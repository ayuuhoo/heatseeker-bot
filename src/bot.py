from typing import override

from rlbot.flat import AirState, ControllerState, GamePacket, MatchPhase
from rlbot.managers import Bot

from util.drive import steer_toward_target
from util.orientation import Orientation
from util.vec import Vec3

# ---- Tunable constants ----
HOVER_HEIGHT = 300        # target hover height (UU)
GOAL_RADIUS = 700         # how close to goal center counts as "in position"
RECOVER_DIST = 2500       # past this, abandon hover and drive back (bumped far)
STOP_SPEED = 120          # consider ourselves stopped below this speed (UU/s)

ORI_P = 4.0               # orientation proportional gain (how hard we correct tilt)
ORI_D = 0.9               # orientation derivative gain (damping, stops tumbling)

ALT_P = 2.5               # height error -> desired climb rate (UU/s per UU)
MAX_CLIMB = 400           # cap on desired vertical speed (UU/s)
BOOST_NOSE_MIN = 0.5      # nose must point this far up before we allow boost

POS_P = 0.0015            # position error -> candle lean (per UU)
POS_D = 0.0045            # velocity damping for position hold (per UU/s)
MAX_LEAN = 0.35           # max candle lean while holding position


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class HeatseekGoalie(Bot):
    # State machine: DRIVE -> LAUNCH -> HOVER
    _state: str = "DRIVE"
    _launch_t0: float = -1.0
    _goal_pos: Vec3 = Vec3(0, 0, 0)
    _toward_field_y: float = 1.0

    @override
    def initialize(self):
        goal_y = 0.0
        for g in self.field_info.goals:
            if g.team_num == self.team:
                goal_y = float(g.location.y)
                break
        # Sit a little in front of the goal line, facing the field.
        self._toward_field_y = -1.0 if goal_y > 0 else 1.0
        self._goal_pos = Vec3(0, goal_y + self._toward_field_y * 150, 0)

    @override
    def get_output(self, packet: GamePacket) -> ControllerState:
        if not packet.balls:
            return ControllerState()

        phase = packet.match_info.match_phase
        car = packet.players[self.index]
        pos = Vec3(car.physics.location)
        vel = Vec3(car.physics.velocity)
        ori = Orientation(car.physics.rotation)
        angvel = Vec3(car.physics.angular_velocity)
        t = packet.match_info.seconds_elapsed
        on_ground = car.air_state == AirState.OnGround

        # Run the full goalie machine during BOTH kickoff and active play, so it
        # gets home and airborne as soon as the countdown releases, instead of
        # waiting for the match timer to start.
        if phase not in (MatchPhase.Active, MatchPhase.Kickoff):
            return ControllerState()

        flat_dist = pos.flat().dist(self._goal_pos.flat())

        # ---- State transitions ----
        if flat_dist > RECOVER_DIST:
            # Bumped really far out -> abandon the hover, come down and drive
            # back on the ground (flying back from here is too slow).
            self._state = "DRIVE"
        elif on_ground:
            if flat_dist > GOAL_RADIUS:
                # Not in position -> drive back.
                self._state = "DRIVE"
            elif vel.length() < STOP_SPEED:
                # In position and stopped -> jump up.
                if self._state != "LAUNCH":
                    self._state = "LAUNCH"
                    self._launch_t0 = -1.0
            else:
                # In position but still rolling -> keep braking via DRIVE.
                self._state = "DRIVE"

        # ---- Debug overlay ----
        self.renderer.begin_rendering()
        self.renderer.draw_string_3d(
            f"{self._state}  z={pos.z:.0f}  spd={vel.length():.0f}",
            pos, 1, self.renderer.white,
        )
        self.renderer.end_rendering()

        # ---- Dispatch ----
        if self._state == "DRIVE":
            return self._drive_to_goal(car, pos, vel)
        if self._state == "LAUNCH":
            return self._launch(t, car, ori)
        return self._hover(pos, vel, ori, angvel)

    # ---------------------------------------------------------------
    # 1) Drive to the goal and stop.
    # ---------------------------------------------------------------
    def _drive_to_goal(self, car, pos: Vec3, vel: Vec3) -> ControllerState:
        controls = ControllerState()
        flat_dist = pos.flat().dist(self._goal_pos.flat())
        if flat_dist > 400:
            # Still far -> drive toward the goal spot.
            controls.throttle = 1.0
            controls.steer = steer_toward_target(car, self._goal_pos)
            # Boost when far and roughly facing the goal (not mid-turn).
            if flat_dist > 900 and abs(controls.steer) < 0.3:
                controls.boost = True
        else:
            # Close -> brake until stopped, then idle.
            if vel.flat().length() > STOP_SPEED:
                controls.throttle = -1.0
            else:
                controls.throttle = 0.0
        return controls

    # ---------------------------------------------------------------
    # 2) Jump off the ground and pitch the nose up until vertical.
    # ---------------------------------------------------------------
    def _launch(self, t: float, car, ori: Orientation) -> ControllerState:
        if self._launch_t0 < 0:
            self._launch_t0 = t
        dt = t - self._launch_t0

        controls = ControllerState()
        if dt < 0.20:
            # One continuous jump (no second press, so no flip/dodge).
            controls.jump = True
        # Pitch nose up the whole time; boost once it's tilted up enough.
        controls.pitch = 1.0
        if ori.forward.z > BOOST_NOSE_MIN:
            controls.boost = True

        if car.air_state != AirState.OnGround and ori.forward.z > 0.85:
            # Airborne and nearly vertical -> hand off to hover.
            self._state = "HOVER"
        elif car.air_state == AirState.OnGround and dt > 1.0:
            # Jump never took -> retry.
            self._launch_t0 = -1.0
        return controls

    # ---------------------------------------------------------------
    # 3) Hover: stay vertical (nose straight up) and hold height.
    #    Orientation is corrected with proportional pitch/yaw/roll inputs
    #    (a "calculated amount" each tick, not a held button), and altitude
    #    is held by feathering boost on/off to track a target climb rate.
    # ---------------------------------------------------------------
    def _hover(
        self, pos: Vec3, vel: Vec3, ori: Orientation, angvel: Vec3
    ) -> ControllerState:
        # Hold our home position: lean the candle to push back toward it. The
        # boost (fired for altitude) provides the horizontal thrust; the velocity
        # term damps the lean so we settle instead of drifting/oscillating.
        # At rest on-target both leans are 0, so the car sits perfectly vertical.
        home = self._goal_pos
        lean_x = clamp(POS_P * (home.x - pos.x) - POS_D * vel.x, -MAX_LEAN, MAX_LEAN)
        lean_y = clamp(POS_P * (home.y - pos.y) - POS_D * vel.y, -MAX_LEAN, MAX_LEAN)

        # Desired orientation: nose up (leaned toward home), roof toward the field.
        f_d = Vec3(lean_x, lean_y, 1.0).normalized()
        up_ref = Vec3(0.0, self._toward_field_y, 0.0)
        right_d = up_ref.cross(f_d).normalized()
        up_d = f_d.cross(right_d).normalized()

        f, r, u = ori.forward, ori.right, ori.up

        # Orientation error as a world-frame rotation axis (~ sin of angle off).
        err = (f.cross(f_d) + r.cross(right_d) + u.cross(up_d)) * 0.5

        # Project error and spin onto the car's own axes.
        e_f, e_r, e_u = err.dot(f), err.dot(r), err.dot(u)
        w_f, w_r, w_u = angvel.dot(f), angvel.dot(r), angvel.dot(u)

        # PD per axis -> the exact correction to apply this tick.
        about_f = ORI_P * e_f - ORI_D * w_f  # about nose  -> roll
        about_r = ORI_P * e_r - ORI_D * w_r  # about right -> pitch
        about_u = ORI_P * e_u - ORI_D * w_u  # about roof  -> yaw

        controls = ControllerState()
        # Control-sign mapping (RL conventions):
        #   +pitch -> nose up   = rotation about -right
        #   +roll  -> roll right = rotation about -forward
        #   +yaw   -> nose right = rotation about +up
        controls.roll = clamp(-about_f, -1.0, 1.0)
        controls.pitch = clamp(-about_r, -1.0, 1.0)
        controls.yaw = clamp(about_u, -1.0, 1.0)

        # Feathered boost for altitude: target a vertical speed proportional to
        # the height error, then boost only while we're slower than that. Near
        # the target height the desired speed -> 0, so boost pulses on/off to
        # cancel gravity -> a steady hover instead of flying up.
        desired_vz = clamp(ALT_P * (HOVER_HEIGHT - pos.z), -MAX_CLIMB, MAX_CLIMB)
        controls.boost = f.z > BOOST_NOSE_MIN and vel.z < desired_vz
        return controls


if __name__ == "__main__":
    HeatseekGoalie("rlbot_community/heatseek_goalie").run()

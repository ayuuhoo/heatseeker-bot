import math
from typing import override

from rlbot.flat import AirState, ControllerState, GamePacket, MatchPhase
from rlbot.managers import Bot

from util.orientation import Orientation, relative_location
from util.vec import Vec3

# ---- Tunable constants ----
HOVER_HEIGHT = 300        # target hover height (UU)
GOAL_RADIUS = 700         # how close to goal center counts as "in position"
STOP_SPEED = 120          # consider ourselves stopped below this speed (UU/s)
# When airborne and knocked off position, aerial back if within this distance of
# home, else come down and drive. Bigger = prefer aerialing. We aerial back more
# readily when knocked INTO the net, and drive back more readily when knocked OUT.
AERIAL_BACK_FRONT = 1200  # in front of the goal line (knocked out)
AERIAL_BACK_IN_NET = 1600 # behind the goal line (knocked into the net)
ARRIVE_RADIUS = 250       # within this of home, just brake to a stop (then launch)
APPROACH_GAIN = 1.8       # desired drive speed per UU of distance (arrive behavior)
MAX_DRIVE_SPEED = 2300    # cap on desired drive speed (UU/s)
IN_NET_SPEED = 600        # slow speed cap while escaping the net (avoid wall-climb)
BOOST_DIST = 1200         # only boost when farther than this from home
ALIGN_FALLOFF = 1.2       # heading error (rad) at which we slow to the min turn speed
MIN_TURN_SPEED = 250      # lowest approach speed when turning sharply
STEER_GAIN = 3.0          # steering response to heading error
BEHIND_ANGLE = 2.0        # heading error (rad) past which home counts as "behind" us
REVERSE_MAX_DIST = 1500   # only reverse (vs turn+boost) when home is closer than this
REVERSE_SPEED = 1200      # reverse speed cap (no boost while reversing)

ORI_P = 5.0               # orientation proportional gain (how hard we correct tilt)
ORI_D = 1.0               # orientation derivative gain (damping, stops tumbling)

ALT_P = 2.5               # height error -> desired climb rate (UU/s per UU)
MAX_CLIMB = 400           # cap on desired vertical speed (UU/s)
BOOST_NOSE_MIN = 0.5      # nose must point this far up before we allow boost

POS_P = 0.0022            # side-to-side lean gain (per UU)
POS_D = 0.0055            # side-to-side velocity damping (per UU/s)
POS_P_Y = 0.0034          # distance-from-line lean gain (higher = quicker depth adjust)
POS_D_Y = 0.0072          # distance-from-line velocity damping
MAX_LEAN = 0.5            # max candle lean while holding position


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class HeatseekGoalie(Bot):
    # State machine: DRIVE -> LAUNCH -> HOVER
    _state: str = "DRIVE"
    _launch_t0: float = -1.0
    _goal_pos: Vec3 = Vec3(0, 0, 0)
    _goal_line_y: float = 0.0
    _toward_field_y: float = 1.0

    @override
    def initialize(self):
        goal_y = 0.0
        for g in self.field_info.goals:
            if g.team_num == self.team:
                goal_y = float(g.location.y)
                break
        # Sit a little in front of the goal line, facing the field.
        self._goal_line_y = goal_y
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
        in_net = (pos.y - self._goal_line_y) * self._toward_field_y < 0.0

        # ---- State transitions ----
        if on_ground:
            # On the ground we can't aerial -> drive out of the net / back to
            # position, or launch if we're parked in position and stopped.
            if in_net or flat_dist > GOAL_RADIUS:
                self._state = "DRIVE"
            elif vel.length() < STOP_SPEED:
                # In position and stopped -> jump up.
                if self._state != "LAUNCH":
                    self._state = "LAUNCH"
                    self._launch_t0 = -1.0
            else:
                # In position but still rolling -> keep braking via DRIVE.
                self._state = "DRIVE"
        elif self._state != "LAUNCH":
            # Airborne (and not mid-launch): aerial back to position if we're
            # close enough, otherwise come down and drive back. We aerial back
            # more readily when knocked INTO the net, and drive back more readily
            # when knocked OUT in front of it.
            aerial_back_dist = AERIAL_BACK_IN_NET if in_net else AERIAL_BACK_FRONT
            self._state = "DRIVE" if flat_dist > aerial_back_dist else "HOVER"

        # ---- Debug overlay ----
        self.renderer.begin_rendering()
        self.renderer.draw_string_3d(
            f"{self._state}  z={pos.z:.0f}  spd={vel.length():.0f}",
            pos, 1, self.renderer.white,
        )
        self.renderer.end_rendering()

        # ---- Dispatch ----
        if self._state == "DRIVE":
            return self._drive_to_goal(car, pos, vel, ori, angvel)
        if self._state == "LAUNCH":
            return self._launch(t, car, ori)
        return self._hover(pos, vel, ori, angvel)

    # ---------------------------------------------------------------
    # 1) Drive to the goal and stop.
    # ---------------------------------------------------------------
    def _drive_to_goal(
        self, car, pos: Vec3, vel: Vec3, ori: Orientation, angvel: Vec3
    ) -> ControllerState:
        # If we're airborne (knocked up / falling), first orient for a clean
        # landing pointed at the goal.
        if car.air_state != AirState.OnGround:
            return self._air_recover(pos, ori, angvel)

        controls = ControllerState()
        in_net = (pos.y - self._goal_line_y) * self._toward_field_y < 0.0

        to_home = (self._goal_pos - pos).flat()
        dist = to_home.length()
        if dist < 1.0:
            return controls  # essentially on the spot

        speed = vel.flat().length()
        # Heading error to home: 0 = pointing straight at it, +/-pi = behind us.
        local = relative_location(pos, ori, self._goal_pos)
        angle = math.atan2(local.y, local.x)
        abs_angle = abs(angle)
        v_fwd = vel.dot(ori.forward)  # signed: + driving forward, - reversing

        # Once we're OUT of the net and close to home, brake to a stop (oppose our
        # motion) so the state machine can launch. We must be out of the net first,
        # or we'd stop behind the line and never launch.
        if not in_net and dist < ARRIVE_RADIUS:
            if speed < STOP_SPEED:
                controls.throttle = 0.0
            else:
                controls.throttle = -1.0 if v_fwd > 0 else 1.0
            return controls

        # Choose the faster way home. If home is behind us AND fairly close, back
        # straight out (no turn needed, but no boost). Otherwise turn to face it
        # and drive forward — that costs turning time but lets us boost, which
        # wins over longer distances.
        if abs_angle > BEHIND_ANGLE and dist < REVERSE_MAX_DIST:
            # --- Reverse straight out: steer the REAR toward home. ---
            rev_angle = math.atan2(local.y, -local.x)
            controls.steer = clamp(rev_angle * STEER_GAIN, -1.0, 1.0)
            desired_speed = min(dist * APPROACH_GAIN, REVERSE_SPEED)
            closing = -v_fwd  # how fast we're reversing toward home
            controls.throttle = -1.0 if closing < desired_speed else 1.0
        else:
            # --- Forward: slow down to turn tightly, boost when lined up. ---
            controls.steer = clamp(angle * STEER_GAIN, -1.0, 1.0)
            # The more we have to turn, the slower we go -> pivot tightly toward
            # home instead of arcing out in a big circle.
            align_cap = clamp(
                MAX_DRIVE_SPEED * (1.0 - abs_angle / ALIGN_FALLOFF),
                MIN_TURN_SPEED,
                MAX_DRIVE_SPEED,
            )
            max_speed = IN_NET_SPEED if in_net else MAX_DRIVE_SPEED
            desired_speed = min(dist * APPROACH_GAIN, align_cap, max_speed)
            controls.throttle = 1.0 if speed < desired_speed else -1.0
            # Boost only when far and pointed nearly straight at home.
            if not in_net and dist > BOOST_DIST and abs_angle < 0.2:
                controls.boost = True
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
        # The y axis (distance from the goal line) uses stronger gains so depth
        # is corrected more quickly.
        lean_y = clamp(
            POS_P_Y * (home.y - pos.y) - POS_D_Y * vel.y, -MAX_LEAN, MAX_LEAN
        )

        # Desired orientation: nose up (leaned toward home), roof toward the field.
        f_d = Vec3(lean_x, lean_y, 1.0).normalized()
        up_ref = Vec3(0.0, self._toward_field_y, 0.0)
        right_d = up_ref.cross(f_d).normalized()
        up_d = f_d.cross(right_d).normalized()

        controls = self._orient_controls(ori, angvel, f_d, right_d, up_d)

        # Feathered boost for altitude: target a vertical speed proportional to
        # the height error, then boost only while we're slower than that. Near
        # the target height the desired speed -> 0, so boost pulses on/off to
        # cancel gravity -> a steady hover instead of flying up.
        desired_vz = clamp(ALT_P * (HOVER_HEIGHT - pos.z), -MAX_CLIMB, MAX_CLIMB)
        controls.boost = ori.forward.z > BOOST_NOSE_MIN and vel.z < desired_vz
        return controls

    # ---------------------------------------------------------------
    # Shared orientation PD: drive the car's (forward, right, up) axes
    # toward a desired frame using proportional pitch/yaw/roll inputs.
    # ---------------------------------------------------------------
    def _orient_controls(
        self,
        ori: Orientation,
        angvel: Vec3,
        f_d: Vec3,
        right_d: Vec3,
        up_d: Vec3,
    ) -> ControllerState:
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
        #   +pitch -> nose up    = rotation about -right
        #   +roll  -> roll right = rotation about -forward
        #   +yaw   -> nose right = rotation about +up
        controls.roll = clamp(-about_f, -1.0, 1.0)
        controls.pitch = clamp(-about_r, -1.0, 1.0)
        controls.yaw = clamp(about_u, -1.0, 1.0)
        return controls

    # ---------------------------------------------------------------
    # Falling: air-roll wheels-down and point the nose at the goal so we
    # land already lined up to drive back (no flip / turn-around needed).
    # We don't boost here — just position; boosting toward home happens once
    # we're on the ground.
    # ---------------------------------------------------------------
    def _air_recover(
        self, pos: Vec3, ori: Orientation, angvel: Vec3
    ) -> ControllerState:
        to_goal = (self._goal_pos - pos).flat()
        if to_goal.length() > 50:
            f_d = to_goal.normalized()
        else:
            # Basically over the goal spot -> just keep our current heading.
            flat_fwd = ori.forward.flat()
            f_d = (
                flat_fwd.normalized()
                if flat_fwd.length() > 0.1
                else Vec3(0.0, self._toward_field_y, 0.0)
            )
        up_d = Vec3(0.0, 0.0, 1.0)
        right_d = up_d.cross(f_d).normalized()

        controls = self._orient_controls(ori, angvel, f_d, right_d, up_d)
        controls.throttle = 1.0  # spin the wheels so we drive the instant we land
        return controls


if __name__ == "__main__":
    HeatseekGoalie("rlbot_community/heatseek_goalie").run()

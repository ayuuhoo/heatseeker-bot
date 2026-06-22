from typing import override

from rlbot.flat import AirState, ControllerState, GamePacket, MatchPhase
from rlbot.managers import Bot

from util.drive import steer_toward_target
from util.orientation import Orientation
from util.vec import Vec3

# ---- Tunable constants (tweak these live while testing) ----
HOVER_HEIGHT = 250      # default hover z (UU) when just holding center
GOAL_RADIUS = 600       # flat dist from goal center within which we launch
ORI_P = 5.0             # orientation proportional gain
ORI_D = 1.2             # orientation derivative (damping) gain
LAT_TILT_GAIN = 0.0015  # how hard we lean per UU of lateral error
MAX_TILT = 0.6          # max nose lean for strafing (≈ tan of lean angle)
BOOST_NOSE_MIN = 0.5    # forward.z must exceed this before we allow boost
MAX_STRAFE_X = 800      # clamp target x to stay between the goal posts


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def find_goal_crossing(slices, goal_y: float) -> "Vec3 | None":
    """Interpolate where the predicted ball path first crosses our goal y-plane."""
    prev = None
    for s in slices:
        loc = s.physics.location
        if prev is not None and prev.y != loc.y:
            if (prev.y - goal_y) * (loc.y - goal_y) <= 0:
                t = (goal_y - prev.y) / (loc.y - prev.y)
                return Vec3(
                    prev.x + t * (loc.x - prev.x),
                    goal_y,
                    prev.z + t * (loc.z - prev.z),
                )
        prev = loc
    return None


class HeatseekGoalie(Bot):
    # States: RETURN → STARTUP → HOVER
    _state: str = "RETURN"
    _startup_t0: float = -1.0
    _goal_y: float = 0.0
    _goal_pos: Vec3 = Vec3(0, 0, 0)
    _toward_field_y: float = 1.0
    _active_initialized: bool = False

    @override
    def initialize(self):
        for g in self.field_info.goals:
            if g.team_num == self.team:
                self._goal_y = float(g.location.y)
                self._goal_pos = Vec3(0, g.location.y, 0)
                break
        # The field interior is the opposite y-direction from our goal.
        self._toward_field_y = -1.0 if self._goal_y > 0 else 1.0

    @override
    def get_output(self, packet: GamePacket) -> ControllerState:
        if not packet.balls:
            return ControllerState()

        phase = packet.match_info.match_phase
        my_car = packet.players[self.index]

        # ----- Step 1 & 2: kickoff decision + drive -----
        if phase == MatchPhase.Kickoff:
            self._active_initialized = False
            return self._do_kickoff(packet)

        if phase != MatchPhase.Active:
            return ControllerState()

        # First active tick after a kickoff: get back into position.
        if not self._active_initialized:
            self._active_initialized = True
            self._state = "RETURN"

        physics = my_car.physics
        pos = Vec3(physics.location)
        ori = Orientation(physics.rotation)
        angvel = Vec3(physics.angular_velocity)
        t = packet.match_info.seconds_elapsed
        on_ground = my_car.air_state == AirState.OnGround

        # ----- State transitions based on ground contact -----
        if on_ground:
            flat_dist = pos.flat().dist(self._goal_pos.flat())
            if flat_dist > GOAL_RADIUS:
                if self._state != "RETURN":
                    self._state = "RETURN"
            elif self._state != "STARTUP":
                self._state = "STARTUP"
                self._startup_t0 = -1.0

        # ----- Step 5: is the ball heading at our net? Where will it cross? -----
        target_x = 0.0
        target_z = float(HOVER_HEIGHT)
        ball = packet.balls[0].physics
        ball_pos = Vec3(ball.location)
        ball_vel = Vec3(ball.velocity)
        heading_at_net = (self._goal_y - ball_pos.y) * ball_vel.y > 0
        if heading_at_net and self.ball_prediction.slices:
            crossing = find_goal_crossing(self.ball_prediction.slices, self._goal_y)
            if crossing is not None:
                target_x = clamp(float(crossing.x), -MAX_STRAFE_X, MAX_STRAFE_X)
                target_z = clamp(float(crossing.z), 60.0, 560.0)

        self.renderer.begin_rendering()
        self.renderer.draw_line_3d(
            pos, Vec3(target_x, self._goal_y, target_z), self.renderer.yellow
        )
        self.renderer.draw_string_3d(
            f"{self._state} z={pos.z:.0f} tx={target_x:.0f}", pos, 1, self.renderer.white
        )
        self.renderer.end_rendering()

        # ----- Dispatch -----
        if self._state == "RETURN":
            return self._do_return(my_car, pos)
        if self._state == "STARTUP":
            return self._do_startup(t, my_car, pos, ori, angvel)
        # Step 4, 6, 7: hover, strafe to target, maintain
        return self._hover_controls(pos, ori, angvel, target_x, target_z)

    # ----- Step 1 & 2: kickoff -----
    def _do_kickoff(self, packet: GamePacket) -> ControllerState:
        my_car = packet.players[self.index]
        my_pos = Vec3(my_car.physics.location)
        ball_pos = Vec3(packet.balls[0].physics.location)
        my_dist = my_pos.dist(ball_pos)

        # Should I go for it? Only if no teammate is closer to the ball.
        teammate_closer = any(
            i != self.index
            and packet.players[i].team == self.team
            and Vec3(packet.players[i].physics.location).dist(ball_pos) < my_dist
            for i in range(len(packet.players))
        )

        controls = ControllerState()
        controls.throttle = 1.0
        if teammate_closer:
            # Let the teammate take it; drive back toward our goal line.
            controls.steer = steer_toward_target(my_car, self._goal_pos)
        else:
            # Go for the kickoff.
            controls.boost = True
            controls.steer = steer_toward_target(my_car, ball_pos)
        return controls

    # ----- Step 2: drive back to the middle of the goal line -----
    def _do_return(self, my_car, pos: Vec3) -> ControllerState:
        controls = ControllerState()
        controls.throttle = 1.0
        controls.steer = steer_toward_target(my_car, self._goal_pos)
        if pos.flat().dist(self._goal_pos.flat()) > 1500:
            controls.boost = True
        return controls

    # ----- Step 3: jump up, tilt nose up, start boosting -----
    def _do_startup(
        self, t: float, my_car, pos: Vec3, ori: Orientation, angvel: Vec3
    ) -> ControllerState:
        if self._startup_t0 < 0:
            self._startup_t0 = t
        dt = t - self._startup_t0

        # Let the hover controller do the pitch-up + boost work.
        controls = self._hover_controls(pos, ori, angvel, 0.0, HOVER_HEIGHT)

        if dt < 0.12:
            # Single continuous jump (NOT a second press → no flip/dodge).
            controls.jump = True

        if my_car.air_state != AirState.OnGround and ori.forward.z > 0.7:
            # Airborne and nose pointing up enough — hand off to hover.
            self._state = "HOVER"
        elif my_car.air_state == AirState.OnGround and dt > 0.6:
            # Jump didn't take — retry.
            self._startup_t0 = -1.0

        return controls

    # ----- Steps 4, 6, 7: candle hover + lateral strafe -----
    def _hover_controls(
        self,
        pos: Vec3,
        ori: Orientation,
        angvel: Vec3,
        target_x: float,
        target_z: float,
    ) -> ControllerState:
        """
        Hold the car vertical (nose straight up) with its belly parallel to the
        goal line, while leaning slightly to strafe toward target_x.

        Desired orientation (world frame):
          forward (nose) ≈ +Z (straight up, leaned toward the strafe target)
          up   (roof)    ≈ ±Y (toward the field) → belly faces the field
          right          ≈ ±X (along the goal line) → boost-lean = strafe
        """
        # Lean the nose toward the strafe target so boost thrust gets an X push.
        tilt = clamp(LAT_TILT_GAIN * (target_x - pos.x), -MAX_TILT, MAX_TILT)
        f_d = Vec3(tilt, 0.0, 1.0).normalized()
        up_ref = Vec3(0.0, self._toward_field_y, 0.0)
        right_d = up_ref.cross(f_d).normalized()
        up_d = f_d.cross(right_d).normalized()

        f, r, u = ori.forward, ori.right, ori.up

        # Orientation error as a world-frame rotation axis (≈ sin of angle off).
        err = (f.cross(f_d) + r.cross(right_d) + u.cross(up_d)) * 0.5

        # Project error and angular velocity onto the car's local axes.
        e_f, e_r, e_u = err.dot(f), err.dot(r), err.dot(u)
        w_f, w_r, w_u = angvel.dot(f), angvel.dot(r), angvel.dot(u)

        # PD per axis → desired rotation about that local axis.
        about_f = ORI_P * e_f - ORI_D * w_f  # about nose  (roll)
        about_r = ORI_P * e_r - ORI_D * w_r  # about right (pitch)
        about_u = ORI_P * e_u - ORI_D * w_u  # about roof  (yaw)

        controls = ControllerState()
        # Control-sign mapping derived from RL conventions:
        #   +pitch control → nose up  → rotation about −right
        #   +roll  control → roll right → rotation about −forward
        #   +yaw   control → nose right → rotation about +up
        controls.roll = clamp(-about_f, -1.0, 1.0)
        controls.pitch = clamp(-about_r, -1.0, 1.0)
        controls.yaw = clamp(about_u, -1.0, 1.0)

        # Boost only when the nose is pointed up enough, and we need height or push.
        need_height = pos.z < target_z
        need_strafe = abs(target_x - pos.x) > 60
        controls.boost = f.z > BOOST_NOSE_MIN and (need_height or need_strafe)
        return controls


if __name__ == "__main__":
    HeatseekGoalie("rlbot_community/heatseek_goalie").run()

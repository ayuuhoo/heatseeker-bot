import math
from typing import override

from rlbot.flat import AirState, ControllerState, GamePacket
from rlbot.managers import Bot

from util.orientation import Orientation
from util.vec import Vec3

HALF_PI = math.pi / 2
HOVER_HEIGHT = 200  # base hover z (UU); adjustable per testing


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def find_goal_crossing(slices, goal_y: float) -> "Vec3 | None":
    """Return interpolated ball position where prediction first crosses the goal y-plane."""
    prev_loc = None
    for s in slices:
        loc = s.physics.location
        if prev_loc is not None and prev_loc.y != loc.y:
            if (prev_loc.y - goal_y) * (loc.y - goal_y) <= 0:
                t = (goal_y - prev_loc.y) / (loc.y - prev_loc.y)
                return Vec3(
                    prev_loc.x + t * (loc.x - prev_loc.x),
                    goal_y,
                    prev_loc.z + t * (loc.z - prev_loc.z),
                )
        prev_loc = loc
    return None


class HeatseekGoalie(Bot):
    _state: str = "STARTUP"
    _startup_t0: float = -1.0

    @override
    def initialize(self):
        pass

    @override
    def get_output(self, packet: GamePacket) -> ControllerState:
        if not packet.balls:
            return ControllerState()

        my_car = packet.players[self.index]
        physics = my_car.physics
        pos = Vec3(physics.location)
        rot = physics.rotation
        angvel = Vec3(physics.angular_velocity)
        t = packet.match_info.seconds_elapsed

        # Landing or getting knocked down → restart airborne sequence
        if my_car.air_state == AirState.OnGround and self._state != "STARTUP":
            self._state = "STARTUP"
            self._startup_t0 = -1.0

        # Find the y-coordinate of our own goal from field info
        goal_y = 0.0
        for g in self.field_info.goals:
            if g.team_num == self.team:
                goal_y = float(g.location.y)
                break

        # Default: hover at goal center; override with predicted crossing point
        target_x = 0.0
        target_z = float(HOVER_HEIGHT)
        if self.ball_prediction.slices:
            crossing = find_goal_crossing(self.ball_prediction.slices, goal_y)
            if crossing is not None:
                target_x = clamp(float(crossing.x), -700.0, 700.0)
                target_z = clamp(float(crossing.z), 60.0, 560.0)

        self.renderer.begin_rendering()
        hover_target = Vec3(target_x, goal_y, target_z)
        self.renderer.draw_line_3d(pos, hover_target, self.renderer.yellow)
        self.renderer.draw_string_3d(
            f"{self._state}  z={pos.z:.0f}", pos, 1, self.renderer.white
        )
        self.renderer.end_rendering()

        if self._state == "STARTUP":
            return self._do_startup(t, my_car)
        return self._do_hover(pos, rot, angvel, target_x, target_z)

    def _do_startup(self, t: float, my_car) -> ControllerState:
        if self._startup_t0 < 0:
            self._startup_t0 = t
        dt = t - self._startup_t0

        controls = ControllerState()
        if dt < 0.05:
            # First jump press
            controls.jump = True
        elif dt < 0.20:
            # Release jump; pitch nose hard upward while rising
            controls.pitch = 1.0
        elif dt < 0.25:
            # Double jump for extra height; nose still pitching up
            controls.jump = True
            controls.pitch = 1.0
            controls.boost = True
        else:
            # Keep boosting nose-up until clearly airborne, then hand off to hover
            controls.pitch = 1.0
            controls.boost = True
            if my_car.air_state != AirState.OnGround:
                self._state = "HOVER"

        return controls

    def _do_hover(
        self,
        pos: Vec3,
        rot,
        angvel: Vec3,
        target_x: float,
        target_z: float,
    ) -> ControllerState:
        ori = Orientation(rot)

        # Project world-space angular velocity onto car-local axes for PD damping
        pitch_rate = angvel.dot(ori.right)
        roll_rate = angvel.dot(ori.forward)
        yaw_rate = angvel.dot(ori.up)

        # PD: drive pitch → +π/2 (candle / nose-up)
        pitch_err = HALF_PI - float(rot.pitch)
        pitch_ctrl = clamp(3.0 * pitch_err - 0.8 * pitch_rate, -1.0, 1.0)

        # PD: drive roll → 0 (no sideways spin)
        roll_err = -float(rot.roll)
        roll_ctrl = clamp(3.0 * roll_err - 0.8 * roll_rate, -1.0, 1.0)

        # Lateral drift: apply a small yaw bias to tilt the nose toward target_x.
        # In candle orientation, yaw rotates around the car's "up" vector, which is
        # horizontal. The mapping to world-x drift depends on car heading:
        #   facing +y (blue goal, ori.up.y ≈ −1): +yaw → +x tilt
        #   facing −y (orange goal, ori.up.y ≈ +1): +yaw → −x tilt
        lat_err = target_x - float(pos.x)
        yaw_sign = 1.0 if ori.up.y <= 0 else -1.0
        yaw_bias = clamp(lat_err * yaw_sign / 600.0, -0.35, 0.35)
        yaw_ctrl = clamp(-0.5 * yaw_rate + yaw_bias, -1.0, 1.0)

        # Height: boost on/off to hold target_z
        boost = pos.z < target_z

        controls = ControllerState()
        controls.pitch = pitch_ctrl
        controls.roll = roll_ctrl
        controls.yaw = yaw_ctrl
        controls.boost = boost
        return controls


if __name__ == "__main__":
    HeatseekGoalie("rlbot_community/heatseek_goalie").run()

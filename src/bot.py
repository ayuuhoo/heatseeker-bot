import math
from typing import override

from rlbot.flat import AirState, ControllerState, GamePacket, MatchPhase
from rlbot.managers import Bot

from util.drive import steer_toward_target
from util.orientation import Orientation
from util.vec import Vec3

HALF_PI = math.pi / 2
HOVER_HEIGHT = 200  # default hover z (UU); tune per testing


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
    _goal_y: float = 0.0
    _goal_pos: Vec3 = Vec3(0, 0, 0)

    @override
    def initialize(self):
        for g in self.field_info.goals:
            if g.team_num == self.team:
                self._goal_y = float(g.location.y)
                self._goal_pos = Vec3(g.location.x, g.location.y, 0)
                break

    @override
    def get_output(self, packet: GamePacket) -> ControllerState:
        if not packet.balls:
            return ControllerState()

        match_phase = packet.match_info.match_phase

        # Kickoff: dedicated logic; also reset so hover starts fresh after kickoff
        if match_phase == MatchPhase.Kickoff:
            self._state = "STARTUP"
            self._startup_t0 = -1.0
            return self._do_kickoff(packet)

        if match_phase != MatchPhase.Active:
            return ControllerState()

        my_car = packet.players[self.index]
        physics = my_car.physics
        pos = Vec3(physics.location)
        rot = physics.rotation
        angvel = Vec3(physics.angular_velocity)
        t = packet.match_info.seconds_elapsed

        if my_car.air_state == AirState.OnGround and self._state != "STARTUP":
            self._state = "STARTUP"
            self._startup_t0 = -1.0

        target_x = 0.0
        target_z = float(HOVER_HEIGHT)
        if self.ball_prediction.slices:
            crossing = find_goal_crossing(self.ball_prediction.slices, self._goal_y)
            if crossing is not None:
                target_x = clamp(float(crossing.x), -700.0, 700.0)
                target_z = clamp(float(crossing.z), 60.0, 560.0)

        self.renderer.begin_rendering()
        self.renderer.draw_line_3d(
            pos, Vec3(target_x, self._goal_y, target_z), self.renderer.yellow
        )
        self.renderer.draw_string_3d(
            f"{self._state}  z={pos.z:.0f}", pos, 1, self.renderer.white
        )
        self.renderer.end_rendering()

        if self._state == "STARTUP":
            return self._do_startup(t, my_car)
        return self._do_hover(pos, rot, angvel, target_x, target_z)

    def _do_kickoff(self, packet: GamePacket) -> ControllerState:
        my_car = packet.players[self.index]
        ball_pos = Vec3(packet.balls[0].physics.location)
        my_pos = Vec3(my_car.physics.location)
        my_dist = my_pos.dist(ball_pos)

        # Check if we are the closest teammate to the ball
        is_closest = all(
            i == self.index
            or packet.players[i].team != self.team
            or Vec3(packet.players[i].physics.location).dist(ball_pos) >= my_dist
            for i in range(len(packet.players))
        )

        controls = ControllerState()
        if is_closest:
            controls.throttle = 1.0
            controls.boost = True
            controls.steer = steer_toward_target(my_car, ball_pos)
        else:
            # Return to goal and get ready to hover
            controls.throttle = 1.0
            controls.steer = steer_toward_target(my_car, self._goal_pos)

        return controls

    def _do_startup(self, t: float, my_car) -> ControllerState:
        if self._startup_t0 < 0:
            self._startup_t0 = t
        dt = t - self._startup_t0

        controls = ControllerState()
        if dt < 0.1:
            # Single jump only — no pitch here to avoid triggering a flip/dodge
            controls.jump = True
        elif my_car.air_state != AirState.OnGround:
            # Airborne: hand off to hover; the PD will pitch nose up from here
            self._state = "HOVER"
        elif dt > 0.4:
            # Still on ground after jump — reset and retry
            self._startup_t0 = -1.0

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

        pitch_rate = angvel.dot(ori.right)
        roll_rate = angvel.dot(ori.forward)
        yaw_rate = angvel.dot(ori.up)

        # PD: pitch → +π/2 (candle / nose-up)
        pitch_err = HALF_PI - float(rot.pitch)
        pitch_ctrl = clamp(3.0 * pitch_err - 0.8 * pitch_rate, -1.0, 1.0)

        # PD: roll → 0
        roll_err = -float(rot.roll)
        roll_ctrl = clamp(3.0 * roll_err - 0.8 * roll_rate, -1.0, 1.0)

        # Yaw bias for lateral drift toward predicted ball crossing
        # Sign flips based on which goal we defend (facing +y vs -y in candle orientation)
        lat_err = target_x - float(pos.x)
        yaw_sign = 1.0 if ori.up.y <= 0 else -1.0
        yaw_bias = clamp(lat_err * yaw_sign / 600.0, -0.35, 0.35)
        yaw_ctrl = clamp(-0.5 * yaw_rate + yaw_bias, -1.0, 1.0)

        boost = pos.z < target_z

        controls = ControllerState()
        controls.pitch = pitch_ctrl
        controls.roll = roll_ctrl
        controls.yaw = yaw_ctrl
        controls.boost = boost
        return controls


if __name__ == "__main__":
    HeatseekGoalie("rlbot_community/heatseek_goalie").run()

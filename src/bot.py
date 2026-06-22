import math
from typing import override

from rlbot.flat import AirState, ControllerState, GamePacket, MatchPhase
from rlbot.managers import Bot

from util.drive import steer_toward_target
from util.orientation import Orientation
from util.vec import Vec3

HALF_PI = math.pi / 2
HOVER_HEIGHT = 250   # default hover z (UU); tune per testing
GOAL_RADIUS = 500    # flat distance from goal center before starting jump


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
    # States: RETURN → drive back to goal; STARTUP → jump airborne; HOVER → candle hover
    _state: str = "RETURN"
    _startup_t0: float = -1.0
    _goal_y: float = 0.0
    _goal_pos: Vec3 = Vec3(0, 0, 0)

    @override
    def initialize(self):
        for g in self.field_info.goals:
            if g.team_num == self.team:
                self._goal_y = float(g.location.y)
                self._goal_pos = Vec3(0, g.location.y, 0)
                break

    @override
    def get_output(self, packet: GamePacket) -> ControllerState:
        if not packet.balls:
            return ControllerState()

        match_phase = packet.match_info.match_phase

        # During kickoff: always return to goal (we're a goalie; let opponents or
        # a forward teammate handle the ball)
        if match_phase == MatchPhase.Kickoff:
            self._state = "RETURN"
            my_car = packet.players[self.index]
            return self._do_return(my_car, Vec3(my_car.physics.location))

        if match_phase != MatchPhase.Active:
            return ControllerState()

        my_car = packet.players[self.index]
        physics = my_car.physics
        pos = Vec3(physics.location)
        rot = physics.rotation
        angvel = Vec3(physics.angular_velocity)
        t = packet.match_info.seconds_elapsed
        is_on_ground = my_car.air_state == AirState.OnGround

        # State transitions driven by ground contact
        if is_on_ground:
            goal_dist = pos.flat().dist(self._goal_pos.flat())
            if goal_dist > GOAL_RADIUS:
                # Not near the goal — go back before trying to hover
                self._state = "RETURN"
            elif self._state != "STARTUP":
                # Near the goal and not already in jump sequence — start one
                self._state = "STARTUP"
                self._startup_t0 = -1.0

        # Predict where ball will cross our goal line
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

        if self._state == "RETURN":
            return self._do_return(my_car, pos)
        elif self._state == "STARTUP":
            return self._do_startup(t, my_car, rot)
        else:
            return self._do_hover(pos, rot, angvel, target_x, target_z)

    def _do_return(self, my_car, pos: Vec3) -> ControllerState:
        """Drive back to the goal at ground level."""
        controls = ControllerState()
        controls.throttle = 1.0
        controls.steer = steer_toward_target(my_car, self._goal_pos)
        if pos.flat().dist(self._goal_pos.flat()) > 1500:
            controls.boost = True
        return controls

    def _do_startup(self, t: float, my_car, rot) -> ControllerState:
        """Jump and pitch nose up; hand off to hover once past ~45 degrees."""
        if self._startup_t0 < 0:
            self._startup_t0 = t
        dt = t - self._startup_t0

        controls = ControllerState()
        if dt < 0.1:
            # Single jump with nose-up pitch (not a second press, so no flip/dodge)
            controls.jump = True
            controls.pitch = 1.0
        elif my_car.air_state != AirState.OnGround:
            # Airborne: keep pitching and boost upward; hand off once nose is ~45° up
            controls.pitch = 1.0
            controls.boost = True
            if float(rot.pitch) > math.pi / 4:
                self._state = "HOVER"
        elif dt > 0.5:
            # Still on ground after 0.5s — reset and retry
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
        """Candle hover: nose up, roll=0, lateral yaw drift toward predicted crossing."""
        ori = Orientation(rot)

        # Angular velocity projected onto car-local axes for PD damping
        pitch_rate = angvel.dot(ori.right)
        roll_rate = angvel.dot(ori.forward)
        yaw_rate = angvel.dot(ori.up)

        # PD: pitch → +π/2 (candle / nose straight up)
        pitch_err = HALF_PI - float(rot.pitch)
        pitch_ctrl = clamp(3.0 * pitch_err - 0.8 * pitch_rate, -1.0, 1.0)

        # PD: roll → 0 (no sideways spin)
        roll_err = -float(rot.roll)
        roll_ctrl = clamp(3.0 * roll_err - 0.8 * roll_rate, -1.0, 1.0)

        # Yaw bias for lateral drift toward predicted ball crossing.
        # In candle orientation, yaw tilts the nose sideways; the direction mapping
        # to world-x flips based on which goal we defend (facing +y vs -y).
        lat_err = target_x - float(pos.x)
        yaw_sign = 1.0 if ori.up.y <= 0 else -1.0
        yaw_bias = clamp(lat_err * yaw_sign / 600.0, -0.35, 0.35)
        yaw_ctrl = clamp(-0.5 * yaw_rate + yaw_bias, -1.0, 1.0)

        # Boost on/off to hold target height
        boost = pos.z < target_z

        controls = ControllerState()
        controls.pitch = pitch_ctrl
        controls.roll = roll_ctrl
        controls.yaw = yaw_ctrl
        controls.boost = boost
        return controls


if __name__ == "__main__":
    HeatseekGoalie("rlbot_community/heatseek_goalie").run()

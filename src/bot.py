from typing import override

from rlbot.flat import AirState, ControllerState, GamePacket, MatchPhase
from rlbot.managers import Bot

from util.drive import steer_toward_target
from util.orientation import Orientation
from util.vec import Vec3

# ---- Tunable constants (tweak these live while testing) ----
HOVER_HEIGHT = 250        # default hover z (UU) when just holding center
GOAL_RADIUS = 900         # flat dist from goal center within which we launch
GOAL_LINE_OFFSET = 120    # how far in front of the goal to sit (toward field)
STARTUP_SPEED_GATE = 250  # must be slower than this (UU/s) before jumping
ORI_P = 5.0               # orientation proportional gain
ORI_D = 1.2               # orientation derivative (damping) gain
LAT_TILT_GAIN = 0.0015    # how hard we lean per UU of lateral error
LAT_DAMP = 0.002          # how hard we lean against lateral velocity (damping)
MAX_TILT = 0.6            # max nose lean for strafing (≈ tan of lean angle)
BOOST_NOSE_MIN = 0.4      # forward.z must exceed this before we allow boost
MAX_STRAFE_X = 800        # clamp target x to stay between the goal posts
ALT_P = 3.0               # height error → desired climb rate (UU/s per UU)
MAX_CLIMB = 500           # cap on desired vertical speed (UU/s)


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
    _goal_y: float = 0.0           # actual goal-line y (for ball crossing math)
    _goal_pos: Vec3 = Vec3(0, 0, 0)  # where we sit, a bit in front of the line
    _toward_field_y: float = 1.0
    _active_initialized: bool = False
    _prev_phase = None

    # Kickoff handling
    _our_kickoff: bool = False
    _our_kickoff_decided: bool = False
    _kickoff_airborne: bool = False  # we jumped to hit the kickoff; wait to land

    @override
    def initialize(self):
        for g in self.field_info.goals:
            if g.team_num == self.team:
                self._goal_y = float(g.location.y)
                break
        # The field interior is the opposite y-direction from our goal.
        self._toward_field_y = -1.0 if self._goal_y > 0 else 1.0
        # Sit slightly in front of the goal line, toward the field.
        self._goal_pos = Vec3(0, self._goal_y + self._toward_field_y * GOAL_LINE_OFFSET, 0)

    @override
    def get_output(self, packet: GamePacket) -> ControllerState:
        if not packet.balls:
            return ControllerState()

        phase = packet.match_info.match_phase
        my_car = packet.players[self.index]

        # Detect entering a fresh kickoff so we can reset our decision.
        if phase != self._prev_phase:
            if phase == MatchPhase.Kickoff:
                self._our_kickoff_decided = False
                self._our_kickoff = False
                self._kickoff_airborne = False
                self._active_initialized = False
            self._prev_phase = phase

        # ----- Steps 1 & 2: kickoff -----
        if phase == MatchPhase.Kickoff:
            return self._do_kickoff(packet)

        if phase != MatchPhase.Active:
            return ControllerState()

        # If we jumped to hit the kickoff, wait until we land before anything else.
        if self._kickoff_airborne:
            if my_car.air_state == AirState.OnGround:
                self._kickoff_airborne = False
                self._state = "RETURN"
            else:
                return ControllerState()

        # First active tick after a kickoff: get back into position.
        if not self._active_initialized:
            self._active_initialized = True
            self._state = "RETURN"

        physics = my_car.physics
        pos = Vec3(physics.location)
        vel = Vec3(physics.velocity)
        ori = Orientation(physics.rotation)
        angvel = Vec3(physics.angular_velocity)
        t = packet.match_info.seconds_elapsed
        on_ground = my_car.air_state == AirState.OnGround

        # ----- State transitions based on ground contact -----
        if on_ground:
            flat_dist = pos.flat().dist(self._goal_pos.flat())
            if flat_dist > GOAL_RADIUS:
                self._state = "RETURN"
            elif vel.length() < STARTUP_SPEED_GATE:
                # In position and slow enough — launch (don't reset an ongoing one).
                if self._state != "STARTUP":
                    self._state = "STARTUP"
                    self._startup_t0 = -1.0
            else:
                # In position but still moving fast — keep braking via RETURN.
                self._state = "RETURN"

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
            return self._do_return(my_car, pos, vel)
        if self._state == "STARTUP":
            return self._do_startup(t, my_car)
        # Steps 4, 6, 7: hover, strafe to target, maintain
        return self._hover_controls(pos, vel, ori, angvel, target_x, target_z)

    # ----- Step 1: whose kickoff is it? (ball served toward our goal) -----
    def _ball_coming_to_us(self, packet: GamePacket) -> bool:
        ball = packet.balls[0].physics
        vy = float(ball.velocity.y)
        if abs(vy) > 50:
            return (self._goal_y - float(ball.location.y)) * vy > 0
        # Ball not moving horizontally yet — peek ~1s into the prediction.
        if self.ball_prediction.slices:
            n = len(self.ball_prediction.slices)
            future = self.ball_prediction.slices[min(120, n - 1)].physics.location
            cur_y = float(ball.location.y)
            return abs(future.y - self._goal_y) < abs(cur_y - self._goal_y) - 50
        return False

    # ----- Steps 1 & 2: kickoff: go for it only if it's ours, then hit & land -----
    def _do_kickoff(self, packet: GamePacket) -> ControllerState:
        my_car = packet.players[self.index]
        my_pos = Vec3(my_car.physics.location)
        my_vel = Vec3(my_car.physics.velocity)
        ball = packet.balls[0].physics
        ball_pos = Vec3(ball.location)

        # Decide once the ball is actually moving; recheck until then.
        if not self._our_kickoff_decided:
            coming = self._ball_coming_to_us(packet)
            # No teammate should be closer for us to commit.
            teammate_closer = any(
                i != self.index
                and packet.players[i].team == self.team
                and Vec3(packet.players[i].physics.location).dist(ball_pos)
                < my_pos.dist(ball_pos)
                for i in range(len(packet.players))
            )
            self._our_kickoff = coming and not teammate_closer
            if abs(float(ball.velocity.y)) > 50:
                self._our_kickoff_decided = True

        controls = ControllerState()

        if not self._our_kickoff:
            # Not our kickoff — drive back and set up in goal.
            controls.throttle = 1.0
            controls.steer = steer_toward_target(my_car, self._goal_pos)
            return controls

        # Our kickoff: approach the ball, stop under it, then jump to pop it.
        ball_ground = Vec3(ball_pos.x, ball_pos.y, 0)
        flat_dist = my_pos.flat().dist(ball_ground.flat())

        if flat_dist > 250:
            # Approach
            controls.throttle = 1.0
            controls.boost = True
            controls.steer = steer_toward_target(my_car, ball_ground)
        elif my_vel.flat().length() > 120:
            # Under the ball but still rolling — brake to a stop first.
            controls.throttle = -1.0
        elif ball_pos.z < 300:
            # Stopped under the ball and it has dropped into range — jump to hit it.
            controls.jump = True
            self._kickoff_airborne = True
        # else: stopped, waiting for the ball to fall into range.

        return controls

    # ----- Step 2: drive back to position, braking so we don't overshoot -----
    def _do_return(self, my_car, pos: Vec3, vel: Vec3) -> ControllerState:
        controls = ControllerState()
        flat_dist = pos.flat().dist(self._goal_pos.flat())
        if flat_dist > 800:
            controls.throttle = 1.0
            controls.steer = steer_toward_target(my_car, self._goal_pos)
            if flat_dist > 2000:
                controls.boost = True
        else:
            # Close to position — kill momentum so we can launch cleanly.
            if vel.flat().length() > 150:
                controls.throttle = -1.0
            else:
                controls.throttle = 0.0
        return controls

    # ----- Step 3: jump up, tilt nose up, start boosting -----
    def _do_startup(self, t: float, my_car) -> ControllerState:
        if self._startup_t0 < 0:
            self._startup_t0 = t
        dt = t - self._startup_t0

        controls = ControllerState()
        if dt < 0.20:
            # Sustained single jump for height (NOT a 2nd press → no flip/dodge).
            controls.jump = True
        # Pitch the nose up the whole time; boost up once tilted enough.
        controls.pitch = 1.0
        ori = Orientation(my_car.physics.rotation)
        if ori.forward.z > BOOST_NOSE_MIN:
            controls.boost = True

        if my_car.air_state != AirState.OnGround and ori.forward.z > 0.8:
            # Airborne and nearly vertical — hand off to the hover controller.
            self._state = "HOVER"
        elif my_car.air_state == AirState.OnGround and dt > 0.9:
            # Never got airborne — retry.
            self._startup_t0 = -1.0

        return controls

    # ----- Steps 4, 6, 7: candle hover + lateral strafe -----
    def _hover_controls(
        self,
        pos: Vec3,
        vel: Vec3,
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
        # The velocity (damping) term stops it overshooting between the posts.
        tilt = clamp(
            LAT_TILT_GAIN * (target_x - pos.x) - LAT_DAMP * vel.x,
            -MAX_TILT,
            MAX_TILT,
        )
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

        # Altitude control by feathering boost. We target a vertical *speed*
        # proportional to the height error (capped), then boost only when we're
        # climbing slower than that target. Near the target height the desired
        # speed → 0, so boost pulses on/off to exactly cancel gravity (hover);
        # below it we boost to climb, above it we cut boost and fall.
        desired_vz = clamp(ALT_P * (target_z - pos.z), -MAX_CLIMB, MAX_CLIMB)
        controls.boost = f.z > BOOST_NOSE_MIN and vel.z < desired_vz
        return controls


if __name__ == "__main__":
    HeatseekGoalie("rlbot_community/heatseek_goalie").run()

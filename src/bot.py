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
UPRIGHT_MIN = 0.6         # car.up.z must exceed this to count as upright on the ground
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

ORI_P = 7.0               # orientation proportional gain (higher = tilts/reacts faster)
ORI_D = 2.25              # orientation derivative gain (damping; matches ORI_P so air
                          # recovery settles wheels-down instead of landing sideways)

CLIMB_DECEL = 550         # gravity-limited decel used to plan climb braking (UU/s^2),
                          # so we don't sail past the target height on the way up
MAX_CLIMB = 900           # cap on desired vertical speed (UU/s)
BOOST_NOSE_MIN = 0.5      # nose must point this far up before we allow boost

# Hover acceleration controller: compute a desired acceleration per axis, point
# the nose along (desired accel + gravity) so one boost serves every axis at the
# right ratio, then feather boost to track the vertical profile.
ACC_P = 4.0               # lateral position error -> accel (UU/s^2 per UU)
ACC_D = 6.0               # lateral velocity damping (UU/s^2 per UU/s)
ACC_P_Y = 4.0             # depth (distance-from-line) position gain
ACC_D_Y = 6.0             # depth velocity damping
ACC_Z = 3.0               # vertical: velocity error -> accel
A_HORIZ_MAX = 760         # cap on commanded horizontal accel (UU/s^2)
A_VERT_MAX = 340          # cap on commanded vertical accel (~boost up budget, UU/s^2)
GRAVITY = 650             # gravity the boost must counter (UU/s^2)
MAX_TILT = 1.17           # cap on horizontal/vertical thrust ratio (tilt ≈ 49.5°)
MIN_UP_THRUST = 150       # keep some up-thrust so the nose never points down
DESCENT_DECEL = 300       # vertical decel boost can manage when arresting a fall (UU/s^2)
BALL_DIR_SPEED = 400      # min ball y-speed (UU/s) to treat it as committed to a net

# Heatseeker homing simulation (RLBot's prediction ignores the homing curve, so
# we model it ourselves). The ball steers toward the goal each step.
HEATSEEKER_TURN_ACCEL = 700 # lateral steering accel toward goal (UU/s^2). Turn radius
                             # = speed^2/accel, so faster balls curve less — TUNE this
HEATSEEKER_TARGET_Z = 270    # height of the goal point the ball seeks (UU) — lower if
                             # the prediction still reads too high
HEATSEEKER_SIM_DT = 1.0 / 60 # simulation timestep (s)
HEATSEEKER_SIM_TIME = 5.0    # how far ahead to simulate (s)

# Ball intercept: where to meet the ball, clamped to a reachable goal-mouth area.
INTERCEPT_X = 850         # max lateral reach chasing the ball (UU, near the posts)
INTERCEPT_Z_MIN = 60      # lowest height to chase to (UU)
INTERCEPT_Z_MAX = 620     # highest height to chase to (UU, near the crossbar)
# If the predicted impact is beyond this (clearly going to miss), idle instead of
# chasing the clamped target. Posts are ~893 wide / 642 tall, so these leave a
# margin: a just-outside shot is still covered, a way-wide one is ignored.
SUPER_FAR_X = 2200        # idle if predicted impact x is wider than this (UU)
SUPER_FAR_Z = 1400        # idle if predicted impact z is higher than this (UU)

# ---- Movement test mode ----
# True  -> patrol the goal corners (predictable targets for tuning the movement).
# False -> real ball-tracking targeting. The hover controller is the same either
# way, so movement tuning here transfers straight to the ball version.
PATROL_CORNERS = True
CORNER_X = 750            # lateral reach toward each goalpost (UU)
CORNER_LOW_Z = 160        # low corner height (UU)
CORNER_HIGH_Z = 500       # high corner height (UU)
CORNER_REACH = 80         # advance to the next corner once within this (UU)
CORNER_DWELL = 2.5        # seconds to hold each corner (watch it settle) before moving on


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class HeatseekGoalie(Bot):
    # State machine: DRIVE -> LAUNCH -> HOVER
    _state: str = "DRIVE"
    _launch_t0: float = -1.0
    _goal_pos: Vec3 = Vec3(0, 0, 0)
    _goal_line_y: float = 0.0
    _toward_field_y: float = 1.0
    _target_x: float = 0.0
    _target_z: float = HOVER_HEIGHT
    _sim_path: list[Vec3] = []  # last simulated homing path, for debug rendering
    _corner_idx: int = 0
    _corners: list[tuple[float, float]] = []
    _corner_arrive_t: float = -1.0

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

        # Patrol targets (x, z) over all four corners, ordered so the across-net
        # moves are diagonal (up + sideways): bottom-left -> top-right (up-right),
        # then down to bottom-right -> top-left (up-left), then down to repeat.
        self._corners = [
            (-CORNER_X, CORNER_LOW_Z),   # bottom-left
            (CORNER_X, CORNER_HIGH_Z),   # top-right   (diagonal up-right)
            (CORNER_X, CORNER_LOW_Z),    # bottom-right
            (-CORNER_X, CORNER_HIGH_Z),  # top-left    (diagonal up-left)
        ]

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

        # On kickoff (e.g. just after a goal), reset the hover target to center.
        # During active play, update where we want to be based on the ball.
        if phase == MatchPhase.Kickoff:
            self._target_x = 0.0
            self._target_z = HOVER_HEIGHT
        elif PATROL_CORNERS:
            self._patrol_corners(pos, t)
        else:
            self._update_target(packet)

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
        target_world = Vec3(self._target_x, self._goal_pos.y, self._target_z)
        self.renderer.draw_line_3d(pos, target_world, self.renderer.cyan)
        # Red curve = our simulated Heatseeker homing path (tune SEEK_RATE so it
        # matches where the ball actually goes).
        for i in range(len(self._sim_path) - 1):
            self.renderer.draw_line_3d(
                self._sim_path[i], self._sim_path[i + 1], self.renderer.red
            )
        self.renderer.draw_string_3d(
            f"{self._state} z={pos.z:.0f} tgt=({self._target_x:.0f},{self._target_z:.0f})",
            pos, 1, self.renderer.white,
        )
        self.renderer.end_rendering()

        # ---- Dispatch ----
        # Stuck on our side / roof on the ground -> jump to pop airborne AND roll
        # toward wheels-down at the same time, so the moment we get any air we're
        # already righting ourselves (otherwise we're wheels-sideways, can't drive).
        if on_ground and ori.up.z < UPRIGHT_MIN:
            self._state = "DRIVE"
            controls = self._air_recover(pos, ori, angvel)
            controls.jump = True
            return controls
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
    # Simulate the Heatseeker homing ourselves and find where the ball will
    # cross our defending plane (our y). RLBot's built-in ball prediction does
    # NOT model the Heatseeker curve (it's plain Soccar physics), so we roll our
    # own: each step we steer the ball's velocity toward the goal it's seeking
    # while keeping its speed, then advance. Returns (x, z) clamped to the goal,
    # or None if it never crosses our plane.
    # ---------------------------------------------------------------
    def _simulate_heatseeker(self, pos: Vec3, vel: Vec3) -> "tuple[float, float] | None":
        speed = vel.length()
        if speed < 1.0:
            return None

        # The ball seeks our goal (it's coming at us). Aim at the goal center.
        goal_target = Vec3(0.0, self._goal_line_y, HEATSEEKER_TARGET_Z)
        defend_y = self._goal_pos.y
        dt = HEATSEEKER_SIM_DT
        steps = int(HEATSEEKER_SIM_TIME / dt)

        self._sim_path = [pos]  # record the path for debug rendering
        for i in range(steps):
            to_goal = goal_target - pos
            to_goal_len = to_goal.length()
            if to_goal_len > 1.0:
                to_goal_dir = to_goal * (1.0 / to_goal_len)
                v_dir = vel * (1.0 / speed)
                # Steer with a fixed LATERAL acceleration toward the goal (only the
                # part perpendicular to our heading). Turn radius = speed^2 / accel,
                # so a faster ball curves LESS over a given distance and a slow ball
                # curves tightly -- i.e. ball speed is accounted for in the curve.
                align = clamp(v_dir.dot(to_goal_dir), -1.0, 1.0)
                lateral = to_goal_dir - v_dir * align
                lat_len = lateral.length()
                if lat_len > 1e-3:
                    lateral = lateral * (1.0 / lat_len)
                    vel = vel + lateral * (HEATSEEKER_TURN_ACCEL * dt)
                    vlen = vel.length()
                    if vlen > 1.0:
                        vel = vel * (speed / vlen)  # homing keeps speed ~constant

            # Homes to the goal center: when the ball is already below center it
            # curves UP toward it, never further down.
            if pos.z < HEATSEEKER_TARGET_Z and vel.z < 0.0:
                vel = Vec3(vel.x, vel.y, 0.0)

            new_pos = pos + vel * dt
            if i % 4 == 0:
                self._sim_path.append(new_pos)
            if (pos.y - defend_y) * (new_pos.y - defend_y) <= 0.0 and pos.y != new_pos.y:
                frac = (defend_y - pos.y) / (new_pos.y - pos.y)
                x = pos.x + frac * (new_pos.x - pos.x)
                z = pos.z + frac * (new_pos.z - pos.z)
                self._sim_path.append(new_pos)
                return (x, z)  # raw crossing; caller decides clamp vs idle
            pos = new_pos
        return None

    # ---------------------------------------------------------------
    # Decide our hover target based on where the ball is going:
    #   - homing AT our net   -> aim at its simulated impact point
    #   - homing toward THEM  -> recenter (we / the wall just cleared it)
    #   - not committed       -> hold our current target
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # Test mode: cycle the hover target around the goal corners so we can tune
    # the movement against predictable targets (no ball involved).
    # ---------------------------------------------------------------
    def _patrol_corners(self, pos: Vec3, t: float):
        self._sim_path = []
        tx, tz = self._corners[self._corner_idx]
        reached = abs(pos.x - tx) < CORNER_REACH and abs(pos.z - tz) < CORNER_REACH
        if reached and self._corner_arrive_t < 0:
            self._corner_arrive_t = t  # first arrival -> start the dwell timer
        # Hold the corner for CORNER_DWELL seconds (so we can watch it settle),
        # then move to the next one.
        if self._corner_arrive_t >= 0 and t - self._corner_arrive_t > CORNER_DWELL:
            self._corner_idx = (self._corner_idx + 1) % len(self._corners)
            self._corner_arrive_t = -1.0
            tx, tz = self._corners[self._corner_idx]
        self._target_x, self._target_z = tx, tz

    def _update_target(self, packet: GamePacket):
        self._sim_path = []  # cleared; repopulated only if we simulate a shot
        ball = packet.balls[0].physics
        ball_pos = Vec3(ball.location)
        ball_vel = Vec3(ball.velocity)

        # >0 means moving toward the field/opponent, <0 means toward our net.
        threat = float(ball.velocity.y) * self._toward_field_y
        if threat < -BALL_DIR_SPEED:
            # Coming at us -> aim where the homing curve will bring it to our plane.
            crossing = self._simulate_heatseeker(ball_pos, ball_vel)
            if crossing is not None:
                raw_x, raw_z = crossing
                if abs(raw_x) > SUPER_FAR_X or raw_z > SUPER_FAR_Z:
                    # Impact is way wide/high of the goal -> it'll clearly miss, so
                    # don't bother chasing it; just idle where we are.
                    pass
                else:
                    # On or near the goal -> go to the impact, clamped to our reach
                    # (so a near-post/just-outside shot still gets covered).
                    self._target_x = clamp(raw_x, -INTERCEPT_X, INTERCEPT_X)
                    self._target_z = clamp(raw_z, INTERCEPT_Z_MIN, INTERCEPT_Z_MAX)
        elif threat > BALL_DIR_SPEED:
            # Ball was hit back toward the opponent (we / the wall cleared it) ->
            # threat is gone. Drop the old impact target and idle/stabilize right
            # where we are, instead of flying off to a now-stale target.
            car_loc = packet.players[self.index].physics.location
            self._target_x = float(car_loc.x)
            self._target_z = clamp(float(car_loc.z), INTERCEPT_Z_MIN, INTERCEPT_Z_MAX)
        # else: ball not committed to a direction -> hold current target.

    # ---------------------------------------------------------------
    # 3) Hover: stay vertical (nose straight up) and hold height.
    #    Orientation is corrected with proportional pitch/yaw/roll inputs
    #    (a "calculated amount" each tick, not a held button), and altitude
    #    is held by feathering boost on/off to track a target climb rate.
    # ---------------------------------------------------------------
    def _hover(
        self, pos: Vec3, vel: Vec3, ori: Orientation, angvel: Vec3
    ) -> ControllerState:
        # Target (_target_x/_target_z) is set each tick in get_output via
        # _update_target, based on where the ball is homing.
        target_x, target_z = self._target_x, self._target_z

        # --- Desired acceleration per axis (PD on position), capped to what the
        # candle can actually produce. ---
        a_x = clamp(
            ACC_P * (target_x - pos.x) - ACC_D * vel.x, -A_HORIZ_MAX, A_HORIZ_MAX
        )
        a_y = clamp(
            ACC_P_Y * (self._goal_pos.y - pos.y) - ACC_D_Y * vel.y,
            -A_HORIZ_MAX,
            A_HORIZ_MAX,
        )

        # Vertical: track a braking velocity profile both ways (so we never sail
        # past the target height). Climb is gravity-limited (we can't thrust down),
        # descent is boost-limited.
        dz = target_z - pos.z
        if dz < 0.0:
            desired_vz = -min(MAX_CLIMB, math.sqrt(2.0 * DESCENT_DECEL * -dz))
        else:
            desired_vz = min(MAX_CLIMB, math.sqrt(2.0 * CLIMB_DECEL * dz))
        a_z = clamp(ACC_Z * (desired_vz - vel.z), -A_VERT_MAX, A_VERT_MAX)

        # --- Point the nose along the thrust we need (desired accel + gravity), so
        # one boost serves climb AND strafe in the right proportion. Limit the tilt
        # for stability. ---
        up_thrust = max(a_z + GRAVITY, MIN_UP_THRUST)
        max_horiz = MAX_TILT * up_thrust
        horiz = math.hypot(a_x, a_y)
        if horiz > max_horiz and horiz > 1e-3:
            s = max_horiz / horiz
            a_x *= s
            a_y *= s

        f_d = Vec3(a_x, a_y, up_thrust)
        flen = f_d.length()
        f_d = f_d * (1.0 / flen) if flen > 1e-3 else Vec3(0.0, 0.0, 1.0)
        up_ref = Vec3(0.0, self._toward_field_y, 0.0)
        right_d = up_ref.cross(f_d).normalized()
        up_d = f_d.cross(right_d).normalized()
        controls = self._orient_controls(ori, angvel, f_d, right_d, up_d)

        # Feather boost to track the vertical profile. Because the nose is tilted
        # toward the target, each boost also pushes us horizontally -- so this one
        # duty cycle drives the climb/descent AND the strafe together, in the ratio
        # the nose is pointing. No separate strafe-boost needed, so no overshoot.
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

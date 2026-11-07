"""
One arm playing Push-T (PushTTask): the diffusion policy trained in the 2-D Push-T simulator
(../diffusion_policy) drives the arm's pusher rod on the MuJoCo table, one control tick at a time.
pusht_sim.py runs one arm, dual_pusht_sim.py two.

Phases:
  reach    SQP plan + PathVelocityField, as in reach_sim: the probe goes to HOVER_HEIGHT above a
           pre-push spot next to the T -- clear of it, so coming down can't land on it
  descend  the probe goes down to the pre-push spot at PUSH_HEIGHT
  push     every 0.1 s (Push-T's 10 Hz): T keypoints + probe position -> Push-T pixels -> policy ->
           8 target positions, used one per 0.1 s; every tick a Cartesian DS drives the probe toward
           the current target, through the CBF (the rod may touch the T, nothing else may). The policy
           runs in a background thread, so the sim keeps going while it thinks (pusht.policy.ChunkedPolicy)
  lift     the T covers SUCCESS_COVERAGE of the goal (Push-T's success rule) or PUSH_TIME_LIMIT
           passes. The probe rises straight up to RETREAT_HEIGHT above the T's top face -- the rod may
           still touch the T, but never pushes it
  hover    the probe glides over the T's centre of mass and holds there, following the T if it moves
"""
import time

import numpy as np

from core.planner import plan_trajectory
from mj.model import TOOL_DOWN_QUAT
from mj.tasks import IK_DAMPING, probe_ds
from mj.utils import draw_sphere

HOVER_HEIGHT = 0.10      # m: how far above PUSH_HEIGHT the reach phase leaves the probe
PUSH_HEIGHT = 0.025      # m: probe (rod tip center) height while pushing -- the T is 4 cm tall, the tip clears the table by 1 cm
PREPUSH_MARGIN_PX = 20   # px: the pre-push spot stays this far inside the Push-T workspace
REACH_TOL = 0.02         # rad: the reach phase is done once the arm is this close to the planned path's end...
REACH_PROBE_TOL = 0.03   # m: ...or the probe this close to the hover point (the descend DS takes care of the rest)
DESCEND_TOL = 0.005      # m: at push height once the probe is this close
RETREAT_HEIGHT = 0.10    # m: once pushing is over, the probe hovers this far above the T's top face
LIFT_TOL = 0.01          # m: lifted once the probe is this close to RETREAT_HEIGHT straight above where it stopped

PROBE_GAIN = 10.0        # 1/s: Push-T's pusher PD (k_p 100, k_v 20) is critically damped with time constant 0.1 s
PROBE_MAX_SPEED = 0.3    # m/s
TASK_METRIC_REG = 0.01   # how much the CBF still minds joint motion the probe task doesn't care about

POLICY_DT = 0.1          # s: Push-T's control period
PREFETCH_ACTIONS = 2     # start inferring the next chunk when this few actions are left, so it's ready in time
SUCCESS_COVERAGE = 0.9   # Push-T's success threshold: fraction of the goal T covered by the block
PUSH_TIME_LIMIT = 300.0  # s of sim time in the push phase


def above_block(arm):
    """RETREAT_HEIGHT above the T's top face, over its centre of mass: where the probe hovers once pushing is over."""
    top = max(box.center[2] + box.half_extents[2] for box in arm.env.update(arm.data))  # the T stands upright
    return np.append(arm.data.body("t_block").xipos[:2], top + RETREAT_HEIGHT)


def lift_point(arm):
    """Straight above the probe, as high as above_block: where the lift phase takes it."""
    return np.append(arm.probe_pos()[:2], above_block(arm)[2])


def draw_targets(scn, frame, current_px, queued_px):
    """The policy's current target (larger) and its queued ones as spheres at push height."""
    if current_px is not None:
        draw_sphere(scn, np.append(frame.to_table(current_px), PUSH_HEIGHT), 0.008, (0.1, 0.4, 1.0, 0.9))
    for px in queued_px:
        draw_sphere(scn, np.append(frame.to_table(px), PUSH_HEIGHT), 0.004, (0.1, 0.4, 1.0, 0.5))


class PushTTask:
    """One arm playing Push-T: the phases above, one control tick at a time. The caller syncs the arm's
    model before each control() and sends its joint velocity to the servos.

    prepush_spot(data) -> table xy to start pushing from, next to the T where it is in data
    hover_spot(data) -> table xy to hover over before descending to the pre-push spot; None: right over it
    note(so_far) -> text for the end of the push log lines, e.g. the obstacle monitor's summary"""

    def __init__(self, arm, safety, frame, policy, prepush_spot, name=None, note=None, hover_spot=None):
        self.arm, self.safety, self.frame, self.policy = arm, safety, frame, policy
        self.prepush_spot, self.hover_spot, self.name = prepush_spot, hover_spot, name
        self.note = note or (lambda so_far: "")
        self.phase, self.field, self.path_end = None, None, None
        self.next_policy_time, self.push_start, self.lift_to = 0.0, None, None
        self.succeeded = False

    @property
    def finished(self):
        """Pushing is over: success or the time limit."""
        return self.phase in ("lift", "hover")

    def log(self, now, message):
        print(f"[{now:6.2f}s] " + (f"{self.name:6s} " if self.name else "") + message)

    def plan(self, planner, rng, boxes=()):
        """Reach: plan to hover above the pre-push spot next to the T (or over hover_spot), rod pointing
        straight down, and follow that path. boxes: more to keep clear of, as SafetyFilter.filter's -- e.g. another arm's T. Returns
        plan_trajectory's plan; None if there's no IK goal, and then it descends with the Cartesian DS."""
        arm, frame = self.arm, self.frame
        who = f"{self.name}: " if self.name else ""
        block_px, block_angle = frame.block_pose_px(arm.data)
        print(f"{who}T start (Push-T px) {np.round(block_px, 1).tolist()}, {np.degrees(block_angle):.1f} deg; "
              f"goal covered {frame.coverage(arm.data) * 100:.1f}%")
        spot = self.prepush_spot(arm.data)
        hover_xy = spot if self.hover_spot is None else self.hover_spot(arm.data)
        self.hover_point = np.append(hover_xy, PUSH_HEIGHT + HOVER_HEIGHT)
        self.push_start_point = np.append(spot, PUSH_HEIGHT)
        self.hover_flange = self.hover_point + arm.pusher.flange_above_probe
        t0 = time.perf_counter()
        plan = plan_trajectory(arm.model, arm.data, arm.geometry, arm.env, arm.q, self.hover_flange, TOOL_DOWN_QUAT,
                               planner=planner, rng=rng, boxes=boxes)
        print(f"{who}pre-push spot {np.round(frame.to_px(spot), 1).tolist()} px"
              + ("" if self.hover_spot is None else f", hovering over {np.round(frame.to_px(hover_xy), 1).tolist()} px")
              + f", planned in {time.perf_counter() - t0:.3f}s")
        if plan is None:
            print(f"{who}WARNING: no IK candidate converged -- descending with the Cartesian DS")
            self.phase = "descend"
            return None
        path = plan.best.path
        self.field, self.path_end, self.phase = arm.path_field(path), path[-1], "reach"
        return plan

    def _probe_cmd(self, target, touch_block, caps, others, boxes, obstacle):
        """One tick of the probe DS toward target, through the CBF."""
        arm = self.arm
        u_des, metric = probe_ds(arm, target, PROBE_GAIN, PROBE_MAX_SPEED, arm.robot.k_o, IK_DAMPING, TASK_METRIC_REG)
        return self.safety.filter(u_des, caps=caps, others=others, boxes=boxes, obstacle=obstacle, metric=metric,
                                  touch_block=touch_block)

    def _finish(self, now):
        """Pushing is over -- success, or out of time: lift."""
        coverage = self.frame.coverage(self.arm.data)
        self.succeeded = coverage >= SUCCESS_COVERAGE
        self.phase, self.lift_to = "lift", lift_point(self.arm)
        self.policy.stop()
        self.log(now, f"{'SUCCESS' if self.succeeded else 'time limit'}: goal covered {coverage * 100:.1f}% after "
                      f"{now - self.push_start:.1f}s of pushing{self.note(False)}")

    def control(self, now, caps, obstacle=None, others=None, boxes=()):
        """This control tick's joint velocity. caps: the arm's RobotGeometry.world this tick; obstacle:
        the obstacle's capsule this tick, or None; others, boxes: another arm's capsules and T, if any
        (all as SafetyFilter.filter's)."""
        arm, frame, policy = self.arm, self.frame, self.policy
        if self.phase == "push" and now >= self.next_policy_time:  # Push-T's 10 Hz tick
            self.next_policy_time += POLICY_DT
            coverage = frame.coverage(arm.data)
            if coverage >= SUCCESS_COVERAGE or now - self.push_start >= PUSH_TIME_LIMIT:
                self._finish(now)
            else:
                policy.observe(frame.observation(arm.data))
                late = policy.collect(now)
                if late is not None:
                    self.log(now, f"goal covered {coverage * 100:5.1f}% | new action chunk, {late} tick(s) late, "
                                  f"{len(policy.queue)} actions left{self.note(True)}")
                policy.next_target(now)

        if self.phase == "reach":
            u = self.safety.filter(self.field(arm.q), caps=caps, others=others, boxes=boxes, obstacle=obstacle)
            if (np.linalg.norm(arm.q - self.path_end) < REACH_TOL
                    or np.linalg.norm(arm.probe_pos() - self.hover_point) < REACH_PROBE_TOL):
                self.phase = "descend"
                self.log(now, "reached the hover point -- descending")
            return u
        if self.phase == "descend":
            u = self._probe_cmd(self.push_start_point, False, caps, others, boxes, obstacle)
            if np.linalg.norm(arm.probe_pos() - self.push_start_point) < DESCEND_TOL:
                self.phase, self.push_start, self.next_policy_time = "push", now, now
                self.log(now, "at push height -- policy takes over")
            return u
        if self.phase == "push":  # toward the policy's current target; until the first lands, hold the start point
            target = (self.push_start_point if policy.target is None
                      else np.append(frame.to_table(policy.target), PUSH_HEIGHT))
            return self._probe_cmd(target, True, caps, others, boxes, obstacle)
        if self.phase == "lift":  # straight up: the rod may still touch the T, but rising never pushes it
            u = self._probe_cmd(self.lift_to, True, caps, others, boxes, obstacle)
            if np.linalg.norm(arm.probe_pos() - self.lift_to) < LIFT_TOL:
                self.phase = "hover"
                self.log(now, f"lifted -- hovering {RETREAT_HEIGHT * 100:.0f} cm above the T")
            return u
        return self._probe_cmd(above_block(arm), False, caps, others, boxes, obstacle)  # hover: over the T, wherever it is now

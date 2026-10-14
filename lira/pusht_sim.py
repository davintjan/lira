"""
Push-T sim: the diffusion policy trained in the 2-D Push-T simulator (../diffusion_policy) drives the
arm's pusher rod on the MuJoCo table.

Phases (one state machine in main):
  reach    SQP plan + PathVelocityField, as in reach_sim: the probe goes to HOVER_HEIGHT above a
           pre-push spot next to the T -- clear of it, so coming down can't land on it
  descend  the probe goes straight down to PUSH_HEIGHT
  push     every 0.1 s (Push-T's 10 Hz): T keypoints + probe position -> Push-T pixels -> policy ->
           8 target positions, used one per 0.1 s; every tick a Cartesian DS drives the probe toward
           the current target, through the CBF (the rod may touch the T, nothing else may). The policy
           runs in a background thread, so the sim keeps going while it thinks (pusht.policy.ChunkedPolicy)
  done     the T covers SUCCESS_COVERAGE of the goal (Push-T's success rule) or PUSH_TIME_LIMIT
           passes; the probe holds still, and on success the obstacle stops where it is

The policy only ever sees Push-T pixels: PushTFrame maps the table to the Push-T workspace at
1 px = 1 mm, axes aligned, with Push-T's fixed goal pose on world.xml's goal body.

Run from this directory:  .venv/bin/python pusht_sim.py [--robot ur5e] [--seed N] [--no-obstacle] [--record out.mp4]
"""
import argparse
import contextlib
import os
import pathlib
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering for --record; must precede `import mujoco`
# the viewer window: the glfw package's Wayland build crashes on a GNOME setting it can't read (GLib-GIO-ERROR ... antialiasing), its X11 build runs fine through XWayland
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco
import mujoco.viewer
import numpy as np
import torch

from common import add_common_args, make_planner, resolve_seed
from core.path_field import PathVelocityField
from core.planner import plan_trajectory
from mj.arm import Arm
from mj.model import TOOL_DOWN_QUAT
from mj.recorder import Recorder
from mj.robots import ROBOTS
from mj.safety import CBFConfig, ObstacleMonitor, SafetyFilter
from mj.scene import MovingObstacle
from mj.tasks import probe_ds
from mj.utils import draw_sphere
from pusht.frame import PushTFrame
from pusht.policy import REFERENCE_CHECKPOINT, ChunkedPolicy, load_policy

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CBF = CBFConfig()

HOVER_HEIGHT = 0.10      # m: how far above PUSH_HEIGHT the reach phase leaves the probe
PUSH_HEIGHT = 0.025      # m: probe (rod tip center) height while pushing -- the T is 4 cm tall, the tip clears the table by 1 cm
PREPUSH_DISTANCE = 0.13  # m from the T's center of mass: clear of the T (it reaches ~0.076 m), its CBF margin and the rod
PREPUSH_MARGIN_PX = 20   # px: the pre-push spot stays this far inside the Push-T workspace
REACH_TOL = 0.02         # rad: the reach phase is done once the arm is this close to the planned path's end...
REACH_PROBE_TOL = 0.03   # m: ...or the probe this close to the hover point (the descend DS takes care of the rest)
DESCEND_TOL = 0.005      # m: at push height once the probe is this close

PROBE_GAIN = 10.0        # 1/s: Push-T's pusher PD (k_p 100, k_v 20) is critically damped with time constant 0.1 s
PROBE_MAX_SPEED = 0.3    # m/s
IK_DAMPING = 1e-2        # damped least squares, keeps J^+ finite near singularities
TASK_METRIC_REG = 0.01   # how much the CBF still minds joint motion the probe task doesn't care about

POLICY_DT = 0.1          # s: Push-T's control period
PREFETCH_ACTIONS = 2     # start inferring the next chunk when this few actions are left, so it's ready in time
SUCCESS_COVERAGE = 0.9  # Push-T's success threshold: fraction of the goal T covered by the block
PUSH_TIME_LIMIT = 300.0   # s of sim time in the push phase

OBSTACLE_CENTER = [0.3545, 0.146, 0.2589]  # m, world frame: sweep around this point; None = OBSTACLE_HEIGHT above the T's start
OBSTACLE_HEIGHT = 0.30   # m: above the T, through the wrist/forearm while pushing (the rod tip is at 2.5 cm, flange ~17.5 cm)
OBSTACLE_AXIS = [0, 1, 0]  # world frame: sweeps left and right along this
OBSTACLE_AMPLITUDE = 0.45  # m either side of its center
OBSTACLE_SPEED = 0.2       # m/s, constant
OBSTACLE_DWELL = 5.0       # s: starts at -OBSTACLE_AMPLITUDE (along OBSTACLE_AXIS), crosses, parks this long at
                           # each end before crossing back; None = sweep back and forth without stopping


def make_obstacle(arm):
    """The obstacle sweeping through the arm while it pushes."""
    if OBSTACLE_CENTER is not None:
        center, where = np.asarray(OBSTACLE_CENTER, dtype=float), "set by OBSTACLE_CENTER"
    else:
        center, where = np.append(arm.data.body("t_block").xipos[:2], OBSTACLE_HEIGHT), f"{OBSTACLE_HEIGHT} m above the T"
    obstacle = MovingObstacle(arm.model, center, axis=OBSTACLE_AXIS, amplitude=OBSTACLE_AMPLITUDE,
                              speed=OBSTACLE_SPEED, dwell=OBSTACLE_DWELL)
    motion = (f"crosses +-{OBSTACLE_AMPLITUDE} m along {OBSTACLE_AXIS}, waiting {OBSTACLE_DWELL} s at each end"
              if OBSTACLE_DWELL is not None else f"sweeps +-{OBSTACLE_AMPLITUDE} m along {OBSTACLE_AXIS}")
    print(f"Obstacle: around {np.round(obstacle.center, 3).tolist()} ({where}), {motion} at {OBSTACLE_SPEED} m/s")
    return obstacle


def park_obstacle(arm):
    arm.data.mocap_pos[arm.model.body_mocapid[arm.model.body("obstacle").id]] = [-0.5, 0.0, 1.0]  # behind the robot


def draw_targets(scn, frame, current_px, queued_px):
    """The policy's current target (larger) and its queued ones as spheres at push height."""
    if current_px is not None:
        draw_sphere(scn, np.append(frame.to_table(current_px), PUSH_HEIGHT), 0.008, (0.1, 0.4, 1.0, 0.9))
    for px in queued_px:
        draw_sphere(scn, np.append(frame.to_table(px), PUSH_HEIGHT), 0.004, (0.1, 0.4, 1.0, 0.5))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--no-obstacle", action="store_true", help="park the obstacle out of the way")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=REFERENCE_CHECKPOINT)
    args = parser.parse_args(argv)
    robot = ROBOTS[args.robot]

    seed = resolve_seed(args.seed)
    print(f"Seed: {seed}   (--seed {seed} replays IK seeds and diffusion noise -- inference timing still varies run to run)")
    torch.manual_seed(seed)

    arm = Arm(robot, joint_vel_max=CBF.joint_vel_max)
    safety = SafetyFilter(arm, CBF)
    frame = PushTFrame(arm.model)
    print(f"Robot: {robot.name}. Loading policy {args.checkpoint.name} on {DEVICE}...")
    policy = ChunkedPolicy(load_policy(args.checkpoint, DEVICE), DEVICE, period=POLICY_DT, prefetch=PREFETCH_ACTIONS)

    block_px, block_angle = frame.block_pose_px(arm.data)
    print(f"T start (Push-T px): pos {np.round(block_px, 1).tolist()}, angle {np.degrees(block_angle):.1f} deg; "
          f"goal covered {frame.coverage(arm.data) * 100:.1f}%")

    # reach: plan to hover above the pre-push spot, rod pointing straight down
    spot = frame.prepush_spot(arm.data, PREPUSH_DISTANCE, PREPUSH_MARGIN_PX)
    print(f"Pre-push spot: table {np.round(spot, 3).tolist()} m = {np.round(frame.to_px(spot), 1).tolist()} px")
    hover_point = np.append(spot, PUSH_HEIGHT + HOVER_HEIGHT)
    push_start_point = np.append(spot, PUSH_HEIGHT)
    planner = make_planner(args.planner, CBF.margin)
    t0 = time.perf_counter()
    plan = plan_trajectory(arm.model, arm.data, arm.geometry, arm.env, arm.q,
                           hover_point + arm.pusher.flange_above_probe, TOOL_DOWN_QUAT,
                           planner=planner, rng=np.random.default_rng(seed))
    print(f"Planning took {time.perf_counter() - t0:.3f}s")
    if plan is None:
        print("WARNING: no IK candidate converged -- skipping the reach phase, descending with the Cartesian DS")
        field, path_end, phase = None, None, "descend"
    else:
        path = plan["best"]["path"]
        field = PathVelocityField(path, k_tangent=robot.path_k_tangent, k_corrective=robot.path_k_corrective)
        path_end, phase = path[-1], "reach"

    if args.no_obstacle:
        obstacle, monitor = None, None
        park_obstacle(arm)
    else:
        obstacle, monitor = make_obstacle(arm), ObstacleMonitor(arm.model)

    def obstacle_note(so_far):
        return f" | {monitor.summary(so_far)}" if monitor else ""

    def probe_to(target, touch_block, obstacle_now, caps):
        """One tick of the probe DS through the safety filter."""
        u_des, metric = probe_ds(arm, target, PROBE_GAIN, PROBE_MAX_SPEED, robot.k_o, IK_DAMPING, TASK_METRIC_REG)
        arm.command(safety.filter(u_des, caps=caps, obstacle=obstacle_now, metric=metric, touch_block=touch_block))

    def draw_overlays(scn):
        """The policy's targets, while it's pushing."""
        draw_targets(scn, frame, policy.target if phase == "push" else None, policy.queue)

    dt = arm.model.opt.timestep
    next_policy_time, push_start, hold_point = 0.0, None, None
    recording = Recorder(arm.model, args.record) if args.record else contextlib.nullcontext()
    with mujoco.viewer.launch_passive(arm.model, arm.data) as viewer, recording as recorder:
        while viewer.is_running():
            t_wall = time.perf_counter()
            now = arm.data.time
            obstacle_now = obstacle.update(arm.data) if obstacle is not None else None
            caps = arm.geometry.world(arm.data)  # every capsule in world frame, once per tick
            if obstacle_now is not None and monitor.update(arm.data, caps["env"], obstacle_now):
                print(f"[{now:6.2f}s] WARNING: the obstacle hit the robot")

            if phase == "push" and now >= next_policy_time:  # Push-T's 10 Hz tick
                next_policy_time += POLICY_DT
                coverage = frame.coverage(arm.data)
                if coverage >= SUCCESS_COVERAGE or now - push_start >= PUSH_TIME_LIMIT:
                    phase, hold_point = "done", arm.probe_pos()
                    policy.stop()
                    verdict = "SUCCESS" if coverage >= SUCCESS_COVERAGE else "time limit"
                    if coverage >= SUCCESS_COVERAGE and obstacle is not None:
                        obstacle.freeze(arm.data)  # task done: the obstacle stops where it is
                    print(f"[{now:6.2f}s] {verdict}: goal covered {coverage * 100:.1f}% after "
                          f"{now - push_start:.1f}s of pushing{obstacle_note(so_far=False)}")
                else:
                    policy.observe(frame.observation(arm.data))
                    late = policy.collect(now)
                    if late is not None:
                        print(f"[{now:6.2f}s] goal covered {coverage * 100:5.1f}% | new action chunk, "
                              f"{late} tick(s) late, {len(policy.queue)} actions left{obstacle_note(so_far=True)}")
                    policy.next_target(now)

            if phase == "reach":
                arm.command(safety.filter(field(arm.q), caps=caps, obstacle=obstacle_now))
                if (np.linalg.norm(arm.q - path_end) < REACH_TOL
                        or np.linalg.norm(arm.probe_pos() - hover_point) < REACH_PROBE_TOL):
                    phase = "descend"
                    print(f"[{now:6.2f}s] reached the hover point -- descending")

            elif phase == "descend":
                probe_to(push_start_point, False, obstacle_now, caps)
                if np.linalg.norm(arm.probe_pos() - push_start_point) < DESCEND_TOL:
                    phase, push_start, next_policy_time = "push", now, now
                    print(f"[{now:6.2f}s] at push height -- policy takes over")

            elif phase == "push":  # toward the policy's current target; until the first lands, hold the start point
                target = push_start_point if policy.target is None else np.append(frame.to_table(policy.target), PUSH_HEIGHT)
                probe_to(target, True, obstacle_now, caps)

            else:  # done: hold still where the probe stopped
                probe_to(hold_point, True, obstacle_now, caps)

            arm.step()
            viewer.user_scn.ngeom = 0
            draw_overlays(viewer.user_scn)
            if recorder is not None:
                recorder.capture(arm.data, draw_overlays)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - t_wall)))  # pace to real time


if __name__ == "__main__":
    main()

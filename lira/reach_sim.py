"""
Reach sim: the arm plans a collision-free path to hover over the T, then follows it while a CBF keeps
it clear of itself, the table, the T and an obstacle sweeping through the workspace.

  plan     once, before the loop (~1 s, too slow for the 500 Hz loop): IK candidates + gradient
           descent or SQP (--planner), cheapest feasible path wins
  follow   every tick: PathVelocityField -> SafetyFilter -> velocity servos. If planning failed,
           a straight-line pose DS stands in for the field.

Run from this directory:  .venv/bin/python reach_sim.py [--robot ur5e] [--seed N] [--record out.mp4]
"""
import argparse
import contextlib
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering for --record; must precede `import mujoco`
# the viewer window: the glfw package's Wayland build crashes on a GNOME setting it can't read (GLib-GIO-ERROR ... antialiasing), its X11 build runs fine through XWayland
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco
import mujoco.viewer
import numpy as np

from common import add_common_args, make_planner, print_field, print_plan, resolve_seed
from core.path_field import PathVelocityField
from core.planner import plan_trajectory
from mj.arm import Arm
from mj.model import TOOL_DOWN_QUAT
from mj.recorder import Recorder
from mj.robots import ROBOTS
from mj.safety import CBFConfig, SafetyFilter
from mj.scene import MovingObstacle
from mj.tasks import flange_pose_ds
from mj.utils import draw_capsules

CBF = CBFConfig()
HOVER_OFFSET = [0, 0, 0.1]  # m: where the probe hovers relative to the T's center of mass
MAX_SPEED = 0.8             # m/s: cap on the fallback DS's flange speed
IK_DAMPING = 1e-2           # damped least squares, keeps J^+ finite near singularities

DISTURBANCE_MAGNITUDE = 0.0         # N: push applied at the flange (0 disables it)
DISTURBANCE_DIRECTION = [1, 1, 1]   # world frame
DISTURBANCE_START = 0               # s of sim time
DISTURBANCE_DURATION = 2            # s

OBSTACLE_CENTER = [0.2545, -0.046, 0.3389]  # m, world frame: sweeps around this point; None = middle of the planned path
OBSTACLE_AXIS = [0, 1, 0]                   # world frame: sweeps left and right along this
OBSTACLE_AMPLITUDE = 0.25                   # m either side of its center
OBSTACLE_SPEED = 0.5                        # m/s, constant


def hover_target(arm):
    """Flange pose that puts the probe HOVER_OFFSET above the T's center of mass (its body origin is
    the bar's edge, see world.xml), rod pointing down, turned with the T."""
    block = arm.data.body("t_block")
    pos = block.xipos + HOVER_OFFSET + arm.pusher.flange_above_probe
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, block.xquat, TOOL_DOWN_QUAT)
    return pos, quat


def obstacle_center(arm, plan):
    """Where the obstacle sweeps around, and why there."""
    if OBSTACLE_CENTER is not None:
        return np.asarray(OBSTACLE_CENTER, dtype=float), "set by OBSTACLE_CENTER"
    if plan is not None:  # the flange's position halfway along the planned path
        path = plan["best"]["path"]
        mid = mujoco.MjData(arm.model)
        mid.qpos[:arm.nu] = path[len(path) // 2]
        mujoco.mj_kinematics(arm.model, mid)
        return mid.site("attachment_site").xpos.copy(), "middle of the planned path"
    return arm.model.body("obstacle").pos.copy(), "world.xml default, no path"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--hide-collision", action="store_true",
                        help="don't overlay the capsules the planner and CBF see")
    args = parser.parse_args(argv)
    robot = ROBOTS[args.robot]

    arm = Arm(robot, joint_vel_max=CBF.joint_vel_max)
    safety = SafetyFilter(arm, CBF)
    print(f"Robot: {robot.name} ({arm.nu} joints, scene {robot.scene})")
    print(f"Collision geometry: {len(arm.geometry.link_capsules)} capsules/spheres, {len(arm.geometry.link_pairs)} "
          f"link-vs-link pairs guarded (sampled in {arm.geometry_seconds:.1f}s)")

    # the IK seeds are the only randomness in a run (planning, physics and the obstacle are
    # deterministic given the path), so this one number recreates the whole run
    seed = resolve_seed(args.seed)
    print(f"Seed: {seed}   (--seed {seed} replays this run)")

    print("Planning coarse-reach path...")
    planner = make_planner(args.planner, CBF.margin)
    target_pos, target_quat = hover_target(arm)
    t0 = time.perf_counter()
    plan = plan_trajectory(arm.model, arm.data, arm.geometry, arm.env, arm.q, target_pos, target_quat,
                           planner=planner, parallel=True, rng=np.random.default_rng(seed))
    print(f"Planning took {time.perf_counter() - t0:.3f}s")
    if plan is None:
        print("WARNING: no IK candidate converged -- falling back to the straight-line DS")
        field = None
    else:
        print_plan(plan, planner, args.planner)
        field = PathVelocityField(plan["best"]["path"], k_tangent=robot.path_k_tangent,
                                  k_corrective=robot.path_k_corrective)
        print_field(field)

    center, where = obstacle_center(arm, plan)
    obstacle = MovingObstacle(arm.model, center, axis=OBSTACLE_AXIS, amplitude=OBSTACLE_AMPLITUDE, speed=OBSTACLE_SPEED)
    print(f"Obstacle: starts at {np.round(obstacle.center, 4).tolist()} ({where}), "
          f"sweeps +-{OBSTACLE_AMPLITUDE} m along {OBSTACLE_AXIS} at {OBSTACLE_SPEED} m/s, heading + first")

    def draw_overlays(scn):
        """The capsules the planner and CBF see: the arm's (red) and the obstacle's (yellow)."""
        if not args.hide_collision:
            draw_capsules(scn, arm.geometry.link_capsules_world(arm.data))
            draw_capsules(scn, [obstacle_now], rgba=(0.95, 0.75, 0.1, 0.35))

    dt = arm.model.opt.timestep  # 0.002 s -> 500 Hz, one control update per sim step
    recording = Recorder(arm.model, args.record) if args.record else contextlib.nullcontext()
    with mujoco.viewer.launch_passive(arm.model, arm.data) as viewer, recording as recorder:
        while viewer.is_running():
            t_wall = time.perf_counter()
            obstacle_now = obstacle.update(arm.data)
            if field is not None:
                u_des = field(arm.q)
            else:  # re-aimed every tick: the T may have been knocked
                u_des = flange_pose_ds(arm, *hover_target(arm), k_p=robot.k_p, k_o=robot.k_o,
                                       max_speed=MAX_SPEED, damping=IK_DAMPING)
            arm.command(safety.filter(u_des, obstacle=obstacle_now))
            arm.push_flange(DISTURBANCE_MAGNITUDE, DISTURBANCE_DIRECTION, DISTURBANCE_START, DISTURBANCE_DURATION)
            arm.step()
            viewer.user_scn.ngeom = 0
            draw_overlays(viewer.user_scn)
            if recorder is not None:
                recorder.capture(arm.data, draw_overlays)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - t_wall)))  # pace to real time


if __name__ == "__main__":
    main()

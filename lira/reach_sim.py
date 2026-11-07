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
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering for --record; must precede `import mujoco`
# the viewer window: the glfw package's Wayland build crashes on a GNOME setting it can't read, its X11 build runs fine
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import numpy as np

from common import add_common_args, make_planner, print_field, print_plan, resolve_seed, run_viewer
from core.planner import plan_trajectory
from mj.arm import Arm
from mj.robots import DEFAULT_ROBOT, ROBOTS
from mj.safety import CBFConfig, SafetyFilter
from mj.scene import MovingObstacle
from mj.tasks import hover_ds, hover_target
from mj.utils import draw_capsules

CBF = CBFConfig()

OBSTACLE_CENTER = [0.2545, -0.046, 0.3389]  # m, world frame: sweeps around this point
OBSTACLE_AXIS = [0, 1, 0]                   # world frame: sweeps left and right along this
OBSTACLE_AMPLITUDE = 0.25                   # m either side of its center
OBSTACLE_SPEED = 0.5                        # m/s, constant


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", choices=sorted(ROBOTS), default=DEFAULT_ROBOT)
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
        field = arm.path_field(plan.best.path)
        print_field(field)

    obstacle = MovingObstacle(arm.model, OBSTACLE_CENTER, axis=OBSTACLE_AXIS, amplitude=OBSTACLE_AMPLITUDE,
                              speed=OBSTACLE_SPEED)
    print(f"Obstacle: starts at {np.round(obstacle.center, 4).tolist()}, "
          f"sweeps +-{OBSTACLE_AMPLITUDE} m along {OBSTACLE_AXIS} at {OBSTACLE_SPEED} m/s, heading + first")
    obstacle_now = None  # the obstacle's capsule this tick

    def tick():
        nonlocal obstacle_now
        obstacle_now = obstacle.update(arm.data)
        if field is not None:
            u_des = field(arm.q)
        else:
            u_des = hover_ds(arm)
        arm.command(safety.filter(u_des, obstacle=obstacle_now))
        arm.step()

    def draw_overlays(scn):
        """The capsules the planner and CBF see: the arm's (red) and the obstacle's (yellow)."""
        if not args.hide_collision:
            draw_capsules(scn, arm.geometry.world(arm.data).all)
            draw_capsules(scn, obstacle_now, rgba=(0.95, 0.75, 0.1, 0.35))

    run_viewer(arm.model, arm.data, tick, draw_overlays, args.record)


if __name__ == "__main__":
    main()

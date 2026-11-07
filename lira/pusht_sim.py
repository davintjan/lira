"""
Push-T sim: one arm plays Push-T on the MuJoCo table (pusht.task.PushTTask: reach, descend, push, lift,
hover) while an obstacle sweeps through; once the T is at its goal, the obstacle stops where it is.

The policy only ever sees Push-T pixels: PushTFrame maps the table to the Push-T workspace at
1 px = 1 mm, axes aligned, with Push-T's fixed goal pose on world.xml's goal body.

Run from this directory:  .venv/bin/python pusht_sim.py [--robot ur5e] [--seed N] [--no-obstacle] [--record out.mp4]
"""
import argparse
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering for --record; must precede `import mujoco`
# the viewer window: the glfw package's Wayland build crashes on a GNOME setting it can't read, its X11 build runs fine
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import numpy as np
import torch

from common import add_common_args, make_planner, resolve_seed, run_viewer
from mj.arm import Arm
from mj.monitor import ContactMonitor
from mj.robots import DEFAULT_ROBOT, ROBOTS
from mj.safety import CBFConfig, SafetyFilter
from mj.scene import MovingObstacle
from pusht.frame import PushTFrame
from pusht.policy import DEVICE, REFERENCE_CHECKPOINT, ChunkedPolicy, load_policy
from pusht.task import POLICY_DT, PREFETCH_ACTIONS, PREPUSH_MARGIN_PX, PushTTask, draw_targets

CBF = CBFConfig()
PREPUSH_DISTANCE = 0.13  # m from the T's center of mass: clear of the T (it reaches ~0.076 m), its CBF margin and the rod

OBSTACLE_CENTER = [0.3545, 0.146, 0.2589]  # m, world frame: sweeps around this point
OBSTACLE_AXIS = [0, 1, 0]  # world frame: sweeps left and right along this
OBSTACLE_AMPLITUDE = 0.45  # m either side of its center
OBSTACLE_SPEED = 0.2       # m/s, constant
OBSTACLE_DWELL = 5.0       # s: starts at -OBSTACLE_AMPLITUDE (along OBSTACLE_AXIS), crosses, parks this long at
                           # each end before crossing back; None = sweep back and forth without stopping


def make_obstacle(arm):
    """The obstacle sweeping through the arm while it pushes."""
    obstacle = MovingObstacle(arm.model, OBSTACLE_CENTER, axis=OBSTACLE_AXIS, amplitude=OBSTACLE_AMPLITUDE,
                              speed=OBSTACLE_SPEED, dwell=OBSTACLE_DWELL)
    motion = (f"crosses +-{OBSTACLE_AMPLITUDE} m along {OBSTACLE_AXIS}, waiting {OBSTACLE_DWELL} s at each end"
              if OBSTACLE_DWELL is not None else f"sweeps +-{OBSTACLE_AMPLITUDE} m along {OBSTACLE_AXIS}")
    print(f"Obstacle: around {np.round(obstacle.center, 3).tolist()}, {motion} at {OBSTACLE_SPEED} m/s")
    return obstacle


def park_obstacle(arm):
    arm.data.mocap_pos[arm.model.body_mocapid[arm.model.body("obstacle").id]] = [-0.5, 0.0, 1.0]  # behind the robot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", choices=sorted(ROBOTS), default=DEFAULT_ROBOT)
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

    if args.no_obstacle:
        obstacle, monitor = None, None
        park_obstacle(arm)
    else:
        obstacle = make_obstacle(arm)
        arm_bodies = {cap.body_id for cap in arm.geometry.env_capsules}
        monitor = ContactMonitor(arm.model, arm_bodies, {arm.model.body("obstacle").id})

    def obstacle_note(so_far):
        return f" | obstacle: {monitor.summary(so_far)}" if monitor else ""

    task = PushTTask(arm, safety, frame, policy, note=obstacle_note,
                     prepush_spot=lambda data: frame.prepush_spot(data, PREPUSH_DISTANCE, PREPUSH_MARGIN_PX))
    task.plan(make_planner(args.planner, CBF.margin), np.random.default_rng(seed))

    def tick():
        now = arm.data.time
        obstacle_now = obstacle.update(arm.data) if obstacle is not None else None
        caps = arm.geometry.world(arm.data)  # every capsule in world frame, once per tick
        if obstacle_now is not None and monitor.update(arm.data, caps.env, obstacle_now):
            print(f"[{now:6.2f}s] WARNING: the obstacle hit the robot")
        arm.command(task.control(now, caps, obstacle_now))
        if task.succeeded and obstacle is not None:
            obstacle.freeze(arm.data)  # task done: the obstacle stops where it is
        arm.step()

    def draw_overlays(scn):
        """The policy's targets, while it's pushing."""
        draw_targets(scn, frame, policy.target if task.phase == "push" else None, policy.queue)

    run_viewer(arm.model, arm.data, tick, draw_overlays, args.record)


if __name__ == "__main__":
    main()

"""
Dual Push-T sim: two arms facing each other across one table (scenes/scene_dual.xml), each playing
Push-T with its own copy of the diffusion policy on its own board -- the iiwa14 the red T to goal_a, the
ur5e the blue T to goal_b. The boards overlap in the middle, where both Ts start, so the arms push in
the same space. Both start with every joint at 0.

Each arm runs pusht.task.PushTTask's phases on its own single-arm model (mj.dual.DualArm), as in
dual_reach_sim.py. Neither the plans nor the policies know the other arm exists; the CBF keeps them apart,
each arm holding the other's links where they are right now (core.geometry.capsule_set_rows). The
obstacle sweeps across both boards between the arms, and both dodge it too (--no-obstacle parks it); it
stops where it is once both arms are done pushing.

The policy only ever sees its own board in Push-T pixels: PushTFrame maps each board through its own
goal body (goal_a, goal_b). Each arm's CBF guards its own T (the rod may touch it while pushing); the
other arm's T its plan and CBF keep clear of entirely, rod included, held where it is now.

Run from this directory:  .venv/bin/python dual_pusht_sim.py [--seed N] [--zigzag] [--no-obstacle] [--record out.mp4]
"""
import argparse
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering for --record; must precede `import mujoco`
# the viewer window: the glfw package's Wayland build crashes on a GNOME setting it can't read, its X11 build runs fine
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco
import numpy as np
import torch

from common import add_common_args, make_planner, resolve_seed, run_viewer
from mj.dual import CONTROL_EVERY, DualArm, make_monitors, make_obstacle
from mj.model import Pusher, load_dual_model
from mj.safety import CBFConfig
from pusht.frame import PushTFrame
from pusht.policy import DEVICE, REFERENCE_CHECKPOINT, ChunkedPolicy, load_policy
from pusht.task import POLICY_DT, PREFETCH_ACTIONS, PREPUSH_MARGIN_PX, PushTTask, draw_targets

CBF = CBFConfig()
# m from the probe's centre to the T's outline at the pre-push spot: the rod, the CBF's margin (descending,
# the rod may not touch the T) and 1 cm so it isn't held right at that margin
PREPUSH_CLEARANCE = Pusher().radius + CBF.margin + 0.01
DDIM_STEPS = 10  # the checkpoint's own DDPM (100 steps, ~0.7 s here) is too slow for two policies at 10 Hz
# (prefix in scene_dual.xml, robot, its T, its goal): each arm plays its own board
ARMS = (("a_", "iiwa14", "t_block_a", "goal_a"),
        ("b_", "ur5e", "t_block_b", "goal_b"))
# --zigzag: each arm plays the other's board -- the iiwa14 the blue T to goal_b, the ur5e the red T to
# goal_a -- so their arms cross over the middle the whole time they push. The boards sit 0.24 m off where
# each arm was placed to reach them best, so the far corners may be out of reach.
ZIGZAG_ARMS = (("a_", "iiwa14", "t_block_b", "goal_b"),
               ("b_", "ur5e", "t_block_a", "goal_a"))


class PushTArm(PushTTask):
    """One arm's Push-T: pusht.task.PushTTask on a DualArm, with its own policy and board."""

    def __init__(self, dual_model, prefix, robot_name, block, goal, policy):
        self.dual = DualArm(dual_model, prefix, robot_name, block, CBF)
        frame = PushTFrame(dual_model, goal=goal)
        super().__init__(self.dual.arm, self.dual.safety, frame, policy, name=robot_name,
                         prepush_spot=lambda data: frame.prepush_spot_near(data, PREPUSH_CLEARANCE, PREPUSH_MARGIN_PX),
                         hover_spot=lambda data: data.body("t_block").xipos[:2].copy())  # over the T's centre of mass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=REFERENCE_CHECKPOINT)
    parser.add_argument("--zigzag", action="store_true",
                        help="each arm plays the other's board: the ur5e the red T, the iiwa14 the blue one")
    parser.add_argument("--no-obstacle", action="store_true", help="leave the obstacle parked off the table")
    parser.add_argument("--ddim-steps", type=int, default=DDIM_STEPS,
                        help="DDIM denoising steps per inference; 0 = the checkpoint's DDPM")
    args = parser.parse_args(argv)

    seed = resolve_seed(args.seed)
    print(f"Seed: {seed}   (--seed {seed} replays IK seeds and diffusion noise -- inference timing still varies run to run)")
    torch.manual_seed(seed)

    prefixes = tuple(prefix for prefix, *_ in ARMS)
    model = load_dual_model(prefixes=prefixes, joint_vel_max=CBF.joint_vel_max)
    data = mujoco.MjData(model)  # qpos0: every arm joint at 0, the Ts at scene_dual.xml's Push-T start
    mujoco.mj_forward(model, data)
    sampler = f"DDIM, {args.ddim_steps} steps" if args.ddim_steps else "DDPM"
    print(f"Loading policy {args.checkpoint.name} on {DEVICE} ({sampler}), one copy per arm...")
    # a copy each: inference runs in each arm's own thread, and the noise scheduler keeps per-call state
    arms = [PushTArm(model, *spec, ChunkedPolicy(load_policy(args.checkpoint, DEVICE, args.ddim_steps or None), DEVICE,
                                                 period=POLICY_DT, prefetch=PREFETCH_ACTIONS))
            for spec in (ZIGZAG_ARMS if args.zigzag else ARMS)]
    planner = make_planner(args.planner, CBF.margin)
    for k, push_arm in enumerate(arms):
        push_arm.dual.sync(data)
        push_arm.plan(planner, np.random.default_rng([seed, k]), boxes=push_arm.dual.other_boxes)

    arm_monitor, obstacle_monitor = make_monitors(model, prefixes)
    obstacle = None if args.no_obstacle else make_obstacle(model)
    step = 0

    def tick():
        nonlocal step
        obstacle_now = obstacle.update(data) if obstacle is not None else None
        if step % CONTROL_EVERY == 0:
            for push_arm in arms:
                push_arm.dual.sync(data)
            caps = [push_arm.arm.geometry.world(push_arm.arm.data) for push_arm in arms]
            for push_arm, own, other in zip(arms, caps, caps[::-1]):
                data.ctrl[push_arm.dual.ctrl_idx] = push_arm.control(data.time, own, obstacle_now, others=other.env,
                                                                     boxes=push_arm.dual.other_boxes)
            if arm_monitor.update(data, caps[0].env, caps[1].env):
                print(f"[{data.time:6.2f}s] WARNING: the arms touched")
            if obstacle_now is not None:
                if obstacle_monitor.update(data, caps[0].env + caps[1].env, [obstacle_now]):
                    print(f"[{data.time:6.2f}s] WARNING: the obstacle hit an arm")
                if all(push_arm.finished for push_arm in arms):
                    obstacle.freeze(data)  # both tasks over: the obstacle stops where it is
        mujoco.mj_step(model, data)
        step += 1

    def draw_overlays(scn):
        """Each arm's policy targets, while it's pushing."""
        for push_arm in arms:
            draw_targets(scn, push_arm.frame, push_arm.policy.target if push_arm.phase == "push" else None,
                         push_arm.policy.queue)

    run_viewer(model, data, tick, draw_overlays, args.record)
    print(f"Arms: {arm_monitor.summary()}" + ("" if obstacle is None else f" | obstacle: {obstacle_monitor.summary()}"))


if __name__ == "__main__":
    main()

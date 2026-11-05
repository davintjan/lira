"""
Dual Push-T sim: two arms facing each other across one table (scenes/scene_dual.xml), each playing
Push-T with its own copy of the diffusion policy on its own board -- the iiwa14 the red T to goal_a, the
ur5e the blue T to goal_b. The boards overlap in the middle, where both Ts start, so the arms push in
the same space. Both start with every joint at 0.

Each arm runs pusht_sim.py's phases (pusht_sim.PushTTask) on its own single-arm model, as
in dual_reach_sim.py: the physics runs in the two-arm model and every control tick each arm's model is
synced from it. Neither the plans nor the policies know the other arm exists; the CBF keeps them apart,
each arm holding the other's links where they are right now (core.geometry.capsule_set_rows). The
obstacle sweeps across both boards between the arms, and both dodge it too (--no-obstacle parks it); it
stops where it is once both arms are done pushing.

The policy only ever sees its own board in Push-T pixels: PushTFrame maps each board through its own
goal body (goal_a, goal_b). Each arm's CBF guards its own T (the rod may touch it while pushing); the
other arm's T its plan and CBF keep clear of entirely, rod included, held where it is now.

Run from this directory:  .venv/bin/python dual_pusht_sim.py [--seed N] [--zigzag] [--no-obstacle] [--record out.mp4]
"""
import argparse
import contextlib
import os
import pathlib
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering for --record; must precede `import mujoco`
# the viewer window: the glfw package's Wayland build crashes on a GNOME setting it can't read, its X11 build runs fine
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco
import mujoco.viewer
import numpy as np
import torch

from common import PLANNERS, make_planner, resolve_seed
from dual_reach_sim import CBF, CONTROL_EVERY, ArmArmMonitor, DualArm, DualObstacleMonitor, make_obstacle
from mj.model import Pusher, load_dual_model
from mj.recorder import Recorder
from pusht.frame import PushTFrame
from pusht.policy import REFERENCE_CHECKPOINT, ChunkedPolicy, load_policy
from pusht_sim import POLICY_DT, PREFETCH_ACTIONS, PREPUSH_MARGIN_PX, PushTTask, draw_targets

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    """One arm's Push-T: pusht_sim.PushTTask on a DualArm, with its own policy and board."""

    def __init__(self, dual_model, prefix, robot_name, block, goal, policy):
        self.dual = DualArm(dual_model, prefix, robot_name, block, rgba=None)
        frame = PushTFrame(dual_model, goal=goal)
        super().__init__(self.dual.arm, self.dual.safety, frame, policy, name=robot_name,
                         prepush_spot=lambda data: frame.prepush_spot_near(data, PREPUSH_CLEARANCE, PREPUSH_MARGIN_PX),
                         hover_spot=lambda data: data.body("t_block").xipos[:2].copy())  # over the T's centre of mass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planner", choices=sorted(PLANNERS), default="sqp")
    parser.add_argument("--seed", type=int, default=None,
                        help="replay a run: the seed it printed at startup (default: a fresh one)")
    parser.add_argument("--record", metavar="PATH", default=None,
                        help="also record video: PATH=out.mp4 writes out_front.mp4 and out_top.mp4")
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

    model = load_dual_model(prefixes=tuple(prefix for prefix, *_ in ARMS), joint_vel_max=CBF.joint_vel_max)
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

    monitor = ArmArmMonitor(model)
    obstacle = None if args.no_obstacle else make_obstacle(model)
    obstacle_monitor = DualObstacleMonitor(model)
    obstacle_now = None

    def draw_overlays(scn):
        """Each arm's policy targets, while it's pushing."""
        for push_arm in arms:
            draw_targets(scn, push_arm.frame, push_arm.policy.target if push_arm.phase == "push" else None,
                         push_arm.policy.queue)

    dt = model.opt.timestep
    step = 0
    recording = Recorder(model, args.record) if args.record else contextlib.nullcontext()
    with mujoco.viewer.launch_passive(model, data) as viewer, recording as recorder:
        while viewer.is_running():
            t_wall = time.perf_counter()
            if obstacle is not None:
                obstacle_now = obstacle.update(data)
            if step % CONTROL_EVERY == 0:
                for push_arm in arms:
                    push_arm.dual.sync(data)
                caps = [push_arm.arm.geometry.world(push_arm.arm.data) for push_arm in arms]
                for push_arm, own, other in zip(arms, caps, caps[::-1]):
                    data.ctrl[push_arm.dual.ctrl_idx] = push_arm.control(data.time, own, obstacle_now, others=other["env"],
                                                                         boxes=push_arm.dual.other_boxes)
                touched_before = monitor.contact_ticks
                monitor.update(data, *[c["env"] for c in caps])
                if monitor.contact_ticks and not touched_before:
                    print(f"[{data.time:6.2f}s] WARNING: the arms touched")
                if obstacle_now is not None:
                    if obstacle_monitor.update(data, [c["env"] for c in caps], obstacle_now):
                        print(f"[{data.time:6.2f}s] WARNING: the obstacle hit an arm")
                    if all(push_arm.finished for push_arm in arms):
                        obstacle.freeze(data)  # both tasks over: the obstacle stops where it is
            mujoco.mj_step(model, data)
            step += 1
            viewer.user_scn.ngeom = 0
            draw_overlays(viewer.user_scn)
            if recorder is not None:
                recorder.capture(data, draw_overlays)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - t_wall)))
    print(f"Arms: closest {monitor.min_clearance * 100:.1f} cm, {monitor.contact_ticks} contact ticks"
          + ("" if obstacle is None else f" | {obstacle_monitor.summary()}"))


if __name__ == "__main__":
    main()

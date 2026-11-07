"""
Dual reach sim, zig zag: two arms facing each other across one table (scenes/scene_dual.xml), each
reaching to hover over the T on the OTHER arm's board -- the iiwa14 over the ur5e's (blue) T, the ur5e
over the iiwa14's (red) T -- so their paths cross over the seam. Both start with every joint at 0, the Ts
at BLOCK_POSES.

Each arm is planned and filtered exactly as in reach_sim.py, on its own single-arm model (mj.dual.DualArm).
Neither plan knows the other arm exists: the arms stay apart only through the CBF, where each holds the
other's links where they are right now (no velocity, no plan) and keeps its own share of every clearance
between them (core.geometry.capsule_set_rows). The obstacle sweeps across both boards between the arms,
and both dodge it too (--no-obstacle parks it).

  plan     once per arm, before the loop: IK candidates + SQP or gradient descent (--planner)
  follow   every tick, each arm: PathVelocityField -> SafetyFilter(others=the other arm, the other T, obstacle) -> servos

Run from this directory:  .venv/bin/python dual_reach_sim.py [--seed N] [--no-obstacle] [--record out.mp4]
"""
import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering for --record; must precede `import mujoco`
# the viewer window: the glfw package's Wayland build crashes on a GNOME setting it can't read, its X11 build runs fine
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco
import numpy as np

from common import add_common_args, make_planner, print_field, print_plan, resolve_seed, run_viewer
from core.planner import plan_trajectory
from mj.dual import CONTROL_EVERY, DualArm, make_monitors, make_obstacle
from mj.model import load_dual_model
from mj.safety import CBFConfig
from mj.tasks import hover_ds, hover_target
from mj.utils import draw_capsules

CBF = CBFConfig()
# (prefix in scene_dual.xml, robot, the T it reaches for): the zig zag -- each the other's T
ARMS = (("a_", "iiwa14", "t_block_b"),
        ("b_", "ur5e", "t_block_a"))
CAPSULE_COLORS = ((1.0, 0.15, 0.15, 0.35), (0.15, 0.35, 1.0, 0.35))  # each arm's capsules, with --show-collision
# (x, y m, yaw deg) each T starts at in this sim, overriding scene_dual.xml's Push-T start: in the shared
# middle, bars 4 cm either side of the centre line, 20 cm apart along it.
BLOCK_POSES = {"t_block_a": (0.2, -0.04, 180.0), "t_block_b": (0.0, 0.04, 0.0)}
REPORT_EVERY = 1.0  # s of sim time between progress lines


class ReachArm(DualArm):
    """A DualArm reaching to hover over its target T, along its own plan."""

    def __init__(self, dual_model, prefix, robot_name, target_block):
        super().__init__(dual_model, prefix, robot_name, target_block, CBF)
        self.field = None

    def plan(self, planner, kind, rng):
        """This arm's own plan to hover over its T, blind to the other arm (None: no IK goal), and a field
        along it."""
        arm = self.arm
        print(f"Planning {self.name}: q = 0 -> over {self.target_block}...")
        t0 = time.perf_counter()
        plan = plan_trajectory(arm.model, arm.data, arm.geometry, arm.env, arm.q, *hover_target(arm),
                               planner=planner, parallel=True, rng=rng, boxes=self.other_boxes)
        print(f"  took {time.perf_counter() - t0:.3f}s")
        if plan is None:
            print(f"  WARNING: no IK candidate converged -- {self.name} falls back to the straight-line DS")
            return None
        print_plan(plan, planner, kind)
        self.field = arm.path_field(plan.best.path)
        print_field(self.field)
        return plan

    def nominal(self):
        if self.field is not None:
            return self.field(self.arm.q)
        return hover_ds(self.arm)

    def hover_error(self):
        """m from the flange to where it should be to hover the probe over the T."""
        target_pos, _ = hover_target(self.arm)
        return np.linalg.norm(self.arm.flange_pose()[0] - target_pos)


def place_blocks(model, data):
    """The Ts to BLOCK_POSES, resting on the table."""
    for name, (x, y, yaw) in BLOCK_POSES.items():
        adr = model.jnt_qposadr[model.body(name).jntadr[0]]
        half = np.radians(yaw) / 2
        data.qpos[adr:adr + 7] = [x, y, 0.02, np.cos(half), 0.0, 0.0, np.sin(half)]
    mujoco.mj_forward(model, data)


def control(arms, data, obstacle=None):
    """One control update: every arm's model synced to the sim, then each arm's nominal command filtered
    against its own hazards, the other arm's links where they are now and the obstacle (its capsule this
    tick, or None), into the servos. Returns each arm's capsules in world frame (RobotGeometry.world's
    env), for the monitors and the overlay."""
    for dual_arm in arms:
        dual_arm.sync(data)
    caps = [dual_arm.arm.geometry.world(dual_arm.arm.data) for dual_arm in arms]
    for dual_arm, own, other in zip(arms, caps, caps[::-1]):
        data.ctrl[dual_arm.ctrl_idx] = dual_arm.safety.filter(dual_arm.nominal(), caps=own, others=other.env,
                                                              boxes=dual_arm.other_boxes, obstacle=obstacle)
    return [c.env for c in caps]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--no-obstacle", action="store_true", help="leave the obstacle parked off the table")
    parser.add_argument("--show-collision", action="store_true",
                        help="overlay the capsules the planner and CBF see")
    args = parser.parse_args(argv)

    prefixes = tuple(prefix for prefix, *_ in ARMS)
    model = load_dual_model(prefixes=prefixes, joint_vel_max=CBF.joint_vel_max)
    data = mujoco.MjData(model)  # qpos0: every arm joint at 0
    place_blocks(model, data)
    arms = [ReachArm(model, *spec) for spec in ARMS]
    for dual_arm in arms:
        dual_arm.sync(data)
        print(f"{dual_arm.name}: {len(dual_arm.arm.geometry.link_capsules)} capsules/spheres, "
              f"base at {np.round(dual_arm.arm.robot.base_pos, 3).tolist()}, "
              f"yaw {np.degrees(dual_arm.arm.robot.base_yaw):.0f} deg")

    seed = resolve_seed(args.seed)
    print(f"Seed: {seed}   (--seed {seed} replays this run)")
    planner = make_planner(args.planner, CBF.margin)
    for k, dual_arm in enumerate(arms):
        dual_arm.plan(planner, args.planner, np.random.default_rng([seed, k]))

    arm_monitor, obstacle_monitor = make_monitors(model, prefixes)
    obstacle = None if args.no_obstacle else make_obstacle(model)
    world_caps, obstacle_now = [None, None], None
    step, next_report = 0, 0.0

    def tick():
        nonlocal world_caps, obstacle_now, step, next_report
        if obstacle is not None:
            obstacle_now = obstacle.update(data)
        if step % CONTROL_EVERY == 0:
            world_caps = control(arms, data, obstacle_now)
            arm_monitor.update(data, *world_caps)
            if obstacle_now is not None and obstacle_monitor.update(data, world_caps[0] + world_caps[1], [obstacle_now]):
                print(f"[{data.time:6.2f}s] WARNING: the obstacle hit an arm")
        if data.time >= next_report:
            next_report += REPORT_EVERY
            print(f"t={data.time:5.1f}s  arm-arm clearance {arm_monitor.clearance * 100:5.1f} cm "
                  f"(min {arm_monitor.min_clearance * 100:.1f}), "
                  + ", ".join(f"{a.name} {a.hover_error() * 100:.1f} cm from goal" for a in arms)
                  + f", {arm_monitor.contact_ticks} contact ticks")
        mujoco.mj_step(model, data)
        step += 1

    def draw_overlays(scn):
        if args.show_collision:
            for caps, color in zip(world_caps, CAPSULE_COLORS):
                draw_capsules(scn, caps, rgba=color)
            if obstacle_now is not None:
                draw_capsules(scn, [obstacle_now], rgba=(0.95, 0.75, 0.1, 0.35))

    run_viewer(model, data, tick, draw_overlays, args.record)
    print(f"Arms: {arm_monitor.summary()}" + ("" if obstacle is None else f" | obstacle: {obstacle_monitor.summary()}"))


if __name__ == "__main__":
    main()

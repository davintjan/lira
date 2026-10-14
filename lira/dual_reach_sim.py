"""
Dual reach sim, zig zag: two arms facing each other across one table (scenes/scene_dual.xml), each
reaching to hover over the T on the OTHER arm's board -- the iiwa14 over the ur5e's (blue) T, the ur5e
over the iiwa14's (red) T -- so their paths cross over the seam. Both start with every joint at 0, the Ts
at BLOCK_POSES.

Each arm is planned and filtered exactly as in reach_sim.py, on its own single-arm model (an Arm, its
base moved to where scene_dual.xml mounts it, so both share world coordinates). The physics runs in
the two-arm model; every tick each arm's model is synced from it. Neither plan knows the other arm
exists: the arms stay apart only through the CBF, where each holds the other's links where they are
right now (no velocity, no plan) and keeps its own share of every clearance between them
(core.geometry.capsule_set_rows). The obstacle sweeps across both boards between the arms, and both
dodge it too (--no-obstacle parks it).

  plan     once per arm, before the loop: IK candidates + SQP or gradient descent (--planner)
  follow   every tick, each arm: PathVelocityField -> SafetyFilter(others=the other arm, obstacle) -> servos

Run from this directory:  .venv/bin/python dual_reach_sim.py [--seed N] [--no-obstacle] [--record out.mp4]
"""
import argparse
import contextlib
import dataclasses
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering for --record; must precede `import mujoco`
# the viewer window: the glfw package's Wayland build crashes on a GNOME setting it can't read, its X11 build runs fine
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco
import mujoco.viewer
import numpy as np

from common import PLANNERS, make_planner, print_field, print_plan, resolve_seed
from core.geometry import capsule_capsule_distance_batch
from core.path_field import PathVelocityField
from core.planner import plan_trajectory
from mj.arm import Arm
from mj.model import load_dual_model
from mj.recorder import Recorder
from mj.robots import ROBOTS
from mj.safety import CBFConfig, SafetyFilter
from mj.scene import MovingObstacle
from mj.tasks import flange_pose_ds
from mj.utils import draw_capsules
from reach_sim import IK_DAMPING, MAX_SPEED, hover_target

CBF = CBFConfig()
# (prefix in scene_dual.xml, robot, the T it reaches for, overlay colour): the zig zag -- each the other's T
ARMS = (("a_", "iiwa14", "t_block_b", (1.0, 0.15, 0.15, 0.35)),
        ("b_", "ur5e", "t_block_a", (0.15, 0.35, 1.0, 0.35)))
# (x, y m, yaw deg) each T starts at in this sim, overriding scene_dual.xml's Push-T start: in the shared
# middle, bars 4 cm either side of the centre line, 20 cm apart along it.
BLOCK_POSES = {"t_block_a": (0.2, -0.04, 180.0), "t_block_b": (0.0, 0.04, 0.0)}
CONTROL_EVERY = 2   # physics steps per control update: 250 Hz. Two arms' filters take ~3 ms, over a 2 ms step
REPORT_EVERY = 1.0  # s of sim time between progress lines

# The obstacle: crosses both boards (along y) between the two arms, at wrist height while they push (the
# rod tip at 2.5 cm, the flange ~17.5 cm), waiting OBSTACLE_DWELL at each end. Both arms dodge it.
OBSTACLE_CENTER = [0.1, 0.0, 0.30]  # m, world frame: the middle of the shared strip
OBSTACLE_AXIS = [0, 1, 0]
OBSTACLE_AMPLITUDE = 0.40           # m either side of the centre: past both goals
OBSTACLE_SPEED = 0.2                # m/s
OBSTACLE_DWELL = 3.0                # s parked at each end; None = sweep back and forth without stopping


class DualArm:
    """One arm of the dual scene: its single-arm Arm (planning, geometry, CBF) and where its joints and
    servos sit in the two-arm model."""

    def __init__(self, dual_model, prefix, robot_name, target_block, rgba):
        mount = dual_model.body(prefix + "mount")
        w, _, _, z = mount.quat  # scene_dual.xml mounts the arms turned about z only
        robot = dataclasses.replace(ROBOTS[robot_name], base_pos=tuple(mount.pos),
                                    base_yaw=float(2 * np.arctan2(z, w)), start_key=None)
        self.arm = Arm(robot, joint_vel_max=CBF.joint_vel_max)
        self.safety = SafetyFilter(self.arm, CBF)
        self.name, self.target_block, self.rgba = robot_name, target_block, rgba
        model = self.arm.model
        joints = [dual_model.joint(prefix + model.joint(j).name) for j in range(self.arm.nu)]
        self.qpos_idx = np.array([j.qposadr[0] for j in joints])
        self.dof_idx = np.array([j.dofadr[0] for j in joints])
        self.ctrl_idx = np.array([dual_model.actuator(prefix + model.actuator(a).name).id for a in range(self.arm.nu)])
        # the arm's model has one T (t_block): it stands in for the T this arm reaches for
        self.block_qpos = model.jnt_qposadr[model.body("t_block").jntadr[0]]
        self.dual_block_qpos = dual_model.jnt_qposadr[dual_model.body(target_block).jntadr[0]]
        self.field = None

    def sync(self, dual_data):
        """The arm's model to the two-arm sim's state: its joints and its target T."""
        data = self.arm.data
        data.qpos[:self.arm.nu] = dual_data.qpos[self.qpos_idx]
        data.qvel[:self.arm.nu] = dual_data.qvel[self.dof_idx]
        data.qpos[self.block_qpos:self.block_qpos + 7] = dual_data.qpos[self.dual_block_qpos:self.dual_block_qpos + 7]
        mujoco.mj_forward(self.arm.model, data)

    def plan(self, planner, kind, rng):
        arm = self.arm
        print(f"Planning {self.name}: q = 0 -> over {self.target_block}...")
        t0 = time.perf_counter()
        plan = plan_trajectory(arm.model, arm.data, arm.geometry, arm.env, arm.q, *hover_target(arm),
                               planner=planner, parallel=True, rng=rng)
        print(f"  took {time.perf_counter() - t0:.3f}s")
        if plan is None:
            print(f"  WARNING: no IK candidate converged -- {self.name} falls back to the straight-line DS")
            return
        print_plan(plan, planner, kind)
        self.field = PathVelocityField(plan["best"]["path"], k_tangent=arm.robot.path_k_tangent,
                                       k_corrective=arm.robot.path_k_corrective)
        print_field(self.field)

    def nominal(self):
        if self.field is not None:
            return self.field(self.arm.q)
        return flange_pose_ds(self.arm, *hover_target(self.arm), k_p=self.arm.robot.k_p, k_o=self.arm.robot.k_o,
                              max_speed=MAX_SPEED, damping=IK_DAMPING)

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


def make_obstacle(model):
    """scene_dual.xml's obstacle, sweeping as OBSTACLE_* say."""
    print(f"Obstacle: crosses +-{OBSTACLE_AMPLITUDE} m along {OBSTACLE_AXIS} around {OBSTACLE_CENTER} at "
          f"{OBSTACLE_SPEED} m/s" + (f", waiting {OBSTACLE_DWELL} s at each end" if OBSTACLE_DWELL else ""))
    return MovingObstacle(model, OBSTACLE_CENTER, axis=OBSTACLE_AXIS, amplitude=OBSTACLE_AMPLITUDE,
                          speed=OBSTACLE_SPEED, dwell=OBSTACLE_DWELL)


def control(arms, data, obstacle=None):
    """One control update: every arm's model synced to the sim, then each arm's nominal command filtered
    against its own hazards, the other arm's links where they are now and the obstacle (its capsule this
    tick, or None), into the servos. Returns each arm's capsules in world frame (RobotGeometry.world's
    env list), for the monitors and the overlay."""
    for dual_arm in arms:
        dual_arm.sync(data)
    caps = [dual_arm.arm.geometry.world(dual_arm.arm.data) for dual_arm in arms]
    for dual_arm, own, other in zip(arms, caps, caps[::-1]):
        data.ctrl[dual_arm.ctrl_idx] = dual_arm.safety.filter(dual_arm.nominal(), caps=own, others=other["env"],
                                                              obstacle=obstacle)
    return [c["env"] for c in caps]


class ArmArmMonitor:
    """Closest the two arms' capsules get, and whether MuJoCo ever has them in contact."""

    def __init__(self, dual_model):
        self.model = dual_model
        self.bodies = [{b for b in range(dual_model.nbody) if dual_model.body(b).name.startswith(prefix)}
                       for prefix, *_ in ARMS]
        self.min_clearance = np.inf
        self.contact_ticks = 0

    def update(self, dual_data, caps_a, caps_b):
        """Records this tick; returns this tick's clearance (m)."""
        I, J = (idx.ravel() for idx in np.meshgrid(np.arange(len(caps_a)), np.arange(len(caps_b)), indexing="ij"))
        get = lambda caps, key, idx: np.array([caps[k][key] for k in idx])  # noqa: E731
        clearance = capsule_capsule_distance_batch(
            get(caps_a, "a", I), get(caps_a, "b", I), get(caps_a, "radius", I),
            get(caps_b, "a", J), get(caps_b, "b", J), get(caps_b, "radius", J))[0].min()
        self.min_clearance = min(self.min_clearance, clearance)
        body_of = self.model.geom_bodyid
        a, b = self.bodies
        self.contact_ticks += any(
            {body_of[c.geom1], body_of[c.geom2]} & a and {body_of[c.geom1], body_of[c.geom2]} & b
            for c in dual_data.contact[:dual_data.ncon])
        return clearance


class DualObstacleMonitor:
    """Closest the obstacle gets to either arm, and whether MuJoCo ever has it touching one. (mj.safety's
    ObstacleMonitor matches bodies by the arm model's ids, which aren't the two-arm model's.)"""

    def __init__(self, dual_model):
        self.model = dual_model
        self.obstacle_body = dual_model.body("obstacle").id
        self.arm_bodies = {b for b in range(dual_model.nbody)
                           if dual_model.body(b).name.startswith(tuple(prefix for prefix, *_ in ARMS))}
        self.min_clearance = np.inf
        self.contact_ticks = 0

    def update(self, dual_data, world_caps, obstacle):
        """Records this tick; returns True on the first tick the obstacle touches an arm."""
        caps = [cap for arm_caps in world_caps for cap in arm_caps]
        A = np.array([c["a"] for c in caps])
        self.min_clearance = min(self.min_clearance, capsule_capsule_distance_batch(
            A, np.array([c["b"] for c in caps]), np.array([c["radius"] for c in caps]),
            np.broadcast_to(obstacle["a"], A.shape), np.broadcast_to(obstacle["b"], A.shape),
            np.full(len(caps), obstacle["radius"]))[0].min())
        body_of = self.model.geom_bodyid
        touching = any(self.obstacle_body in (body_of[c.geom1], body_of[c.geom2])
                       and {body_of[c.geom1], body_of[c.geom2]} & self.arm_bodies
                       for c in dual_data.contact[:dual_data.ncon])
        first_hit = touching and self.contact_ticks == 0
        self.contact_ticks += touching
        return first_hit

    def summary(self):
        return f"obstacle: closest {self.min_clearance * 100:.1f} cm, {self.contact_ticks} contact ticks"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planner", choices=sorted(PLANNERS), default="sqp")
    parser.add_argument("--seed", type=int, default=None,
                        help="replay a run: the seed it printed at startup (default: a fresh one)")
    parser.add_argument("--record", metavar="PATH", default=None,
                        help="also record video: PATH=out.mp4 writes out_front.mp4 and out_top.mp4")
    parser.add_argument("--no-obstacle", action="store_true", help="leave the obstacle parked off the table")
    parser.add_argument("--show-collision", action="store_true",
                        help="overlay the capsules the planner and CBF see")
    args = parser.parse_args(argv)

    model = load_dual_model(prefixes=tuple(prefix for prefix, *_ in ARMS), joint_vel_max=CBF.joint_vel_max)
    data = mujoco.MjData(model)  # qpos0: every arm joint at 0
    place_blocks(model, data)
    arms = [DualArm(model, *spec) for spec in ARMS]
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

    monitor = ArmArmMonitor(model)
    obstacle = None if args.no_obstacle else make_obstacle(model)
    obstacle_monitor = DualObstacleMonitor(model)
    world_caps, obstacle_now = [None, None], None

    def draw_overlays(scn):
        if args.show_collision:
            for dual_arm, caps in zip(arms, world_caps):
                draw_capsules(scn, caps, rgba=dual_arm.rgba)
            if obstacle_now is not None:
                draw_capsules(scn, [obstacle_now], rgba=(0.95, 0.75, 0.1, 0.35))

    dt = model.opt.timestep
    step, next_report = 0, 0.0
    recording = Recorder(model, args.record) if args.record else contextlib.nullcontext()
    with mujoco.viewer.launch_passive(model, data) as viewer, recording as recorder:
        while viewer.is_running():
            t_wall = time.perf_counter()
            if obstacle is not None:
                obstacle_now = obstacle.update(data)
            if step % CONTROL_EVERY == 0:
                world_caps = control(arms, data, obstacle_now)
                clearance = monitor.update(data, *world_caps)
                if obstacle_now is not None and obstacle_monitor.update(data, world_caps, obstacle_now):
                    print(f"[{data.time:6.2f}s] WARNING: the obstacle hit an arm")
            if data.time >= next_report:
                next_report += REPORT_EVERY
                print(f"t={data.time:5.1f}s  arm-arm clearance {clearance * 100:5.1f} cm (min {monitor.min_clearance * 100:.1f}), "
                      + ", ".join(f"{a.name} {a.hover_error() * 100:.1f} cm from goal" for a in arms)
                      + f", {monitor.contact_ticks} contact ticks")
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

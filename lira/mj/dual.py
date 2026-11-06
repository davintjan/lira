"""
The two-arm scene (scenes/scene_dual.xml), as both dual sims run it. The physics runs in the two-arm
model; each arm is planned and filtered on its own single-arm model (DualArm), its base moved to where
the scene mounts it so both share world coordinates, and synced from the two-arm sim every control tick.
The obstacle sweeps across both boards between the arms, and both dodge it.
"""
import dataclasses

import mujoco
import numpy as np

from mj.arm import Arm
from mj.monitor import ContactMonitor
from mj.robots import ROBOTS
from mj.safety import SafetyFilter
from mj.scene import EnvironmentGeometry, MovingObstacle

CONTROL_EVERY = 2  # physics steps per control update: 250 Hz. Two arms' filters take ~3 ms, over a 2 ms step

# The obstacle: crosses both boards (along y) between the two arms, at wrist height while they push (the
# rod tip at 2.5 cm, the flange ~17.5 cm), waiting OBSTACLE_DWELL at each end.
OBSTACLE_CENTER = [0.1, 0.0, 0.30]  # m, world frame: the middle of the shared strip
OBSTACLE_AXIS = [0, 1, 0]
OBSTACLE_AMPLITUDE = 0.40           # m either side of the centre: past both goals
OBSTACLE_SPEED = 0.2                # m/s
OBSTACLE_DWELL = 3.0                # s parked at each end; None = sweep back and forth without stopping


class DualArm:
    """One arm of the two-arm scene: its single-arm Arm (planning, geometry, CBF) and where its joints and
    servos sit in the two-arm model.

    prefix: the arm's prefix in scene_dual.xml; target_block: the T it works on; cbf: its CBFConfig."""

    def __init__(self, dual_model, prefix, robot_name, target_block, cbf):
        mount = dual_model.body(prefix + "mount")
        w, _, _, z = mount.quat  # scene_dual.xml mounts the arms turned about z only
        robot = dataclasses.replace(ROBOTS[robot_name], base_pos=tuple(mount.pos),
                                    base_yaw=float(2 * np.arctan2(z, w)), start_key=None)
        self.arm = Arm(robot, joint_vel_max=cbf.joint_vel_max)
        self.safety = SafetyFilter(self.arm, cbf)
        self.name, self.target_block = robot_name, target_block
        model = self.arm.model
        joints = [dual_model.joint(prefix + model.joint(j).name) for j in range(self.arm.nu)]
        self.qpos_idx = np.array([j.qposadr[0] for j in joints])
        self.dof_idx = np.array([j.dofadr[0] for j in joints])
        self.ctrl_idx = np.array([dual_model.actuator(prefix + model.actuator(a).name).id for a in range(self.arm.nu)])
        # the arm's model has one T (t_block): it stands in for the T this arm works on
        self.block_qpos = model.jnt_qposadr[model.body("t_block").jntadr[0]]
        self.dual_block_qpos = dual_model.jnt_qposadr[dual_model.body(target_block).jntadr[0]]
        # every other T, which the arm's model doesn't have: its boxes in world frame (both models share
        # it) from the two-arm sim, refreshed by sync -- the plan and the CBF keep clear of them
        names = [dual_model.body(b).name for b in range(dual_model.nbody)]
        self.other_blocks = [EnvironmentGeometry(dual_model, object_body_name=name) for name in names
                             if name.startswith("t_block") and name != target_block]
        self.other_boxes = []

    def sync(self, dual_data):
        """The arm's model to the two-arm sim's state: the time, its joints and its target T; and where
        the other Ts are."""
        data = self.arm.data
        data.time = dual_data.time
        data.qpos[:self.arm.nu] = dual_data.qpos[self.qpos_idx]
        data.qvel[:self.arm.nu] = dual_data.qvel[self.dof_idx]
        data.qpos[self.block_qpos:self.block_qpos + 7] = dual_data.qpos[self.dual_block_qpos:self.dual_block_qpos + 7]
        mujoco.mj_forward(self.arm.model, data)
        self.other_boxes = [box for block in self.other_blocks for box in block.update(dual_data)]


def make_obstacle(dual_model):
    """scene_dual.xml's obstacle, sweeping as OBSTACLE_* say."""
    print(f"Obstacle: crosses +-{OBSTACLE_AMPLITUDE} m along {OBSTACLE_AXIS} around {OBSTACLE_CENTER} at "
          f"{OBSTACLE_SPEED} m/s" + (f", waiting {OBSTACLE_DWELL} s at each end" if OBSTACLE_DWELL else ""))
    return MovingObstacle(dual_model, OBSTACLE_CENTER, axis=OBSTACLE_AXIS, amplitude=OBSTACLE_AMPLITUDE,
                          speed=OBSTACLE_SPEED, dwell=OBSTACLE_DWELL)


def make_monitors(dual_model, prefixes):
    """ContactMonitors for the two arms against each other, and for both against the obstacle. prefixes:
    the two arms' prefixes in scene_dual.xml."""
    bodies_a, bodies_b = (arm_bodies(dual_model, prefix) for prefix in prefixes)
    arms = ContactMonitor(dual_model, bodies_a, bodies_b)
    obstacle = ContactMonitor(dual_model, bodies_a | bodies_b, {dual_model.body("obstacle").id})
    return arms, obstacle


def arm_bodies(dual_model, prefix):
    """The body ids of the arm with this prefix in the two-arm model."""
    return {b for b in range(dual_model.nbody) if dual_model.body(b).name.startswith(prefix)}

"""Builds the MuJoCo model for one robot: its scene xml, with the edits every sim needs applied via
MjSpec, so the vendored robot xml stays untouched."""
import contextlib
from dataclasses import dataclass

import mujoco
import numpy as np

from mj.robots import SCENE_DIR

# [w, x, y, z]: 180 deg about y -- turns a frame's z axis upside down. The flange's z is the tool
# axis, so a flange at this orientation (or at a body's orientation composed with it) points the
# rod straight down.
TOOL_DOWN_QUAT = np.array([0.0, 0.0, 1.0, 0.0])


@dataclass(frozen=True)
class Pusher:
    """The rod mounted on the flange, pointing out along the tool axis. Its tip plays Push-T's pusher."""
    radius: float = 0.015  # m: matches Push-T's pusher circle (15 px at 1 px = 1 mm)
    length: float = 0.15   # m, flange to tip: keeps the wrist well above the T while the tip pushes it

    @property
    def tip_offset(self):
        """m along the tool axis from the flange to the probe (the centre of the rounded tip)."""
        return self.length - self.radius

    @property
    def flange_above_probe(self):
        """World offset from the probe to the flange while the rod points straight down."""
        return np.array([0.0, 0.0, self.tip_offset])


def load_model(robot, pusher=Pusher(), joint_vel_max=10.0):
    """robot's scene (a RobotConfig), with:
      - the base moved to robot.base_pos
      - the pusher rod on the flange (attachment_site): a collision capsule (group 3, so
        RobotGeometry, the planner and the CBF guard it like any link), a visible cylinder + tip
        (group-3 geoms aren't drawn by default), and a "probe" site at the centre of its tip
      - the arm switched to velocity control: each position servo becomes a velocity servo with the
        same kv, force = kv * (ctrl - qvel), so ctrl is simply the joint velocity the controller
        wants. A velocity servo has no position spring to hold the arm up, so gravity is compensated
        through the actuators (actgravcomp: counted against the motor force limits, as on hardware).
    Opened by a relative path from inside SCENE_DIR, where the vendored robots' meshes resolve
    (see scenes/generate_vendor_links.py)."""
    with contextlib.chdir(SCENE_DIR):
        return _build(robot, pusher, joint_vel_max)


def _build(robot, pusher, joint_vel_max):
    spec = mujoco.MjSpec.from_file(robot.scene)  # relative: MuJoCo only applies the vendored xml's meshdir then
    spec.body("base").pos = robot.base_pos
    _add_pusher(spec, pusher)

    kv = -spec.compile().actuator_biasprm[:, 2]  # position servo bias is [0, -kp, -kv]
    arm_joints = []
    for act, gain in zip(spec.actuators, kv):
        act.set_to_velocity(kv=gain)
        act.ctrlrange = [-joint_vel_max, joint_vel_max]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        arm_joints.append(spec.joint(act.target))
    for joint in arm_joints:
        joint.actgravcomp = True
        joint.parent.gravcomp = 1.0  # the body this joint moves
    return spec.compile()


def _add_pusher(spec, pusher):
    flange = spec.site("attachment_site")
    rod = flange.parent.add_body(name="pusher", pos=flange.pos, quat=flange.quat)
    rod.gravcomp = 1.0  # rides on the wrist: held up like the links are
    r, length = pusher.radius, pusher.length
    rod.add_geom(name="pusher", type=mujoco.mjtGeom.mjGEOM_CAPSULE, group=3, contype=1, conaffinity=1,
                 size=[r, length / 2 - r, 0], pos=[0, 0, length / 2], mass=0.1)
    rod.add_geom(name="pusher_visual", type=mujoco.mjtGeom.mjGEOM_CYLINDER, group=2, contype=0, conaffinity=0,
                 size=[r, length / 2 - r, 0], pos=[0, 0, length / 2 - r / 2],
                 density=0, rgba=[0.15, 0.15, 0.15, 1])
    rod.add_geom(name="pusher_tip_visual", type=mujoco.mjtGeom.mjGEOM_SPHERE, group=2, contype=0, conaffinity=0,
                 size=[r, 0, 0], pos=[0, 0, pusher.tip_offset], density=0, rgba=[0.9, 0.9, 0.9, 1])
    rod.add_site(name="probe", pos=[0, 0, pusher.tip_offset])

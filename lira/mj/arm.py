"""Arm: one robot in its scene -- the compiled model, the live sim data and the collision geometry the
planner and the CBF read. Built from a RobotConfig, so code holding an Arm never needs to know
which robot it is."""
import time

import mujoco
import numpy as np

from mj.mj_interface import site_pose
from mj.model import Pusher, load_model
from mj.robots import SCENE_DIR
from mj.scene import EnvironmentGeometry, RobotGeometry


class Arm:
    def __init__(self, robot, pusher=Pusher(), joint_vel_max=10.0):
        self.robot = robot
        self.pusher = pusher
        self.model = load_model(robot, pusher, joint_vel_max)
        self.data = mujoco.MjData(self.model)
        self.nu = self.model.nu  # the arm's joints: actuated, and first in qpos/qvel
        t0 = time.perf_counter()
        self.geometry = RobotGeometry(self.model, excluded_bodies=robot.excluded_bodies,
                                      base_capsule=robot.base_capsule,
                                      cache_dir=SCENE_DIR / ".cache", cache_name=robot.name)
        self.geometry_seconds = time.perf_counter() - t0  # first run samples the hazards; later runs hit the cache
        self.env = EnvironmentGeometry(self.model)
        self.reset()

    def reset(self):
        """Arm at the robot's start pose (its start_key keyframe, or qpos0). Only the arm's joints:
        the robot xml's keyframe doesn't know about the T's freejoint."""
        if self.robot.start_key is not None:
            self.data.qpos[:self.nu] = self.model.key(self.robot.start_key).qpos[:self.nu]
        mujoco.mj_forward(self.model, self.data)

    @property
    def q(self):
        """Joint positions (a copy)."""
        return self.data.qpos[:self.nu].copy()

    def flange_pose(self):
        """(position, [w, x, y, z]) of the flange (attachment_site), where the rod is mounted."""
        return site_pose(self.data)

    def probe_pos(self):
        """The rod tip's centre: the point that plays Push-T's pusher."""
        return self.data.site("probe").xpos.copy()

    def command(self, qdot):
        """Joint velocity to the velocity servos (see mj.model.load_model)."""
        self.data.ctrl[:self.nu] = qdot

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def push_flange(self, magnitude, direction, start, duration, site="attachment_site"):
        """Simulated external push: magnitude N along direction (world frame, normalized here) at
        `site`, while start <= time < start + duration; nothing otherwise. Call it every tick, before
        step(). data.xfrc_applied acts at the body's centre of mass, so a force at the site also needs
        the torque it makes about it; the row is overwritten every call, so the push really stops."""
        model, data = self.model, self.data
        body = model.site_bodyid[model.site(site).id]
        active = start <= data.time < start + duration
        force = magnitude * np.asarray(direction, dtype=float) / np.linalg.norm(direction) if active else np.zeros(3)
        data.xfrc_applied[body, :3] = force
        data.xfrc_applied[body, 3:] = np.cross(data.site(site).xpos - data.xipos[body], force)

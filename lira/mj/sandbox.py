"""KinematicsSandbox: an arm's kinematics at an arbitrary q, for planning and IK (core/planner.py,
core/ik.py), on a scratch MjData so the live sim is never touched."""
import mujoco
import numpy as np

# aliased so KinematicsSandbox's same-named methods below don't look recursive
from mj.mj_interface import site_jacobian as _mj_site_jacobian, point_jacobian as _mj_point_jacobian


class KinematicsSandbox:
    """Evaluates arm kinematics at an arbitrary q via a scratch MjData, without touching the live sim.
    Call set_q(q) before any accessor."""

    def __init__(self, model, geometry):
        self.model = model
        self.data = mujoco.MjData(model)
        self.geometry = geometry
        self._site_id = model.site("attachment_site").id

    def set_q(self, q):
        self.data.qpos[:len(q)] = q  # the arm's joints come first in qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def ee_pos(self):
        return self.data.site(self._site_id).xpos.copy()

    def ee_quat(self):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site(self._site_id).xmat)
        return quat

    def ee_jacobian_full(self):
        return _mj_site_jacobian(self.model, self.data)

    def point_jacobian(self, body_id, point):
        return _mj_point_jacobian(self.model, self.data, body_id, point)

    def self_capsules(self):
        return self.geometry.update(self.data)

    def env_capsules(self):
        return self.geometry.update_all(self.data)

    def link_pairs(self):
        return self.geometry.link_pairs_world(self.data)

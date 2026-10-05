"""Glue between live MuJoCo `data` and numpy: pose/Jacobian reads, and one-shot velocity-level IK (diff_ik)."""
import mujoco
import numpy as np


def pose_pub(mj_data, site="attachment_site"):
    site = mj_data.site(site)
    ee_quat = np.zeros(4)
    mujoco.mju_mat2Quat(ee_quat, site.xmat)  # sites have no xquat
    return site.xpos.copy(), ee_quat


def obj_pose_pub(mj_data, name):
    body = mj_data.body(name)
    return body.xpos.copy(), body.xquat.copy()


def site_jacobian(model, data, site="attachment_site"):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, model.site(site).id)
    return np.vstack([jacp, jacr])[:, :6]


def point_jacobian(model, data, body_id, point):
    """3x6 position Jacobian of a world-frame point rigidly attached to body_id."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
    return jacp[:, :6]


def diff_ik(model, data, xdot, omega, damping=1e-2, site="attachment_site"):
    """Damped-least-squares velocity IK: twist (xdot, omega) of `site` -> joint velocity."""
    J = site_jacobian(model, data, site)
    twist = np.hstack([xdot, omega])
    return J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), twist)

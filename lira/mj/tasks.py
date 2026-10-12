"""Nominal controllers: what the arm would do with no hazards around. Each returns the joint velocity
u_des at the live state; SafetyFilter.filter then makes it safe. (The planned path's field,
core.path_field.PathVelocityField, is the third one: it maps q to u_des directly.)"""
import numpy as np

from core.ds import clip_norm, damped_least_squares, linear_pos_ds, orientation_ds
from mj.mj_interface import diff_ik, site_jacobian


def flange_pose_ds(arm, target_pos, target_quat, k_p, k_o, max_speed, damping):
    """Straight-line DS on the flange's full pose: linear in position (capped at max_speed m/s),
    log-map in orientation, through damped-least-squares IK."""
    pos, quat = arm.flange_pose()
    xdot = clip_norm(linear_pos_ds(pos, target_pos, k_p), max_speed)
    omega = orientation_ds(quat, target_quat, k_o)
    return diff_ik(arm.model, arm.data, xdot, omega, damping=damping)


def probe_ds(arm, target, gain, max_speed, k_o, damping, metric_reg):
    """Cartesian DS on the probe, xdot = gain * (target - probe) capped at max_speed, while the rod is
    turned toward vertical. Returns (u_des, metric) -- pass the metric on to SafetyFilter.filter.

    The task is 5-D -- probe position plus the rod's tilt -- and leaves the spin about the rod free:
    the rod is round, so spinning changes nothing for the push, and it's the arm's spare freedom to
    stay unfolded near its base (holding the full orientation there folds the wrist into the upper
    arm). The metric is the same task's (J^T J + metric_reg * I), so the CBF spends the free spin
    first and only minds other joint motion a little."""
    xdot = clip_norm(gain * (np.asarray(target) - arm.probe_pos()), max_speed)
    axis = arm.data.site("probe").xmat.reshape(3, 3)[:, 2]  # the rod's direction, flange -> tip
    omega = k_o * np.cross(axis, [0.0, 0.0, -1.0])         # turns the rod toward pointing down; never about itself
    tilt_basis = np.linalg.svd(axis[None, :])[2][1:]       # two unit vectors perpendicular to the rod
    J = site_jacobian(arm.model, arm.data, "probe")
    J_task = np.vstack([J[:3], tilt_basis @ J[3:]])
    u_des = damped_least_squares(J_task, np.concatenate([xdot, tilt_basis @ omega]), damping)
    metric = J_task.T @ J_task + metric_reg * np.eye(arm.nu)
    return u_des, metric

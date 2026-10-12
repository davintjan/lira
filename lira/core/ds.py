"""Reactive dynamical systems and velocity-level IK: the nominal commands, before the CBF filters
them. Pure math: no MuJoCo, no state."""
import numpy as np

_POS_DS_SCALING = np.array([1.0, 1.0, 0.8])  # per-axis: z converges a little slower


def linear_pos_ds(pos, target, k):
    """xdot = A @ x + b: linear DS pulling pos toward target, A = -k * diag(scaling)."""
    A = -k * np.diag(_POS_DS_SCALING)
    b = k * _POS_DS_SCALING * np.array(target)
    return A @ np.array(pos) + b


def orientation_ds(quat, target_quat, k):
    """omega = -k * log(dq): angular velocity toward target_quat via the quaternion log-map error.
    Quaternions are scalar-first [w, x, y, z]. At k = 1 its norm is the angle error."""
    s1, u1 = quat[0], np.asarray(quat[1:])
    s2, u2 = target_quat[0], np.asarray(target_quat[1:])
    dori = np.array([s1 * s2 + u1 @ u2, *(-s1 * u2 + s2 * u1 - np.cross(u1, u2))])
    if dori[0] < 0:  # q and -q are the same rotation: take the short way, not the 2pi-minus-angle one
        dori = -dori
    dori[0] = np.clip(dori[0], -1.0, 1.0)
    v = dori[1:]
    v_norm = np.linalg.norm(v)
    logdq = np.arccos(dori[0]) * (v / v_norm) if v_norm > 1e-6 else np.zeros(3)
    return -k * logdq


def clip_norm(v, max_norm):
    """v scaled down to max_norm if it's longer."""
    norm = np.linalg.norm(v)
    return v * (max_norm / norm) if norm > max_norm else v


def damped_least_squares(J, task, damping):
    """Joint velocity for a task velocity: J^T (J J^T + damping^2 I)^-1 task. The damping keeps it
    finite near singularities."""
    return J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(len(task)), task)

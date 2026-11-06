"""
Numerical IK on a KinematicsSandbox (mj/sandbox.py): one seed to convergence (numeric_ik), or many
seeds for every distinct branch (solve_ik_candidates) -- both inside the joint limits less a margin
(joint_limits), which the planners keep to as well.
"""
import numpy as np

from core.ds import damped_least_squares, orientation_ds

JOINT_LIMIT_MARGIN = 0.02  # rad: IK and the planner keep joints this far inside their limits


def joint_limits(model, n_dof=None):
    """(lo, hi) for the arm's joints, JOINT_LIMIT_MARGIN inside the model's limits: a goal planned
    exactly on a limit is one the sim's soft limit keeps pushing the arm back from, so it never
    settles there."""
    n_dof = n_dof or model.nu
    return model.jnt_range[:n_dof, 0] + JOINT_LIMIT_MARGIN, model.jnt_range[:n_dof, 1] - JOINT_LIMIT_MARGIN


def numeric_ik(sandbox, target_pos, target_quat, q_seed, model,
               max_iters=150, pos_tol=5e-3, rot_tol=5e-2, step_scale=0.5, damping=1e-2):
    """Iterative damped-least-squares IK from a single seed (the same
    twist-tracking math as mj_interface.diff_ik, applied to convergence). Clips to joint_limits (JOINT_LIMIT_MARGIN inside the real ones) every iteration.
    Returns (q, converged)."""
    q = np.array(q_seed, dtype=float)
    lo, hi = joint_limits(model)
    for _ in range(max_iters):
        sandbox.set_q(q)
        pos_err = np.array(target_pos) - sandbox.ee_pos()
        omega = orientation_ds(sandbox.ee_quat(), target_quat, 1.0)  # norm ~ angle error at k=1
        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(omega) < rot_tol:
            return q, True
        dq = damped_least_squares(sandbox.ee_jacobian_full(), np.hstack([pos_err, omega]), damping)
        q = np.clip(q + step_scale * dq, lo, hi)
    return q, False


def _wrap_near(q, reference, lo, hi):
    """Rewrap each joint to its 2pi-equivalent closest to `reference`, so random-seed IK doesn't
    produce a needlessly-long "long way around" solution. Only where the rewrapped angle is still
    within the joint's limits: a +-170 deg joint (iiwa) can't take the shortcut through +-180."""
    wrapped = reference + (q - reference + np.pi) % (2 * np.pi) - np.pi
    return np.where((wrapped >= lo) & (wrapped <= hi), wrapped, q)


def solve_ik_candidates(sandbox, target_pos, target_quat, model, q_current,
                        n_random_seeds=8, dedupe_tol=0.15, rng=None, known=()):
    """Multi-start numerical IK: q_current plus random seeds, keep every seed that converges,
    dedupe near-identical solutions (same kinematic branch). Each result is wrapped near
    q_current first (see _wrap_near). known: solutions found already (an earlier call) -- only new
    ones are returned."""
    rng = rng or np.random.default_rng()
    lo, hi = joint_limits(model)
    q_current = np.array(q_current, dtype=float)
    seeds = [q_current] + [rng.uniform(lo, hi) for _ in range(n_random_seeds)]

    candidates = list(known)
    for seed in seeds:
        q, ok = numeric_ik(sandbox, target_pos, target_quat, seed, model)
        if not ok:
            continue
        q = np.clip(_wrap_near(q, q_current, lo, hi), lo, hi)
        if any(np.max(np.abs(q - c)) < dedupe_tol for c in candidates):
            continue
        candidates.append(q)
    return candidates[len(known):]

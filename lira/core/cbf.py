"""The CBF-QP safety filter: the joint velocity closest to the nominal one that keeps every clearance
row safe. Runs every 500 Hz tick. Pure math: the rows come from core.geometry."""
import numpy as np
import osqp
from scipy import sparse


def cbf_qp(u_des, rows, alpha=5.0, u_max=None, metric=None):
    """Closest u to u_des s.t. grad_i . u >= -alpha * h_i for every clearance row (h_i, grad_i) from
    geometry.clearance_rows, plus optional -u_max <= u <= u_max. Falls back to u = 0 (freeze) if
    infeasible -- including a violated row no joint can currently fix (zero gradient): a real
    collision, so stopping is right.
    "Closest" is (u - u_des)^T metric (u - u_des); metric=None is plain joint-space distance.
    A task-space metric (J^T J + a little identity) makes motions the task doesn't care about
    -- e.g. spinning a round tool about its axis -- nearly free, so the filter uses them first."""
    n_dof = len(u_des)
    if u_max is not None:
        # |grad . u| <= |grad|_1 * u_max inside the box, so these rows can never bind -- dropping them is exact
        rows = [(h, grad) for h, grad in rows if alpha * h < np.abs(grad).sum() * u_max]
    if not rows:
        return np.array(u_des) if u_max is None else np.clip(u_des, -u_max, u_max)  # box-only QP = clip (exact for metric=None)

    G = -np.vstack([grad for _, grad in rows])
    h_ub = alpha * np.array([h for h, _ in rows])
    l_bound = np.full_like(h_ub, -np.inf)

    if u_max is not None:
        G = np.vstack([G, np.eye(n_dof)])
        h_ub = np.concatenate([h_ub, np.full(n_dof, u_max)])
        l_bound = np.concatenate([l_bound, np.full(n_dof, -u_max)])

    W = np.eye(n_dof) if metric is None else np.asarray(metric)
    solver = osqp.OSQP()
    solver.setup(sparse.csc_matrix(2.0 * W), -2.0 * W @ np.asarray(u_des), sparse.csc_matrix(G), l_bound, h_ub,
                 verbose=False)
    result = solver.solve()

    if result.info.status not in ("solved", "solved inaccurate"):
        return np.zeros(n_dof)
    return result.x

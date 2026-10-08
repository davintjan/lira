"""
Turns a planned joint-space path into a velocity field u(q) with a convergence guarantee.

Shape: every path segment contributes a forward push along it plus a pull back onto it,
blended by smooth (Gaussian) weights on the distance to each segment -- continuous, unlike
picking the single nearest segment. The forward push tapers off with the distance still to go
along the path, and near the goal the blend hands over to the last segment alone, so the arm
arrives along the planned approach (not straight at the goal from wherever it drifted to).

Guarantee: one energy V(q) = (q - q*)^T P (q - q*) that every motion must decrease, as in
LPV-DS (Figueroa & Billard, 2018) -- but instead of learning local linear systems with a GMM
and an SDP, P is fitted once per plan (a small LP, milliseconds) so that the planned path
itself decreases V, and a closed-form filter then makes every velocity decrease it:
    dV/dt <= -2 * epsilon * V   everywhere   ->   the goal is globally exponentially stable,
whatever the blending weights do. P lets the path move away from the goal in plain distance
(a detour) as long as it still crosses to smaller ellipsoids (q - q*)^T P (q - q*).
"""
import numpy as np
from scipy.optimize import linprog


def _symmetric_basis(n):
    """Index pairs (i, j), i <= j, for the free entries of a symmetric n x n matrix."""
    return [(i, j) for i in range(n) for j in range(i, n)]


def _quad_coefficients(x, y, basis):
    """Coefficients c with x^T P y = c . p, p the free entries of symmetric P."""
    return np.array([x[i] * y[i] if i == j else x[i] * y[j] + x[j] * y[i] for i, j in basis])


def _to_matrix(p, basis, n):
    P = np.zeros((n, n))
    for (i, j), value in zip(basis, p):
        P[i, j] = P[j, i] = value
    return P


def _lp_with_psd_cuts(cost, A_ub, b_ub, A_eq, b_eq, bounds, basis, n, min_eig, max_cuts):
    """linprog whose first len(basis) variables are the free entries of a symmetric P, with
    P >= min_eig * I enforced lazily: after each solve, every eigenvector v whose eigenvalue is
    below min_eig adds the cut v^T P v >= min_eig. Cutting planes approach the bound only slowly,
    so any P with eigenvalues >= min_eig / 2 is accepted -- still safely positive definite, which
    is all the guarantee needs. Returns the solution vector, or None. If the cuts haven't settled
    after max_cuts rounds, the last solution is still returned as long as its P is positive definite."""
    A_ub, b_ub = list(A_ub), list(b_ub)
    n_extra = len(cost) - len(basis)
    last = None
    for _ in range(max_cuts):
        res = linprog(cost, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if res.status != 0:
            return None
        eigvals, eigvecs = np.linalg.eigh(_to_matrix(res.x[:len(basis)], basis, n))
        if eigvals[0] >= 0.5 * min_eig:
            return res.x
        last = res.x if eigvals[0] > 0.0 else None
        for value, v in zip(eigvals, eigvecs.T):
            if value < min_eig:
                A_ub.append(np.concatenate([-_quad_coefficients(v, v, basis), np.zeros(n_extra)]))
                b_ub.append(-min_eig)
    return last


def fitLyapunovP(path, samples_per_segment=5, min_eig=0.05, required_margin=0.2, max_cuts=60):
    """Fits P (n x n for n joints, symmetric, trace n, eigenvalues >= min_eig) so the planned path's own direction
    decreases V = (q - q*)^T P (q - q*) everywhere along it. With unit path directions d_k at
    sample points q_k, the margin m is the smallest  -(q_k - q*)^T P d_k / |q_k - q*|  -- for
    P = I that's the cosine of the angle between the path and the straight line to the goal.
    Every constraint is linear in P, so both stages are LPs:
      1. maximize m: is there any P that fits this path, and how well?
      2. among P with m >= min(required_margin, half of the best), take the one closest to the
         identity (smallest sum |P - I|) -- bend the energy only as much as the path needs. A path
         that already heads toward the goal keeps P = I.
    Returns (P, margin). margin > 0: the whole path decreases V. margin <= 0: no quadratic V fits
    this path; the field's filter still guarantees convergence, but cuts the corners that don't fit."""
    path = np.asarray(path, dtype=float)
    n = path.shape[1]
    goal = path[-1]
    basis = _symmetric_basis(n)
    n_p = len(basis)
    identity = np.array([1.0 if i == j else 0.0 for i, j in basis])

    decrease = []  # one row per sample: (q_k - q*)^T P d_k / |q_k - q*| = c . p
    for a, b in zip(path[:-1], path[1:]):
        seg = b - a
        if seg @ seg < 1e-12:
            continue
        d = seg / np.linalg.norm(seg)
        for s in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            r = a + s * seg - goal
            if np.linalg.norm(r) > 1e-6:
                decrease.append(_quad_coefficients(r, d, basis) / np.linalg.norm(r))
    decrease = np.array(decrease)
    p_bounds = [(min_eig, n) if i == j else (-n, n) for i, j in basis]  # diag >= min_eig: necessary for P >= min_eig * I

    # stage 1 -- variables [p, m]: maximize m  s.t.  c . p + m <= 0,  trace(P) = n
    x = _lp_with_psd_cuts(
        cost=np.append(np.zeros(n_p), -1.0),
        A_ub=np.hstack([decrease, np.ones((len(decrease), 1))]), b_ub=np.zeros(len(decrease)),
        A_eq=np.append(identity, 0.0)[None, :], b_eq=[float(n)],
        bounds=p_bounds + [(None, n)], basis=basis, n=n, min_eig=min_eig, max_cuts=max_cuts,
    )
    if x is None:  # no positive definite P found: plain distance is always a valid energy for the filter
        return np.eye(n), float(np.min(-decrease @ identity))
    best_margin = x[-1]
    if best_margin <= 0.0:
        return np.eye(n), best_margin  # no quadratic V fits: plain distance is the most neutral choice

    # stage 2 -- variables [p, s]: minimize sum s  s.t.  c . p <= -m_req,  |p - I| <= s,  trace(P) = n
    m_req = min(required_margin, 0.5 * best_margin)
    eye_p = np.eye(n_p)
    x = _lp_with_psd_cuts(
        cost=np.append(np.zeros(n_p), np.ones(n_p)),
        A_ub=np.vstack([np.hstack([decrease, np.zeros((len(decrease), n_p))]),
                        np.hstack([eye_p, -eye_p]), np.hstack([-eye_p, -eye_p])]),
        b_ub=np.concatenate([np.full(len(decrease), -m_req), identity, -identity]),
        A_eq=np.append(identity, np.zeros(n_p))[None, :], b_eq=[float(n)],
        bounds=p_bounds + [(0.0, None)] * n_p, basis=basis, n=n, min_eig=min_eig, max_cuts=max_cuts,
    )
    if x is None:
        return np.eye(n), best_margin
    P = _to_matrix(x[:n_p], basis, n)
    return P, float(np.min(-decrease @ x[:n_p]))


class PathVelocityField:
    """u(q) for following `path` (N, n_dof) to its last point. Built once per plan: fits P and caches
    the segments; calling it costs only the per-tick evaluation.

    k_tangent:    rad/s, forward speed along the path
    k_corrective: 1/s, pull back onto the path when off it
    sigma:        rad, width of the segment blending -- larger = smoother, cuts corners more
    goal_dist:    rad, braking distance: the forward push ramps down linearly to zero over the
                  last goal_dist of path length (gain k_tangent / goal_dist); also the radius of
                  the hand-over to the last segment alone
    epsilon:      1/s, minimum convergence rate enforced by the filter: dV/dt <= -2 epsilon V"""

    def __init__(self, path, k_tangent=1.0, k_corrective=2.0, sigma=0.15, goal_dist=0.15, epsilon=1.0):
        self.path = np.asarray(path, dtype=float)
        self.goal = self.path[-1]
        self.k_tangent = k_tangent
        self.k_corrective = k_corrective
        self.sigma = sigma
        self.goal_dist = goal_dist
        self.epsilon = epsilon

        starts, ends = self.path[:-1], self.path[1:]
        keep = np.einsum("ij,ij->i", ends - starts, ends - starts) > 1e-12
        self.seg_start = starts[keep]
        self.seg_vec = (ends - starts)[keep]
        self.seg_len_sq = np.einsum("ij,ij->i", self.seg_vec, self.seg_vec)
        self.seg_dir = self.seg_vec / np.sqrt(self.seg_len_sq)[:, None]
        seg_len = np.sqrt(self.seg_len_sq)
        self.length_after = np.cumsum(seg_len[::-1])[::-1] - seg_len  # path length after each segment's end

        self.P, self.margin = fitLyapunovP(self.path)

    def __call__(self, q):
        q = np.asarray(q, dtype=float)
        r = q - self.goal
        dist_goal = np.linalg.norm(r)

        # every segment's field: forward push (tapering over the last goal_dist of path length)
        # plus a pull onto the segment
        t = np.clip(np.einsum("ij,ij->i", q - self.seg_start, self.seg_vec) / self.seg_len_sq, 0.0, 1.0)
        closest = self.seg_start + t[:, None] * self.seg_vec
        to_go = self.length_after + (1.0 - t) * np.sqrt(self.seg_len_sq)
        speed = self.k_tangent * np.minimum(1.0, to_go / self.goal_dist)
        seg_fields = speed[:, None] * self.seg_dir - self.k_corrective * (q - closest)

        # path term: segments blended by distance
        dist_sq = np.einsum("ij,ij->i", q - closest, q - closest)
        weights = np.exp(-(dist_sq - dist_sq.min()) / (2.0 * self.sigma ** 2))  # shift by min: no underflow
        weights /= weights.sum()
        v_path = weights @ seg_fields

        # near the goal, hand over to the last segment alone: it's exactly 0 at the goal (no push
        # left, nothing to pull onto), so u(goal) = 0 -- the goal is an equilibrium -- while the
        # neighbouring segments' pulls, which aren't 0 there, fade out
        beta = np.exp(-dist_goal ** 2 / (2.0 * self.goal_dist ** 2))
        v = (1.0 - beta) * v_path + beta * seg_fields[-1]

        # stability filter: smallest change to v with dV/dt = 2 r^T P u <= -2 epsilon r^T P r
        g = self.P @ r
        excess = g @ v + self.epsilon * (r @ g)
        if excess > 0.0:
            v = v - excess / (g @ g) * g
        return v

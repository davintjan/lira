"""
Turns a planned joint-space path into a velocity field u(q) with a convergence guarantee.

Shape: every path segment contributes a forward push along it plus a pull back onto it,
blended by smooth (Gaussian) weights on the distance to each segment -- continuous, unlike
picking the single nearest segment. The forward push tapers off with the distance still to go
along the path, and near the goal the blend hands over to the last segment alone, so the arm
arrives along the planned approach (not straight at the goal from wherever it drifted to).

Guarantee: one energy V(q) that every motion must decrease -- how much path is still to go, plus a
penalty for being off the path:
    V(q) = min over segments k of   s_k(q) + (k_tube / 2) * d_k(q)^2
s_k is the path length from segment k's closest point to the goal, d_k the distance to that point.
Along the path V is exactly the path still to go, so ANY plan decreases it -- detours, S-bends,
backing away from the goal first -- where a quadratic (q - q*)^T P (q - q*) can only accept paths
that keep closing in on the goal. Taking the min over segments (rather than reading s from the
nearest segment) keeps V continuous: where the minimizing segment switches, both values are equal,
so V never jumps. k_tube sets where that switch happens between two parts of the path that pass
close to each other: the larger it is, the more an arm is credited only with the part it's actually
on. A closed-form filter then makes every velocity decrease V:
    dV/dt <= -gamma(V),   gamma(V) = min_progress * tanh(epsilon * V / min_progress)
exponential (rate epsilon) near the goal, at least min_progress far from it -> the goal is globally
asymptotically stable, locally exponentially, whatever the blending weights do.
"""
import numpy as np


class PathVelocityField:
    """u(q) for following `path` (N, n_dof) to its last point. Built once per plan: caches the
    segments; calling it costs only the per-tick evaluation.

    k_tangent:    rad/s, forward speed along the path
    k_corrective: 1/s, pull back onto the path when off it
    sigma:        rad, width of the segment blending -- larger = smoother, cuts corners more
    goal_dist:    rad, braking distance: the forward push ramps down linearly to zero over the
                  last goal_dist of path length (gain k_tangent / goal_dist); also the radius of
                  the hand-over to the last segment alone
    epsilon:      1/s, convergence rate the filter enforces near the goal: dV/dt <= -epsilon V there
    min_progress: rad/s, how fast V must fall far from the goal (default: half of k_tangent, so
                  following the path at full speed always satisfies it)
    k_tube:       1/rad, weight of the off-path penalty in V (see the module docstring)"""

    def __init__(self, path, k_tangent=1.0, k_corrective=2.0, sigma=0.15, goal_dist=0.15, epsilon=1.0,
                 min_progress=None, k_tube=100.0):
        self.path = np.asarray(path, dtype=float)
        self.goal = self.path[-1]
        self.k_tangent = k_tangent
        self.k_corrective = k_corrective
        self.sigma = sigma
        self.goal_dist = goal_dist
        self.epsilon = epsilon
        self.min_progress = 0.5 * k_tangent if min_progress is None else min_progress
        self.k_tube = k_tube

        starts, ends = self.path[:-1], self.path[1:]
        keep = np.einsum("ij,ij->i", ends - starts, ends - starts) > 1e-12
        self.seg_start = starts[keep]
        self.seg_vec = (ends - starts)[keep]
        self.seg_len_sq = np.einsum("ij,ij->i", self.seg_vec, self.seg_vec)
        self.seg_dir = self.seg_vec / np.sqrt(self.seg_len_sq)[:, None]
        self.seg_len = np.sqrt(self.seg_len_sq)
        self.length_after = np.cumsum(self.seg_len[::-1])[::-1] - self.seg_len  # path length after each segment's end

    def _closest(self, q):
        """Per segment: the parameter t of the closest point (clipped to [0, 1]) and the point itself."""
        t = np.clip(np.einsum("ij,ij->i", q - self.seg_start, self.seg_vec) / self.seg_len_sq, 0.0, 1.0)
        return t, self.seg_start + t[:, None] * self.seg_vec

    def energy(self, q):
        """V(q) and dV/dq, from the segment minimizing s_k + (k_tube / 2) d_k^2."""
        q = np.asarray(q, dtype=float)
        t, closest = self._closest(q)
        offset = q - closest
        values = self.length_after + (1.0 - t) * self.seg_len + 0.5 * self.k_tube * np.einsum("ij,ij->i", offset, offset)
        j = int(np.argmin(values))
        grad = self.k_tube * offset[j]
        if 0.0 < t[j] < 1.0:  # beside the segment: moving along it uses up path
            grad = grad - self.seg_dir[j]
        elif grad @ grad < 1e-18 and values[j] > 0.0:
            # exactly on a waypoint, where s has a corner: take the direction the path goes on in
            grad = -self.seg_dir[min(j + 1, len(self.seg_dir) - 1) if t[j] >= 1.0 else j]
        return values[j], grad

    def decrease_rate(self, V):
        """gamma(V): how fast V must fall at this V -- epsilon * V near the goal, min_progress far away."""
        return self.min_progress * np.tanh(self.epsilon * V / self.min_progress)

    def __call__(self, q):
        q = np.asarray(q, dtype=float)
        dist_goal = np.linalg.norm(q - self.goal)

        # every segment's field: forward push (tapering over the last goal_dist of path length)
        # plus a pull onto the segment
        t, closest = self._closest(q)
        to_go = self.length_after + (1.0 - t) * self.seg_len
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

        # stability filter: smallest change to v with dV/dt = grad . u <= -gamma(V)
        V, g = self.energy(q)
        excess = g @ v + self.decrease_rate(V)
        if excess > 0.0 and g @ g > 0.0:
            v = v - excess / (g @ g) * g
        return v


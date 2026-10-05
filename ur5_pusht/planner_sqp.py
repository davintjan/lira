"""
SQP path optimizer: a drop-in alternative to TrajectoryPlanner's gradient descent (planner.py).
Pick between them with PLANNER in ur5_sim.py.

Collision is a constraint here, not a soft cost. Each iteration builds a local model of the
problem at the current path and solves it exactly as one QP (OSQP):
  - smoothness exactly (it's already quadratic in the waypoints)
  - singularity linearized
  - every clearance row (h, dh/dq) from geometry.clearanceRows as h + dh/dq . dq >= 0
  - a trust region: each joint of each waypoint moves at most `trust` rad per step
The step is kept only if the real objective improves by at least accept_ratio of what the model
promised; otherwise the trust region shrinks and the QP is solved again.

Constraints use an exact L1 penalty mu * max(0, -h), as in TrajOpt (Schulman et al., 2014):
above some mu the optimum has zero violation, like a hard constraint, but the QP stays solvable
even when the straight-line seed runs through an obstacle. mu grows until the path is clear.
"""
import numpy as np
import osqp
from scipy import sparse

from planner import TrajectoryPlanner


class SQPTrajectoryPlanner(TrajectoryPlanner):
    """Same interface and scoring as TrajectoryPlanner -- optimize() returns (path, pathCost
    breakdown) -- so plan_trajectory ranks candidates from either planner the same way. Only how
    the path is found differs. Inherits margins, weights and feasibility_eps from TrajectoryPlanner;
    iters and lr there are gradient-descent settings and unused here."""

    def __init__(self, activation=0.05, mu_init=10.0, mu_scale=10.0, max_mu_rounds=4,
                 max_sqp_iters=20, trust_init=0.2, trust_min=1e-4, accept_ratio=0.25,
                 min_improve=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation        # m: rows start this far before their margin, so steps see them coming
        self.mu_init = mu_init              # L1 penalty weight per m of violation, first round
        self.mu_scale = mu_scale            # mu multiplier each round the result still intrudes
        self.max_mu_rounds = max_mu_rounds
        self.max_sqp_iters = max_sqp_iters  # accepted steps per mu round
        self.trust_init = trust_init        # rad: per-joint, per-waypoint step limit
        self.trust_min = trust_min          # rad: give up shrinking below this -- no step helps
        self.accept_ratio = accept_ratio    # keep a step if real gain >= this * predicted gain
        self.min_improve = min_improve      # converged once the model predicts less gain than this

    def optimize(self, q_start, q_goal, sandbox, object_boxes, table_height):
        """SQP from a straight-line seed (endpoints fixed), raising mu until the path is clear."""
        path = np.linspace(np.array(q_start, dtype=float), np.array(q_goal, dtype=float), self.n_waypoints)
        limits = sandbox.model.jnt_range[:6]

        mu = self.mu_init
        for _ in range(self.max_mu_rounds):
            path = self._sqp(path, sandbox, object_boxes, table_height, limits, mu)
            depth = max(self._deepest_intrusion(self._collision_rows(q, sandbox, object_boxes, table_height))
                        for q in path[1:-1])
            if depth <= self.feasibility_eps:
                break
            mu *= self.mu_scale

        return path, self.pathCost(path, sandbox, object_boxes, table_height)

    def _sqp(self, path, sandbox, object_boxes, table_height, limits, mu):
        """Trust-region SQP at a fixed mu. Returns the improved path."""
        trust = self.trust_init
        merit = self._merit(path, sandbox, object_boxes, table_height, mu)

        for _ in range(self.max_sqp_iters):
            rows, sing_cost, sing_grad = self._linearize(path, sandbox, object_boxes, table_height)
            while True:
                dq = self._solve_step(path, rows, sing_grad, mu, trust, limits)
                predicted = merit - self._model_merit(path, dq, rows, sing_cost, sing_grad, mu)
                if predicted < self.min_improve:
                    return path  # converged: the local model sees nothing left to gain

                trial = path.copy()
                trial[1:-1] += dq
                trial_merit = self._merit(trial, sandbox, object_boxes, table_height, mu)
                if (merit - trial_merit) / predicted >= self.accept_ratio:
                    path, merit = trial, trial_merit
                    trust *= 1.5  # the model was trustworthy: allow bigger steps
                    break
                trust *= 0.5  # the model overpromised: try a smaller step
                if trust < self.trust_min:
                    return path

        return path

    def _merit(self, path, sandbox, object_boxes, table_height, mu):
        """The real objective: smoothness + singularity + mu * total violation (m) at the interior waypoints."""
        merit = self._smoothness_cost(path)
        for q in path[1:-1]:
            merit += mu * sum(-h for h, _ in self._collision_rows(q, sandbox, object_boxes, table_height))
            merit += self._singularity_cost_grad(q, sandbox, with_grad=False)[0]
        return merit

    def _linearize(self, path, sandbox, object_boxes, table_height):
        """Local model ingredients at every interior waypoint: clearance rows as (waypoint index, h,
        dh/dq), the singularity cost, and its gradient (n_interior, 6)."""
        rows = []
        sing_cost = 0.0
        sing_grad = np.zeros((len(path) - 2, 6))
        for k, q in enumerate(path[1:-1]):
            for h, grad in self._collision_rows(q, sandbox, object_boxes, table_height, activation=self.activation):
                rows.append((k, h, grad))
            cost, sing_grad[k] = self._singularity_cost_grad(q, sandbox)
            sing_cost += cost
        return rows, sing_cost, sing_grad

    def _model_merit(self, path, dq, rows, sing_cost, sing_grad, mu):
        """What the local model predicts the merit is after stepping by dq."""
        stepped = path.copy()
        stepped[1:-1] += dq
        violation = sum(max(0.0, -(h + grad @ dq[k])) for k, h, grad in rows)
        return self._smoothness_cost(stepped) + sing_cost + float(np.sum(sing_grad * dq)) + mu * violation

    def _solve_step(self, path, rows, sing_grad, mu, trust, limits):
        """One QP over z = [dq (interior waypoints x 6 joints, flattened), t (one slack per row)]:
            minimize   smoothness(path + dq) + sing_grad . dq + mu * sum(t)
            subject to h_i + grad_i . dq[k_i] + t_i >= 0,   t >= 0     (L1 penalty via slacks)
                       |dq| <= trust,  joint limits hold after the step
        Smoothness is w_smooth * sum ||p[k+1] - p[k]||^2; with D the (N-1, N) difference matrix and
        L its interior columns, it's w_smooth * ||D @ path + L @ dq||^2 per joint."""
        n_interior = len(path) - 2
        n_dq = 6 * n_interior
        n_t = len(rows)

        D = np.diff(np.eye(len(path)), axis=0)
        L = D[:, 1:-1]
        P_dq = 2.0 * self.w_smooth * np.kron(L.T @ L, np.eye(6))
        q_dq = (2.0 * self.w_smooth * L.T @ (D @ path)).ravel() + sing_grad.ravel()
        P = sparse.block_diag([sparse.csc_matrix(P_dq), sparse.csc_matrix((n_t, n_t))], format="csc")
        q = np.concatenate([q_dq, np.full(n_t, mu)])

        # clearance rows: grad . dq[k] + t >= -h
        A_rows = np.zeros((n_t, n_dq + n_t))
        for i, (k, h, grad) in enumerate(rows):
            A_rows[i, 6 * k:6 * k + 6] = grad
            A_rows[i, n_dq + i] = 1.0
        lower_rows = np.array([-h for _, h, _ in rows])

        # trust region and joint limits on dq, t >= 0 on the slacks
        interior = path[1:-1]
        lower_dq = np.maximum(-trust, limits[:, 0] - interior).ravel()
        upper_dq = np.minimum(trust, limits[:, 1] - interior).ravel()

        A = sparse.vstack([sparse.csc_matrix(A_rows), sparse.eye(n_dq + n_t)], format="csc")
        lower = np.concatenate([lower_rows, lower_dq, np.zeros(n_t)])
        upper = np.concatenate([np.full(n_t, np.inf), upper_dq, np.full(n_t, np.inf)])

        solver = osqp.OSQP()
        solver.setup(P, q, A, lower, upper, verbose=False, polish=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=20000)
        result = solver.solve()
        if result.info.status not in ("solved", "solved inaccurate"):
            return np.zeros((n_interior, 6))  # no step: _sqp sees zero predicted gain and stops
        return result.x[:n_dq].reshape(n_interior, 6)

"""
Offline motion planning. TrajectoryPlanner is what every path planner shares: the cost of a joint-space
path -- smoothness, collision, a manipulability hinge -- and its score (path_cost), total path cost, not
just endpoint quality. Its subclasses find the path: planner_gd.GDTrajectoryPlanner by gradient
descent, planner_sqp.SQPTrajectoryPlanner by SQP. plan_trajectory ties it together: IK candidates
(core.ik), a path to each, keep the cheapest feasible one. Kinematics come from a KinematicsSandbox
(mj/sandbox.py). Runs once before the control loop starts, not every tick.
"""
import multiprocessing as mp
import os

import numpy as np

from core.geometry import clearance_rows, manipulability
from core.ik import solve_ik_candidates
from mj.sandbox import KinematicsSandbox


class TrajectoryPlanner:
    """A joint-space path's cost and score, the same for every planner, so plan_trajectory ranks
    candidates from either the same way. Subclasses implement optimize(q_start, q_goal, sandbox,
    object_boxes, table_height) -> (path, path_cost breakdown)."""

    def __init__(self, n_waypoints=12, margin=0.03, table_margin=0.005, manip_thresh=0.01, w_smooth=1.0,
                 w_collision=50.0, w_singularity=20.0, feasibility_eps=0.002, feasibility_penalty=1000.0):
        self.n_waypoints = n_waypoints
        self.margin = margin
        self.table_margin = table_margin
        self.manip_thresh = manip_thresh
        self.w_smooth = w_smooth
        self.w_collision = w_collision
        self.w_singularity = w_singularity
        # dominant penalty once intrusion_depth > feasibility_eps -- collision
        # is only a soft term above, so this stops a smoother-but-touching
        # path from beating a genuinely clear one (see _deepest_intrusion).
        self.feasibility_eps = feasibility_eps
        self.feasibility_penalty = feasibility_penalty

    def optimize(self, q_start, q_goal, sandbox, object_boxes, table_height):
        raise NotImplementedError

    def _collision_rows(self, q, sandbox, object_boxes, table_height, activation=0.0):
        """(h, dh/dq) for every hazard within `activation` m of its margin at q -- see
        geometry.clearance_rows. The default 0 keeps only the violated ones: the cost and the intrusion
        check need nothing else."""
        sandbox.set_q(q)
        caps = sandbox.world_capsules()
        return clearance_rows(
            sandbox.ee_pos(), sandbox.ee_jacobian_full()[:3], caps["self"], caps["pairs"], caps["env"],
            object_boxes, table_height, sandbox.point_jacobian,
            margin=self.margin, table_margin=self.table_margin, activation=activation,
        )

    @staticmethod
    def _deepest_intrusion(rows):
        """Deepest intrusion (m) into any collision margin among these rows, 0 if none. Every row
        counts: clearances fixed by design (e.g. shoulder_link vs the table) never become rows in
        the first place (see geometry.clearance_rows). Collision only, not singularity: nothing live
        enforces manipulability."""
        return max((-h for h, _ in rows), default=0.0)

    def _waypoint_collision_singularity_cost_grad(self, q, sandbox, object_boxes, table_height):
        """Collision + manipulability-hinge cost/gradient at one waypoint (smoothness is handled
        separately). Returns (collision_cost, singularity_cost, grad, intrusion_depth), see
        _deepest_intrusion for the last."""
        rows = self._collision_rows(q, sandbox, object_boxes, table_height)
        collision_cost = 0.0
        grad = np.zeros(len(q))

        # collision: squared hinge on how far each hazard is inside its margin
        for h, grad_h in rows:
            viol = -h
            collision_cost += self.w_collision * viol ** 2
            grad += -2.0 * self.w_collision * viol * grad_h

        singularity_cost, grad_sing = self._singularity_cost_grad(q, sandbox)
        grad += grad_sing

        return collision_cost, singularity_cost, grad, self._deepest_intrusion(rows)

    def _singularity_cost_grad(self, q, sandbox, with_grad=True):
        """Squared hinge on manipulability below manip_thresh, and its gradient -- finite
        differences, only paid when actually below threshold (and with_grad)."""
        sandbox.set_q(q)
        manip = manipulability(sandbox.ee_jacobian_full())
        if manip >= self.manip_thresh:
            return 0.0, np.zeros(len(q))
        viol_m = self.manip_thresh - manip
        cost = self.w_singularity * viol_m ** 2
        if not with_grad:
            return cost, np.zeros(len(q))
        grad_manip = np.zeros(len(q))
        eps = 1e-4
        for i in range(len(q)):
            dq = np.zeros(len(q))
            dq[i] = eps
            sandbox.set_q(q + dq)
            w_plus = manipulability(sandbox.ee_jacobian_full())
            sandbox.set_q(q - dq)
            w_minus = manipulability(sandbox.ee_jacobian_full())
            grad_manip[i] = (w_plus - w_minus) / (2 * eps)
        sandbox.set_q(q)  # restore
        return cost, -2.0 * self.w_singularity * viol_m * grad_manip

    def _smoothness_cost(self, path):
        diffs = path[1:] - path[:-1]
        return self.w_smooth * float(np.sum(diffs ** 2))

    def path_cost(self, path, sandbox, object_boxes, table_height):
        """Total path cost with a breakdown; adds a large feasibility_penalty if
        intrusion_depth exceeds feasibility_eps (see __init__)."""
        collision = 0.0
        singularity = 0.0
        intrusion_depth = 0.0
        for i in range(1, len(path) - 1):
            c, s, _, depth = self._waypoint_collision_singularity_cost_grad(path[i], sandbox, object_boxes, table_height)
            collision += c
            singularity += s
            intrusion_depth = max(intrusion_depth, depth)
        smooth = self._smoothness_cost(path)
        penalty = 0.0
        if intrusion_depth > self.feasibility_eps:
            penalty = self.feasibility_penalty + 1000.0 * intrusion_depth  # + term also ranks infeasible candidates among themselves
        return dict(smooth=smooth, collision=collision, singularity=singularity, intrusion_depth=intrusion_depth,
                    feasibility_penalty=penalty, total=smooth + collision + singularity + penalty)


# Per-process globals, set once by _init_worker after the pool forks -- each
# worker gets its own MjData (MjData isn't thread- or fork-safe to share) on the
# model it inherited through the fork, reused across every candidate that worker
# is assigned.
_worker_sandbox = None


def _init_worker(model, geometry):
    global _worker_sandbox
    _worker_sandbox = KinematicsSandbox(model, geometry)


def _optimize_worker(args):
    """Runs optimize() for one candidate in a worker process; same steps plan_trajectory
    used to do inline in its loop, just against this process's own sandbox."""
    planner, q_start, q_goal, object_boxes, table_height = args
    path, breakdown = planner.optimize(q_start, q_goal, _worker_sandbox, object_boxes, table_height)
    _worker_sandbox.set_q(q_goal)
    goal_manip = manipulability(_worker_sandbox.ee_jacobian_full())
    return dict(q_goal=q_goal, path=path, breakdown=breakdown, goal_manip=goal_manip)


def plan_trajectory(model, data, geometry, env, q_start, target_pos, target_quat,
                    planner, n_random_seeds=8, ik_rounds=4, parallel=False, rng=None,
                    boxes=()):
    """Full pipeline: multi-seed IK for candidate goals, optimize a path to each with `planner` (a
    TrajectoryPlanner subclass), pick the winner by total path cost (not isolated endpoint quality).
    Returns None if no seed converged, else {'best': ..., 'all_candidates': ...}.

    ik_rounds: up to this many rounds of n_random_seeds IK seeds, stopping at the first that finds a
    goal clear of collisions. One round's seeds can all land on colliding branches (e.g. every one
    elbow-down, through the table) while a clear branch exists.

    parallel: if True, each surviving candidate's optimize() call runs in its own worker
    process. Workers are forked, so they inherit this exact model -- including anything added
    after loading the XML, like mj.model.load_model's pusher rod (reloading the XML would drop it
    and shift every body id after it) -- and each makes its own MjData, which can't be shared
    across processes. False (default): sequential, single-process. Candidates are independent of
    each other, so this only changes wall-clock time, not the result.

    rng: numpy Generator for the random IK seeds -- the only randomness in planning, so a seeded
    one makes the whole plan reproducible. None: a fresh unseeded one each call.

    boxes: more boxes (center, rot, half_extents) to keep clear of besides env's block, where they are
    now -- e.g. another arm's T."""
    sandbox = KinematicsSandbox(model, geometry)
    object_boxes = env.update(data) + list(boxes)  # blocks don't move during planning -- one snapshot suffices
    table_height = env.table_height

    # a goal that itself collides (e.g. an IK branch with the elbow through the table) can't
    # have a clear path -- the endpoint is fixed -- so don't spend optimize() on it. If every
    # goal collides after ik_rounds, keep them all so the least-bad path still wins.
    candidates, clear_goals = [], []
    for _ in range(ik_rounds):
        found = solve_ik_candidates(sandbox, target_pos, target_quat, model, q_start,
                                    n_random_seeds=n_random_seeds, rng=rng, known=candidates)
        candidates += found
        clear_goals += [q for q in found
                        if planner._deepest_intrusion(planner._collision_rows(q, sandbox, object_boxes, table_height))
                        <= planner.feasibility_eps]
        if clear_goals:
            break
    if not candidates:
        return None
    candidates = clear_goals or candidates

    if parallel and len(candidates) > 1:
        n_workers = min(os.cpu_count() or 1, len(candidates))
        tasks = [(planner, q_start, q_goal, object_boxes, table_height) for q_goal in candidates]
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers, initializer=_init_worker, initargs=(model, geometry)) as pool:
            results = pool.map(_optimize_worker, tasks)
    else:
        results = []
        for q_goal in candidates:
            path, breakdown = planner.optimize(q_start, q_goal, sandbox, object_boxes, table_height)
            sandbox.set_q(q_goal)
            goal_manip = manipulability(sandbox.ee_jacobian_full())
            results.append(dict(q_goal=q_goal, path=path, breakdown=breakdown, goal_manip=goal_manip))

    best = min(results, key=lambda r: r["breakdown"]["total"])
    return dict(best=best, all_candidates=results)

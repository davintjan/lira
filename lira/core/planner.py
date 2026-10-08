"""
Offline motion planning. KinematicsSandbox evaluates kinematics at an
arbitrary q. TrajectoryPlanner (CHOMP-style) plans a joint-space path and
scores it by total path cost, not just endpoint quality. plan_trajectory
ties it together: try every IK candidate, keep the cheapest feasible path.
Runs once before the control loop starts (see arm_sim.main), not every tick.
"""
import multiprocessing as mp
import os

import mujoco
import numpy as np

from core.geometry import clearanceRows, manipulability
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


class TrajectoryPlanner:
    """CHOMP-style joint-space path optimizer: gradient descent against smoothness + collision
    (self/table/object) + a manipulability hinge, integrated over the whole path so IK candidates
    are compared by total path cost, not endpoint quality alone."""

    def __init__(self, n_waypoints=12, iters=60, tol=1e-4, lr=0.02, margin=0.03, table_margin=0.005,
                 manip_thresh=0.01, w_smooth=1.0, w_collision=50.0, w_singularity=20.0,
                 feasibility_eps=0.002, feasibility_penalty=1000.0):
        self.n_waypoints = n_waypoints
        self.iters = iters  # max passes over the path
        self.tol = tol      # rad: stop early once no joint of any waypoint moves more than this in a pass
        self.lr = lr
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

    def _collision_rows(self, q, sandbox, object_boxes, table_height, activation=0.0):
        """(h, dh/dq) for every hazard within `activation` m of its margin at q -- see
        geometry.clearanceRows. The default 0 keeps only the violated ones: the cost and the intrusion
        check need nothing else."""
        sandbox.set_q(q)
        return clearanceRows(
            sandbox.ee_pos(), sandbox.ee_jacobian_full()[:3], sandbox.self_capsules(), sandbox.link_pairs(),
            sandbox.env_capsules(), object_boxes, table_height, sandbox.point_jacobian,
            margin=self.margin, table_margin=self.table_margin, activation=activation,
        )

    @staticmethod
    def _deepest_intrusion(rows):
        """Deepest intrusion (m) into any collision margin among these rows, 0 if none. Every row
        counts: clearances fixed by design (e.g. shoulder_link vs the table) never become rows in
        the first place (see geometry.clearanceRows). Collision only, not singularity: nothing live
        enforces manipulability."""
        return max((-h for h, _ in rows), default=0.0)

    def _waypoint_collision_singularity_cost_grad(self, q, sandbox, object_boxes, table_height):
        """Collision + manipulability-hinge cost/gradient at one waypoint (smoothness is handled
        separately in optimize()). Returns (collision_cost, singularity_cost, grad, intrusion_depth),
        see _deepest_intrusion for the last."""
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

    def pathCost(self, path, sandbox, object_boxes, table_height):
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

    def optimize(self, q_start, q_goal, sandbox, object_boxes, table_height):
        """Gradient-descends the interior waypoints of a straight-line seed (endpoints fixed), for up
        to `iters` passes -- fewer once a pass moves no joint more than `tol` (the forces have
        balanced; more passes won't change the path). The breakdown's "iters" is how many it took."""
        q_start = np.array(q_start, dtype=float)
        q_goal = np.array(q_goal, dtype=float)
        path = np.linspace(q_start, q_goal, self.n_waypoints)

        for n_iters in range(1, self.iters + 1):
            step = 0.0
            for i in range(1, self.n_waypoints - 1):
                smooth_grad = 2.0 * self.w_smooth * (2 * path[i] - path[i - 1] - path[i + 1])
                _, _, cs_grad, _ = self._waypoint_collision_singularity_cost_grad(
                    path[i], sandbox, object_boxes, table_height
                )
                delta = self.lr * (smooth_grad + cs_grad)
                path[i] = path[i] - delta
                step = max(step, np.abs(delta).max())
            if step < self.tol:
                break

        breakdown = self.pathCost(path, sandbox, object_boxes, table_height)
        breakdown["iters"] = n_iters
        return path, breakdown


JOINT_LIMIT_MARGIN = 0.02  # rad: IK and the planner keep joints this far inside their limits


def joint_limits(model, n_dof=None):
    """(lo, hi) for the arm's joints, JOINT_LIMIT_MARGIN inside the model's limits: a goal planned
    exactly on a limit is one the sim's soft limit keeps pushing the arm back from, so it never
    settles there."""
    n_dof = n_dof or model.nu
    return model.jnt_range[:n_dof, 0] + JOINT_LIMIT_MARGIN, model.jnt_range[:n_dof, 1] - JOINT_LIMIT_MARGIN


def numeric_ik(sandbox, target_pos, target_quat, q_seed, controller, model,
                max_iters=150, pos_tol=5e-3, rot_tol=5e-2, step_scale=0.5, damping=1e-2):
    """Iterative damped-least-squares IK from a single seed (same twist-tracking math as
    mj_interface.diff_ik, applied to convergence). Clips to joint_limits (JOINT_LIMIT_MARGIN inside the real ones) every iteration.
    Returns (q, converged)."""
    q = np.array(q_seed, dtype=float)
    lo, hi = joint_limits(model)
    for _ in range(max_iters):
        sandbox.set_q(q)
        pos_err = np.array(target_pos) - sandbox.ee_pos()
        omega = controller.linearOrientationDS(sandbox.ee_quat(), target_quat, 1.0)  # norm ~ angle error at k=1
        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(omega) < rot_tol:
            return q, True
        J = sandbox.ee_jacobian_full()
        twist = np.hstack([pos_err, omega])
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), twist)
        q = np.clip(q + step_scale * dq, lo, hi)
    return q, False


def _wrap_near(q, reference, lo, hi):
    """Rewrap each joint to its 2pi-equivalent closest to `reference`, so random-seed IK doesn't
    produce a needlessly-long "long way around" solution. Only where the rewrapped angle is still
    within the joint's limits: a +-170 deg joint (iiwa) can't take the shortcut through +-180."""
    wrapped = reference + (q - reference + np.pi) % (2 * np.pi) - np.pi
    return np.where((wrapped >= lo) & (wrapped <= hi), wrapped, q)


def solve_ik_candidates(sandbox, target_pos, target_quat, controller, model, q_current,
                         n_random_seeds=8, dedupe_tol=0.15, rng=None):
    """Multi-start numerical IK: q_current plus random seeds, keep every seed that converges,
    dedupe near-identical solutions (same kinematic branch). Each result is wrapped near
    q_current first (see _wrap_near)."""
    rng = rng or np.random.default_rng()
    lo, hi = joint_limits(model)
    q_current = np.array(q_current, dtype=float)
    seeds = [q_current] + [rng.uniform(lo, hi) for _ in range(n_random_seeds)]

    candidates = []
    for seed in seeds:
        q, ok = numeric_ik(sandbox, target_pos, target_quat, seed, controller, model)
        if not ok:
            continue
        q = np.clip(_wrap_near(q, q_current, lo, hi), lo, hi)
        if any(np.max(np.abs(q - c)) < dedupe_tol for c in candidates):
            continue
        candidates.append(q)
    return candidates


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


def plan_trajectory(model, data, controller, geometry, env, q_start, target_pos, target_quat,
                     planner=None, n_random_seeds=8, parallel=False, n_workers=None, rng=None):
    """Full pipeline: multi-seed IK for candidate goals, CHOMP-optimize a path to each, pick
    the winner by total path cost (not isolated endpoint quality). Returns None if no seed
    converged, else {'best': ..., 'all_candidates': ...}.

    parallel: if True, each surviving candidate's optimize() call runs in its own worker
    process. Workers are forked, so they inherit this exact model -- including anything added
    after loading the XML, like arm_sim.load_model's pusher rod (reloading the XML would drop it
    and shift every body id after it) -- and each makes its own MjData, which can't be shared
    across processes. False (default): sequential, single-process. Candidates are independent of
    each other, so this only changes wall-clock time, not the result.

    rng: numpy Generator for the random IK seeds -- the only randomness in planning, so a seeded
    one makes the whole plan reproducible. None: a fresh unseeded one each call."""
    planner = planner or TrajectoryPlanner()
    sandbox = KinematicsSandbox(model, geometry)
    object_boxes = env.update(data)  # block doesn't move during planning -- one snapshot suffices
    table_height = env.table_height

    candidates = solve_ik_candidates(sandbox, target_pos, target_quat, controller, model, q_start,
                                      n_random_seeds=n_random_seeds, rng=rng)
    if not candidates:
        return None

    # a goal that itself collides (e.g. an IK branch with the elbow through the table) can't
    # have a clear path -- the endpoint is fixed -- so don't spend optimize() on it. If every
    # goal collides, keep them all so the least-bad path still wins.
    clear_goals = [q for q in candidates
                   if planner._deepest_intrusion(planner._collision_rows(q, sandbox, object_boxes, table_height))
                   <= planner.feasibility_eps]
    candidates = clear_goals or candidates

    if parallel and len(candidates) > 1:
        n_workers = n_workers or min(os.cpu_count() or 1, len(candidates))
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

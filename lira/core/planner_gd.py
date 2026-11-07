"""
Gradient-descent path optimizer (CHOMP-style), one of the two TrajectoryPlanners (planner.py); the other
is planner_sqp.py. Pick between them with make_planner (common.py) or a sim's --planner flag.

Collision is a soft cost here, a squared hinge on how far each hazard is inside its margin, so a path can
end slightly inside one -- path_cost's feasibility_penalty then ranks it below any clear path.
"""
import numpy as np

from core.planner import TrajectoryPlanner


class GDTrajectoryPlanner(TrajectoryPlanner):
    """Gradient descent against smoothness + collision (self/table/object) + a manipulability hinge,
    one waypoint at a time, with a fixed step size."""

    def __init__(self, iters=60, tol=1e-4, lr=0.02, **kwargs):
        super().__init__(**kwargs)
        self.iters = iters  # max passes over the path
        self.tol = tol      # rad: stop early once no joint of any waypoint moves more than this in a pass
        self.lr = lr

    def optimize(self, q_start, q_goal, sandbox, object_boxes, table_height):
        """Gradient-descends the interior waypoints of a straight-line seed (endpoints fixed), for up
        to `iters` passes -- fewer once a pass moves no joint more than `tol` (the forces have
        balanced; more passes won't change the path). Its PathCost's iters is how many it took."""
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

        cost = self.path_cost(path, sandbox, object_boxes, table_height)
        cost.iters = n_iters
        return path, cost

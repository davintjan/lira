"""Setup both sims share: the seed, the planner, and the startup report."""
import numpy as np

from core.planner import TrajectoryPlanner
from core.planner_sqp import SQPTrajectoryPlanner
from mj.robots import DEFAULT_ROBOT, ROBOTS

PLANNERS = {"gd": TrajectoryPlanner, "sqp": SQPTrajectoryPlanner}  # gradient descent / SQP with collision as a constraint


def add_common_args(parser):
    parser.add_argument("--robot", choices=sorted(ROBOTS), default=DEFAULT_ROBOT)
    parser.add_argument("--planner", choices=sorted(PLANNERS), default="sqp")
    parser.add_argument("--seed", type=int, default=None,
                        help="replay a run: the seed it printed at startup (default: a fresh one)")
    parser.add_argument("--record", metavar="PATH", default=None,
                        help="also record video: PATH=out.mp4 writes out_front.mp4 and out_top.mp4 "
                             "(world.xml's cameras), finished when the viewer closes")


def resolve_seed(seed):
    """seed, or a fresh random one if None."""
    return seed if seed is not None else int(np.random.default_rng().integers(2 ** 31))


def make_planner(kind, cbf_margin):
    """The planner `kind` names, planning with cbf_margin plus a buffer. The planner accepts a path up
    to its feasibility_eps (2 mm) inside its margin, but the CBF enforces its margin exactly -- so
    planned with cbf_margin itself, a path or goal ending inside that sliver is a spot the CBF never
    lets the arm reach, and it stalls just short of the goal."""
    planner_cls = PLANNERS[kind]
    return planner_cls(margin=cbf_margin + 2 * planner_cls().feasibility_eps)


def print_plan(plan, planner, kind):
    """Every candidate's cost (gradient descent), then the winner's cost breakdown."""
    for k, cand in enumerate(plan["all_candidates"]):
        if "iters" in cand["breakdown"]:  # gradient descent reports its passes; SQP doesn't
            print(f"  candidate {k}: {cand['breakdown']['iters']:2d}/{planner.iters} GD passes, "
                  f"total cost {cand['breakdown']['total']:.4f}{'  <- best' if cand is plan['best'] else ''}")
    cost = plan["best"]["breakdown"]
    print(f"Path found ({kind}): goal_manip={plan['best']['goal_manip']:.4f}")
    for term in ("smooth", "collision", "singularity"):
        print(f"  {term:<19} {cost[term]:10.4f}")
    print(f"  feasibility_penalty {cost['feasibility_penalty']:10.4f}"
          f"   (intrusion_depth {cost['intrusion_depth'] * 100:.2f} cm)")
    print(f"  total               {cost['total']:10.4f}")


def print_field(field):
    eigvals = np.linalg.eigvalsh(field.P)
    fit = "path decreases V everywhere" if field.margin > 0 else "no quadratic V fits -- filter will cut corners"
    print(f"Lyapunov fit: margin {field.margin:.3f} ({fit}), P eigenvalues {eigvals.min():.2f}..{eigvals.max():.2f}")

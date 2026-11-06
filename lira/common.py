"""Setup every sim shares: the CLI flags, the seed, the planner, the startup report and the viewer loop."""
import contextlib
import time

import mujoco
import mujoco.viewer
import numpy as np

from core.planner_gd import GDTrajectoryPlanner
from core.planner_sqp import SQPTrajectoryPlanner
from mj.recorder import Recorder

PLANNERS = {"gd": GDTrajectoryPlanner, "sqp": SQPTrajectoryPlanner}  # gradient descent / SQP with collision as a constraint


def add_common_args(parser):
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
    print(f"Path field: {field.seg_len.sum():.2f} rad of path, energy V = path left + {field.k_tube:g}/2 * "
          f"(distance to path)^2, must fall >= {field.min_progress:g} rad/s far from the goal")


def run_viewer(model, data, tick, draw_overlays, record=None):
    """The loop every sim runs until the viewer window closes, paced to real time: tick() -- one physics
    step, with whatever control and monitoring goes with it -- then the overlays and, with record (a path,
    see Recorder), a video frame. draw_overlays(scene) adds the sim's extra geoms to a scene."""
    dt = model.opt.timestep
    recording = Recorder(model, record) if record else contextlib.nullcontext()
    with mujoco.viewer.launch_passive(model, data) as viewer, recording as recorder:
        while viewer.is_running():
            t_wall = time.perf_counter()
            tick()
            viewer.user_scn.ngeom = 0
            draw_overlays(viewer.user_scn)
            if recorder is not None:
                recorder.capture(data, draw_overlays)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - t_wall)))

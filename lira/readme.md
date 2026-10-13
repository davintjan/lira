# LiRA — Lightweight Reactive Arm control

Plan once offline, then react live: a collision-free joint-space path (gradient descent or SQP)
becomes a stable velocity field, and a CBF-QP filters it every tick against self-collision, the
table, objects and moving obstacles.

```
core/      the robot-agnostic math: numpy, osqp, scipy
  planner.py       trajectory planner (gradient descent) + IK candidates + plan_trajectory
  planner_sqp.py   the same planner as trust-region SQP
  path_field.py    planned path -> stable velocity field (Lyapunov P fitted to the path)
  ds.py            reactive DS (position, orientation) and damped-least-squares IK
  cbf.py           the CBF-QP safety filter
  geometry.py      capsule/box distances, CBF clearance rows
mj/        MuJoCo glue
  robots.py        RobotConfig + ROBOTS: everything that differs between arms
  model.py         load_model: scene xml + pusher rod + velocity servos (MjSpec edits)
  arm.py           Arm: model, live data and collision geometry for one robot
  safety.py        SafetyFilter (clearance rows at the live state -> cbf_qp), ObstacleMonitor
  tasks.py         nominal controllers: flange pose DS, probe DS
  scene.py         robot/environment capsules from the model, moving obstacle
  mj_interface.py  poses, Jacobians, differential IK
  utils.py         viewer overlays (draw_capsules: what the planner and CBF see)
pusht/     Push-T glue
  frame.py         table <-> Push-T pixels, observation, coverage, pre-push spot
  policy.py        load the diffusion policy; ChunkedPolicy runs it in a background thread
scenes/    the MuJoCo world
  world.xml        table, T, goal, obstacle -- shared by every robot
  scene_<robot>.xml  one robot (vendored from mujoco_menagerie) + world.xml
common.py          setup both sims share: CLI flags, seed, planner, plan report
reach_sim.py       plan to the T, follow the field, dodge the obstacle
pusht_sim.py       the arm plays the Push-T diffusion policy
```

(core/planner.py still uses MuJoCo for kinematics through mj/. The joint count comes from the model,
so core/ works for any arm MuJoCo can load.)

Adding a robot: put its menagerie folder in `scenes/generate_vendor_links.py`'s `ROBOTS` (and in the
submodule's sparse checkout), write `scenes/scene_<robot>.xml` like the others, and add an entry to
`mj/robots.py`'s `ROBOTS`.

## Run

From this directory (the scripts import `core.*` / `mj.*` relative to it):

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # first time only
.venv/bin/python scenes/generate_vendor_links.py                     # first time, and after updating mujoco_menagerie
.venv/bin/python reach_sim.py                                        # --robot ur5e, --seed N, --planner gd; see --help
.venv/bin/python pusht_sim.py                                        # --no-obstacle, --checkpoint PATH
```

Just the scene in the viewer (from inside `scenes/`: the vendored meshes only resolve when the scene
is opened by a relative path from there):

```
cd scenes && PYGLFW_LIBRARY_VARIANT=x11 ../.venv/bin/python -m mujoco.viewer --mjcf=scene_ur5e.xml
```

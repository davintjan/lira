# LiRA — Lightweight Reactive Arm control

Plan once offline, then react live: a collision-free joint-space path (gradient descent or SQP)
becomes a stable velocity field, and a CBF-QP filters it every tick against self-collision, the
table, objects and moving obstacles.

```
core/      the robot-agnostic math: numpy, osqp, scipy
  planner.py       trajectory planner (gradient descent) + IK candidates + plan_trajectory
  planner_sqp.py   the same planner as trust-region SQP
  path_field.py    planned path -> stable velocity field (Lyapunov P fitted to the path)
  control.py       DS controllers, CBF-QP safety filter, disturbances
  geometry.py      capsule/box distances, CBF clearance rows
mj/        MuJoCo glue
  scene.py         robot/environment capsules from the model, moving obstacle
  mj_interface.py  poses, Jacobians, differential IK
  utils.py         viewer overlays (draw_capsules: what the planner and CBF see)
scenes/    the MuJoCo world
  world.xml        table, T, goal, obstacle -- shared by every robot
  scene_<robot>.xml  one robot (vendored from mujoco_menagerie) + world.xml
arm_sim.py         plan to the T, follow the field, dodge the obstacle -- any arm in ROBOTS
                   (ROBOT = "ur5e" or "iiwa14"; per-robot settings at the top)
ur5_pusht_sim.py   arm_sim's robot plays the Push-T diffusion policy (parked)
```

(core/planner.py still uses MuJoCo for kinematics through mj/. The joint count comes from the model,
so core/ works for any arm MuJoCo can load.)

Adding a robot: put its menagerie folder in `scenes/generate_vendor_links.py`'s `ROBOTS` (and in the
submodule's sparse checkout), write `scenes/scene_<robot>.xml` like the others, and add an entry to
`arm_sim.ROBOTS`.

## Run

From this directory (the scripts import `core.*` / `mj.*` relative to it):

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # first time only
.venv/bin/python scenes/generate_vendor_links.py                     # first time, and after updating mujoco_menagerie
.venv/bin/python arm_sim.py
```

Just the scene in the viewer: `.venv/bin/python -m mujoco.viewer --mjcf=scenes/scene_ur5e.xml`

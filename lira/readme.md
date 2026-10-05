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
scenes/    the MuJoCo world: scene.xml + the vendored UR5e it includes
ur5_sim.py         UR5e: plan to the T, follow the field, dodge the obstacle
ur5_pusht_sim.py   UR5e plays the Push-T diffusion policy (parked)
```

(core/planner.py still uses MuJoCo for kinematics through mj/ — not robot-agnostic yet.)

## Run

From this directory (the scripts import `core.*` / `mj.*` relative to it):

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # first time only
.venv/bin/python scenes/generate_ur5e_vendor_links.py                # first time, and after updating mujoco_menagerie
.venv/bin/python ur5_sim.py
```

Just the scene in the viewer: `.venv/bin/python -m mujoco.viewer --mjcf=scenes/scene.xml`

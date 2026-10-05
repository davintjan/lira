"""
Simulation that implement planner (Gradient Descent and sequential quadratic programming) 
and collision avoidance
"""

"""Entry point: wires geometry (scene.py), control (control.py), and planning (planner.py) into
the sim loop. Plans once before the loop starts, then drives the arm every tick."""
import os
import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np

from core.control import RobotController, disturbances
from core.geometry import clearanceRows, movingObstacleRows
from mj.scene import RobotGeometry, EnvironmentGeometry, MovingObstacle
from core.planner import TrajectoryPlanner, plan_trajectory
from core.planner_sqp import SQPTrajectoryPlanner
from core.path_field import PathVelocityField
from mj.mj_interface import pose_pub, obj_pose_pub, site_jacobian, point_jacobian, diff_ik

SCENE_DIR = pathlib.Path(__file__).resolve().parent / "scenes"  # scene.xml and the vendored UR5e it includes
SEED = None  # IK seed: None = a fresh random one each run (printed at startup); set it to a printed value to replay that run
PLANNER = "gd"  # "gd": gradient descent (planner.py), "sqp": SQP with collision as a constraint (planner_sqp.py)
K_P = 1.0
K_O = 1.0
MAX_VEL = 0.8       # m/s cap on commanded ee speed
IK_DAMPING = 1e-2   # damped least squares, keeps J^+ finite near singularities
CBF_ALPHA = 5.0     # self-collision CBF class-K gain: higher = closer/faster approach allowed before braking
CBF_MARGIN = 0.03   # m: extra clearance the CBF holds beyond the capsules' actual surfaces
CBF_TABLE_MARGIN = 0.005  # m: much tighter -- several links rest just mm above the table by design
PUSHER_RADIUS = 0.015  # m: matches Push-T's pusher circle (15 px at 1 px = 1 mm)
PUSHER_LENGTH = 0.15   # m, flange to tip: keeps the wrist well above the T while the tip pushes it
JOINT_VEL_MAX = 10.0  # rad/s: box constraint on the CBF-QP's joint velocity solution
PATH_K_TANGENT = 8.0     # rad/s: nominal joint-space speed along the planned path
PATH_K_CORRECTIVE = 2.0  # rad/s per rad: pull-back gain toward the path when off it
DISTURBANCE_MAGNITUDE = 0.0       # N: push applied at the end effector (0 disables it)
DISTURBANCE_DIRECTION = [1, 1, 1]   # world frame, normalized inside disturbances()
DISTURBANCE_START = 0            # s of sim time: after the arm has reached the target
DISTURBANCE_DURATION = 2          # s
OBSTACLE_CENTER = [0.3545, 0.046, 0.3389]          # m, world frame: [x, y, z] = sweep around this point; None = around the middle of the planned path
OBSTACLE_AXIS = [0, 1, 0]       # world frame: the ellipsoid sweeps left and right along this
OBSTACLE_AMPLITUDE = 0.25       # m: how far it goes either side of its center
OBSTACLE_SPEED = 0.3            # m/s, constant
_OBJ_OFFSET = [0, 0, 0.1]  # m: where the probe (pusher tip) hovers relative to the T-block
# IK and the fallback DS steer the flange (attachment_site); with the rod pointing straight down,
# the flange sits this far above the probe
_FLANGE_ABOVE_PROBE = [0, 0, PUSHER_LENGTH - PUSHER_RADIUS]
_QUAT_X180 = np.array([0.0, 0.0, 1.0, 0.0])  # 180 deg about local x, [w, x, y, z]


def load_model(xml_path="scene.xml"):
    """scene.xml with the arm switched to velocity control: each of ur5e.xml's position servos
    becomes a velocity servo, force = kv * (ctrl - qvel), keeping its kv -- so ctrl is simply the
    joint velocity control_step wants, no position target to wind up. A velocity servo has no
    position spring to hold the arm up, so gravity is compensated through the actuators
    (actgravcomp: counted against the motors' force limits, as on hardware). Edited via MjSpec so
    the vendored ur5e.xml stays untouched.

    Also mounts a pusher rod on the flange (attachment_site), pointing out along the tool axis:
    a collision capsule (group 3, so RobotGeometry, the planner and the CBF guard it like any link)
    plus a visible cylinder (group-3 geoms aren't drawn by default), and a "probe" site at the
    centre of its rounded tip -- the point that plays Push-T's pusher. ur5e.xml defines the wrist,
    so the rod can only be added here, not in scene.xml."""
    spec = mujoco.MjSpec.from_file(xml_path)
    flange = spec.site("attachment_site")
    rod = flange.parent.add_body(name="pusher", pos=flange.pos, quat=flange.quat)
    rod.gravcomp = 1.0  # rides on the wrist: held up like the links are
    rod.add_geom(name="pusher", type=mujoco.mjtGeom.mjGEOM_CAPSULE, group=3, contype=1, conaffinity=1,
                 size=[PUSHER_RADIUS, PUSHER_LENGTH / 2 - PUSHER_RADIUS, 0], pos=[0, 0, PUSHER_LENGTH / 2],
                 mass=0.1)
    rod.add_geom(name="pusher_visual", type=mujoco.mjtGeom.mjGEOM_CYLINDER, group=2, contype=0, conaffinity=0,
                 size=[PUSHER_RADIUS, PUSHER_LENGTH / 2 - PUSHER_RADIUS, 0], pos=[0, 0, PUSHER_LENGTH / 2 - PUSHER_RADIUS / 2],
                 density=0, rgba=[0.15, 0.15, 0.15, 1])
    rod.add_geom(name="pusher_tip_visual", type=mujoco.mjtGeom.mjGEOM_SPHERE, group=2, contype=0, conaffinity=0,
                 size=[PUSHER_RADIUS, 0, 0], pos=[0, 0, PUSHER_LENGTH - PUSHER_RADIUS], density=0,
                 rgba=[0.9, 0.9, 0.9, 1])
    rod.add_site(name="probe", pos=[0, 0, PUSHER_LENGTH - PUSHER_RADIUS])

    kv = -spec.compile().actuator_biasprm[:6, 2]  # position servo bias is [0, -kp, -kv]
    arm_joints = []
    for act, gain in zip(spec.actuators, kv):
        act.set_to_velocity(kv=gain)
        act.ctrlrange = [-JOINT_VEL_MAX, JOINT_VEL_MAX]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        arm_joints.append(spec.joint(act.target))
    for joint in arm_joints:
        joint.actgravcomp = True
        joint.parent.gravcomp = 1.0  # the body this joint moves
    return spec.compile()


def obstacle_rows(model, data, geometry, obstacle):
    """CBF rows keeping every robot capsule CBF_MARGIN clear of the moving obstacle (MovingObstacle.update).
    The obstacle's own motion changes the clearance too: grad . u + h_dot_obstacle >= -alpha * h,
    which is collisionQP's grad . u >= -alpha * h' with h' = h + h_dot_obstacle / alpha."""
    return [(h + h_dot_obstacle / CBF_ALPHA, grad) for h, grad, h_dot_obstacle in movingObstacleRows(
        geometry.update_all(data), obstacle, lambda body_id, point: point_jacobian(model, data, body_id, point),
        margin=CBF_MARGIN)]


def control_step(model, data, controller, geometry, env, field=None, obstacle=None):
    """field=None: straight-line DS (linearPosDS + linearOrientationDS) through diff_ik.
    field=<PathVelocityField>: the planned path's joint-space field instead, bypassing
    the DS/diff_ik entirely. collisionQP filters u_des the same way either case, and its joint
    velocity goes straight to the velocity servos (see load_model). Everything is evaluated at
    the measured state, with no stored command -- time-invariant, nothing to wind up.
    obstacle: the moving obstacle's capsule and velocity this tick (MovingObstacle.update), or None."""
    ee_pos, ee_quat = pose_pub(data)
    obj_pos, obj_quat = obj_pose_pub(data, "t_block")
    obj_pos = data.body("t_block").xipos.copy()  # hover over the T's center of mass -- its body origin is the bar's edge (scene.xml)
    setpoint = obj_pos + _OBJ_OFFSET + _FLANGE_ABOVE_PROBE
    setpoint_quat = np.zeros(4)
    mujoco.mju_mulQuat(setpoint_quat, obj_quat, _QUAT_X180)  # flip target 180 deg about its local x

    if field is not None:
        u_des = field(data.qpos[:6])
    else:
        xdot_des = controller.linearPosDS(ee_pos, setpoint, K_P)
        w_des = controller.linearOrientationDS(ee_quat, setpoint_quat, K_O)
        speed = np.linalg.norm(xdot_des)
        if speed > MAX_VEL:
            xdot_des *= MAX_VEL / speed
        u_des = diff_ik(model, data, xdot_des, w_des, damping=IK_DAMPING)  # nominal joint velocity, no self-collision awareness

    rows = clearanceRows(
        ee_pos, site_jacobian(model, data)[:3], geometry.update(data), geometry.link_pairs_world(data),
        geometry.update_all(data), env.update(data), env.table_height,
        lambda body_id, point: point_jacobian(model, data, body_id, point),
        margin=CBF_MARGIN, table_margin=CBF_TABLE_MARGIN, activation=np.inf,
    )
    if obstacle is not None:
        rows += obstacle_rows(model, data, geometry, obstacle)
    qdot = controller.collisionQP(u_des, rows, alpha=CBF_ALPHA, u_max=JOINT_VEL_MAX)

    data.ctrl[:6] = qdot

def draw_capsules(viewer, world_capsules, rgba=(1.0, 0.15, 0.15, 0.35)):
    """Overlay keep-out capsules on the passive viewer -- what collisionQP is guarding against.
    Resets user_scn each call rather than growing it unbounded."""
    scn = viewer.user_scn
    scn.ngeom = 0
    for cap in world_capsules:
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom], type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.zeros(3), pos=np.zeros(3), mat=np.eye(3).flatten(),
            rgba=np.array(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
                              cap["radius"], cap["a"], cap["b"])
        scn.ngeom += 1

def main():
    os.chdir(SCENE_DIR)
    model = load_model("scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep  # 0.002 s -> 500 Hz, one control update per sim step
    ee_pos, ee_quat = pose_pub(data)
    controller = RobotController(data.qpos[:6].copy(), data.qvel[:6].copy(), ee_pos, np.zeros(3))
    geometry = RobotGeometry(model)
    env = EnvironmentGeometry(model)

    # Plan once before the loop starts (and before the viewer opens) --
    # planning takes ~1s, too slow for the 500Hz loop.
    q_start = data.qpos[:6].copy()
    obj_pos, obj_quat = obj_pose_pub(data, "t_block")
    obj_pos = data.body("t_block").xipos.copy()  # hover over the T's center of mass -- its body origin is the bar's edge (scene.xml)
    target_pos = obj_pos + _OBJ_OFFSET + _FLANGE_ABOVE_PROBE
    target_quat = np.zeros(4)
    mujoco.mju_mulQuat(target_quat, obj_quat, _QUAT_X180)

    # the IK seeds are the only randomness in a run (planning, physics and the obstacle are
    # deterministic given the path), so this one number recreates the whole run
    seed = SEED if SEED is not None else int(np.random.default_rng().integers(2 ** 31))
    print(f"Seed: {seed}   (set SEED = {seed} to replay this run)")

    print("Planning coarse-reach path...")
    planner = SQPTrajectoryPlanner() if PLANNER == "sqp" else TrajectoryPlanner()
    t0 = time.perf_counter()
    plan = plan_trajectory(model, data, controller, geometry, env, q_start, target_pos, target_quat,
                           planner=planner, parallel=True, rng=np.random.default_rng(seed))
    print(f"Planning took {time.perf_counter() - t0:.3f}s")
    if plan is None:
        print("WARNING: no IK candidate converged -- falling back to the straight-line DS")
        field = None
    else:
        cost = plan["best"]["breakdown"]
        print(f"Path found ({PLANNER}): goal_manip={plan['best']['goal_manip']:.4f}")
        print(f"  smooth              {cost['smooth']:10.4f}")
        print(f"  collision           {cost['collision']:10.4f}")
        print(f"  singularity         {cost['singularity']:10.4f}")
        print(f"  feasibility_penalty {cost['feasibility_penalty']:10.4f}"
              f"   (intrusion_depth {cost['intrusion_depth'] * 100:.2f} cm)")
        print(f"  total               {cost['total']:10.4f}")
        field = PathVelocityField(plan["best"]["path"], k_tangent=PATH_K_TANGENT, k_corrective=PATH_K_CORRECTIVE)
        eigvals = np.linalg.eigvalsh(field.P)
        print(f"Lyapunov fit: margin {field.margin:.3f} ({'path decreases V everywhere' if field.margin > 0 else 'no quadratic V fits -- filter will cut corners'}), "
              f"P eigenvalues {eigvals.min():.2f}..{eigvals.max():.2f}")

    # the obstacle sweeps around OBSTACLE_CENTER, or by default through the end effector's position
    # at the middle of the planned path (scene.xml's position if there's no path)
    if OBSTACLE_CENTER is not None:
        obstacle_center, where = np.asarray(OBSTACLE_CENTER, dtype=float), "set by OBSTACLE_CENTER"
    elif field is not None:
        mid = mujoco.MjData(model)
        mid.qpos[:6] = plan["best"]["path"][len(plan["best"]["path"]) // 2]
        mujoco.mj_kinematics(model, mid)
        obstacle_center, where = mid.site("attachment_site").xpos.copy(), "middle of the planned path"
    else:
        obstacle_center, where = model.body("obstacle").pos.copy(), "scene.xml default, no path"
    obstacle = MovingObstacle(model, obstacle_center, axis=OBSTACLE_AXIS,
                              amplitude=OBSTACLE_AMPLITUDE, speed=OBSTACLE_SPEED)
    print(f"Obstacle: starts at {np.round(obstacle.center, 4).tolist()} ({where}), "
          f"sweeps +-{OBSTACLE_AMPLITUDE} m along {OBSTACLE_AXIS} at {OBSTACLE_SPEED} m/s, heading + first")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()
            control_step(model, data, controller, geometry, env, field=field, obstacle=obstacle.update(data))
            disturbances(model, data, DISTURBANCE_MAGNITUDE, DISTURBANCE_DIRECTION,
                         DISTURBANCE_DURATION, start=DISTURBANCE_START)
            mujoco.mj_step(model, data)
            # print("qvel:", np.round(data.qvel[:6], 3))
            # draw_capsules(viewer, geometry.update(data))
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))  # pace to real time


if __name__ == "__main__":
    main()

"""
UR5 plays Push-T: the diffusion policy trained in the 2-D Push-T simulator (../diffusion_policy)
drives the UR5's pusher rod on the MuJoCo table.

Phases (one state machine in main):
  reach    SQP plan + PathVelocityField, as in arm_sim: the probe goes to HOVER_HEIGHT above a
           pre-push spot next to the T -- clear of it, so coming down can't land on it
  descend  the probe goes straight down to PUSH_HEIGHT
  push     every 0.1 s (Push-T's 10 Hz): T keypoints + probe position -> Push-T pixels -> policy ->
           8 target positions, used one per 0.1 s; every tick a Cartesian DS drives the probe toward
           the current target, through the CBF (the rod may touch the T, nothing else may). The policy
           runs in a background thread, so the sim keeps going while it thinks -- like a real robot,
           the chunk lands late and its already-past actions are dropped
  done     the T covers SUCCESS_COVERAGE of the goal (Push-T's success rule) or PUSH_TIME_LIMIT
           passes; the probe holds still

The policy only ever sees Push-T pixels: PushTFrame maps the table to the Push-T workspace at
1 px = 1 mm, axes aligned, with Push-T's fixed goal pose on world.xml's goal body.

Run from this directory:  .venv/bin/python ur5_pusht_sim.py
"""
import os
from concurrent.futures import ThreadPoolExecutor
import sys
import time

import dill
import mujoco
import mujoco.viewer
import numpy as np
import torch
from omegaconf import OmegaConf

import arm_sim as U  # the robot is arm_sim.ROBOT, and its settings
from core.control import RobotController
from core.geometry import clearanceRows, capsuleCapsuleDistanceBatch
from mj.mj_interface import pose_pub, site_jacobian, point_jacobian
from core.path_field import PathVelocityField
from core.planner import plan_trajectory
from mj.scene import EnvironmentGeometry, MovingObstacle

# the policy code lives in the diffusion_policy fork, used in place (see pusht_simple/ for why not pip install)
DIFFUSION_POLICY_DIR = U.SCENE_DIR.parent.parent / "diffusion_policy"
sys.path.insert(0, str(DIFFUSION_POLICY_DIR))
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # noqa: E402
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D  # noqa: E402
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy  # noqa: E402

CHECKPOINT = DIFFUSION_POLICY_DIR / "data/outputs/reference/epoch=0550-test_mean_score=0.969.ckpt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = None              # None = a fresh random one each run (printed); set it to a printed value to replay that run
PLANNER = "sqp"          # "gd" or "sqp", as in arm_sim
HOVER_HEIGHT = 0.10      # m: how far above PUSH_HEIGHT the reach phase leaves the probe
PUSH_HEIGHT = 0.025      # m: probe (rod tip center) height while pushing -- the T is 4 cm tall, the tip clears the table by 1 cm
PREPUSH_DISTANCE = 0.13  # m from the T's center of mass: clear of the T (it reaches ~0.076 m), its CBF margin and the rod
PREPUSH_MARGIN_PX = 20   # px: the pre-push spot stays this far inside the Push-T workspace
PROBE_GAIN = 10.0        # 1/s: Push-T's pusher PD (k_p 100, k_v 20) is critically damped with time constant 0.1 s
PROBE_MAX_SPEED = 0.3    # m/s
POLICY_DT = 0.1          # s: Push-T's control period
PREFETCH_ACTIONS = 2     # start inferring the next chunk when this few actions are left, so it's ready in time
SUCCESS_COVERAGE = 0.95  # Push-T's success threshold: fraction of the goal T covered by the block
PUSH_TIME_LIMIT = 60.0   # s of sim time in the push phase
REACH_TOL = 0.02         # rad: the reach phase is done once the arm is this close to the planned path's end...
REACH_PROBE_TOL = 0.03   # m: ...or the probe this close to the hover point (the descend DS takes care of the rest)
TASK_METRIC_REG = 0.01   # how much the CBF still minds joint motion the probe task doesn't care about
OBSTACLE = True                # sweep world.xml's ellipsoid through the arm, as in arm_sim; False = park it out of the way
OBSTACLE_CENTER = [0.0545, -0.046, 0.3389]         # m, world frame: sweep around this point; None = OBSTACLE_HEIGHT above the T's start
OBSTACLE_HEIGHT = 0.30         # m: above the T, through the wrist/forearm while pushing (the rod tip is at 2.5 cm, flange ~17.5 cm)
OBSTACLE_AXIS = [0, 1, 0]      # world frame: sweeps left and right along this
OBSTACLE_AMPLITUDE = 0.25      # m either side of its center
OBSTACLE_SPEED = 0.3           # m/s, constant
OBSTACLE_DWELL = 5.0           # s: starts at -OBSTACLE_AMPLITUDE (along OBSTACLE_AXIS), crosses, parks this long at
                               # each end before crossing back; None = sweep back and forth without stopping

# Push-T's 9 block keypoints in the T's local frame (px), as PushTKeypointsEnv.genenerate_keypoint_manager_params()
# makes them: farthest-point samples of the rendered T with seed 0, so always these same values. Generated in
# ../diffusion_policy's venv (needs pygame/pymunk); the observation they produce matched that env's exactly.
LOCAL_KEYPOINTS = np.array([
    [9.269787, 90.040974], [-60.983472, -1.18673], [60.912756, -1.393364],
    [-0.270503, 25.543625], [-14.064928, 122.815854], [39.002739, 32.857404],
    [-39.966414, 32.729655], [-16.824344, 62.863179], [23.541461, -1.700288],
])
# the T's two rectangles in its local frame (px), counter-clockwise -- pusht_env.add_tee with scale 30
T_RECTS_LOCAL = [
    np.array([[-60.0, 0.0], [60.0, 0.0], [60.0, 30.0], [-60.0, 30.0]]),     # bar
    np.array([[-15.0, 30.0], [15.0, 30.0], [15.0, 120.0], [-15.0, 120.0]]),  # stem
]
T_AREA_PX = 120 * 30 + 30 * 90


def load_policy(checkpoint, device):
    """The Push-T lowdim diffusion policy from either checkpoint format:
      - the original repo's Hydra workspace checkpoint (train.py): built from its own saved config
        (e.g. the reference one feeds observations by inpainting, obs_as_global_cond=False)
      - train_pusht_simple.py's plain checkpoint: no config inside, so the architecture is rebuilt
        exactly as eval_pusht_simple.py does (observations as global conditioning)
    EMA weights either way; they include the observation/action normalizer."""
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    if "cfg" in payload:
        cfg = OmegaConf.to_container(payload["cfg"].policy, resolve=True)
        strip = lambda d: {k: v for k, v in d.items() if k != "_target_"}  # noqa: E731
        policy = DiffusionUnetLowdimPolicy(
            model=ConditionalUnet1D(**strip(cfg["model"])),
            noise_scheduler=DDPMScheduler(**strip(cfg["noise_scheduler"])),
            **{k: cfg[k] for k in ("horizon", "obs_dim", "action_dim", "n_action_steps", "n_obs_steps",
                                   "num_inference_steps", "obs_as_local_cond", "obs_as_global_cond",
                                   "pred_action_steps_only", "oa_step_convention")},
        )
        state = payload["state_dicts"]["ema_model"]
    else:
        obs_dim, action_dim, n_obs_steps = 20, 2, 2
        policy = DiffusionUnetLowdimPolicy(
            model=ConditionalUnet1D(input_dim=action_dim, local_cond_dim=None, global_cond_dim=obs_dim * n_obs_steps,
                                    diffusion_step_embed_dim=256, down_dims=[256, 512, 1024], kernel_size=5,
                                    n_groups=8, cond_predict_scale=True),
            noise_scheduler=DDPMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                                          beta_schedule="squaredcos_cap_v2", variance_type="fixed_small",
                                          clip_sample=True, prediction_type="epsilon"),
            horizon=16, obs_dim=obs_dim, action_dim=action_dim, n_action_steps=8, n_obs_steps=n_obs_steps,
            num_inference_steps=100, obs_as_global_cond=True, oa_step_convention=True,
        )
        state = payload["ema_state_dict"]
    policy.load_state_dict(state)
    return policy.to(device).eval()


def _rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _clip_polygon(subject, clip):
    """Sutherland-Hodgman: the part of convex polygon `subject` inside convex polygon `clip`
    (both counter-clockwise)."""
    out = list(subject)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        if not out:
            break
        edge = b - a
        inside = lambda p: edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= 0  # noqa: E731
        poly, out = out, []
        for p, q in zip(poly, poly[1:] + poly[:1]):
            if inside(q):
                if not inside(p):
                    out.append(_line_intersection(p, q, a, b))
                out.append(q)
            elif inside(p):
                out.append(_line_intersection(p, q, a, b))
    return out


def _line_intersection(p, q, a, b):
    d1, d2 = q - p, b - a
    t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / (d1[0] * d2[1] - d1[1] * d2[0])
    return p + t * d1


def _area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = np.array(poly).T
    return 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))


class PushTFrame:
    """Table <-> Push-T pixels. 1 px = 1 mm and the axes are aligned (pixel x = table x, pixel y =
    table y, angles counter-clockwise seen from above), so a pose maps by a shift and a scale. The
    shift puts Push-T's fixed goal pose (256, 256 px) on world.xml's goal body, which must sit at
    Push-T's goal angle, 45 deg."""

    SCALE = 0.001  # m per px
    GOAL_PX = np.array([256.0, 256.0])
    GOAL_ANGLE = np.pi / 4
    SIZE_PX = 512

    def __init__(self, model):
        goal = model.body("goal")
        self.origin = goal.pos[:2] - self.GOAL_PX * self.SCALE  # table xy of pixel (0, 0)
        w, x, y, z = goal.quat
        goal_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        assert abs(goal_yaw - self.GOAL_ANGLE) < 1e-6, "world.xml's goal body must be at Push-T's goal angle (45 deg)"

    def to_px(self, xy):
        return (np.asarray(xy)[..., :2] - self.origin) / self.SCALE

    def to_table(self, px):
        return self.origin + np.asarray(px) * self.SCALE

    def block_pose_px(self, data):
        """The T's Push-T pose: (position px, angle rad) of its body origin -- world.xml builds the
        T in Push-T's local frame, so no offset."""
        body = data.body("t_block")
        rot = body.xmat.reshape(3, 3)
        return self.to_px(body.xpos), np.arctan2(rot[1, 0], rot[0, 0])

    def observation(self, data):
        """Push-T's 20-number observation: the 9 block keypoints (x, y) then the pusher (x, y), in px."""
        pos, angle = self.block_pose_px(data)
        keypoints = LOCAL_KEYPOINTS @ _rotation(angle).T + pos
        return np.concatenate([keypoints.ravel(), self.to_px(data.site("probe").xpos)]).astype(np.float32)

    def coverage(self, data):
        """Push-T's success measure: the fraction of the goal T's area the block covers."""
        pos, angle = self.block_pose_px(data)
        block = [r @ _rotation(angle).T + pos for r in T_RECTS_LOCAL]
        goal = [r @ _rotation(self.GOAL_ANGLE).T + self.GOAL_PX for r in T_RECTS_LOCAL]
        return sum(_area(_clip_polygon(b, g)) for b in block for g in goal) / T_AREA_PX

    def inside(self, px, margin=0.0):
        return bool(np.all(px >= margin) and np.all(px <= self.SIZE_PX - margin))


def prepush_spot(frame, data):
    """Table xy to start pushing from: PREPUSH_DISTANCE from the T's center of mass -- clear of it --
    inside the Push-T workspace (the policy never saw a pusher outside it), and as far behind the T
    (away from the goal) as that allows."""
    com = data.body("t_block").xipos[:2]
    goal = frame.to_table(frame.GOAL_PX)
    away = com - goal
    away = away / np.linalg.norm(away) if np.linalg.norm(away) > 1e-9 else np.array([1.0, 0.0])
    best = None
    for theta in np.radians(np.arange(0, 360, 5)):
        spot = com + PREPUSH_DISTANCE * np.array([np.cos(theta), np.sin(theta)])
        if frame.inside(frame.to_px(spot), PREPUSH_MARGIN_PX) and (best is None or (spot - com) @ away > (best - com) @ away):
            best = spot
    return best if best is not None else com + PREPUSH_DISTANCE * away


def probe_step(model, data, controller, geometry, env, target, touch_block, obstacle=None, caps=None):
    """One tick of the Cartesian DS on the probe: xdot = PROBE_GAIN * (target - probe), capped at
    PROBE_MAX_SPEED, while the rod is turned toward vertical. The task is 5-D -- probe position plus
    the rod's tilt -- and leaves the spin about the rod free: the rod is round, so spinning changes
    nothing for the push, and it's the arm's spare freedom to stay unfolded near its base (holding
    the full orientation there folds the wrist into the upper arm). Damped least squares turns the
    task into joint velocities; the CBF filters them in the same task metric, so it spends the free
    spin first; they go to the velocity servos. touch_block=True lets the rod touch the T (that's the
    pushing) -- every other link stays guarded from it. obstacle: MovingObstacle.update's capsule, or None;
    the whole arm, rod included, stays clear of it. caps: this tick's RobotGeometry.world(data), or
    None to compute it here."""
    caps = caps or geometry.world(data)
    probe_pos = data.site("probe").xpos.copy()
    xdot = PROBE_GAIN * (np.asarray(target) - probe_pos)
    speed = np.linalg.norm(xdot)
    if speed > PROBE_MAX_SPEED:
        xdot *= PROBE_MAX_SPEED / speed
    axis = data.site("probe").xmat.reshape(3, 3)[:, 2]  # the rod's direction, flange -> tip
    omega = U.K_O * np.cross(axis, [0.0, 0.0, -1.0])     # turns the rod toward pointing down; never about itself
    tilt_basis = np.linalg.svd(axis[None, :])[2][1:]    # two unit vectors perpendicular to the rod
    J = site_jacobian(model, data, "probe")
    J_task = np.vstack([J[:3], tilt_basis @ J[3:]])
    task = np.concatenate([xdot, tilt_basis @ omega])
    u_des = J_task.T @ np.linalg.solve(J_task @ J_task.T + U.IK_DAMPING ** 2 * np.eye(5), task)

    env_capsules = caps["env"]
    if touch_block:
        env_capsules = [dict(cap, guard_objects=False) if cap["name"] == "pusher" else cap for cap in env_capsules]
    rows = clearanceRows(
        pose_pub(data)[0], site_jacobian(model, data)[:3], caps["self"], caps["pairs"],
        env_capsules, env.update(data), env.table_height,
        lambda body_id, point: point_jacobian(model, data, body_id, point),
        margin=U.CBF_MARGIN, table_margin=U.CBF_TABLE_MARGIN, activation=U.CBF_ACTIVATION,
    )
    if obstacle is not None:
        rows += U.obstacle_rows(model, data, caps["env"], obstacle)
    metric = J_task.T @ J_task + TASK_METRIC_REG * np.eye(model.nu)
    data.ctrl[:model.nu] = controller.collisionQP(u_des, rows, alpha=U.CBF_ALPHA, u_max=U.JOINT_VEL_MAX, metric=metric)


def obstacle_clearance(model, data, env_capsules, obstacle):
    """Closest distance (m) from any robot capsule (env_capsules: RobotGeometry.world's "env") to the
    obstacle's capsule, and whether MuJoCo has the obstacle in contact with the robot right now -- the
    proof the CBF is dodging, not just lucky."""
    A = np.array([c["a"] for c in env_capsules])
    clearance = capsuleCapsuleDistanceBatch(
        A, np.array([c["b"] for c in env_capsules]), np.array([c["radius"] for c in env_capsules]),
        np.broadcast_to(obstacle["a"], A.shape), np.broadcast_to(obstacle["b"], A.shape),
        np.full(len(env_capsules), obstacle["radius"]))[0].min()
    obstacle_body = model.body("obstacle").id
    robot_bodies = {c["body_id"] for c in env_capsules}
    touching = any({model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]} & {obstacle_body}
                   and {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]} & robot_bodies
                   for c in data.contact[:data.ncon])
    return clearance, touching


def draw_targets(viewer, frame, targets_px, current):
    """The policy's queued targets as small spheres at push height; the current one larger."""
    scn = viewer.user_scn
    scn.ngeom = 0
    for i, px in enumerate([current] + list(targets_px)):
        if px is None:
            continue
        xyz = np.append(frame.to_table(px), PUSH_HEIGHT)
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=np.array([0.008 if i == 0 else 0.004, 0, 0]), pos=xyz, mat=np.eye(3).flatten(),
                            rgba=np.array([0.1, 0.4, 1.0, 0.9 if i == 0 else 0.5], dtype=np.float32))
        scn.ngeom += 1


def main():
    os.chdir(U.SCENE_DIR)
    seed = SEED if SEED is not None else int(np.random.default_rng().integers(2 ** 31))
    print(f"Seed: {seed}   (set SEED = {seed} to replay: IK seeds and diffusion noise -- inference timing still varies run to run)")
    torch.manual_seed(seed)

    model = U.load_model()
    data = mujoco.MjData(model)
    U.reset_to_start(model, data)
    dt = model.opt.timestep
    frame = PushTFrame(model)
    print(f"Loading policy {CHECKPOINT.name} on {DEVICE}...")
    policy = load_policy(CHECKPOINT, DEVICE)
    controller = RobotController(data.qpos[:model.nu].copy(), data.qvel[:model.nu].copy(), pose_pub(data)[0], np.zeros(3))
    geometry = U.robot_geometry(model)
    env = EnvironmentGeometry(model)

    block_px, block_angle = frame.block_pose_px(data)
    print(f"T start (Push-T px): pos {np.round(block_px, 1).tolist()}, angle {np.degrees(block_angle):.1f} deg; "
          f"goal covered {frame.coverage(data) * 100:.1f}%")

    # reach: plan to hover above the pre-push spot, rod pointing straight down
    spot = prepush_spot(frame, data)
    print(f"Pre-push spot: table {np.round(spot, 3).tolist()} m = {np.round(frame.to_px(spot), 1).tolist()} px")
    flange_target = np.append(spot, PUSH_HEIGHT + HOVER_HEIGHT) + U._FLANGE_ABOVE_PROBE
    planner = U.make_planner(PLANNER)  # plans with a margin buffer over the CBF's -- see make_planner
    t0 = time.perf_counter()
    plan = plan_trajectory(model, data, controller, geometry, env, data.qpos[:model.nu].copy(), flange_target,
                           U._QUAT_X180, planner=planner, rng=np.random.default_rng(seed))
    print(f"Planning took {time.perf_counter() - t0:.3f}s")
    if plan is None:
        print("WARNING: no IK candidate converged -- skipping the reach phase, descending with the Cartesian DS")
        field, path_end, phase = None, None, "descend"
    else:
        path = plan["best"]["path"]
        field = PathVelocityField(path, k_tangent=U.PATH_K_TANGENT, k_corrective=U.PATH_K_CORRECTIVE)
        path_end, phase = path[-1], "reach"

    if OBSTACLE:
        if OBSTACLE_CENTER is not None:
            obstacle_center, where = np.asarray(OBSTACLE_CENTER, dtype=float), "set by OBSTACLE_CENTER"
        else:
            obstacle_center = np.append(data.body("t_block").xipos[:2], OBSTACLE_HEIGHT)
            where = f"{OBSTACLE_HEIGHT} m above the T"
        mover = MovingObstacle(model, obstacle_center, axis=OBSTACLE_AXIS, amplitude=OBSTACLE_AMPLITUDE, speed=OBSTACLE_SPEED,
                                dwell=OBSTACLE_DWELL)
        print(f"Obstacle: around {np.round(mover.center, 3).tolist()} ({where}), "
              + (f"crosses +-{OBSTACLE_AMPLITUDE} m along {OBSTACLE_AXIS}, waiting {OBSTACLE_DWELL} s at each end"
                 if OBSTACLE_DWELL is not None else f"sweeps +-{OBSTACLE_AMPLITUDE} m along {OBSTACLE_AXIS}")
              + f" at {OBSTACLE_SPEED} m/s")
    else:
        mover = None
        data.mocap_pos[model.body_mocapid[model.body("obstacle").id]] = [-0.5, 0.0, 1.0]  # parked behind the robot
    min_clearance, hits = np.inf, 0

    descend_target = np.append(spot, PUSH_HEIGHT)
    obs_history, queue, target_px = [], [], None
    # one worker thread for inference: the sim loop never waits on it
    inference = ThreadPoolExecutor(max_workers=1)
    pending, pending_time = None, None  # the chunk being inferred, and the sim time of the observation it's from

    def infer(obs_stack):
        with torch.no_grad():  # thread-local in torch, so it goes here, not around submit()
            return policy.predict_action({"obs": torch.from_numpy(obs_stack[None]).to(DEVICE)})["action"][0].cpu().numpy()
    next_policy_time, push_start, hold_point = 0.0, None, None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t_wall = time.perf_counter()
            obstacle = mover.update(data) if mover is not None else None
            caps = geometry.world(data)  # every capsule in world frame, once per tick
            if obstacle is not None:
                clearance, touching = obstacle_clearance(model, data, caps["env"], obstacle)
                min_clearance = min(min_clearance, clearance)
                if touching and hits == 0:
                    print(f"[{data.time:6.2f}s] WARNING: the obstacle hit the robot")
                hits += touching

            if phase == "reach":
                U.control_step(model, data, controller, geometry, env, field=field, obstacle=obstacle, caps=caps)
                hover_point = np.append(spot, PUSH_HEIGHT + HOVER_HEIGHT)
                if (np.linalg.norm(data.qpos[:model.nu] - path_end) < REACH_TOL
                        or np.linalg.norm(data.site("probe").xpos - hover_point) < REACH_PROBE_TOL):
                    phase = "descend"
                    print(f"[{data.time:6.2f}s] reached the hover point -- descending")

            elif phase == "descend":
                probe_step(model, data, controller, geometry, env, descend_target, touch_block=False, obstacle=obstacle, caps=caps)
                if np.linalg.norm(data.site("probe").xpos - descend_target) < 0.005:
                    phase, push_start, next_policy_time = "push", data.time, data.time
                    print(f"[{data.time:6.2f}s] at push height -- policy takes over")

            elif phase == "push":
                if data.time >= next_policy_time:
                    obs = frame.observation(data)
                    obs_history = (obs_history + [obs])[-2:] if obs_history else [obs, obs]  # Push-T pads by repeating
                    coverage = frame.coverage(data)
                    if coverage >= SUCCESS_COVERAGE or data.time - push_start >= PUSH_TIME_LIMIT:
                        phase, hold_point, queue = "done", data.site("probe").xpos.copy(), []
                        verdict = "SUCCESS" if coverage >= SUCCESS_COVERAGE else "time limit"
                        print(f"[{data.time:6.2f}s] {verdict}: goal covered {coverage * 100:.1f}% after "
                              f"{data.time - push_start:.1f}s of pushing"
                              + (f" | obstacle: closest {min_clearance * 100:.1f} cm, {hits} contact ticks" if mover else ""))
                        continue
                    if pending is not None and pending.done():
                        # chunk[0] was meant for the tick its observation came from; skip the ones already past
                        late = int(round((data.time - pending_time) / POLICY_DT))
                        queue, pending = list(pending.result()[late:]), None
                        print(f"[{data.time:6.2f}s] goal covered {coverage * 100:5.1f}% | new action chunk, "
                              f"{late} tick(s) late, {len(queue)} actions left"
                              + (f" | obstacle: closest {min_clearance * 100:.1f} cm so far, {hits} contact ticks" if mover else ""))
                    if queue:
                        target_px = queue.pop(0)  # an empty queue keeps the last target until the chunk lands
                    if pending is None and len(queue) <= PREFETCH_ACTIONS:
                        pending, pending_time = inference.submit(infer, np.stack(obs_history)), data.time
                    next_policy_time += POLICY_DT
                push_target = descend_target if target_px is None else np.append(frame.to_table(target_px), PUSH_HEIGHT)
                probe_step(model, data, controller, geometry, env, push_target, touch_block=True, obstacle=obstacle, caps=caps)

            else:  # done: hold still where the probe stopped
                probe_step(model, data, controller, geometry, env, hold_point, touch_block=True, obstacle=obstacle, caps=caps)

            mujoco.mj_step(model, data)
            draw_targets(viewer, frame, queue, target_px if phase == "push" else None)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - t_wall)))  # pace to real time


if __name__ == "__main__":
    main()

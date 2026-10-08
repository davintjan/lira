"""Real-time control: reactive position/orientation DS, and the CBF-QP safety filter guarding it. Runs every 500Hz tick. Offline planning lives in planner.py."""
import numpy as np
import osqp
from scipy import sparse


class RobotController:
    def __init__(self, qpos, qvel, ee_pos, ee_vel):
        pass

    def compute_A_b(self, x_target, k_p):
        scaling = np.array([1.0, 1.0, 0.8])
        A = -k_p * np.diag(scaling)
        b = k_p * scaling * np.array(x_target)
        return A, b

    def linearPosDS(self, ee_curr_pos, ee_target_pos, k):
        """xdot = A @ x + b: linear DS pulling ee_curr_pos toward ee_target_pos."""
        A, b = self.compute_A_b(ee_target_pos, k)
        return A @ np.array(ee_curr_pos) + b

    def linearOrientationDS(self, ee_curr_quat, ee_target_quat, k):
        """omega = -k * logdq: angular velocity toward ee_target_quat via the quaternion log-map error."""
        q1 = np.array([*ee_curr_quat[1:], ee_curr_quat[0]])  # to scalar-last [x,y,z,w]
        q2 = np.array([*ee_target_quat[1:], ee_target_quat[0]])
        s1, u1 = q1[3], q1[:3]
        s2, u2 = q2[3], q2[:3]
        Su1 = np.array([
            [0, -u1[2], u1[1]],
            [u1[2], 0, -u1[0]],
            [-u1[1], u1[0], 0],
        ])
        dori = np.array([s1 * s2 + u1 @ u2.T, *(-s1 * u2 + s2 * u1 - Su1 @ u2)])
        if dori[0] < 0:  # q and -q are the same rotation: take the short way, not the 2pi-minus-angle one
            dori = -dori
        dori[0] = np.clip(dori[0], -1.0, 1.0)
        v = dori[1:]
        v_norm = np.linalg.norm(v)
        logdq = np.arccos(dori[0]) * (v / v_norm) if v_norm > 1e-6 else np.zeros(3)
        return -k * logdq

    def collisionQP(self, u_des, rows, alpha=5.0, u_max=None, metric=None):
        """
        CBF-QP safety filter: closest u to u_des s.t. grad_i . u >= -alpha * h_i for every
        clearance row (h_i, grad_i) from geometry.clearanceRows, plus optional
        -u_max <= u <= u_max. Falls back to u=0 (freeze) if infeasible -- including a violated
        row no joint can currently fix (zero gradient): a real collision, so stopping is right.
        "Closest" is (u - u_des)^T metric (u - u_des); metric=None is plain joint-space distance.
        A task-space metric (J^T J + a little identity) makes motions the task doesn't care about
        -- e.g. spinning a round tool about its axis -- nearly free, so the filter uses them first.
        """
        n_dof = len(u_des)
        if u_max is not None:
            # |grad . u| <= |grad|_1 * u_max inside the box, so these rows can never bind -- dropping them is exact
            rows = [(h, grad) for h, grad in rows if alpha * h < np.abs(grad).sum() * u_max]
        if not rows:
            return np.array(u_des) if u_max is None else np.clip(u_des, -u_max, u_max)  # box-only QP = clip (exact for metric=None)

        G = -np.vstack([grad for _, grad in rows])
        h_ub = alpha * np.array([h for h, _ in rows])
        l_bound = np.full_like(h_ub, -np.inf)

        if u_max is not None:
            box = np.eye(n_dof)
            G = np.vstack([G, box])
            h_ub = np.concatenate([h_ub, np.full(n_dof, u_max)])
            l_bound = np.concatenate([l_bound, np.full(n_dof, -u_max)])

        W = np.eye(n_dof) if metric is None else np.asarray(metric)
        P = sparse.csc_matrix(2.0 * W)
        q = -2.0 * W @ np.asarray(u_des)
        A = sparse.csc_matrix(G)

        solver = osqp.OSQP()
        solver.setup(P, q, A, l_bound, h_ub, verbose=False)
        result = solver.solve()

        if result.info.status not in ("solved", "solved inaccurate"):
            return np.zeros(n_dof)
        return result.x

def disturbances(model, data, magnitude, direction, duration, start=0.0, site="attachment_site"):
    """Simulated external push at the end effector: a force of `magnitude` N along `direction`
    (world-frame 3-vector, normalized here), applied at `site` while start <= data.time <
    start + duration, and nothing otherwise. Call it every tick, before mj_step.

    Uses data.xfrc_applied, which MuJoCo applies at the body's centre of mass, so a force at the
    site point also needs the torque it creates about that centre: (site - com) x force. The row
    is overwritten each call -- xfrc_applied persists between steps -- so the push really stops."""
    body = model.site_bodyid[model.site(site).id]
    active = start <= data.time < start + duration
    force = magnitude * np.asarray(direction, dtype=float) / np.linalg.norm(direction) if active else np.zeros(3)
    lever = data.site(site).xpos - data.xipos[body]
    data.xfrc_applied[body, :3] = force
    data.xfrc_applied[body, 3:] = np.cross(lever, force)

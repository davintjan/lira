"""Collision-distance/gradient primitives and the manipulability index. Pure math: no MuJoCo, no state."""
from dataclasses import dataclass, fields, replace

import numpy as np


@dataclass
class Capsules:
    """N capsules in world frame, as arrays: capsule k is the segment a[k]-b[k], swept by a sphere of
    radius[k] (a[k] == b[k]: a sphere). Indexing with an index array, a slice or a boolean mask gives
    those capsules, as a set of the same kind."""
    a: np.ndarray       # (N, 3)
    b: np.ndarray       # (N, 3)
    radius: np.ndarray  # (N,)

    def __len__(self):
        return len(self.radius)

    def __getitem__(self, index):
        picked = {name: values[index] for name, values in vars(self).items()}
        return type(self)(**picked)


@dataclass
class LinkCapsules(Capsules):
    """An arm's collision capsules in world frame (mj.scene.RobotGeometry.world): the body each is on,
    its name, and which of its clearances can change at all (see RobotGeometry._sample_hazards)."""
    body_id: np.ndarray        # (N,) int
    name: np.ndarray           # (N,) str
    guard_table: np.ndarray    # (N,) bool
    guard_objects: np.ndarray  # (N,) bool


@dataclass
class MovingCapsules(Capsules):
    """Capsules moving at vel[k] (m/s, world frame) -- the obstacle (mj.scene.MovingObstacle.update)."""
    vel: np.ndarray  # (N, 3)


@dataclass
class ArmCapsules:
    """An arm's capsules in world frame at one state (mj.scene.RobotGeometry.world)."""
    all: LinkCapsules             # every capsule, the synthetic base included (e.g. to draw them)
    self_collision: LinkCapsules  # the ones the end effector is kept clear of
    env: LinkCapsules             # the real collision geoms, for table and object checks
    pairs: tuple                  # the guarded link-vs-link pairs (first, second): first[k] against second[k]


@dataclass
class Box:
    """An oriented box: centred at center, its axes rot's columns, half_extents along each."""
    name: str
    center: np.ndarray
    rot: np.ndarray
    half_extents: np.ndarray


def concatenate(capsule_sets):
    """One set of all the capsules in capsule_sets, which are all of the same kind."""
    first = capsule_sets[0]
    joined = {}
    for field in fields(first):
        joined[field.name] = np.concatenate([getattr(caps, field.name) for caps in capsule_sets])
    return replace(first, **joined)


def quat2mat(q):
    """Scalar-first [w,x,y,z] quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def point_capsule_distance(P, A, B, R):
    """Signed distance from each point P[i] to capsule i (segment A[i]-B[i], radius R[i]). P, A, B are
    (N,3), R is (N,). Returns h (N,), the closest points C (N,3) on the segments and their parameters t (N,)."""
    AB = B - A
    ab_len_sq = np.sum(AB * AB, axis=1)
    safe_len_sq = np.where(ab_len_sq < 1e-12, 1.0, ab_len_sq)
    t = np.where(ab_len_sq < 1e-12, 0.0,
                 np.clip(np.sum((P - A) * AB, axis=1) / safe_len_sq, 0.0, 1.0))
    C = A + t[:, None] * AB
    diff = P - C
    h = np.sqrt(np.sum(diff * diff, axis=1)) - R
    return h, C, t


def capsule_capsule_distance(A1, B1, R1, A2, B2, R2):
    """Signed distance between capsules (A1[i], B1[i], R1[i]) and (A2[i], B2[i], R2[i]), and the closest
    points C1, C2 on their segments. Every array is (N,3) except R1 and R2, which are (N,). Segment-segment
    closest points per Ericson, Real-Time Collision Detection 5.1.9. Zero-length capsules (spheres, e.g.
    the iiwa's collision geoms) included: the general formula runs on safe denominators, then the point
    cases override it."""
    D1, D2, R = B1 - A1, B2 - A2, A1 - A2
    len1_sq = np.sum(D1 * D1, axis=1)
    len2_sq = np.sum(D2 * D2, axis=1)
    point1, point2 = len1_sq < 1e-12, len2_sq < 1e-12
    safe_len1_sq = np.where(point1, 1.0, len1_sq)
    safe_len2_sq = np.where(point2, 1.0, len2_sq)
    f = np.sum(D2 * R, axis=1)
    c = np.sum(D1 * R, axis=1)
    b = np.sum(D1 * D2, axis=1)
    denom = len1_sq * len2_sq - b * b
    safe_denom = np.where(denom > 1e-12, denom, 1.0)
    s = np.where(denom > 1e-12, np.clip((b * f - c * len2_sq) / safe_denom, 0.0, 1.0), 0.0)
    t = (b * s + f) / safe_len2_sq

    below, above = t < 0.0, t > 1.0
    t = np.clip(t, 0.0, 1.0)
    s = np.where(below, np.clip(-c / safe_len1_sq, 0.0, 1.0),
                 np.where(above, np.clip((b - c) / safe_len1_sq, 0.0, 1.0), s))

    # segment 1 a point: closest spot on segment 2 to it; segment 2 a point: closest spot on segment 1
    s = np.where(point1, 0.0, np.where(point2, np.clip(-c / safe_len1_sq, 0.0, 1.0), s))
    t = np.where(point2, 0.0, np.where(point1, np.clip(f / safe_len2_sq, 0.0, 1.0), t))

    C1 = A1 + s[:, None] * D1
    C2 = A2 + t[:, None] * D2
    diff = C1 - C2
    h = np.sqrt(np.sum(diff * diff, axis=1)) - R1 - R2
    return h, C1, C2


def half_space_capsule_distance(A, B, R, plane_height, normal=np.array([0.0, 0.0, 1.0])):
    """Signed clearance of each capsule above a horizontal plane. A, B are (N,3), R is (N,). Returns
    h (N,) and t (N,): which endpoint binds, 0 for A, 1 for B."""
    da, db = A @ normal, B @ normal
    use_a = da <= db
    h = np.where(use_a, da, db) - R - plane_height
    t = np.where(use_a, 0.0, 1.0)
    return h, t


def point_box_distance(P, box_center, box_rot, half_extents):
    """Signed distance from each point P[i] (P is (N,3)) to ONE oriented box, and the world-frame
    direction it grows in. Returns h (N,), grad (N,3)."""
    p_local = (P - box_center) @ box_rot  # box_rot.T @ v per row, row-vector form
    q = np.abs(p_local) - half_extents
    outside = np.maximum(q, 0.0)
    outside_dist = np.sqrt(np.sum(outside * outside, axis=1))
    inside_dist = np.minimum(np.max(q, axis=1), 0.0)
    h = outside_dist + inside_dist

    safe_od = np.where(outside_dist > 1e-9, outside_dist, 1.0)
    grad_local = np.sign(p_local) * outside / safe_od[:, None]

    inside_mask = outside_dist <= 1e-9
    if np.any(inside_mask):
        rows = np.nonzero(inside_mask)[0]
        axis = np.argmax(q[rows], axis=1)
        signs = np.sign(p_local[rows, axis])
        fallback = np.zeros((len(rows), 3))
        fallback[np.arange(len(rows)), axis] = np.where(signs >= 0, 1.0, -1.0)
        grad_local[rows] = fallback

    grad_world = grad_local @ box_rot.T  # box_rot @ g per row, row-vector form
    return h, grad_world


def capsule_box_distance(A, B, R, box_center, box_rot, half_extents):
    """Signed distance from each capsule (A, B are (N,3), R is (N,)) to ONE box, approximated by the worse
    of its two endpoints. Returns h (N,), t (N,) (which endpoint binds, 0 for A, 1 for B) and grad (N,3)."""
    h_a, g_a = point_box_distance(A, box_center, box_rot, half_extents)
    h_b, g_b = point_box_distance(B, box_center, box_rot, half_extents)
    h_a, h_b = h_a - R, h_b - R
    use_a = h_a <= h_b
    h = np.where(use_a, h_a, h_b)
    t = np.where(use_a, 0.0, 1.0)
    g = np.where(use_a[:, None], g_a, g_b)
    return h, t, g


def separating_direction(disp, axis1, axis2=None):
    """Unit direction to push two touching shapes apart: along disp (closest point to closest
    point) normally. disp ~ 0 means a point sits on a capsule's axis, or two axes cross -- the
    deepest overlap there is, not a safe case -- and then any direction perpendicular to the
    axes separates them, so pick one."""
    dist = np.linalg.norm(disp)
    if dist > 1e-9:
        return disp / dist
    n = np.cross(axis1, axis2) if axis2 is not None else np.zeros(3)
    if np.linalg.norm(n) < 1e-9:  # one axis only, or parallel axes: anything perpendicular to axis1
        n = np.cross(axis1, [1.0, 0.0, 0.0])
        if np.linalg.norm(n) < 1e-9:
            n = np.cross(axis1, [0.0, 1.0, 0.0])
        if np.linalg.norm(n) < 1e-9:  # zero-length axis: every direction separates
            n = np.array([1.0, 0.0, 0.0])
    return n / np.linalg.norm(n)


def every_pair(caps_a, caps_b):
    """Every capsule of caps_a against every capsule of caps_b: for pair k, the indices i[k], j[k] of
    its two capsules, their distance h[k] and their closest points c_a[k], c_b[k]."""
    i, j = np.meshgrid(np.arange(len(caps_a)), np.arange(len(caps_b)), indexing="ij")
    i, j = i.ravel(), j.ravel()
    h, c_a, c_b = capsule_capsule_distance(caps_a.a[i], caps_a.b[i], caps_a.radius[i],
                                           caps_b.a[j], caps_b.b[j], caps_b.radius[j])
    return i, j, h, c_a, c_b


def clearance_rows(ee_pos, jac_ee, caps, boxes, table_height, jac_fn, margin, table_margin, activation):
    """Every guarded clearance of one arm at one pose, as (h, grad) rows -- the single source for the CBF
    (cbf.cbf_qp), the planner cost and the planner's intrusion check:
      h    = clearance - its margin, in m (negative = inside the margin)
      grad = dh/dq, shape (n_dof,)
    caps: the arm's ArmCapsules. The hazards: the end effector against the self-collision capsules, the
    guarded link pairs, the table (table_margin: several links rest mm above it by design, so the full
    margin there would make the QP infeasible at rest) and the boxes.
    Left out: hazards with h >= activation (too far away to matter), and clearances fixed by design (a
    capsule's guard_table / guard_objects flags). Every other hazard gets a row, even at a pose where its
    gradient happens to be zero or the direction has to be picked (see separating_direction).
    jac_fn(body_id, point): the position Jacobian of a world point fixed to body_id."""
    env = caps.env
    return (ee_rows(ee_pos, jac_ee, caps.self_collision, jac_fn, margin, activation)
            + pair_rows(caps.pairs, jac_fn, margin, activation)
            + table_rows(env[env.guard_table], table_height, jac_fn, table_margin, activation)
            + box_rows(env[env.guard_objects], boxes, jac_fn, margin, activation))


def ee_rows(ee_pos, jac_ee, caps, jac_fn, margin, activation):
    """The end effector point against each capsule. Both move, so the clearance changes at
    n . (velocity of the end effector - velocity of the capsule's closest point)."""
    h, closest, _ = point_capsule_distance(np.broadcast_to(ee_pos, caps.a.shape), caps.a, caps.b, caps.radius)
    h = h - margin
    rows = []
    for k in np.nonzero(h < activation)[0]:
        n = separating_direction(ee_pos - closest[k], caps.b[k] - caps.a[k])
        rows.append((h[k], n @ (jac_ee - jac_fn(caps.body_id[k], closest[k]))))
    return rows


def pair_rows(pairs, jac_fn, margin, activation):
    """Link against link, for each pair (first[k], second[k]). Both links move, so the clearance changes
    at n . (velocity of first's closest point - velocity of second's)."""
    first, second = pairs
    h, c1, c2 = capsule_capsule_distance(first.a, first.b, first.radius, second.a, second.b, second.radius)
    h = h - margin
    rows = []
    for k in np.nonzero(h < activation)[0]:
        n = separating_direction(c1[k] - c2[k], first.b[k] - first.a[k], second.b[k] - second.a[k])
        rows.append((h[k], n @ (jac_fn(first.body_id[k], c1[k]) - jac_fn(second.body_id[k], c2[k]))))
    return rows


def table_rows(caps, table_height, jac_fn, table_margin, activation):
    """Each capsule against the table top, by its lower endpoint."""
    h, t = half_space_capsule_distance(caps.a, caps.b, caps.radius, table_height)
    h = h - table_margin
    rows = []
    for k in np.nonzero(h < activation)[0]:
        lower = caps.a[k] if t[k] == 0.0 else caps.b[k]
        rows.append((h[k], np.array([0.0, 0.0, 1.0]) @ jac_fn(caps.body_id[k], lower)))
    return rows


def box_rows(caps, boxes, jac_fn, margin, activation):
    """Each capsule against each box, by its endpoint nearer the box."""
    rows = []
    for box in boxes:
        h, t, grad = capsule_box_distance(caps.a, caps.b, caps.radius, box.center, box.rot, box.half_extents)
        h = h - margin
        for k in np.nonzero(h < activation)[0]:
            nearer = caps.a[k] if t[k] == 0.0 else caps.b[k]
            rows.append((h[k], grad[k] @ jac_fn(caps.body_id[k], nearer)))
    return rows


def moving_obstacle_rows(env_capsules, obstacle, jac_fn, margin, activation=np.inf, alpha=np.inf):
    """Clearance rows for every robot capsule that can move (guard_objects) against a moving obstacle (a
    MovingCapsules of one, see scene.MovingObstacle), as (h, grad, h_dot_obstacle):
      h              = clearance - margin, in m
      grad           = dh/dq, shape (n_dof,)
      h_dot_obstacle = how fast the clearance shrinks from the obstacle's own motion, -n . vel
                       (n points from the obstacle toward the robot)
    so the clearance changes at grad . qdot + h_dot_obstacle. Kept apart from clearance_rows: the
    planner plans once and can't know where a moving obstacle will be, so only the CBF uses these.
    Left out, as in clearance_rows: rows whose effective clearance h + h_dot_obstacle / alpha (what
    the CBF actually holds >= 0) is >= activation -- so an obstacle closing in fast gets its row early."""
    caps = env_capsules[env_capsules.guard_objects]
    if len(caps) == 0:
        return []
    i, _, h, c_robot, c_obstacle = every_pair(caps, obstacle)
    h = h - margin
    obstacle_axis = obstacle.b[0] - obstacle.a[0]
    rows = []
    for k in range(len(h)):
        n = separating_direction(c_robot[k] - c_obstacle[k], caps.b[i[k]] - caps.a[i[k]], obstacle_axis)
        h_dot_obstacle = -n @ obstacle.vel[0]
        if h[k] + h_dot_obstacle / alpha < activation:
            rows.append((h[k], n @ jac_fn(caps.body_id[i[k]], c_robot[k]), h_dot_obstacle))
    return rows


def capsule_set_rows(env_capsules, others, jac_fn, margin, activation=np.inf):
    """Clearance rows for every robot capsule that can move (guard_objects) against each of `others`
    -- capsules held where they are now, e.g. another arm's links -- as (h, grad):
      h    = clearance - margin, in m
      grad = dh/dq from this arm's joints only, shape (n_dof,)
    The other side's motion is left out. When two arms each hold grad_i . u_i >= -alpha * h for the
    same pair, the pair's clearance changes at h_dot = grad_a . u_a + grad_b . u_b >= -2 * alpha * h:
    still a CBF (with twice the gain), so the pair stays apart without either arm knowing how the other
    moves. One row per (this arm's body, other body): the closest of their capsule pairs -- a link made
    of many spheres (the iiwa's) would otherwise add a row per sphere pair, all pushing the same way.
    Left out, as in clearance_rows: rows with h >= activation."""
    caps = env_capsules[env_capsules.guard_objects]
    if len(caps) == 0 or len(others) == 0:
        return []
    i, j, h, c_robot, c_other = every_pair(caps, others)
    h = h - margin
    closest = {}  # (body, other body) -> index of their closest capsule pair
    for k in np.nonzero(h < activation)[0]:
        key = (caps.body_id[i[k]], others.body_id[j[k]])
        if key not in closest or h[k] < h[closest[key]]:
            closest[key] = k
    rows = []
    for k in closest.values():
        mine, theirs = i[k], j[k]
        my_axis = caps.b[mine] - caps.a[mine]
        their_axis = others.b[theirs] - others.a[theirs]
        n = separating_direction(c_robot[k] - c_other[k], my_axis, their_axis)
        rows.append((h[k], n @ jac_fn(caps.body_id[mine], c_robot[k])))
    return rows


def manipulability(J):
    """Yoshikawa manipulability index sqrt(det(J J^T))."""
    return np.sqrt(max(np.linalg.det(J @ J.T), 0.0))

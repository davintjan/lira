"""Collision-distance/gradient primitives and the manipulability index. Pure math: no MuJoCo, no state."""
import numpy as np


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


def clearance_rows(ee_pos, jac_ee, self_capsules, link_pairs, env_capsules, object_boxes, table_height,
                  jac_fn, margin, table_margin, activation):
    """Every guarded clearance at one arm pose, as (h, grad) rows -- the single source for the CBF
    (cbf.cbf_qp), the planner cost and the planner's intrusion check.
      h    = clearance - its margin, in m (negative = inside the margin)
      grad = dh/dq, shape (n_dof,)
    Hazards: the ee point vs each self_capsule (other links + synthetic base); link vs link for
    each of link_pairs; each env_capsule vs the table (table_margin -- several links rest mm above
    it by design, so margin there would make the QP infeasible at rest) and each object box.
    Left out: hazards with h >= activation (too far away to matter), and clearances that are fixed
    by design (the capsule's guard_table / guard_objects flags, see RobotGeometry._sample_hazards).
    Every other hazard always gets a row -- even at a pose where its gradient happens to be zero,
    or where two axes touch and the direction has to be picked (see separating_direction).
    jac_fn(body_id, point) -> 3x6 position Jacobian of a world point fixed to body_id. A point's
    Jacobian is linear in the point, so jac_fn at an interior point c equals blending the endpoint
    Jacobians -- one call instead of two.

    Distances for every hazard of one kind are computed together via the array primitives above
    (plain numpy over all of them at once) -- profiled to be ~80% of a GD optimize() call's time
    when done one hazard at a time in a Python loop. jac_fn (a real mj_jac call) can't be batched
    the same way and is the one genuinely per-state MuJoCo cost here, so it still runs per row --
    but only for rows that actually survive the activation cutoff, which is the common case's
    minority."""
    rows = []

    # ee point vs link
    if self_capsules:
        A = np.array([cap["a"] for cap in self_capsules])
        B = np.array([cap["b"] for cap in self_capsules])
        R = np.array([cap["radius"] for cap in self_capsules])
        h, C, _ = point_capsule_distance(np.broadcast_to(ee_pos, A.shape), A, B, R)
        h = h - margin
        for k in np.nonzero(h < activation)[0]:
            cap = self_capsules[k]
            n = separating_direction(ee_pos - C[k], cap["b"] - cap["a"])
            rows.append((h[k], n @ (jac_ee - jac_fn(cap["body_id"], C[k]))))

    # link vs link: both links move, so clearance rate = n . (velocity of c1 - velocity of c2)
    if link_pairs:
        A1 = np.array([p[0]["a"] for p in link_pairs])
        B1 = np.array([p[0]["b"] for p in link_pairs])
        R1 = np.array([p[0]["radius"] for p in link_pairs])
        A2 = np.array([p[1]["a"] for p in link_pairs])
        B2 = np.array([p[1]["b"] for p in link_pairs])
        R2 = np.array([p[1]["radius"] for p in link_pairs])
        h, C1, C2 = capsule_capsule_distance(A1, B1, R1, A2, B2, R2)
        h = h - margin
        for k in np.nonzero(h < activation)[0]:
            cap1, cap2 = link_pairs[k]
            n = separating_direction(C1[k] - C2[k], cap1["b"] - cap1["a"], cap2["b"] - cap2["a"])
            rows.append((h[k], n @ (jac_fn(cap1["body_id"], C1[k]) - jac_fn(cap2["body_id"], C2[k]))))

    # link vs table: only capsules whose height is actually q-dependent (guard_table)
    table_caps = [cap for cap in env_capsules if cap["guard_table"]]
    if table_caps:
        A = np.array([cap["a"] for cap in table_caps])
        B = np.array([cap["b"] for cap in table_caps])
        R = np.array([cap["radius"] for cap in table_caps])
        h, t = half_space_capsule_distance(A, B, R, table_height)
        h = h - table_margin
        for k in np.nonzero(h < activation)[0]:
            cap = table_caps[k]
            bind_pt = cap["a"] if t[k] == 0.0 else cap["b"]
            rows.append((h[k], np.array([0.0, 0.0, 1.0]) @ jac_fn(cap["body_id"], bind_pt)))

    # link vs object: only capsules that actually move (guard_objects), one batched call per box
    obj_caps = [cap for cap in env_capsules if cap["guard_objects"]]
    if obj_caps and object_boxes:
        A = np.array([cap["a"] for cap in obj_caps])
        B = np.array([cap["b"] for cap in obj_caps])
        R = np.array([cap["radius"] for cap in obj_caps])
        for box in object_boxes:
            h, t, g = capsule_box_distance(A, B, R, box["center"], box["rot"], box["half_extents"])
            h = h - margin
            for k in np.nonzero(h < activation)[0]:
                cap = obj_caps[k]
                bind_pt = cap["a"] if t[k] == 0.0 else cap["b"]
                rows.append((h[k], g[k] @ jac_fn(cap["body_id"], bind_pt)))

    return rows


def moving_obstacle_rows(env_capsules, obstacle, jac_fn, margin, activation=np.inf, alpha=np.inf):
    """Clearance rows for every robot capsule that can move (guard_objects) against a moving
    capsule-shaped obstacle (a, b, radius, vel -- see scene.MovingObstacle), as
    (h, grad, h_dot_obstacle):
      h              = clearance - margin, in m
      grad           = dh/dq, shape (n_dof,)
      h_dot_obstacle = how fast the clearance shrinks from the obstacle's own motion, -n . vel
                       (n points from the obstacle toward the robot)
    so the clearance changes at grad . qdot + h_dot_obstacle. Kept apart from clearance_rows: the
    planner plans once and can't know where a moving obstacle will be, so only the CBF uses these.
    Left out, as in clearance_rows: rows whose effective clearance h + h_dot_obstacle / alpha (what
    the CBF actually holds >= 0) is >= activation -- so an obstacle closing in fast gets its row early."""
    caps = [cap for cap in env_capsules if cap["guard_objects"]]
    if not caps:
        return []
    A = np.array([cap["a"] for cap in caps])
    B = np.array([cap["b"] for cap in caps])
    R = np.array([cap["radius"] for cap in caps])
    h, C_robot, C_obstacle = capsule_capsule_distance(
        A, B, R, np.broadcast_to(obstacle["a"], A.shape), np.broadcast_to(obstacle["b"], A.shape),
        np.full(len(caps), obstacle["radius"]))
    h = h - margin
    rows = []
    for k, cap in enumerate(caps):
        n = separating_direction(C_robot[k] - C_obstacle[k], cap["b"] - cap["a"], obstacle["b"] - obstacle["a"])
        h_dot_obstacle = -n @ obstacle["vel"]
        if h[k] + h_dot_obstacle / alpha < activation:
            rows.append((h[k], n @ jac_fn(cap["body_id"], C_robot[k]), h_dot_obstacle))
    return rows



def capsule_set_rows(env_capsules, others, jac_fn, margin, activation=np.inf):
    """Clearance rows for every robot capsule that can move (guard_objects) against each of `others`
    -- capsules (a, b, radius) held where they are now, e.g. another arm's links -- as (h, grad):
      h    = clearance - margin, in m
      grad = dh/dq from this arm's joints only, shape (n_dof,)
    The other side's motion is left out. When two arms each hold grad_i . u_i >= -alpha * h for the
    same pair, the pair's clearance changes at h_dot = grad_a . u_a + grad_b . u_b >= -2 * alpha * h:
    still a CBF (with twice the gain), so the pair stays apart without either arm knowing how the other
    moves. One row per (this arm's body, other body): the closest of their capsule pairs -- a link made
    of many spheres (the iiwa's) would otherwise add a row per sphere pair, all pushing the same way.
    Left out, as in clearance_rows: rows with h >= activation."""
    caps = [cap for cap in env_capsules if cap["guard_objects"]]
    if not caps or not others:
        return []
    I, J = (idx.ravel() for idx in np.meshgrid(np.arange(len(caps)), np.arange(len(others)), indexing="ij"))
    A = np.array([cap["a"] for cap in caps])
    B = np.array([cap["b"] for cap in caps])
    R = np.array([cap["radius"] for cap in caps])
    OA = np.array([cap["a"] for cap in others])
    OB = np.array([cap["b"] for cap in others])
    OR = np.array([cap["radius"] for cap in others])
    h, C_robot, C_other = capsule_capsule_distance(A[I], B[I], R[I], OA[J], OB[J], OR[J])
    h = h - margin
    closest = {}  # (body, other body) -> index of their closest capsule pair
    for k in np.nonzero(h < activation)[0]:
        key = (caps[I[k]]["body_id"], others[J[k]]["body_id"])
        if key not in closest or h[k] < h[closest[key]]:
            closest[key] = k
    rows = []
    for k in closest.values():
        cap, other = caps[I[k]], others[J[k]]
        n = separating_direction(C_robot[k] - C_other[k], cap["b"] - cap["a"], other["b"] - other["a"])
        rows.append((h[k], n @ jac_fn(cap["body_id"], C_robot[k])))
    return rows

def manipulability(J):
    """Yoshikawa manipulability index sqrt(det(J J^T))."""
    return np.sqrt(max(np.linalg.det(J @ J.T), 0.0))

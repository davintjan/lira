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


def pointCapsuleDistance(p, a, b, r):
    """Signed distance from point p to capsule (segment [a,b], radius r); also returns closest point c and clamp t."""
    p, a, b = np.array(p), np.array(a), np.array(b)
    ab = b - a
    ab_len_sq = ab @ ab
    t = 0.0 if ab_len_sq < 1e-12 else np.clip((p - a) @ ab / ab_len_sq, 0.0, 1.0)
    c = a + t * ab
    h = np.linalg.norm(p - c) - r
    return h, c, t


def pointCapsuleDistanceBatch(P, A, B, R):
    """Batched pointCapsuleDistance: P, A, B are (N,3) arrays, R is (N,). Returns h (N,), C (N,3), t (N,).
    np.linalg.norm's generic dispatch is expensive for many tiny 3-vectors one at a time (profiled:
    ~25% of a GD optimize() call) -- batching amortizes that overhead across all N at once."""
    AB = B - A
    ab_len_sq = np.sum(AB * AB, axis=1)
    safe_len_sq = np.where(ab_len_sq < 1e-12, 1.0, ab_len_sq)
    t = np.where(ab_len_sq < 1e-12, 0.0,
                 np.clip(np.sum((P - A) * AB, axis=1) / safe_len_sq, 0.0, 1.0))
    C = A + t[:, None] * AB
    diff = P - C
    h = np.sqrt(np.sum(diff * diff, axis=1)) - R
    return h, C, t


def capsuleCapsuleDistance(a1, b1, r1, a2, b2, r2):
    """Signed distance between capsules [a1,b1] and [a2,b2]; also returns the closest points c1, c2 on
    their center segments. Segment-segment closest points per Ericson, Real-Time Collision Detection 5.1.9."""
    a1, b1, a2, b2 = np.array(a1), np.array(b1), np.array(a2), np.array(b2)
    d1, d2, r = b1 - a1, b2 - a2, a1 - a2
    len1_sq, len2_sq = d1 @ d1, d2 @ d2
    f = d2 @ r
    if len1_sq < 1e-12 and len2_sq < 1e-12:
        s, t = 0.0, 0.0
    elif len1_sq < 1e-12:
        s, t = 0.0, np.clip(f / len2_sq, 0.0, 1.0)
    else:
        c = d1 @ r
        if len2_sq < 1e-12:
            s, t = np.clip(-c / len1_sq, 0.0, 1.0), 0.0
        else:
            b = d1 @ d2
            denom = len1_sq * len2_sq - b * b
            s = np.clip((b * f - c * len2_sq) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0  # parallel: any s works
            t = (b * s + f) / len2_sq
            if t < 0.0:
                s, t = np.clip(-c / len1_sq, 0.0, 1.0), 0.0
            elif t > 1.0:
                s, t = np.clip((b - c) / len1_sq, 0.0, 1.0), 1.0
    c1, c2 = a1 + s * d1, a2 + t * d2
    h = np.linalg.norm(c1 - c2) - r1 - r2
    return h, c1, c2


def capsuleCapsuleDistanceBatch(A1, B1, R1, A2, B2, R2):
    """Batched capsuleCapsuleDistance for N pairs: every array is (N,3) except R1/R2 which are (N,).
    Assumes every capsule has nonzero length (true for every real link capsule this project guards),
    so it skips the scalar version's zero-length-capsule branches -- see capsuleCapsuleDistance."""
    D1, D2, R = B1 - A1, B2 - A2, A1 - A2
    len1_sq = np.sum(D1 * D1, axis=1)
    len2_sq = np.sum(D2 * D2, axis=1)
    f = np.sum(D2 * R, axis=1)
    c = np.sum(D1 * R, axis=1)
    b = np.sum(D1 * D2, axis=1)
    denom = len1_sq * len2_sq - b * b
    safe_denom = np.where(denom > 1e-12, denom, 1.0)
    s = np.where(denom > 1e-12, np.clip((b * f - c * len2_sq) / safe_denom, 0.0, 1.0), 0.0)
    t = (b * s + f) / len2_sq

    below, above = t < 0.0, t > 1.0
    t = np.clip(t, 0.0, 1.0)
    s = np.where(below, np.clip(-c / len1_sq, 0.0, 1.0),
                 np.where(above, np.clip((b - c) / len1_sq, 0.0, 1.0), s))

    C1 = A1 + s[:, None] * D1
    C2 = A2 + t[:, None] * D2
    diff = C1 - C2
    h = np.sqrt(np.sum(diff * diff, axis=1)) - R1 - R2
    return h, C1, C2


def halfSpaceCapsuleDistance(a, b, radius, plane_height, normal=np.array([0.0, 0.0, 1.0])):
    """Signed clearance of capsule [a,b] above a horizontal plane; t says which endpoint binds."""
    da, db = a @ normal, b @ normal
    if da <= db:
        return da - radius - plane_height, 0.0
    return db - radius - plane_height, 1.0


def halfSpaceCapsuleDistanceBatch(A, B, R, plane_height, normal=np.array([0.0, 0.0, 1.0])):
    """Batched halfSpaceCapsuleDistance: A, B are (N,3), R is (N,). Returns h (N,), t (N,)."""
    da, db = A @ normal, B @ normal
    use_a = da <= db
    h = np.where(use_a, da, db) - R - plane_height
    t = np.where(use_a, 0.0, 1.0)
    return h, t


def pointBoxDistance(p, box_center, box_rot, half_extents):
    """Signed distance from point p to an oriented box, plus the world-frame gradient direction."""
    p_local = box_rot.T @ (np.array(p) - box_center)
    q = np.abs(p_local) - half_extents
    outside = np.maximum(q, 0.0)
    outside_dist = np.linalg.norm(outside)
    inside_dist = min(max(q[0], q[1], q[2]), 0.0)
    h = outside_dist + inside_dist

    if outside_dist > 1e-9:
        grad_local = np.sign(p_local) * outside / outside_dist  # abs() above dropped the side p is on
    else:
        axis = np.argmax(q)
        grad_local = np.zeros(3)
        grad_local[axis] = 1.0 if p_local[axis] >= 0 else -1.0
    return h, box_rot @ grad_local


def pointBoxDistanceBatch(P, box_center, box_rot, half_extents):
    """Batched pointBoxDistance: N points (P, shape (N,3)) against ONE box. Returns h (N,), grad (N,3).
    (Only points are batched, not boxes -- object_boxes has ~2 entries, capsule endpoints ~9-18, so
    the win is batching over capsules; looping the ~2 boxes around this call is negligible.)"""
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


def capsuleBoxDistanceBatch(A, B, R, box_center, box_rot, half_extents):
    """Batched capsuleBoxDistance: N capsules (A, B shape (N,3), R shape (N,)) against ONE box."""
    h_a, g_a = pointBoxDistanceBatch(A, box_center, box_rot, half_extents)
    h_b, g_b = pointBoxDistanceBatch(B, box_center, box_rot, half_extents)
    h_a, h_b = h_a - R, h_b - R
    use_a = h_a <= h_b
    h = np.where(use_a, h_a, h_b)
    t = np.where(use_a, 0.0, 1.0)
    g = np.where(use_a[:, None], g_a, g_b)
    return h, t, g


def capsuleBoxDistance(a, b, radius, box_center, box_rot, half_extents):
    """Capsule-to-box distance, approximated by the worse of the capsule's two endpoints."""
    h_a, g_a = pointBoxDistance(a, box_center, box_rot, half_extents)
    h_b, g_b = pointBoxDistance(b, box_center, box_rot, half_extents)
    h_a -= radius
    h_b -= radius
    if h_a <= h_b:
        return h_a, 0.0, g_a
    return h_b, 1.0, g_b


def separatingDirection(disp, axis1, axis2=None):
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


def clearanceRows(ee_pos, jac_ee, self_capsules, link_pairs, env_capsules, object_boxes, table_height,
                  jac_fn, margin, table_margin, activation):
    """Every guarded clearance at one arm pose, as (h, grad) rows -- the single source for the CBF
    (control.collisionQP), the planner cost and the planner's intrusion check.
      h    = clearance - its margin, in m (negative = inside the margin)
      grad = dh/dq, shape (6,)
    Hazards: the ee point vs each self_capsule (other links + synthetic base); link vs link for
    each of link_pairs; each env_capsule vs the table (table_margin -- several links rest mm above
    it by design, so margin there would make the QP infeasible at rest) and each object box.
    Left out: hazards with h >= activation (too far away to matter), and clearances that are fixed
    by design (the capsule's guard_table / guard_objects flags, see RobotGeometry._sample_hazards).
    Every other hazard always gets a row -- even at a pose where its gradient happens to be zero,
    or where two axes touch and the direction has to be picked (see separatingDirection).
    jac_fn(body_id, point) -> 3x6 position Jacobian of a world point fixed to body_id. A point's
    Jacobian is linear in the point, so jac_fn at an interior point c equals blending the endpoint
    Jacobians -- one call instead of two.

    Distances for every hazard of one kind are computed together via the *Batch primitives above
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
        h, C, _ = pointCapsuleDistanceBatch(np.broadcast_to(ee_pos, A.shape), A, B, R)
        h = h - margin
        for k in np.nonzero(h < activation)[0]:
            cap = self_capsules[k]
            n = separatingDirection(ee_pos - C[k], cap["b"] - cap["a"])
            rows.append((h[k], n @ (jac_ee - jac_fn(cap["body_id"], C[k]))))

    # link vs link: both links move, so clearance rate = n . (velocity of c1 - velocity of c2)
    if link_pairs:
        A1 = np.array([p[0]["a"] for p in link_pairs])
        B1 = np.array([p[0]["b"] for p in link_pairs])
        R1 = np.array([p[0]["radius"] for p in link_pairs])
        A2 = np.array([p[1]["a"] for p in link_pairs])
        B2 = np.array([p[1]["b"] for p in link_pairs])
        R2 = np.array([p[1]["radius"] for p in link_pairs])
        h, C1, C2 = capsuleCapsuleDistanceBatch(A1, B1, R1, A2, B2, R2)
        h = h - margin
        for k in np.nonzero(h < activation)[0]:
            cap1, cap2 = link_pairs[k]
            n = separatingDirection(C1[k] - C2[k], cap1["b"] - cap1["a"], cap2["b"] - cap2["a"])
            rows.append((h[k], n @ (jac_fn(cap1["body_id"], C1[k]) - jac_fn(cap2["body_id"], C2[k]))))

    # link vs table: only capsules whose height is actually q-dependent (guard_table)
    table_caps = [cap for cap in env_capsules if cap["guard_table"]]
    if table_caps:
        A = np.array([cap["a"] for cap in table_caps])
        B = np.array([cap["b"] for cap in table_caps])
        R = np.array([cap["radius"] for cap in table_caps])
        h, t = halfSpaceCapsuleDistanceBatch(A, B, R, table_height)
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
            h, t, g = capsuleBoxDistanceBatch(A, B, R, box["center"], box["rot"], box["half_extents"])
            h = h - margin
            for k in np.nonzero(h < activation)[0]:
                cap = obj_caps[k]
                bind_pt = cap["a"] if t[k] == 0.0 else cap["b"]
                rows.append((h[k], g[k] @ jac_fn(cap["body_id"], bind_pt)))

    return rows


def movingObstacleRows(env_capsules, obstacle, jac_fn, margin):
    """Clearance rows for every robot capsule that can move (guard_objects) against a moving
    capsule-shaped obstacle (a, b, radius, vel -- see scene.MovingObstacle), as
    (h, grad, h_dot_obstacle):
      h              = clearance - margin, in m
      grad           = dh/dq, shape (6,)
      h_dot_obstacle = how fast the clearance shrinks from the obstacle's own motion, -n . vel
                       (n points from the obstacle toward the robot)
    so the clearance changes at grad . qdot + h_dot_obstacle. Kept apart from clearanceRows: the
    planner plans once and can't know where a moving obstacle will be, so only the CBF uses these."""
    rows = []
    for cap in env_capsules:
        if not cap["guard_objects"]:
            continue
        h, c_robot, c_obstacle = capsuleCapsuleDistance(cap["a"], cap["b"], cap["radius"],
                                                        obstacle["a"], obstacle["b"], obstacle["radius"])
        n = separatingDirection(c_robot - c_obstacle, cap["b"] - cap["a"], obstacle["b"] - obstacle["a"])
        rows.append((h - margin, n @ jac_fn(cap["body_id"], c_robot), -n @ obstacle["vel"]))
    return rows


def manipulability(J):
    """Yoshikawa manipulability index sqrt(det(J J^T))."""
    return np.sqrt(max(np.linalg.det(J @ J.T), 0.0))

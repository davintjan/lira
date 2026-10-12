"""Table <-> Push-T pixels, and Push-T's observation and success measure computed from the MuJoCo T.
The policy only ever sees Push-T pixels; this is the one place that knows how the table maps to them."""
import numpy as np

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

    def inside(self, px, margin=0.0):
        return bool(np.all(px >= margin) and np.all(px <= self.SIZE_PX - margin))

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

    def prepush_spot(self, data, distance, margin_px):
        """Table xy to start pushing from: `distance` m from the T's center of mass -- clear of it --
        inside the Push-T workspace by margin_px (the policy never saw a pusher outside it), and as
        far behind the T (away from the goal) as that allows."""
        com = data.body("t_block").xipos[:2]
        away = com - self.to_table(self.GOAL_PX)
        away = away / np.linalg.norm(away) if np.linalg.norm(away) > 1e-9 else np.array([1.0, 0.0])
        best = None
        for theta in np.radians(np.arange(0, 360, 5)):
            spot = com + distance * np.array([np.cos(theta), np.sin(theta)])
            if self.inside(self.to_px(spot), margin_px) and (best is None or (spot - com) @ away > (best - com) @ away):
                best = spot
        return best if best is not None else com + distance * away


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

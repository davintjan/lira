"""Scene geometry: RobotGeometry tracks the arm's self-collision capsules; EnvironmentGeometry tracks
the tabletop and t-block. Both read model.* once at init and refresh world poses via update(data)."""
import itertools

import mujoco
import numpy as np

from geometry import quat2mat, capsuleCapsuleDistance


class RobotGeometry:
    """Self-collision capsules: MuJoCo's real collision geoms (group 3) plus a synthetic base capsule (base ships with none).
    Also which clearances are worth guarding at all (see _sample_hazards)."""

    # ee sits rigidly off wrist_2/wrist_3 (and the pusher rod, which starts at it -- see
    # ur5_sim.load_model) by construction, so guarding them for self-collision would just fight
    # the arm's own fixed geometry.
    _EXCLUDED_BODIES = {"wrist_2_link", "wrist_3_link", "pusher"}

    # base has no collision geom in ur5e.xml; fitted by hand to the real
    # base_0/base_1 mesh vertices (z in [0,0.099], radius 0.076) -- NOT the
    # base-to-shoulder joint offset (0.163), which overlaps shoulder_link's capsule.
    _SYNTHETIC_BASE_CAPSULE = dict(
        body_name="base", local_pos=np.array([0.0, 0.0, 0.0495]),
        local_quat=np.array([1.0, 0.0, 0.0, 0.0]), radius=0.076, half_length=0.0495,
    )

    def __init__(self, model):
        # capsules: filtered (excludes wrist_2/3), used for self-collision.
        # env_capsules: unfiltered, used for table/object checks.
        self.capsules = []
        self.env_capsules = []

        for gid in range(model.ngeom):
            if model.geom_group[gid] != 3:
                continue
            body_id = model.geom_bodyid[gid]
            body_name = model.body(body_id).name
            radius, half_length = model.geom_size[gid][:2]
            entry = dict(
                body_id=body_id,
                name=model.geom(gid).name or f"{body_name}_geom{gid}",
                local_pos=model.geom_pos[gid].copy(),
                local_rot=quat2mat(model.geom_quat[gid]),
                radius=radius, half_length=half_length,
            )
            self.env_capsules.append(entry)
            if body_name not in self._EXCLUDED_BODIES:
                self.capsules.append(entry)

        # base_synthetic: self-collision only -- it's fixed to the world, so
        # its table/object clearance can never change with q.
        base = self._SYNTHETIC_BASE_CAPSULE
        base_entry = dict(
            body_id=model.body(base["body_name"]).id, name="base_synthetic",
            local_pos=base["local_pos"], local_rot=quat2mat(base["local_quat"]),
            radius=base["radius"], half_length=base["half_length"],
        )
        self.capsules.append(base_entry)

        # link_capsules: every capsule incl. base_synthetic; link_pairs index into it.
        self.link_capsules = self.env_capsules + [base_entry]
        self.link_pairs = self._sample_hazards(model)

    def _sample_hazards(self, model, n_samples=2000, near=0.10, pair_fixed_tol=0.03, still_tol=1e-3, seed=0):
        """Decides once, by sampling random arm poses, which clearances are worth guarding -- so no
        check ever has to guess per pose. A clearance that never changes is fixed by design, not a
        collision; one that does change always counts, even at a pose where its gradient happens
        to be zero. Sets on each env capsule:
          guard_table   -- False if its lowest point never changes height (e.g. shoulder_link,
                           upper_arm's joint cylinder): no joint can bring it closer to the table.
          guard_objects -- False if it never moves at all (shoulder_link spins in place, like the
                           base): no joint can bring it closer to anything.
        Returns the link-vs-link capsule index pairs to guard: drops same-body and parent/child
        pairs (they meet at their joint by design), pairs that never come within `near` m, and
        pairs whose clearance varies by less than `pair_fixed_tol` m -- those sit rigidly close by
        construction (e.g. upper_arm vs base), so guarding them would only block the arm."""
        caps = self.link_capsules
        candidates = []
        for i, j in itertools.combinations(range(len(caps)), 2):
            bi, bj = caps[i]["body_id"], caps[j]["body_id"]
            if bi == bj or model.body_parentid[bi] == bj or model.body_parentid[bj] == bi:
                continue
            candidates.append((i, j))

        data = mujoco.MjData(model)
        lo, hi = model.jnt_range[:6, 0], model.jnt_range[:6, 1]
        rng = np.random.default_rng(seed)
        h_min = np.full(len(candidates), np.inf)
        h_max = np.full(len(candidates), -np.inf)
        endpoints, lowest = [], []
        for _ in range(n_samples):
            data.qpos[:6] = rng.uniform(lo, hi)
            mujoco.mj_kinematics(model, data)
            world = self._capsules_world(data, caps)
            for k, (i, j) in enumerate(candidates):
                h, _, _ = capsuleCapsuleDistance(world[i]["a"], world[i]["b"], world[i]["radius"],
                                                 world[j]["a"], world[j]["b"], world[j]["radius"])
                h_min[k] = min(h_min[k], h)
                h_max[k] = max(h_max[k], h)
            env_world = world[:len(self.env_capsules)]
            endpoints.append([np.concatenate([cap["a"], cap["b"]]) for cap in env_world])
            lowest.append([min(cap["a"][2], cap["b"][2]) for cap in env_world])

        endpoints, lowest = np.array(endpoints), np.array(lowest)
        for k, cap in enumerate(self.env_capsules):
            cap["guard_table"] = bool(np.ptp(lowest[:, k]) > still_tol)
            cap["guard_objects"] = bool(np.ptp(endpoints[:, k], axis=0).max() > still_tol)

        return [pair for k, pair in enumerate(candidates)
                if h_min[k] < near and h_max[k] - h_min[k] > pair_fixed_tol]

    @staticmethod
    def _capsules_world(data, capsule_list):
        world = []
        for cap in capsule_list:
            body_id = cap["body_id"]
            body_pos = data.xpos[body_id]
            body_rot = data.xmat[body_id].reshape(3, 3)
            center = body_pos + body_rot @ cap["local_pos"]
            z_axis = body_rot @ cap["local_rot"][:, 2]
            half = cap["half_length"]
            world.append(dict(
                body_id=body_id, name=cap["name"],
                a=center - half * z_axis, b=center + half * z_axis,
                radius=cap["radius"],
                guard_table=cap.get("guard_table", True), guard_objects=cap.get("guard_objects", True),
            ))
        return world

    def update(self, data):
        """Self-collision capsules in world frame this step."""
        return self._capsules_world(data, self.capsules)

    def update_all(self, data):
        """Unfiltered capsules in world frame, for table/object checks."""
        return self._capsules_world(data, self.env_capsules)

    def link_pairs_world(self, data):
        """(capsule, capsule) pairs in world frame this step, for link-vs-link self-collision."""
        world = self._capsules_world(data, self.link_capsules)
        return [(world[i], world[j]) for i, j in self.link_pairs]


class EnvironmentGeometry:
    """Non-robot obstacles: the tabletop (fixed) and the t-block (moves)."""

    def __init__(self, model, table_geom_name="table_top", object_body_name="t_block"):
        # table is a fixed body with no joints above it, so body_pos is
        # already its world position -- no forward kinematics needed.
        table_body_id = model.body("table").id
        table_gid = model.geom(table_geom_name).id
        self.table_height = (
            model.body_pos[table_body_id][2] + model.geom_pos[table_gid][2] + model.geom_size[table_gid][2]
        )

        self.object_body_id = model.body(object_body_name).id
        self.object_boxes = []
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != self.object_body_id:
                continue
            self.object_boxes.append(dict(
                name=model.geom(gid).name or f"{object_body_name}_geom{gid}",
                local_pos=model.geom_pos[gid].copy(),
                local_rot=quat2mat(model.geom_quat[gid]),
                half_extents=model.geom_size[gid][:3].copy(),
            ))

    def update(self, data):
        """The block's boxes in world frame this step."""
        body_pos = data.xpos[self.object_body_id]
        body_rot = data.xmat[self.object_body_id].reshape(3, 3)
        world = []
        for box in self.object_boxes:
            world.append(dict(
                name=box["name"],
                center=body_pos + body_rot @ box["local_pos"],
                rot=body_rot @ box["local_rot"],
                half_extents=box["half_extents"],
            ))
        return world


class MovingObstacle:
    """The floating ellipsoid in scene.xml, swept back and forth along `axis` at constant `speed`:
    from `center` out to +amplitude, back through center to -amplitude, and so on. update(data)
    moves it (through its mocap pose) and returns it as the CBF sees it.

    The exact distance from a capsule to an ellipsoid has no closed form, so the CBF gets a capsule
    that encloses the ellipsoid instead: along the longest semi-axis a, radius = the middle
    semi-axis b, half-length a - b. Every cross-section of the ellipsoid fits inside, so keeping
    clear of the capsule keeps clear of the ellipsoid."""

    def __init__(self, model, center, axis=(0.0, 1.0, 0.0), amplitude=0.25, speed=0.3, body_name="obstacle"):
        body_id = model.body(body_name).id
        self.mocap_id = model.body_mocapid[body_id]
        geom_id = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id)
        semi_axes = model.geom_size[geom_id]
        order = np.argsort(semi_axes)[::-1]  # longest first
        self.long_axis_local = np.eye(3)[order[0]]
        self.radius = semi_axes[order[1]]
        self.half_length = semi_axes[order[0]] - semi_axes[order[1]]
        self.center = np.asarray(center, dtype=float)
        self.axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        self.amplitude = amplitude
        self.speed = speed

    def update(self, data):
        """Moves the obstacle to where it is at data.time; returns its capsule (a, b, radius) and
        its velocity vel, all world frame."""
        a = self.amplitude
        phase = (self.speed * data.time) % (4.0 * a)  # one cycle: 0 -> +a -> 0 -> -a -> 0
        if phase < a:
            offset, direction = phase, 1.0
        elif phase < 3.0 * a:
            offset, direction = 2.0 * a - phase, -1.0
        else:
            offset, direction = phase - 4.0 * a, 1.0
        pos = self.center + offset * self.axis
        data.mocap_pos[self.mocap_id] = pos

        long_axis = quat2mat(data.mocap_quat[self.mocap_id]) @ self.long_axis_local
        return dict(a=pos - self.half_length * long_axis, b=pos + self.half_length * long_axis,
                    radius=self.radius, vel=direction * self.speed * self.axis)

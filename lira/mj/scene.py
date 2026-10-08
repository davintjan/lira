"""Scene geometry: RobotGeometry tracks the arm's self-collision capsules; EnvironmentGeometry tracks
the tabletop and t-block. Both read model.* once at init and refresh world poses via update(data)."""
import hashlib
import itertools
import json
import pathlib

import mujoco
import numpy as np

from core.geometry import quat2mat, capsuleCapsuleDistanceBatch


class RobotGeometry:
    """Self-collision capsules: MuJoCo's real collision geoms (group 3) -- a sphere is a capsule
    with half-length 0 -- plus an optional synthetic base capsule for robots whose base ships
    without one. Also which clearances are worth guarding at all (see _sample_hazards).

    The robot-specific parts are arguments (arm_sim.ROBOTS holds them per robot):
      excluded_bodies -- bodies the ee sits rigidly off (the last wrist links, the pusher rod) by
                         construction, so guarding them for the ee's self-collision would just fight
                         the arm's own fixed geometry
      base_capsule    -- dict(body_name, local_pos, local_quat, radius, half_length), or None
    cache_dir, cache_name: where to keep _sample_hazards' result between runs (None = recompute every
    time). It depends only on the robot's geometry, so the file is named <cache_name>_<hash of
    everything the sampling reads> -- change the rod, a joint limit, a capsule or a sampling setting
    and the hash changes, so it's recomputed instead of reused stale."""

    def __init__(self, model, excluded_bodies=(), base_capsule=None, cache_dir=None, cache_name="robot"):
        # capsules: filtered (excludes excluded_bodies), used for self-collision.
        # env_capsules: unfiltered, used for table/object checks.
        self.capsules = []
        self.env_capsules = []
        excluded_bodies = set(excluded_bodies)

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
            if body_name not in excluded_bodies:
                self.capsules.append(entry)

        # base_synthetic: self-collision only -- it's fixed to the world, so
        # its table/object clearance can never change with q.
        base_entries = []
        if base_capsule is not None:
            base_entries.append(dict(
                body_id=model.body(base_capsule["body_name"]).id, name="base_synthetic",
                local_pos=np.asarray(base_capsule["local_pos"], dtype=float),
                local_rot=quat2mat(np.asarray(base_capsule["local_quat"], dtype=float)),
                radius=base_capsule["radius"], half_length=base_capsule["half_length"],
            ))
        self.capsules += base_entries

        # link_capsules: every capsule incl. base_synthetic; link_pairs index into it.
        self.link_capsules = self.env_capsules + base_entries
        self.link_pairs = self._cached_hazards(model, cache_dir, cache_name)

    _HAZARD_SETTINGS = dict(n_samples=2000, near=0.10, pair_fixed_tol=0.03, still_tol=1e-3, seed=0)
    _HAZARD_VERSION = 2  # bump whenever _sample_hazards' logic changes, so old cache files stop matching

    def _cached_hazards(self, model, cache_dir, cache_name):
        """_sample_hazards' result from cache_dir if it was computed for this exact geometry before;
        otherwise computes it and saves it there."""
        if cache_dir is None:
            return self._sample_hazards(model, **self._HAZARD_SETTINGS)
        path = pathlib.Path(cache_dir) / f"{cache_name}_{self._hazard_key(model)}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            for cap, table, objects in zip(self.env_capsules, cached["guard_table"], cached["guard_objects"]):
                cap["guard_table"], cap["guard_objects"] = table, objects
            return [tuple(pair) for pair in cached["link_pairs"]]
        pairs = self._sample_hazards(model, **self._HAZARD_SETTINGS)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(
            capsules=[cap["name"] for cap in self.link_capsules],  # for a human reading the file
            link_pairs=[[int(i), int(j)] for i, j in pairs],
            guard_table=[cap["guard_table"] for cap in self.env_capsules],
            guard_objects=[cap["guard_objects"] for cap in self.env_capsules],
        ), indent=1))
        return pairs

    def _hazard_key(self, model):
        """Hash of everything _sample_hazards reads: the kinematic tree (body offsets, joints and their
        limits), the xml's contact excludes, every capsule (incl. the rod and the synthetic base), the
        sampling settings and _HAZARD_VERSION."""
        h = hashlib.sha256()
        for array in (model.body_parentid, model.body_pos, model.body_quat, model.jnt_type, model.jnt_bodyid,
                      model.jnt_pos, model.jnt_axis, model.jnt_range[:model.nu], model.exclude_signature):
            h.update(np.ascontiguousarray(array).tobytes())
        for cap in self.link_capsules:
            h.update(np.array([cap["body_id"], *cap["local_pos"], *cap["local_rot"].ravel(),
                               cap["radius"], cap["half_length"]], dtype=float).tobytes())
        h.update(json.dumps(dict(self._HAZARD_SETTINGS, version=self._HAZARD_VERSION), sort_keys=True).encode())
        return h.hexdigest()[:12]

    def _sample_hazards(self, model, n_samples, near, pair_fixed_tol, still_tol, seed):
        """Decides once, by sampling random arm poses, which clearances are worth guarding -- so no
        check ever has to guess per pose. A clearance that never changes is fixed by design, not a
        collision; one that does change always counts, even at a pose where its gradient happens
        to be zero. Sets on each env capsule:
          guard_table   -- False if its lowest point never changes height (e.g. shoulder_link,
                           upper_arm's joint cylinder): no joint can bring it closer to the table.
          guard_objects -- False if it never moves at all (shoulder_link spins in place, like the
                           base): no joint can bring it closer to anything.
        Returns the link-vs-link capsule index pairs to guard: drops same-body and parent/child
        pairs (they meet at their joint by design), pairs the robot xml excludes from contact, pairs that never come within `near` m, and
        pairs whose clearance varies by less than `pair_fixed_tol` m -- those sit rigidly close by
        construction (e.g. upper_arm vs base), so guarding them would only block the arm."""
        caps = self.link_capsules
        # the robot xml's <contact><exclude> pairs: its author says these bodies don't really collide
        # (e.g. the iiwa's coarse spheres overlap across a joint where the real links don't)
        excluded = {frozenset((int(sig) >> 16, int(sig) & 0xFFFF)) for sig in model.exclude_signature}
        candidates = []
        for i, j in itertools.combinations(range(len(caps)), 2):
            bi, bj = caps[i]["body_id"], caps[j]["body_id"]
            if bi == bj or model.body_parentid[bi] == bj or model.body_parentid[bj] == bi:
                continue
            if frozenset((bi, bj)) in excluded:
                continue
            candidates.append((i, j))
        I, J = np.array(candidates, dtype=int).reshape(-1, 2).T

        # every capsule at once: center = body_pos + body_rot @ local_pos, axis = body_rot @ local z
        body = np.array([cap["body_id"] for cap in caps])
        local_pos = np.array([cap["local_pos"] for cap in caps])
        local_z = np.array([cap["local_rot"][:, 2] for cap in caps])
        half = np.array([cap["half_length"] for cap in caps])[:, None]
        radius = np.array([cap["radius"] for cap in caps])

        data = mujoco.MjData(model)
        lo, hi = model.jnt_range[:model.nu, 0], model.jnt_range[:model.nu, 1]
        rng = np.random.default_rng(seed)
        h_min = np.full(len(candidates), np.inf)
        h_max = np.full(len(candidates), -np.inf)
        n_env = len(self.env_capsules)
        endpoints, lowest = [], []
        for _ in range(n_samples):
            data.qpos[:model.nu] = rng.uniform(lo, hi)
            mujoco.mj_kinematics(model, data)
            rot = data.xmat[body].reshape(-1, 3, 3)
            center = data.xpos[body] + np.einsum("nij,nj->ni", rot, local_pos)
            axis = np.einsum("nij,nj->ni", rot, local_z)
            A, B = center - half * axis, center + half * axis
            h, _, _ = capsuleCapsuleDistanceBatch(A[I], B[I], radius[I], A[J], B[J], radius[J])
            np.minimum(h_min, h, out=h_min)
            np.maximum(h_max, h, out=h_max)
            endpoints.append(np.hstack([A[:n_env], B[:n_env]]))
            lowest.append(np.minimum(A[:n_env, 2], B[:n_env, 2]))

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

    def link_capsules_world(self, data):
        """Every capsule, incl. the synthetic base, in world frame -- e.g. to draw them."""
        return self._capsules_world(data, self.link_capsules)

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
    """The floating ellipsoid in world.xml, swept back and forth along `axis` at constant `speed`:
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

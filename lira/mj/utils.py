"""Viewer helpers: overlay what the planner and CBF actually see on the passive viewer."""
import mujoco
import numpy as np


def draw_capsules(viewer, world_capsules, rgba=(1.0, 0.15, 0.15, 0.35), reset=True):
    """Overlay capsules (dicts with a, b, radius in world frame -- RobotGeometry.update_all,
    MovingObstacle.update, ...) on the passive viewer: the keep-out shapes collisionQP guards.
    A capsule with a == b is a sphere (e.g. the iiwa's collision geoms) and is drawn as one.
    reset=True clears earlier overlays first, so calling it every frame doesn't pile them up;
    reset=False adds to them (draw several sets in different colours)."""
    scn = viewer.user_scn
    if reset:
        scn.ngeom = 0
    rgba = np.array(rgba, dtype=np.float32)
    for cap in world_capsules:
        if scn.ngeom >= scn.maxgeom:
            break
        geom = scn.geoms[scn.ngeom]
        a, b = np.asarray(cap["a"], dtype=float), np.asarray(cap["b"], dtype=float)
        if np.linalg.norm(b - a) < 1e-9:
            mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([cap["radius"], 0, 0]),
                                pos=a, mat=np.eye(3).flatten(), rgba=rgba)
        else:
            mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=np.zeros(3),
                                pos=np.zeros(3), mat=np.eye(3).flatten(), rgba=rgba)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, cap["radius"], a, b)
        scn.ngeom += 1

"""Overlay helpers: draw extra geoms (what the planner and CBF see, the policy's targets) into an
MjvScene -- the passive viewer's user_scn, or a Recorder's renderer.scene. They only add geoms; to
redraw the viewer's overlays each frame, set viewer.user_scn.ngeom = 0 first."""
import mujoco
import numpy as np


def draw_sphere(scn, pos, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([radius, 0, 0]),
                        pos=np.asarray(pos, dtype=float), mat=np.eye(3).flatten(), rgba=np.array(rgba, dtype=np.float32))
    scn.ngeom += 1


def draw_capsules(scn, world_capsules, rgba=(1.0, 0.15, 0.15, 0.35)):
    """World-frame capsules (a core.geometry.Capsules set: RobotGeometry.world's, MovingObstacle.update's):
    the keep-out shapes cbf_qp guards. A capsule with a == b is a sphere (e.g. the iiwa's collision
    geoms) and is drawn as one."""
    for a, b, radius in zip(world_capsules.a, world_capsules.b, world_capsules.radius):
        if np.linalg.norm(b - a) < 1e-9:
            draw_sphere(scn, a, radius, rgba)
            continue
        if scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=np.zeros(3),
                            pos=np.zeros(3), mat=np.eye(3).flatten(), rgba=np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
        scn.ngeom += 1

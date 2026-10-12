"""Per-robot settings: everything that differs between arms lives here. Nothing else knows which arm
it is driving -- the joint count comes from the model (n_dof = model.nu).

Adding a robot: vendor it (scenes/generate_vendor_links.py), write scenes/scene_<robot>.xml, add an
entry to ROBOTS."""
import pathlib
from dataclasses import dataclass

SCENE_DIR = pathlib.Path(__file__).resolve().parent.parent / "scenes"  # scene_<robot>.xml, world.xml and the vendored robots


@dataclass(frozen=True)
class RobotConfig:
    name: str
    scene: str                    # file in SCENE_DIR: the robot + world.xml
    excluded_bodies: tuple        # the ee sits rigidly off these by construction (see RobotGeometry)
    base_capsule: dict | None     # synthetic self-collision capsule for a base that ships without one
    start_key: str | None         # keyframe in the robot xml to start from; None = qpos0
    base_pos: tuple               # m, world frame: where the base body is mounted on the table
    k_p: float = 1.0              # 1/s: position DS gain (straight-line fallback)
    k_o: float = 1.0              # 1/s: orientation DS gain
    path_k_tangent: float = 4.0   # rad/s: nominal joint-space speed along the planned path
    path_k_corrective: float = 4.0  # rad/s per rad: pull-back gain toward the path when off it


ROBOTS = {
    "ur5e": RobotConfig(
        name="ur5e",
        scene="scene_ur5e.xml",
        excluded_bodies=("wrist_2_link", "wrist_3_link", "pusher"),
        # ur5e.xml's base has no collision geom: fitted by hand to the base_0/base_1 mesh vertices
        # (z in [0, 0.099], radius 0.076) -- NOT the base-to-shoulder offset (0.163), which overlaps
        # shoulder_link's capsule
        base_capsule=dict(body_name="base", local_pos=[0.0, 0.0, 0.0495], local_quat=[1.0, 0.0, 0.0, 0.0],
                          radius=0.076, half_length=0.0495),
        start_key=None,  # qpos0: all joints 0, arm stretched out
        base_pos=(0.0, 0.0, 0.0),
    ),
    "iiwa14": RobotConfig(
        name="iiwa14",
        scene="scene_iiwa14.xml",
        excluded_bodies=("link6", "link7", "pusher"),
        base_capsule=None,  # iiwa14.xml's base already has collision spheres
        start_key="home",   # qpos0 is straight up -- joints 1/3/5/7 aligned, a singularity
        # pulled back from the origin: with the rod pointing down its wrist centre rides ~0.28 m above
        # the probe, and at the origin the near half of the Push-T workspace needs more elbow (A4)
        # bend than its +-120 deg allows. -0.2 reaches the most of the workspace (65/81 grid points vs 45)
        base_pos=(-0.2, 0.0, 0.0),
    ),
}
DEFAULT_ROBOT = "iiwa14"

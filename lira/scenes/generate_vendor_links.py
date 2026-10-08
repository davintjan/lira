"""
Regenerates two symlink trees per robot inside this repo (_vendor/<robot>/<robot>.xml and
assets/_vendor/<robot>/<meshfile>) that the scene_*.xml files include instead of
../../mujoco_menagerie directly. One entry per robot in ROBOTS.

Why: MuJoCo (confirmed empirically on 3.13.0, undocumented) only applies an
included file's <compiler meshdir> when that file is reached via a path
INSIDE the root's own directory tree -- a cross-directory "../..." include
can't find its meshes. These symlinks make the robot xml reachable from inside
the tree without ever modifying mujoco_menagerie/ itself.

Usage: python generate_vendor_links.py
"""
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
MENAGERIE = HERE.parent.parent / "mujoco_menagerie"  # lira/scenes/ -> repo root
# menagerie folder -> its robot xml; each folder must be in the submodule's sparse checkout
ROBOTS = {
    "universal_robots_ur5e": "ur5e.xml",
    "kuka_iiwa_14": "iiwa14.xml",
}


def relink(out_dir, name, target):
    """Symlink out_dir/name -> target, stored relative so the tree survives being moved or cloned elsewhere."""
    out_dir.mkdir(parents=True, exist_ok=True)
    link = out_dir / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target, out_dir))


def link_robot(folder, xml_name):
    src = MENAGERIE / folder
    xml_link_dir = HERE / "_vendor" / folder
    mesh_link_dir = HERE / "assets" / "_vendor" / folder
    src_xml = src / xml_name
    mesh_files = sorted(set(re.findall(r'<mesh\s+file="([^"]+)"', src_xml.read_text())))

    for out_dir in (xml_link_dir, mesh_link_dir):
        if out_dir.exists():
            shutil.rmtree(out_dir)

    relink(xml_link_dir, xml_name, src_xml)
    for name in mesh_files:
        relink(mesh_link_dir, name, src / "assets" / name)
    print(f"{folder}: linked {xml_name} and {len(mesh_files)} mesh files")


def main():
    for folder, xml_name in ROBOTS.items():
        link_robot(folder, xml_name)


if __name__ == "__main__":
    main()

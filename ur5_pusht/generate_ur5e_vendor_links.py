"""
Regenerates two symlink trees inside this repo (_vendor/.../ur5e.xml and
assets/_vendor/.../<meshfile>) that scene.xml includes instead of
../mujoco_menagerie directly.

Why: MuJoCo (confirmed empirically on 3.13.0, undocumented) only applies an
included file's <compiler meshdir> when that file is reached via a path
INSIDE the root's own directory tree -- a cross-directory "../..." include
can't find its meshes. These symlinks make ur5e.xml reachable from inside
the tree without ever modifying mujoco_menagerie/ itself.

Usage: python generate_ur5e_vendor_links.py
"""
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "mujoco_menagerie" / "universal_robots_ur5e"
XML_LINK_DIR = HERE / "_vendor" / "universal_robots_ur5e"
MESH_LINK_DIR = HERE / "assets" / "_vendor" / "universal_robots_ur5e"


def relink(out_dir, name, target):
    out_dir.mkdir(parents=True, exist_ok=True)
    link = out_dir / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def main():
    src_xml = SRC / "ur5e.xml"
    mesh_files = sorted(set(re.findall(r'<mesh\s+file="([^"]+)"', src_xml.read_text())))

    for out_dir in (XML_LINK_DIR, MESH_LINK_DIR):
        if out_dir.exists():
            shutil.rmtree(out_dir)

    relink(XML_LINK_DIR, "ur5e.xml", "../../../mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
    for name in mesh_files:
        relink(MESH_LINK_DIR, name, f"../../../../mujoco_menagerie/universal_robots_ur5e/assets/{name}")

    print(f"Linked ur5e.xml into {XML_LINK_DIR}")
    print(f"Linked {len(mesh_files)} mesh files into {MESH_LINK_DIR}")


if __name__ == "__main__":
    main()

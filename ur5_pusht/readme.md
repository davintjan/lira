How to start the mujoco viewer with the scene.xml

cd ur5_pusht_scene
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # first time only
.venv/bin/python generate_ur5e_vendor_links.py                        # first time, and after updating mujoco_menagerie
.venv/bin/python -m mujoco.viewer --mjcf=scene.xml
# diff_pol

Hermetic layout: two vendored dependencies as untouched (or minimally, visibly
patched) git submodules, plus our own code as plain sibling directories that
depend on them without editing them in place.

```
diffusion_policy/     fork of real-stanford/diffusion_policy (submodule)
mujoco_menagerie/     fork of google-deepmind/mujoco_menagerie (submodule),
                      sparse-checked-out to universal_robots_ur5e/ only
pusht_simple/         our train/eval scripts + requirements.txt, depends on
                      ../diffusion_policy via sys.path (see comments in the
                      scripts / requirements.txt for why not pip install -e)
lira/                 LiRA (Lightweight Reactive Arm control): an offline planner
                      (gradient descent or SQP) turned into a stable velocity field,
                      filtered live by a CBF; UR5e MuJoCo sims, incl. one running
                      the Push-T policy (see lira/readme.md)
```

## Setup from a fresh clone

```
git submodule update --init --recursive
```

`mujoco_menagerie` is sparse-checked-out to just `universal_robots_ur5e/`
(menagerie contains many other robots we don't need). That sparse-checkout
config lives in the submodule's local `.git/info/sparse-checkout`, which is
not part of git history, so it has to be set again after the submodule
clones in:

```
cd mujoco_menagerie
git sparse-checkout init --cone
git sparse-checkout set universal_robots_ur5e
```

## diffusion_policy fork patch

`diffusion_policy` carries exactly one commit on top of upstream: a fix to
`diffusion_policy/model/common/lr_scheduler.py`'s imports for modern
`diffusers` (see that commit's message for details). Nothing else in that
submodule is modified.

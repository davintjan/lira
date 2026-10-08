# LiRA — Lightweight Reactive Arm control

Plan once offline, then react live. LiRA plans a collision-free joint-space path for a robot
arm (gradient descent or trust-region SQP), turns it into a stable velocity field, and filters
that field every control tick with a CBF-QP against self-collision, the table, objects and
moving obstacles. It runs on a UR5e in MuJoCo.

The code, and how to run it, is in [lira/](lira/readme.md).

```
git clone --recursive git@github.com:davintjan/lira.git
```

## Upcoming work

Combining LiRA with diffusion policy, so that LiRA can serve a learned policy's actions
(starting with the Push-T policy in `pusht_simple/`) under the same planning and CBF safety
layer.

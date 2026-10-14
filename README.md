# LiRA — Lightweight Reactive Arm control

Plan once offline, then react live. LiRA plans a collision-free joint-space path for a robot
arm (gradient descent or trust-region SQP), turns it into a stable velocity field, and filters
that field every control tick with a CBF-QP against self-collision, the table, objects and
moving obstacles. It runs on a UR5e and a KUKA iiwa14 in MuJoCo.

The code, and how to run it, is in [lira/](lira/readme.md).

```
git clone --recursive git@github.com:davintjan/lira.git
```

## Push-T demo

| | Front | Top |
|---|---|---|
| iiwa14 | <img src="lira/videos/iiwa_pusht_front.webp" width="400"> | <img src="lira/videos/iiwa_pusht_top.webp" width="400"> |
| UR5e | <img src="lira/videos/ur_pusht_front.webp" width="400"> | <img src="lira/videos/ur_pusht_top.webp" width="400"> |

In these videos the arm plays Push-T with the diffusion policy checkpoint from the original paper,
used as is with no retraining. Each run starts with the arm away from the table, so we first plan a
path with SQP to bring it into the workspace. Once it gets there, the policy takes over.

While the policy is running, its actions go through a DS-based controller. The controller follows
what the policy wants to do and also moves the arm out of the way of the obstacle. The robot only
knows where the obstacle is at the current moment. It doesn't know where the obstacle is heading or
how fast it's moving, so all of the dodging is a reaction to where the obstacle is right now.

This isn't vision-based yet. The robot gets the exact pose of the T straight from the simulator
instead of estimating it from a camera.

## Dual-arm Push-T demo

| Front | Top |
|---|---|
| <img src="lira/videos/dual_pusht_front.webp" width="400"> | <img src="lira/videos/dual_pusht_top.webp" width="400"> |

Here an iiwa14 and a UR5e face each other across one table, and each plays Push-T on its own board:
the iiwa14 pushes the red T and the UR5e the blue one. The two boards overlap in the middle, where both
Ts start, so the arms push in the same space while an obstacle sweeps between them.

Each arm runs its own copy of the same checkpoint, and neither policy knows the other arm exists. What
keeps the arms apart is the CBF: each one treats the other arm's links as obstacles at wherever they are
right now. Each arm only limits its own part of how fast the gap between them closes, so the arms never
need to share their plans or velocities.

Here the policy samples with DDIM in 10 steps instead of the checkpoint's original 100-step DDPM, using
the same weights. Running two policies at 10 Hz on one laptop GPU is too slow with DDPM.

"""
A Hydra-free, single-file version of train_diffusion_unet_lowdim_workspace.py
for the Push-T (state/keypoint) task.

No yaml files, no @hydra.main, no _target_ string lookups. Every object below
is built with a plain constructor call, so you can read this file top to
bottom and see exactly what runs, in what order, with what values.

The hyperparameters below are copied from:
  - diffusion_policy/config/train_diffusion_unet_lowdim_workspace.yaml
  - diffusion_policy/config/task/pusht_lowdim.yaml
so a run of this script should behave the same as:
  python train.py --config-name=train_diffusion_unet_lowdim_workspace

What's deliberately simplified vs. the original workspace:
  - No wandb / JsonLogger / TopKCheckpointManager -- just print() and a
    single "latest.ckpt" file, overwritten every `--checkpoint_every` epochs.
  - EMA uses this repo's own diffusion_policy.model.diffusion.ema_model.EMAModel
    (not diffusers.training_utils.EMAModel, whose constructor signature changed
    across diffusers versions), with the same warmup schedule as the original
    workspace config (power=0.75, max_value=0.9999).
  - The full simulated rollout eval (PushTKeypointsRunner) runs every
    `--rollout_every` epochs (default 50, same cadence as checkpointing) and
    prints test_mean_score alongside train/val loss. Pass --no_eval_rollout
    to skip it and keep only the cheap train/val loss forward passes.

Usage:
    python train_pusht_simple.py --data data/pusht/pusht_cchi_v7_replay.zarr
"""
import argparse
import copy
import os
import sys
import time
from pathlib import Path

# This script lives in a sibling directory of the vendored diffusion_policy
# fork (../diffusion_policy), which isn't pip-installed -- its own setup.py
# predates PEP 420 namespace packages and find_packages() picks up nothing.
# Point sys.path at its repo root directly instead of touching that repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import tqdm

from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.common.pytorch_util import dict_apply


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/pusht/pusht_cchi_v7_replay.zarr",
                    help="path to the pusht_cchi_v7_replay.zarr dataset (from pusht.zip)")
    p.add_argument("--output_dir", default="data/outputs/pusht_simple")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--eval_rollout", action="store_true", default=True,
                    help="also run real PushT simulator rollouts periodically (on by default)")
    p.add_argument("--no_eval_rollout", dest="eval_rollout", action="store_false",
                    help="disable periodic simulator rollouts, keep only train/val loss")
    p.add_argument("--rollout_every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"No dataset at {args.data}. Download it with:\n"
            f"  mkdir -p data && cd data && "
            f"wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip && "
            f"unzip pusht.zip"
        )

    # ---- hyperparameters (copied from the two yaml files, see module docstring) ----
    horizon = 16
    n_obs_steps = 2
    n_action_steps = 8
    obs_dim = 20      # 9 keypoints * 2 + 2D agent position
    action_dim = 2
    num_train_timesteps = 100

    device = torch.device(args.device)

    # ---- data ----
    dataset = PushTLowdimDataset(
        zarr_path=args.data,
        horizon=horizon,
        pad_before=n_obs_steps - 1,
        pad_after=n_action_steps - 1,
        seed=42,
        val_ratio=0.02,
        max_train_episodes=90,
    )
    val_dataset = dataset.get_validation_dataset()
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=1, pin_memory=True)
    normalizer = dataset.get_normalizer()
    print(f"train batches/epoch: {len(train_loader)}, val batches/epoch: {len(val_loader)}")

    # ---- model: this is the actual DDPM from the paper ----
    # epsilon_theta(O_t, A^k_t, k) -- the noise-prediction network (Eq. 4/5 in the paper)
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,          # predicts noise in *action* space only
        local_cond_dim=None,
        global_cond_dim=obs_dim * n_obs_steps,   # observations enter as FiLM conditioning
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
    )
    # alpha, gamma, sigma schedule from Sec III-C of the paper (squared-cosine schedule)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon",
    )
    policy = DiffusionUnetLowdimPolicy(
        model=noise_pred_net,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=num_train_timesteps,
        obs_as_global_cond=True,
        oa_step_convention=True,
    )
    # set_normalizer() replaces the normalizer's internal buffers wholesale with
    # freshly-cloned CPU tensors (see DictOfTensorMixin._load_from_state_dict), so
    # it MUST run before .to(device), or those buffers silently stay on CPU while
    # the rest of the model is on GPU -> device-mismatch crash during training.
    policy.set_normalizer(normalizer)
    policy = policy.to(device)

    # ---- optimizer / lr schedule ----
    optimizer = AdamW(policy.parameters(), lr=1e-4, betas=(0.95, 0.999),
                       eps=1e-8, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * args.epochs,
    )

    # ---- EMA: a slowly-drifting copy of the weights, used at eval/checkpoint time.
    # Plain DDPM training is noisy step-to-step; averaging weights over time gives a
    # much more stable policy to actually evaluate/deploy, at basically zero cost.
    #
    # Uses the same warmup schedule as the original workspace (see
    # diffusion_policy/model/diffusion/ema_model.py): decay ramps up from ~0 toward
    # max_value=0.9999 as 1 - (1+step)^-power, instead of being fixed at 0.9999 from
    # step 0. A *fixed* 0.9999 decay from step 0 barely moves away from the random
    # initialization for the first several thousand steps (0.9999^6000 ~= 0.55), so
    # early training deceptively looks fine on train/val loss while the EMA weights
    # actually evaluated/checkpointed are still ~half random noise. ----
    ema_model = copy.deepcopy(policy)  # separate weight buffer the EMA is averaged into
    ema = EMAModel(model=ema_model, update_after_step=0, inv_gamma=1.0,
                   power=0.75, min_value=0.0, max_value=0.9999)

    # ---- optional: real simulator rollouts (success rate), same object the
    # original workspace uses. Off by default because it's the slow/heavy part. ----
    env_runner = None
    if args.eval_rollout:
        from diffusion_policy.env_runner.pusht_keypoints_runner import PushTKeypointsRunner
        env_runner = PushTKeypointsRunner(
            output_dir=args.output_dir,
            n_train=6, n_train_vis=2, train_start_seed=0,
            n_test=50, n_test_vis=4, legacy_test=True, test_start_seed=100000,
            max_steps=300, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
            fps=10,
        )

    # ============================= training loop =============================
    global_step = 0
    for epoch in range(args.epochs):
        policy.train()
        t0 = time.time()
        train_losses = []
        for batch in tqdm.tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            loss = policy.compute_loss(batch)   # the DDPM loss: MSE(noise, eps_theta(...))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ema.step(policy)

            train_losses.append(loss.item())
            global_step += 1

        train_loss = sum(train_losses) / len(train_losses)
        msg = f"[epoch {epoch}] train_loss={train_loss:.5f} ({time.time()-t0:.1f}s)"

        # ---- validation loss ----
        if epoch % args.val_every == 0:
            policy.eval()
            with torch.no_grad():
                val_losses = [
                    policy.compute_loss(dict_apply(b, lambda x: x.to(device))).item()
                    for b in val_loader
                ]
            msg += f" val_loss={sum(val_losses)/len(val_losses):.5f}"

        # ---- optional: real simulator success rate, evaluated on the EMA weights
        # (ema_model is a standalone policy instance kept in eval() by EMAModel, so
        # no need to rebuild/reload a separate copy here). ----
        if env_runner is not None and epoch % args.rollout_every == 0:
            runner_log = env_runner.run(ema_model)
            for k, v in runner_log.items():
                if "mean_score" in k:
                    msg += f" {k}={v:.3f}"

        print(msg)

        # ---- checkpoint ----
        if epoch % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.output_dir, "latest_1.ckpt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": policy.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()

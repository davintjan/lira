"""
Standalone rollout eval for checkpoints saved by train_pusht_simple.py.

train_pusht_simple.py's checkpoint format ({"model_state_dict", "ema_state_dict",
"epoch", ...}) has no `cfg`/`_target_`, so the repo's eval.py (which expects a
Hydra workspace checkpoint) can't load it. This script rebuilds the exact same
policy architecture used in train_pusht_simple.py, loads the EMA weights from a
plain checkpoint, and runs the real PushT simulator rollout (PushTKeypointsRunner)
against it -- i.e. it computes the test_mean_score that --eval_rollout would have
logged during training, after the fact.

Usage:
    python eval_pusht_simple.py --checkpoint data/outputs/pusht_simple/latest.ckpt
"""
import argparse
import json
import os
import sys
from pathlib import Path

# See train_pusht_simple.py for why this is needed instead of a pip install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy"))

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.env_runner.pusht_keypoints_runner import PushTKeypointsRunner


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="data/outputs/pusht_simple/latest.ckpt")
    p.add_argument("--output_dir", default="data/outputs/pusht_simple/eval")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", dest="use_ema", action="store_false",
                   help="evaluate the raw (non-EMA) weights instead")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- same hyperparameters as train_pusht_simple.py ----
    horizon = 16
    n_obs_steps = 2
    n_action_steps = 8
    obs_dim = 20
    action_dim = 2
    num_train_timesteps = 100

    device = torch.device(args.device)

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        local_cond_dim=None,
        global_cond_dim=obs_dim * n_obs_steps,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
    )
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

    print(f"loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  epoch={ckpt.get('epoch')} global_step={ckpt.get('global_step')}")
    key = "ema_state_dict" if args.use_ema else "model_state_dict"
    policy.load_state_dict(ckpt[key])
    policy = policy.to(device)
    policy.eval()

    env_runner = PushTKeypointsRunner(
        output_dir=args.output_dir,
        n_train=6, n_train_vis=2, train_start_seed=0,
        n_test=50, n_test_vis=4, legacy_test=True, test_start_seed=100000,
        max_steps=300, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
        fps=10,
    )
    runner_log = env_runner.run(policy)

    scores = {k: v for k, v in runner_log.items() if "mean_score" in k}
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    json_log = {k: (v if not hasattr(v, "_path") else v._path) for k, v in runner_log.items()}
    out_path = os.path.join(args.output_dir, "eval_log.json")
    with open(out_path, "w") as f:
        json.dump(json_log, f, indent=2, sort_keys=True, default=str)
    print(f"full log -> {out_path}")


if __name__ == "__main__":
    main()

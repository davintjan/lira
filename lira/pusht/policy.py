"""The Push-T lowdim diffusion policy: loading a checkpoint, and running it the way a real robot would
-- in a background thread, so the sim keeps going while it thinks."""
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import dill
import numpy as np
import torch
from omegaconf import OmegaConf

# the policy code lives in the diffusion_policy fork, used in place (see pusht_simple/ for why not pip install)
DIFFUSION_POLICY_DIR = pathlib.Path(__file__).resolve().parents[2] / "diffusion_policy"
sys.path.insert(0, str(DIFFUSION_POLICY_DIR))
from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # noqa: E402
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # noqa: E402
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D  # noqa: E402
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy  # noqa: E402

REFERENCE_CHECKPOINT = DIFFUSION_POLICY_DIR / "data/outputs/reference/epoch=0550-test_mean_score=0.969.ckpt"


def load_policy(checkpoint, device, ddim_steps=None):
    """The Push-T lowdim diffusion policy from either checkpoint format:
      - the original repo's Hydra workspace checkpoint (train.py): built from its own saved config
        (e.g. the reference one feeds observations by inpainting, obs_as_global_cond=False)
      - train_pusht_simple.py's plain checkpoint: no config inside, so the architecture is rebuilt
        exactly as eval_pusht_simple.py does (observations as global conditioning)
    EMA weights either way; they include the observation/action normalizer.
    ddim_steps: sample with DDIM in this many denoising steps instead of the checkpoint's DDPM (100):
    same weights, ~10x faster -- what the Diffusion Policy paper does on its real robots to keep up
    with 10 Hz. None keeps DDPM."""
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    if "cfg" in payload:
        cfg = OmegaConf.to_container(payload["cfg"].policy, resolve=True)
        strip = lambda d: {k: v for k, v in d.items() if k != "_target_"}  # noqa: E731
        policy = DiffusionUnetLowdimPolicy(
            model=ConditionalUnet1D(**strip(cfg["model"])),
            noise_scheduler=DDPMScheduler(**strip(cfg["noise_scheduler"])),
            **{k: cfg[k] for k in ("horizon", "obs_dim", "action_dim", "n_action_steps", "n_obs_steps",
                                   "num_inference_steps", "obs_as_local_cond", "obs_as_global_cond",
                                   "pred_action_steps_only", "oa_step_convention")},
        )
        state = payload["state_dicts"]["ema_model"]
    else:
        obs_dim, action_dim, n_obs_steps = 20, 2, 2
        policy = DiffusionUnetLowdimPolicy(
            model=ConditionalUnet1D(input_dim=action_dim, local_cond_dim=None, global_cond_dim=obs_dim * n_obs_steps,
                                    diffusion_step_embed_dim=256, down_dims=[256, 512, 1024], kernel_size=5,
                                    n_groups=8, cond_predict_scale=True),
            noise_scheduler=DDPMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                                          beta_schedule="squaredcos_cap_v2", variance_type="fixed_small",
                                          clip_sample=True, prediction_type="epsilon"),
            horizon=16, obs_dim=obs_dim, action_dim=action_dim, n_action_steps=8, n_obs_steps=n_obs_steps,
            num_inference_steps=100, obs_as_global_cond=True, oa_step_convention=True,
        )
        state = payload["ema_state_dict"]
    policy.load_state_dict(state)
    if ddim_steps is not None:
        policy.noise_scheduler = DDIMScheduler(**{k: v for k, v in policy.noise_scheduler.config.items()
                                                  if k in ("num_train_timesteps", "beta_start", "beta_end",
                                                           "beta_schedule", "clip_sample", "prediction_type")})
        policy.num_inference_steps = ddim_steps
    return policy.to(device).eval()


class ChunkedPolicy:
    """Runs the policy asynchronously and hands out its action chunk one action per `period`.

    Call once per period, in this order:
      observe(obs)        the latest observation (Push-T pads the first one by repeating it)
      collect(t)          picks up a finished chunk, if any, and drops its actions already in the past
      next_target(t)      the current target; starts inferring the next chunk once `prefetch` or
                          fewer actions are left, so it's ready in time
    Like a real robot, a chunk lands late: its first action was meant for the tick its observation
    came from."""

    def __init__(self, policy, device, period=0.1, prefetch=2):
        self.policy = policy
        self.device = device
        self.period = period
        self.prefetch = prefetch
        self.history = []   # the last n_obs_steps observations
        self.queue = []     # actions still to come, Push-T px
        self.target = None  # the current action: kept until a new one arrives
        self._executor = ThreadPoolExecutor(max_workers=1)  # the sim loop never waits on it
        self._pending = None       # the chunk being inferred...
        self._pending_time = None  # ...and the sim time of the observation it's from

    def observe(self, obs):
        n = self.policy.n_obs_steps
        self.history = (self.history + [obs])[-n:] if self.history else [obs] * n

    def collect(self, t):
        """If a chunk finished: queues its actions from now on and returns how many periods late it
        landed. Otherwise None."""
        if self._pending is None or not self._pending.done():
            return None
        late = int(round((t - self._pending_time) / self.period))
        self.queue, self._pending = list(self._pending.result()[late:]), None
        return late

    def next_target(self, t):
        if self.queue:
            self.target = self.queue.pop(0)  # an empty queue keeps the last target until the chunk lands
        if self._pending is None and len(self.queue) <= self.prefetch:
            self._pending, self._pending_time = self._executor.submit(self._infer, np.stack(self.history)), t
        return self.target

    def stop(self):
        """Forget the queued actions."""
        self.queue = []

    def _infer(self, obs_stack):
        with torch.no_grad():  # thread-local in torch, so it goes here, not around submit()
            obs = torch.from_numpy(obs_stack[None]).to(self.device)
            return self.policy.predict_action({"obs": obs})["action"][0].cpu().numpy()

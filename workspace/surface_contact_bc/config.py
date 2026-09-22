"""Shared settings. Method files change only the component being ablated."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = (
    "action_bridge_contact_frame_full",
    "action_bridge_contact_frame_no_kl",
    "action_bridge_isotropic",
    "diffusion_world",
    "diffusion_contact_frame",
    "reference_only",
    "diffusion_residual",
)

DEFAULTS = dict(
    obs_history=2,
    action_history=2,
    horizon=8,
    n_exec=2,
    hidden_dim=128,
    h_emb_dim=128,
    time_emb_dim=32,
    unet_channels=[32, 64],
    num_train_timesteps=100,
    num_inference_steps=20,
    prediction_type="epsilon",
    beta_kl=0.001,
    lambda_unroll=1.0,
    sigma=1.0,
    lambda_q=1.0,
    lambda_ref_reg=0.0,
    gamma_min=0.0,
    gamma_max=0.8,
    stiffness_min=0.0,
    stiffness_max=0.5,
    offset_max=0.04,
    control_scale=1.0,
    steps=10000,
    batch_size=128,
    lr=0.001,
    weight_decay=0.000001,
    ema_decay=0.995,
    val_every=250,
    val_windows=256,
    log_every=50,
    seed=0,
    train_episodes=None,
    device="cpu",
    threads=2,
)


def method_config(method):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    config = DEFAULTS | json.loads((ROOT / "configs" / f"{method}.json").read_text())
    return config | {"method": method}

"""Short-horizon variant of latent path-Sinkhorn delayed modes.

This ablation tests whether long chunks over-weight late near-goal behavior in
the path-level OT objective. It keeps the same dataset, model, and loss setup as
`latent_path_sinkhorn_delayed_modes.py`, but supervises shorter action paths.
"""

from __future__ import annotations

from action_bridge.configs.latent_path_sinkhorn_delayed_modes import get_config as get_base_config


def get_config():
    cfg = get_base_config()

    cfg.data.horizon = 10
    cfg.eval.replan_every = 2

    return cfg

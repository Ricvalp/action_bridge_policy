"""Gaussian-initialized residual bridge ablation."""

from __future__ import annotations

from action_bridge.configs import base


def get_config():
    cfg = base.get_config()
    cfg.model.type = "bridge"
    cfg.model.init_type = "gaussian"
    cfg.model.init_noise_scale = 0.6
    cfg.model.noise_dim = 0
    cfg.loss.bridge_weight = 0.03
    cfg.loss.jerk_weight = 0.002
    return cfg

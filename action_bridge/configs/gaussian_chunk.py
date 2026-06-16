"""Gaussian-to-action-chunk MLP baseline."""

from __future__ import annotations

from action_bridge.configs import base


def get_config():
    cfg = base.get_config()
    cfg.model.type = "chunk"
    cfg.model.noise_dim = 8
    cfg.model.noise_scale = 1.0
    cfg.loss.bridge_weight = 0.0
    cfg.loss.jerk_weight = 0.0
    return cfg

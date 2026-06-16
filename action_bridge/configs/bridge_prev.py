"""Previous-action initialized residual bridge with bridge path energy."""

from __future__ import annotations

from action_bridge.configs import base


def get_config():
    cfg = base.get_config()
    cfg.model.type = "bridge"
    cfg.model.init_type = "prev_action"
    cfg.model.noise_dim = 0
    cfg.loss.bridge_weight = 0.03
    cfg.loss.jerk_weight = 0.002
    return cfg

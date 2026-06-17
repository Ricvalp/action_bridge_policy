"""More ambiguous Sinkhorn bridge ablation without action-history conditioning."""

from __future__ import annotations

from action_bridge.configs.sinkhorn_bridge import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.data.context = 1
    cfg.model.use_context_actions = False
    cfg.model.particles = 16
    cfg.model.init_type = "gaussian"
    cfg.model.init_noise_scale = 0.8
    cfg.loss.sinkhorn_context_weight = 0.01
    cfg.eval.policy_sample = "sample"
    return cfg

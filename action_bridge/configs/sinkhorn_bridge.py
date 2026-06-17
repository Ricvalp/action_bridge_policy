"""Probabilistic CWG-style particle bridge with Sinkhorn marginal matching."""

from __future__ import annotations

from action_bridge.configs import base


def get_config():
    cfg = base.get_config()
    cfg.model.type = "sinkhorn_bridge"
    cfg.model.init_type = "prev_action"
    cfg.model.init_noise_scale = 0.35
    cfg.model.particles = 8
    cfg.model.tau = 0.35
    cfg.model.use_context_actions = True

    cfg.loss.action_weight = 0.0
    cfg.loss.endpoint_weight = 0.0
    cfg.loss.first_action_weight = 0.0
    cfg.loss.sinkhorn_weight = 1.0
    cfg.loss.bridge_weight = 0.08
    cfg.loss.jerk_weight = 0.0
    cfg.loss.mean_action_weight = 0.0
    cfg.loss.diversity_weight = 0.0
    cfg.loss.sinkhorn_epsilon = 0.05
    cfg.loss.sinkhorn_iterations = 35
    cfg.loss.sinkhorn_context_weight = 0.05
    cfg.loss.sinkhorn_intermediate_weight = 1.0
    cfg.loss.sinkhorn_endpoint_weight = 1.0

    cfg.eval.deterministic = False
    cfg.eval.policy_sample = "mean"
    return cfg

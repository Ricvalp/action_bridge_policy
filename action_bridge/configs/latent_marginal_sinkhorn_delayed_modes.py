"""Latent bridge trained only with per-timestep Sinkhorn marginals."""

from __future__ import annotations

from action_bridge.configs import base


def get_config():
    cfg = base.get_config()

    cfg.data.context = 6
    cfg.data.horizon = 18
    cfg.data.shared_prefix_steps = 6
    cfg.data.shared_prefix_speed = 0.55
    cfg.data.shared_prefix_target_x = 0.30
    cfg.data.paired_modes = True

    cfg.model.type = "latent_sinkhorn_bridge"
    cfg.model.init_type = "prev_action"
    cfg.model.init_noise_scale = 0.02
    cfg.model.particles = 24
    cfg.model.latent_dim = 8
    cfg.model.latent_init_scale = 1.0
    cfg.model.latent_limit = 2.0
    cfg.model.tau = 0.35
    cfg.model.use_context_actions = True

    cfg.loss.action_weight = 0.0
    cfg.loss.endpoint_weight = 0.0
    cfg.loss.first_action_weight = 0.0
    cfg.loss.mean_action_weight = 0.0
    cfg.loss.path_sinkhorn_weight = 0.0
    cfg.loss.sinkhorn_weight = 1.0
    cfg.loss.sinkhorn_epsilon = 0.05
    cfg.loss.sinkhorn_iterations = 35
    cfg.loss.sinkhorn_context_weight = 0.05
    cfg.loss.sinkhorn_intermediate_weight = 1.0
    cfg.loss.sinkhorn_endpoint_weight = 1.0
    cfg.loss.bridge_weight = 0.0
    cfg.loss.jerk_weight = 0.0
    cfg.loss.diversity_weight = 0.0

    cfg.train.batch_size = 64
    cfg.train.epochs = 12

    cfg.eval.deterministic = False
    cfg.eval.policy_sample = "sample"
    cfg.eval.replan_every = 2
    cfg.eval.multimodal_examples = 8
    cfg.eval.multimodal_samples = 24
    cfg.eval.marginal_examples = 4
    cfg.eval.marginal_samples = 24
    cfg.eval.marginal_time_slices = 6

    return cfg

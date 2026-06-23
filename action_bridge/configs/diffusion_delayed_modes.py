"""Context-conditioned diffusion policy on delayed top/bottom modes."""

from __future__ import annotations

from action_bridge.configs import base


def get_config():
    cfg = base.get_config()

    cfg.data.context = 6
    cfg.data.horizon = 10
    cfg.data.shared_prefix_steps = 12
    cfg.data.shared_prefix_speed = 0.28
    cfg.data.shared_prefix_target_x = 0.30
    cfg.data.paired_modes = True

    cfg.model.type = "diffusion"
    cfg.model.history_dim = 96
    cfg.model.hidden_dim = 192
    cfg.model.action_limit = 1.0
    cfg.model.use_context_actions = True
    cfg.model.diffusion_steps = 50
    cfg.model.diffusion_inference_steps = 50
    cfg.model.diffusion_schedule = "squaredcos_cap_v2"
    cfg.model.diffusion_beta_start = 1e-4
    cfg.model.diffusion_beta_end = 0.02
    cfg.model.diffusion_time_dim = 32
    cfg.model.diffusion_eval_samples = 24
    cfg.model.particles = 24

    cfg.loss.diffusion_weight = 1.0
    cfg.loss.action_weight = 0.0
    cfg.loss.endpoint_weight = 0.0
    cfg.loss.first_action_weight = 0.0
    cfg.loss.bridge_weight = 0.0
    cfg.loss.jerk_weight = 0.0
    cfg.loss.sinkhorn_weight = 0.0
    cfg.loss.path_sinkhorn_weight = 0.0
    cfg.loss.mean_action_weight = 0.0
    cfg.loss.diversity_weight = 0.0

    cfg.train.batch_size = 64
    cfg.train.epochs = 12

    cfg.eval.deterministic = False
    cfg.eval.policy_sample = "sample"
    cfg.eval.replan_every = 2
    cfg.eval.multimodal_examples = 8
    cfg.eval.multimodal_samples = 24

    return cfg

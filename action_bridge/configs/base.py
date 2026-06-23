"""Base config for execution-time action bridge experiments."""

from __future__ import annotations

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()

    cfg.run_name = ""
    cfg.seed = 7
    cfg.device = "auto"

    cfg.data = ConfigDict()
    cfg.data.num_trajectories = 1000
    cfg.data.trajectory_length = 72
    cfg.data.context = 6
    cfg.data.horizon = 16
    cfg.data.train_fraction = 0.8
    cfg.data.force_regenerate = False
    cfg.data.path = ""
    cfg.data.paired_modes = True
    cfg.data.shared_prefix_steps = 0
    cfg.data.shared_prefix_speed = 0.55
    cfg.data.shared_prefix_target_x = 0.30

    cfg.model = ConfigDict()
    cfg.model.type = "bridge"
    cfg.model.state_dim = 4
    cfg.model.action_dim = 2
    cfg.model.history_dim = 96
    cfg.model.hidden_dim = 192
    cfg.model.tau = 0.45
    cfg.model.init_type = "prev_action"
    cfg.model.init_noise_scale = 0.6
    cfg.model.noise_dim = 0
    cfg.model.noise_scale = 1.0
    cfg.model.action_limit = 1.0
    cfg.model.use_context_actions = True
    cfg.model.particles = 8
    cfg.model.latent_dim = 8
    cfg.model.latent_init_scale = 1.0
    cfg.model.latent_limit = 2.0
    cfg.model.diffusion_steps = 50
    cfg.model.diffusion_beta_start = 1e-4
    cfg.model.diffusion_beta_end = 0.02
    cfg.model.diffusion_time_dim = 32
    cfg.model.diffusion_eval_samples = 24

    cfg.train = ConfigDict()
    cfg.train.epochs = 12
    cfg.train.batch_size = 128
    cfg.train.lr = 5e-4
    cfg.train.weight_decay = 1e-4
    cfg.train.grad_clip = 5.0
    cfg.train.num_workers = 0

    cfg.logging = ConfigDict()
    cfg.logging.wandb = False
    cfg.logging.wandb_project = "action-bridge-policy"
    cfg.logging.wandb_entity = ""
    cfg.logging.wandb_group = ""
    cfg.logging.wandb_mode = "online"
    cfg.logging.log_every_steps = 50
    cfg.logging.path_plot_every_steps = 0
    cfg.logging.path_plot_examples = 4
    cfg.logging.path_plot_particles = 12
    cfg.logging.save_local_path_plots = True

    cfg.loss = ConfigDict()
    cfg.loss.action_weight = 1.0
    cfg.loss.endpoint_weight = 0.15
    cfg.loss.first_action_weight = 0.1
    cfg.loss.bridge_weight = 0.03
    cfg.loss.jerk_weight = 0.0
    cfg.loss.phi_final = 1.4
    cfg.loss.sinkhorn_weight = 1.0
    cfg.loss.sinkhorn_epsilon = 0.05
    cfg.loss.sinkhorn_iterations = 40
    cfg.loss.sinkhorn_context_weight = 0.05
    cfg.loss.sinkhorn_intermediate_weight = 1.0
    cfg.loss.sinkhorn_endpoint_weight = 1.0
    cfg.loss.path_sinkhorn_weight = 0.0
    cfg.loss.path_sinkhorn_epsilon = 0.10
    cfg.loss.path_sinkhorn_iterations = 35
    cfg.loss.path_context = "state"
    cfg.loss.path_context_weight = 1.0
    cfg.loss.mean_action_weight = 0.0
    cfg.loss.diversity_weight = 0.0
    cfg.loss.diffusion_weight = 1.0

    cfg.eval = ConfigDict()
    cfg.eval.rollout_episodes = 96
    cfg.eval.replan_every = 4
    cfg.eval.deterministic = True
    cfg.eval.policy_sample = "mean"
    cfg.eval.plot_examples = 8
    cfg.eval.multimodal_examples = 0
    cfg.eval.multimodal_samples = 32
    cfg.eval.marginal_examples = 0
    cfg.eval.marginal_samples = 32
    cfg.eval.marginal_time_slices = 6

    return cfg

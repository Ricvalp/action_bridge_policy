"""Shared ml_collections config factories."""

from __future__ import annotations

from ml_collections import ConfigDict


def _logging_config() -> ConfigDict:
    config = ConfigDict()
    config.log_every_steps = 25
    config.eval_every_steps = 100
    config.full_eval_every_steps = 5000
    config.full_eval_split = "val"
    config.full_eval_max_batches = 4
    config.full_eval_closed_loop_episodes = 32
    config.full_eval_num_samples = 16
    config.wandb = ConfigDict()
    config.wandb.enabled = False
    config.wandb.project = "action-bridge-policy"
    config.wandb.entity = None
    config.wandb.mode = "online"
    config.wandb.name = None
    config.wandb.group = None
    config.wandb.tags = []
    config.wandb.log_model = False
    return config


def _toy_eval_config() -> ConfigDict:
    config = ConfigDict()
    config.closed_loop = True
    config.closed_loop_episodes = 128
    config.closed_loop_plot_rollouts = 24
    config.generated_history_trajectory_fraction = 0.5
    config.generated_history_time_fraction = 0.5
    return config


def _toy_common(benchmark: str) -> ConfigDict:
    config = ConfigDict()
    config.seed = 0
    config.device = "cuda"
    config.benchmark = benchmark
    config.trajectory_len = 64
    config.chunk_horizon = 16
    config.obs_history = 2
    config.action_history = 2
    config.action_dim = 2
    config.obs_dim = 4
    config.output_dir = "outputs"
    config.logging = _logging_config()
    config.eval = _toy_eval_config()
    return config


def _toy_model(latent_type: str) -> ConfigDict:
    config = ConfigDict()
    config.policy_type = "action_bridge"
    config.hidden_dim = 128
    config.h_emb_dim = 128
    config.time_emb_dim = 32
    config.z_embed_dim = 32
    config.latent_type = latent_type
    if latent_type == "categorical":
        config.num_categories = 2
    elif latent_type == "continuous":
        config.z_dim = 4
        config.continuous_prior = "learned_conditional_gaussian"
    else:
        raise ValueError(f"Unsupported latent_type {latent_type!r}.")
    return config


def _toy_reference() -> ConfigDict:
    config = ConfigDict()
    config.type = "continuation"
    config.coordinate_mode = "raw_action"
    config.dt = 1.0
    config.alpha = 0.8
    config.sigma = 0.05
    config.learn_alpha = False
    config.learn_sigma = False
    config.control_is_whitened = True
    config.gamma_mode = "constant"
    config.gamma_const = 0.2
    config.gamma_min = 0.0
    config.gamma_max = 0.95
    config.potential_type = "none"
    config.stiffness_mode = "learned_diag"
    config.k_const = 0.0
    config.k_min = 0.0
    config.k_max = 2.0
    config.attractor_mode = "learned"
    config.time_emb_dim = 32
    config.hidden_dim = 128
    config.beta_kl = 1.0
    config.lambda_q = 1.0
    config.lambda_ref_reg = 0.0001
    config.lambda_m_smooth = 0.001
    config.deterministic_inference = True
    return config


def _toy_loss(latent_type: str) -> ConfigDict:
    config = ConfigDict()
    config.beta_R = 0.01
    config.beta_z_start = 0.0
    config.beta_z_end = 0.01
    config.beta_z_warmup_steps = 10000
    config.free_nats = 0.1
    config.tube_training = False
    config.tube_noise_std_start = 0.0
    config.tube_noise_std_end = 0.02
    config.tube_noise_warmup_steps = 5000
    if latent_type == "continuous":
        config.num_z_samples_train = 1
    return config


def _toy_optim() -> ConfigDict:
    config = ConfigDict()
    config.lr = 0.0003
    config.batch_size = 256
    config.max_steps = 100000
    config.grad_clip = 1.0
    return config


def _toy_inference() -> ConfigDict:
    config = ConfigDict()
    config.deterministic = True
    config.num_samples = 32
    config.latent_commitment = "chunk"
    return config


def toy_delayed_config(latent_type: str) -> ConfigDict:
    config = _toy_common("toy_delayed")
    config.data = ConfigDict()
    config.data.num_contexts = 256
    config.data.paired_fraction = 0.5
    config.data.obstacle_center = [0.5, 0.5]
    config.data.obstacle_radius = 0.13
    config.data.lane_margin = 0.08
    config.data.start_mean = [0.12, 0.5]
    config.data.start_jitter = [0.03, 0.08]
    config.data.goal_mean = [0.88, 0.5]
    config.data.goal_jitter = [0.03, 0.08]
    config.data.shared_prefix_steps = 8
    config.data.shared_prefix_target_x = 0.30
    config.data.action_noise_std = 0.005
    config.data.speed = 0.035
    config.data.train_absolute_actions = False
    config.data.env_accepts_absolute_actions = False
    config.model = _toy_model(latent_type)
    if latent_type == "categorical":
        config.model.z_dim = 4
    config.reference = _toy_reference()
    config.loss = _toy_loss(latent_type)
    if latent_type == "categorical":
        config.loss.beta_aux_mode_ce = 0.0
    config.optim = _toy_optim()
    config.inference = _toy_inference()
    return config


def toy_annular_config(latent_type: str) -> ConfigDict:
    config = _toy_common("toy_annular")
    config.data = ConfigDict()
    config.data.num_contexts = 256
    config.data.paired_fraction = 0.3
    config.data.obstacle_center = [0.5, 0.5]
    config.data.obstacle_radius = 0.15
    config.data.margin = 0.08
    config.data.r_min = 0.28
    config.data.r_max = 0.48
    config.data.min_start_goal_distance = 0.35
    config.data.require_interaction = True
    config.data.interaction_distance_threshold = 0.18
    config.data.p_min = 0.08
    config.data.temperature = 0.08
    config.data.speed_noise_std = 0.002
    config.data.train_absolute_actions = False
    config.data.env_accepts_absolute_actions = False
    config.model = _toy_model(latent_type)
    config.reference = _toy_reference()
    config.loss = _toy_loss(latent_type)
    config.optim = _toy_optim()
    config.inference = _toy_inference()
    return config


def pusht_lowdim_config(latent_type: str) -> ConfigDict:
    config = ConfigDict()
    config.seed = 0
    config.device = "cuda"
    config.benchmark = "pusht_lowdim"
    config.chunk_horizon = 16
    config.obs_history = 2
    config.action_history = 2
    config.action_dim = 2
    config.obs_dim = 20
    config.output_dir = "outputs"
    config.data = ConfigDict()
    config.data.backend = "auto"
    config.data.dataset_path = None
    config.data.train_fraction = 0.8
    config.data.val_fraction = 0.1
    config.data.obs_key = None
    config.data.action_key = None
    config.data.episode_ends_key = None
    config.data.max_episodes = None
    config.model = ConfigDict()
    config.model.policy_type = "action_bridge"
    config.model.hidden_dim = 512
    config.model.h_emb_dim = 512
    config.model.time_emb_dim = 32
    config.model.z_embed_dim = 64
    config.model.latent_type = latent_type
    if latent_type == "categorical":
        config.model.num_categories = 4
    elif latent_type == "continuous":
        config.model.z_dim = 8
        config.model.continuous_prior = "learned_conditional_gaussian"
    else:
        raise ValueError(f"Unsupported latent_type {latent_type!r}.")
    config.reference = ConfigDict()
    config.reference.type = "continuation"
    config.reference.coordinate_mode = "raw_action"
    config.reference.dt = 1.0
    config.reference.alpha = 0.8
    config.reference.sigma = 0.05
    config.reference.learn_alpha = True
    config.reference.learn_sigma = False
    config.reference.control_is_whitened = True
    config.reference.gamma_mode = "constant"
    config.reference.gamma_const = 0.2
    config.reference.gamma_min = 0.0
    config.reference.gamma_max = 0.95
    config.reference.potential_type = "none"
    config.reference.stiffness_mode = "learned_diag"
    config.reference.k_const = 0.0
    config.reference.k_min = 0.0
    config.reference.k_max = 2.0
    config.reference.attractor_mode = "learned"
    config.reference.time_emb_dim = 32
    config.reference.hidden_dim = 128
    config.reference.beta_kl = 1.0
    config.reference.lambda_q = 1.0
    config.reference.lambda_ref_reg = 0.0001
    config.reference.lambda_m_smooth = 0.001
    config.reference.deterministic_inference = True
    config.loss = ConfigDict()
    config.loss.beta_R = 0.005
    config.loss.beta_z_start = 0.0
    config.loss.beta_z_end = 0.001
    config.loss.beta_z_warmup_steps = 20000
    config.loss.free_nats = 0.05
    config.loss.tube_training = True
    config.loss.tube_noise_std_start = 0.0
    config.loss.tube_noise_std_end = 0.01
    config.loss.tube_noise_warmup_steps = 10000
    if latent_type == "continuous":
        config.loss.num_z_samples_train = 1
    config.optim = ConfigDict()
    config.optim.lr = 0.0002
    config.optim.batch_size = 256
    config.optim.max_steps = 200000
    config.optim.grad_clip = 1.0
    config.inference = ConfigDict()
    config.inference.deterministic = True
    config.inference.num_samples = 8
    config.inference.latent_commitment = "episode"
    config.inference.n_exec = 8
    config.logging = _logging_config()
    config.eval = ConfigDict()
    config.eval.batch_size = 256
    config.eval.offline_rollout_episodes = 64
    config.eval.n_exec = 8
    return config

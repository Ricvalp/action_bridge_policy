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

    cfg.train = ConfigDict()
    cfg.train.epochs = 12
    cfg.train.batch_size = 128
    cfg.train.lr = 5e-4
    cfg.train.weight_decay = 1e-4
    cfg.train.grad_clip = 5.0
    cfg.train.num_workers = 0

    cfg.loss = ConfigDict()
    cfg.loss.action_weight = 1.0
    cfg.loss.endpoint_weight = 0.15
    cfg.loss.first_action_weight = 0.1
    cfg.loss.bridge_weight = 0.03
    cfg.loss.jerk_weight = 0.0
    cfg.loss.phi_final = 1.4

    cfg.eval = ConfigDict()
    cfg.eval.rollout_episodes = 96
    cfg.eval.replan_every = 4
    cfg.eval.deterministic = True
    cfg.eval.plot_examples = 8

    return cfg

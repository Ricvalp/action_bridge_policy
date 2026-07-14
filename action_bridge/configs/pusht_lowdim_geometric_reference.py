"""Push-T fixed geometric reference with a small residual policy."""

from action_bridge.configs.base import pusht_lowdim_config


def get_config():
    config = pusht_lowdim_config("continuous")
    config.obs_dim = 5
    config.action_dim = 2
    config.model.latent_type = "none"
    config.model.hidden_dim = 256
    config.model.h_emb_dim = 256
    config.model.encoder_depth = 2
    config.model.control_depth = 3
    config.model.time_emb_dim = 32
    config.model.z_embed_dim = 0
    if "z_dim" in config.model:
        del config.model.z_dim
    config.model.control_scale = 1.0

    config.reference.type = "geometric_pusht"
    config.reference.coordinate_mode = "absolute_action"
    config.reference.dt = 1.0
    config.reference.sigma = 3.0
    config.reference.control_is_whitened = True
    config.reference.beta_kl = 0.003
    config.reference.lambda_q = 1.0
    config.reference.target_pose = [256.0, 256.0, 0.7853981633974483]
    config.reference.n_per_edge = 16
    config.reference.pusher_radius = 5.0
    config.reference.delta_pre = 8.0
    config.reference.delta_push_far = 7.0
    config.reference.delta_push_near = 2.0
    config.reference.lambda_tau = 0.35
    config.reference.lambda_travel = 0.0005
    config.reference.k_trans = 1.0
    config.reference.k_rot = 0.35
    config.reference.soft_contact_selection = True
    config.reference.contact_softmax_temp = 0.05
    config.reference.contact_gate_d0 = 8.0
    config.reference.tau_contact = 4.0
    config.reference.tau_goal = 35.0
    config.reference.lambda_theta = 25.0
    config.reference.K_free = 0.04
    config.reference.K_contact = 0.12
    config.reference.K_goal_gain = 1.5
    config.reference.K_min = 0.0
    config.reference.K_max = 0.35
    config.reference.gamma_free = 0.08
    config.reference.gamma_contact = 0.18
    config.reference.gamma_goal = 0.35
    config.reference.gamma_min = 0.02
    config.reference.gamma_max = 0.85
    config.reference.max_step_norm = 12.0
    config.reference.deterministic_inference = True

    config.loss.beta_z_start = 0.0
    config.loss.beta_z_end = 0.0
    config.loss.beta_z_warmup_steps = 1
    config.loss.free_nats = 0.0
    config.loss.lambda_unroll = 1.0
    config.loss.lambda_unroll_warmup_steps = 5000

    config.optim.lr = 0.0001
    config.optim.batch_size = 512
    config.optim.max_steps = 300000

    config.inference.deterministic = True
    config.inference.latent_commitment = "chunk"
    config.inference.n_exec = 8
    config.eval.sim_n_exec = 8
    config.eval.sim_collect_contact_diagnostics = True
    return config

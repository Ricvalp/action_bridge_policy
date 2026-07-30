"""Push-T contact bridge with stop-gradient damped-continuation reference loss."""

from action_bridge.configs.pusht_lowdim_contact_quadratic import get_config as quadratic_config


def get_config():
    config = quadratic_config()
    config.loss.contact_objective = "stopgrad_reference"
    config.loss.passive_target = "damped_continuation"
    config.loss.passive_alpha_max = 1.0
    config.loss.passive_ema_decay = 0.85
    config.loss.passive_contact_lambda_parallel = 0.8
    config.loss.passive_contact_lambda_perp = 0.2
    config.loss.passive_contact_distance = 18.0
    config.loss.passive_contact_temperature = 4.0
    config.loss.passive_contact_boundary_samples_per_edge = 8
    config.loss.passive_contact_goal_xy = [256.0, 256.0]
    config.loss.passive_eps = 1e-8
    config.loss.lambda_ref = 0.5
    config.loss.lambda_ref_warmup_steps = 5000
    config.loss.lambda_slow = 0.01
    config.loss.lambda_slow_warmup_steps = 0
    config.loss.lambda_diss = 0.0001
    config.loss.lambda_diss_warmup_steps = 0
    config.loss.ema_decay = 0.995
    config.loss.lambda_unroll = 0.0
    config.loss.lambda_unroll_warmup_steps = 0
    return config

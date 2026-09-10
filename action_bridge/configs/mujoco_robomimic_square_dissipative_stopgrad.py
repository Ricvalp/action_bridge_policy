"""Square: dissipative reference trained separately from its residual control."""

from action_bridge.configs.mujoco_robomimic_square_dissipative import get_config as joint_config


def get_config():
    config = joint_config()
    config.loss.contact_objective = "stopgrad_reference"
    # This is a demonstration-derived passive TARGET, not a reference model.
    # The reference still learns its potential, attractor, and damping.
    config.loss.passive_target = "damped_continuation"
    config.loss.passive_alpha_max = 0.4
    config.loss.passive_eps = 1e-8
    config.loss.lambda_ref = 0.5
    config.loss.lambda_ref_warmup_steps = 5000
    config.loss.lambda_slow = 0.01
    config.loss.lambda_slow_warmup_steps = 0
    config.loss.lambda_diss = 0.0001
    config.loss.lambda_diss_warmup_steps = 0
    config.loss.ema_decay = 0.995
    # Keep the same unroll weight/schedule as the joint-training comparison.
    # This objective sends unroll gradients through control, not the reference.
    return config

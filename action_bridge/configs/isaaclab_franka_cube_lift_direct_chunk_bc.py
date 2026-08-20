"""Direct action-chunk BC baseline for PHI Franka cube lift."""

from action_bridge.configs.base import isaaclab_franka_cube_lift_config


def get_config():
    config = isaaclab_franka_cube_lift_config("continuous")
    config.model.policy_type = "direct_bc"
    config.model.latent_type = "none"
    config.model.z_embed_dim = 0
    config.model.depth = 3
    if "z_dim" in config.model:
        del config.model.z_dim
    if "continuous_prior" in config.model:
        del config.model.continuous_prior
    return config

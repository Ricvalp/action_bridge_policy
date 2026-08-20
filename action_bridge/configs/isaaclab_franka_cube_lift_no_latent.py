"""No-latent Action Bridge baseline for PHI Franka cube lift."""

from action_bridge.configs.base import isaaclab_franka_cube_lift_config


def get_config():
    config = isaaclab_franka_cube_lift_config("continuous")
    config.model.latent_type = "none"
    config.model.z_embed_dim = 0
    del config.model.z_dim
    del config.model.continuous_prior
    return config

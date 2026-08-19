"""No-latent Action Bridge baseline for PHI planar reach."""

from action_bridge.configs.base import mujoco_planar_reach_config


def get_config():
    config = mujoco_planar_reach_config("continuous")
    config.model.latent_type = "none"
    config.model.z_embed_dim = 0
    del config.model.z_dim
    del config.model.continuous_prior
    return config

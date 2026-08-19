"""Direct action-chunk BC baseline for PHI planar reach."""

from action_bridge.configs.base import mujoco_planar_reach_config


def get_config():
    config = mujoco_planar_reach_config("continuous")
    config.model.policy_type = "direct_bc"
    config.model.latent_type = "none"
    config.model.z_embed_dim = 0
    config.model.depth = 3
    if "z_dim" in config.model:
        del config.model.z_dim
    return config

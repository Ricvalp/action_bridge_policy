"""Push-T low-dimensional config with a categorical chunk latent."""

from action_bridge.configs.base import pusht_lowdim_config


def get_config():
    return pusht_lowdim_config("categorical")

"""Continuous-latent Action Bridge baseline for PHI planar reach."""

from action_bridge.configs.base import mujoco_planar_reach_config


def get_config():
    return mujoco_planar_reach_config("continuous")

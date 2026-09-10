"""DDIM baseline using the same Square data, training, and rollout pipeline."""

from ml_collections import ConfigDict

from action_bridge.configs.base import mujoco_config


def get_config():
    config = mujoco_config("robomimic_square", "none")
    config.model = ConfigDict(
        dict(
            policy_type="diffusion",
            hidden_dim=256,
            h_emb_dim=256,
            time_emb_dim=64,
            unet_channels=[80, 160, 320],
            num_train_timesteps=100,
            num_inference_steps=20,
            beta_schedule="squaredcos_cap_v2",
            timestep_spacing="leading",
        )
    )
    # Noise-prediction MSE only: no bridge reference, KL, or smoothing losses.
    config.loss = ConfigDict()
    del config.reference
    config.eval.sampling_seed = 0
    return config

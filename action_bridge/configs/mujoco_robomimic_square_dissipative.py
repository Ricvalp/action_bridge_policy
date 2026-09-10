"""Square: jointly trained dissipative reference and control, without latents."""

from ml_collections import ConfigDict

from action_bridge.configs.base import mujoco_config


def get_config():
    # Reuse the MuJoCo data/runtime settings, not its old model or objective.
    config = mujoco_config("robomimic_square", "none")
    config.model = ConfigDict(
        dict(
            policy_type="action_bridge",
            latent_type="none",
            hidden_dim=960,
            h_emb_dim=960,
            encoder_depth=3,
            control_depth=4,
            time_emb_dim=64,
            control_scale=1.0,
            # Used only if model.latent_type=continuous is requested.
            z_dim=4,
            z_embed_dim=128,
            continuous_prior="learned_conditional_gaussian",
        )
    )
    config.reference = ConfigDict(
        dict(
            type="contact_langevin",
            # q is the normalized 7D OSC command; p is its finite difference.
            # Do not integrate these commands as if they were absolute poses.
            coordinate_mode="raw_action",
            dt=1.0,
            sigma=0.5,
            control_is_whitened=True,
            potential_type="quadratic",
            attractor_mode="learned",
            stiffness_mode="learned_diag",
            k_min=0.0,
            k_max=2.0,
            gamma_mode="learned_scalar",
            gamma_min=0.0,
            gamma_max=0.95,
            hidden_dim=128,
            time_emb_dim=64,
            beta_kl=0.001,
            lambda_q=1.0,
            lambda_ref_reg=0.0001,
            lambda_m_smooth=0.001,
            deterministic_inference=True,
        )
    )
    config.loss = ConfigDict(
        dict(
            contact_objective="standard",
            lambda_unroll=1.0,
            lambda_unroll_warmup_steps=1000,
            # Inactive without latents; shared by both reference objectives.
            beta_z_start=0.001,
            beta_z_end=0.01,
            beta_z_warmup_steps=5000,
            free_nats=0.0,
            num_z_samples_train=1,
            vectorize_z_samples_train=False,
        )
    )
    config.chunk_horizon = 8
    config.inference.n_exec = 4
    config.eval.actions_per_plan = 4
    config.optim.batch_size = 256
    config.optim.max_steps = 100_000
    config.logging.sim_eval_every_steps = 5000
    config.logging.sim_eval_n_exec = 4
    return config

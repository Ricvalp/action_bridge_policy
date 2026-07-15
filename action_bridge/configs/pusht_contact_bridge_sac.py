"""Default overrides for ContactBridgeSAC fine-tuning on Push-T."""

from ml_collections import ConfigDict

from action_bridge.configs.base import pusht_lowdim_config


def get_config():
    config = pusht_lowdim_config("continuous")
    config.run_id = "pusht_contact_bridge_sac"
    config.rl = ConfigDict()
    config.rl.algorithm = "contact_bridge_sac"
    config.rl.checkpoint = None
    config.rl.n_exec = 8
    config.rl.gamma = 0.99
    config.rl.replay_size = 1000000
    config.rl.batch_size = 256
    config.rl.num_envs = 1
    config.rl.prefill_bc_episodes = 200
    config.rl.critic_pretrain_steps = 50000
    config.rl.total_env_steps = 200000
    config.rl.collect_episodes_per_iter = None
    config.rl.updates_per_env_step = 1.0
    config.rl.actor_update_delay = 2
    config.rl.success_bonus = 0.0
    config.rl.stochastic_latent_collection = True

    config.rl.critic_hidden_dim = 512
    config.rl.critic_depth = 3
    config.rl.critic_lr = 3.0e-4
    config.rl.critic_grad_clip = 1.0
    config.rl.target_tau = 0.005

    config.rl.actor_lr = 1.0e-5
    config.rl.actor_grad_clip = 1.0
    config.rl.freeze_reference = True
    config.rl.train_history_encoder = False

    config.rl.alpha_init = 0.05
    config.rl.alpha_lr = 1.0e-4
    config.rl.alpha_min = 1.0e-4
    config.rl.alpha_max = 10.0
    config.rl.target_ref_cost = 0.05

    config.rl.lambda_bc_start = 10.0
    config.rl.lambda_bc_end = 0.5
    config.rl.lambda_bc_anneal_steps = 100000

    config.rl.log_every_updates = 100
    config.rl.checkpoint_every_env_steps = 10000
    config.rl.eval_every_env_steps = 5000
    config.rl.eval_episodes = 20
    config.rl.eval_max_steps = 500
    config.rl.eval_render_episodes = 0
    config.rl.eval_save_gifs = False
    config.rl.continue_eval_on_error = True

    config.logging.wandb.enabled = False
    config.logging.wandb.project = "action-bridge-policy"
    return config

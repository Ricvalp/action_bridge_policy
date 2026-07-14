"""Push-T small autoregressive BC baseline matched to the geometric residual."""

from action_bridge.configs.base import pusht_lowdim_config


def get_config():
    config = pusht_lowdim_config("continuous")
    config.obs_dim = 5
    config.action_dim = 2
    config.model.policy_type = "autoregressive_bc"
    config.model.latent_type = "none"
    config.model.hidden_dim = 256
    config.model.h_emb_dim = 256
    config.model.time_emb_dim = 32
    config.model.depth = 3
    config.model.z_embed_dim = 0
    if "z_dim" in config.model:
        del config.model.z_dim

    config.optim.lr = 0.0001
    config.optim.batch_size = 512
    config.optim.max_steps = 300000

    config.inference.deterministic = True
    config.inference.n_exec = 8
    config.eval.sim_n_exec = 8
    return config

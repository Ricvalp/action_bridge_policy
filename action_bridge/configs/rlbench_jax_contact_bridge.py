"""JAX RLBench contact bridge with a continuous chunk latent."""

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.seed = 0
    config.run_id = "rlbench_jax_contact_bridge"
    config.output_dir = "outputs"
    config.policy_type = "action_bridge"

    config.data = ConfigDict()
    config.data.cache_root = "data/rlbench_cache"
    config.data.tasks = []
    config.data.exclude_tasks = []
    config.data.variation_ids = []
    config.data.obs_history = 2
    config.data.action_history = 2
    config.data.chunk_horizon = 16
    config.data.obs_stride = 1
    config.data.action_stride = 1
    config.data.action_offset = 1
    config.data.action_representation = "absolute"
    config.data.train_fraction = 0.8
    config.data.val_fraction = 0.1
    config.data.split_seed = 0
    config.data.pad_episode_starts = True
    config.data.pad_episode_ends = False
    config.data.include_rgb = True
    config.data.include_mask_id = False
    config.data.point_count = 1024
    config.data.point_sampling = "random"
    config.data.sampling_strategy = "task_uniform"
    config.data.max_episodes_per_variation = None
    config.data.preload_to_memory = False
    config.data.prefetch_workers = 4
    config.data.prefetch_batches = 4

    config.encoder = ConfigDict()
    config.encoder.type = "supernode"
    config.encoder.d_model = 256
    config.encoder.n_heads = 4
    config.encoder.mlp_mult = 4
    config.encoder.dropout = 0.0
    config.encoder.frame_num_latents = 8
    config.encoder.frame_layers = 2
    config.encoder.supernodes = 64
    config.encoder.supernode_temperature = 0.005
    config.encoder.supernode_center_sampling = "linspace"
    config.encoder.supernode_layers = 2
    config.encoder.history_layers = 1
    config.encoder.query_num_latents = 64
    config.encoder.query_layers = 1
    config.encoder.max_obs_history = 16
    config.encoder.max_action_history = 32
    config.encoder.mask_id_vocab = 256
    config.encoder.use_rgb = True
    config.encoder.use_mask_id = False
    config.encoder.use_task_tokens = True

    config.decoder = ConfigDict()
    config.decoder.num_layers = 4

    config.bridge = ConfigDict()
    config.bridge.z_dim = 4
    config.bridge.z_embed_dim = 64
    config.bridge.hidden_dim = 512
    config.bridge.control_depth = 3
    config.bridge.reference_depth = 2
    config.bridge.auxiliary_depth = 2
    config.bridge.dt = 1.0
    config.bridge.sigma = 0.1
    config.bridge.k_min = 0.0
    config.bridge.k_max = 2.0
    config.bridge.gamma_min = 0.0
    config.bridge.gamma_max = 0.95
    config.bridge.xyz_center = [0.0, 0.0, 1.25]
    config.bridge.xyz_scale = [1.0, 1.0, 1.25]

    config.loss = ConfigDict()
    config.loss.xyz_weight = 1.0
    config.loss.momentum_weight = 0.25
    config.loss.unroll_weight = 0.1
    config.loss.quaternion_weight = 1.0
    config.loss.gripper_weight = 0.1
    config.loss.beta_R = 0.001
    config.loss.beta_z_start = 0.0
    config.loss.beta_z_end = 0.001
    config.loss.beta_z_warmup_steps = 20000
    config.loss.free_nats = 0.05

    config.optim = ConfigDict()
    config.optim.batch_size = 128
    config.optim.max_steps = 300000
    config.optim.lr = 0.0002
    config.optim.weight_decay = 0.000001
    config.optim.grad_clip = 1.0

    config.logging = ConfigDict()
    config.logging.log_every_steps = 100
    config.logging.val_every_steps = 2000
    config.logging.val_batches = 8
    config.logging.checkpoint_every_steps = 20000
    config.logging.artifact_every_steps = 10000
    config.logging.artifact_examples = 4
    config.logging.wandb = ConfigDict()
    config.logging.wandb.enabled = False
    config.logging.wandb.project = "action-bridge-policy-rlbench"
    config.logging.wandb.entity = None
    config.logging.wandb.mode = "online"
    config.logging.wandb.group = None
    config.logging.wandb.tags = []
    config.logging.wandb.log_plotly = False

    config.checkpoint = ConfigDict()
    config.checkpoint.resume_path = None
    config.checkpoint.resume_wandb = True
    return config


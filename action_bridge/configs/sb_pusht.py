"""Five jobs, one seed. No inheritance from the old Action Bridge configs."""
METHODS = ("ddim", "fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic")
COMPLETIONS = ("repeat", "fixed_damped", "learned_dissipative")


def get_config(method="ddim"):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    return dict(method=method, seed=0, obs_dim=5, action_dim=2, obs_history=2,
                action_history=2, horizon=16, execute=8, channels=[80, 160, 320],
                history_dim=256, hidden_dim=256, time_dim=64,
                num_train_timesteps=100, num_inference_steps=32,
                updates=300_000, rounds=4, phase_updates=37_500,
                batch_size=256, lr=1e-4, weight_decay=1e-6, ema_decay=.999,
                grad_clip=1., checkpoint_every=5000, log_every=100,
                validation_every=10_000, reference_updates=20_000,
                reference_hidden_dim=64, innovation_floor=1e-3,
                robot_dt=1., temperature=.05, revision_gamma=2.,
                prior_ridge=.05, max_rate=4., mobility_smoothing=0.,
                source_std=.01, endpoint_std=.001, time_cutoff=.01,
                proposals_per_history=2, proposal_seed=17,
                coupling_records=4096, coupling_refresh_every=1000,
                coupling_steps=32, ot_block_size=8, ot_entropy=.1,
                ot_context_weight=1., validation_seeds=list(range(500_000, 500_005)),
                evaluation_seeds=list(range(1_000_000, 1_000_050)),
                max_episode_steps=300, completion_id=2,
                encoder_spec={"kind": "HistoryEncoder", "observations": "flat_tensor"},
                sampler_spec={"ddim": "eta=0", "fm": "midpoint16", "sb": "exact_linear_cosine32"})

# Action Bridge Policy

Fail-fast experiment for the execution-time action bridge idea in:

```text
../../ideas/execution_time_action_bridge.md
```

The experiment intentionally starts with a cheap state-based surrogate rather than image Push-T. A point agent must reach a goal while going around a circular obstacle. Demonstrations contain paired top/bottom modes from similar starts, and the model receives recent state-action history before predicting future action chunks.

The delayed-branch variant is the cleaner multimodality test. In that dataset, paired top/bottom demonstrations share the same first `K` approach actions and the same state at the fork. Previous actions are therefore useful for continuity but do not reveal whether the future should go above or below the obstacle.

The key model is a residual action bridge:

```text
a_{k+1} = a_k + tau * v_theta(a_k, h_t, k)
```

Here bridge time is execution time, so the intermediate outputs are the future actions that would be executed.

## Variants

- `regression`: deterministic MLP action-chunk regression.
- `gaussian_chunk`: MLP chunk generator conditioned on Gaussian noise.
- `bridge_no_energy`: previous-action initialized residual action bridge with no path-energy term.
- `bridge_prev`: previous-action initialized residual action bridge with path energy.
- `bridge_gaussian`: Gaussian-initialized residual bridge with path energy.
- `sinkhorn_bridge`: probabilistic particle bridge with Sinkhorn marginal matching.
- `sinkhorn_bridge_state_only`: more ambiguous Sinkhorn ablation without action-history conditioning.
- `latent_path_sinkhorn_delayed_modes`: latent particle bridge trained with path-level Sinkhorn matching on the delayed-branch dataset.
- `latent_marginal_sinkhorn_delayed_modes`: latent particle bridge trained only with per-timestep Sinkhorn marginal matching on the delayed-branch dataset.

`bridge_no_energy` is the most important sanity baseline: if `bridge_prev` does not beat it, the path/bridge objective is probably not adding much.

The Sinkhorn variants are closer to the Contact Wasserstein Geodesic code: a cloud of source particles is pushed through a stack of residual maps, each block gives an action marginal, and Sinkhorn divergence matches generated marginals to expert action marginals.

The latent path-Sinkhorn variant evolves particles in `y_k = (a_k, z_k)`. The action coordinate is decoded directly as the executable action, while the latent coordinate can carry unresolved mode intent. Its primary loss matches whole generated action paths to whole expert action paths using a joint `(current_state, action_path)` Sinkhorn divergence.

## Quick Smoke Run

From this directory:

```bash
uv run python -m action_bridge.train \
  --config=action_bridge/configs/bridge_prev.py \
  --config.train.epochs=1 \
  --config.data.num_trajectories=128 \
  --config.train.batch_size=64 \
  --config.run_name=smoke_bridge_prev
```

Fallback without `uv`:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m action_bridge.train \
  --config=action_bridge/configs/bridge_prev.py \
  --config.train.epochs=1 \
  --config.data.num_trajectories=128 \
  --config.train.batch_size=64 \
  --config.run_name=smoke_bridge_prev
```

## Small Comparison

```bash
uv run python -m action_bridge.run_all \
  --epochs=5 \
  --num_trajectories=1000 \
  --horizon=16
```

Run the probabilistic Sinkhorn bridge:

```bash
uv run python -m action_bridge.train \
  --config=action_bridge/configs/sinkhorn_bridge.py \
  --config.train.epochs=12 \
  --config.data.num_trajectories=1000 \
  --config.data.horizon=16 \
  --config.train.batch_size=64 \
  --config.model.particles=8 \
  --config.device=cuda
```

Run the delayed-branch latent path-Sinkhorn experiment:

```bash
uv run python -m action_bridge.train \
  --config=action_bridge/configs/latent_path_sinkhorn_delayed_modes.py \
  --config.train.epochs=12 \
  --config.data.num_trajectories=1000 \
  --config.train.batch_size=64 \
  --config.model.particles=24 \
  --config.device=cuda
```

Run the delayed-branch latent marginal-only Sinkhorn ablation:

```bash
uv run python -m action_bridge.train \
  --config=action_bridge/configs/latent_marginal_sinkhorn_delayed_modes.py \
  --config.train.epochs=12 \
  --config.data.num_trajectories=1000 \
  --config.train.batch_size=64 \
  --config.model.particles=24 \
  --config.device=cuda
```

Outputs are written to:

```text
runs/<run-name>/
```

Each run contains:

- `metrics.json`
- `losses.csv`
- `model.pt`
- `rollouts.png`
- `multimodal_samples.png` for configs with `eval.multimodal_examples > 0`
- `position_marginals.png` for configs with `eval.marginal_examples > 0`
- `training_path_plots/step_*.png` when `logging.path_plot_every_steps > 0`

## W&B Logging

W&B is off by default so local breakpoint debugging stays lightweight. Enable it with config overrides:

```bash
uv run python -m action_bridge.train \
  --config=action_bridge/configs/latent_path_sinkhorn_delayed_modes.py \
  --config.logging.wandb=True \
  --config.logging.wandb_project=action-bridge-policy \
  --config.logging.log_every_steps=25 \
  --config.logging.path_plot_every_steps=100 \
  --config.device=cuda
```

Useful logging knobs:

- `logging.wandb`: enable/disable W&B.
- `logging.wandb_mode`: use `online`, `offline`, or `disabled`.
- `logging.log_every_steps`: scalar training metrics frequency.
- `logging.path_plot_every_steps`: predicted-vs-target path snapshot frequency; `0` disables it.
- `logging.path_plot_examples`: batch examples per snapshot.
- `logging.path_plot_particles`: generated particles shown per example.

## Readout

Primary metrics:

- `rollout_success_rate`
- `rollout_final_distance`
- `rollout_obstacle_cross_rate`
- `rollout_chunk_discontinuity`
- `rollout_jerk`
- `action_mse`
- `particle_diversity` for probabilistic variants
- `particle_path_diversity` for probabilistic variants
- `marginal_sinkhorn` for marginal Sinkhorn variants
- `path_sinkhorn` for path-level Sinkhorn variants
- `multimodal_mode_entropy`, `multimodal_top_fraction`, and `multimodal_bottom_fraction` for delayed-branch multimodality tests
- `position_marginal_nearest_gt_distance` and `position_marginal_mode_entropy` for marginal-distribution diagnostics

The idea is weak if `bridge_prev` only improves smoothness while hurting success, or if `bridge_no_energy` matches it.

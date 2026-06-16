# Action Bridge Policy

Fail-fast experiment for the execution-time action bridge idea in:

```text
../../ideas/execution_time_action_bridge.md
```

The experiment intentionally starts with a cheap state-based surrogate rather than image Push-T. A point agent must reach a goal while going around a circular obstacle. Demonstrations contain paired top/bottom modes from similar starts, and the model receives recent state-action history before predicting future action chunks.

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

`bridge_no_energy` is the most important sanity baseline: if `bridge_prev` does not beat it, the path/bridge objective is probably not adding much.

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

Outputs are written to:

```text
runs/<run-name>/
```

Each run contains:

- `metrics.json`
- `losses.csv`
- `model.pt`
- `rollouts.png`

## Readout

Primary metrics:

- `rollout_success_rate`
- `rollout_final_distance`
- `rollout_obstacle_cross_rate`
- `rollout_chunk_discontinuity`
- `rollout_jerk`
- `action_mse`

The idea is weak if `bridge_prev` only improves smoothness while hurting success, or if `bridge_no_energy` matches it.

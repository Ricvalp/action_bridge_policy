# Latent Path-KL Action Bridge Policy

This sandbox implements a toy-first research pilot for action chunks as stochastic action-path laws. The main policy is a reference action process plus a learned control residual, with an optional latent variable held fixed over the chunk for path-level mode commitment. It is SB-inspired, but it is not an exact Schrodinger Bridge solver and does not use Sinkhorn, IPF, score matching, or diffusion noising.

## Install

From this directory:

```bash
uv sync --extra cpu --extra test
```

Choose the PyTorch backend explicitly for each machine:

```bash
# MacBook / CPU-only evaluation
uv sync --extra cpu --extra test --locked

# Linux GPU nodes; pick the CUDA build that matches the machine
uv sync --extra cu118 --extra test --locked
uv sync --extra cu126 --extra test --locked
uv sync --extra cu128 --extra test --locked
uv sync --extra cu130 --extra test --locked
```

Only one of `cpu`, `cu118`, `cu126`, `cu128`, or `cu130` should be enabled in a given environment. Use `uv lock` locally after dependency changes, then use `uv sync --locked` on experiment machines so they do not rewrite `uv.lock`.

If uv is unavailable, create a virtualenv and install dependencies manually. For PyTorch, use the install command appropriate for the target machine from the PyTorch selector.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Toy Data

Generate a small delayed-branch top/bottom dataset artifact:

```bash
uv run python -m action_bridge.scripts.generate_toy_delayed \
  --num-contexts 32 \
  --out /tmp/toy_delayed_small.pt
```

Generate annular clockwise/counterclockwise data:

```bash
uv run python -m action_bridge.scripts.generate_toy_annular \
  --num-contexts 32 \
  --out /tmp/toy_annular_small.pt
```

The training scripts generate toy data in memory from the Python `ml_collections` config, so these artifacts are mostly for inspection and debugging.

Visualize the toy datasets before training:

```bash
uv run python -m action_bridge.scripts.visualize_toy_dataset \
  --config-name toy_delayed_categorical \
  --out-dir outputs/dataset_viz/toy_delayed

uv run python -m action_bridge.scripts.visualize_toy_dataset \
  --config-name toy_annular_categorical \
  --out-dir outputs/dataset_viz/toy_annular
```

The visualizer writes full trajectory overlays, paired-mode examples, chunk windows, action traces, clearance histograms, start/goal plots, and dataset-specific diagnostics.

## Training

Categorical latent delayed-branch pilot:

```bash
uv run python -m action_bridge.training.train_toy \
  --config-name toy_delayed_categorical
```

Continuous latent delayed-branch pilot:

```bash
uv run python -m action_bridge.training.train_toy \
  --config-name toy_delayed_continuous
```

Annular variants:

```bash
uv run python -m action_bridge.training.train_toy --config-name toy_annular_categorical
uv run python -m action_bridge.training.train_toy --config-name toy_annular_continuous
```

Short CPU smoke:

```bash
uv run python -m action_bridge.training.train_toy \
  --config-name toy_delayed_categorical \
  optim.max_steps=50 \
  device=cpu
```

Useful ablations are CLI overrides:

```bash
model.latent_type=none
reference.type=brownian
model.policy_type=direct_bc
model.policy_type=autoregressive_bc
loss.lambda_acc=0.01 loss.lambda_jerk=0.001
loss.tube_training=true
```

Configs live in `action_bridge/configs/*.py` and expose `get_config()`. CLI overrides still use the same dotted style:

```bash
uv run python -m action_bridge.training.train_toy \
  --config-name toy_delayed_continuous \
  device=cpu \
  optim.max_steps=10000 \
  reference.type=continuation
```

Training periodically writes reduced validation eval artifacts under `outputs/<run_id>/eval/step_<step>/` according to:

```text
logging.full_eval_every_steps
logging.full_eval_split
logging.full_eval_max_batches
logging.full_eval_closed_loop_episodes
logging.full_eval_num_samples
```

Enable Weights & Biases logging with:

```bash
uv run python -m action_bridge.training.train_toy \
  --config-name toy_delayed_continuous \
  logging.wandb.enabled=true \
  logging.wandb.project=action-bridge-policy
```

When enabled, training logs train/validation scalars, periodic closed-loop/open-loop eval metrics, and the generated figures: `closed_loop_rollouts`, `continuous_latent_scatter`, `energy_histograms`, and `generated_same_history`.

Run the compact delayed-branch comparison sweep:

```bash
uv run python -m action_bridge.scripts.run_toy_sweep \
  --max-steps 2000 \
  --device cpu
```

## Evaluation

Reload a checkpoint:

```bash
uv run python -m action_bridge.eval.eval_toy \
  --checkpoint outputs/<run_id>/checkpoints/latest.pt \
  device=cpu
```

By default this creates a timestamped eval artifact directory:

```text
outputs/eval/<run_id>_<YYYYmmdd_HHMMSS>/
```

Use `--output-dir` to choose a specific destination:

```bash
uv run python -m action_bridge.eval.eval_toy \
  --checkpoint outputs/<run_id>/checkpoints/latest.pt \
  --output-dir outputs/eval/manual_eval \
  device=cpu
```

Toy evaluation includes open-loop chunk metrics and closed-loop receding-horizon rollout metrics by default. Useful closed-loop overrides:

```bash
uv run python -m action_bridge.eval.eval_toy \
  --checkpoint outputs/<run_id>/checkpoints/best.pt \
  device=cpu \
  eval.n_exec=4 \
  eval.closed_loop_episodes=128 \
  eval.success_radius=0.08
```

Every training run writes:

```text
outputs/<run_id>/
  config.json
  checkpoints/latest.pt
  checkpoints/best.pt
  metrics/train_metrics.csv
  metrics/val_metrics.csv
  metrics/periodic_eval_metrics.csv
  metrics/test_metrics.json
  metrics/closed_loop_metrics.json
  figures/dataset_samples.png
  figures/generated_same_history.png
  figures/closed_loop_rollouts.png
  figures/energy_histograms.png
```

Categorical annular runs also attempt a prior calibration plot; continuous runs attempt a latent scatter plot.

## Model Variants

The path-KL policy predicts each action as:

```python
a_next = mu_reference(a_prev, a_prevprev, h, k) + u_theta(a_prev, a_prevprev, h, k, z)
```

The reference process is used at inference too. Implemented references include Brownian/raw action, fixed or learned continuation, low-acceleration learned alpha, and low-jerk.

Categorical latent runs enumerate all categories during training, add an analytic categorical KL between posterior and prior, and sample one category per chunk at inference. Continuous latent runs use a VAE-style Gaussian posterior/prior with reparameterization and hold one unconstrained latent vector fixed over the chunk.

Implemented baselines include direct chunk BC, autoregressive BC without a reference, BC with acceleration/jerk penalties, path-KL without a latent, and tube-training variants.

## Metrics

Toy evaluation reports action MSE, goal error, path length, collision rate, minimum clearance, acceleration energy, jerk energy, path-KL energy, mode switch rate, hybrid rate, valid top/bottom rates for delayed data, cw/ccw rates for annular data, and an RBF-MMD over simple trajectory features. Closed-loop evaluation additionally reports success rate, clean success rate, receding-horizon chunk-boundary discontinuity, closed-loop collision/hybrid rates, and final goal error.

## RLBench

The RLBench data layer caches complete point-cloud demonstration episodes in
HDF5 and builds standard state-conditioned imitation-learning windows at load
time. Temporal histories, action horizons and strides, episode splits, point
subsampling, and absolute versus `delta_xyz` actions can therefore change
without recaching.

See [RLBENCH_DATA.md](RLBENCH_DATA.md) for the raw-data contract, cache command,
schema, and `RLBenchDataset` API. Point-cloud policy architectures and an
RLBench training entrypoint are intentionally not connected yet.

## Push-T

`action_bridge.data.pusht_adapter.PushTLowDimDataset` loads local low-dimensional offline Push-T data. Supported backends are:

```text
data.backend=zarr   # Diffusion Policy-style zarr with data/state, data/action, meta/episode_ends
data.backend=npz    # arrays such as obs/actions/episode_ends
data.backend=torch  # .pt/.pth dict with the same arrays
data.backend=auto   # infer from the path suffix
```

Train the continuous latent Push-T pilot:

```bash
uv run python -m action_bridge.training.train_pusht \
  --config-name pusht_lowdim_continuous \
  data.dataset_path=/path/to/pusht_lowdim.zarr
```

If your local file uses nonstandard names, pass explicit keys:

```bash
data.obs_key=data/state \
data.action_key=data/action \
data.episode_ends_key=meta/episode_ends
```

The Push-T entrypoint reports offline metrics by default: action MSE/L1, acceleration/jerk energy, chunk-boundary discontinuity, path-KL energy, and logged-history receding-horizon action error.

Simulator closed-loop evaluation is optional and uses `gym-pusht`:

```bash
uv pip install gym-pusht

uv run python -m action_bridge.scripts.eval_pusht_sim \
  --checkpoint outputs/<run_id>/checkpoints/best.pt \
  --device cuda \
  --episodes 50 \
  --max-steps 300 \
  --n-exec 8 \
  --render-episodes 4 \
  --gif-fps 10
```

It logs simulator success rate, final/max coverage reward, episode length, replanning count, path-KL energy, chunk-boundary discontinuity, rollout figures, and one GIF per rendered episode under `figures/pusht_sim_gifs/`. Add `--save-videos --video-fps 10` to request MP4 files when `imageio` is installed.

## Tests

```bash
uv run pytest tests -q
```

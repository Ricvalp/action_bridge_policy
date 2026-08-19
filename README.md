# Latent Path-KL Action Bridge Policy

This sandbox implements a toy-first research pilot for action chunks as stochastic action-path laws. The main policy is a reference action process plus a learned control residual, with an optional latent variable held fixed over the chunk for path-level mode commitment. It is SB-inspired, but it is not an exact Schrodinger Bridge solver and does not use Sinkhorn, IPF, score matching, or diffusion noising.

## Install

This branch supports Python 3.11 and 3.12. RLBench-neutral Action Bridge code
and RLBench training currently share that project-level requirement because
`phi-rlbench` itself is restricted to `>=3.11,<3.13`.

The current dependency source is an editable development checkout at
`workspace/phi-rlbench`. That directory is ignored by this repository and is
not present in a fresh clone, so the checked-in source setting is suitable only
for local co-development. With that directory present, run from this directory:

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

Do not deploy the editable `workspace/phi-rlbench` source to an HPC clone. Once
`phi-rlbench` has its own private or appropriately licensed remote, replace the
editable source with an immutable full Git commit, regenerate `uv.lock`, and
commit both files. For example, after substituting the real organization and
40-character revision:

```toml
[tool.uv.sources]
phi-rlbench = { git = "ssh://git@github.com/<PHI-ORG>/phi-rlbench.git", rev = "<FULL-COMMIT>" }
```

Shared jobs should then use `uv sync --locked` and record the Action Bridge
commit, the `phi-rlbench` commit, and the lockfile digest. A moving branch or an
uncommitted local path is not a reproducible HPC dependency.

If uv is unavailable, create a virtualenv and install dependencies manually. For PyTorch, use the install command appropriate for the target machine from the PyTorch selector.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The `requirements.txt` fallback does not reproduce the editable `phi-rlbench`
source or the locked RLBench environment; use `uv` for RLBench work.

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

RLBench support is now split across two repositories with one-way ownership:

- `phi-rlbench` owns the immutable HDF5 schema, whole-root cache builder,
  read-only cache access, NumPy/PyTorch windows, preprocessing, simulator
  runtime, and framework-neutral evaluation loop;
- this repository owns the JAX policies, losses, training loop, checkpoint
  format/loading, offline checkpoint evaluation, and the policy-specific
  adapter used by online evaluation.

The JAX trainer is connected to the `phi-rlbench` NumPy loader. Existing HDF5
v1 caches remain readable without rewriting, and temporal histories, horizons,
splits, point subsampling, and action representation remain loader-time
choices. The deprecated `action_bridge.data.rlbench_*` imports are temporary
compatibility shims; new code should import data APIs directly from
`phi_rlbench`.

`phi-rlbench` can build a new cache from an existing raw demonstration tree,
but persistent RLBench demonstration collection is not implemented yet. Cache
construction is an immutable whole-new-root operation: it never appends to or
overwrites an existing cache destination.

The policy-owned online command now exists at
`python -m action_bridge.jax.eval.rlbench_online`. It has passed pure metadata,
adapter, and cached-input tests, but no learned GUI or headless CoppeliaSim
episode has been run successfully on this workstation. It must not yet be
treated as native-simulator acceptance.

See [RLBENCH_DATA.md](RLBENCH_DATA.md) for the raw-data contract, cache and
loader commands, offline JAX training, dependency pinning, and the provisional
online-evaluation procedure.

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

# RLBench Data and Action Bridge Integration

RLBench support has a one-way repository boundary:

- `phi-rlbench` owns cache construction and validation, read-only cache access,
  NumPy and optional PyTorch windows, observation preprocessing, the native
  RLBench runtime, and the framework-neutral evaluation loop;
- `action_bridge_policy` owns JAX models and losses, training, checkpoint
  serialization/loading, cached checkpoint evaluation, and the policy adapter
  that feeds the generic `phi-rlbench` evaluator.

The dependency direction is `action_bridge_policy -> phi-rlbench`; the backend
does not import Action Bridge or depend on JAX. Both projects are independent
of the audited ICIL checkout.

The current Action Bridge branch supports Python 3.11 and 3.12. Its
`phi-rlbench` dependency is an editable local checkout under
`workspace/phi-rlbench`, which is ignored by this repository and therefore is
not available in a fresh clone. See [HPC dependency pinning](#hpc-dependency-pinning)
before moving this branch to a cluster.

Current acceptance status:

- synthetic cache construction and NumPy/PyTorch window tests pass;
- JAX training and cached offline evaluation remain Action Bridge entrypoints;
- the online checkpoint adapter and command are installed, but learned-policy
  GUI/headless evaluation against CoppeliaSim has not passed on this host;
- persistent raw demonstration collection is not implemented in
  `phi-rlbench` yet.

## Design

The cache stores complete episodes frame-by-frame. It does **not** store
precomputed training windows. The following choices therefore remain loader
settings and do not require recaching:

- observation and action history lengths;
- action chunk horizon;
- observation and action strides;
- episode-level train/validation/test splits;
- absolute or `delta_xyz` action targets;
- point-cloud downsampling up to the cached point count;
- RGB, segmentation masks, and additional low-dimensional observation fields.

The canonical action in the cache is the absolute 8D RLBench gripper target:

```text
[x, y, z, qx, qy, qz, qw, gripper_open]
```

The same vector is cached as `state` and `action`. By default, a state at frame
`t` predicts actions beginning at frame `t + 1`.

## Raw Layout

The converter expects an already-generated RLBench point-cloud demonstration
tree:

```text
RAW_ROOT/
  <task>/
    variation0/
      mask_to_label.json
      variation_descriptions.pkl
      episodes/
        episode0/
          low_dim_obs.pkl
          merged_point_cloud/
            0.npz
            1.npz
```

Each point-cloud NPZ must contain `points`, `colors`, and `masks`.

Raw `low_dim_obs.pkl` files may require the same RLBench/PyRep installation used
to generate them. They are pickle files and must only be loaded from a trusted
source. The resulting HDF5 cache only requires NumPy and `h5py` to read.

This integration does not currently provide persistent raw demonstration
generation. Generation remains a separate prerequisite until the
`phi-rlbench` collection milestone is implemented and validated with a licensed
CoppeliaSim installation.

## Build The Cache

From the `action_bridge_policy` repository root:

```bash
uv run python -m action_bridge.scripts.cache_rlbench \
  --raw-root data/rlbench_raw \
  --cache-root data/rlbench_cache \
  --tasks reach_target push_button \
  --num-points 1024
```

Omit `--tasks` to convert every task. Useful options include:

```text
--num-variations N
--max-episodes-per-variation N
--compression gzip|lzf|none
--no-include-rgb
--no-include-mask-id
--no-ignore-background
--no-workspace-filter
--workspace-bounds XMIN XMAX YMIN YMAX ZMIN ZMAX
--allow-length-mismatch
```

`CACHE_ROOT` must not already exist, even as an empty directory. The builder
stages the complete cache beside the requested destination, validates it, and
publishes the whole root with one rename. It never appends variations or
rewrites an existing cache. The legacy wrapper still parses `--overwrite` to
produce an explicit migration error, but `phi-rlbench` deliberately rejects
that operation; choose a new destination instead.

The output is:

```text
CACHE_ROOT/
  manifest.json
  preprocessing-profile.json
  <task>/
    variation0.h5
```

`preprocessing-profile.json` records the resolved preprocessing recipe without
changing the immutable `action_bridge.rlbench_dense` schema v1. Existing v1
caches remain read-only and do not need to be regenerated for this migration.

Each variation file contains:

```text
episode_ids
episodes/<episode_id>/xyz
episodes/<episode_id>/valid
episodes/<episode_id>/rgb             # optional
episodes/<episode_id>/mask_id         # optional
episodes/<episode_id>/state
episodes/<episode_id>/action
episodes/<episode_id>/gripper_pose
episodes/<episode_id>/gripper_open
episodes/<episode_id>/joint_positions # when present
...other selected low-dimensional fields
```

## Load Training Windows

```python
from torch.utils.data import DataLoader

from phi_rlbench.data.torch_dataset import RLBenchDataset

dataset = RLBenchDataset(
    "data/rlbench_cache",
    split="train",
    tasks=["reach_target"],
    obs_history=2,
    action_history=2,
    chunk_horizon=16,
    action_offset=1,
    action_representation="absolute",
    point_count=1024,
    include_rgb=True,
    include_mask_id=False,
)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
```

To balance tasks, variations, or episodes without changing the cache:

```python
from torch.utils.data import WeightedRandomSampler

weights = dataset.sampling_weights("task_uniform")
sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=8)
```

Available strategies are `window_uniform`, `episode_uniform`,
`variation_uniform`, and `task_uniform`.

Each item contains:

```text
obs_hist                  [T_obs, state_dim]
point_cloud_hist          [T_obs, N, 3]
point_valid_hist          [T_obs, N]
act_hist                  [T_act, action_dim]
future_actions            [H, action_dim]
obs_history_mask          [T_obs]
action_history_mask       [T_act]
future_action_mask        [H]
context/task_id
context/task_variation_id
context/variation_id
context/episode_id
context/time_index
rgb_hist                  [T_obs, N, 3]  # optional, float in [0, 1]
mask_id_hist              [T_obs, N]     # optional
low_dim/<field>           [T_obs, D]     # requested extra fields
```

Use `dataset.set_epoch(epoch)` to obtain a new deterministic random point subset
when `point_count` is below the cached point count. HDF5 handles are opened
lazily per process, so the dataset is safe with multiprocessing data loaders.

For `action_representation="delta_xyz"`, only XYZ is differenced. Quaternion
and gripper-open channels remain absolute. Use `decode_action_chunk()` to
recover absolute targets from `phi_rlbench.data.actions`.

The framework-neutral loader has matching window semantics and returns NumPy
arrays with flat metadata keys:

```python
import numpy as np

from phi_rlbench.data.numpy_dataset import NumpyRLBenchDataset

dataset = NumpyRLBenchDataset(
    "data/rlbench_cache",
    split="train",
    obs_history=2,
    action_history=2,
    chunk_horizon=16,
    action_representation="absolute",
    point_count=1024,
)
batch = dataset.sample_batch(32, np.random.default_rng(0), strategy="task_uniform")
```

The old `action_bridge.data.rlbench_cache`, `rlbench_cache_builder`,
`rlbench_dataset`, and `rlbench_numpy_dataset` modules now provide only
deprecated compatibility re-exports. They emit `FutureWarning` and are planned
for removal in Action Bridge 0.2.0.

## Train The JAX Policy

The cache and loader are supplied by `phi-rlbench`; the model, optimizer,
losses, checkpoint, and training loop in this section remain owned by Action
Bridge.

Install one JAX environment:

```bash
# macOS or CPU node
uv sync --extra jax-cpu --locked

# Linux H200 with a CUDA 13-capable NVIDIA driver
uv sync --extra jax-cu13 --locked
```

Train the XYZ contact bridge:

```bash
uv run --extra jax-cu13 python -m action_bridge.jax.training.train_rlbench \
  --config-name rlbench_jax_contact_bridge \
  run_id=rlbench_contact_bridge_seed0 \
  seed=0 \
  logging.wandb.enabled=true \
  logging.wandb.project=action-bridge-policy-rlbench
```

Train the parameter-matched frontend with direct chunk BC:

```bash
uv run --extra jax-cu13 python -m action_bridge.jax.training.train_rlbench \
  --config-name rlbench_jax_direct_chunk_bc \
  run_id=rlbench_direct_chunk_bc_seed0
```

Training saves `latest.pt`, periodic checkpoints, `best_val.pt`, scalar metrics,
and local interactive 3D chunk diagnostics. It can fully resume model,
optimizer, RNG, step, best validation loss, and optionally the same W&B run:

```bash
uv run --extra jax-cu13 python -m action_bridge.jax.training.train_rlbench \
  --config-name rlbench_jax_contact_bridge \
  checkpoint.resume_path=outputs/rlbench_contact_bridge_seed0/checkpoints/latest.pt \
  optim.max_steps=500000
```

Evaluate cached validation windows from a checkpoint:

```bash
uv run --extra jax-cpu python -m action_bridge.jax.eval.eval_rlbench \
  --checkpoint outputs/rlbench_contact_bridge_seed0/checkpoints/best_val.pt \
  --split val \
  --num-batches 32 \
  --batch-size 64
```

### HPC dependency pinning

The current `[tool.uv.sources]` entry points to the ignored local directory
`workspace/phi-rlbench`. That is useful for development on this workstation,
but neither that directory nor its source revision travels with an Action
Bridge clone. Consequently, this feature branch is not yet a portable HPC
environment even though offline training code is connected.

Before submitting shared jobs:

1. initialize and publish `phi-rlbench` to an access-controlled or otherwise
   appropriately licensed repository;
2. replace the editable path source with a Git source pinned to one full,
   immutable commit, not a moving branch;
3. run `uv lock` locally and review the `pyproject.toml` and `uv.lock` changes;
4. commit both files, then use `uv sync --locked` on the cluster;
5. record the Action Bridge commit, `phi-rlbench` commit, `uv.lock` SHA-256,
   cache manifest identity, resolved task order, and training configuration with
   each run.

Jobs launched from a pre-migration commit are unaffected. New shared jobs
should not be redirected to this feature branch until that pin exists and a
fixed-cache old/new batch plus one forward/loss-step comparison has passed.

### Local migration acceptance (2026-08-19)

Acceptance used the synthetic five-episode `reach_target` fixture and the
pre-migration Action Bridge checkout at commit
`a20d149a80712458af6c6095930a9329ae4e440d`. It is evidence for the integration,
not for a production dataset or native simulator:

- PHI old/new loader and preprocessing compatibility: 9 passed, 1 optional
  Torch check skipped in the PHI-only environment; parent Torch parity passed
  in the Action Bridge suite;
- complete Action Bridge test suite with CPU Torch and JAX: 99 passed;
- a one-step direct-chunk training run consumed the PHI cache, compiled,
  optimized, validated, and wrote a checkpoint with top-level
  `online_evaluation` metadata;
- cached offline evaluation loaded that checkpoint successfully;
- the online adapter loaded the same trusted checkpoint and returned one finite
  action chunk with shape `(2, 8)` from a cached validation observation;
- the fixture manifest and HDF5 SHA-256 values were unchanged before and after:
  `47c9f0d28ec5b92e970ae161e7fcc581aa153384cbc9f57f234352755169aa52`
  and `05453b4bd637eed2f878f4e0420280da9ae3fd8bcdc935f0b2f3d95efe186aea`.

The ignored local evidence is under
`workspace/runs/action-bridge-integration-smoke/one_step`; it is intentionally
not a portable experiment artifact or a learned-policy result.

## Provisional Native Checkpoint Evaluation

The policy-owned adapter is available as:

```bash
uv run --extra jax-cu13 python -m action_bridge.jax.eval.rlbench_online \
  --checkpoint /absolute/path/to/trusted-checkpoint.pt \
  --online-metadata /absolute/path/to/online-metadata.json \
  --trusted-checkpoint \
  --task reach_target \
  --variation 0 \
  --episodes 1 \
  --actions-per-plan 1 \
  --preprocessing-profile legacy \
  --reject-out-of-bounds-actions \
  --gui \
  --record-video \
  --output-root /absolute/path/to/new-evaluation-root
```

Existing checkpoints are pickle files, so `--trusted-checkpoint` is an
explicit acknowledgement that loading them may execute code. An older
checkpoint normally needs `--online-metadata` to provide its exact task and
task-variation vocabularies, state/action component layouts, window geometry,
modalities, and training-cache identity. The argument may be omitted only when
equivalent unambiguous metadata is embedded in the checkpoint or its saved
configuration.

The embedded cache identity hashes the bounded manifest and optional
preprocessing sidecar, not every HDF5 payload. It records which immutable cache
metadata was selected without scanning a large dataset at checkpoint time.
For legacy caches without a sidecar, keep `--preprocessing-profile legacy`
explicit. Automatic semantic comparison between a future sidecar's full
preprocessing recipe and the selected live profile remains follow-up work.

The command requires the `phi-rlbench` simulator and visualization extras plus
a compatible RLBench/PyRep/CoppeliaSim installation; those simulator extras are
not part of the current core Action Bridge lock. Run one GUI episode first and
inspect the actions and video before attempting headless or multi-episode
evaluation. Action bounds are intentionally explicit, and actions are rejected
rather than silently clipped.

Install and lock those extras only on a machine where CoppeliaSim is permitted
and set `COPPELIASIM_ROOT` first. PyRep's package build fails during metadata
preparation when that variable does not identify an installation; this is the
current blocker on the institute workstation, not a failure of cached training.

This command has passed pure import, metadata, adapter, and cached-input parity
tests. It has not loaded a verified production checkpoint on this workstation,
and no learned GUI episode, learned headless episode, 20-episode evaluation, or
native success-rate claim has passed. CoppeliaSim remains unavailable under the
institute's current licensing decision, so native evaluation is paused rather
than complete.

## Visualize Episodes And Training Batches

Generate interactive 3D HTML views directly from the cache:

```bash
uv run --extra cpu python -m action_bridge.scripts.visualize_rlbench_data \
  --cache-root data/rlbench_cache \
  --tasks open_fridge stack_cups basketball_in_hoop \
  --episodes-per-task 1 \
  --obs-history 2 \
  --action-history 2 \
  --chunk-horizon 16 \
  --action-representation absolute \
  --batch-size 6 \
  --num-batches 2
```

The command writes a timestamped directory under
`outputs/rlbench_visualizations/` with:

- animated episode views showing the RGB point cloud, full expert trajectory,
  current gripper pose, orientation axes, and the next target action chunk;
- batch views built from actual collated `DataLoader` output, including tensor
  shapes and one 3D scene per sample;
- `index.html` linking every generated view;
- `visualization_config.json` recording the loader and display settings.

Use `--action-representation delta_xyz` to inspect loader-time delta targets;
the orange chunk is decoded back into world coordinates for display. HTML
files use Plotly from a CDN by default. Add `--embed-plotly` for fully
self-contained files.

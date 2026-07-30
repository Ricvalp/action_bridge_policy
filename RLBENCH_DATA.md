# RLBench Data Layer

This module provides RLBench caching and PyTorch data loading for standard
state-conditioned imitation learning. It is independent of the ICIL project:
`action_bridge_policy` neither imports from nor modifies `icil-jax-rlbench`.

No point-cloud policy is connected yet. The data layer deliberately stops at
producing model-ready histories and action chunks.

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

The converter expects the layout produced by the RLBench point-cloud dataset
generator:

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
to generate them. The resulting HDF5 cache only requires `h5py`.

## Build The Cache

From `sandbox/action_bridge_policy`:

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
--overwrite
```

Writes are atomic at the variation-file level. The output is:

```text
CACHE_ROOT/
  manifest.json
  <task>/
    variation0.h5
```

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

from action_bridge.data.rlbench_dataset import RLBenchDataset

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
recover absolute targets.

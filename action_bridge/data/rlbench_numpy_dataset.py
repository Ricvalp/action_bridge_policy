"""Deprecated compatibility exports for the phi-rlbench NumPy dataset."""

from __future__ import annotations

import warnings

from phi_rlbench.data import (
    EpisodeKey,
    NumpyRLBenchDataset,
    RLBenchCacheStore,
    RLBenchWindowKey,
    SUPPORTED_ACTION_REPRESENTATIONS,
    VariationKey,
    build_cache_keys,
    encode_action_chunk,
    encode_action_history,
    normalize_action_representation,
    split_episode_ids,
)

warnings.warn(
    "action_bridge.data.rlbench_numpy_dataset is deprecated; import the dataset "
    "and helpers from phi_rlbench.data. This compatibility module will be removed in "
    "action-bridge-policy 0.2.0.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "SUPPORTED_ACTION_REPRESENTATIONS",
    "EpisodeKey",
    "NumpyRLBenchDataset",
    "RLBenchCacheStore",
    "RLBenchWindowKey",
    "VariationKey",
    "build_cache_keys",
    "encode_action_chunk",
    "encode_action_history",
    "normalize_action_representation",
    "split_episode_ids",
]

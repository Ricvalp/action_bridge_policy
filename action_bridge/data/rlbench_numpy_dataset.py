"""Deprecated compatibility exports for the phi-rlbench NumPy dataset."""

from __future__ import annotations

import warnings

from phi_rlbench.data.actions import (
    SUPPORTED_ACTION_REPRESENTATIONS,
    encode_action_chunk,
    encode_action_history,
    normalize_action_representation,
)
from phi_rlbench.data.cache import (
    EpisodeKey,
    RLBenchCacheStore,
    VariationKey,
    build_cache_keys,
)
from phi_rlbench.data.indexing import split_episode_ids
from phi_rlbench.data.numpy_dataset import (
    NumpyRLBenchDataset,
    RLBenchWindowKey,
)

warnings.warn(
    "action_bridge.data.rlbench_numpy_dataset is deprecated; import the dataset "
    "from phi_rlbench.data.numpy_dataset and helpers from phi_rlbench.data.actions "
    "or phi_rlbench.data.indexing. This compatibility module will be removed in "
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

"""Deprecated compatibility exports for the phi-rlbench Torch dataset."""

from __future__ import annotations

import warnings

from phi_rlbench.data.actions import (
    SUPPORTED_ACTION_REPRESENTATIONS,
    decode_action_chunk,
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
from phi_rlbench.data.torch_dataset import (
    RLBenchDataset,
    RLBenchWindowKey,
)

warnings.warn(
    "action_bridge.data.rlbench_dataset is deprecated; import the Torch adapter "
    "from phi_rlbench.data.torch_dataset and action helpers from "
    "phi_rlbench.data.actions. This compatibility module will be removed in "
    "action-bridge-policy 0.2.0.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "SUPPORTED_ACTION_REPRESENTATIONS",
    "EpisodeKey",
    "RLBenchCacheStore",
    "RLBenchDataset",
    "RLBenchWindowKey",
    "VariationKey",
    "build_cache_keys",
    "decode_action_chunk",
    "encode_action_chunk",
    "encode_action_history",
    "normalize_action_representation",
    "split_episode_ids",
]

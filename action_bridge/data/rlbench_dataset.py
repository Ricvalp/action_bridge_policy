"""Deprecated compatibility exports for the phi-rlbench Torch dataset."""

from __future__ import annotations

import warnings

from phi_rlbench.data import (
    EpisodeKey,
    RLBenchCacheStore,
    RLBenchDataset,
    RLBenchWindowKey,
    SUPPORTED_ACTION_REPRESENTATIONS,
    VariationKey,
    build_cache_keys,
    decode_action_chunk,
    encode_action_chunk,
    encode_action_history,
    normalize_action_representation,
    split_episode_ids,
)

warnings.warn(
    "action_bridge.data.rlbench_dataset is deprecated; import the Torch adapter "
    "and action helpers from phi_rlbench.data. This compatibility module will be removed in "
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

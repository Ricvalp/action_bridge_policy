"""Deprecated compatibility exports for the phi-rlbench cache API."""

from __future__ import annotations

import warnings

from phi_rlbench.data import (
    CACHE_SCHEMA_NAME,
    CACHE_SCHEMA_VERSION,
    CORE_EPISODE_FIELDS,
    OPTIONAL_POINT_FIELDS,
    EpisodeKey,
    RLBenchCacheStore,
    VariationKey,
    build_cache_keys,
    build_variation_keys,
    discover_tasks,
)

warnings.warn(
    "action_bridge.data.rlbench_cache is deprecated; import cache access from "
    "phi_rlbench.data. "
    "This compatibility module will be removed in action-bridge-policy 0.2.0.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "CACHE_SCHEMA_NAME",
    "CACHE_SCHEMA_VERSION",
    "CORE_EPISODE_FIELDS",
    "OPTIONAL_POINT_FIELDS",
    "EpisodeKey",
    "RLBenchCacheStore",
    "VariationKey",
    "build_cache_keys",
    "build_variation_keys",
    "discover_tasks",
]

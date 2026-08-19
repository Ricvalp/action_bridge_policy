"""Deprecated compatibility exports for the phi-rlbench cache builder."""

from __future__ import annotations

import warnings

from phi_rlbench.data.builder import (
    DEFAULT_LOW_DIM_FIELDS,
    DEFAULT_WORKSPACE_BOUNDS,
    MASK_NAME_SUBSTRINGS_TO_IGNORE,
    MASK_NAMES_TO_IGNORE,
    convert_rlbench_dataset,
    write_cache_manifest,
)
from phi_rlbench.data.schema import (
    CACHE_SCHEMA_NAME,
    CACHE_SCHEMA_VERSION,
)

warnings.warn(
    "action_bridge.data.rlbench_cache_builder is deprecated; import from "
    "phi_rlbench.data.builder. This compatibility module will be removed in "
    "action-bridge-policy 0.2.0.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "CACHE_SCHEMA_NAME",
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_LOW_DIM_FIELDS",
    "DEFAULT_WORKSPACE_BOUNDS",
    "MASK_NAMES_TO_IGNORE",
    "MASK_NAME_SUBSTRINGS_TO_IGNORE",
    "convert_rlbench_dataset",
    "write_cache_manifest",
]

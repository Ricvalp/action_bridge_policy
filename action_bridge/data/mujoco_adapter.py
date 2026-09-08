"""Connect phi-mujoco's state windows to Action Bridge's batch format."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from phi_mujoco.offline import (
    SplitConfig,
    StandardNormalization,
    WindowConfig,
    WindowDataset,
    get_integration,
)


class MujocoStateDataset:
    """Use backend windowing, splits, and normalization without simulator imports.

    The only batch conversion is replacing ``obs_hist['state']`` with the
    state array expected by the low-dimensional Action Bridge model.
    """

    def __init__(
        self,
        cache_root: str | Path,
        *,
        integration: str,
        split: str,
        window_config: WindowConfig,
        split_config: SplitConfig,
        successful_only: bool = True,
        normalize: bool = True,
        normalization: StandardNormalization | Mapping[str, object] | None = None,
        normalization_eps: float = 1e-6,
    ) -> None:
        self.integration = get_integration(integration)
        self.spec = self.integration.spec
        state_spec = self.spec.observations.get("state")
        if state_spec is None or len(state_spec.shape) != 1:
            raise ValueError("Action Bridge requires a vector 'state' observation")
        if len(self.spec.action.shape) != 1:
            raise ValueError("Action Bridge requires vector actions")
        self.obs_dim = state_spec.shape[0]
        self.action_dim = self.spec.action.shape[0]
        self.windows = WindowDataset(
            cache_root,
            integration=self.integration,
            split=split,
            window_config=window_config,
            split_config=split_config,
            successful_only=successful_only,
            observation_modalities=("state",),
            normalize=normalize,
            normalization_modality="state",
            normalization=normalization,
            normalization_eps=normalization_eps,
        )
        self.bundle = self.windows.bundle
        self.window_config = self.windows.window_config
        self.split_plan = self.windows.split_plan
        self.split = self.windows.split
        self.episode_indices = self.windows.episode_indices
        self.normalization = self.windows.normalization
        self.normalization_stats = (
            {
                **self.normalization.to_dict(),
                "type": "standard",
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
            }
            if self.normalization is not None
            else None
        )

    @staticmethod
    def _model_batch(item: dict[str, Any]) -> dict[str, Any]:
        return {**item, "obs_hist": item["obs_hist"]["state"]}

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._model_batch(self.windows[index])

    def item_from_episode_time(
        self, episode_index: int, time_index: int
    ) -> dict[str, Any]:
        return self._model_batch(
            self.windows.item_from_episode_time(episode_index, time_index)
        )

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> dict[str, Any]:
        return self._model_batch(self.windows.sample_batch(batch_size, rng))

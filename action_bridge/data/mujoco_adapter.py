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
from tqdm.auto import tqdm


class MujocoStateDataset:
    """Keep the backend's low-dimensional training windows in CPU RAM.

    The backend defines splits, normalization, padding, and action alignment.
    Materialize its outputs once so minibatches never reread HDF5 episodes.
    Only ``obs_hist['state']`` is flattened into the model's batch format.
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
        progress: bool = False,
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

        first = self._model_batch(self.windows[0])
        self._arrays = {
            name: np.empty((len(self.windows), *value.shape), dtype=value.dtype)
            for name, value in first.items()
        }
        for index in tqdm(
            range(len(self.windows)),
            desc=f"Caching {split} windows in RAM",
            unit="window",
            disable=not progress,
        ):
            item = first if index == 0 else self._model_batch(self.windows[index])
            for name, value in item.items():
                self._arrays[name][index] = value
        self._window_indices = {
            (int(episode), int(time)): index
            for index, (episode, time) in enumerate(
                zip(self._arrays["episode_index"], self._arrays["time_index"])
            )
        }

    @property
    def cached_nbytes(self) -> int:
        """Array storage only, excluding small Python metadata objects."""

        return sum(array.nbytes for array in self._arrays.values())

    @staticmethod
    def _model_batch(item: dict[str, Any]) -> dict[str, Any]:
        return {**item, "obs_hist": item["obs_hist"]["state"]}

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Return independent writable arrays, as the backend does.
        return {
            name: np.array(array[index], copy=True)
            for name, array in self._arrays.items()
        }

    def item_from_episode_time(
        self, episode_index: int, time_index: int
    ) -> dict[str, Any]:
        try:
            index = self._window_indices[episode_index, time_index]
        except KeyError as exc:
            raise ValueError(
                f"no cached window for episode {episode_index}, time {time_index} "
                f"in split {self.split!r}"
            ) from exc
        return self[index]

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> dict[str, Any]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = rng.integers(0, len(self), size=batch_size)
        return {name: array[indices] for name, array in self._arrays.items()}

"""NumPy RLBench windows with the same semantics as ``RLBenchDataset``."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zlib

import numpy as np

from action_bridge.data.rlbench_cache import (
    EpisodeKey,
    RLBenchCacheStore,
    VariationKey,
    build_cache_keys,
)


SUPPORTED_ACTION_REPRESENTATIONS = ("absolute", "delta_xyz")


def normalize_action_representation(value: str) -> str:
    value = str(value).strip().lower()
    if value not in SUPPORTED_ACTION_REPRESENTATIONS:
        raise ValueError(
            f"action_representation must be one of {SUPPORTED_ACTION_REPRESENTATIONS}, "
            f"got {value!r}."
        )
    return value


def encode_action_chunk(
    action_chunk: np.ndarray,
    *,
    observation_state: np.ndarray,
    representation: str,
) -> np.ndarray:
    representation = normalize_action_representation(representation)
    actions = np.asarray(action_chunk, dtype=np.float32)
    if representation == "absolute":
        return actions
    if actions.ndim != 2 or actions.shape[-1] < 3:
        raise ValueError(f"Expected action chunk [H, A>=3], got {actions.shape}.")
    state = np.asarray(observation_state, dtype=np.float32)
    if state.ndim != 2 or state.shape[-1] < 3:
        raise ValueError(f"Expected observation state [T_obs, S>=3], got {state.shape}.")
    output = actions.copy()
    output[0, :3] = actions[0, :3] - state[-1, :3]
    if actions.shape[0] > 1:
        output[1:, :3] = actions[1:, :3] - actions[:-1, :3]
    return output


def encode_action_history(
    action_history: np.ndarray,
    *,
    previous_actions: np.ndarray,
    representation: str,
) -> np.ndarray:
    representation = normalize_action_representation(representation)
    actions = np.asarray(action_history, dtype=np.float32)
    if representation == "absolute":
        return actions
    previous = np.asarray(previous_actions, dtype=np.float32)
    if actions.shape != previous.shape or actions.shape[-1] < 3:
        raise ValueError(
            f"Expected matching action histories [T, A>=3], got "
            f"{actions.shape} and {previous.shape}."
        )
    output = actions.copy()
    output[:, :3] = actions[:, :3] - previous[:, :3]
    return output


def split_episode_ids(
    episode_ids: Sequence[int] | np.ndarray,
    split: str,
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> List[int]:
    ids = np.asarray(episode_ids, dtype=np.int64)
    if split == "all":
        return sorted(int(value) for value in ids)
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split {split!r}; expected train, val, test, or all.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 <= val_fraction < 1.0 or train_fraction + val_fraction >= 1.0:
        raise ValueError("val_fraction must be non-negative and train_fraction + val_fraction < 1.")

    ids = ids[np.random.default_rng(int(seed)).permutation(len(ids))]
    count = len(ids)
    n_train = int(np.floor(train_fraction * count))
    n_val = int(np.floor(val_fraction * count))
    if count >= 3:
        n_train = min(max(1, n_train), count - 2)
        n_val = min(max(1, n_val), count - n_train - 1)
    elif count == 2:
        n_train, n_val = 1, 1
    elif count == 1:
        n_train, n_val = 1, 0

    if split == "train":
        selected = ids[:n_train]
    elif split == "val":
        selected = ids[n_train : n_train + n_val]
    else:
        selected = ids[n_train + n_val :]
    return sorted(int(value) for value in selected)


@dataclass(frozen=True)
class RLBenchWindowKey:
    variation_index: int
    episode_id: int
    time_index: int


class NumpyRLBenchDataset:
    """Framework-neutral standard imitation-learning windows from the HDF5 cache."""

    def __init__(
        self,
        cache_root: str,
        *,
        split: str = "train",
        tasks: Optional[Sequence[str]] = None,
        exclude_tasks: Sequence[str] = (),
        variation_ids: Optional[Sequence[int]] = None,
        obs_history: int = 2,
        action_history: int = 2,
        chunk_horizon: int = 16,
        obs_stride: int = 1,
        action_stride: int = 1,
        action_offset: int = 1,
        action_representation: str = "absolute",
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        split_seed: int = 0,
        pad_episode_starts: bool = True,
        pad_episode_ends: bool = False,
        include_rgb: bool = True,
        include_mask_id: bool = False,
        extra_observation_fields: Sequence[str] = (),
        point_count: Optional[int] = None,
        point_sampling: str = "random",
        point_sampling_seed: int = 0,
        max_episodes_per_variation: Optional[int] = None,
        keep_h5_open: bool = True,
        preload_to_memory: bool = False,
        return_absolute_actions: bool = False,
        store: Optional[RLBenchCacheStore] = None,
    ):
        for name, value in (
            ("obs_history", obs_history),
            ("action_history", action_history),
            ("chunk_horizon", chunk_horizon),
            ("obs_stride", obs_stride),
            ("action_stride", action_stride),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive, got {value}.")
        if int(action_offset) < 0:
            raise ValueError("action_offset must be non-negative.")
        if point_sampling not in {"random", "first"}:
            raise ValueError("point_sampling must be 'random' or 'first'.")

        if store is None:
            keys, selected_tasks = build_cache_keys(
                cache_root,
                tasks,
                exclude_tasks=exclude_tasks,
                variation_ids=variation_ids,
            )
            store = RLBenchCacheStore(
                keys,
                keep_open=keep_h5_open,
                preload_to_memory=preload_to_memory,
            )
        else:
            keys = store.keys
            selected_tasks = sorted({key.task for key in keys})

        self.store = store
        self.keys: List[VariationKey] = list(keys)
        self.selected_tasks = tuple(selected_tasks)
        self.split = str(split)
        self.obs_history = int(obs_history)
        self.action_history = int(action_history)
        self.chunk_horizon = int(chunk_horizon)
        self.obs_stride = int(obs_stride)
        self.action_stride = int(action_stride)
        self.action_offset = int(action_offset)
        self.action_representation = normalize_action_representation(action_representation)
        self.pad_episode_starts = bool(pad_episode_starts)
        self.pad_episode_ends = bool(pad_episode_ends)
        self.include_rgb = bool(include_rgb)
        self.include_mask_id = bool(include_mask_id)
        self.extra_observation_fields = tuple(str(name) for name in extra_observation_fields)
        self.point_count = None if point_count is None else int(point_count)
        self.point_sampling = str(point_sampling)
        self.point_sampling_seed = int(point_sampling_seed)
        self.return_absolute_actions = bool(return_absolute_actions)
        self._epoch = 0

        dimensions = [self.store.variation_dims(index) for index in range(len(self.keys))]
        if len(set(dimensions)) != 1:
            raise ValueError(f"Selected RLBench caches have inconsistent dimensions: {dimensions}")
        cached_points, self.state_dim, self.action_dim = dimensions[0]
        self.cached_point_count = int(cached_points)
        if self.point_count is None:
            self.point_count = self.cached_point_count
        if not 1 <= self.point_count <= self.cached_point_count:
            raise ValueError(
                f"point_count={self.point_count} must be in [1, {self.cached_point_count}]."
            )

        self.task_to_id = {task: index for index, task in enumerate(sorted(self.selected_tasks))}
        self.task_variation_to_id = {
            f"{key.task}:{key.variation}": index for index, key in enumerate(self.keys)
        }
        required_fields = {"xyz", "valid", "state", "action", *self.extra_observation_fields}
        if self.include_rgb:
            required_fields.add("rgb")
        if self.include_mask_id:
            required_fields.add("mask_id")
        for variation_index, key in enumerate(self.keys):
            episode_ids = self.store.list_episode_ids(variation_index)
            if not len(episode_ids):
                raise ValueError(f"{key.task}/variation{key.variation} has no episodes.")
            available = set(self.store.available_fields(variation_index, int(episode_ids[0])))
            missing = sorted(required_fields - available)
            if missing:
                raise ValueError(
                    f"{key.task}/variation{key.variation} is missing requested fields {missing}."
                )

        self.episodes: List[EpisodeKey] = []
        self.indices: List[RLBenchWindowKey] = []
        for variation_index, key in enumerate(self.keys):
            episode_ids = split_episode_ids(
                self.store.list_episode_ids(variation_index),
                self.split,
                train_fraction=float(train_fraction),
                val_fraction=float(val_fraction),
                seed=int(split_seed) + zlib.crc32(f"{key.task}:{key.variation}".encode("utf-8")),
            )
            if max_episodes_per_variation is not None:
                episode_ids = episode_ids[: int(max_episodes_per_variation)]
            for episode_id in episode_ids:
                episode = EpisodeKey(
                    variation_index=variation_index,
                    task=key.task,
                    variation=key.variation,
                    episode_id=episode_id,
                    length=self.store.episode_length(variation_index, episode_id),
                )
                windows = self._episode_windows(episode)
                if windows:
                    self.episodes.append(episode)
                    self.indices.extend(windows)
        if not self.indices:
            raise ValueError(f"No valid RLBench windows for split={self.split!r}.")

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def sampling_weights(self, strategy: str = "window_uniform") -> np.ndarray:
        if strategy == "window_uniform":
            return np.ones(len(self.indices), dtype=np.float64)
        if strategy == "episode_uniform":
            groups = [(window.variation_index, window.episode_id) for window in self.indices]
        elif strategy == "variation_uniform":
            groups = [window.variation_index for window in self.indices]
        elif strategy == "task_uniform":
            groups = [self.keys[window.variation_index].task for window in self.indices]
        else:
            raise ValueError(
                "sampling strategy must be window_uniform, episode_uniform, "
                "variation_uniform, or task_uniform."
            )
        counts = Counter(groups)
        return np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)

    def close(self) -> None:
        self.store.close()

    def _episode_windows(self, episode: EpisodeKey) -> List[RLBenchWindowKey]:
        min_time = 0
        if not self.pad_episode_starts:
            min_time = max(
                (self.obs_history - 1) * self.obs_stride,
                self.action_history * self.action_stride - self.action_offset,
            )
        max_time = episode.length - 1 - self.action_offset
        if not self.pad_episode_ends:
            max_time -= (self.chunk_horizon - 1) * self.action_stride
        if max_time < min_time:
            return []
        return [
            RLBenchWindowKey(episode.variation_index, episode.episode_id, time_index)
            for time_index in range(min_time, max_time + 1)
        ]

    @staticmethod
    def _clipped_indices(raw_indices: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
        valid = (raw_indices >= 0) & (raw_indices < int(length))
        clipped = np.clip(raw_indices, 0, int(length) - 1).astype(np.int64)
        return clipped, valid.astype(np.bool_)

    def _window_indices(self, time_index: int, episode_length: int):
        obs_raw = time_index - np.arange(
            self.obs_history - 1, -1, -1, dtype=np.int64
        ) * self.obs_stride
        future_start = int(time_index) + self.action_offset
        history_raw = future_start - np.arange(
            self.action_history, 0, -1, dtype=np.int64
        ) * self.action_stride
        future_raw = future_start + np.arange(
            self.chunk_horizon, dtype=np.int64
        ) * self.action_stride
        obs, obs_mask = self._clipped_indices(obs_raw, episode_length)
        history, history_mask = self._clipped_indices(history_raw, episode_length)
        future, future_mask = self._clipped_indices(future_raw, episode_length)
        return obs, obs_mask, history, history_mask, future, future_mask

    def _point_indices(self, dataset_index: int, valid: np.ndarray) -> np.ndarray:
        if self.point_count == self.cached_point_count:
            return np.arange(self.cached_point_count, dtype=np.int64)
        if self.point_sampling == "first":
            return np.arange(self.point_count, dtype=np.int64)
        rng = np.random.default_rng(
            np.random.SeedSequence([self.point_sampling_seed, self._epoch, int(dataset_index)])
        )
        usable = np.flatnonzero(np.asarray(valid, dtype=np.bool_).any(axis=0))
        if usable.size == 0:
            usable = np.arange(self.cached_point_count, dtype=np.int64)
        return np.asarray(
            rng.choice(usable, size=self.point_count, replace=usable.size < self.point_count),
            dtype=np.int64,
        )

    def item_from_episode_time(
        self,
        variation_index: int,
        episode_id: int,
        time_index: int,
        *,
        dataset_index: int = 0,
    ) -> Dict[str, Any]:
        length = self.store.episode_length(variation_index, episode_id)
        obs_idx, obs_mask, hist_idx, hist_mask, future_idx, future_mask = self._window_indices(
            int(time_index), length
        )
        observation_fields = ["xyz", "valid", "state"]
        if self.include_rgb:
            observation_fields.append("rgb")
        if self.include_mask_id:
            observation_fields.append("mask_id")
        observation_fields.extend(self.extra_observation_fields)
        observations = self.store.load_episode_slices(
            variation_index,
            episode_id,
            obs_idx,
            fields=list(dict.fromkeys(observation_fields)),
        )
        history = self.store.load_episode_slices(
            variation_index, episode_id, hist_idx, fields=("action",)
        )["action"].astype(np.float32)
        previous_history = self.store.load_episode_slices(
            variation_index,
            episode_id,
            np.clip(hist_idx - self.action_stride, 0, length - 1),
            fields=("action",),
        )["action"].astype(np.float32)
        history = encode_action_history(
            history,
            previous_actions=previous_history,
            representation=self.action_representation,
        )
        future_absolute = self.store.load_episode_slices(
            variation_index, episode_id, future_idx, fields=("action",)
        )["action"].astype(np.float32)
        state = observations["state"].astype(np.float32)
        future = encode_action_chunk(
            future_absolute,
            observation_state=state,
            representation=self.action_representation,
        )
        point_indices = self._point_indices(dataset_index, observations["valid"])
        key = self.keys[int(variation_index)]
        task_variation_name = f"{key.task}:{key.variation}"
        output: Dict[str, Any] = {
            "obs_hist": state,
            "point_cloud_hist": observations["xyz"][:, point_indices].astype(np.float32),
            "point_valid_hist": observations["valid"][:, point_indices].astype(np.bool_),
            "act_hist": history,
            "future_actions": future,
            "obs_history_mask": obs_mask,
            "action_history_mask": hist_mask,
            "future_action_mask": future_mask,
            "action_is_absolute": np.asarray(
                self.action_representation == "absolute", dtype=np.bool_
            ),
            "task_id": np.asarray(self.task_to_id[key.task], dtype=np.int32),
            "task_variation_id": np.asarray(
                self.task_variation_to_id[task_variation_name], dtype=np.int32
            ),
            "variation_id": np.asarray(key.variation, dtype=np.int32),
            "episode_id": np.asarray(episode_id, dtype=np.int32),
            "time_index": np.asarray(time_index, dtype=np.int32),
        }
        if self.return_absolute_actions and self.action_representation != "absolute":
            output["future_actions_absolute"] = future_absolute
        if self.include_rgb:
            output["rgb_hist"] = (
                observations["rgb"][:, point_indices].astype(np.float32) / 255.0
            )
        if self.include_mask_id:
            output["mask_id_hist"] = observations["mask_id"][:, point_indices].astype(np.int32)
        for field in self.extra_observation_fields:
            output[f"low_dim/{field}"] = np.asarray(observations[field])
        return output

    def __getitem__(self, index: int) -> Dict[str, Any]:
        window = self.indices[int(index)]
        return self.item_from_episode_time(
            window.variation_index,
            window.episode_id,
            window.time_index,
            dataset_index=int(index),
        )

    def sample_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        strategy: str = "window_uniform",
    ) -> Dict[str, np.ndarray]:
        weights = self.sampling_weights(strategy)
        probabilities = weights / weights.sum()
        indices = rng.choice(len(self), size=int(batch_size), replace=True, p=probabilities)
        items = [self[int(index)] for index in indices]
        return {
            key: np.stack([np.asarray(item[key]) for item in items], axis=0)
            for key in items[0]
        }

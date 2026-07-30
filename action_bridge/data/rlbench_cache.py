"""Worker-safe access to dense, per-episode RLBench HDF5 caches."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np


CACHE_SCHEMA_NAME = "action_bridge.rlbench_dense"
CACHE_SCHEMA_VERSION = 1
CORE_EPISODE_FIELDS = ("xyz", "valid", "state", "action")
OPTIONAL_POINT_FIELDS = ("rgb", "mask_id")


@dataclass(frozen=True)
class VariationKey:
    """Location and identity of one cached task variation."""

    task: str
    variation: int
    path: str


@dataclass(frozen=True)
class EpisodeKey:
    """Identity and length of one episode in a cache store."""

    variation_index: int
    task: str
    variation: int
    episode_id: int
    length: int


def _variation_number(path: Path) -> int:
    suffix = path.stem.removeprefix("variation")
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(f"Expected variation<number>.h5, got {path.name!r}.") from exc


def discover_tasks(cache_root: Path | str) -> List[str]:
    """Return task directories containing at least one variation cache."""

    root = Path(cache_root)
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and any(path.glob("variation*.h5")))


def build_variation_keys(
    cache_root: Path | str,
    task: str,
    variation_ids: Optional[Sequence[int]] = None,
) -> List[VariationKey]:
    """Discover variation files for one task."""

    allowed = None if variation_ids is None else {int(value) for value in variation_ids}
    paths = sorted((Path(cache_root) / task).glob("variation*.h5"), key=_variation_number)
    keys = []
    for path in paths:
        variation = _variation_number(path)
        if allowed is None or variation in allowed:
            keys.append(VariationKey(task=str(task), variation=variation, path=str(path)))
    return keys


def build_cache_keys(
    cache_root: Path | str,
    tasks: Optional[Sequence[str]] = None,
    *,
    exclude_tasks: Sequence[str] = (),
    variation_ids: Optional[Sequence[int]] = None,
) -> Tuple[List[VariationKey], List[str]]:
    """Discover selected task/variation cache files."""

    root = Path(cache_root)
    if not root.is_dir():
        raise FileNotFoundError(f"RLBench cache root not found: {root}")
    selected = list(tasks) if tasks else discover_tasks(root)
    excluded = {str(task) for task in exclude_tasks}
    selected = [str(task) for task in selected if str(task) not in excluded]
    if not selected:
        raise RuntimeError(f"No RLBench tasks selected under {root}.")

    keys: List[VariationKey] = []
    missing: List[str] = []
    for task in selected:
        task_keys = build_variation_keys(root, task, variation_ids=variation_ids)
        if not task_keys:
            missing.append(task)
        keys.extend(task_keys)
    if missing:
        raise RuntimeError(f"No selected variation cache files found for tasks: {missing}")
    return keys, selected


class RLBenchCacheStore:
    """Read cached RLBench episodes without baking in a sampling strategy.

    HDF5 handles are opened lazily in each process. This makes the store safe to
    use through a PyTorch ``DataLoader`` with multiprocessing workers.
    """

    def __init__(
        self,
        keys: Sequence[VariationKey],
        *,
        keep_open: bool = True,
        preload_to_memory: bool = False,
    ):
        if not keys:
            raise ValueError("RLBenchCacheStore requires at least one variation key.")
        self.keys = list(keys)
        self.keep_open = bool(keep_open) and not bool(preload_to_memory)
        self.preload_to_memory = bool(preload_to_memory)
        self.task_names = tuple(sorted({key.task for key in self.keys}))
        self.task_to_id = {task: index for index, task in enumerate(self.task_names)}
        self.task_variation_names = tuple(f"{key.task}:{key.variation}" for key in self.keys)
        self.task_variation_to_id = {
            name: index for index, name in enumerate(self.task_variation_names)
        }
        self._handles: Dict[int, h5py.File] = {}
        self._handle_pid = os.getpid()
        self._preloaded: Dict[int, Dict[str, Any]] = {}
        self.preloaded_bytes = 0
        self._validate_files()
        if self.preload_to_memory:
            self._preload_all()

    def __len__(self) -> int:
        return len(self.keys)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = {}
        state["_handle_pid"] = None
        return state

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        for handle in list(getattr(self, "_handles", {}).values()):
            try:
                handle.close()
            except Exception:
                pass
        self._handles = {}

    def _ensure_process_local_handles(self) -> None:
        pid = os.getpid()
        if self._handle_pid != pid:
            self.close()
            self._handle_pid = pid

    def _validate_files(self) -> None:
        for key in self.keys:
            path = Path(key.path)
            if not path.is_file():
                raise FileNotFoundError(f"RLBench variation cache not found: {path}")
            with h5py.File(path, "r") as handle:
                missing = [name for name in ("episode_ids", "episodes") if name not in handle]
                if missing:
                    raise ValueError(f"Invalid RLBench cache {path}: missing {missing}.")
                cached_task = str(handle.attrs.get("task", key.task))
                cached_variation = int(handle.attrs.get("variation", key.variation))
                if cached_task != key.task or cached_variation != key.variation:
                    raise ValueError(
                        f"Cache identity mismatch for {path}: expected "
                        f"{key.task}/variation{key.variation}, found "
                        f"{cached_task}/variation{cached_variation}."
                    )

    def _handle(self, variation_index: int) -> h5py.File:
        self._ensure_process_local_handles()
        variation_index = int(variation_index)
        if self.keep_open:
            handle = self._handles.get(variation_index)
            if handle is None:
                handle = h5py.File(self.keys[variation_index].path, "r")
                self._handles[variation_index] = handle
            return handle
        return h5py.File(self.keys[variation_index].path, "r")

    @contextmanager
    def _temporary_or_kept_handle(self, variation_index: int) -> Iterator[h5py.File]:
        handle = self._handle(variation_index)
        try:
            yield handle
        finally:
            if not self.keep_open:
                handle.close()

    @staticmethod
    def _read_rows(dataset: h5py.Dataset | np.ndarray, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return np.asarray(dataset[indices])
        if indices.size == 1 or np.all(indices[1:] > indices[:-1]):
            return np.asarray(dataset[indices])
        unique, inverse = np.unique(indices, return_inverse=True)
        return np.asarray(dataset[unique])[inverse]

    @staticmethod
    def _episode_group(handle: h5py.File, episode_id: int) -> h5py.Group:
        name = str(int(episode_id))
        if name not in handle["episodes"]:
            raise KeyError(f"Episode {episode_id} is not present in {handle.filename}.")
        return handle["episodes"][name]

    def _preload_one(self, variation_index: int) -> Dict[str, Any]:
        output: Dict[str, Any] = {"episodes": {}}
        with h5py.File(self.keys[variation_index].path, "r") as handle:
            episode_ids = np.asarray(handle["episode_ids"][:], dtype=np.int64)
            output["episode_ids"] = episode_ids
            output["attrs"] = dict(handle.attrs)
            self.preloaded_bytes += episode_ids.nbytes
            for episode_id in episode_ids:
                group = self._episode_group(handle, int(episode_id))
                episode: Dict[str, Any] = {"attrs": dict(group.attrs)}
                for name, value in group.items():
                    if isinstance(value, h5py.Dataset):
                        array = np.asarray(value[:])
                        episode[name] = array
                        self.preloaded_bytes += array.nbytes
                output["episodes"][int(episode_id)] = episode
        return output

    def _preload_all(self) -> None:
        for variation_index in range(len(self.keys)):
            self._preloaded[variation_index] = self._preload_one(variation_index)

    def variation_metadata(self, variation_index: int) -> Dict[str, Any]:
        variation_index = int(variation_index)
        if self.preload_to_memory:
            return dict(self._preloaded[variation_index]["attrs"])
        with self._temporary_or_kept_handle(variation_index) as handle:
            return dict(handle.attrs)

    def list_episode_ids(self, variation_index: int) -> np.ndarray:
        variation_index = int(variation_index)
        if self.preload_to_memory:
            return np.asarray(self._preloaded[variation_index]["episode_ids"], dtype=np.int64)
        with self._temporary_or_kept_handle(variation_index) as handle:
            return np.asarray(handle["episode_ids"][:], dtype=np.int64)

    def episode_metadata(self, variation_index: int, episode_id: int) -> Dict[str, Any]:
        variation_index = int(variation_index)
        episode_id = int(episode_id)
        if self.preload_to_memory:
            return dict(self._preloaded[variation_index]["episodes"][episode_id]["attrs"])
        with self._temporary_or_kept_handle(variation_index) as handle:
            return dict(self._episode_group(handle, episode_id).attrs)

    def episode_length(self, variation_index: int, episode_id: int) -> int:
        metadata = self.episode_metadata(variation_index, episode_id)
        if "T" in metadata:
            return int(metadata["T"])
        if self.preload_to_memory:
            episode = self._preloaded[int(variation_index)]["episodes"][int(episode_id)]
            return int(episode["action"].shape[0])
        with self._temporary_or_kept_handle(int(variation_index)) as handle:
            return int(self._episode_group(handle, int(episode_id))["action"].shape[0])

    def available_fields(self, variation_index: int, episode_id: int) -> Tuple[str, ...]:
        variation_index = int(variation_index)
        episode_id = int(episode_id)
        if self.preload_to_memory:
            episode = self._preloaded[variation_index]["episodes"][episode_id]
            return tuple(sorted(name for name in episode if name != "attrs"))
        with self._temporary_or_kept_handle(variation_index) as handle:
            group = self._episode_group(handle, episode_id)
            return tuple(sorted(name for name, value in group.items() if isinstance(value, h5py.Dataset)))

    def load_episode_slices(
        self,
        variation_index: int,
        episode_id: int,
        indices: Sequence[int] | np.ndarray,
        *,
        fields: Sequence[str] = CORE_EPISODE_FIELDS,
    ) -> Dict[str, np.ndarray]:
        """Load arbitrary, possibly repeated frame indices from an episode."""

        variation_index = int(variation_index)
        episode_id = int(episode_id)
        row_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        length = self.episode_length(variation_index, episode_id)
        if row_indices.size and (row_indices.min() < 0 or row_indices.max() >= length):
            raise IndexError(
                f"Frame indices [{row_indices.min()}, {row_indices.max()}] are outside "
                f"episode length {length}."
            )

        if self.preload_to_memory:
            episode = self._preloaded[variation_index]["episodes"][episode_id]
            missing = [field for field in fields if field not in episode]
            if missing:
                raise KeyError(f"Episode is missing requested fields: {missing}")
            return {
                field: self._read_rows(episode[field], row_indices)
                for field in fields
            }

        def read(group: h5py.Group) -> Dict[str, np.ndarray]:
            missing = [field for field in fields if field not in group]
            if missing:
                raise KeyError(f"{group.name} is missing requested fields: {missing}")
            return {
                field: self._read_rows(group[field], row_indices)
                for field in fields
            }

        with self._temporary_or_kept_handle(variation_index) as handle:
            return read(self._episode_group(handle, episode_id))

    def iter_episode_keys(self) -> Iterator[EpisodeKey]:
        for variation_index, key in enumerate(self.keys):
            for episode_id in self.list_episode_ids(variation_index):
                yield EpisodeKey(
                    variation_index=variation_index,
                    task=key.task,
                    variation=key.variation,
                    episode_id=int(episode_id),
                    length=self.episode_length(variation_index, int(episode_id)),
                )

    def infer_dims(self) -> Tuple[int, int, int]:
        """Return ``(cached_points, state_dim, action_dim)``."""

        for episode in self.iter_episode_keys():
            return self.variation_dims(episode.variation_index)
        raise RuntimeError("The selected RLBench cache contains no episodes.")

    def variation_dims(self, variation_index: int) -> Tuple[int, int, int]:
        """Return dimensions from the first episode in one variation."""

        episode_ids = self.list_episode_ids(variation_index)
        if not len(episode_ids):
            raise RuntimeError(
                f"{self.keys[int(variation_index)].path} contains no episodes."
            )
        sample = self.load_episode_slices(
            variation_index,
            int(episode_ids[0]),
            [0],
            fields=("xyz", "state", "action"),
        )
        return (
            int(sample["xyz"].shape[1]),
            int(sample["state"].shape[-1]),
            int(sample["action"].shape[-1]),
        )

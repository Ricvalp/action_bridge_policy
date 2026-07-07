"""Low-dimensional Push-T adapter.

The adapter intentionally avoids vendoring external Push-T code. It reads local
offline low-dimensional datasets and emits the common action-bridge batch
schema used by the toy benchmarks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


OBS_KEYS = [
    "obs",
    "observations",
    "observation",
    "state",
    "states",
    "lowdim",
    "lowdim_obs",
    "data/obs",
    "data/observations",
    "data/state",
    "data/states",
    "data/lowdim",
    "data/lowdim_obs",
    "data/keypoint",
    "data/keypoints",
]
ACTION_KEYS = ["action", "actions", "data/action", "data/actions"]
EPISODE_END_KEYS = ["episode_ends", "episode_end", "ends", "meta/episode_ends"]
EPISODE_LENGTH_KEYS = ["episode_lengths", "lengths", "meta/episode_lengths"]


def _setup_error() -> RuntimeError:
    return RuntimeError(
        "Push-T lowdim data was not found. Provide data.dataset_path pointing to "
        "a local Diffusion Policy zarr dataset or a local .npz/.pt export with "
        "observations, actions, and episode_ends. This project does not vendor "
        "external repositories or datasets."
    )


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _lookup(mapping: Dict[str, Any], candidates: Sequence[str], explicit: Optional[str] = None) -> Any:
    if explicit:
        if explicit not in mapping:
            raise KeyError(f"Requested key {explicit!r} was not found. Available keys: {sorted(mapping)}")
        return mapping[explicit]
    for key in candidates:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"None of the expected keys {list(candidates)} were found. Available keys: {sorted(mapping)}")


def _lookup_optional(mapping: Dict[str, Any], candidates: Sequence[str], explicit: Optional[str] = None) -> Optional[Any]:
    if explicit:
        if explicit not in mapping:
            raise KeyError(f"Requested key {explicit!r} was not found. Available keys: {sorted(mapping)}")
        return mapping[explicit]
    for key in candidates:
        if key in mapping:
            return mapping[key]
    return None


def _flatten_loaded_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_loaded_dict(value, name))
        else:
            flat[name] = value
            flat[str(key)] = value
    return flat


def _zarr_arrays(group, prefix: str = "") -> Dict[str, Any]:
    arrays: Dict[str, Any] = {}
    if hasattr(group, "items"):
        members = group.items()
    elif hasattr(group, "keys"):
        members = ((name, group[name]) for name in group.keys())
    else:
        raise TypeError(f"Unsupported zarr group object {type(group)!r}.")
    for name, value in members:
        key = f"{prefix}/{name}" if prefix else name
        if hasattr(value, "shape") and hasattr(value, "__getitem__"):
            arrays[key] = value
            arrays[name] = value
        elif hasattr(value, "items") or hasattr(value, "keys"):
            arrays.update(_zarr_arrays(value, key))
    return arrays


def _load_zarr(path: Path) -> Dict[str, Any]:
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("Loading Push-T zarr data requires `zarr`. Run `uv sync` from the sandbox.") from exc
    root = zarr.open(str(path), mode="r")
    return _zarr_arrays(root)


def _load_npz(path: Path) -> Dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def _load_torch(path: Path) -> Dict[str, Any]:
    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(loaded)!r}.")
    return _flatten_loaded_dict(loaded)


def _load_arrays(path: Path, backend: str) -> Dict[str, Any]:
    if backend == "auto":
        if path.suffix == ".zarr" or path.is_dir():
            backend = "zarr"
        elif path.suffix == ".npz":
            backend = "npz"
        elif path.suffix in {".pt", ".pth"}:
            backend = "torch"
        else:
            raise ValueError(f"Could not infer Push-T backend for {path}. Use data.backend=zarr|npz|torch.")
    if backend in {"zarr", "diffusion_policy"}:
        return _load_zarr(path)
    if backend == "npz":
        return _load_npz(path)
    if backend in {"torch", "pt"}:
        return _load_torch(path)
    if backend == "lerobot":
        raise NotImplementedError(
            "Direct LeRobot loading is not implemented in this lightweight adapter. "
            "Export the local LeRobot Push-T dataset to .npz/.pt with observation/state, action, "
            "and episode_ends arrays, then use data.backend=npz or data.backend=torch."
        )
    raise ValueError(f"Unsupported Push-T backend {backend!r}.")


def _episode_ends_from_lengths(lengths: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(lengths, dtype=np.int64))


def _as_episode_tensors(obs: Any, actions: Any, episode_ends: Optional[Any] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    obs_np = _to_numpy(obs).astype(np.float32)
    action_np = _to_numpy(actions).astype(np.float32)
    if obs_np.ndim == 3 and action_np.ndim == 3:
        count = min(obs_np.shape[0], action_np.shape[0])
        return [torch.from_numpy(obs_np[i]) for i in range(count)], [torch.from_numpy(action_np[i]) for i in range(count)]
    if obs_np.ndim != 2 or action_np.ndim != 2:
        raise ValueError(
            "Expected either [N,T,D] episode arrays or flat [T,D] arrays with episode_ends; "
            f"got obs shape {obs_np.shape}, action shape {action_np.shape}."
        )
    if episode_ends is None:
        raise ValueError("Flat Push-T arrays require episode_ends or episode_lengths.")
    ends = _to_numpy(episode_ends).astype(np.int64).reshape(-1)
    starts = np.concatenate([[0], ends[:-1]])
    obs_eps = []
    action_eps = []
    for start, end in zip(starts, ends):
        if end <= start:
            continue
        obs_eps.append(torch.from_numpy(obs_np[start:end]))
        action_eps.append(torch.from_numpy(action_np[start:end]))
    return obs_eps, action_eps


def _split_episode_ids(num_episodes: int, split: str, train_fraction: float, val_fraction: float) -> List[int]:
    if split == "all":
        return list(range(num_episodes))
    if num_episodes <= 0:
        return []
    if num_episodes < 3:
        return list(range(num_episodes))
    n_train = max(1, int(train_fraction * num_episodes))
    n_train = min(n_train, num_episodes - 2)
    remaining = num_episodes - n_train
    n_val = max(1, int(val_fraction * num_episodes))
    n_val = min(n_val, remaining - 1)
    if split == "train":
        return list(range(0, n_train))
    if split == "val":
        return list(range(n_train, n_train + n_val))
    if split == "test":
        return list(range(n_train + n_val, num_episodes))
    raise ValueError(f"Unknown split {split!r}; expected train, val, test, or all.")


def _valid_indices(
    observations: Sequence[torch.Tensor],
    actions: Sequence[torch.Tensor],
    episode_ids: Iterable[int],
    obs_history: int,
    action_history: int,
    chunk_horizon: int,
) -> List[Tuple[int, int]]:
    start_t = max(obs_history - 1, action_history)
    indices: List[Tuple[int, int]] = []
    for episode_id in episode_ids:
        length = min(int(observations[episode_id].shape[0]), int(actions[episode_id].shape[0]))
        end_t = length - chunk_horizon
        for t in range(start_t, end_t + 1):
            indices.append((int(episode_id), int(t)))
    return indices


class PushTLowDimDataset(Dataset):
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        backend: str = "auto",
        split: str = "train",
        obs_history: int = 2,
        action_history: int = 2,
        chunk_horizon: int = 16,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        obs_key: Optional[str] = None,
        action_key: Optional[str] = None,
        episode_ends_key: Optional[str] = None,
        max_episodes: Optional[int] = None,
    ):
        if dataset_path is None:
            raise _setup_error()
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Push-T dataset path does not exist: {path}")

        arrays = _load_arrays(path, backend)
        observations = _lookup(arrays, OBS_KEYS, explicit=obs_key)
        actions = _lookup(arrays, ACTION_KEYS, explicit=action_key)
        episode_ends = _lookup_optional(arrays, EPISODE_END_KEYS, explicit=episode_ends_key)
        if episode_ends is None:
            lengths = _lookup_optional(arrays, EPISODE_LENGTH_KEYS)
            episode_ends = _episode_ends_from_lengths(lengths) if lengths is not None else None

        obs_eps, action_eps = _as_episode_tensors(observations, actions, episode_ends)
        if max_episodes is not None:
            obs_eps = obs_eps[: int(max_episodes)]
            action_eps = action_eps[: int(max_episodes)]

        self.observations = obs_eps
        self.actions = action_eps
        self.obs_history = int(obs_history)
        self.action_history = int(action_history)
        self.chunk_horizon = int(chunk_horizon)
        self.episode_ids = _split_episode_ids(len(obs_eps), split, float(train_fraction), float(val_fraction))
        self.indices = _valid_indices(
            self.observations,
            self.actions,
            self.episode_ids,
            self.obs_history,
            self.action_history,
            self.chunk_horizon,
        )
        if not self.indices:
            raise ValueError(
                f"No valid Push-T chunks for split={split!r}. Need episodes longer than "
                f"max(obs_history-1, action_history)+chunk_horizon = "
                f"{max(self.obs_history - 1, self.action_history) + self.chunk_horizon}."
            )
        first_episode = self.indices[0][0]
        self.obs_dim = int(self.observations[first_episode].shape[-1])
        self.action_dim = int(self.actions[first_episode].shape[-1])
        self.dataset_path = str(path)
        self.backend = backend
        self.split = split

    def __len__(self) -> int:
        return len(self.indices)

    def item_from_episode_time(self, episode_id: int, t: int) -> Dict[str, Any]:
        obs = self.observations[episode_id]
        actions = self.actions[episode_id]
        obs_start = t - self.obs_history + 1
        act_start = t - self.action_history
        obs_hist = obs[obs_start : t + 1]
        act_hist = actions[act_start:t]
        future_actions = actions[t : t + self.chunk_horizon]
        context = {
            "traj_id": torch.tensor(episode_id, dtype=torch.long),
            "time_index": torch.tensor(t, dtype=torch.long),
        }
        return {
            "obs_hist": obs_hist.float(),
            "act_hist": act_hist.float(),
            "future_actions": future_actions.float(),
            "context": context,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode_id, t = self.indices[idx]
        return self.item_from_episode_time(episode_id, t)

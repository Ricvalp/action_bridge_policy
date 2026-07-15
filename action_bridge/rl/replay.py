"""Chunk-level replay buffer for Push-T bridge RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class ReplayBatch:
    obs_hist: torch.Tensor
    act_hist: torch.Tensor
    exec_actions: torch.Tensor
    planned_actions: torch.Tensor
    reward_m: torch.Tensor
    next_obs_hist: torch.Tensor
    next_act_hist: torch.Tensor
    done: torch.Tensor
    discount_m: torch.Tensor
    path_kl: torch.Tensor
    bc_cost: torch.Tensor
    success: torch.Tensor
    coverage_t: torch.Tensor
    coverage_tp: torch.Tensor


class ChunkReplayBuffer:
    """Simple ring buffer storing normalized model inputs and action chunks."""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(int(seed))
        self.storage: Dict[str, np.ndarray] = {}
        self.size = 0
        self.pos = 0

    def __len__(self) -> int:
        return int(self.size)

    def _ensure_field(self, key: str, value: np.ndarray) -> None:
        if key in self.storage:
            return
        shape = (self.capacity,) + tuple(value.shape)
        self.storage[key] = np.zeros(shape, dtype=value.dtype)

    def add(self, **transition) -> None:
        for key, value in transition.items():
            arr = np.asarray(value)
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            if arr.dtype == np.int64:
                arr = arr.astype(np.float32)
            self._ensure_field(key, arr)
            self.storage[key][self.pos] = arr
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        idx = self.rng.integers(0, self.size, size=int(batch_size))

        def tensor(key: str) -> torch.Tensor:
            return torch.from_numpy(self.storage[key][idx]).to(device=device, dtype=torch.float32)

        return ReplayBatch(
            obs_hist=tensor("obs_hist"),
            act_hist=tensor("act_hist"),
            exec_actions=tensor("exec_actions"),
            planned_actions=tensor("planned_actions"),
            reward_m=tensor("reward_m").view(-1),
            next_obs_hist=tensor("next_obs_hist"),
            next_act_hist=tensor("next_act_hist"),
            done=tensor("done").view(-1),
            discount_m=tensor("discount_m").view(-1),
            path_kl=tensor("path_kl").view(-1),
            bc_cost=tensor("bc_cost").view(-1),
            success=tensor("success").view(-1),
            coverage_t=tensor("coverage_t").view(-1),
            coverage_tp=tensor("coverage_tp").view(-1),
        )

    def save_npz(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: value[: self.size] for key, value in self.storage.items()}
        arrays["size"] = np.asarray([self.size], dtype=np.int64)
        arrays["pos"] = np.asarray([self.pos], dtype=np.int64)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load_npz(cls, path, capacity: Optional[int] = None, seed: int = 0) -> "ChunkReplayBuffer":
        raw = np.load(path)
        size = int(raw["size"][0])
        cap = int(capacity or max(size, 1))
        buffer = cls(capacity=cap, seed=seed)
        for key in raw.files:
            if key in {"size", "pos"}:
                continue
            arr = np.asarray(raw[key])
            field = np.zeros((cap,) + arr.shape[1:], dtype=arr.dtype)
            count = min(size, cap)
            field[:count] = arr[:count]
            buffer.storage[key] = field
        buffer.size = min(size, cap)
        buffer.pos = buffer.size % cap
        return buffer

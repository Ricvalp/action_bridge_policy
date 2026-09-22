"""Small, explicit data and action contracts shared by the revision experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch


def tree_map(fn, value):
    """Apply a tensor operation without assuming observations are flat vectors."""
    if isinstance(value, Mapping):
        return {key: tree_map(fn, item) for key, item in value.items()}
    return fn(value) if isinstance(value, torch.Tensor) else value


def take(batch, indices, device=None):
    return tree_map(lambda x: x[indices].to(device=device), batch)


@dataclass(frozen=True)
class ActionCodec:
    """Absolute Euclidean targets in explicit raw units; normalize exactly once.

    Delta targets, torques, rotations and categorical actions require another
    codec AND corresponding completion semantics, not just another dimension.
    """

    mean: list[float]
    std: list[float]
    low: list[float]
    high: list[float]
    units: str
    semantics: str = "absolute_target"

    def __post_init__(self):
        if self.semantics != "absolute_target":
            raise ValueError("This codec/completion pair supports absolute Euclidean targets only")
        if len({len(self.mean), len(self.std), len(self.low), len(self.high)}) != 1:
            raise ValueError("Action statistics and bounds must have the same dimension")
        if not self.mean or min(self.std) <= 0:
            raise ValueError("Action standard deviations must be positive")

    @property
    def raw_dim(self):
        return len(self.mean)

    @property
    def model_dim(self):
        return len(self.mean)

    def encode(self, raw):
        return (raw - raw.new_tensor(self.mean)) / raw.new_tensor(self.std)

    def decode(self, model):
        return model * model.new_tensor(self.std) + model.new_tensor(self.mean)

    def bound(self, raw):
        return raw.maximum(raw.new_tensor(self.low)).minimum(raw.new_tensor(self.high))

    def specification(self):
        return asdict(self)


class PlanCache:
    """One environment's prediction, distinct from its actually executed history."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.plan = None
        self.executed = 0

    @property
    def needs_bootstrap(self):
        return self.plan is None or self.executed >= self.plan.shape[1]

    def store(self, plan):
        if plan.ndim != 3 or plan.shape[0] != 1:
            raise ValueError("Use a separate PlanCache for each environment")
        self.plan = plan.detach().clone()
        self.executed = 0

    def advance(self, count=1):
        if self.plan is None or not 0 <= self.executed + count <= self.plan.shape[1]:
            raise ValueError("Executed count exceeds the cached plan")
        self.executed += count

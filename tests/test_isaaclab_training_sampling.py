from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ml_collections import ConfigDict

from action_bridge.training.train_isaaclab import (
    _first_close_sampling_weights,
    _training_sampler,
)


@dataclass(frozen=True)
class _Key:
    episode_index: int
    time_index: int


def _dataset(actions: list[float]):
    values = np.zeros((len(actions), 8), dtype=np.float32)
    values[:, -1] = np.asarray(actions, dtype=np.float32)
    episode = SimpleNamespace(
        episode_index=7,
        arrays=SimpleNamespace(actions=values),
    )
    return SimpleNamespace(
        episodes=(episode,),
        indices=tuple(_Key(7, time) for time in range(len(actions))),
        __len__=lambda self: len(actions),
    )


class _Dataset:
    def __init__(self, actions: list[float]):
        value = _dataset(actions)
        self.episodes = value.episodes
        self.indices = value.indices

    def __len__(self) -> int:
        return len(self.indices)


def test_first_close_sampling_weights_are_task_exact_and_opt_in():
    dataset = _Dataset([1.0, -1.0, -1.0, 1.0, -1.0])

    assert _first_close_sampling_weights(dataset, 1.0) is None
    weights = _first_close_sampling_weights(dataset, 16.0)

    assert weights is not None
    assert weights.dtype == torch.float64
    assert weights.tolist() == [1.0, 16.0, 1.0, 1.0, 16.0]


@pytest.mark.parametrize("weight", [0.0, 0.5, float("nan"), float("inf")])
def test_first_close_sampling_weight_rejects_invalid_values(weight: float):
    with pytest.raises(ValueError, match="finite and >= 1"):
        _first_close_sampling_weights(_Dataset([1.0, -1.0]), weight)


def test_first_close_sampling_rejects_a_transition_free_split():
    with pytest.raises(ValueError, match="no open-to-close transition"):
        _first_close_sampling_weights(_Dataset([1.0, 1.0, 1.0]), 4.0)


def test_weighted_sampler_is_seeded_and_reproducible():
    dataset = _Dataset([1.0, -1.0, -1.0, 1.0])
    config = ConfigDict(
        {
            "seed": 13,
            "data": {"first_close_sampling_weight": 8.0},
        }
    )

    first = list(_training_sampler(dataset, config))
    second = list(_training_sampler(dataset, config))

    assert first == second
    assert len(first) == len(dataset)


def test_uniform_sampling_returns_no_sampler():
    config = ConfigDict(
        {
            "seed": 0,
            "data": {"first_close_sampling_weight": 1.0},
        }
    )
    assert _training_sampler(_Dataset([1.0, -1.0]), config) is None

from __future__ import annotations

import numpy as np
import pytest
from phi_mujoco.evaluation import PolicyInput
from test_mujoco_online_metadata import make_metadata

from action_bridge.eval.mujoco_online.adapter import ActionBridgeMujocoPolicyAdapter


class RecordingBackend:
    def __init__(self):
        self.batches = []

    def reset(self, *, seed):
        self.seed = seed

    def predict(self, batch):
        self.batches.append({key: value.copy() for key, value in batch.items()})
        # First command must be clipped; second remains inside the bounds.
        return np.stack([np.full(7, 2.0), np.full(7, -0.5), np.zeros(7)])[None], {}


def policy_input(metadata, step):
    return PolicyInput(
        observations={
            "state": np.full(metadata.observation_dim, step, dtype=np.float32)
        },
        episode_index=0,
        step_index=step,
        seed=123,
        integration=metadata.integration,
    )


def test_chunk_buffer_preserves_every_observation_and_projected_action():
    metadata = make_metadata(actions_per_plan=2)
    backend = RecordingBackend()
    adapter = ActionBridgeMujocoPolicyAdapter(
        metadata=metadata, backend=backend, checkpoint_identifier="test"
    )
    adapter.reset(integration=metadata.integration, seed=123)
    first = adapter.predict(policy_input(metadata, 0))
    second = adapter.predict(policy_input(metadata, 1))
    third = adapter.predict(policy_input(metadata, 2))
    assert len(backend.batches) == 2
    assert first.shape == second.shape == third.shape == (7,)
    np.testing.assert_array_equal(first, np.ones(7))
    np.testing.assert_array_equal(second, np.full(7, -0.5))
    np.testing.assert_array_equal(backend.batches[0]["act_hist"], np.zeros((1, 2, 7)))
    np.testing.assert_array_equal(
        backend.batches[1]["obs_hist"], np.stack([np.ones(23), np.full(23, 2)])[None]
    )
    np.testing.assert_array_equal(
        backend.batches[1]["act_hist"], np.stack([first, second])[None]
    )
    adapter.reset(integration=metadata.integration, seed=123)
    adapter.predict(policy_input(metadata, 0))
    np.testing.assert_array_equal(backend.batches[-1]["act_hist"], np.zeros((1, 2, 7)))


def test_adapter_requires_every_intermediate_observation():
    metadata = make_metadata()
    adapter = ActionBridgeMujocoPolicyAdapter(
        metadata=metadata, backend=RecordingBackend(), checkpoint_identifier="test"
    )
    adapter.reset(integration=metadata.integration, seed=123)
    adapter.predict(policy_input(metadata, 0))
    with pytest.raises(ValueError, match="consecutive"):
        adapter.predict(policy_input(metadata, 2))


def test_execution_override_cannot_exceed_trained_horizon():
    with pytest.raises(ValueError, match="trained action horizon"):
        ActionBridgeMujocoPolicyAdapter(
            metadata=make_metadata(),
            backend=RecordingBackend(),
            checkpoint_identifier="test",
            actions_per_plan=4,
        )

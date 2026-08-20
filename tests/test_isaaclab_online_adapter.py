from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from phi_isaaclab.windows import reset_hold_open_action

from action_bridge.eval.isaaclab_online.adapter import (
    ActionBridgeIsaacLabPolicyAdapter,
)
from action_bridge.eval.isaaclab_online.metadata import OnlineEvaluationMetadata


def _metadata() -> OnlineEvaluationMetadata:
    return OnlineEvaluationMetadata.from_mapping(
        {
            "schema_name": "action_bridge.isaaclab_online",
            "schema_version": 1,
            "task_name": "franka_cube_lift",
            "variation_id": 0,
            "observation_profile": "phi.isaaclab.franka_cube_lift.state.v1",
            "action_profile": "phi.isaaclab.franka_cube_lift.ee_pose_abs_gripper.v1",
            "observation_dim": 35,
            "action_dim": 8,
            "observation_history": 2,
            "action_history": 2,
            "action_horizon": 4,
            "actions_per_plan": 1,
            "control_timestep_s": 0.02,
            "normalization": {
                "type": "standard",
                "eps": 1e-6,
                "obs_mean": [0.0] * 35,
                "obs_std": [1.0] * 35,
                "action_mean": [0.0] * 8,
                "action_std": [1.0] * 8,
            },
            "collection_identity": {
                "schema_name": "phi.isaaclab.episode_hdf5",
                "schema_version": 1,
                "manifest_sha256": "a" * 64,
            },
            "action_projection": {
                "position_lower_m": [0.2, -0.5, 0.02],
                "position_upper_m": [0.8, 0.5, 0.8],
                "position_projection": "clamp",
                "quaternion_order": "xyzw",
                "quaternion_projection": "normalize_nonnegative_w",
                "quaternion_epsilon": 1e-8,
                "gripper_threshold": 0.0,
                "gripper_open_action": 1.0,
                "gripper_close_action": -1.0,
            },
            "policy_type": "direct_bc",
            "latent_commitment": "chunk",
            "deterministic_latent": True,
        }
    )


@dataclass
class _Input:
    observation: torch.Tensor
    episode_ids: torch.Tensor
    step_indices: torch.Tensor
    task_id: str = "franka_cube_lift"
    variation_id: int = 0


class _Backend:
    def __init__(self, normalized_actions: torch.Tensor) -> None:
        self.normalized_actions = normalized_actions
        self.resets: list[torch.Tensor | None] = []
        self.batches: list[dict[str, torch.Tensor]] = []

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.resets.append(None if env_ids is None else env_ids.clone())

    def predict(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Mapping[str, object]]:
        self.batches.append({key: value.clone() for key, value in batch.items()})
        return self.normalized_actions.clone(), {}


def _reset_observations(batch_size: int = 2) -> torch.Tensor:
    value = torch.zeros((batch_size, 35), dtype=torch.float32)
    value[:, 18:25] = torch.tensor([0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
    return value


def _adapter(backend: _Backend) -> ActionBridgeIsaacLabPolicyAdapter:
    return ActionBridgeIsaacLabPolicyAdapter(
        metadata=_metadata(),
        backend=backend,
        checkpoint_identifier="sha256:" + "a" * 64,
    )


def test_reset_history_matches_canonical_numpy_helper_and_stays_batched() -> None:
    source = torch.tensor(
        [1.0, -1.0, 2.0, 0.0, 0.0, 0.0, -2.0, 0.3], dtype=torch.float32
    )
    backend = _Backend(source.repeat(2, 4, 1))
    adapter = _adapter(backend)
    observations = _reset_observations()

    with pytest.raises(RuntimeError, match="reset"):
        adapter.predict(
            _Input(
                observations,
                torch.zeros(2, dtype=torch.int64),
                torch.zeros(2, dtype=torch.int64),
            )
        )
    adapter.reset()
    output = adapter.predict(
        _Input(observations, torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64))
    )

    assert output.shape == (2, 8)
    assert output.device == observations.device
    expected = torch.tensor([0.8, -0.5, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0])
    torch.testing.assert_close(output, expected.repeat(2, 1))
    expected_padding = reset_hold_open_action(observations[0].numpy())
    np.testing.assert_array_equal(
        backend.batches[0]["act_hist"][0].numpy(),
        np.repeat(expected_padding[None], 2, axis=0),
    )
    torch.testing.assert_close(
        backend.batches[0]["obs_hist"], observations.unsqueeze(1).repeat(1, 2, 1)
    )


def test_subset_reset_tracks_asynchronous_vector_episodes_and_commits_actions() -> None:
    source = torch.tensor(
        [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, -1.0], dtype=torch.float32
    )
    backend = _Backend(source.repeat(2, 4, 1))
    adapter = _adapter(backend)
    observations = _reset_observations()
    adapter.reset()
    first = adapter.predict(
        _Input(observations, torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64))
    )

    adapter.reset(torch.tensor([1], dtype=torch.int64))
    next_observations = observations.clone()
    next_observations[0, 0] = 1.0
    next_observations[1, 18] = 0.45
    second = adapter.predict(
        _Input(
            next_observations,
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([1, 0], dtype=torch.int64),
        )
    )

    torch.testing.assert_close(first, second)
    assert backend.resets[0] is None
    torch.testing.assert_close(backend.resets[1], torch.tensor([1], dtype=torch.int64))
    # The continuing environment commits the previously executed action.
    torch.testing.assert_close(backend.batches[1]["act_hist"][0, -1], first[0])
    # The reset environment receives a fresh hold/open history from its new O0.
    expected_reset = torch.tensor([0.45, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0])
    torch.testing.assert_close(backend.batches[1]["act_hist"][1, -1], expected_reset)


def test_adapter_rejects_episode_change_without_explicit_reset() -> None:
    source = torch.tensor(
        [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float32
    )
    backend = _Backend(source.repeat(2, 4, 1))
    adapter = _adapter(backend)
    observations = _reset_observations()
    adapter.reset()
    adapter.predict(
        _Input(observations, torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64))
    )

    with pytest.raises(ValueError, match="changed without reset"):
        adapter.predict(
            _Input(
                observations,
                torch.tensor([0, 1], dtype=torch.int64),
                torch.tensor([1, 0], dtype=torch.int64),
            )
        )


def test_adapter_rejects_near_zero_quaternion() -> None:
    source = torch.zeros((2, 4, 8), dtype=torch.float32)
    source[..., :3] = torch.tensor([0.4, 0.0, 0.5])
    backend = _Backend(source)
    adapter = _adapter(backend)
    adapter.reset()
    with pytest.raises(ValueError, match="near-zero quaternion"):
        adapter.predict(
            _Input(
                _reset_observations(),
                torch.zeros(2, dtype=torch.int64),
                torch.zeros(2, dtype=torch.int64),
            )
        )

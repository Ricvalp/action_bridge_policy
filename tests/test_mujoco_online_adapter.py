from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from phi_mujoco.evaluation import PolicyInput

from action_bridge.eval.mujoco_online.adapter import ActionBridgeMujocoPolicyAdapter
from action_bridge.eval.mujoco_online.metadata import OnlineEvaluationMetadata


def _metadata(*, clip_actions: bool = False) -> OnlineEvaluationMetadata:
    return OnlineEvaluationMetadata.from_mapping(
        {
            "schema_name": "action_bridge.mujoco_online",
            "schema_version": 1,
            "task_name": "planar_reach",
            "variation_id": 0,
            "observation_profile": "phi.mujoco.planar_reach.state.v1",
            "action_profile": "phi.mujoco.planar_reach.joint_torque.v1",
            "observation_dim": 8,
            "action_dim": 2,
            "observation_history": 2,
            "action_history": 2,
            "action_horizon": 3,
            "actions_per_plan": 1,
            "action_lower": [-2.0, -2.0],
            "action_upper": [2.0, 2.0],
            "control_timestep_s": 0.02,
            "normalization": {
                "type": "standard",
                "eps": 1e-6,
                "obs_mean": [float(index) for index in range(8)],
                "obs_std": [float(index + 2) for index in range(8)],
                "action_mean": [0.5, -0.5],
                "action_std": [2.0, 4.0],
            },
            "collection_identity": {
                "schema_name": "phi.mujoco.episode_npz",
                "schema_version": 1,
                "manifest_sha256": "a" * 64,
            },
            "policy_type": "direct_bc",
            "latent_commitment": "episode",
            "deterministic_latent": True,
            "clip_actions": clip_actions,
        }
    )


def _input(values: np.ndarray, *, step: int) -> PolicyInput:
    return PolicyInput(
        observation=values,
        task_name="planar_reach",
        variation_id=0,
        episode_step=step,
    )


class _Backend:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions)
        self.seeds: list[int] = []
        self.batches: list[dict[str, np.ndarray]] = []

    def reset(self, *, seed: int) -> None:
        self.seeds.append(seed)

    def predict(self, batch: Mapping[str, np.ndarray]) -> tuple[np.ndarray, Mapping[str, object]]:
        self.batches.append({key: value.copy() for key, value in batch.items()})
        return self.actions.copy(), {"backend_value": 7}


def _adapter(backend: _Backend, *, clip_actions: bool = False):
    return ActionBridgeMujocoPolicyAdapter(
        metadata=_metadata(clip_actions=clip_actions),
        backend=backend,
        checkpoint_identifier="sha256:" + "a" * 64,
    )


def test_adapter_matches_offline_initial_padding_and_commits_one_action() -> None:
    normalized_actions = np.asarray(
        [[[0.25, 0.125], [0.0, 0.0], [-0.25, -0.125]]],
        dtype=np.float32,
    )
    backend = _Backend(normalized_actions)
    adapter = _adapter(backend)
    observation_0 = np.arange(8, dtype=np.float32)
    observation_1 = observation_0 + 1.0

    with pytest.raises(RuntimeError, match="reset"):
        adapter.predict(_input(observation_0, step=0))
    adapter.reset(task_name="planar_reach", variation_id=0, seed=17)
    output_0 = adapter.predict(_input(observation_0, step=0))
    output_1 = adapter.predict(_input(observation_1, step=1))

    assert backend.seeds == [17]
    np.testing.assert_allclose(
        output_0.actions,
        [[1.0, 0.0], [0.5, -0.5], [0.0, -1.0]],
    )
    np.testing.assert_allclose(output_1.actions, output_0.actions)
    expected_observation_0 = np.zeros(8, dtype=np.float32)
    np.testing.assert_allclose(
        backend.batches[0]["obs_hist"][0],
        np.stack([expected_observation_0, expected_observation_0]),
    )
    np.testing.assert_allclose(
        backend.batches[1]["obs_hist"][0],
        np.stack(
            [
                expected_observation_0,
                (observation_1 - np.arange(8, dtype=np.float32))
                / np.arange(2, 10, dtype=np.float32),
            ]
        ),
    )
    np.testing.assert_allclose(
        backend.batches[0]["act_hist"][0],
        [[-0.25, 0.125], [-0.25, 0.125]],
    )
    np.testing.assert_allclose(
        backend.batches[1]["act_hist"][0],
        [[-0.25, 0.125], [0.25, 0.125]],
    )
    assert output_0.diagnostics["policy_call"] == 1
    assert output_0.diagnostics["action_clipped"] is False


def test_adapter_requires_exact_consecutive_observations() -> None:
    backend = _Backend(np.zeros((1, 3, 2), dtype=np.float32))
    adapter = _adapter(backend)
    adapter.reset(task_name="planar_reach", variation_id=0, seed=1)
    adapter.predict(_input(np.zeros(8, dtype=np.float32), step=0))

    with pytest.raises(ValueError, match="consecutive episode_step=1"):
        adapter.predict(_input(np.ones(8, dtype=np.float32), step=2))


def test_adapter_rejects_out_of_bounds_by_default_and_reports_explicit_clipping() -> None:
    # With action_mean/std above, this denormalizes to [2.5, 3.5].
    backend = _Backend(np.ones((1, 3, 2), dtype=np.float32))
    rejecting = _adapter(backend)
    rejecting.reset(task_name="planar_reach", variation_id=0, seed=1)
    with pytest.raises(ValueError, match="violates the exact torque bounds"):
        rejecting.predict(_input(np.zeros(8, dtype=np.float32), step=0))

    clipping = _adapter(backend, clip_actions=True)
    clipping.reset(task_name="planar_reach", variation_id=0, seed=1)
    output = clipping.predict(_input(np.zeros(8, dtype=np.float32), step=0))
    np.testing.assert_array_equal(output.actions, np.full((3, 2), 2.0, np.float32))
    assert output.diagnostics["action_clipped"] is True
    assert output.diagnostics["clipped_value_count"] == 6
    assert output.diagnostics["max_clip_correction_nm"] == pytest.approx(1.5)
    assert output.diagnostics["clipped_action_calls_total"] == 1
    assert output.diagnostics["clipped_value_count_total"] == 6
    assert output.diagnostics["max_clip_correction_nm_episode"] == pytest.approx(1.5)


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((3, 2), dtype=np.float32),
        np.zeros((1, 2, 2), dtype=np.float32),
        np.full((1, 3, 2), np.nan, dtype=np.float32),
        np.full((1, 3, 2), np.finfo(np.float64).max, dtype=np.float64),
        np.zeros((1, 3, 2), dtype=np.bool_),
    ],
)
def test_adapter_rejects_invalid_backend_output(actions: np.ndarray) -> None:
    adapter = _adapter(_Backend(actions))
    adapter.reset(task_name="planar_reach", variation_id=0, seed=1)
    with pytest.raises(ValueError):
        adapter.predict(_input(np.zeros(8, dtype=np.float32), step=0))

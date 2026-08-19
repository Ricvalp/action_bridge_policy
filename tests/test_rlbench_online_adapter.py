from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping

import numpy as np
import pytest
from phi_rlbench.data.schema import ACTION_COMPONENTS, STATE_COMPONENTS
from phi_rlbench.evaluation import PolicyInput

from action_bridge.jax.eval.rlbench_online import (
    CANONICAL_RLBENCH_ACTION_LAYOUT,
    CANONICAL_RLBENCH_STATE_LAYOUT,
    ActionBridgeRLBenchPolicyAdapter,
    OnlineEvaluationMetadata,
    OnlineMetadataError,
    build_online_batch,
    resolve_online_metadata,
)
from action_bridge.jax.eval.rlbench_online.jax_backend import (
    _load_checkpoint_snapshot,
    validate_checkpoint_config,
)

_UNAVAILABLE_CACHE_IDENTITY = {
    "available": False,
    "manifest_sha256": None,
    "schema_name": None,
    "schema_version": None,
}


def _metadata(**overrides: object) -> OnlineEvaluationMetadata:
    values: dict[str, object] = {
        "task_to_id": {"reach_target": 0},
        "task_variation_to_id": {"reach_target:0": 0},
        "point_count": 4,
        "observation_history": 2,
        "observation_stride": 1,
        "action_history": 2,
        "action_stride": 1,
        "action_offset": 1,
        "action_horizon": 3,
        "action_representation": "absolute",
        "state_layout": list(CANONICAL_RLBENCH_STATE_LAYOUT),
        "action_layout": list(CANONICAL_RLBENCH_ACTION_LAYOUT),
        "training_cache_identity": _UNAVAILABLE_CACHE_IDENTITY,
        "include_rgb": True,
        "include_mask_id": False,
        "policy_type": "direct_chunk_bc",
    }
    values.update(overrides)
    return OnlineEvaluationMetadata.from_mapping(values)


def _input() -> PolicyInput:
    state = np.asarray(
        [
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.1, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    return PolicyInput(
        point_cloud_history=np.zeros((2, 4, 3), dtype=np.float32),
        point_valid_history=np.ones((2, 4), dtype=np.bool_),
        state_history=state,
        observation_history_mask=np.asarray([False, True], dtype=np.bool_),
        task_name="reach_target",
        task_id=0,
        variation_id=0,
        episode_step=0,
        rgb_history=np.full((2, 4, 3), 0.5, dtype=np.float32),
    )


class FakeBackend:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.batches: list[Mapping[str, np.ndarray]] = []

    def reset(self, *, seed: int) -> None:
        self.seeds.append(seed)

    def predict(
        self,
        batch: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, Mapping[str, object]]:
        self.batches.append(batch)
        state = batch["obs_hist"][:, -1]
        actions = np.repeat(state[:, None], 3, axis=1).astype(np.float32)
        return actions, {"backend_scalar": 2.0, "policy_call": 999}


def test_canonical_layout_is_owned_by_phi_rlbench_schema() -> None:
    assert CANONICAL_RLBENCH_STATE_LAYOUT == STATE_COMPONENTS
    assert CANONICAL_RLBENCH_ACTION_LAYOUT == ACTION_COMPONENTS
    assert CANONICAL_RLBENCH_STATE_LAYOUT == (
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "gripper_open",
    )


def test_batch_matches_current_action_bridge_inference_geometry() -> None:
    batch = build_online_batch(_input(), _metadata())

    assert "future_actions" not in batch
    assert batch["obs_hist"].shape == (1, 2, 8)
    assert batch["point_cloud_hist"].shape == (1, 2, 4, 3)
    assert batch["point_valid_hist"].dtype == np.bool_
    np.testing.assert_array_equal(batch["act_hist"], batch["obs_hist"])
    np.testing.assert_array_equal(
        batch["action_history_mask"],
        batch["obs_history_mask"],
    )
    np.testing.assert_array_equal(batch["task_id"], np.asarray([0], np.int32))
    np.testing.assert_array_equal(
        batch["task_variation_id"],
        np.asarray([0], np.int32),
    )
    assert batch["rgb_hist"].shape == (1, 2, 4, 3)


def test_adapter_resets_backend_and_returns_one_canonical_chunk() -> None:
    backend = FakeBackend()
    adapter = ActionBridgeRLBenchPolicyAdapter(
        metadata=_metadata(),
        backend=backend,
        checkpoint_identifier="sha256:" + "a" * 64,
    )

    with pytest.raises(RuntimeError, match="reset"):
        adapter.predict(_input())
    adapter.reset(task_name="reach_target", variation_id=0, seed=7)
    output = adapter.predict(_input())

    assert backend.seeds == [7]
    assert output.actions.shape == (3, 8)
    assert output.actions.dtype == np.float32
    assert output.diagnostics["policy_call"] == 1
    assert output.diagnostics["backend_scalar"] == 2.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"task_to_id": {"reach_target": 1}},
        {"task_to_id": {" reach_target": 0}},
        {"task_to_id": {"reach_target": 0, "wipe_desk": 1}},
        {"task_variation_to_id": {"unknown:0": 0}},
        {"observation_history": 3},
        {"observation_stride": 2, "action_stride": 2, "action_offset": 2},
        {"action_stride": 2},
        {"action_offset": 2},
        {"action_representation": "delta_xyz"},
        {"state_layout": [*CANONICAL_RLBENCH_STATE_LAYOUT[:-1], "closed"]},
        {"action_layout": list(reversed(CANONICAL_RLBENCH_ACTION_LAYOUT))},
        {
            "training_cache_identity": {
                "available": True,
                "manifest_sha256": "A" * 64,
                "schema_name": "action_bridge.rlbench_dense",
                "schema_version": 1,
            }
        },
        {"policy_type": "unknown"},
        {"deterministic_latent": False},
    ],
)
def test_metadata_rejects_ambiguous_online_semantics(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(OnlineMetadataError):
        _metadata(**overrides)


def test_metadata_accepts_optional_cache_bundle_digests() -> None:
    metadata = _metadata(
        training_cache_identity={
            "available": True,
            "manifest_sha256": "a" * 64,
            "schema_name": "action_bridge.rlbench_dense",
            "schema_version": 1,
            "cache_bundle_sha256": "b" * 64,
            "preprocessing_sidecar_sha256": "c" * 64,
        }
    )

    identity = metadata.to_json_dict()["training_cache_identity"]
    assert identity["cache_bundle_sha256"] == "b" * 64
    assert identity["preprocessing_sidecar_sha256"] == "c" * 64


def test_action_bridge_metadata_can_request_stochastic_latent() -> None:
    metadata = _metadata(
        policy_type="action_bridge",
        deterministic_latent=False,
    )
    assert metadata.deterministic_latent is False


def test_metadata_resolution_never_guesses_old_checkpoint_task_order(tmp_path) -> None:
    old_checkpoint: dict[str, object] = {"config": {"data": {"tasks": []}}}
    with pytest.raises(OnlineMetadataError, match="must not be inferred"):
        resolve_online_metadata(old_checkpoint)

    path = tmp_path / "online.json"
    path.write_text(json.dumps(_metadata().to_json_dict()), encoding="utf-8")
    resolved = resolve_online_metadata(old_checkpoint, explicit_path=path)
    assert resolved.task_to_id == {"reach_target": 0}

    embedded = {
        "online_evaluation": _metadata(policy_type="action_bridge").to_json_dict()
    }
    resolved = resolve_online_metadata(embedded, explicit_path=path)
    assert resolved.policy_type == "action_bridge"


def _checkpoint_config() -> dict[str, object]:
    return {
        "policy_type": "direct_chunk_bc",
        "data": {
            "point_count": 4,
            "obs_history": 2,
            "obs_stride": 1,
            "action_history": 2,
            "action_stride": 1,
            "action_offset": 1,
            "chunk_horizon": 3,
            "action_representation": "absolute",
            "include_rgb": True,
            "include_mask_id": False,
        },
        "encoder": {
            "max_obs_history": 16,
            "max_action_history": 32,
            "use_rgb": True,
            "use_mask_id": False,
        },
    }


def test_checkpoint_config_must_match_external_metadata() -> None:
    config = _checkpoint_config()
    validate_checkpoint_config(config, _metadata())
    config["data"]["chunk_horizon"] = 4  # type: ignore[index]
    with pytest.raises(OnlineMetadataError, match="chunk_horizon"):
        validate_checkpoint_config(config, _metadata())


def test_checkpoint_config_validates_effective_modalities_and_capacity() -> None:
    config = _checkpoint_config()
    config["data"]["include_rgb"] = False  # type: ignore[index]
    validate_checkpoint_config(config, _metadata(include_rgb=False))

    with pytest.raises(OnlineMetadataError, match="effective checkpoint input"):
        validate_checkpoint_config(config, _metadata(include_rgb=True))

    config["encoder"]["max_obs_history"] = 1  # type: ignore[index]
    with pytest.raises(OnlineMetadataError, match="max_obs_history"):
        validate_checkpoint_config(config, _metadata(include_rgb=False))


def test_checkpoint_snapshot_digest_identifies_unpickled_bytes(tmp_path) -> None:
    payload = pickle.dumps({"params": {"weight": 1}, "config": {"seed": 7}})
    path = tmp_path / "checkpoint.pkl"
    path.write_bytes(payload)

    checkpoint, identifier = _load_checkpoint_snapshot(path)

    assert checkpoint["config"] == {"seed": 7}
    assert identifier == f"sha256:{hashlib.sha256(payload).hexdigest()}"


def test_batch_rejects_missing_required_sensor_or_vocabulary() -> None:
    observation = _input()
    without_rgb = PolicyInput(
        point_cloud_history=observation.point_cloud_history,
        point_valid_history=observation.point_valid_history,
        state_history=observation.state_history,
        observation_history_mask=observation.observation_history_mask,
        task_name=observation.task_name,
        task_id=observation.task_id,
        variation_id=observation.variation_id,
        episode_step=observation.episode_step,
    )
    with pytest.raises(ValueError, match="requires rgb_history"):
        build_online_batch(without_rgb, _metadata())
    with pytest.raises(KeyError, match="variation"):
        build_online_batch(
            observation,
            _metadata(task_variation_to_id={"reach_target:1": 0}),
        )

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from action_bridge.config import load_config
from action_bridge.jax.eval.rlbench_online.metadata import OnlineMetadataError
from action_bridge.jax.training import rlbench_online_metadata as metadata_module
from action_bridge.jax.training.rlbench_online_metadata import (
    RLBenchOnlineMetadataWarning,
    configure_rlbench_online_metadata,
)


@dataclass
class _Dataset:
    task_to_id: dict[str, int]
    task_variation_to_id: dict[str, int]
    state_dim: int = 8
    action_dim: int = 8
    point_count: int = 64
    obs_history: int = 2
    obs_stride: int = 1
    action_history: int = 2
    action_stride: int = 1
    action_offset: int = 1
    chunk_horizon: int = 16
    action_representation: str = "absolute"
    include_rgb: bool = True
    include_mask_id: bool = True


def _dataset() -> _Dataset:
    return _Dataset(
        task_to_id={"open_drawer": 0, "reach_target": 1},
        task_variation_to_id={
            "open_drawer:0": 0,
            "reach_target:0": 1,
            "reach_target:1": 2,
        },
    )


def _config():
    config = load_config("rlbench_jax_contact_bridge")
    config.data.cache_root = "/read-only/cache"
    config.data.point_count = 64
    config.data.include_rgb = True
    config.data.include_mask_id = True
    config.encoder.use_rgb = False
    config.encoder.use_mask_id = True
    return config


def _cache_identity() -> dict[str, object]:
    return {
        "available": True,
        "sha256": "a" * 64,
        "bundle_sha256": "b" * 64,
        "schema_name": "action_bridge.rlbench_dense",
        "schema_version": 1,
        "valid_json": True,
        "preprocessing_sidecar": {
            "available": True,
            "sha256": "c" * 64,
            "valid_json": True,
        },
    }


@pytest.fixture
def cache_identity(monkeypatch):
    seen = []

    def identity(path):
        seen.append(path)
        return _cache_identity()

    monkeypatch.setattr(metadata_module, "cache_manifest_identity", identity)
    return seen


def test_configure_metadata_uses_dataset_semantics_and_effective_modalities(
    cache_identity,
):
    config = _config()
    train = _dataset()
    validation = _dataset()

    metadata = configure_rlbench_online_metadata(config, train, validation)

    assert metadata is not None
    assert dict(metadata.task_to_id) == train.task_to_id
    assert dict(metadata.task_variation_to_id) == train.task_variation_to_id
    assert metadata.point_count == train.point_count
    assert metadata.observation_history == train.obs_history
    assert metadata.action_horizon == train.chunk_horizon
    assert metadata.include_rgb is False
    assert metadata.include_mask_id is True
    assert metadata.training_cache_identity == {
        "available": True,
        "manifest_sha256": "a" * 64,
        "schema_name": "action_bridge.rlbench_dense",
        "schema_version": 1,
        "cache_bundle_sha256": "b" * 64,
        "preprocessing_sidecar_sha256": "c" * 64,
    }
    assert config.online_evaluation.to_dict() == metadata.to_json_dict()
    assert cache_identity == ["/read-only/cache"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"task_to_id": {"open_drawer": 0}}, "task_to_id mappings differ"),
        (
            {"task_variation_to_id": {"open_drawer:0": 0, "reach_target:0": 1}},
            "task_variation_to_id mappings differ",
        ),
        ({"state_dim": 7}, "state_dim differ"),
        ({"action_dim": 7}, "action_dim differ"),
        ({"point_count": 32}, "point_count differ"),
    ],
)
def test_configure_metadata_rejects_train_validation_identity_mismatches(
    cache_identity,
    change,
    message,
):
    train = _dataset()
    validation = replace(_dataset(), **change)

    with pytest.raises(ValueError, match=message):
        configure_rlbench_online_metadata(_config(), train, validation)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"state_dim": 7}, "RLBench state_dim must be 8"),
        ({"action_dim": 7}, "RLBench action_dim must be 8"),
    ],
)
def test_configure_metadata_rejects_noncanonical_dimensions(
    cache_identity,
    change,
    message,
):
    train = replace(_dataset(), **change)
    validation = replace(_dataset(), **change)

    with pytest.raises(ValueError, match=message):
        configure_rlbench_online_metadata(_config(), train, validation)


def test_configure_metadata_rejects_configured_point_count_mismatch(cache_identity):
    config = _config()
    config.data.point_count = 32

    with pytest.raises(ValueError, match="dataset point_count disagrees"):
        configure_rlbench_online_metadata(config, _dataset(), _dataset())


def test_unsupported_offline_geometry_removes_stale_metadata_and_warns(cache_identity):
    config = _config()
    config.data.obs_stride = 2
    config.online_evaluation = {"stale": True}
    train = replace(_dataset(), obs_stride=2)
    validation = replace(_dataset(), obs_stride=2)

    with pytest.warns(RLBenchOnlineMetadataWarning, match="unsupported"):
        metadata = configure_rlbench_online_metadata(config, train, validation)

    assert metadata is None
    assert "online_evaluation" not in config
    assert cache_identity == []


def test_unsupported_geometry_rejects_resume_that_claimed_online_metadata(
    cache_identity,
):
    original = configure_rlbench_online_metadata(_config(), _dataset(), _dataset())
    assert original is not None
    config = _config()
    config.data.action_stride = 2
    train = replace(_dataset(), action_stride=2)
    validation = replace(_dataset(), action_stride=2)

    with pytest.raises(OnlineMetadataError, match="Cannot resume"):
        configure_rlbench_online_metadata(
            config,
            train,
            validation,
            resume_payload={"online_evaluation": original.to_json_dict()},
        )


def test_resume_may_change_only_deterministic_latent_default(cache_identity):
    original = configure_rlbench_online_metadata(_config(), _dataset(), _dataset())
    assert original is not None
    config = _config()
    config.checkpoint.online_deterministic_latent_default = False

    current = configure_rlbench_online_metadata(
        config,
        _dataset(),
        _dataset(),
        resume_payload={"online_evaluation": original.to_json_dict()},
    )

    assert current is not None
    assert current.deterministic_latent is False
    assert config.online_evaluation.deterministic_latent is False


def test_resume_rejects_every_other_semantic_mismatch(cache_identity):
    original = configure_rlbench_online_metadata(_config(), _dataset(), _dataset())
    assert original is not None
    embedded = original.to_json_dict()
    embedded["point_count"] = 65

    with pytest.raises(OnlineMetadataError, match="point_count"):
        configure_rlbench_online_metadata(
            _config(),
            _dataset(),
            _dataset(),
            resume_payload={"online_evaluation": embedded},
        )


def test_legacy_resume_without_metadata_warns_and_attaches_current_semantics(
    cache_identity,
):
    config = _config()

    with pytest.warns(RLBenchOnlineMetadataWarning, match="legacy checkpoint"):
        metadata = configure_rlbench_online_metadata(
            config,
            _dataset(),
            _dataset(),
            resume_payload={"config": {"seed": 0}},
        )

    assert metadata is not None
    assert config.online_evaluation.to_dict() == metadata.to_json_dict()


def test_unavailable_manifest_uses_all_null_cache_identity(monkeypatch):
    monkeypatch.setattr(
        metadata_module,
        "cache_manifest_identity",
        lambda path: {"available": False, "sha256": None},
    )

    metadata = configure_rlbench_online_metadata(_config(), _dataset(), _dataset())

    assert metadata is not None
    assert metadata.training_cache_identity == {
        "available": False,
        "manifest_sha256": None,
        "schema_name": None,
        "schema_version": None,
        "cache_bundle_sha256": None,
        "preprocessing_sidecar_sha256": None,
    }

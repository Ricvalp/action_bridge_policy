"""Driver provenance checks using tiny dummy caches, never a synthetic experiment."""

import copy
import json

import pytest
import torch

from action_bridge.configs.sb_pusht import get_config
from action_bridge.plan_revision import checkpoints
from action_bridge.scripts.sb_pusht import (
    dataset_digest, run_stage, source_spec, validate_dependencies,
    validate_evaluation_sources, window_spec,
)


@pytest.fixture(autouse=True)
def no_cuda_initialization(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def cached_driver(tmp_path):
    config = get_config()
    dataset = tmp_path / "dummy-immutable-data"
    dataset.write_bytes(b"dummy cache identity; not a replay dataset")
    root = tmp_path / "run"
    metadata = {"dataset_path": str(dataset.resolve()), "dataset_sha256": dataset_digest(dataset),
                "normalizer_id": "unit-test-normalizer", "splits": {"train": [0], "val": [1], "test": [2]}}
    checkpoints.save(root / "windows.pt", {"metadata": metadata, "data_spec": window_spec(config), "records": {}})
    reference = {"complete": True, "metadata": metadata, "config": config,
                 "completion_config": {"robot_dt": 1.}, "completion_state": {},
                 "innovation_variance": torch.ones(2)}
    proposal = {"complete": True, "metadata": metadata, "config": config, "ema": {}}
    checkpoints.save(root / "reference" / "latest.pt", reference)
    checkpoints.save(root / "ddim" / "latest.pt", proposal)
    source_metadata = metadata | {
        "reference_hash": checkpoints.digest(root / "reference" / "latest.pt"),
        "proposal_hash": checkpoints.digest(root / "ddim" / "latest.pt"),
    }
    sources = {"metadata": source_metadata, "source_spec": source_spec(config), "config": config,
               "records": {}, "dependencies": {}}
    checkpoints.save(root / "sources.pt", sources)
    return root, dataset, config, metadata, reference, proposal, sources


@pytest.mark.parametrize("key,new_value", [
    ("mobility_smoothing", 0.5), ("prior_ridge", 0.2), ("max_rate", 3.),
    ("proposal_seed", 18), ("proposals_per_history", 3), ("batch_size", 128),
    ("ot_block_size", 16),
])
def test_source_cache_semantic_changes_are_rejected(tmp_path, key, new_value):
    root, dataset, config, *_ = cached_driver(tmp_path)
    assert run_stage("sources", root, dataset, config, "cpu")["source_spec"] == source_spec(config)
    with pytest.raises(ValueError, match="Source cache configuration mismatch"):
        run_stage("sources", root, dataset, config | {key: new_value}, "cpu")
    with pytest.raises(ValueError, match="Source cache configuration mismatch"):
        run_stage("sb_ou", root, dataset, config | {"method": "sb_ou", key: new_value}, "cpu")


def test_learner_settings_do_not_change_frozen_source_identity():
    config = get_config()
    assert source_spec(config) == source_spec(config | {"method": "sb_kinetic", "lr": 2e-4, "updates": 2})


@pytest.mark.parametrize("name", ["reference", "proposal"])
def test_source_dependencies_must_share_dataset_and_normalizer(tmp_path, name):
    root, dataset, config, _, reference, proposal, _ = cached_driver(tmp_path)
    state = copy.deepcopy(reference if name == "reference" else proposal)
    state["metadata"]["normalizer_id"] = "wrong-coordinate-system"
    directory = "reference" if name == "reference" else "ddim"
    checkpoints.save(root / directory / "latest.pt", state)
    with pytest.raises(ValueError, match="another dataset, split or normalizer"):
        run_stage("sources", root, dataset, config, "cpu")


def test_dependency_shapes_and_robot_index_time_are_checked(tmp_path):
    _, _, config, metadata, reference, proposal, _ = cached_driver(tmp_path)
    with pytest.raises(ValueError, match="robot-index dt"):
        validate_dependencies(reference, proposal, metadata, config | {"robot_dt": .1})
    proposal = copy.deepcopy(proposal)
    proposal["config"]["horizon"] = 8
    with pytest.raises(ValueError, match="incompatible horizon"):
        validate_dependencies(reference, proposal, metadata, config)


def test_prepare_preserves_completed_manifest_jobs_when_all_is_resumed(tmp_path):
    root, dataset, config, *_ = cached_driver(tmp_path)
    run_stage("prepare", root, dataset, config, "cpu")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["jobs"]["ddim"] = "complete"
    manifest["stages"] = {"reference": "complete", "ddim": "complete"}
    manifest_path.write_text(json.dumps(manifest))
    run_stage("prepare", root, dataset, config, "cpu")
    assert json.loads(manifest_path.read_text()) == manifest
    with pytest.raises(ValueError, match="manifest has different configuration"):
        run_stage("prepare", root, dataset, config | {"lr": 2e-4}, "cpu")


def test_window_noise_annotations_cannot_go_stale(tmp_path):
    root, dataset, config, *_ = cached_driver(tmp_path)
    with pytest.raises(ValueError, match="another dataset or shape configuration"):
        run_stage("prepare", root, dataset, config | {"source_std": .2}, "cpu")


def test_evaluation_requires_same_frozen_proposal_and_matching_source_cache(tmp_path):
    root, _, config, metadata, _, proposal, sources = cached_driver(tmp_path)
    proposal_hash = sources["metadata"]["proposal_hash"]
    source_hash = checkpoints.digest(root / "sources.pt")
    validate_evaluation_sources(proposal, proposal_hash, sources, source_hash)
    with pytest.raises(ValueError, match="same checkpoint as the frozen bootstrap"):
        validate_evaluation_sources(proposal, "another-ddim-checkpoint", sources, source_hash)
    state = {"config": config | {"method": "sb_kinetic"},
             "metadata": sources["metadata"] | {"source_hash": source_hash},
             "direction": "forward", "dependencies": {
                 "proposal_sha256": proposal_hash,
                 "reference_sha256": sources["metadata"]["reference_hash"]}}
    validate_evaluation_sources(state, "kinetic-checkpoint", sources, source_hash)
    with pytest.raises(ValueError, match="different dataset/dependency provenance"):
        validate_evaluation_sources(state, "kinetic-checkpoint", sources, "another-source-cache")
    with pytest.raises(ValueError, match="completed forward DSBM phase"):
        validate_evaluation_sources(state | {"direction": "reverse"}, "kinetic-checkpoint", sources, source_hash)
    changed = copy.deepcopy(state)
    changed["dependencies"]["reference_sha256"] = "another-reference"
    with pytest.raises(ValueError, match="different frozen reference"):
        validate_evaluation_sources(changed, "kinetic-checkpoint", sources, source_hash)


def test_primary_evaluation_does_not_select_an_unfinished_job(tmp_path):
    root, dataset, config, metadata, _, proposal, _ = cached_driver(tmp_path)
    checkpoints.save(root / "ddim" / "latest.pt", proposal | {"complete": False})
    with pytest.raises(ValueError, match="Finish ddim training"):
        run_stage("evaluate", root, dataset, config, "cpu")

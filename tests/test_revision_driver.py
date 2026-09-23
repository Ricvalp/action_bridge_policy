"""Active-protocol manifest and dependency checks; no scientific runs."""
import json

import pytest
import torch

from action_bridge.configs.sb_pusht import METHODS, get_config
from action_bridge.plan_revision import checkpoints
from action_bridge.scripts import sb_pusht as driver


@pytest.fixture(autouse=True)
def no_cuda_initialization(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def cached_driver(tmp_path):
    config = get_config()
    dataset = tmp_path / "dummy-immutable-data"
    dataset.write_bytes(b"dummy cache identity; not a replay dataset")
    root = tmp_path / "run"
    metadata = {"dataset_path": str(dataset.resolve()), "dataset_sha256": driver.dataset_digest(dataset),
                "normalizer_id": "unit-test", "splits": {"train": [0], "val": [1], "test": [2]}}
    checkpoints.save(root / "windows.pt", {"metadata": metadata,
                     "data_spec": driver.window_spec(config), "records": {}})
    reference = {"complete": True, "metadata": metadata, "config": config,
                 "completion_config": {"robot_dt": 1.}, "completion_state": {},
                 "innovation_variance": torch.ones(2)}
    checkpoints.save(root / "reference" / "latest.pt", reference)
    return root, dataset, config, metadata, reference


def test_manifest_has_only_replacement_jobs_and_no_ddim_dependencies(tmp_path, capsys):
    root = tmp_path / "not-created"
    assert driver.main(["all", "--run-root", str(root), "--dry-run"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["protocol"] == "self_source_v1"
    assert tuple(manifest["jobs"]) == METHODS
    assert manifest["dependencies"]["ddim"] == []
    assert all(manifest["dependencies"][key] == ["reference"] for key in METHODS[1:])
    assert len(manifest["completion_evaluations"]) == 2
    assert manifest["source_curriculum"] == [.1, .5, 1., 1.]
    assert not root.exists()


def test_shared_legacy_source_stage_is_retired_without_mutation(tmp_path):
    with pytest.raises(ValueError, match="retired"):
        driver.run_stage("sources", tmp_path / "absent", None, get_config(), "cpu")
    assert not (tmp_path / "absent").exists()


def test_reference_dataset_shape_and_time_checks(tmp_path):
    _, _, config, metadata, reference = cached_driver(tmp_path)
    driver.validate_reference(reference, metadata, config)
    with pytest.raises(ValueError, match="robot-index dt"):
        driver.validate_reference(reference, metadata, config | {"robot_dt": .1})
    with pytest.raises(ValueError, match="incompatible horizon"):
        driver.validate_reference(reference, metadata, config | {"horizon": 8})
    with pytest.raises(ValueError, match="another dataset"):
        driver.validate_reference(reference, metadata | {"normalizer_id": "wrong"}, config)


def test_prepare_preserves_finished_jobs_on_resume(tmp_path):
    root, dataset, config, *_ = cached_driver(tmp_path)
    driver.run_stage("prepare", root, dataset, config, "cpu")
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["jobs"]["ddim"] = "complete"
    path.write_text(json.dumps(manifest))
    driver.run_stage("prepare", root, dataset, config, "cpu")
    assert json.loads(path.read_text()) == manifest
    with pytest.raises(ValueError, match="manifest has different configuration"):
        driver.run_stage("prepare", root, dataset, config | {"lr": 2e-4}, "cpu")


def test_windows_reject_old_protocol_and_stale_noise_annotations(tmp_path):
    root, dataset, config, *_ = cached_driver(tmp_path)
    with pytest.raises(ValueError, match="another dataset or shape configuration"):
        driver.run_stage("prepare", root, dataset, config | {"source_std": .2}, "cpu")
    with pytest.raises(ValueError, match="Only self_source_v1"):
        driver.run_stage("prepare", root, dataset, config | {"protocol": "frozen_ddim"}, "cpu")


def test_evaluation_rejects_legacy_dependency_or_wrong_budget(tmp_path):
    _, _, config, metadata, _ = cached_driver(tmp_path)
    state = dict(config=config, metadata=metadata, dependencies={})
    driver.validate_evaluation(state, metadata, config)
    with pytest.raises(ValueError, match="Legacy results"):
        driver.validate_evaluation(state | {"config": config | {"protocol": "old"}}, metadata, config)
    with pytest.raises(ValueError, match="External proposal"):
        driver.validate_evaluation(state | {"dependencies": {"proposal_state": {}}}, metadata, config)
    with pytest.raises(ValueError, match="optimizer budget"):
        driver.validate_evaluation(state, metadata, config | {"updates": 1})


def test_primary_evaluation_rejects_unfinished_training(tmp_path):
    root, dataset, config, metadata, _ = cached_driver(tmp_path)
    checkpoints.save(root / "ddim" / "latest.pt", {"complete": False})
    with pytest.raises(ValueError, match="Finish ddim training"):
        driver.run_stage("evaluate", root, dataset, config, "cpu")


def test_reviser_stage_never_reads_ddim_or_shared_sources(tmp_path, monkeypatch):
    root, dataset, config, metadata, reference = cached_driver(tmp_path)
    data = checkpoints.load(root / "windows.pt")
    data["records"] = {"train": {}, "val": {}}
    checkpoints.save(root / "windows.pt", data)
    captured = {}
    def training(records, config, output, metadata, dependencies, device, **kwargs):
        captured.update(dependencies)
        return {"step": 0}
    monkeypatch.setattr(driver, "train", training)
    driver.run_stage("sb_ou", root, dataset, config | {"method": "sb_ou"}, "cpu", evaluation=False)
    assert "completion_state" in captured
    assert not any("proposal" in key for key in captured)
    assert not (root / "ddim").exists()
    assert not (root / "sources.pt").exists()

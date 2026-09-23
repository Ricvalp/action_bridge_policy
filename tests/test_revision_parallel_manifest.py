"""Independent Slurm method jobs may finish in the same shared run directory."""
import json
import multiprocessing
from pathlib import Path
import time
from unittest.mock import patch

import pytest

from action_bridge.configs.sb_pusht import METHODS, get_config
from action_bridge.scripts import sb_pusht as driver


def complete_in_process(root, stage, start):
    # Make racing read/modify/write operations overlap reliably if locking is
    # removed. Each process still uses its own real filesystem lock and writes.
    read_text = Path.read_text

    def slow_read(path, *args, **kwargs):
        value = read_text(path, *args, **kwargs)
        if path.name == "manifest.json":
            time.sleep(.01)
        return value

    with patch.object(Path, "read_text", slow_read):
        start.wait(timeout=10)
        for _ in range(3):
            driver.record_stage_completion(root, stage)


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="Unix Slurm locking")
def test_concurrent_jobs_preserve_all_updates_and_valid_json(tmp_path):
    manifest = driver.experiment_manifest(get_config())
    manifest["metadata"] = {"unchanged": "dataset identity" * 10000}
    manifest["stages"] = {"prepare": "complete", "reference": "complete"}
    path = tmp_path / "manifest.json"
    driver.write_json(path, manifest)
    context = multiprocessing.get_context("fork")
    start = context.Barrier(len(METHODS) + 1)
    jobs = [context.Process(target=complete_in_process, args=(tmp_path, method, start))
            for method in METHODS]
    for job in jobs:
        job.start()
    try:
        start.wait(timeout=10)
        while any(job.is_alive() for job in jobs):
            # Deliberately read without taking the lock, just like a monitor.
            current = json.loads(path.read_text())
            assert current["metadata"] == manifest["metadata"]
            for job in jobs:
                job.join(timeout=.001)
        assert all(job.exitcode == 0 for job in jobs)
    finally:
        for job in jobs:
            if job.is_alive():
                job.terminate()
            job.join(timeout=5)
    result = json.loads(path.read_text())
    assert result["jobs"] == {method: "complete" for method in METHODS}
    assert result["stages"] == {**manifest["stages"], **result["jobs"]}
    assert result["metadata"] == manifest["metadata"]
    assert not path.with_suffix(".json.tmp").exists()


def test_auxiliary_stage_does_not_add_a_method(tmp_path):
    manifest = driver.experiment_manifest(get_config())
    driver.write_json(tmp_path / "manifest.json", manifest)
    driver.record_stage_completion(tmp_path, "reference")
    result = json.loads((tmp_path / "manifest.json").read_text())
    assert result["jobs"] == manifest["jobs"]
    assert result["stages"] == {"reference": "complete"}


def test_main_records_its_finished_method(tmp_path, monkeypatch):
    driver.write_json(tmp_path / "manifest.json", driver.experiment_manifest(get_config()))
    monkeypatch.setattr(driver.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(driver, "run_stage", lambda *args, **kwargs: None)
    assert driver.main(["ddim", "--run-root", str(tmp_path), "--device", "cpu"]) == 0
    result = json.loads((tmp_path / "manifest.json").read_text())
    assert result["jobs"]["ddim"] == "complete"
    assert result["stages"] == {"ddim": "complete"}


def test_missing_manifest_is_not_created(tmp_path):
    driver.record_stage_completion(tmp_path / "absent", "ddim")
    assert not (tmp_path / "absent").exists()

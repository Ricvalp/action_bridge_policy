"""Audit and prepare remain usable without the optional ripgrep executable."""

import json
from pathlib import Path

import numpy as np
import pytest

from action_bridge.plan_revision import checkpoints
from action_bridge.scripts import sb_pusht as driver


def write_source(checkout, relative_path, contents):
    path = checkout / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


def test_discover_implementations_matches_python_sources_in_sorted_order(tmp_path):
    sources = {
        "z_policy.py": "scheduler = DDPMScheduler()\n",
        "nested/a_policy.py": "scheduler = DDIMScheduler()\n",
        ".hidden/flow.py": "from policy import flow_matching\n",
        "ema.py": "class PolicyEMA:\n    pass\n",
        "multiple.py": "DDIMScheduler\nDDPMScheduler\nflow_matching\n",
    }
    for path, contents in sources.items():
        write_source(tmp_path, path, contents)
    write_source(tmp_path, "ordinary.py", "class Policy:\n    pass\n")
    write_source(tmp_path, "line_break.py", "class Policy:\n    EMA = None\n")
    write_source(tmp_path, "README.md", "DDIMScheduler\n")
    write_source(tmp_path, "compiled.pyc", "flow_matching\n")
    write_source(tmp_path, ".gitignore", "nested/\n")

    assert driver.discover_implementations(tmp_path) == sorted(
        str(tmp_path / path) for path in sources
    )


@pytest.mark.parametrize("directory", [
    ".git", "site-packages", ".native", "node_modules",
    ".venv", ".venv-cpu", ".uv-cache", ".uv-cache-test",
])
def test_discover_implementations_excludes_runtime_directories(tmp_path, directory):
    expected = write_source(tmp_path, "policy.py", "DDIMScheduler\n")
    write_source(tmp_path, f"{directory}/ignored.py", "DDIMScheduler\n")
    write_source(tmp_path, f"nested/{directory}/ignored.py", "flow_matching\n")

    assert driver.discover_implementations(tmp_path) == [str(expected)]


def test_discover_implementations_does_not_follow_symlinks(tmp_path):
    checkout = tmp_path / "checkout"
    expected = write_source(checkout, "policy.py", "DDIMScheduler\n")
    external = write_source(tmp_path, "external/implementation.py", "flow_matching\n")
    (checkout / "linked.py").symlink_to(external)
    (checkout / "linked-directory").symlink_to(external.parent, target_is_directory=True)
    (checkout / "loop").symlink_to(checkout, target_is_directory=True)

    assert driver.discover_implementations(checkout) == [str(expected)]


@pytest.mark.parametrize("stage", ["audit", "prepare"])
def test_cli_audit_and_fresh_prepare_do_not_require_rg(tmp_path, monkeypatch, stage):
    checkout = tmp_path / "checkout"
    script = write_source(checkout, "action_bridge/scripts/sb_pusht.py", "# test checkout\n")
    implementation = write_source(checkout, "policy.py", "DDIMScheduler\n")
    write_source(checkout, ".venv/lib/ignored.py", "DDPMScheduler\n")
    lock = write_source(checkout, "uv.lock", "# test lock identity\n")
    monkeypatch.setattr(driver, "__file__", str(script))
    monkeypatch.setattr(driver.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(driver.torch, "set_num_threads", lambda count: None)

    real_run = driver.subprocess.run
    commands = []

    def without_rg(command, *args, **kwargs):
        commands.append(command)
        if Path(command[0]).name == "rg":
            raise FileNotFoundError("rg is deliberately unavailable in this regression test")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(driver.subprocess, "run", without_rg)
    real_check_output = driver.subprocess.check_output
    git_results = {
        ("rev-parse", "HEAD"): "test-policy-commit\n",
        ("status", "--short"): "",
        ("diff", "HEAD"): "",
    }

    def checkout_git(command, *args, **kwargs):
        if command[:3] == ["git", "-C", str(checkout)]:
            return git_results[tuple(command[3:])]
        return real_check_output(command, *args, **kwargs)

    monkeypatch.setattr(driver.subprocess, "check_output", checkout_git)

    # This is a tiny regression fixture, never an experiment or real Push-T data.
    dataset = tmp_path / "dummy-replay.npz"
    rng = np.random.default_rng(0)
    np.savez(dataset, observations=rng.normal(size=(10, 30, 5)).astype(np.float32),
             actions=rng.normal(size=(10, 30, 2)).astype(np.float32))
    dataset_sha256 = checkpoints.digest(dataset)
    run_root = tmp_path / "run"

    assert driver.main([stage, "--dataset", str(dataset), "--run-root", str(run_root),
                        "--device", "cpu"]) == 0

    report = json.loads((run_root / "audit.json").read_text())
    assert report["discovered_implementation_paths"] == [str(implementation)]
    assert report["commit"] == "test-policy-commit"
    assert report["lock_sha256"] == checkpoints.digest(lock)
    assert report["dataset_path"] == str(dataset)
    assert report["dataset_exists"] is True
    assert all(Path(command[0]).name != "rg" for command in commands)
    assert checkpoints.digest(dataset) == dataset_sha256

    if stage == "prepare":
        windows = checkpoints.load(run_root / "windows.pt")
        assert set(windows["records"]) == {"train", "val", "test"}
        assert all(len(records["future_actions"]) > 0 for records in windows["records"].values())
        assert windows["metadata"]["dataset_sha256"] == dataset_sha256
        assert windows["metadata"]["policy_commit"] == report["commit"]
        assert windows["metadata"]["lock_sha256"] == report["lock_sha256"]
        manifest = json.loads((run_root / "manifest.json").read_text())
        assert manifest["stages"]["prepare"] == "complete"
        assert set(manifest["jobs"].values()) == {"pending"}

from __future__ import annotations

import csv
import json
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from action_bridge.training import mujoco_sim_eval as sim_eval


def _config(**logging):
    return {
        "chunk_horizon": 8,
        "data": {"integration": "robomimic_square"},
        "eval": {"actions_per_plan": 4},
        "logging": {"sim_eval_episodes": 4, **logging},
    }


class FakeProcess:
    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.pid = 246810
        self.returncode = None
        self.waits = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waits.append(timeout)
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


class FakeWandbRun:
    def __init__(self):
        self.definitions = []
        self.logs = []

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, payload):
        # Deliberately no step argument: late evaluations need their own x-axis.
        self.logs.append(payload)


@pytest.fixture(autouse=True)
def protect_process_signals(monkeypatch):
    # All processes below have invented PIDs: never signal the real machine.
    monkeypatch.setattr(sim_eval.os, "killpg", lambda pid, sig: None)


@pytest.fixture
def launched(monkeypatch):
    processes = []
    snapshots = []

    def launch(command, **kwargs):
        checkpoint = command[command.index("--checkpoint") + 1]
        output_dir = command[command.index("--run-dir") + 1]
        from pathlib import Path

        assert Path(checkpoint).is_file()  # Fully saved before the child is launched.
        assert not Path(output_dir).exists()  # The backend claims this directory.
        process = FakeProcess(command, **kwargs)
        processes.append(process)
        return process

    def save(path, model, optimizer, config, step, best_mse):
        snapshots.append((model, optimizer, config, step, best_mse))
        path.write_bytes(f"immutable checkpoint {step}".encode())

    monkeypatch.setattr(sim_eval.subprocess, "Popen", launch)
    monkeypatch.setattr(sim_eval, "save_checkpoint", save)
    return processes, snapshots


def _summary(manager, successes=2, **overrides):
    job = manager.pending
    job.output_dir.mkdir()
    result = {
        "attempted_episodes": manager.episodes,
        "successful_episodes": successes,
        "success_rate": successes / manager.episodes,
        "actions_per_plan": manager.n_exec,
        "checkpoint_identifier": "sha256:immutable-test-checkpoint",
        "episodes": [],
        "selected_videos": [],
        **overrides,
    }
    (job.output_dir / "summary.json").write_text(json.dumps(result))
    return result


def test_submit_is_nonblocking_and_skips_busy_without_saving(tmp_path, launched, monkeypatch):
    processes, snapshots = launched
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-slurm-assigned")
    config = _config(sim_eval_n_exec=8, sim_eval_max_steps=400)
    manager = sim_eval.AsyncMujocoEvaluator(config, tmp_path, None)
    model, optimizer = object(), object()
    assert manager.submit(model, optimizer, config, 2000, 0.04)
    assert not manager.submit(model, optimizer, config, 4000, 0.03)
    manager.poll()
    assert len(processes) == len(snapshots) == 1
    assert snapshots[0] == (model, optimizer, config, 2000, 0.04)
    process = processes[0]
    assert not process.waits
    assert process.command[:3] == [sys.executable, "-m", "action_bridge.eval.mujoco_online.parallel"]
    assert process.command[process.command.index("--actions-per-plan") + 1] == "8"
    assert process.command[process.command.index("--max-steps") + 1] == "400"
    assert process.command[process.command.index("--seed") + 1] == "2000000"
    assert process.kwargs["start_new_session"] is True
    assert process.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-slurm-assigned"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert process.kwargs["env"][name] == "1"
    assert process.kwargs["stdout"].name == str(manager.pending.log_path)
    assert manager.pending.log_path.parent != manager.pending.output_dir


def test_finished_evaluation_logs_checkpoint_step_and_exact_best_snapshot(tmp_path, launched):
    config = _config()
    wandb = FakeWandbRun()
    manager = sim_eval.AsyncMujocoEvaluator(config, tmp_path, wandb)
    manager.submit(None, None, config, 2000, 0.04)
    snapshot = manager.pending.checkpoint
    _summary(manager)
    manager.pending.process.returncode = 0
    manager.poll()
    assert manager.pending is None
    assert not snapshot.exists()
    assert (tmp_path / "checkpoints" / "best_success.pt").read_bytes() == b"immutable checkpoint 2000"
    metadata = json.loads((tmp_path / "metrics" / "best_success.json").read_text())
    assert metadata["step"] == 2000
    assert metadata["protocol"]["seed"] == 2_000_000
    assert metadata["checkpoint_identifier"] == "sha256:immutable-test-checkpoint"
    assert manager.best_success == 0.5
    assert wandb.logs == [{"sim_eval/checkpoint_step": 2000, "sim_eval/success_rate": 0.5}]
    assert (("sim_eval/*",), {"step_metric": "sim_eval/checkpoint_step"}) in wandb.definitions
    with (tmp_path / "metrics" / "sim_eval_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{"step": "2000", "success_rate": "0.5", "successful_episodes": "2", "attempted_episodes": "4", "actions_per_plan": "4"}]


def test_success_ties_keep_earlier_checkpoint_and_resume_preserves_best(tmp_path, launched):
    config = _config()
    manager = sim_eval.AsyncMujocoEvaluator(config, tmp_path, None)
    for step, successes in ((2000, 2), (4000, 2), (6000, 1), (8000, 3)):
        manager.submit(None, None, config, step, 0.04)
        _summary(manager, successes)
        manager.pending.process.returncode = 0
        manager.poll()
        expected_step = 8000 if step == 8000 else 2000
        assert (tmp_path / "checkpoints" / "best_success.pt").read_bytes() == f"immutable checkpoint {expected_step}".encode()
    resumed = sim_eval.AsyncMujocoEvaluator(config, tmp_path, None)
    assert resumed.best_success == 0.75
    with pytest.raises(ValueError, match="protocol differs"):
        sim_eval.AsyncMujocoEvaluator(_config(sim_eval_n_exec=8), tmp_path, None)


@pytest.mark.parametrize("problem", ["exit", "missing", "partial", "invalid_json", "nan"])
def test_failed_evaluations_are_not_zero_success_and_do_not_stop_training(tmp_path, launched, problem):
    config = _config()
    wandb = FakeWandbRun()
    manager = sim_eval.AsyncMujocoEvaluator(config, tmp_path, wandb)
    manager.submit(None, None, config, 2000, 0.04)
    job = manager.pending
    job.process.returncode = 1 if problem == "exit" else 0
    if problem == "partial":
        _summary(manager, attempted_episodes=3)
    elif problem == "invalid_json":
        _summary(manager)
        (job.output_dir / "summary.json").write_text("not json")
    elif problem == "nan":
        _summary(manager, success_rate=float("nan"))
    manager.poll()
    assert manager.pending is None
    assert not wandb.logs
    assert not (tmp_path / "metrics" / "sim_eval_metrics.csv").exists()
    assert not (tmp_path / "checkpoints" / "best_success.pt").exists()
    assert not job.checkpoint.exists()
    error = json.loads((tmp_path / "metrics" / "sim_eval_errors.jsonl").read_text())
    assert error["step"] == 2000
    assert manager.submit(None, None, config, 4000, 0.03)


def test_launch_failure_is_recorded_and_snapshot_removed(tmp_path, launched, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("unable to spawn")

    monkeypatch.setattr(sim_eval.subprocess, "Popen", fail)
    config = _config()
    manager = sim_eval.AsyncMujocoEvaluator(config, tmp_path, None)
    assert not manager.submit(None, None, config, 1, 0.1)
    assert not list((tmp_path / "eval").glob("*.pt"))
    assert "unable to spawn" in (tmp_path / "metrics" / "sim_eval_errors.jsonl").read_text()


def test_finish_drains_pending_work_and_optional_snapshot_retention(tmp_path, launched):
    config = _config(sim_eval_keep_checkpoints=True)
    manager = sim_eval.AsyncMujocoEvaluator(config, tmp_path, None)
    manager.submit(None, None, config, 2000, 0.04)
    job = manager.pending
    _summary(manager, successes=0)
    manager.finish()
    assert job.process.waits == [None]
    assert job.checkpoint.exists()
    assert manager.pending is None
    assert manager.best_success == 0.0
    manager.finish()
    manager.close()


@pytest.mark.parametrize("timeout", [False, True])
def test_close_terminates_only_owned_subprocess_group(tmp_path, launched, monkeypatch, timeout):
    config = _config()
    manager = sim_eval.AsyncMujocoEvaluator(config, tmp_path, None)
    manager.submit(None, None, config, 2000, 0.04)
    job = manager.pending
    signals = []
    monkeypatch.setattr(sim_eval.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    if timeout:
        original_wait = job.process.wait

        def wait(timeout=None):
            if len(signals) == 1:
                raise subprocess.TimeoutExpired(job.process.command, timeout)
            return original_wait(timeout)

        monkeypatch.setattr(job.process, "wait", wait)
    manager.close()
    assert signals == [(job.process.pid, signal.SIGTERM), (job.process.pid, signal.SIGKILL)]
    assert manager.pending is None
    assert not job.checkpoint.exists()
    manager.close()  # Idempotent; no more signals.
    assert len(signals) == 2


@pytest.mark.parametrize("operation,return_code", [("poll", 1), ("close", 1), ("close", 0)])
def test_exited_coordinator_still_has_owned_children_terminated(
    tmp_path, launched, monkeypatch, operation, return_code
):
    config = _config()
    manager = sim_eval.AsyncMujocoEvaluator(config, tmp_path, None)
    manager.submit(None, None, config, 2000, 0.04)
    job = manager.pending
    job.process.returncode = return_code
    signals = []
    monkeypatch.setattr(sim_eval.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    getattr(manager, operation)()
    assert signals == [(job.process.pid, signal.SIGTERM), (job.process.pid, signal.SIGKILL)]
    assert all(timeout == 5 for timeout in job.process.waits)
    assert manager.pending is None
    assert not job.checkpoint.exists()


def test_successful_poll_does_not_signal_already_joined_workers(tmp_path, launched, monkeypatch):
    config = _config()
    manager = sim_eval.AsyncMujocoEvaluator(config, tmp_path, None)
    manager.submit(None, None, config, 2000, 0.04)
    _summary(manager)
    manager.pending.process.returncode = 0
    signals = []
    monkeypatch.setattr(sim_eval.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    manager.poll()
    assert not signals
    assert manager.pending is None


def test_video_failures_preserve_success_and_selected_videos_are_logged(tmp_path, launched, monkeypatch):
    recorded = []

    def video(path, **kwargs):
        recorded.append((path, kwargs))
        return "video-object"

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Video=video))
    config = _config()
    wandb = FakeWandbRun()
    manager = sim_eval.AsyncMujocoEvaluator(config, tmp_path, wandb)
    manager.submit(None, None, config, 2000, 0.04)
    _summary(manager, selected_videos=[
        {"path": "success.mp4", "success": True, "seed": 2000000},
        {"path": "missing.mp4", "success": False, "seed": 2000001},
    ], video_errors=["rendering seed 2000002 failed"])
    (manager.pending.output_dir / "success.mp4").write_bytes(b"fake mp4")
    manager.finish()
    assert wandb.logs[0]["sim_eval/success_rate"] == 0.5
    assert wandb.logs[0]["sim_eval/success_1"] == "video-object"
    assert len(recorded) == 1
    errors = (tmp_path / "metrics" / "sim_eval_errors.jsonl").read_text().splitlines()
    assert len(errors) == 2


def test_wandb_failure_does_not_discard_local_score_or_best_checkpoint(tmp_path, launched):
    class BrokenWandb(FakeWandbRun):
        def log(self, payload):
            raise RuntimeError("W&B disconnected")

    config = _config()
    manager = sim_eval.AsyncMujocoEvaluator(config, tmp_path, BrokenWandb())
    manager.submit(None, None, config, 2000, 0.04)
    _summary(manager)
    manager.finish()
    assert manager.best_success == 0.5
    assert (tmp_path / "checkpoints" / "best_success.pt").is_file()
    assert (tmp_path / "metrics" / "sim_eval_metrics.csv").is_file()
    assert "W&B disconnected" in (tmp_path / "metrics" / "sim_eval_errors.jsonl").read_text()


@pytest.mark.parametrize("options", [
    {"sim_eval_episodes": 0}, {"sim_eval_num_workers": -1},
    {"sim_eval_worker_threads": 1.5}, {"sim_eval_n_exec": 9},
    {"sim_eval_seed": -1}, {"sim_eval_success_videos": -1},
    {"sim_eval_video_backend": "glfw"}, {"sim_eval_max_steps": 0},
])
def test_invalid_settings_fail_early(tmp_path, options):
    with pytest.raises(ValueError):
        sim_eval.AsyncMujocoEvaluator(_config(**options), tmp_path, None)


def test_snapshot_uses_real_checkpoint_with_optimizer_and_offline_best_metric(tmp_path, monkeypatch):
    monkeypatch.setattr(sim_eval.subprocess, "Popen", FakeProcess)
    config = _config()
    manager = sim_eval.AsyncMujocoEvaluator(config, tmp_path, None)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    manager.submit(model, optimizer, config, 2000, 0.04)
    _summary(manager)
    manager.finish()
    checkpoint = torch.load(tmp_path / "checkpoints" / "best_success.pt", weights_only=False)
    assert checkpoint["step"] == 2000
    assert checkpoint["optimizer_state"]["state"]
    assert checkpoint["best_metric"] == 0.04
    assert checkpoint["config"] == config
    for name, weights in model.state_dict().items():
        torch.testing.assert_close(checkpoint["model_state"][name], weights)

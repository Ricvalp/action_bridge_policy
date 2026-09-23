"""Asynchronous evaluation owns its snapshot and exactly one child process."""
import json
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest
import torch

from action_bridge.configs.sb_pusht import get_config
from action_bridge.eval import revision_pusht_async as asynchronous
from action_bridge.plan_revision import checkpoints


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.wait = Mock(side_effect=self._wait)
        self.terminate = Mock()
        self.kill = Mock()

    def poll(self):
        return self.returncode

    def _wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake evaluation", timeout)
        return self.returncode


@pytest.fixture
def setup(tmp_path, monkeypatch):
    processes, calls = [], []

    def popen(command, **kwargs):
        processes.append(FakeProcess())
        calls.append((command, kwargs))
        return processes[-1]

    monkeypatch.setattr(asynchronous.subprocess, "Popen", popen)
    monkeypatch.setattr(checkpoints, "runtime_identity", lambda: {"policy_commit": "test"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = get_config("ddim") | {"validation_seeds": [12, 56, 90]}
    callback = Mock()
    manager = asynchronous.AsyncEvaluation(tmp_path, config, callback)
    payload = {"config": config, "step": 10000, "ema": {"weight": torch.tensor([1.])},
               "metadata": {}, "dependencies": {}}
    return manager, payload, callback, processes, calls


def complete(manager, process, *, metrics=None, code=0):
    directory = manager._job.directory / "results"
    directory.mkdir(exist_ok=True)
    if metrics is not None:
        (directory / "metrics.json").write_text(json.dumps(metrics))
    process.returncode = code


def test_snapshot_is_immutable_and_callback_receives_matching_checkpoint(setup):
    manager, payload, callback, processes, calls = setup
    assert manager.submit(payload)
    snapshot = manager._job.checkpoint
    saved = snapshot.read_bytes()
    payload["ema"]["weight"].fill_(99)
    assert checkpoints.load(snapshot)["ema"]["weight"].item() == 1
    assert snapshot.read_bytes() == saved
    metrics = {"success_rate": .4, "episodes": 3}

    def consume(step, result, checkpoint):
        assert step == 10000
        assert result == metrics
        assert Path(checkpoint).read_bytes() == saved

    callback.side_effect = consume
    complete(manager, processes[0], metrics=metrics)
    manager.poll()
    callback.assert_called_once_with(10000, metrics, snapshot)
    assert not snapshot.exists()
    assert (snapshot.parent / "results/metrics.json").is_file()
    assert (snapshot.parent / "worker.log").is_file()
    assert not manager.busy
    assert manager.last_submitted_step == 10000


def test_busy_interval_does_not_wait_or_queue_another_worker(setup):
    manager, payload, callback, processes, calls = setup
    assert manager.submit(payload)
    assert manager.busy
    assert not manager.submit(payload | {"step": 20000})
    manager.poll()
    assert len(processes) == 1
    processes[0].wait.assert_not_called()
    callback.assert_not_called()
    assert manager.last_submitted_step == 10000


def test_worker_uses_common_cli_exact_seeds_and_cpu_isolation(setup, monkeypatch):
    manager, payload, callback, processes, calls = setup
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("WANDB_MODE", "online")
    manager.submit(payload)
    command, arguments = calls[0]
    assert command[:3] == [asynchronous.sys.executable, "-m", "action_bridge.scripts.eval_pusht_ddim"]
    assert command[command.index("--seeds") + 1:] == ["12", "56", "90"]
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--threads") + 1] == "2"
    assert "--no-save-videos" not in command
    environment = arguments["env"]
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["WANDB_MODE"] == "disabled"
    assert environment["OMP_NUM_THREADS"] == environment["MKL_NUM_THREADS"] == "2"
    assert arguments["stderr"] == subprocess.STDOUT
    assert arguments["stdout"].closed
    assert asynchronous.os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert asynchronous.os.environ["WANDB_MODE"] == "online"


def test_gpu_override_keeps_visible_devices_and_can_disable_media(setup, monkeypatch, tmp_path):
    _, payload, callback, processes, calls = setup
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    manager = asynchronous.AsyncEvaluation(tmp_path, payload["config"], callback,
                                          device="cuda", threads=1, save_videos=False)
    manager.submit(payload)
    assert "--no-save-videos" in calls[0][0]
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "3"


@pytest.mark.parametrize("metrics,code", [
    ({"success_rate": .9}, 1), (None, 0), ({"other": .9}, 0),
    ({"success_rate": float("nan")}, 0), ({"success_rate": 2}, 0),
    ({"success_rate": "0.5"}, 0), ([], 0),
])
def test_worker_failure_is_visible_and_never_fabricates_success(setup, metrics, code):
    manager, payload, callback, processes, calls = setup
    manager.submit(payload)
    snapshot = manager._job.checkpoint
    complete(manager, processes[0], metrics=metrics, code=code)
    with pytest.warns(RuntimeWarning, match="Closed-loop evaluation at step 10000 failed"):
        manager.poll()
    callback.assert_not_called()
    errors = [json.loads(line) for line in (manager.output / "errors.jsonl").read_text().splitlines()]
    assert errors[0]["step"] == 10000
    assert errors[0]["error"]
    assert not manager.busy
    assert not snapshot.exists()
    assert (snapshot.parent / "worker.log").exists()
    # A failure does not permanently prevent future intervals from evaluating.
    assert manager.submit(payload | {"step": 20000})


def test_launch_failure_is_reported_without_becoming_busy(setup, monkeypatch):
    manager, payload, callback, processes, calls = setup
    monkeypatch.setattr(asynchronous.subprocess, "Popen", Mock(side_effect=OSError("cannot fork")))
    with pytest.warns(RuntimeWarning, match="Cannot start evaluation: cannot fork"):
        assert not manager.submit(payload)
    assert not manager.busy
    assert manager.last_submitted_step is None
    callback.assert_not_called()
    assert not list(manager.output.glob("*/checkpoint.pt"))


def test_finish_waits_until_outstanding_result_is_collected(setup):
    manager, payload, callback, processes, calls = setup
    manager.submit(payload)
    process = processes[0]

    def finish_wait(timeout):
        if process.wait.call_count == 1:
            raise subprocess.TimeoutExpired("fake evaluation", timeout)
        complete(manager, process, metrics={"success_rate": .2})
        return 0

    process.wait.side_effect = finish_wait
    manager.finish()
    assert process.wait.call_count == 2
    assert all(call.kwargs == {"timeout": 1} for call in process.wait.call_args_list)
    callback.assert_called_once()
    assert not manager.busy


def test_close_terminates_only_owned_child_and_retains_logs(setup):
    manager, payload, callback, processes, calls = setup
    manager.submit(payload)
    snapshot = manager._job.checkpoint
    process = processes[0]
    process.wait.side_effect = [subprocess.TimeoutExpired("fake", 5), 0]
    unrelated = FakeProcess()
    manager.close()
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_count == 2
    unrelated.terminate.assert_not_called()
    unrelated.kill.assert_not_called()
    callback.assert_not_called()
    assert not snapshot.exists()
    assert (snapshot.parent / "worker.log").is_file()
    assert not manager.busy
    manager.close()  # Idempotent.
    process.terminate.assert_called_once()


def test_context_cancels_on_training_exception(setup):
    manager, payload, callback, processes, calls = setup
    with pytest.raises(RuntimeError, match="training failed"):
        with manager:
            manager.submit(payload)
            processes[0].wait.side_effect = None
            raise RuntimeError("training failed")
    processes[0].terminate.assert_called_once()
    callback.assert_not_called()


def test_parent_callback_exception_propagates_and_keeps_snapshot(setup):
    manager, payload, callback, processes, calls = setup
    manager.submit(payload)
    snapshot = manager._job.checkpoint
    callback.side_effect = RuntimeError("logging failed")
    complete(manager, processes[0], metrics={"success_rate": .5})
    with pytest.raises(RuntimeError, match="logging failed"):
        manager.poll()
    assert snapshot.exists()
    assert not (manager.output / "errors.jsonl").exists()
    assert not manager.busy


def test_same_step_resubmission_uses_fresh_directory(setup):
    manager, payload, callback, processes, calls = setup
    manager.submit(payload)
    first = manager._job.directory
    complete(manager, processes[0], metrics={"success_rate": .1})
    assert manager.submit(payload)
    assert manager._job.directory != first
    callback.assert_called_once()


@pytest.mark.parametrize("seeds", [[], [-1], ["12"]])
def test_invalid_seeds_fail_before_a_worker_is_created(tmp_path, seeds):
    with pytest.raises(ValueError, match="validation_seeds"):
        asynchronous.AsyncEvaluation(tmp_path, get_config() | {"validation_seeds": seeds}, Mock())

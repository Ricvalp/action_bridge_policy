"""Gap diagnostics travel through the existing evaluator and async logger."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from action_bridge.configs.sb_pusht import get_config
from action_bridge.eval import revision_pusht_async as asynchronous
from action_bridge.eval import revision_pusht_cli as cli
from action_bridge.scripts import sb_pusht


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    windows = tmp_path / "windows.pt"
    windows.write_bytes(b"prepared windows")
    state = {
        "config": get_config("sb_ou"), "ema": {}, "direction": "forward",
        "step": 75000, "metadata": {"normalizer_id": "original"}, "dependencies": {},
    }
    data = {"records": {"validation": "held-out windows"}}
    policy = object()
    restore = Mock(return_value=policy)
    online_metrics = {"success_rate": .5, "gap_samples": 10, "pusher_gap_mean_px": 3.}
    diagnostic = {
        "metrics": {"offline_self_gap_samples": 4, "offline_self_pusher_gap_mean_px": 40.,
                    "offline_expert_gap_samples": 4, "offline_expert_pusher_gap_mean_px": 2.},
        "records": {"self": [{"gap": 40.}], "expert": [{"gap": 2.}]},
        "selection": {"episode_ids": [101], "seed": 7301},
    }
    diagnose = Mock(return_value=diagnostic)
    serial = Mock(return_value=online_metrics)
    parallel = Mock(return_value=online_metrics)
    monkeypatch.setattr(cli.checkpoints, "load", lambda stream: (
        data if Path(stream.name) == windows else state))
    monkeypatch.setattr(cli.checkpoints, "restore_policy", restore)
    monkeypatch.setattr(cli.checkpoints, "runtime_identity", lambda: {})
    monkeypatch.setattr(cli, "evaluate", serial)
    monkeypatch.setattr(cli, "evaluate_parallel", parallel)
    monkeypatch.setattr(cli, "diagnose_offline_sources", diagnose)
    monkeypatch.setattr(cli.torch, "set_num_threads", Mock())
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    return SimpleNamespace(
        checkpoint=checkpoint, windows=windows, state=state, data=data,
        output=tmp_path / "eval", policy=policy, restore=restore, diagnostic=diagnostic,
        diagnose=diagnose, serial=serial, parallel=parallel, online_metrics=online_metrics,
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_gap_diagnostics_use_ema_and_merge_with_online_results(evaluation, workers, capsys):
    run = evaluation
    original = copy.deepcopy(run.state)
    assert cli.main("sb_ou", [
        "--checkpoint", str(run.checkpoint), "--source-gap-windows", str(run.windows),
        "--source-gap-episodes", "3", "--source-gap-replans", "5", "--completion", "repeat",
        "--workers", str(workers), "--device", "cpu", "--output-dir", str(run.output),
        "--episodes", "2", "--no-save-videos", "--no-progress",
    ]) == 0

    run.restore.assert_called_once_with(run.state, "cpu")
    assert run.serial.call_count == (workers == 1)
    assert run.parallel.call_count == (workers == 2)
    args, kwargs = run.diagnose.call_args
    assert args[0] is run.policy
    assert args[1]["completion_id"] == 0
    assert args[2:] == (run.state["metadata"], run.state["dependencies"], run.data, "cpu")
    assert kwargs == {"episodes": 3, "replans": 5, "completion_id": 0}
    assert run.state == original
    expected = run.online_metrics | run.diagnostic["metrics"]
    assert json.loads((run.output / "metrics.json").read_text()) == expected
    assert json.loads((run.output / "source_gaps.json").read_text()) == run.diagnostic
    identity = json.loads((run.output / "evaluation.json").read_text())
    assert identity["source_gap_diagnostic"] == {
        "windows": str(run.windows), "windows_sha256": hashlib.sha256(b"prepared windows").hexdigest(),
        "episodes": 3, "replans": 5, "selection": run.diagnostic["selection"],
    }
    assert '"offline_self_pusher_gap_mean_px": 40.0' in capsys.readouterr().out


def test_checkpoint_alone_does_not_need_training_windows(evaluation):
    run = evaluation
    assert cli.main("sb_ou", [
        "--checkpoint", str(run.checkpoint), "--output-dir", str(run.output), "--device", "cpu",
    ]) == 0
    run.diagnose.assert_not_called()
    assert not (run.output / "source_gaps.json").exists()
    assert "source_gap_diagnostic" not in json.loads((run.output / "evaluation.json").read_text())


@pytest.mark.parametrize("workers", [1, 2])
def test_mismatched_windows_fail_before_serial_or_parallel_simulation(evaluation, workers):
    run = evaluation
    run.diagnose.side_effect = ValueError("Window cache and evaluation differ in execute")
    with pytest.raises(ValueError, match="differ in execute"):
        cli.main("sb_ou", [
            "--checkpoint", str(run.checkpoint), "--source-gap-windows", str(run.windows),
            "--workers", str(workers), "--device", "cpu", "--output-dir", str(run.output),
        ])
    run.serial.assert_not_called()
    run.parallel.assert_not_called()
    assert not (run.output / "metrics.json").exists()


@pytest.mark.parametrize("extra", [
    ["--source-gap-episodes", "0"], ["--source-gap-replans", "0"],
    ["--source-gap-replans", "1"],
    ["--source-gap-episodes", "-1"], ["--source-gap-windows", "/missing/windows.pt"],
])
def test_invalid_offline_options_fail_before_running_simulation(evaluation, extra):
    run = evaluation
    with pytest.raises(SystemExit):
        cli.main("sb_ou", ["--checkpoint", str(run.checkpoint),
                            "--output-dir", str(run.output), *extra])
    run.serial.assert_not_called()
    run.parallel.assert_not_called()
    run.diagnose.assert_not_called()
    assert not run.output.exists()


def test_ddim_does_not_have_a_carried_plan_gap(evaluation):
    run = evaluation
    with pytest.raises(SystemExit):
        cli.main("ddim", ["--checkpoint", str(run.checkpoint),
                           "--source-gap-windows", str(run.windows)])
    run.serial.assert_not_called()
    run.diagnose.assert_not_called()


def test_async_worker_receives_windows_and_returns_all_diagnostic_metrics(tmp_path, monkeypatch):
    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr(asynchronous.subprocess, "Popen", popen)
    monkeypatch.setattr(asynchronous.checkpoints, "save", lambda path, payload: path.write_bytes(b"ema"))
    callback = Mock()
    config = get_config("sb_ou")
    windows = tmp_path / "windows.pt"
    windows.write_bytes(b"prepared")
    manager = asynchronous.AsyncEvaluation(
        tmp_path / "sb_ou", config, callback, source_gap_windows=windows,
        source_gap_episodes=3, source_gap_replans=7,
    )
    assert manager.submit({"config": config, "step": 75000})
    command = popen.call_args.args[0]
    assert command[command.index("--source-gap-windows") + 1] == str(windows)
    assert command[command.index("--source-gap-episodes") + 1] == "3"
    assert command[command.index("--source-gap-replans") + 1] == "7"
    assert command[command.index("--seeds") + 1:] == list(map(str, config["validation_seeds"]))
    checkpoint = manager._job.checkpoint
    results = manager._job.directory / "results"
    results.mkdir()
    metrics = {"success_rate": .3, "offline_self_gap_samples": 12,
               "offline_expert_pusher_gap_mean_px": 2.5, "pusher_gap_mean_px": 6.}
    (results / "metrics.json").write_text(json.dumps(metrics))
    process.poll.return_value = 0
    manager.poll()
    callback.assert_called_once_with(75000, metrics, checkpoint)
    assert not checkpoint.exists()


@pytest.mark.parametrize("method,has_windows,enabled", [
    ("sb_ou", True, True), ("sb_kinetic", True, True), ("fm_paired", True, True),
    ("ddim", True, False), ("sb_ou", False, False),
])
def test_training_attaches_available_windows_without_changing_config(
        tmp_path, monkeypatch, method, has_windows, enabled):
    windows = tmp_path / "windows.pt"
    if has_windows:
        windows.write_bytes(b"prepared")
    constructor = Mock()
    monkeypatch.setattr(asynchronous, "AsyncEvaluation", constructor)
    config = get_config(method)
    before = copy.deepcopy(config)
    options = {"device": "cpu"}
    callback = Mock()
    factory = sb_pusht.stage_evaluation(tmp_path / method, config, options)
    factory(callback)
    assert constructor.call_args.args == (tmp_path / method, config, callback)
    expected = {"device": "cpu"}
    if enabled:
        expected["source_gap_windows"] = windows
    assert constructor.call_args.kwargs == expected
    assert config == before
    assert options == {"device": "cpu"}


def test_explicit_offline_disable_and_no_sim_eval_are_preserved(tmp_path, monkeypatch):
    (tmp_path / "windows.pt").write_bytes(b"prepared")
    constructor = Mock()
    monkeypatch.setattr(asynchronous, "AsyncEvaluation", constructor)
    config = get_config("sb_ou")
    assert sb_pusht.stage_evaluation(tmp_path / "sb_ou", config, False) is None
    constructor.assert_not_called()
    factory = sb_pusht.stage_evaluation(tmp_path / "sb_ou", config, {"source_gap_windows": None})
    factory(Mock())
    assert constructor.call_args.kwargs == {"source_gap_windows": None}

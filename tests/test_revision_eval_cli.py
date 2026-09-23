"""Method-specific evaluation commands use only a self-contained checkpoint."""

import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from action_bridge.configs.sb_pusht import METHODS, get_config
from action_bridge.eval import revision_pusht_cli as cli


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    """Replace inference and simulation, but exercise argument handling and files."""
    signature = inspect.signature(cli.evaluate)
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cli.torch, "set_num_threads", Mock())
    monkeypatch.setattr(cli, "CHECKOUT", tmp_path / "checkout")
    runtime = {"policy_commit": "evaluation-commit", "lock_sha256": "evaluation-lock"}
    monkeypatch.setattr(cli.checkpoints, "runtime_identity", lambda: runtime)

    def setup(method="ddim"):
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.write_bytes(b"dummy checkpoint; only used to test file identity")
        payload = {
            "config": get_config(method),
            "metadata": {
                "dataset_path": "/does/not/exist/replay.zarr",
                "dataset_sha256": "training-data-hash",
                "normalizer_id": "saved-normalizer",
                "observation_profile": "pusht-state",
                "action_profile": "absolute-target",
            },
            "dependencies": {} if method == "ddim" else {"saved_dependency": "weights"},
            "ema": {"saved_policy": "ema-weights"},
            "step": 12500,
            "complete": False,
            "direction": "forward",
            "runtime": {"policy_commit": "training-commit"},
        }
        policy = object()
        load = Mock(return_value=payload)
        restore = Mock(return_value=policy)
        evaluate = Mock(return_value={"success_rate": 0.5})
        monkeypatch.setattr(cli.checkpoints, "load", load)
        monkeypatch.setattr(cli.checkpoints, "restore_policy", restore)
        monkeypatch.setattr(cli, "evaluate", evaluate)
        return SimpleNamespace(
            checkpoint=checkpoint, payload=payload, policy=policy,
            output=tmp_path / "evaluation", load=load, restore=restore,
            evaluate=evaluate, signature=signature, runtime=runtime,
        )

    return setup


def evaluated_arguments(run):
    args, kwargs = run.evaluate.call_args
    return run.signature.bind(*args, **kwargs).arguments


@pytest.mark.parametrize("method", METHODS)
def test_method_scripts_evaluate_checkpoint_without_original_training_files(evaluation, method):
    run = evaluation(method)
    original = copy.deepcopy(run.payload)
    script = importlib.import_module(f"action_bridge.scripts.eval_pusht_{method}")

    assert script.main([
        "--checkpoint", str(run.checkpoint), "--device", "cpu",
        "--output-dir", str(run.output), "--episodes", "3", "--seed", "123",
        "--execute", "4", "--max-steps", "20", "--no-save-videos",
    ]) == 0

    run.load.assert_called_once()
    run.restore.assert_called_once_with(run.payload, "cpu")
    values = evaluated_arguments(run)
    assert values["policy"] is run.policy
    assert values["config"] == original["config"] | {
        "execute": 4, "max_episode_steps": 20, "evaluation_seeds": [123, 124, 125],
    }
    assert values["metadata"] == original["metadata"]
    assert values["dependencies"] == original["dependencies"]
    assert values["device"] == "cpu"
    assert list(values["seeds"]) == [123, 124, 125]
    assert values["render"] is False
    assert values["save_videos"] is False
    assert values["save_gifs"] is False
    assert Path(values["output"]) == run.output
    assert run.payload == original
    assert not Path(run.payload["metadata"]["dataset_path"]).exists()

    report = json.loads((run.output / "evaluation.json").read_text())
    assert report["method"] == method
    assert report["checkpoint"] == str(run.checkpoint.resolve())
    assert report["checkpoint_sha256"] == hashlib.sha256(run.checkpoint.read_bytes()).hexdigest()
    assert report["checkpoint_step"] == 12500
    assert report["weights"] == "ema"
    assert report["config"] == values["config"]
    assert report["metadata"] == original["metadata"]
    assert report["checkpoint_runtime"] == original["runtime"]
    assert report["evaluation_runtime"] == run.runtime
    assert report["seeds"] == [123, 124, 125]
    assert report["save_gifs"] is False
    assert report["save_videos"] is False


def test_defaults_preserve_trained_horizon_and_create_unique_workspace_outputs(evaluation):
    run = evaluation()
    arguments = ["--checkpoint", str(run.checkpoint)]
    outputs = []
    for _ in range(2):
        assert cli.main("ddim", arguments) == 0
        values = evaluated_arguments(run)
        assert values["config"] == run.payload["config"] | {
            "evaluation_seeds": list(range(1000000, 1000010)),
        }
        assert values["device"] == "cpu"
        assert list(values["seeds"]) == list(range(1000000, 1000010))
        assert values["render"] is True
        assert values["save_videos"] is True
        assert values["save_gifs"] is False
        output = Path(values["output"])
        assert output.parent == cli.CHECKOUT / "workspace/sb_pusht/evaluation"
        assert output.name.startswith("ddim-")
        assert (output / "evaluation.json").is_file()
        outputs.append(output)
    assert outputs[0] != outputs[1]
    cli.torch.set_num_threads.assert_called_with(4)


@pytest.mark.parametrize("option,value", [
    ("--episodes", "0"), ("--episodes", "-1"), ("--threads", "0"),
    ("--max-steps", "0"), ("--execute", "0"), ("--execute", "17"),
])
def test_invalid_arguments_fail_before_restoring_or_running_policy(evaluation, option, value):
    run = evaluation()
    with pytest.raises((SystemExit, ValueError)):
        cli.main("ddim", ["--checkpoint", str(run.checkpoint),
                          "--output-dir", str(run.output), option, value])
    run.restore.assert_not_called()
    run.evaluate.assert_not_called()
    assert not run.output.exists()


@pytest.mark.parametrize("method", METHODS[1:])
def test_method_mismatch_is_rejected_before_simulation(evaluation, method):
    run = evaluation("ddim")
    with pytest.raises((SystemExit, ValueError)):
        cli.main(method, ["--checkpoint", str(run.checkpoint), "--output-dir", str(run.output)])
    run.restore.assert_not_called()
    run.evaluate.assert_not_called()
    assert not run.output.exists()


@pytest.mark.parametrize("method", ["sb_ou", "sb_kinetic"])
def test_reverse_phase_sb_checkpoint_is_not_an_evaluation_policy(evaluation, method):
    run = evaluation(method)
    run.payload["direction"] = "reverse"
    with pytest.raises((SystemExit, ValueError)):
        cli.main(method, ["--checkpoint", str(run.checkpoint), "--output-dir", str(run.output)])
    run.restore.assert_not_called()
    run.evaluate.assert_not_called()


def test_existing_evaluation_directory_is_not_overwritten(evaluation):
    run = evaluation()
    run.output.mkdir()
    existing = run.output / "metrics.json"
    existing.write_text('{"previous_result": true}\n')
    with pytest.raises((SystemExit, ValueError, FileExistsError)):
        cli.main("ddim", ["--checkpoint", str(run.checkpoint), "--output-dir", str(run.output)])
    run.evaluate.assert_not_called()
    assert existing.read_text() == '{"previous_result": true}\n'
    assert sorted(run.output.iterdir()) == [existing]


@pytest.mark.parametrize("method", METHODS[1:])
def test_revision_completion_choice_reaches_evaluator(evaluation, method):
    run = evaluation(method)
    assert cli.main(method, ["--checkpoint", str(run.checkpoint),
                            "--output-dir", str(run.output), "--completion", "repeat"]) == 0
    assert evaluated_arguments(run)["completion_id"] == 0
    assert run.payload["config"]["completion_id"] == 2


def test_ddim_rejects_revision_only_completion_flag(evaluation):
    run = evaluation()
    with pytest.raises(SystemExit):
        cli.main("ddim", ["--checkpoint", str(run.checkpoint), "--completion", "repeat"])
    run.evaluate.assert_not_called()


def test_checkpoint_hash_identifies_loaded_file_when_training_replaces_latest(evaluation):
    run = evaluation()
    original_hash = hashlib.sha256(run.checkpoint.read_bytes()).hexdigest()

    def load_while_training_saves(stream):
        replacement = run.checkpoint.with_suffix(".tmp")
        replacement.write_bytes(b"newer checkpoint from concurrent training")
        replacement.replace(run.checkpoint)
        stream.read()
        return run.payload

    run.load.side_effect = load_while_training_saves
    assert cli.main("ddim", ["--checkpoint", str(run.checkpoint),
                             "--output-dir", str(run.output)]) == 0
    report = json.loads((run.output / "evaluation.json").read_text())
    assert report["checkpoint_sha256"] == original_hash
    assert report["checkpoint_sha256"] != hashlib.sha256(run.checkpoint.read_bytes()).hexdigest()


def test_explicit_validation_seeds_are_not_replaced_with_consecutive_seeds(evaluation, capsys):
    run = evaluation()
    assert cli.main("ddim", ["--checkpoint", str(run.checkpoint),
                            "--output-dir", str(run.output), "--episodes", "20", "--seed", "456",
                            "--seeds", "5", "17", "99"]) == 0
    assert evaluated_arguments(run)["seeds"] == [5, 17, 99]
    assert evaluated_arguments(run)["config"]["evaluation_seeds"] == [5, 17, 99]
    report = json.loads((run.output / "evaluation.json").read_text())
    assert report["seeds"] == [5, 17, 99]
    assert "episodes=3" in capsys.readouterr().out


def test_negative_explicit_seed_is_rejected_before_simulation(evaluation):
    run = evaluation()
    with pytest.raises(SystemExit):
        cli.main("ddim", ["--checkpoint", str(run.checkpoint), "--seeds", "5", "-1"])
    run.restore.assert_not_called()
    run.evaluate.assert_not_called()


def test_gif_only_cli_rendering_is_explicit(evaluation):
    run = evaluation()
    assert cli.main("ddim", ["--checkpoint", str(run.checkpoint),
                            "--output-dir", str(run.output), "--no-save-videos", "--save-gifs"]) == 0
    values = evaluated_arguments(run)
    assert values["render"] is True
    assert values["save_videos"] is False and values["save_gifs"] is True

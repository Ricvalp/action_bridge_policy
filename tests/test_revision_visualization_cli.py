"""Revision visualization is a read-only checkpoint consumer with fresh traces."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from action_bridge.configs.sb_pusht import get_config
from action_bridge.scripts import visualize_pusht_revision as cli


@pytest.fixture
def visualization(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.torch, "set_num_threads", Mock())
    monkeypatch.setattr(cli, "CHECKOUT", tmp_path / "checkout")
    runtime = {"policy_commit": "visualization-commit", "lock_sha256": "saved-lock"}
    monkeypatch.setattr(cli.checkpoints, "runtime_identity", lambda: runtime)
    for name in ("SDL_VIDEODRIVER", "SDL_AUDIODRIVER", "MPLBACKEND", "MPLCONFIGDIR"):
        monkeypatch.delenv(name, raising=False)

    def setup(method="sb_kinetic"):
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.write_bytes(b"trusted checkpoint fixture")
        payload = dict(
            config=get_config(method),
            metadata={"dataset_path": "/missing/replay.zarr", "dataset_sha256": "dataset-hash",
                      "observation_profile": "pusht-state", "action_profile": "absolute-target"},
            dependencies={"frozen_completion": "weights"}, ema={"policy": "weights"},
            step=12500, direction="forward", runtime={"policy_commit": "training-commit"})
        policy = object()
        episode = {"seed": 1_000_000, "plan_traces": [{"step": 0}]}
        artifacts = {"video": "revision.mp4", "storyboards": ["replan-1.png"],
                     "gif": None, "replans": [1, 2, 3]}

        def collect(*args, **kwargs):
            assert cli.os.environ["SDL_VIDEODRIVER"] == "dummy"
            assert cli.os.environ["SDL_AUDIODRIVER"] == "dummy"
            assert cli.os.environ["MPLBACKEND"] == "Agg"
            output = kwargs["output"]
            seed = kwargs["seeds"][0]
            (output / f"episode-seed{seed}.json").write_text(json.dumps(episode))
            return {"success_rate": 0.0}

        load = Mock(return_value=payload)
        restore = Mock(return_value=policy)
        evaluate = Mock(side_effect=collect)
        renderer = Mock(return_value=artifacts)
        monkeypatch.setattr(cli.checkpoints, "load", load)
        monkeypatch.setattr(cli.checkpoints, "restore_policy", restore)
        monkeypatch.setattr(cli, "evaluate", evaluate)
        monkeypatch.setattr(cli, "render_revision_process", renderer)
        return SimpleNamespace(
            checkpoint=checkpoint, payload=payload, policy=policy, episode=episode,
            artifacts=artifacts, output=tmp_path / "visualization", runtime=runtime,
            load=load, restore=restore, evaluate=evaluate, renderer=renderer)

    return setup


@pytest.mark.parametrize("method", cli.METHODS)
def test_infers_method_collects_fresh_trace_and_records_identity(visualization, method, capsys):
    run = visualization(method)
    original = copy.deepcopy(run.payload)
    assert cli.main(["--checkpoint", str(run.checkpoint), "--output-dir", str(run.output)]) == 0

    run.load.assert_called_once()
    assert run.load.call_args.args[0].closed
    run.restore.assert_called_once_with(run.payload, "cpu")
    config = original["config"] | {"max_episode_steps": 32, "evaluation_seeds": [1_000_000]}
    run.evaluate.assert_called_once_with(
        run.policy, config, original["metadata"], original["dependencies"], "cpu",
        output=run.output, completion_id=2, seeds=[1_000_000], render=False,
        save_videos=False, save_gifs=False, progress=True, trace_generation=True, write_metrics=False)
    run.renderer.assert_called_once_with(
        run.episode, config, run.output, start_replan=1, replans=3, fps=10, save_gif=False)
    cli.torch.set_num_threads.assert_called_once_with(2)
    assert run.payload == original
    assert not Path(original["metadata"]["dataset_path"]).exists()

    report = json.loads((run.output / "visualization.json").read_text())
    assert report["method"] == method
    assert report["checkpoint"] == str(run.checkpoint.resolve())
    assert report["checkpoint_sha256"] == hashlib.sha256(run.checkpoint.read_bytes()).hexdigest()
    assert report["checkpoint_step"] == 12500
    assert report["weights"] == "ema"
    assert report["checkpoint_runtime"] == original["runtime"]
    assert report["visualization_runtime"] == run.runtime
    assert report["config"] == config
    assert report["metadata"] == original["metadata"]
    assert report["episode"] == "episode-seed1000000.json"
    assert report["seed"] == 1_000_000
    assert report["horizon"] == 16 and report["execute"] == 8
    assert report["completion"] == "learned_dissipative"
    assert report["diagnostic_only"] is True
    assert report["artifacts"] == run.artifacts
    assert {"completion", "source_noise", "revision", "execution", "startup"} == set(report["phase_semantics"])
    assert "torch" in report["versions"]
    assert report["arguments"]["checkpoint"] == str(run.checkpoint)
    assert "not a success-rate benchmark" in capsys.readouterr().out


def test_default_outputs_are_fresh_timestamped_workspace_directories(visualization):
    run = visualization()
    outputs = []
    for _ in range(2):
        assert cli.main(["--checkpoint", str(run.checkpoint)]) == 0
        output = run.evaluate.call_args.kwargs["output"]
        assert output.parent == cli.CHECKOUT / "workspace/sb_pusht/visualizations"
        assert output.name.startswith("sb_kinetic-")
        assert (output / "visualization.json").is_file()
        outputs.append(output)
    assert outputs[0] != outputs[1]


@pytest.mark.parametrize("completion_id,mode", list(enumerate(cli.COMPLETION_NAMES)))
def test_overrides_completion_execution_range_and_media(visualization, completion_id, mode):
    run = visualization("fm_paired")
    run.payload["config"]["max_episode_steps"] = 25
    if mode == "direct_mlp":
        run.payload["config"].update(training_completion_modes=[0, 1, 3], execute=4)
    assert cli.main([
        "--checkpoint", str(run.checkpoint), "--output-dir", str(run.output),
        "--device", "cpu", "--threads", "3", "--seed", "7", "--execute", "4",
        "--completion", mode, "--start-replan", "10", "--replans", "2",
        "--fps", "6.5", "--save-gif", "--no-progress",
    ]) == 0
    config = run.evaluate.call_args.args[1]
    assert config["execute"] == 4
    assert config["completion_id"] == completion_id
    assert config["max_episode_steps"] == 25  # Never expand the saved episode limit.
    assert config["evaluation_seeds"] == [7]
    assert run.evaluate.call_args.kwargs["progress"] is False
    run.renderer.assert_called_once_with(
        run.episode, config, run.output, start_replan=10, replans=2, fps=6.5, save_gif=True)
    cli.torch.set_num_threads.assert_called_once_with(3)


def test_inherits_completion_and_accepts_startup_and_k_equal_h(visualization):
    run = visualization()
    run.payload["config"]["completion_id"] = 0
    assert cli.main([
        "--checkpoint", str(run.checkpoint), "--output-dir", str(run.output),
        "--execute", "16", "--start-replan", "0", "--replans", "1", "--seed", "0",
    ]) == 0
    config = run.evaluate.call_args.args[1]
    assert config["execute"] == config["horizon"] == config["max_episode_steps"] == 16
    assert config["completion_id"] == 0
    assert run.renderer.call_args.kwargs["start_replan"] == 0


@pytest.mark.parametrize("arguments,message", [
    (["--threads", "0"], "--threads must be positive"),
    (["--execute", "0"], "--execute must be positive"),
    (["--replans", "0"], "--replans must be positive"),
    (["--fps", "0"], "--fps must be positive"),
    (["--fps", "nan"], "--fps must be positive"),
    (["--fps", "inf"], "--fps must be positive"),
    (["--seed", "-1"], "--seed must be nonnegative"),
    (["--start-replan", "-1"], "--start-replan must be nonnegative"),
    (["--execute", "17"], "--execute must be between"),
])
def test_invalid_arguments_do_not_restore_or_write(visualization, arguments, message, capsys):
    run = visualization()
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--checkpoint", str(run.checkpoint), "--output-dir", str(run.output), *arguments])
    assert message in capsys.readouterr().err
    run.restore.assert_not_called()
    run.evaluate.assert_not_called()
    assert not run.output.exists()


@pytest.mark.parametrize("change,message", [
    ({"method": "ddim"}, "DDIM has no old-plan completion/revision process"),
    ({"method": "other"}, "Unsupported revision method"),
    ({"protocol": "retired_protocol"}, "self_source_v1"),
    ({"obs_dim": 6}, "5-state/2-target"),
    ({"action_dim": 3}, "5-state/2-target"),
    ({"completion_id": len(cli.COMPLETION_NAMES)}, "completion_id must be"),
    ({"max_episode_steps": 0}, "max_episode_steps must be positive"),
])
def test_incompatible_checkpoint_config_is_rejected(visualization, change, message, capsys):
    run = visualization()
    run.payload["config"].update(change)
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--checkpoint", str(run.checkpoint), "--output-dir", str(run.output)])
    assert message in capsys.readouterr().err
    run.restore.assert_not_called()
    assert not run.output.exists()


@pytest.mark.parametrize("trained,execute,message", [
    ([0, 1, 2], 8, "not a trained completion mode"),
    ([0, 1, 3], 4, "must match the K"),
])
def test_direct_tail_requires_trained_mode_and_checkpoint_execution_length(
        visualization, trained, execute, message, capsys):
    run = visualization("fm_paired")
    run.payload["config"].update(training_completion_modes=trained, completion_id=3, execute=8)
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--checkpoint", str(run.checkpoint), "--output-dir", str(run.output),
                  "--execute", str(execute)])
    assert message in capsys.readouterr().err
    run.restore.assert_not_called()
    assert not run.output.exists()


@pytest.mark.parametrize("problem,message", [("ema", "no EMA"), ("direction", "forward phase")])
def test_requires_ema_and_forward_sb_weights(visualization, problem, message, capsys):
    run = visualization()
    run.payload.pop(problem)
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--checkpoint", str(run.checkpoint), "--output-dir", str(run.output)])
    assert message in capsys.readouterr().err
    run.restore.assert_not_called()
    assert not run.output.exists()


def test_existing_output_and_missing_checkpoint_are_rejected(visualization, capsys):
    run = visualization()
    run.output.mkdir()
    marker = run.output / "existing.txt"
    marker.write_text("preserve me")
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--checkpoint", str(run.checkpoint), "--output-dir", str(run.output)])
    assert "already exists" in capsys.readouterr().err
    assert marker.read_text() == "preserve me"
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--checkpoint", str(run.checkpoint.with_name("missing.pt"))])
    assert "does not exist" in capsys.readouterr().err
    run.restore.assert_not_called()
    run.evaluate.assert_not_called()


def test_checkpoint_hash_uses_loaded_file_even_if_path_is_atomically_replaced(visualization):
    run = visualization()
    original_hash = hashlib.sha256(run.checkpoint.read_bytes()).hexdigest()

    def load(stream):
        replacement = run.checkpoint.with_name("replacement.pt")
        replacement.write_bytes(b"a newer training checkpoint")
        replacement.replace(run.checkpoint)
        stream.read()
        return run.payload

    run.load.side_effect = load
    assert cli.main(["--checkpoint", str(run.checkpoint), "--output-dir", str(run.output)]) == 0
    report = json.loads((run.output / "visualization.json").read_text())
    assert report["checkpoint_sha256"] == original_hash
    assert report["checkpoint_sha256"] != hashlib.sha256(run.checkpoint.read_bytes()).hexdigest()

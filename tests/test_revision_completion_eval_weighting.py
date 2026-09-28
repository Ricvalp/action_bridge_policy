"""Completion velocity ablations are explicit and propagate to evaluation."""
import json

import pytest
import torch

from action_bridge.eval import revision_pusht as evaluator
from action_bridge.eval import revision_pusht_cli as evaluation_cli
from action_bridge.plan_revision import checkpoints
from action_bridge.scripts import visualize_pusht_candidates as candidates_cli
from action_bridge.scripts import visualize_pusht_revision as visualization_cli
from test_revision_eval_cli import evaluation
from test_revision_generation_trace import scenario
from test_revision_visualization_cli import visualization


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_missing_checkpoint_key_keeps_last_pair_and_explicit_ablation_is_labeled():
    config = {"method": "sb_ou"}
    identity = evaluator.configure_completion_velocity_weighting(config, 2)
    assert identity == {
        "trained_completion_velocity_weighting": "last_pair",
        "completion_velocity_weighting": "last_pair",
        "completion_velocity_weighting_override": None,
    }
    assert config == {"method": "sb_ou"}
    identity = evaluator.configure_completion_velocity_weighting(config, 2, "linear")
    assert identity["trained_completion_velocity_weighting"] == "last_pair"
    assert identity["completion_velocity_weighting"] == "linear"
    assert identity["completion_velocity_weighting_override"] == "linear"
    assert config["completion_velocity_weighting"] == "linear"


@pytest.mark.parametrize("mode", [0, 1, 3])
def test_explicit_weighting_rejects_completion_modes_that_ignore_it(mode):
    with pytest.raises(ValueError, match="requires learned_dissipative"):
        evaluator.configure_completion_velocity_weighting({}, mode, "uniform")


@pytest.mark.parametrize("weighting", [None, "linear", "exp_quarter"])
def test_rollout_passes_trained_weighting_to_completion_and_reports_it(tmp_path, monkeypatch, weighting):
    policy, config, metadata, dependencies = scenario("sb_ou", monkeypatch)
    config.pop("completion_velocity_weighting", None)
    if weighting is not None:
        config["completion_velocity_weighting"] = weighting
    original = evaluator.complete_plan
    called = []

    def complete(*args, **kwargs):
        called.append(kwargs["velocity_weighting"])
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluator, "complete_plan", complete)
    metrics = evaluator.evaluate(policy, config, metadata, dependencies, "cpu", output=tmp_path,
                                 seeds=[71], progress=False, render=False, save_videos=False)
    expected = weighting or "last_pair"
    assert called and set(called) == {expected}
    assert metrics["completion_velocity_weighting"] == expected
    assert metrics["trained_completion_velocity_weighting"] == expected
    assert metrics["completion_velocity_weighting_override"] is None


@pytest.mark.parametrize("workers", [1, 2])
def test_closed_loop_cli_records_override_and_passes_it_to_serial_or_parallel(evaluation, workers):
    run = evaluation("sb_ou")
    run.payload["config"].pop("completion_velocity_weighting", None)
    assert evaluation_cli.main("sb_ou", [
        "--checkpoint", str(run.checkpoint), "--output-dir", str(run.output),
        "--workers", str(workers), "--device", "cpu", "--completion-velocity-weighting", "exp_half",
    ]) == 0
    runner = run.evaluate if workers == 1 else run.evaluate_parallel
    config = runner.call_args.args[1]
    assert config["completion_velocity_weighting"] == "exp_half"
    assert "completion_velocity_weighting" not in run.payload["config"]
    report = json.loads((run.output / "evaluation.json").read_text())
    assert report["trained_completion_velocity_weighting"] == "last_pair"
    assert report["completion_velocity_weighting"] == "exp_half"
    assert report["completion_velocity_weighting_override"] == "exp_half"


def test_revision_visualization_records_changed_initializer(visualization):
    run = visualization("sb_kinetic")
    run.payload["config"]["completion_velocity_weighting"] = "linear"
    assert visualization_cli.main([
        "--checkpoint", str(run.checkpoint), "--output-dir", str(run.output),
        "--completion-velocity-weighting", "uniform",
    ]) == 0
    assert run.evaluate.call_args.args[1]["completion_velocity_weighting"] == "uniform"
    report = json.loads((run.output / "visualization.json").read_text())
    assert report["trained_completion_velocity_weighting"] == "linear"
    assert report["completion_velocity_weighting_override"] == "uniform"


def test_candidate_visualization_uses_override_and_rejects_it_for_fixed_symmetric_scene(tmp_path, monkeypatch):
    policy, config, metadata, dependencies = scenario("sb_ou", monkeypatch)
    config.pop("completion_velocity_weighting", None)
    checkpoint = tmp_path / "best.pt"
    checkpoints.save(checkpoint, dict(config=config, metadata=metadata, dependencies=dependencies,
                                     ema=policy.state_dict(), direction="forward", step=50))
    monkeypatch.setattr(candidates_cli, "render_candidate_batch", lambda *args, **kwargs: {})
    output = tmp_path / "candidates"
    arguments = ["--checkpoint", str(checkpoint), "--output-dir", str(output),
                 "--candidates", "2", "--replan", "1", "--completion-velocity-weighting", "linear"]
    assert candidates_cli.main(arguments) == 0
    report = json.loads((output / "candidates.json").read_text())
    assert report["trained_completion_velocity_weighting"] == "last_pair"
    assert report["completion_velocity_weighting"] == "linear"
    assert report["completion_velocity_weighting_override"] == "linear"
    with pytest.raises(SystemExit, match="2"):
        candidates_cli.main(arguments + ["--scene", "symmetric"])

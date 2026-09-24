"""Full generation traces observe the existing closed loop without changing it."""
import json

import numpy as np
import pytest
import torch

from action_bridge.configs.sb_pusht import get_config
from action_bridge.eval import revision_pusht as evaluator
from action_bridge.plan_revision.completion import LearnedCompletion
from action_bridge.plan_revision.models import build_policy
from action_bridge.plan_revision.training import completion_config
from test_revision_rollout import MockEnv


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def scenario(method, monkeypatch, *, execute=2):
    config = get_config(method) | {
        "horizon": 4, "execute": execute, "channels": [8, 16], "history_dim": 16,
        "hidden_dim": 16, "time_dim": 8, "reference_hidden_dim": 8,
        "num_inference_steps": 4, "max_episode_steps": 10,
    }
    torch.manual_seed(19)
    policy = build_policy(config).eval()
    reference_config = completion_config(config)
    reference = LearnedCompletion(**reference_config)
    dependencies = {"completion_config": reference_config,
                    "completion_state": reference.state_dict(),
                    "innovation_variance": torch.ones(2)}
    metadata = {"codec": {"mean": [256., 256.], "std": [50., 50.],
                          "low": [0., 0.], "high": [512., 512.], "units": "pixels"},
                "normalization": {"obs_mean": [0.] * 5, "obs_std": [256.] * 5}}
    monkeypatch.setattr(evaluator, "_make_pusht_env", lambda **kwargs: MockEnv(episode_steps=100))
    return policy, config, metadata, dependencies


@pytest.mark.parametrize("method", ["fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic"])
@pytest.mark.parametrize("completion_id", [0, 1, 2])
def test_full_trace_preserves_actions_and_records_every_replanning_phase(
        tmp_path, monkeypatch, method, completion_id):
    policy, config, metadata, dependencies = scenario(method, monkeypatch)
    outputs = []
    for name, trace in (("ordinary", False), ("traced", True)):
        output = tmp_path / name
        evaluator.evaluate(policy, config, metadata, dependencies, "cpu", output=output,
                           seeds=[71], completion_id=completion_id, progress=False,
                           trace_generation=trace)
        outputs.append(json.loads((output / "episode-seed71.json").read_text()))
    ordinary, traced = outputs
    for key in ordinary:
        if key not in ("inference_seconds_per_replan", "plan_traces"):
            assert ordinary[key] == traced[key], key
    assert len(ordinary["plan_traces"]) == 3
    traces = traced["plan_traces"]
    assert len(traces) == 5
    assert traces[0]["startup"] and traces[0]["old_plan_raw"] is None
    previous = None
    for index, trace in enumerate(traces):
        assert trace["replan"] == index and trace["step"] == index * config["execute"]
        assert trace["state_raw"] == traced["states_raw"][trace["step"]]
        if previous is not None:
            assert trace["old_plan_raw"] == previous["final_raw"]
            assert trace["old_executed"] == 2
            assert trace["aligned_old_raw"] == previous["final_raw"][2:]
            assert trace["completed_raw"][:2] == trace["aligned_old_raw"]
            assert not trace["startup"]
        candidates = np.asarray(trace["revision_candidate_plans_raw"])
        expected_steps = 3 if method.startswith("fm_") else 5
        assert candidates.shape == (expected_steps, 4, 2)
        np.testing.assert_array_equal(candidates[0], trace["perturbed_source_raw"])
        np.testing.assert_array_equal(candidates[-1], trace["final_raw"])
        assert len(trace["revision_times"]) == expected_steps
        assert trace["revision_times"][0] == 0. and trace["revision_times"][-1] == 1.
        previous = trace


def test_full_execution_records_command_anchor_not_a_fictitious_retained_plan(tmp_path, monkeypatch):
    policy, config, metadata, dependencies = scenario("fm_paired", monkeypatch, execute=4)
    evaluator.evaluate(policy, config, metadata, dependencies, "cpu", output=tmp_path,
                       seeds=[71], progress=False, trace_generation=True)
    episode = json.loads((tmp_path / "episode-seed71.json").read_text())
    first, second, _ = episode["plan_traces"]
    assert second["old_plan_raw"] == first["final_raw"]
    assert second["old_executed"] == 4 and second["startup"]
    assert second["aligned_old_raw"] is None
    np.testing.assert_allclose(second["startup_anchor_raw"], episode["actions_raw"][3], atol=1e-5)
    np.testing.assert_array_equal(second["completed_raw"], [second["startup_anchor_raw"]] * 4)


def test_ddim_is_not_mislabeled_as_completion_and_revision(monkeypatch):
    with pytest.raises(ValueError, match="FM or SB"):
        evaluator.evaluate(None, get_config("ddim"), {}, {}, "cpu", trace_generation=True)

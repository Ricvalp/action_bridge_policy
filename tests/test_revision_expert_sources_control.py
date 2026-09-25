"""Expert-only training changes the old-plan source, not labels or conditioning."""
import json

import pytest
import torch

from action_bridge.configs.sb_pusht import get_config
from action_bridge.plan_revision.data import build_self_sources, immutable_snapshot
from test_revision_ablation_jobs import JOBS, launch  # noqa: F401
from test_revision_self_sources import ObservedOnlySampler, fixture


def test_control_changes_only_self_source_schedule_from_baseline():
    baseline = get_config("sb_ou") | json.loads((JOBS / "configs/baseline_seed0.json").read_text())
    control = get_config("sb_ou") | json.loads((JOBS / "configs/expert_sources_only.json").read_text())
    assert {key for key in baseline if baseline[key] != control[key]} == {"source_self_probabilities"}
    assert control["source_self_probabilities"] == [0.] * control["training_blocks"]


def test_control_submission_defaults_to_ou_with_reference_dependency(launch):
    result = launch.run("submit.sh", "expert_sources_only")
    assert result.returncode == 0, result.stderr
    calls = launch.rows("sbatch")
    assert len(calls) == 3
    assert calls[-1]["args"][-2:] == ["expert_sources_only", "sb_ou"]
    assert "--dependency=afterok:1002" in calls[-1]["args"]


@pytest.mark.parametrize("method", ["fm_paired", "fm_local_ot", "sb_kinetic"])
def test_control_accepts_other_revisers(launch, method):
    result = launch.run("submit.sh", "expert_sources_only", method)
    assert result.returncode == 0, result.stderr
    assert launch.rows("sbatch")[-1]["args"][-1] == method


@pytest.mark.parametrize("script", ["submit.sh", "train.sbatch"])
def test_control_rejects_ddim_before_launch(launch, script):
    result = launch.run(script, "expert_sources_only", "ddim")
    assert result.returncode != 0
    assert "no previous-plan source" in result.stderr
    assert not launch.rows()


def test_expert_source_old_chunk_is_aligned_with_recorded_history():
    config, windows, completion = fixture(episodes=1, replans=3)
    config.update(method="sb_ou", source_std=0., source_self_probabilities=[0.] * 4)
    snapshot = immutable_snapshot(ObservedOnlySampler())
    records, diagnostics = build_self_sources(
        windows, snapshot, completion, torch.ones(config["action_dim"]),
        config, "cpu", block=3, seed=73, modes=(0,))
    assert diagnostics["self_records"] == 0
    assert diagnostics["expert_records"] == 2
    torch.testing.assert_close(records["obs_hist"], windows["obs_hist"])
    torch.testing.assert_close(records["act_hist"], windows["act_hist"])
    torch.testing.assert_close(records["future_actions"], windows["future_actions"])
    torch.testing.assert_close(records["old_actions"][1:], windows["future_actions"][:-1])
    torch.testing.assert_close(records["source_actions"][1:, :2], windows["future_actions"][:-1, 2:])
    assert records["source_origin"].tolist() == [0, 1, 1]

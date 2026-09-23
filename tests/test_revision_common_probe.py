"""One history-aligned probe pool, frozen before all four revision queries."""

import json

import pytest
import torch
from torch import nn

from action_bridge.plan_revision import reporting
from action_bridge.plan_revision.completion import LearnedCompletion


class Reviser(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.register_buffer("shift", torch.tensor(float(shift)))
        self.calls = []

    def sample(self, obs_hist, act_hist, mode, *, source_actions, reference,
               generator, has_previous_plan):
        self.calls.append({"obs_hist": obs_hist.clone(), "act_hist": act_hist.clone(),
                           "source": source_actions.clone(), "available": has_previous_plan.clone()})
        return source_actions + self.shift, {}


def example():
    config = {"protocol": "self_source_v1", "horizon": 4, "execute": 2,
              "batch_size": 8, "source_std": .01, "robot_dt": 1., "prior_ridge": .05,
              "max_rate": 4., "mobility_smoothing": 0., "temperature": .05,
              "revision_gamma": 2.}
    obs = torch.arange(40, dtype=torch.float32).reshape(4, 2, 5) / 100
    records = {"obs_hist": obs, "act_hist": torch.ones(4, 2, 2),
               "episode_id": torch.tensor([0, 0, 1, 1]), "time_index": torch.tensor([0, 2, 0, 2]),
               "future_actions": torch.full((4, 4, 2), 1.5), "valid_mask": torch.ones(4, 4, dtype=torch.bool),
               "startup_actions": torch.ones(4, 4, 2)}
    completion_config = {"obs_dim": 5, "action_dim": 2, "obs_history": 2,
                         "action_history": 2, "horizon": 4, "hidden_dim": 8}
    completion = LearnedCompletion(**completion_config)
    dependencies = {"completion_config": completion_config, "completion_state": completion.state_dict(),
                    "innovation_variance": torch.ones(2)}
    policies = {method: Reviser(index / 4) for index, method in enumerate(reporting.REVISERS)}
    configs = {method: config | {"method": method} for method in reporting.REVISERS}
    return policies, {"validation": records, "test": records}, configs, dependencies


def test_pool_is_equal_frozen_history_aligned_and_shared(tmp_path):
    policies, windows, configs, dependencies = example()
    result = reporting.common_source_probe(policies, windows, configs, dependencies, "cpu", tmp_path, count=2)
    artifact = torch.load(tmp_path / "common_sources.pt", weights_only=False)
    pool = artifact["pools"]["test"]
    assert len(pool["source_actions"]) == 8
    assert pool["origin_model_id"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert pool["episode_id"].tolist() == [0, 1] * 4
    assert pool["time_index"].tolist() == [2, 2] * 4
    assert pool["has_previous_plan"].all() and (pool["source_origin"] == 2).all()
    assert not pool["source_actions"].requires_grad
    assert set(artifact["provenance"]["snapshot_sha256"]) == set(reporting.REVISERS)
    for policy in policies.values():
        # Replay used a distinct frozen copy, not the live diagnostic policy.
        assert len(policy.calls) == 1
        torch.testing.assert_close(policy.calls[0]["source"], pool["source_actions"])
        torch.testing.assert_close(policy.calls[0]["obs_hist"], pool["obs_hist"])
        torch.testing.assert_close(policy.calls[0]["act_hist"], pool["act_hist"])
    assert result["records"] == 8 and set(result["methods"]) == set(reporting.REVISERS)
    assert json.loads((tmp_path / "result.json").read_text()) == result
    original = (tmp_path / "common_sources.pt").read_bytes()
    assert reporting.common_source_probe(policies, windows, configs, dependencies, "cpu", tmp_path, count=2) == result
    assert (tmp_path / "common_sources.pt").read_bytes() == original
    policies["fm_paired"].shift.add_(1.)
    with pytest.raises(ValueError, match="frozen"):
        reporting.common_source_probe(policies, windows, configs, dependencies, "cpu", tmp_path, count=2)


def test_test_labels_do_not_select_threshold_or_change_self_sources(tmp_path):
    policies, windows, configs, dependencies = example()
    first = reporting.common_source_probe(policies, windows, configs, dependencies, "cpu", tmp_path / "first", count=2)
    windows["test"] = {**windows["test"], "future_actions": windows["test"]["future_actions"] + 100.}
    second = reporting.common_source_probe(policies, windows, configs, dependencies, "cpu", tmp_path / "second", count=2)
    assert first["threshold_from_validation_mse"] == second["threshold_from_validation_mse"]
    a = torch.load(tmp_path / "first/common_sources.pt", weights_only=False)["pools"]["test"]
    b = torch.load(tmp_path / "second/common_sources.pt", weights_only=False)["pools"]["test"]
    torch.testing.assert_close(a["source_actions"], b["source_actions"])
    assert first["methods"] != second["methods"]


def test_full_execution_has_no_old_plan_probe(tmp_path):
    config = {"protocol": "self_source_v1", "horizon": 4, "execute": 4}
    from action_bridge.plan_revision.reporting import REVISERS, common_source_probe
    result = common_source_probe(dict.fromkeys(REVISERS), {}, dict.fromkeys(REVISERS, config),
                                 {}, "cpu", tmp_path)
    assert result["status"] == "not_applicable"
    assert not (tmp_path / "common_sources.pt").exists()


def test_replacement_report_excludes_retired_results(tmp_path):
    for name, protocol, mode in (("legacy", "fixed_ddim", 2), ("ddim", "self_source_v1", 2),
                                 ("sb_kinetic-learned", "self_source_v1", 2),
                                 ("sb_kinetic-repeat", "self_source_v1", 0)):
        folder = tmp_path / "evaluation" / name
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps({"protocol": protocol,
            "training_seconds": 10., "cache_seconds": 2., "parameters": 100,
            "metrics": {"success_rate": .5, "completion_id": mode}}))
    rows = reporting.report(tmp_path)
    assert len(rows) == 3 and all(row["name"] != "legacy" for row in rows)
    report = (tmp_path / "REPORT.md").read_text()
    assert "Main comparison" in report and "Same kinetic checkpoint" in report
    assert "legacy" not in report

"""The learned-tail velocity choice follows training and source-cache identity."""
import json

import pytest
import torch

from action_bridge.configs.sb_pusht import get_config
from action_bridge.plan_revision import cache, reporting, training
from action_bridge.plan_revision.completion import COMPLETION_VELOCITY_WEIGHTINGS, complete_plan
from action_bridge.plan_revision.data import immutable_snapshot
from action_bridge.plan_revision.models import build_policy
from action_bridge.scripts import sb_pusht as driver
from test_revision_self_sources import fixture, replay
from test_revision_self_source_training import self_source_case
from test_revision_training import setup


@pytest.fixture(autouse=True)
def cpu_resources(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("weighting", COMPLETION_VELOCITY_WEIGHTINGS)
def test_cli_records_velocity_choice(tmp_path, capsys, weighting):
    assert get_config("sb_ou")["completion_velocity_weighting"] == "linear"
    assert driver.main(["sb_ou", "--run-root", str(tmp_path / "absent"),
                        "--completion-velocity-weighting", weighting, "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["completion_velocity_weighting"] == weighting
    assert not (tmp_path / "absent").exists()


def test_driver_rejects_unknown_choice_before_writing(tmp_path):
    with pytest.raises(ValueError, match="completion_velocity_weighting"):
        driver.run_stage("prepare", tmp_path / "absent", None,
                         get_config() | {"completion_velocity_weighting": "typo"}, "cpu")
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("weighting", COMPLETION_VELOCITY_WEIGHTINGS)
def test_replay_uses_selected_velocity_and_preserves_mixed_modes(weighting):
    config, windows, completion = fixture(episodes=2, replans=3)
    # Three retained targets are enough to distinguish a fit from the last pair.
    config.update(execute=1, source_std=0.)
    windows["time_index"] //= 2
    baseline, _ = replay(windows, config, completion, p_self=0.)
    config["completion_velocity_weighting"] = weighting
    records, diagnostics = replay(windows, config, completion, p_self=0.)
    ordinary = records["has_previous_plan"]
    expected = complete_plan(records["old_actions"][ordinary], config["execute"],
                             records["obs_hist"][ordinary], records["act_hist"][ordinary],
                             records["completion_id"][ordinary], completion,
                             velocity_weighting=weighting)
    torch.testing.assert_close(records["source_actions"][ordinary], expected)
    unchanged = records["completion_id"] != 2
    torch.testing.assert_close(records["source_actions"][unchanged], baseline["source_actions"][unchanged])
    assert diagnostics["completion_velocity_weighting"] == weighting
    learned = ordinary & (records["completion_id"] == 2)
    if weighting == "last_pair":
        torch.testing.assert_close(records["source_actions"], baseline["source_actions"])
    else:
        assert not torch.allclose(records["source_actions"][learned], baseline["source_actions"][learned])


def test_generic_record_sampling_uses_velocity_option():
    config, records, completion, _, _ = setup("fm_paired")
    batch = cache.draw_records(records, 30, "cpu", completion=completion, executed=1,
                               source_std=0., endpoint_std=0., velocity_weighting="exp_quarter")
    expected = complete_plan(batch["old_actions"], 1, batch["obs_hist"], batch["act_hist"],
                             batch["completion_id"], completion, velocity_weighting="exp_quarter")
    torch.testing.assert_close(batch["source_actions"], expected)


def test_coupling_refresh_forwards_velocity_option(monkeypatch):
    config, records, completion, _, _ = setup("sb_ou")
    config["completion_velocity_weighting"] = "exp_half"
    calls = []
    original = cache.draw_records

    def draw(*args, **kwargs):
        calls.append(kwargs["velocity_weighting"])
        return original(*args, **kwargs)

    monkeypatch.setattr(cache, "draw_records", draw)
    cache.refresh_coupling(records, 4, "cpu", config, completion, None, "reverse")
    assert calls == ["exp_half", "exp_half"]


def test_training_records_choice_and_rejects_changed_resume(tmp_path, monkeypatch):
    config, records, _, dependencies, metadata = setup("fm_paired")
    config["completion_velocity_weighting"] = "uniform"
    calls = []
    original = training.draw_records

    def draw(*args, **kwargs):
        calls.append(kwargs["velocity_weighting"])
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "draw_records", draw)
    state = training.train(records, config, tmp_path, metadata, dependencies, "cpu", stop_after=1)
    assert calls == ["uniform"]
    assert state["config"]["completion_velocity_weighting"] == "uniform"
    with pytest.raises(ValueError, match="provenance mismatch"):
        training.train(records, config | {"completion_velocity_weighting": "linear"},
                       tmp_path, metadata, dependencies, "cpu")


def test_source_cache_identity_rejects_changed_velocity(tmp_path):
    config, windows, dependencies, metadata = self_source_case("fm_paired")
    config["completion_velocity_weighting"] = "linear"
    completion = training.restore_completion(dependencies, "cpu")
    policy = immutable_snapshot(build_policy(config))
    _, _, _, diagnostics = training._block_sources(
        windows, policy, completion, dependencies, config, metadata, tmp_path, "cpu", 0)
    assert diagnostics["completion_velocity_weighting"] == "linear"
    with pytest.raises(ValueError, match="identity mismatch"):
        training._block_sources(windows, policy, completion, dependencies,
                                config | {"completion_velocity_weighting": "exp_half"},
                                metadata, tmp_path, "cpu", 0)


def test_evaluation_choice_and_missing_old_key_are_explicit():
    config = get_config("sb_ou")
    state = dict(config=config, metadata={}, dependencies={}, direction="forward")
    driver.validate_evaluation(state, {}, config)
    with pytest.raises(ValueError, match="completion_velocity_weighting"):
        driver.validate_evaluation(state, {}, config | {"completion_velocity_weighting": "uniform"})
    old = {key: value for key, value in config.items() if key != "completion_velocity_weighting"}
    driver.validate_evaluation(state | {"config": old}, {}, old | {"completion_velocity_weighting": "last_pair"})
    with pytest.raises(ValueError, match="completion_velocity_weighting"):
        driver.validate_evaluation(state | {"config": old}, {}, config)


def test_common_source_comparison_rejects_mixed_velocity_choices(tmp_path):
    configs = {method: get_config(method) for method in reporting.REVISERS}
    configs["sb_ou"]["completion_velocity_weighting"] = "exp_quarter"
    with pytest.raises(ValueError, match="same completion velocity weighting"):
        reporting.common_source_probe(dict.fromkeys(reporting.REVISERS), {}, configs,
                                      {}, "cpu", tmp_path / "absent")
    assert not (tmp_path / "absent").exists()

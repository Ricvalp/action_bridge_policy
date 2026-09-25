"""The direct-tail ablation changes completion, never the learned SB reference."""

import copy
import json

import pytest
import torch

from action_bridge.plan_revision.completion import (
    COMPLETION_NAMES, DirectTailPredictor, LearnedCompletion, complete_plan, direct_tail_records,
)
from action_bridge.plan_revision.training import direct_tail_config, fit_direct_tail, restore_completion


@pytest.fixture(autouse=True)
def cpu_resources(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def windows():
    generator = torch.Generator().manual_seed(17)
    return {
        "episode_id": torch.tensor([0, 0, 0, 1, 1, 1]),
        "time_index": torch.tensor([0, 2, 4, 0, 2, 4]),
        "obs_hist": torch.randn(6, 2, 3, generator=generator),
        "act_hist": torch.randn(6, 2, 2, generator=generator),
        "future_actions": torch.randn(6, 4, 2, generator=generator),
        "valid_mask": torch.ones(6, 4, dtype=torch.bool),
    }


def config():
    return dict(horizon=4, execute=2, obs_dim=3, action_dim=2, obs_history=2,
                action_history=2, direct_tail_hidden_dim=8, direct_tail_updates=4,
                seed=21, lr=.003, weight_decay=0., grad_clip=1., batch_size=2,
                log_every=2, checkpoint_every=2, validation_every=2)


def model_pair():
    reference = LearnedCompletion(3, 2, 2, 2, 4, hidden_dim=8)
    reference.direct_tail = DirectTailPredictor(**direct_tail_config(config()))
    return reference, reference.direct_tail


def test_completion_names_preserve_existing_ids():
    assert COMPLETION_NAMES == ("repeat", "fixed_damped", "learned_dissipative", "direct_mlp")


def test_adjacent_expert_records_never_cross_episodes_or_fill_startups():
    source = windows()
    source = {key: value[torch.tensor([5, 0, 2, 3, 1, 4])] for key, value in source.items()}
    records = direct_tail_records(source, 2)
    assert records["time_index"].tolist() == [4, 4, 2, 2]
    assert records["episode_id"].tolist() == [1, 0, 0, 1]
    for row, (episode, time) in enumerate(zip(records["episode_id"], records["time_index"])):
        old = ((source["episode_id"] == episode) & (source["time_index"] == time - 2)).nonzero()[0, 0]
        assert torch.equal(records["old_actions"][row], source["future_actions"][old])
    records["old_actions"].fill_(500)
    assert source["future_actions"].abs().max() < 500


@pytest.mark.parametrize("problem,match", [("duplicate", "duplicate"), ("no_previous", "adjacent"),
                                          ("partial", "complete valid"), ("missing", "fields")])
def test_bad_supervised_windows_rejected(problem, match):
    source = windows()
    if problem == "duplicate":
        source["time_index"][1] = 0
    elif problem == "no_previous":
        source["time_index"] *= 3
    elif problem == "partial":
        source["valid_mask"][0, -1] = False
    else:
        del source["time_index"]
    with pytest.raises(ValueError, match=match):
        direct_tail_records(source, 2)


@pytest.mark.parametrize("execute", [0, 4, 5])
def test_direct_tail_requires_retained_targets(execute):
    with pytest.raises(ValueError, match="execute"):
        DirectTailPredictor(4, execute, 3, 2, 2, 2)
    with pytest.raises(ValueError, match="execute"):
        direct_tail_records(windows(), execute)


def test_direct_prediction_is_unconstrained_and_retains_old_suffix_exactly():
    source = windows()
    reference, predictor = model_pair()
    with torch.no_grad():
        predictor.head[-1].weight.zero_()
        predictor.head[-1].bias.copy_(torch.tensor([20., -20., 50., -50.]))
    old = source["future_actions"].clone()
    result = complete_plan(old, 2, source["obs_hist"], source["act_hist"], 3, reference)
    assert torch.equal(result[:, :2], old[:, 2:])
    assert torch.equal(result[:, -2:], torch.tensor([[20., -20.], [50., -50.]]).expand(6, -1, -1))
    assert torch.equal(old, source["future_actions"])


def test_direct_inputs_exclude_executed_old_targets_and_future_labels():
    records = direct_tail_records(windows(), 2)
    _, model = model_pair()
    baseline = model(records["old_actions"], records["obs_hist"], records["act_hist"])
    records["old_actions"][:, :2] += 500
    records["future_actions"] += 1000
    changed = model(records["old_actions"], records["obs_hist"], records["act_hist"])
    assert torch.equal(changed, baseline)
    old = records["old_actions"].clone().requires_grad_()
    model(old, records["obs_hist"], records["act_hist"]).sum().backward()
    assert torch.equal(old.grad[:, :2], torch.zeros_like(old.grad[:, :2]))
    assert torch.count_nonzero(old.grad[:, 2:]) > 0


def test_direct_loss_supervises_only_appended_tail_and_all_parameters():
    records = direct_tail_records(windows(), 2)
    _, model = model_pair()
    before = model.loss(records)
    records["future_actions"][:, :2] += 1000
    assert torch.equal(before, model.loss(records))
    expected = (model(records["old_actions"], records["obs_hist"], records["act_hist"])
                - records["future_actions"][:, -2:]).square().mean()
    assert torch.equal(before, expected)
    before.backward()
    assert all(parameter.grad is not None and parameter.grad.isfinite().all() for parameter in model.parameters())
    records["valid_mask"][0, 1] = False
    with pytest.raises(ValueError, match="fully valid"):
        model.loss(records)


def test_mixed_modes_leave_all_existing_completion_and_reference_values_unchanged():
    source = windows()
    reference, _ = model_pair()
    old = source["future_actions"][:4]
    obs, history = source["obs_hist"][:4], source["act_hist"][:4]
    before = reference.prior(obs, history, torch.ones(2))
    mixed = complete_plan(old, 2, obs, history, torch.arange(4), reference)
    for mode in range(4):
        individual = complete_plan(old, 2, obs, history, mode, reference)
        assert torch.equal(individual[mode], mixed[mode])
    after = reference.prior(obs, history, torch.ones(2))
    assert all(torch.equal(before[key], after[key]) for key in before)
    with pytest.raises(ValueError, match="K used to train"):
        complete_plan(old, 1, obs, history, 3, reference)
    with pytest.raises(ValueError, match="fitted DirectTailPredictor"):
        complete_plan(old, 2, obs, history, 3)


def test_self_contained_restore_attaches_frozen_tail_without_changing_reference():
    reference, direct = model_pair()
    tail_state = copy.deepcopy(direct.state_dict())
    del reference.direct_tail
    dependencies = {
        "completion_config": dict(obs_dim=3, action_dim=2, obs_history=2, action_history=2,
                                  horizon=4, hidden_dim=8),
        "completion_state": copy.deepcopy(reference.state_dict()),
        "direct_tail_config": direct_tail_config(config()), "direct_tail_state": tail_state,
    }
    restored = restore_completion(dependencies, "cpu")
    assert not restored.training and not restored.direct_tail.training
    assert all(not parameter.requires_grad for parameter in restored.parameters())
    records = direct_tail_records(windows(), 2)
    actual = restored.direct_tail(records["old_actions"], records["obs_hist"], records["act_hist"])
    expected = direct(records["old_actions"], records["obs_hist"], records["act_hist"])
    assert torch.equal(actual, expected)
    assert not any(key.startswith("direct_tail") for key in dependencies["completion_state"])
    for key, value in reference.prior(records["obs_hist"], records["act_hist"], torch.ones(2)).items():
        assert torch.equal(value, restored.prior(records["obs_hist"], records["act_hist"], torch.ones(2))[key])


def test_separate_fit_logs_validation_and_resumes_exactly(tmp_path):
    settings, records = config(), windows()
    metadata = {"train_episodes": [0, 1], "validation_episodes": [10, 11]}
    validation = copy.deepcopy(records)
    validation["episode_id"] += 10
    whole = fit_direct_tail(records, validation, settings, tmp_path / "whole", metadata, "cpu")
    partial = fit_direct_tail(records, validation, settings, tmp_path / "resume", metadata, "cpu", stop_after=1)
    assert partial["step"] == 1 and not partial["complete"]
    torch.manual_seed(971)
    resumed = fit_direct_tail(records, validation, settings, tmp_path / "resume", metadata, "cpu")
    assert resumed["complete"] and resumed["step"] == 4
    for key in whole["model"]:
        assert torch.equal(whole["model"][key], resumed["model"][key])
    assert torch.equal(whole["rng"]["torch"], resumed["rng"]["torch"])
    assert whole["train_records"] == 4 and whole["validation_records"] == 4
    for phase in ("train", "validation"):
        rows = [json.loads(line) for line in (tmp_path / "whole" / f"{phase}.jsonl").read_text().splitlines()]
        assert [row["step"] for row in rows] == [2, 4]
        assert all(row["tail_mse"] >= 0 for row in rows)
    with pytest.raises(ValueError, match="mismatch"):
        fit_direct_tail(records, validation, {**settings, "direct_tail_updates": 8},
                        tmp_path / "whole", metadata, "cpu")


def test_validation_labels_do_not_change_fitted_parameters(tmp_path):
    settings, records = config(), windows()
    validation = copy.deepcopy(records)
    before = fit_direct_tail(records, validation, settings, tmp_path / "one", {}, "cpu")
    validation["future_actions"].add_(100)
    after = fit_direct_tail(records, validation, settings, tmp_path / "two", {}, "cpu")
    assert all(torch.equal(before["model"][key], after["model"][key]) for key in before["model"])

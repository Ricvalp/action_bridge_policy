"""Tiny end-to-end checks of scarcity, direct tails and reference ablations."""

import copy

import pytest
import torch

from action_bridge.configs.sb_pusht import get_config
from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.cache import reference_for
from action_bridge.plan_revision.completion import complete_plan
from action_bridge.plan_revision.data import build_self_sources
from action_bridge.plan_revision.models import build_policy
from action_bridge.plan_revision.training import restore_completion
from action_bridge.scripts import sb_pusht as driver


@pytest.fixture(autouse=True)
def cpu_resources(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def tiny_config(method="sb_ou", **overrides):
    settings = get_config(method)
    settings.update(horizon=4, execute=2, channels=[8, 16], history_dim=16,
                    hidden_dim=16, time_dim=8, mode_dim=4, batch_size=2,
                    num_inference_steps=2, updates=2, rounds=1, phase_updates=1,
                    training_blocks=1, source_self_probabilities=[.5],
                    coupling_records=4, coupling_steps=2, coupling_refresh_every=1,
                    reference_updates=2, reference_hidden_dim=8, direct_tail_updates=2,
                    direct_tail_hidden_dim=8, checkpoint_every=1, log_every=1,
                    validation_every=2)
    return settings | overrides


def window_records(episode_start=0):
    generator = torch.Generator().manual_seed(31 + episode_start)
    records = {
        "episode_id": torch.tensor([0, 0, 0, 1, 1, 1]) + episode_start,
        "time_index": torch.tensor([0, 2, 4, 0, 2, 4]),
        "obs_hist": .1 * torch.randn(6, 2, 5, generator=generator),
        "act_hist": .1 * torch.randn(6, 2, 2, generator=generator),
        "future_actions": .1 * torch.randn(6, 4, 2, generator=generator),
        "valid_mask": torch.ones(6, 4, dtype=torch.bool),
    }
    records["startup_actions"] = records["act_hist"][:, -1:, :].expand(-1, 4, -1).clone()
    return records


def cached_experiment(tmp_path, config):
    dataset = tmp_path / "immutable-test-data"
    dataset.write_bytes(b"unit-test dataset identity; records are explicit synthetic tensors")
    root = tmp_path / "run"
    records = {"train": window_records(0), "val": window_records(10), "test": window_records(20)}
    metadata = {"dataset_path": str(dataset.resolve()), "dataset_sha256": driver.dataset_digest(dataset),
                "normalizer_id": "test-only-normalizer", "splits": {
                    "train": [0, 1], "val": [10, 11], "test": [20, 21]}}
    checkpoints.save(root / "windows.pt", dict(metadata=metadata, records=records,
                                               data_spec=driver.window_spec(config)))
    return root, dataset, records, metadata


@pytest.mark.parametrize("method", ["fm_paired", "sb_ou", "sb_kinetic"])
def test_direct_tail_driver_fit_train_checkpoint_restore_and_replay(tmp_path, method):
    config = tiny_config(method, training_completion_modes=[3], completion_id=3)
    root, dataset, windows, metadata = cached_experiment(tmp_path, config)
    untouched = copy.deepcopy(windows)
    driver.run_stage("reference", root, dataset, config, "cpu", evaluation=False)
    reference_state = checkpoints.load(root / "reference" / "latest.pt")
    direct_state = checkpoints.load(root / "direct_tail" / "latest.pt")
    assert reference_state["complete"] and direct_state["complete"]
    assert direct_state["direct_tail_config"]["execute"] == config["execute"]
    assert not any(key.startswith("direct_tail") for key in reference_state["completion_state"])

    trained = driver.run_stage(method, root, dataset, config, "cpu", evaluation=False)
    assert trained["complete"] and trained["step"] == 2
    assert trained["metadata"] == metadata
    dependencies = trained["dependencies"]
    assert dependencies["direct_tail_sha256"] == checkpoints.digest(root / "direct_tail" / "latest.pt")
    assert checkpoints.content_digest(dependencies["completion_state"]) == checkpoints.content_digest(
        reference_state["completion_state"])
    assert checkpoints.content_digest(dependencies["direct_tail_state"]) == checkpoints.content_digest(
        direct_state["direct_tail_state"])
    policy = checkpoints.restore_policy(trained, "cpu").requires_grad_(False)
    field = policy.forward_field if method.startswith("sb_") else policy.field
    assert field.mode_embedding.num_embeddings == 4
    completion = restore_completion(dependencies, "cpu")
    replay, diagnostics = build_self_sources(
        windows["train"], policy, completion, dependencies["innovation_variance"], config,
        "cpu", block=0, seed=37, p_self=1., modes=[3])
    assert diagnostics["completion_modes"] == [3]
    assert diagnostics["self_records"] == 4 and diagnostics["startup_records"] == 2
    assert replay["generated_actions"].isfinite().all()
    assert torch.equal(replay["completion_id"], torch.full((6,), 3))
    batch = windows["train"]
    completed = complete_plan(batch["future_actions"], 2, batch["obs_hist"], batch["act_hist"],
                              3, completion)
    assert torch.equal(completed[:, :2], batch["future_actions"][:, 2:])
    prior = completion.prior(batch["obs_hist"], batch["act_hist"], dependencies["innovation_variance"])
    reference = reference_for({"precision": prior["precision"], "prior_mean": prior["mean"]}, config
                              ) if method.startswith("sb_") else None
    generated, _ = policy.sample(batch["obs_hist"], batch["act_hist"], 3,
                                  source_actions=completed, reference=reference,
                                  has_previous_plan=torch.ones(6, dtype=torch.bool))
    assert generated.shape == (6, 4, 2) and generated.isfinite().all()
    for split in windows:
        assert all(torch.equal(windows[split][key], untouched[split][key]) for key in windows[split])


@pytest.mark.parametrize("variant", ["brownian", "isotropic_ou"])
def test_reference_variant_driver_training_and_self_contained_restore(tmp_path, variant):
    config = tiny_config(reference_kind=variant, training_completion_modes=[2])
    root, dataset, windows, _ = cached_experiment(tmp_path, config)
    driver.run_stage("reference", root, dataset, config, "cpu", evaluation=False)
    trained = driver.run_stage("sb_ou", root, dataset, config, "cpu", evaluation=False)
    assert trained["complete"] and trained["config"]["reference_kind"] == variant
    assert not (root / "direct_tail").exists()
    assert "direct_tail_state" not in trained["dependencies"]
    policy = checkpoints.restore_policy(trained, "cpu").requires_grad_(False)
    completion = restore_completion(trained["dependencies"], "cpu")
    batch = windows["val"]
    prior = completion.prior(batch["obs_hist"], batch["act_hist"],
                             trained["dependencies"]["innovation_variance"])
    reference = reference_for({"precision": prior["precision"], "prior_mean": prior["mean"]}, config)
    if variant == "brownian":
        assert reference.kind == "brownian" and torch.count_nonzero(reference.precision) == 0
    else:
        expected_rate = prior["precision"].double().diagonal(dim1=-2, dim2=-1).mean(-1)
        assert reference.kind == "ou"
        torch.testing.assert_close(reference.precision, expected_rate[:, None, None] * torch.eye(8))
        assert torch.equal(reference.mean, prior["mean"].double())
    source = complete_plan(batch["future_actions"], 2, batch["obs_hist"], batch["act_hist"], 2, completion)
    generated, _ = policy.sample(batch["obs_hist"], batch["act_hist"], 2,
                                 source_actions=source, reference=reference)
    assert generated.isfinite().all()


@pytest.mark.parametrize("problem", ["foreign_dataset", "incomplete", "wrong_execute"])
def test_direct_tail_dependency_must_be_finished_and_match_current_data(tmp_path, problem):
    config = tiny_config(training_completion_modes=[3], completion_id=3)
    root, dataset, *_ = cached_experiment(tmp_path, config)
    driver.run_stage("reference", root, dataset, config, "cpu", evaluation=False)
    path = root / "direct_tail" / "latest.pt"
    state = checkpoints.load(path)
    if problem == "foreign_dataset":
        state["metadata"]["normalizer_id"] = "different-training-data"
    elif problem == "incomplete":
        state["complete"] = False
    else:
        state["direct_tail_config"]["execute"] = 1
    checkpoints.save(path, state)
    with pytest.raises(ValueError):
        driver.run_stage("sb_ou", root, dataset, config, "cpu", evaluation=False)
    assert not (root / "sb_ou" / "latest.pt").exists()


@pytest.mark.parametrize("method", ["fm_paired", "sb_ou", "sb_kinetic"])
def test_existing_embedding_shapes_stay_unchanged_and_new_mode_is_explicit(method):
    baseline = tiny_config(method)
    without_modes = {key: value for key, value in baseline.items() if key != "training_completion_modes"}
    torch.manual_seed(12)
    old = build_policy(without_modes)
    torch.manual_seed(12)
    current = build_policy(baseline)
    assert all(torch.equal(old.state_dict()[key], current.state_dict()[key]) for key in old.state_dict())
    fields = [current.forward_field, current.reverse_field] if method.startswith("sb_") else [current.field]
    assert all(field.mode_embedding.num_embeddings == 3 for field in fields)
    with pytest.raises(ValueError, match="completion_id"):
        build_policy(baseline | {"completion_id": 3})
    with pytest.raises(ValueError, match="training_completion_modes"):
        build_policy(baseline | {"training_completion_modes": [0, 2, 4]})
    with pytest.raises(ValueError, match="training_completion_modes"):
        build_policy(baseline | {"training_completion_modes": [2, 2]})


def test_scarcity_cache_keys_include_fraction_and_subset_but_preserve_full_data_spec(tmp_path):
    baseline = tiny_config()
    old_config = {key: value for key, value in baseline.items()
                  if key not in {"train_episode_fraction", "subset_seed"}}
    assert driver.window_spec(old_config) == driver.window_spec(baseline)
    assert driver.window_spec(baseline | {"subset_seed": 17}) == driver.window_spec(baseline)
    scarce = baseline | {"train_episode_fraction": .25, "subset_seed": 7}
    spec = driver.window_spec(scarce)
    assert spec["train_episode_fraction"] == .25 and spec["subset_seed"] == 7
    assert spec != driver.window_spec(scarce | {"subset_seed": 8})
    assert spec != driver.window_spec(scarce | {"train_episode_fraction": .5})
    root, dataset, *_ = cached_experiment(tmp_path, scarce)
    with pytest.raises(ValueError, match="Window cache"):
        driver.run_stage("reference", root, dataset, scarce | {"subset_seed": 8}, "cpu", evaluation=False)

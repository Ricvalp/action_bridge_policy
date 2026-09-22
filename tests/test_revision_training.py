"""Tiny optimizer/resume checks on tensors, not additional learning experiments."""

import builtins
import copy

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")

from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.cache import draw_records, refresh_coupling
from action_bridge.plan_revision.completion import LearnedCompletion
from action_bridge.plan_revision.training import train


@pytest.fixture(autouse=True)
def cpu_test_resources(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    # These are CPU interface tests; do not initialize CUDA while saving RNG.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def setup(method="sb_ou", *, horizon=4, action_dim=2):
    config = {
        "method": method, "horizon": horizon, "action_dim": action_dim,
        "obs_history": 2, "action_history": 2, "obs_dim": 3,
        "channels": [8, 16], "history_dim": 16, "hidden_dim": 16,
        "time_dim": 8, "mode_dim": 8, "num_inference_steps": 2,
        "time_cutoff": .05, "temperature": .05, "revision_gamma": 2.,
        "seed": 31, "batch_size": 2, "updates": 8, "rounds": 2,
        "phase_updates": 2, "coupling_records": 4, "coupling_steps": 2,
        "coupling_refresh_every": 2, "execute": 2, "robot_dt": 1.,
        "source_std": .01, "endpoint_std": .001,
        "lr": 1e-4, "weight_decay": 1e-6, "grad_clip": 1., "ema_decay": .9,
        "log_every": 2, "checkpoint_every": 1, "validation_every": 4,
    }
    generator = torch.Generator().manual_seed(44)
    batch = 7
    n = horizon * action_dim
    records = {
        "obs_hist": torch.randn(batch, 2, 3, generator=generator),
        "act_hist": torch.randn(batch, 2, action_dim, generator=generator),
        "old_actions": torch.randn(batch, horizon, action_dim, generator=generator),
        "future_actions": torch.randn(batch, horizon, action_dim, generator=generator),
        "valid_mask": torch.ones(batch, horizon, dtype=torch.bool),
        "precision": torch.eye(n).repeat(batch, 1, 1),
        "prior_mean": torch.zeros(batch, n),
        "episode_id": torch.arange(batch), "time_index": torch.arange(batch) + 20,
    }
    completion_config = dict(obs_dim=3, action_dim=action_dim, obs_history=2,
                             action_history=2, horizon=horizon, hidden_dim=8, robot_dt=1.)
    torch.manual_seed(19)
    completion = LearnedCompletion(**completion_config).eval().requires_grad_(False)
    dependencies = {"completion_config": completion_config,
                    "completion_state": copy.deepcopy(completion.state_dict()),
                    "innovation_variance": torch.ones(action_dim),
                    "normalization": {"kind": "explicit-unit-test-coordinates"}}
    metadata = {"source_hash": "unit-test-source", "reference_hash": "unit-test-reference",
                "train_episodes": list(range(batch)), "validation_episodes": [100]}
    return config, records, completion, dependencies, metadata


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("method", ["sb_ou", "sb_kinetic"])
def test_two_directional_rounds_and_exact_midphase_resume(tmp_path, method):
    config, records, _, dependencies, metadata = setup(method)
    validations = []

    def validation(model, step):
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())
        validations.append(step)
        # Validation may consume randomness; trainer must preserve its own RNG.
        torch.randn(3)
        return {"success_rate": step / config["updates"]}

    uninterrupted = train(records, config, tmp_path / "whole", metadata, dependencies, "cpu", validate=validation)
    assert validations == [4, 8]
    validations.clear()
    interrupted = train(records, config, tmp_path / "resumed", metadata, dependencies, "cpu",
                        validate=validation, stop_after=3)
    assert interrupted["step"] == 3 and interrupted["phase"] == 1
    assert interrupted["direction"] == "forward"
    assert not interrupted["complete"]
    assert interrupted["cache_provenance"]["generated_endpoint"] == "source"
    assert interrupted["opposite_snapshot"] is not None
    # Deliberately disturb all process RNG state before resuming.
    torch.manual_seed(912)
    np.random.seed(234)
    supplied_dependencies = copy.deepcopy(dependencies)
    for tensor in supplied_dependencies["completion_state"].values():
        tensor.add_(1)
    # A matching checkpoint is authoritative: never silently replace its frozen
    # reference just because a caller reconstructed a different current object.
    resumed = train(records, config, tmp_path / "resumed", metadata, supplied_dependencies, "cpu", validate=validation)
    assert validations == [4, 8]
    assert resumed["step"] == 8 and resumed["complete"]
    assert resumed["phase"] == 3 and resumed["outer_round"] == 1 and resumed["direction"] == "forward"
    for key in ("model", "ema", "optimizers", "coupling_cache", "opposite_snapshot", "rng"):
        assert_tree_equal(uninterrupted[key], resumed[key])
    assert_tree_equal(resumed["dependencies"], dependencies)
    for key in ("generated_endpoint", "phase", "refresh_step", "source_hash", "snapshot_phase"):
        assert uninterrupted["cache_provenance"][key] == resumed["cache_provenance"][key]
    assert resumed["cache_provenance"]["snapshot_phase"] == 2
    before = (tmp_path / "resumed" / "latest.pt").stat().st_mtime_ns
    train(records, config, tmp_path / "resumed", metadata, dependencies, "cpu", validate=validation)
    assert (tmp_path / "resumed" / "latest.pt").stat().st_mtime_ns == before


class TaggedSnapshot:
    """Deterministic fake dynamics: expose which endpoint was actually sampled."""

    def __init__(self):
        self.calls = []

    def rollout(self, state, obs_hist, act_hist, completion_id, reference, *, reverse, steps):
        assert not torch.is_grad_enabled()
        self.calls.append({"state": state.clone(), "obs_hist": obs_hist.clone(),
                           "act_hist": act_hist.clone(), "completion_id": completion_id.clone(),
                           "reverse": reverse, "steps": steps})
        return state + (100 if reverse else 200), {}


@pytest.mark.parametrize("method", ["sb_ou", "sb_kinetic"])
@pytest.mark.parametrize("direction,generated", [("forward", "source"), ("reverse", "target")])
def test_cache_refresh_keeps_real_endpoint_and_context_attached(method, direction, generated):
    config, records, completion, _, _ = setup(method)
    snapshot = TaggedSnapshot()
    cache, provenance = refresh_coupling(records, 5, "cpu", config, completion, snapshot, direction)
    n = config["horizon"] * config["action_dim"]
    state_dim = n * (2 if method == "sb_kinetic" else 1)
    assert cache["x0"].shape == cache["x1"].shape == (5, state_dim)
    assert provenance["generated_endpoint"] == generated
    start = torch.cat([call["state"] for call in snapshot.calls])
    expected_offset = 100 if direction == "forward" else 200
    real_key, generated_key = ("x1", "x0") if direction == "forward" else ("x0", "x1")
    torch.testing.assert_close(cache[real_key], start)
    torch.testing.assert_close(cache[generated_key], start + expected_offset)
    real_actions = cache["future_actions"] if direction == "forward" else cache["source_actions"]
    torch.testing.assert_close(cache[real_key][:, :n], real_actions.flatten(1))
    for key in ("obs_hist", "act_hist", "completion_id"):
        torch.testing.assert_close(cache[key], torch.cat([call[key] for call in snapshot.calls]))
    assert all(call["reverse"] == (direction == "forward") for call in snapshot.calls)
    assert all(call["steps"] == config["coupling_steps"] for call in snapshot.calls)
    for key in ("obs_hist", "act_hist", "episode_id", "time_index", "precision", "prior_mean"):
        torch.testing.assert_close(cache[key], records[key][cache["record_id"]])
    assert all(not tensor.requires_grad and tensor.grad_fn is None and tensor.device.type == "cpu"
               for tensor in cache.values())


def test_initial_reverse_phase_uses_natural_endpoints():
    config, records, completion, _, _ = setup("sb_kinetic")
    cache, provenance = refresh_coupling(records, 5, "cpu", config, completion, None, "reverse")
    n = config["horizon"] * config["action_dim"]
    torch.testing.assert_close(cache["x0"][:, :n], cache["source_actions"].flatten(1))
    torch.testing.assert_close(cache["x1"][:, :n], cache["future_actions"].flatten(1))
    assert not torch.equal(cache["x0"][:, n:], cache["x1"][:, n:])
    assert provenance["generated_endpoint"] == "none"


def test_source_completion_respects_configured_robot_index_dt():
    config, records, _, dependencies, _ = setup()
    completion = LearnedCompletion(**{**dependencies["completion_config"], "robot_dt": .2})
    torch.manual_seed(3)
    batch = draw_records(records, 16, "cpu", completion=completion, executed=config["execute"])
    assert (batch["completion_id"] == 2).any()
    assert torch.isfinite(batch["source_actions"]).all()


@pytest.mark.parametrize("method", ["ddim", "fm_paired"])
def test_baseline_training_checkpoint_restores_seeded_sampling_without_simulators(tmp_path, monkeypatch, method):
    config, records, _, dependencies, metadata = setup(method, horizon=9, action_dim=6)
    config["updates"] = 2
    if method == "ddim":
        records = {key: value for key, value in records.items() if key != "old_actions"}
        dependencies = {}
    payload = train(records, config, tmp_path, metadata, dependencies, "cpu")
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert not name.startswith(("gym_pusht", "pymunk", "gymnasium", "pygame")), name
        assert "pusht_adapter" not in name and "pusht_sim" not in name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    first = checkpoints.restore_policy(payload)
    second = checkpoints.restore_policy(checkpoints.load(tmp_path / "latest.pt"))
    kwargs = {} if method == "ddim" else {"source_actions": records["old_actions"][:2]}
    sampled = []
    for policy in (first, second):
        actions, _ = policy.sample(records["obs_hist"][:2], records["act_hist"][:2], torch.zeros(2, dtype=torch.long),
                                   generator=torch.Generator().manual_seed(80), **kwargs)
        sampled.append(actions)
    assert sampled[0].shape == (2, 9, 6)
    torch.testing.assert_close(*sampled, rtol=0, atol=0)
    if method == "ddim":
        assert payload["scheduler"]["num_train_timesteps"] == 100
        assert payload["scheduler"]["clip_sample"] is False
        assert payload["scheduler"]["prediction_type"] == "epsilon"


def test_mismatched_resume_provenance_fails(tmp_path):
    config, records, _, dependencies, metadata = setup("sb_ou")
    train(records, config, tmp_path, metadata, dependencies, "cpu", stop_after=1)
    changed = {**metadata, "source_hash": "another-source"}
    with pytest.raises(ValueError, match="provenance mismatch"):
        train(records, config, tmp_path, changed, dependencies, "cpu")

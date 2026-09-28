"""Tiny curriculum/resume wiring checks, not learning-performance benchmarks."""
import copy
import json

import pytest
import torch

from action_bridge.plan_revision import checkpoints, training
from action_bridge.plan_revision.cache import draw_records
from action_bridge.plan_revision.curriculum import CURRICULUM_PROTOCOL
from test_revision_training import setup, assert_tree_equal


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def case(method="sb_ou", *, updates=8):
    config, base, _, dependencies, metadata = setup(method, horizon=16, action_dim=3)
    config.update(protocol=CURRICULUM_PROTOCOL, condition_on_k=True,
                  model_schema="k_conditioned_v1", k_values_by_block=[1, 2, 4, 8],
                  startup_sampling_fraction=.1, execute=8, updates=updates,
                  training_blocks=4, rounds=4, phase_updates=updates // 8,
                  source_self_probabilities=[.1, .5, 1., 1.], source_seed=17,
                  prior_ridge=.05, max_rate=4., mobility_smoothing=0.,
                  completion_velocity_weighting="last_pair", training_completion_modes=[0, 1, 2],
                  coupling_refresh_every=2, log_every=2)
    # Dense contiguous adapter rows; three-action dimensions are not Push-T.
    indices = torch.arange(9) % len(base["future_actions"])
    records = {key: value[indices].clone() for key, value in base.items()
               if key not in {"old_actions", "prior_mean", "precision"}}
    records["episode_id"] = torch.zeros(9, dtype=torch.long)
    records["time_index"] = torch.arange(9)
    records["startup_actions"] = records["act_hist"][:, -1:].expand(-1, 16, -1).clone()
    return config, records, dependencies, metadata


@pytest.mark.parametrize("method", ["fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic"])
def test_four_stage_smoke_keeps_horizon_weights_and_reference(tmp_path, method):
    config, records, dependencies, metadata = case(method)
    original = copy.deepcopy(dependencies)
    kwargs = {"pairer_factory": lambda source: lambda batch, *_: (batch, {})} if method == "fm_local_ot" else {}
    state = training.train(records, config, tmp_path, metadata, dependencies, "cpu", **kwargs)
    assert state["complete"] and state["step"] == 8
    assert state["curriculum"]["active_k"] == state["curriculum"]["deployment_k"] == 8
    assert state["curriculum"]["final_k_reached"]
    assert not state["curriculum"]["pending_transition"]
    assert_tree_equal(state["dependencies"], original)
    snapshots, versions = [], []
    for block, (k, overlap) in enumerate(zip([1, 2, 4, 8], [15, 14, 12, 8])):
        folder = tmp_path / "sources" / f"block_{block:03d}"
        artifact = torch.load(folder / "sources.pt", weights_only=False)
        rows = artifact["records"]
        ordinary = rows["has_previous_plan"]
        assert rows["source_actions"].shape[1:] == (16, 3)
        assert bool((rows["execution_k"] == k).all())
        assert bool((rows["overlap_mask"][ordinary].sum(1) == overlap).all())
        assert bool((rows["elapsed_count"][ordinary] == k).all())
        assert bool((rows["elapsed_count"][~ordinary] == 0).all())
        assert artifact["provenance"]["active_k"] == k
        versions.append(artifact["provenance"]["version"])
        snapshots.append(checkpoints.load(folder / "snapshot.pt")["ema"])
    assert len(set(versions)) == 4
    assert [tuple((key, tuple(value.shape)) for key, value in s.items()) for s in snapshots].count(
        tuple((key, tuple(value.shape)) for key, value in snapshots[0].items())) == 4
    assert any(not torch.equal(snapshots[0][key], state["ema"][key]) for key in snapshots[0])
    rows = [json.loads(row) for row in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert [row["active_k"] for row in rows] == [1, 2, 4, 8]
    assert all(row["deployment_k"] == 8 for row in rows)
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in rows)
    assert all("source_error_tail" in row for row in rows)


@pytest.mark.parametrize("method", ["sb_ou", "sb_kinetic"])
def test_resume_mid_directions_and_across_boundary_is_exact(tmp_path, method):
    config, records, dependencies, metadata = case(method, updates=16)
    whole = training.train(records, config, tmp_path / "whole", metadata, dependencies, "cpu")
    # Reverse midpoint, forward midpoint, immediately on and after promotion.
    for stop in (1, 3, 4, 5):
        root = tmp_path / f"stop-{stop}"
        partial = training.train(records, config, root, metadata, dependencies, "cpu", stop_after=stop)
        assert not partial["complete"]
        assert partial["curriculum"]["pending_transition"] == (stop == 4)
        if stop == 3:
            (root / partial["source_provenance"]["cache_path"]).unlink()
        torch.manual_seed(876)
        resumed = training.train(records, config, root, metadata, dependencies, "cpu")
        for key in ("model", "ema", "optimizers", "coupling_cache", "opposite_snapshot", "rng", "curriculum"):
            assert_tree_equal(whole[key], resumed[key])


def test_resume_rejects_changed_schedule_and_stale_coupling_k(tmp_path):
    config, records, dependencies, metadata = case()
    partial = training.train(records, config, tmp_path, metadata, dependencies, "cpu", stop_after=1)
    changed = config | {"k_values_by_block": [2, 2, 4, 8]}
    with pytest.raises(ValueError, match="provenance mismatch"):
        training.train(records, changed, tmp_path, metadata, dependencies, "cpu")
    partial["cache_provenance"]["active_k"] = 8
    checkpoints.save(tmp_path / "latest.pt", partial)
    with pytest.raises(ValueError, match="stale K/source/phase"):
        training.train(records, config, tmp_path, metadata, dependencies, "cpu")


def test_startup_sampling_does_not_depend_on_record_fraction():
    for ordinary_count in (10, 1000):
        count = 1 + ordinary_count
        records = dict(future_actions=torch.zeros(count, 16, 3),
                       source_actions=torch.zeros(count, 16, 3),
                       completion_id=torch.zeros(count, dtype=torch.long),
                       has_previous_plan=torch.arange(count) != 0)
        torch.manual_seed(13)
        batch = draw_records(records, 10000, "cpu", startup_fraction=.1, endpoint_std=0.)
        assert .09 < float((~batch["has_previous_plan"]).float().mean()) < .11


def test_ddim_training_matches_the_original_deployment_grid(tmp_path):
    from action_bridge.plan_revision.curriculum import stage_windows
    config, dense, _, metadata = case("ddim")
    legacy = {key: value for key, value in config.items()
              if key not in {"condition_on_k", "model_schema", "k_values_by_block", "startup_sampling_fraction"}}
    legacy["protocol"] = "self_source_v1"
    windows = stage_windows(dense, config)
    before = training.train(windows, legacy, tmp_path / "fixed", metadata, {}, "cpu")
    after = training.train(dense, config, tmp_path / "curriculum", metadata, {}, "cpu")
    for key in ("model", "ema", "optimizers", "rng"):
        assert_tree_equal(before[key], after[key])
    assert after["source_provenance"] is None and after["curriculum"] is None


def test_boundary_evaluations_keep_deployment_k_while_training_k_changes(tmp_path):
    config, records, dependencies, metadata = case()
    config["validation_every"] = 3  # Deliberately not aligned with 2-update blocks.
    submitted = []

    class Evaluator:
        busy = False
        last_submitted_step = None

        def poll(self):
            pass

        def submit(self, payload):
            self.last_submitted_step = payload["step"]
            submitted.append((payload["step"], payload["config"]["execute"],
                              payload["curriculum"]["active_k"]))

        def finish(self):
            pass

        def close(self):
            pass

    state = training.train(records, config, tmp_path, metadata, dependencies, "cpu",
                           evaluation_factory=lambda callback: Evaluator())
    assert state["complete"]
    assert submitted == [(2, 8, 1), (3, 8, 2), (4, 8, 2), (6, 8, 4), (8, 8, 8)]


def test_empty_overlap_diagnostic_is_unavailable_not_zero_or_nan():
    class Policy:
        def sample(self, *args, source_actions, **kwargs):
            return source_actions + 1, {}

    records = dict(obs_hist=torch.zeros(2, 2, 3), act_hist=torch.zeros(2, 2, 3),
                   future_actions=torch.zeros(2, 4, 3), source_actions=torch.ones(2, 4, 3),
                   completion_id=torch.full((2,), 2), has_previous_plan=torch.ones(2, dtype=torch.bool),
                   execution_k=torch.full((2,), 4), overlap_mask=torch.zeros(2, 4, dtype=torch.bool))
    result = training._region_diagnostics(Policy(), records, {"method": "fm_paired"}, "cpu")
    assert result["overlap_available"] == 0
    assert "source_error_overlap" not in result and "revised_error_overlap" not in result
    assert result["source_error_tail"] == result["revision_magnitude_tail"] == 1
    assert result["revised_error_tail"] == 4

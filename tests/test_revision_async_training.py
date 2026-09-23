"""Asynchronous evaluation observes frozen policies, never drives optimization."""

import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")

from action_bridge.configs.sb_pusht import METHODS
from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.training import train
from test_revision_training import assert_tree_equal, setup


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def consume_randomness():
    random.random()
    np.random.rand(3)
    torch.randn(3)


def training_case(method):
    config, records, _, dependencies, metadata = setup(method)
    config["validation_every"] = 2
    if method == "ddim":
        records = {key: value for key, value in records.items() if key != "old_actions"}
        dependencies = {}
    pairer = (lambda batch, completion, device: (batch, {"ot_cost": .125})) if method == "fm_local_ot" else None
    return config, records, dependencies, metadata, pairer


class RecordingTracker:
    def __init__(self):
        self.evaluations = []

    def log(self, step, metrics):
        consume_randomness()

    def log_evaluation(self, step, metrics):
        consume_randomness()
        self.evaluations.append((step, dict(metrics)))


class DelayedEvaluation:
    """A deterministic worker double: results arrive after two later polls.

    Saving each snapshot immediately imitates serialization in the real manager.
    The next optimizer updates must not mutate the policy being evaluated.
    """

    def __init__(self, output, on_result, *, fail=False):
        consume_randomness()
        self.output = Path(output)
        self.on_result = on_result
        self.fail = fail
        self.job = None
        self.last_submitted_step = None
        self.submitted = []
        self.completed = []
        self.finish_calls = 0
        self.close_calls = 0

    @property
    def busy(self):
        return self.job is not None

    def submit(self, payload):
        consume_randomness()
        if self.busy:
            return False
        checkpoint = self.output / "fake_eval" / f"step_{payload['step']}.pt"
        checkpoints.save(checkpoint, payload)
        self.submitted.append((copy.deepcopy(payload), checkpoint))
        self.job = (payload["step"], checkpoint, 0)
        self.last_submitted_step = payload["step"]
        return True

    def poll(self):
        consume_randomness()
        if self.job is None:
            return
        if self.fail:
            raise RuntimeError("evaluation callback failed")
        step, checkpoint, polls = self.job
        self.job = (step, checkpoint, polls + 1)
        if polls + 1 >= 2:
            self._complete()

    def _complete(self):
        step, checkpoint, _ = self.job
        self.job = None
        consume_randomness()
        current_training_step = checkpoints.load(self.output / "latest.pt")["step"]
        self.completed.append((step, current_training_step))
        self.on_result(step, {"success_rate": {2: 1., 6: .75, 8: .25}.get(step, .5),
                              "mean_reward": .4}, checkpoint)

    def finish(self):
        consume_randomness()
        self.finish_calls += 1
        if self.job is not None:
            self._complete()

    def close(self):
        consume_randomness()
        self.close_calls += 1
        self.job = None


@pytest.mark.parametrize("method", METHODS)
def test_async_results_preserve_training_and_promote_the_evaluated_snapshot(tmp_path, method):
    config, records, dependencies, metadata, pairer = training_case(method)
    baseline = train(records, config, tmp_path / "plain", metadata, dependencies, "cpu", pairer=pairer)
    baseline_rng = checkpoints.rng_state()
    output = tmp_path / "async"
    managers = []

    def factory(on_result):
        manager = DelayedEvaluation(output, on_result)
        managers.append(manager)
        return manager

    tracker = RecordingTracker()
    result = train(records, config, output, metadata, dependencies, "cpu", pairer=pairer,
                   evaluation_factory=factory, tracker=tracker)
    for key in ("model", "ema", "optimizers", "coupling_cache", "opposite_snapshot", "rng"):
        assert_tree_equal(baseline[key], result[key])
    assert_tree_equal(baseline_rng, checkpoints.rng_state())
    manager, = managers
    assert [state["step"] for state, _ in manager.submitted] == [2, 6, 8]
    # Optimization advanced during the first evaluation; after step 8 it drains
    # the step-6 job, then submits and drains the final snapshot exactly once.
    assert manager.completed == [(2, 5), (6, 8), (8, 8)]
    assert manager.finish_calls == 2 and manager.close_calls == 1
    assert not manager.busy
    snapshots = {state["step"]: (state, path) for state, path in manager.submitted}
    for state, path in snapshots.values():
        assert "optimizers" not in state and "coupling_cache" not in state
        assert state["direction"] == "forward"
        assert_tree_equal(checkpoints.load(path)["ema"], state["ema"])
    bridge = method.startswith("sb_")
    assert [state["training_direction"] for state, _ in snapshots.values()] == (
        ["reverse", "reverse", "forward"] if bridge else ["forward"] * 3)
    assert [state["forward_updates"] for state, _ in snapshots.values()] == (
        [0, 2, 4] if bridge else [2, 6, 8])
    winner = 6 if bridge else 2
    best = checkpoints.load(output / "best.pt")
    assert best["step"] == winner
    assert_tree_equal(best["ema"], snapshots[winner][0]["ema"])
    assert checkpoints.digest(output / "best.pt") == checkpoints.digest(snapshots[winner][1])
    assert any(not torch.equal(best["ema"][key], result["ema"][key]) for key in best["ema"])
    rows = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows if row.get("status") == "skipped_busy"] == [4, 8]
    successful_rows = [row for row in rows if "success_rate" in row]
    assert [row["step"] for row in successful_rows] == [2, 6, 8]
    assert [step for step, _ in tracker.evaluations] == [2, 6, 8]
    for row, (step, metrics) in zip(successful_rows, tracker.evaluations, strict=True):
        assert row["step"] == step
        assert row["success_rate"] == metrics["success_rate"]
        assert row["checkpoint_sha256"] == checkpoints.digest(snapshots[step][1])
    report = json.loads((output / "best_eval.json").read_text())
    assert report["step"] == winner
    assert result["complete"] and result["best_validation"] == report["success_rate"]


@pytest.mark.parametrize("method", ["sb_ou", "sb_kinetic"])
def test_initial_reverse_results_are_logged_but_cannot_be_best(tmp_path, method):
    config, records, dependencies, metadata, pairer = training_case(method)
    manager = None

    def factory(on_result):
        nonlocal manager
        manager = DelayedEvaluation(tmp_path, on_result)
        return manager

    state = train(records, config, tmp_path, metadata, dependencies, "cpu", pairer=pairer,
                  evaluation_factory=factory, stop_after=2)
    assert not (tmp_path / "best.pt").exists()
    assert state["best_validation"] == float("-inf")
    row = json.loads((tmp_path / "validation.jsonl").read_text())
    assert row["step"] == 2 and row["forward_updates"] == 0 and row["success_rate"] == 1.
    assert manager.close_calls == 1 and manager.completed == [(2, 2)]


def test_evaluation_cadence_can_change_on_resume_but_learning_rate_cannot(tmp_path):
    config, records, dependencies, metadata, pairer = training_case("sb_ou")
    baseline = train(records, config, tmp_path / "plain", metadata, dependencies, "cpu")
    output = tmp_path / "resumed"
    train(records, config, output, metadata, dependencies, "cpu", stop_after=3)
    with pytest.raises(ValueError, match="provenance mismatch"):
        train(records, {**config, "lr": config["lr"] * 2}, output, metadata, dependencies, "cpu")
    changed = {**config, "validation_every": 3}
    managers = []

    def factory(on_result):
        manager = DelayedEvaluation(output, on_result)
        managers.append(manager)
        return manager

    resumed = train(records, changed, output, metadata, dependencies, "cpu", evaluation_factory=factory)
    for key in ("model", "ema", "optimizers", "coupling_cache", "opposite_snapshot", "rng"):
        assert_tree_equal(baseline[key], resumed[key])
    assert resumed["config"]["validation_every"] == 3
    assert [state["step"] for state, _ in managers[0].submitted] == [6, 8]
    # A completed stage does not start a new evaluator, even if cadence changes.
    train(records, config, output, metadata, dependencies, "cpu", evaluation_factory=factory)
    assert len(managers) == 1


@pytest.mark.parametrize("method", METHODS)
def test_runtime_seed_panel_change_resets_best_without_changing_training(tmp_path, method):
    config, records, dependencies, metadata, pairer = training_case(method)
    config["validation_every"] = 1
    original_inputs = copy.deepcopy((config, records, dependencies, metadata))
    baseline = train(records, config, tmp_path / "plain", metadata, dependencies, "cpu", pairer=pairer)
    baseline_rng = checkpoints.rng_state()
    output = tmp_path / "resumed"
    panel = list(range(5))
    managers = []

    def factory(on_result):
        seeds = list(panel)

        def with_panel(step, metrics, checkpoint):
            score = 1. if len(seeds) == 5 else {5: .6, 6: .7}.get(step, .5)
            on_result(step, {**metrics, "success_rate": score,
                             "seeds": seeds, "episodes": len(seeds)}, checkpoint)

        manager = DelayedEvaluation(output, with_panel)
        managers.append(manager)
        return manager

    def resume(**kwargs):
        return train(records, config, output, metadata, dependencies, "cpu", pairer=pairer,
                     evaluation_factory=factory, **kwargs)

    first = resume(stop_after=4)
    initial_report = json.loads((output / "best_eval.json").read_text())
    assert first["best_validation"] == initial_report["success_rate"] == 1.
    assert initial_report["step"] == (3 if method.startswith("sb_") else 1)
    assert initial_report["seeds"] == panel and initial_report["episodes"] == 5

    # The panel belongs to the runtime evaluator, not the training config.
    panel = list(range(20))
    changed = resume(stop_after=5)
    changed_report = json.loads((output / "best_eval.json").read_text())
    assert changed["best_validation"] == changed_report["success_rate"] == .6
    assert changed_report["step"] == checkpoints.load(output / "best.pt")["step"] == 5
    assert changed_report["seeds"] == panel and changed_report["episodes"] == 20

    # An interrupted write can leave the old panel's higher score in latest.pt.
    # The published report must remain authoritative on the following resume.
    changed["best_validation"] = 1.
    checkpoints.save(output / "latest.pt", changed)
    improved = resume(stop_after=6)
    assert improved["best_validation"] == .7
    report = json.loads((output / "best_eval.json").read_text())
    assert report["step"] == 6 and report["success_rate"] == .7
    best_digest = checkpoints.digest(output / "best.pt")

    resumed = resume()
    assert resumed["complete"] and resumed["best_validation"] == .7
    assert json.loads((output / "best_eval.json").read_text()) == report
    assert checkpoints.digest(output / "best.pt") == best_digest
    assert best_digest == checkpoints.digest(managers[2].submitted[0][1])
    rows = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    assert [(row["step"], row["success_rate"]) for row in rows if row.get("seeds") == panel] == [
        (5, .6), (6, .7), (7, .5), (8, .5)]
    for key in ("model", "ema", "optimizers", "coupling_cache", "opposite_snapshot", "rng",
                "config", "metadata", "dependencies", "source_block", "source_provenance"):
        assert_tree_equal(baseline[key], resumed[key])
    assert_tree_equal(baseline_rng, checkpoints.rng_state())
    assert_tree_equal(original_inputs, (config, records, dependencies, metadata))


def test_training_failure_closes_its_evaluation_manager(tmp_path):
    config, records, dependencies, metadata, _ = training_case("ddim")
    managers = []

    def factory(on_result):
        manager = DelayedEvaluation(tmp_path, on_result, fail=True)
        managers.append(manager)
        return manager

    with pytest.raises(RuntimeError, match="evaluation callback failed"):
        train(records, config, tmp_path, metadata, dependencies, "cpu", evaluation_factory=factory)
    assert managers[0].close_calls == 1 and not managers[0].busy
    assert checkpoints.load(tmp_path / "latest.pt")["step"] == 3


def test_sync_and_async_evaluation_cannot_both_own_best_checkpoint(tmp_path):
    config, records, dependencies, metadata, _ = training_case("ddim")
    with pytest.raises(ValueError, match="either synchronous validation or an asynchronous"):
        train(records, config, tmp_path, metadata, dependencies, "cpu",
              validate=lambda *args: {}, evaluation_factory=lambda *args: None)

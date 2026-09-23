"""Tracking is diagnostic: it must not alter any optimizer or sampling state."""

import copy
import math
from pathlib import Path
import random

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")

from action_bridge.configs.sb_pusht import METHODS
from action_bridge.plan_revision.training import fit_completion, track_images, train
from action_bridge.scripts import sb_pusht
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


class RecordingTracker:
    """Use no SDK or network, but stress RNG isolation like a diagnostic might."""

    def __init__(self, images_every=3):
        self.images_every = images_every
        self.rows = []
        self.figures = []

    def log(self, step, metrics):
        consume_randomness()
        self.rows.append((step, dict(metrics)))

    def images_due(self, step, final=False):
        return final or step % self.images_every == 0

    def images(self, step, paths):
        consume_randomness()
        self.figures.append((step, list(paths)))


def visualizer(calls):
    # The hook receives only the inference model and step, not a training batch
    # containing future expert actions. Task-specific held-out data is a closure.
    def visualize(model, step):
        assert not model.training
        assert not torch.is_grad_enabled()
        consume_randomness()
        calls.append((step, copy.deepcopy(model.state_dict())))
        return [Path(f"example-{step}.png")]
    return visualize


def validation(model, step):
    assert not model.training
    consume_randomness()
    return {"success_rate": .5, "mean_reward": .25, "description": "not a scalar"}


def training_case(method):
    config, records, _, dependencies, metadata = setup(method)
    config.update(updates=4, rounds=1)
    if method == "ddim":
        records = {key: value for key, value in records.items() if key != "old_actions"}
        dependencies = {}
    # Real OT behavior has separate tests. Here only exercise metric forwarding.
    pairer = (lambda batch, completion, device: (batch, {"ot_cost": .125})) if method == "fm_local_ot" else None
    return config, records, dependencies, metadata, pairer


@pytest.mark.parametrize("method", METHODS)
def test_every_policy_logs_metrics_and_images_without_changing_training(tmp_path, method):
    config, records, dependencies, metadata, pairer = training_case(method)
    baseline = train(records, config, tmp_path / "plain", metadata, dependencies, "cpu",
                     validate=validation, pairer=pairer)
    tracker, calls = RecordingTracker(), []
    logged = train(records, config, tmp_path / "logged", metadata, dependencies, "cpu",
                   validate=validation, pairer=pairer, tracker=tracker, visualize=visualizer(calls))
    for key in ("model", "ema", "optimizers", "coupling_cache", "opposite_snapshot", "rng"):
        assert_tree_equal(baseline[key], logged[key])
    assert [step for step, row in tracker.rows if "train/loss" in row] == [2, 4]
    values = {key: value for _, row in tracker.rows for key, value in row.items()}
    assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in values.values())
    loss_key = "noise_mse" if method == "ddim" else "control_mse" if method.startswith("sb_") else "velocity_mse"
    assert {f"train/{loss_key}", "train/loss", "train/grad_norm", "train/lr", "train/phase"} <= values.keys()
    assert values["val/success_rate"] == .5
    assert values["val/mean_reward"] == .25
    assert "val/description" not in values
    if method == "fm_local_ot":
        assert values["train/ot_cost"] == .125
    assert [step for step, _ in calls] == [3, 4]
    assert [step for step, _ in tracker.figures] == [3, 4]
    assert_tree_equal(calls[-1][1], logged["ema"])


@pytest.mark.parametrize("method", ["sb_ou", "sb_kinetic"])
def test_sb_always_plots_at_forward_phase_end_never_reverse(tmp_path, method):
    config, records, _, dependencies, metadata = setup(method)
    tracker, calls = RecordingTracker(images_every=99), []
    train(records, config, tmp_path, metadata, dependencies, "cpu",
          tracker=tracker, visualize=visualizer(calls))
    assert [step for step, _ in calls] == [4, 8]
    assert [step for step, _ in tracker.figures] == [4, 8]
    assert [row["train/reverse_phase"] for _, row in tracker.rows] == [1, 0, 1, 0]


@pytest.mark.parametrize("method", METHODS)
def test_logging_can_be_enabled_when_resuming_same_scientific_config(tmp_path, method):
    config, records, dependencies, metadata, pairer = training_case(method)
    baseline = train(records, config, tmp_path / "whole", metadata, dependencies, "cpu", pairer=pairer)
    partial = train(records, config, tmp_path / "resumed", metadata, dependencies, "cpu",
                    pairer=pairer, stop_after=2)
    assert not partial["complete"]
    tracker, calls = RecordingTracker(), []
    resumed = train(records, config, tmp_path / "resumed", metadata, dependencies, "cpu",
                    pairer=pairer, tracker=tracker, visualize=visualizer(calls))
    for key in ("model", "ema", "optimizers", "coupling_cache", "opposite_snapshot", "rng"):
        assert_tree_equal(baseline[key], resumed[key])
    assert resumed["config"] == config
    assert [step for step, row in tracker.rows if "train/loss" in row] == [4]
    assert [step for step, _ in calls] == [3, 4]
    # Re-running a finished stage should not create empty extra W&B runs.
    empty = RecordingTracker()
    train(records, config, tmp_path / "resumed", metadata, dependencies, "cpu",
          pairer=pairer, tracker=empty, visualize=visualizer([]))
    assert empty.rows == empty.figures == []


def test_reference_logs_train_validation_and_images_without_changing_fit(tmp_path):
    config, records, _, _, metadata = setup()
    records = {key: value for key, value in records.items() if key != "old_actions"}
    config.update(reference_updates=4, reference_hidden_dim=8, innovation_floor=1e-3,
                  prior_ridge=.05, max_rate=4.)
    baseline = fit_completion(records, records, config, tmp_path / "plain", metadata, "cpu")
    tracker, calls = RecordingTracker(), []
    logged = fit_completion(records, records, config, tmp_path / "logged", metadata, "cpu",
                            tracker=tracker, visualize=visualizer(calls))
    for key in ("model", "optimizer", "completion_state", "innovation_variance", "diagnostics", "rng"):
        assert_tree_equal(baseline[key], logged[key])
    assert [step for step, row in tracker.rows if "train/reference_mse" in row] == [2, 4]
    values = {key: value for _, row in tracker.rows for key, value in row.items()}
    assert {"train/grad_norm", "train/lr", "val/reference_mse", "val/rollout_mse"} <= values.keys()
    assert [step for step, _ in calls] == [3, 4]
    assert [step for step, _ in tracker.figures] == [3, 4]


def test_failed_preview_restores_mixed_module_modes_and_rng():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout())
    model[1].eval()
    rng = torch.get_rng_state().clone()

    def failed_preview(model, step):
        assert not model.training and not model[0].training
        consume_randomness()
        raise RuntimeError("plot failed")

    with pytest.raises(RuntimeError, match="plot failed"):
        track_images(RecordingTracker(), failed_preview, model, 3)
    assert model.training and model[0].training and not model[1].training
    assert torch.equal(torch.get_rng_state(), rng)


def test_cli_forwards_tracking_to_every_stage_without_changing_model_config(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sb_pusht, "run_stage", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert sb_pusht.main(["all", "--run-root", str(tmp_path), "--device", "cpu", "--wandb",
                          "--wandb-project", "research", "--wandb-entity", "lab",
                          "--wandb-mode", "offline", "--wandb-images-every", "123",
                          "--wandb-image-count", "2"]) == 0
    assert [args[0] for args, _ in calls] == ["prepare", "reference", "ddim", "sources", *METHODS[1:], "evaluate", "report"]
    for args, kwargs in calls:
        assert args[3]["validation_every"] == 10000
        assert kwargs["evaluation"] == {"device": "cpu", "threads": 2, "save_videos": True}
        options = kwargs["tracking"]
        assert options.enabled and options.project == "research" and options.entity == "lab"
        assert options.mode == "offline" and options.images_every == 123 and options.image_count == 2
        assert not any("wandb" in key or "tracking" in key for key in args[3])


def test_cli_can_configure_or_disable_background_evaluation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sb_pusht, "run_stage", lambda *args, **kwargs: calls.append((args, kwargs)))
    sb_pusht.main(["ddim", "--run-root", str(tmp_path), "--device", "cpu", "--eval-every", "5000",
                   "--eval-threads", "3", "--no-eval-videos"])
    args, kwargs = calls[-1]
    assert args[3]["validation_every"] == 5000
    assert kwargs["evaluation"] == {"device": "cpu", "threads": 3, "save_videos": False}
    sb_pusht.main(["ddim", "--run-root", str(tmp_path), "--device", "cpu", "--no-sim-eval"])
    assert calls[-1][1]["evaluation"] is False


def test_cli_tracking_defaults_off_and_allows_disabling_images(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sb_pusht, "run_stage", lambda *args, **kwargs: calls.append(kwargs["tracking"]))
    sb_pusht.main(["ddim", "--run-root", str(tmp_path), "--device", "cpu"])
    assert not calls[-1].enabled
    assert calls[-1].project == "action-bridge-policy"
    sb_pusht.main(["ddim", "--run-root", str(tmp_path), "--device", "cpu", "--wandb", "--wandb-image-count", "0"])
    assert calls[-1].enabled and calls[-1].image_count == 0


@pytest.mark.parametrize("arguments", [["--wandb-images-every", "0"], ["--wandb-image-count", "-1"],
                                      ["--wandb-mode", "typo"]])
def test_cli_rejects_invalid_tracking_options_before_starting(tmp_path, monkeypatch, arguments):
    def unexpected_stage(*args, **kwargs):
        pytest.fail("Invalid tracking options must not start a stage")
    monkeypatch.setattr(sb_pusht, "run_stage", unexpected_stage)
    with pytest.raises(SystemExit) as error:
        sb_pusht.main(["ddim", "--run-root", str(tmp_path), "--device", "cpu", *arguments])
    assert error.value.code == 2

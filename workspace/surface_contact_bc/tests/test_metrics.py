from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from workspace.surface_contact_bc.data import Normalizer
from workspace.surface_contact_bc.env import rollout_expert, sample_context
from workspace.surface_contact_bc.evaluation import (
    aggregate_metrics, episode_metrics, evaluate_policy, rollout,
)
from workspace.surface_contact_bc.plotting import (
    plot_comparison, plot_episode, plot_reference_overlay, save_rollout_gif,
)


def test_expert_task_and_contact_metrics_are_separate():
    context = sample_context(0)
    trace = rollout_expert(context)
    metrics = episode_metrics(trace, context)
    assert metrics["task_success"]
    assert metrics["contact_success"]
    assert metrics["clean_success"]
    assert 0 < metrics["max_penetration"] < 0.02
    assert 0 < metrics["peak_normal_force"] < 30
    assert metrics["contact_loss_count"] == 0  # Initial free approach is not a loss.
    assert metrics["recovery_time"] is None
    assert metrics["post_impulse_tangential_progress_loss"] is None
    json.dumps(metrics, allow_nan=False)


def test_recovery_requires_sustained_contact_and_is_not_zero_for_failure():
    context = sample_context(1)
    baseline = rollout_expert(context)
    trace = {"observations": baseline["observations"].copy(), "actions": baseline["actions"],
             "contact": baseline["infos"]["contact"].copy(), "impulse": np.zeros(150)}
    trace["impulse"][75] = 0.2
    trace["contact"][76:81] = False
    trace["observations"][76:81, 7] = 0.03
    trace["observations"][76:81, 2:4] = context.normal * 0.2
    metrics = episode_metrics(trace, context, counterfactual=baseline)
    assert metrics["recovery_eligible"]
    assert metrics["recovery_success"]
    assert metrics["recovery_time"] == pytest.approx(0.24, abs=1e-5)
    assert metrics["post_impulse_tangential_progress_loss"] == pytest.approx(0)
    trace["contact"][76:] = False
    failed = episode_metrics(trace, context)
    assert not failed["recovery_success"]
    assert failed["recovery_time"] is None
    summary = aggregate_metrics([metrics, failed])
    assert summary["recovery_success_rate"] == 0.5
    assert summary["recovery_time_count"] == 1


def test_impulse_progress_loss_is_paired_and_goal_direction_aware():
    context = sample_context(2)
    baseline = rollout_expert(context)
    trace = {"observations": baseline["observations"].copy(), "actions": baseline["actions"],
             "impulse": np.zeros(150)}
    trace["impulse"][75] = -0.2
    direction = np.sign(context.goal - context.initial_position @ context.tangent)
    trace["observations"][76:, :2] -= np.linspace(0, 0.02, 75)[:, None] * direction * context.tangent
    metrics = episode_metrics(trace, context, counterfactual=baseline)
    assert metrics["post_impulse_tangential_progress_loss"] == pytest.approx(0.02, abs=1e-6)


class HoldTarget(torch.nn.Module):
    obs_history = 2
    action_history = 2
    chunk_horizon = 8

    def __init__(self):
        super().__init__()
        self.histories = []

    def generate(self, obs_hist, act_hist, *, generator, diagnostics):
        self.histories.append((obs_hist.clone(), act_hist.clone()))
        return act_hist[:, -1:, :].repeat(1, self.chunk_horizon, 1), {}


def test_real_simulator_observations_and_executed_actions_reenter_history():
    context = replace(sample_context(4), duration=0.16)
    normalizer = Normalizer(np.zeros(11), np.ones(11), np.zeros(2), 1.0)
    policy = HoldTarget()
    trace = rollout(policy, normalizer, context, n_exec=2)
    assert len(policy.histories) == 2
    np.testing.assert_allclose(policy.histories[0][1][0], np.tile(context.initial_position, (2, 1)), atol=1e-7)
    np.testing.assert_allclose(policy.histories[1][0][0, -1], trace["observations"][2], atol=1e-7)
    np.testing.assert_allclose(policy.histories[1][1][0, -1], trace["actions"][1], atol=1e-7)
    assert trace["replan"].tolist() == [True, False, True, False]
    assert not np.allclose(trace["observations"][2, :2], trace["actions"][1], atol=1e-7)
    metrics = episode_metrics(trace, context)
    assert not metrics["contact_success"]
    assert metrics["contact_loss_count"] == 0


def test_evaluation_writes_reloadable_trace_and_json(tmp_path):
    context = replace(sample_context(5), duration=0.16)
    normalizer = Normalizer(np.zeros(11), np.ones(11), np.zeros(2), 1.0)
    summary = evaluate_policy(HoldTarget(), normalizer, [context], tmp_path, plots=False)
    assert summary["num_episodes"] == 1
    assert "episode" not in summary and "context_seed" not in summary
    assert summary["recovery_success_rate"] is None
    assert summary["recovery_success_rate_count"] == 0
    assert json.loads((tmp_path / "summary.json").read_text())["contexts"][0]["seed"] == 5
    with np.load(tmp_path / "traces/episode_000.npz", allow_pickle=False) as archive:
        assert archive["observations"].shape == (5, 11)
    with pytest.raises(ValueError, match="n_exec"):
        rollout(HoldTarget(), normalizer, context, n_exec=9)


def test_substep_penetration_changes_clean_success():
    context = sample_context(0)
    trace = rollout_expert(context)
    assert episode_metrics(trace, context)["clean_success"]
    trace["peak_penetration"] = np.full(len(trace["observations"]), 0.03)
    metrics = episode_metrics(trace, context)
    assert metrics["task_success"]
    assert not metrics["clean_success"]
    assert metrics["max_penetration"] == pytest.approx(0.03)


def test_evaluation_creates_reference_only_overlay_without_mutating_policy(tmp_path):
    context = replace(sample_context(7), duration=0.16)
    normalizer = Normalizer(np.zeros(11), np.ones(11), np.zeros(2), 1.0)
    policy = HoldTarget()
    policy.reference_only = False
    evaluate_policy(policy, normalizer, [context], tmp_path, plots=True)
    assert not policy.reference_only
    assert (tmp_path / "reference_only_overlay.png").is_file()
    assert (tmp_path / "traces/episode_000_reference_only.npz").is_file()


def test_reference_diagnostics_and_all_figures(tmp_path):
    context = replace(sample_context(3), duration=0.24)
    trace = rollout_expert(context)
    steps = len(trace["actions"])
    for name in ("gamma_normal", "gamma_tangent", "stiffness", "desired_offset", "ref_normal",
                 "ref_tangent", "residual_normal", "residual_tangent", "control_normal", "control_tangent",
                 "control_ref_ratio", "path_kl"):
        trace[name] = np.zeros(steps)
    metrics = episode_metrics(trace, context)
    assert metrics["path_kl_executed_sum"] == 0
    assert metrics["gamma_normal_minus_tangent_mean"] == 0
    plot_episode(trace, context, tmp_path)
    plot_reference_overlay(trace, trace, tmp_path / "reference_only_overlay.png")
    save_rollout_gif(trace, context, tmp_path / "rollout.gif", fps=5)
    rows = []
    for seed in (0, 1):
        for suite, value in (("id", 0), ("normal_impulse_sweep", -0.2),
                             ("normal_impulse_sweep", 0.2), ("unseen_orientation", 45)):
            rows.append(dict(method="test", train_episodes=4, seed=seed, suite=suite, value=value,
                             task_success_rate=1.0, clean_success_rate=0.5, peak_normal_force=3.0,
                             rms_normal_force=2.0, recovery_time=0.2, recovery_success_rate=0.5))
    plot_comparison(rows, tmp_path)
    for filename in ("rollout.png", "learned_reference.png", "reference_residual.png", "rollout.gif",
                     "reference_only_overlay.png", "task_force_pareto.png", "impulse_success.png",
                     "impulse_recovery.png", "low_data.png", "unseen_orientation.png"):
        assert (tmp_path / filename).stat().st_size > 100

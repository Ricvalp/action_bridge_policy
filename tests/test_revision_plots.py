"""Training previews show command targets, not teacher-forced predictions."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from action_bridge.eval import revision_pusht_plots as plots


@pytest.fixture(autouse=True)
def cpu_test_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def examples():
    config = {"method": "ddim", "seed": 7, "execute": 2, "completion_id": 2,
              "source_std": .01, "robot_dt": 1., "temperature": .05, "revision_gamma": 2.}
    records = {
        "obs_hist": torch.tensor([1., 2., 3., 4., .5]).repeat(8, 2, 1),
        "act_hist": torch.tensor([[0., 0.], [1., 2.]]).repeat(8, 1, 1),
        "future_actions": torch.zeros(8, 4, 2),
        "episode_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        "time_index": torch.tensor([10, 10, 11, 11, 20, 20, 21, 21]),
    }
    metadata = {
        "codec": {"mean": [100., 200.], "std": [10., 20.], "low": [0., 0.],
                  "high": [512., 512.], "units": "pixels"},
        "normalization": {"obs_mean": [50., 60., 200., 220., 0.], "obs_std": [10., 20., 10., 10., 1.]},
    }
    return config, records, metadata


class FakePolicy(nn.Module):
    def __init__(self, *, stochastic=False):
        super().__init__()
        self.child = nn.Dropout()
        self.calls = []
        self.stochastic = stochastic

    def sample(self, obs_hist, act_hist, mode, *, source_actions, reference, generator):
        assert not self.training
        self.calls.append((obs_hist.clone(), act_hist.clone(), mode, source_actions, reference))
        shape = (len(obs_hist), 4, 2)
        if self.stochastic:
            return torch.randn(shape, generator=generator), {}
        return torch.full(shape, 2.), {}


class FakeReference(nn.Module):
    horizon = 4
    action_dim = 2
    robot_dt = 1.

    def coefficients(self, obs_hist, act_hist):
        assert not self.training
        zeros = act_hist.new_zeros(len(act_hist), self.horizon, self.action_dim)
        return zeros, zeros, zeros  # Constant observed command velocity.


def capture_plots(monkeypatch):
    seen = []

    def capture(path, state, expert, predicted, **kwargs):
        seen.append({"path": path, "state": state.copy(), "expert": expert.copy(),
                     "predicted": predicted.copy(), **kwargs})

    monkeypatch.setattr(plots, "_plot_chunk", capture)
    return seen


def test_fixed_histories_deduplicate_proposals():
    _, records, _ = examples()
    assert plots._example_indices(records, 3).tolist() == [0, 4, 6]
    assert plots._example_indices(records, 20).tolist() == [0, 2, 4, 6]


def test_scene_and_actions_decode_once_without_future_conditioning(tmp_path, monkeypatch):
    config, records, metadata = examples()
    seen = capture_plots(monkeypatch)
    policy = FakePolicy()
    policy.child.eval()  # Preserve mixed train/eval states, too.
    plot = plots.make_action_chunk_plotter(records, config, metadata, tmp_path, "cpu", count=2)
    paths = plot(policy, 5000)
    assert policy.training and not policy.child.training
    assert len(paths) == 2 and paths[0].name == "step_00005000_example_00.png"
    np.testing.assert_allclose(seen[0]["state"], [60., 100., 230., 260., .5])
    np.testing.assert_allclose(seen[0]["expert"], [[100., 200.]] * 4)
    np.testing.assert_allclose(seen[0]["predicted"], [[120., 240.]] * 4)
    assert policy.calls[0][3] is None and policy.calls[0][4] is None

    different_gt = {**records, "future_actions": torch.full_like(records["future_actions"], 900.)}
    second = plots.make_action_chunk_plotter(different_gt, config, metadata, tmp_path, "cpu", count=2)
    second(policy, 5001)
    np.testing.assert_array_equal(seen[0]["predicted"], seen[2]["predicted"])
    assert not np.array_equal(seen[0]["expert"], seen[2]["expert"])


def test_rendered_png_and_global_random_streams(tmp_path):
    config, records, metadata = examples()
    plot = plots.make_action_chunk_plotter(records, config, metadata, tmp_path, "cpu", count=1)
    torch_state = torch.get_rng_state().clone()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    first = plot(FakePolicy(stochastic=True), 1)[0]
    second = plot(FakePolicy(stochastic=True), 1)[0]
    assert first == second
    assert torch.equal(torch_state, torch.get_rng_state())
    assert random.getstate() == python_state
    assert np.random.get_state()[0] == numpy_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    with Image.open(first) as image:
        assert image.format == "PNG" and image.size == (720, 720)


def test_sampling_noise_is_identical_across_logging_steps(tmp_path, monkeypatch):
    config, records, metadata = examples()
    seen = capture_plots(monkeypatch)
    plot = plots.make_action_chunk_plotter(records, config, metadata, tmp_path, "cpu", count=1)
    policy = FakePolicy(stochastic=True)
    plot(policy, 1)
    plot(policy, 100)
    np.testing.assert_array_equal(seen[0]["predicted"], seen[1]["predicted"])


def test_reference_uses_autonomous_history_dynamics(tmp_path, monkeypatch):
    config, records, metadata = examples()
    seen = capture_plots(monkeypatch)
    policy = FakeReference()
    plot = plots.make_action_chunk_plotter(records, config, metadata, tmp_path, "cpu", count=1, reference=True)
    plot(policy, 10)
    # Last command is [1,2], velocity [1,2]; no future label enters the rollout.
    expected = [[120., 280.], [130., 320.], [140., 360.], [150., 400.]]
    np.testing.assert_allclose(seen[0]["predicted"], expected)
    assert policy.training


@pytest.mark.parametrize("method", ["fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic"])
def test_revisers_use_completed_cached_plan_and_frozen_reference(tmp_path, monkeypatch, method):
    config, records, metadata = examples()
    config = {**config, "method": method, "source_std": 0.}
    records["old_actions"] = torch.tensor([[1., 2.], [2., 4.], [3., 6.], [4., 8.]]).repeat(8, 1, 1)
    records["precision"] = torch.eye(8).repeat(8, 1, 1)
    records["prior_mean"] = torch.zeros(8, 8)
    completion = FakeReference().eval()
    dependencies = {"example": "frozen reference weights"}
    restored = []

    def restore(supplied, device):
        assert supplied is dependencies and device == "cpu"
        restored.append(True)
        torch.rand(4)  # Simulate module initialization; must not shift RNG.
        return completion

    monkeypatch.setattr(plots, "restore_completion", restore)
    seen = capture_plots(monkeypatch)
    rng = torch.get_rng_state().clone()
    plot = plots.make_action_chunk_plotter(records, config, metadata, tmp_path, "cpu",
                                           dependencies=dependencies, count=1)
    assert torch.equal(rng, torch.get_rng_state())
    policy = FakePolicy()
    plot(policy, 4)
    assert restored == [True]
    torch.testing.assert_close(policy.calls[0][3][0], torch.tensor([[3., 6.], [4., 8.], [5., 10.], [6., 12.]]))
    gaussian = policy.calls[0][4]
    assert (gaussian is not None) == method.startswith("sb_")
    if gaussian is not None:
        assert gaussian.kind == ("kinetic" if method == "sb_kinetic" else "ou")
    np.testing.assert_allclose(seen[0]["source"], [[130., 320.], [140., 360.], [150., 400.], [160., 440.]])


def test_no_images_or_invalid_count(tmp_path):
    config, records, metadata = examples()
    plot = plots.make_action_chunk_plotter(records, config, metadata, tmp_path, "cpu", count=0)
    assert plot(FakePolicy(), 1) == []
    with pytest.raises(ValueError, match="negative"):
        plots.make_action_chunk_plotter(records, config, metadata, tmp_path, "cpu", count=-1)


@pytest.mark.parametrize("method", ["ddim", "fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic"])
def test_real_policy_inference_smoke(tmp_path, monkeypatch, method):
    pytest.importorskip("diffusers")
    from action_bridge.configs.sb_pusht import get_config
    from action_bridge.plan_revision.completion import LearnedCompletion
    from action_bridge.plan_revision.models import build_policy

    _, records, metadata = examples()
    config = get_config(method) | {"horizon": 4, "execute": 2, "channels": [8, 16],
                                  "history_dim": 16, "hidden_dim": 16, "time_dim": 8,
                                  "num_inference_steps": 2}
    dependencies = None
    if method != "ddim":
        records["old_actions"] = torch.zeros(8, 4, 2)
        records["precision"] = torch.eye(8).repeat(8, 1, 1)
        records["prior_mean"] = torch.zeros(8, 8)
        completion_config = {"obs_dim": 5, "action_dim": 2, "obs_history": 2,
                             "action_history": 2, "horizon": 4, "hidden_dim": 8}
        completion = LearnedCompletion(**completion_config)
        dependencies = {"completion_config": completion_config, "completion_state": completion.state_dict()}
    policy = build_policy(config)
    seen = capture_plots(monkeypatch)
    before = torch.get_rng_state().clone()
    plot = plots.make_action_chunk_plotter(records, config, metadata, tmp_path, "cpu",
                                           dependencies=dependencies, count=1)
    plot(policy, 1)
    assert torch.equal(before, torch.get_rng_state())
    assert policy.training
    assert seen[0]["predicted"].shape == (4, 2)
    assert np.isfinite(seen[0]["predicted"]).all()


def test_failed_plot_restores_model_state(tmp_path, monkeypatch):
    config, records, metadata = examples()
    policy = FakePolicy()
    policy.child.eval()

    def fail(*args, **kwargs):
        raise RuntimeError("plot failure")

    monkeypatch.setattr(plots, "_plot_chunk", fail)
    plot = plots.make_action_chunk_plotter(records, config, metadata, tmp_path, "cpu", count=1)
    with pytest.raises(RuntimeError, match="plot failure"):
        plot(policy, 1)
    assert policy.training and not policy.child.training

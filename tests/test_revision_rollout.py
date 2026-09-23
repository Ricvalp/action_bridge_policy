"""Dataset-free closed-loop state, clipping, seed and reset checks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from action_bridge.eval import revision_pusht as evaluator
from action_bridge.plan_revision.completion import LearnedCompletion
from action_bridge.plan_revision.contracts import PlanCache


class MockEnv:
    def __init__(self, episode_steps=5):
        self.action_space = SimpleNamespace(low=np.array([0., 0.]), high=np.array([512., 512.]))
        self.episode_steps = episode_steps
        self.episodes = []
        self.closed = False

    def reset(self, *, seed, options=None):
        self.count = 0
        self.actions = []
        self.episodes.append(self.actions)
        self.state = np.array([100., 100., 250., 250., 0.], dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        self.actions.append(action.copy())
        self.count += 1
        self.state[:2] = action
        return self.state.copy(), .2, False, self.count >= self.episode_steps, {"coverage": .19}

    def render(self):
        return np.full((64, 64, 3), 255, dtype=np.uint8)

    def close(self):
        self.closed = True


class Proposal(nn.Module):
    def __init__(self, horizon=4, constant=None):
        super().__init__()
        self.calls, self.horizon, self.constant = [], horizon, constant

    def sample(self, obs_hist, act_hist, mode=None, *, source_actions=None, reference=None, generator=None):
        self.calls.append((obs_hist.clone(), act_hist.clone(), source_actions))
        if self.constant is None:
            result = 100 + torch.randn(1, self.horizon, 2, generator=generator, device=act_hist.device)
        else:
            result = torch.full((1, self.horizon, 2), self.constant, device=act_hist.device)
        return result, {"nfe": 32}


class Revision(nn.Module):
    def __init__(self, constant=None):
        super().__init__()
        self.calls = []
        self.constant = constant

    def sample(self, obs_hist, act_hist, mode=None, *, source_actions=None, reference=None, generator=None,
               has_previous_plan=None):
        assert source_actions is not None
        self.calls.append((obs_hist.clone(), act_hist.clone(), source_actions.clone(), mode, has_previous_plan))
        generated = source_actions + 1 if self.constant is None else torch.full_like(source_actions, self.constant)
        return generated, {"nfe": 32}


def setup(monkeypatch, *, method="fm_paired", constant=None):
    env = MockEnv()
    config = dict(method=method, obs_dim=5, action_dim=2, obs_history=2, action_history=2,
                  horizon=4, execute=2, robot_dt=1., evaluation_seeds=[71, 72], max_episode_steps=8,
                  source_std=0., temperature=.05, revision_gamma=2., prior_ridge=.05, max_rate=4.,
                  mobility_smoothing=0.)
    metadata = {"codec": {"mean": [0., 0.], "std": [1., 1.], "low": [0., 0.],
                           "high": [512., 512.], "units": "pixels"},
                "normalization": {"obs_mean": [0.] * 5, "obs_std": [1.] * 5}}
    completion = LearnedCompletion(5, 2, 2, 2, 4)
    proposal = Proposal(constant=constant)
    dependencies = {"completion_config": {}, "completion_state": {}}
    monkeypatch.setattr(evaluator, "_make_pusht_env", lambda **kwargs: env)
    monkeypatch.setattr(evaluator, "restore_completion", lambda *args: completion)
    return env, config, metadata, dependencies, proposal


def test_actual_history_cached_alignment_reset_and_no_teacher_access(monkeypatch, tmp_path):
    env, config, metadata, dependencies, proposal = setup(monkeypatch)
    policy = Revision()
    caches = []

    class ObservedCache(PlanCache):
        def __init__(self):
            super().__init__()
            caches.append(self)

    monkeypatch.setattr(evaluator, "PlanCache", ObservedCache)
    result = evaluator.evaluate(policy, config, metadata, dependencies, "cpu", output=tmp_path, completion_id=0)
    assert result["episodes"] == 2 and result["episode_length"] == 5
    assert result["startups"] == 1 and result["startup_nfe"] == 32
    assert result["nfe_per_replan"] == 32
    assert not proposal.calls
    assert len(policy.calls) == 6  # Own startup plus two revisions per episode.
    for episode_index in range(2):
        initial_obs, initial_actions, source, mode, available = policy.calls[3 * episode_index]
        assert torch.equal(initial_actions, torch.full_like(initial_actions, 100.))
        assert torch.equal(source, torch.full_like(source, 100.)) and not available.any()
        _, executed_history, _, mode, available = policy.calls[3 * episode_index + 1]
        torch.testing.assert_close(executed_history[0], torch.from_numpy(np.stack(env.episodes[episode_index][:2])))
        assert mode == 0 and available.all()
    first = json.loads((tmp_path / "episode-seed71.json").read_text())
    assert len(first["actions_raw"]) == 5
    assert first["plan_traces"][0]["startup"]
    old_plan = np.array(first["plan_traces"][0]["final_raw"])
    next_trace = first["plan_traces"][1]
    np.testing.assert_array_equal(next_trace["aligned_old_raw"], old_plan[2:])
    np.testing.assert_array_equal(np.array(next_trace["completed_raw"])[:2], old_plan[2:])
    assert caches[0].plan is None and caches[0].executed == 0
    assert env.closed


def test_ddim_and_reviser_generate_their_own_first_prefix(monkeypatch):
    env, config, metadata, dependencies, _ = setup(monkeypatch)
    evaluator.evaluate(Revision(), config, metadata, dependencies, "cpu", completion_id=0, seeds=[71])
    first_prefix = np.stack(env.episodes[0][:2])
    env, config, metadata, dependencies, _ = setup(monkeypatch, method="ddim")
    evaluator.evaluate(Proposal(constant=300.), config, metadata, dependencies, "cpu", seeds=[71])
    np.testing.assert_array_equal(first_prefix, np.full((2, 2), 101.))
    np.testing.assert_array_equal(np.stack(env.episodes[0][:2]), np.full((2, 2), 300.))
    np.testing.assert_array_equal(env.episodes[0][2], [300., 300.])


def test_clipped_commands_not_unexecuted_predictions_enter_history(monkeypatch, tmp_path):
    env, config, metadata, dependencies, _ = setup(monkeypatch, constant=600.)
    policy = Revision(constant=600.)
    result = evaluator.evaluate(policy, config, metadata, dependencies, "cpu", completion_id=0,
                                seeds=[1], output=tmp_path)
    assert result["clipping_rate"] == 1.
    assert torch.equal(policy.calls[1][1], torch.full((1, 2, 2), 512.))
    assert torch.equal(policy.calls[1][2], torch.full((1, 4, 2), 600.))
    trace = json.loads((tmp_path / "episode-seed1.json").read_text())
    assert trace["actions_preclipped_raw"][0] == [600., 600.]
    assert trace["actions_raw"][0] == [512., 512.]


def test_full_execution_reuses_observed_anchor_and_video_selection(monkeypatch, tmp_path):
    env, config, metadata, dependencies, proposal = setup(monkeypatch)
    config.update(execute=4, evaluation_seeds=[1, 2, 3])
    policy = Revision()
    result = evaluator.evaluate(policy, config, metadata, dependencies, "cpu", completion_id=0,
                                output=tmp_path, render=True, save_videos=False, save_gifs=True)
    assert result["startups"] == 2.
    assert not proposal.calls and len(policy.calls) == 6
    assert not any(bool(call[-1]) for call in policy.calls)
    torch.testing.assert_close(policy.calls[1][2], torch.full((1, 4, 2), 101.))
    assert len(list(tmp_path.glob("failure-*.gif"))) == 2
    assert (tmp_path / "failure-seed1.gif").exists() and (tmp_path / "failure-seed2.gif").exists()


def test_native_coverage_distinct_from_legacy_reward_threshold(monkeypatch):
    env, config, metadata, dependencies, _ = setup(monkeypatch, method="ddim")

    def step(action):
        return env.state.copy(), .96, False, True, {"coverage": .912, "is_success": False}

    env.step = step
    metrics = evaluator.evaluate(Proposal(), config, metadata, dependencies, "cpu", seeds=[1])
    assert metrics["success_rate"] == 1 and metrics["env_success_rate"] == 0
    assert metrics["max_coverage"] == .912 and metrics["final_coverage"] == .912


def test_no_bootstrap_dependency_needed(monkeypatch):
    _, config, metadata, dependencies, _ = setup(monkeypatch)
    result = evaluator.evaluate(Revision(), config, metadata, dependencies, "cpu")
    assert result["external_bootstrap"] is False


def test_startup_anchor_normalizes_once_and_uses_no_future_command(monkeypatch):
    env, config, metadata, dependencies, _ = setup(monkeypatch)
    metadata["codec"].update(mean=[50., 20.], std=[10., 5.])
    policy = Revision()
    evaluator.evaluate(policy, config, metadata, dependencies, "cpu", seeds=[1], completion_id=0)
    torch.testing.assert_close(policy.calls[0][2], torch.tensor([5., 16.]).expand(1, 4, 2))
    np.testing.assert_array_equal(env.episodes[0][0], [110., 105.])


def test_nonfinite_startup_is_surfaced_without_fallback(monkeypatch):
    env, config, metadata, dependencies, _ = setup(monkeypatch)
    with pytest.raises(ValueError, match="nonfinite"):
        evaluator.evaluate(Revision(constant=float("nan")), config, metadata, dependencies, "cpu", seeds=[1])
    assert env.closed and env.episodes[0] == []

"""Causal/split/coordinate checks; these are not new learning experiments."""
import torch
import numpy as np
import pytest

from action_bridge.configs.sb_pusht import get_config
from action_bridge.data.revision_pusht import load_windows, context_features
from action_bridge.plan_revision.contracts import ActionCodec, PlanCache
from action_bridge.plan_revision.data import build_source_pairs
from action_bridge.plan_revision.completion import LearnedCompletion


def test_episode_windows_are_causal_and_train_normalized(tmp_path):
    observations = np.arange(10 * 30 * 5, dtype=np.float32).reshape(10, 30, 5)
    actions = np.arange(10 * 30 * 2, dtype=np.float32).reshape(10, 30, 2)
    path = tmp_path / "episodes.npz"
    np.savez(path, obs=observations, actions=actions)
    config = get_config()
    records, metadata = load_windows(path, config)
    assert metadata["splits"] == {"train": list(range(8)), "val": [8], "test": [9]}
    expected_mean = torch.from_numpy(actions[:8]).flatten(0, 1).mean(0)
    torch.testing.assert_close(torch.tensor(metadata["normalization"]["action_mean"]), expected_mean)
    train = records["train"]
    assert bool((train["time_index"] - train["earlier_time_index"] == 8).all())
    assert train["valid_mask"].all()
    # Episode initial history is padded using the observed pusher, not action[0].
    codec = ActionCodec(**metadata["codec"])
    torch.testing.assert_close(codec.decode(train["earlier_act_hist"][0]),
                               torch.from_numpy(observations[0, 0, :2]).expand(2, -1))
    assert not train["earlier_action_history_mask"][0].any()
    torch.testing.assert_close(codec.decode(train["future_actions"][0]), torch.from_numpy(actions[0, 8:24]))
    features = context_features(train, metadata["normalization"])
    modified = dict(train, future_actions=torch.randn_like(train["future_actions"]))
    torch.testing.assert_close(features, context_features(modified, metadata["normalization"]))


def test_fixed_proposals_use_earlier_histories_only():
    config = get_config() | {"horizon": 4, "execute": 2, "batch_size": 2}
    windows = {"obs_hist": torch.zeros(3, 2, 5), "act_hist": torch.zeros(3, 2, 2),
               "earlier_obs_hist": torch.ones(3, 2, 5), "earlier_act_hist": torch.ones(3, 2, 2),
               "future_actions": torch.full((3, 4, 2), 123.), "valid_mask": torch.ones(3, 4, dtype=torch.bool)}
    class Proposal:
        def sample(self, obs_hist, act_hist, *, generator):
            assert obs_hist.eq(1).all() and act_hist.eq(1).all()
            return torch.zeros(len(obs_hist), 4, 2), {}
    completion = LearnedCompletion(5, 2, 2, 2, 4)
    records = build_source_pairs(windows, Proposal(), completion, torch.ones(2), config, "cpu")
    assert len(records["old_actions"]) == 6
    assert records["old_actions"].eq(0).all()
    assert records["future_actions"].eq(123).all()
    assert records["proposal_id"].tolist() == [0, 1, 0, 1, 0, 1]


def test_codec_and_plan_cache():
    codec = ActionCodec([1.] * 6, [2.] * 6, [-10.] * 6, [10.] * 6, "meters")
    actions = torch.randn(1, 9, 6)
    torch.testing.assert_close(codec.decode(codec.encode(actions)), actions)
    cache = PlanCache()
    assert cache.needs_bootstrap
    cache.store(actions)
    cache.advance(9)
    assert cache.needs_bootstrap  # K=H deliberately bootstraps, not a nonexistent tail.
    cache.reset()
    assert cache.plan is None and cache.executed == 0
    with pytest.raises(ValueError, match="absolute"):
        ActionCodec([0.], [1.], [-1.], [1.], "radians", semantics="torque")

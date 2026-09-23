"""Causal/split/coordinate checks; these are not new learning experiments."""
import torch
import numpy as np
import pytest

from action_bridge.configs.sb_pusht import get_config
from action_bridge.data.revision_pusht import load_windows, context_features
from action_bridge.plan_revision.contracts import ActionCodec, PlanCache
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
    assert train["time_index"].tolist() == [0, 8] * 8
    assert train["valid_mask"].all()
    # Episode initial history is padded using the observed pusher, not action[0].
    codec = ActionCodec(**metadata["codec"])
    torch.testing.assert_close(codec.decode(train["act_hist"][0]),
                               torch.from_numpy(observations[0, 0, :2]).expand(2, -1))
    assert not train["act_history_mask"][0].any()
    torch.testing.assert_close(codec.decode(train["startup_actions"][0]),
                               torch.from_numpy(observations[0, 0, :2]).expand(16, -1))
    torch.testing.assert_close(codec.decode(train["future_actions"][0]), torch.from_numpy(actions[0, :16]))
    features = context_features(train, metadata["normalization"])
    modified = dict(train, future_actions=torch.randn_like(train["future_actions"]))
    torch.testing.assert_close(features, context_features(modified, metadata["normalization"]))


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

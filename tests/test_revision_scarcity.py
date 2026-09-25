"""Whole-demonstration scarcity without moving held-out episodes or leaking data."""
import random

import numpy as np
import pytest
import torch

from action_bridge.configs.sb_pusht import get_config
from action_bridge.data.pusht_adapter import PushTLowDimDataset
from action_bridge.data.revision_pusht import load_windows
from action_bridge.plan_revision.contracts import ActionCodec


def arrays(episodes=205, length=12):
    generator = np.random.default_rng(17)
    obs = generator.normal(size=(episodes, length, 5)).astype(np.float32)
    actions = generator.normal(size=(episodes, length, 2)).astype(np.float32)
    # Distinct episode distributions make normalization leaks detectable.
    obs += np.arange(episodes, dtype=np.float32)[:, None, None]
    actions += 2 * np.arange(episodes, dtype=np.float32)[:, None, None]
    return obs, actions


@pytest.fixture
def dataset_path(tmp_path):
    path = tmp_path / "episodes.npz"
    obs, actions = arrays()
    np.savez(path, obs=obs, actions=actions)
    return path


def config(fraction=1.0, subset_seed=0):
    return dict(get_config(), horizon=4, execute=2,
                train_episode_fraction=fraction, subset_seed=subset_seed)


def dataset(path, fraction=1., subset_seed=0, **kwargs):
    return PushTLowDimDataset(
        str(path), chunk_horizon=4, pad_episode_starts=True,
        train_episode_fraction=fraction, subset_seed=subset_seed, **kwargs)


def test_nested_episode_subsets_with_unchanged_validation_and_test(dataset_path):
    selections = []
    for fraction, count in [(1., 164), (.5, 82), (.25, 41), (.1, 16)]:
        windows, metadata = load_windows(dataset_path, config(fraction, 11))
        ids = metadata["splits"]["train"]
        assert len(ids) == count
        assert ids == sorted(ids)
        assert windows["train"]["episode_id"].unique().tolist() == ids
        assert metadata["splits"]["val"] == list(range(164, 184))
        assert metadata["splits"]["test"] == list(range(184, 205))
        assert windows["val"]["episode_id"].unique().tolist() == list(range(164, 184))
        selections.append(set(ids))
        if fraction < 1:
            subset = metadata["train_subset"]
            assert subset == {
                "fraction": fraction, "seed": 11,
                "original_splits": {"train": list(range(164)), "val": list(range(164, 184)),
                                    "test": list(range(184, 205))},
                "selected_train_episode_ids": ids}
    assert selections[3] < selections[2] < selections[1] < selections[0]


def test_scarcity_is_not_first_n_episodes_or_a_new_train_val_split(dataset_path):
    first = dataset(dataset_path, .25, 0)
    repeated = dataset(dataset_path, .25, 0)
    other = dataset(dataset_path, .25, 1)
    assert first.episode_ids == repeated.episode_ids
    assert first.episode_ids != list(range(41))
    assert first.episode_ids != other.episode_ids
    assert first.original_split_ids == other.original_split_ids


def test_excluded_training_episodes_cannot_affect_normalizer_or_any_windows(dataset_path, tmp_path):
    settings = config(.25, 8)
    original, metadata = load_windows(dataset_path, settings)
    obs, actions = arrays()
    excluded = sorted(set(range(164)) - set(metadata["splits"]["train"]))
    obs[excluded] += 10000
    actions[excluded] -= 10000
    changed = tmp_path / "excluded-modified.npz"
    np.savez(changed, obs=obs, actions=actions)
    changed_windows, changed_metadata = load_windows(changed, settings)
    assert metadata | {"dataset_path": str(changed)} == changed_metadata
    for split in ("train", "val", "test"):
        for field in original[split]:
            torch.testing.assert_close(original[split][field], changed_windows[split][field], rtol=0, atol=0)


def test_stats_fit_selected_training_episodes_for_every_split(dataset_path):
    obs, actions = arrays()
    selections = dataset(dataset_path, .1, 2).selected_train_episode_ids
    expected_obs = torch.from_numpy(obs[selections]).flatten(0, 1)
    expected_actions = torch.from_numpy(actions[selections]).flatten(0, 1)
    for split in ("train", "val", "test", "all"):
        selected = dataset(dataset_path, .1, 2, split=split, normalize=True)
        stats = selected.normalization_stats
        for key, expected in [("obs_mean", expected_obs.mean(0)), ("obs_std", expected_obs.std(0)),
                              ("action_mean", expected_actions.mean(0)), ("action_std", expected_actions.std(0))]:
            torch.testing.assert_close(torch.tensor(stats[key]), expected)
        episode = selected.episode_ids[0]
        record = selected.item_from_episode_time(episode, 0)
        expected_anchor = ((torch.from_numpy(obs[episode, 0, :2]) - expected_actions.mean(0))
                           / expected_actions.std(0))
        torch.testing.assert_close(record["act_hist"], expected_anchor.expand(2, -1))
        torch.testing.assert_close(record["obs_hist"], selected.observations[episode][0].expand(2, -1))
        torch.testing.assert_close(record["future_actions"], selected.actions[episode][:4])


def test_explicit_stats_are_respected_without_refitting(dataset_path):
    stats = {"obs_mean": [10.] * 5, "obs_std": [2.] * 5,
             "action_mean": [-3.] * 2, "action_std": [7.] * 2}
    selected = dataset(dataset_path, .1, normalize=True, normalization_stats=stats)
    assert selected.normalization_stats == stats
    obs, actions = arrays()
    episode = selected.episode_ids[0]
    torch.testing.assert_close(selected.observations[episode], (torch.from_numpy(obs[episode]) - 10) / 2)
    torch.testing.assert_close(selected.actions[episode], (torch.from_numpy(actions[episode]) + 3) / 7)


def test_initial_anchor_and_targets_round_trip_through_selected_codec(dataset_path):
    records, metadata = load_windows(dataset_path, config(.1))
    obs, actions = arrays()
    codec = ActionCodec(**metadata["codec"])
    for split in ("train", "val", "test"):
        record = records[split]
        episode = int(record["episode_id"][0])
        torch.testing.assert_close(codec.decode(record["startup_actions"][0]),
                                   torch.from_numpy(obs[episode, 0, :2]).expand(4, -1), atol=3e-5, rtol=1e-5)
        torch.testing.assert_close(codec.decode(record["future_actions"][0]),
                                   torch.from_numpy(actions[episode, :4]), atol=3e-5, rtol=1e-5)


def test_subset_seed_not_training_seed_and_no_global_random_state_changes(dataset_path):
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    _, first = load_windows(dataset_path, config(.25, 5) | {"seed": 1})
    _, second = load_windows(dataset_path, config(.25, 5) | {"seed": 999})
    assert first == second
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_full_dataset_keeps_original_metadata_and_order(dataset_path):
    settings = config()
    del settings["train_episode_fraction"], settings["subset_seed"]
    old, old_metadata = load_windows(dataset_path, settings)
    explicit, explicit_metadata = load_windows(dataset_path, config(1., 17))
    assert old_metadata == explicit_metadata
    assert "train_subset" not in old_metadata
    for split in old:
        for field in old[split]:
            torch.testing.assert_close(old[split][field], explicit[split][field], rtol=0, atol=0)


def test_smallest_subset_retains_one_training_episode(dataset_path):
    selected = dataset(dataset_path, 1e-6)
    assert len(selected.episode_ids) == 1


@pytest.mark.parametrize("fraction", [0., -1., 1.1, float("inf"), float("nan"), True, None, "0.5"])
def test_invalid_fraction_rejected(dataset_path, fraction):
    with pytest.raises(ValueError, match="train_episode_fraction"):
        dataset(dataset_path, fraction)


@pytest.mark.parametrize("seed", [-1, 1.5, True, None, "1"])
def test_invalid_subset_seed_rejected(dataset_path, seed):
    with pytest.raises(ValueError, match="subset_seed"):
        dataset(dataset_path, .5, seed)

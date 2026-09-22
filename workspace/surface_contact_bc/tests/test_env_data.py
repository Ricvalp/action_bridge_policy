import json

import numpy as np
import pytest

from workspace.surface_contact_bc.data import EpisodeWindows, Normalizer, load_episodes, make_dataset
from workspace.surface_contact_bc.env import Context, SurfaceContactEnv, rollout_expert, sample_context


def test_free_space_and_unilateral_contact():
    env = SurfaceContactEnv(sample_context(0))
    assert env.contact_info()["normal_force"] == 0.
    env.x = env.context.normal * (env.context.offset - .01)
    env.v = .1 * env.context.tangent
    info = env.contact_info()
    assert info["normal_force"] > 0.
    assert info["friction_force"] * (env.context.tangent @ env.v) <= 0.
    env.v = env.context.normal * 10.
    assert env.contact_info()["normal_force"] == 0.  # no adhesive damping


def test_penetration_peak_includes_transient_between_control_samples():
    env = SurfaceContactEnv(sample_context(0))
    context = env.context
    env.x = (context.offset - .008) * context.normal
    env.v = -2. * context.normal
    env.step((context.offset + .6) * context.normal)
    # The impact exceeds the clean-contact threshold, yet the next observation
    # has almost recovered. Metrics must retain the substep penetration peak.
    assert env.last_info["peak_penetration"] > .02
    assert env.last_info["penetration"] < .005


@pytest.mark.parametrize("suite,value", [("id", None), ("mass", .5), ("mass", 2.),
    ("stiffness", 3.), ("damping", 2.), ("friction", 3.), ("gains", .5), ("gains", 2.)])
def test_integration_stability(suite, value):
    context = sample_context(2, suite, value)
    env = SurfaceContactEnv(context)
    target = context.normal * (context.offset - .1) + context.goal * context.tangent
    for step in range(150):
        obs = env.step(target, impulse=1. if step == 75 else 0.)
        assert np.isfinite(obs).all()
        assert np.linalg.norm(env.x) < 3.
        assert np.linalg.norm(env.v) < 20.
        assert env.last_info["normal_force"] >= 0.


def test_seeded_expert_and_context_roundtrip():
    context = sample_context(5)
    copy = Context.from_dict(json.loads(json.dumps(context.to_dict())))
    first, second = rollout_expert(context), rollout_expert(copy)
    np.testing.assert_array_equal(first["observations"], second["observations"])
    np.testing.assert_array_equal(first["actions"], second["actions"])
    assert np.max(first["infos"]["penetration"]) < .02
    assert np.all(first["infos"]["contact"][-12:])
    final = first["observations"][-1]
    assert abs(context.tangent @ final[:2] - context.goal) < .025
    assert np.linalg.norm(final[2:4]) < .08


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "demos"
    make_dataset(root, train_episodes=3, val_episodes=2, test_episodes=2, seed=20)
    return root


def test_context_splits_geometry_and_normalization(dataset):
    manifest = json.loads((dataset / "manifest.json").read_text())
    seeds = [entry["seed"] for entry in manifest["episodes"]]
    assert len(set(seeds)) == len(seeds)
    assert manifest["expert_quality"] == {"passed": 7, "failed": 0, "filtered": 0}
    stats = Normalizer.from_dict(manifest["normalizer"])
    for split in ("train", "val", "test"):
        for episode in load_episodes(dataset, split):
            obs, actions = episode["observations"], episode["actions"]
            context = Context.from_dict(episode["context"])
            np.testing.assert_allclose(obs[:, 4:6], np.broadcast_to(context.normal, (len(obs), 2)), atol=1e-7)
            np.testing.assert_allclose(obs[:, 7], obs[:, :2] @ context.normal - context.offset, atol=1e-7)
            np.testing.assert_allclose(stats.denormalize_obs(stats.normalize_obs(obs)), obs, atol=1e-6)
            np.testing.assert_allclose(stats.denormalize_action(stats.normalize_action(actions)), actions, atol=1e-7)


def test_windows_do_not_cross_episodes(dataset):
    windows = EpisodeWindows(dataset, obs_history=2, action_history=2, horizon=8)
    normalizer = windows.normalizer
    length = len(windows.episodes[0]["actions"])
    first, last, next_episode = windows[0], windows[length - 1], windows[length]
    np.testing.assert_allclose(normalizer.denormalize_action(first["act_hist"]),
                               np.broadcast_to(windows.episodes[0]["observations"][0, :2], (2, 2)), atol=1e-7)
    np.testing.assert_allclose(normalizer.denormalize_action(last["future_actions"]),
                               np.broadcast_to(windows.episodes[0]["actions"][-1], (8, 2)), atol=1e-7)
    assert last["episode_id"] == 0 and next_episode["episode_id"] == 1
    assert next_episode["step"] == 0
    expected = windows.episodes[0]["actions"][:8]
    np.testing.assert_allclose(normalizer.denormalize_action(first["future_actions"]), expected, atol=1e-7)


def test_low_data_stats_fit_only_selected_train_episodes(dataset):
    train = EpisodeWindows(dataset, limit=1)
    val = EpisodeWindows(dataset, split="val", limit=1)
    assert len(train.episodes) == 1 and len(val.episodes) == 2
    assert train.normalizer.to_dict() == val.normalizer.to_dict()
    expected = Normalizer.fit(load_episodes(dataset, "train")[:1])
    assert expected.to_dict() == train.normalizer.to_dict()
    assert expected.to_dict() != Normalizer.fit(load_episodes(dataset, "train")).to_dict()


def test_ood_orientations_are_disjoint():
    train_normals = [sample_context(seed).normal for seed in range(20)]
    unseen = sample_context(0, "unseen_orientation").normal
    assert not any(np.allclose(unseen, normal) for normal in train_normals)
    with pytest.raises(ValueError, match="differ"):
        sample_context(0, "unseen_orientation", 20.)


def test_collection_refuses_overwrite(dataset):
    with pytest.raises(FileExistsError):
        make_dataset(dataset, train_episodes=1)

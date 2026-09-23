"""Push-T rollout videos are bounded, local artifacts, enabled by the CLI."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from action_bridge.eval import revision_pusht as evaluator


class VideoEnv:
    def __init__(self):
        self.action_space = SimpleNamespace(low=np.zeros(2), high=np.full(2, 512.))
        self.render_calls = 0
        self.closed = False

    def reset(self, *, seed, options=None):
        self.seed, self.count = seed, 0
        self.state = np.array([100., 100., 250., 250., 0.], dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        self.count += 1
        self.state[:2] = action
        success = self.seed % 2 == 0
        return self.state.copy(), float(success), success, self.count >= 2, {"is_success": success}

    def render(self):
        self.render_calls += 1
        return np.full((64, 64, 3), 200, dtype=np.uint8)

    def close(self):
        self.closed = True


class FixedPolicy(torch.nn.Module):
    def sample(self, obs_hist, act_hist, *args, **kwargs):
        return torch.full((1, 4, 2), 100.), {"nfe": 1}


@pytest.fixture
def rollout(monkeypatch):
    env = VideoEnv()
    monkeypatch.setattr(evaluator, "_make_pusht_env", lambda **kwargs: env)
    config = dict(method="ddim", obs_dim=5, action_dim=2, horizon=4, execute=2,
                  obs_history=2, action_history=2, max_episode_steps=2,
                  robot_dt=1., evaluation_seeds=list(range(1, 9)))
    metadata = {"codec": {"mean": [0., 0.], "std": [1., 1.], "low": [0., 0.],
                          "high": [512., 512.], "units": "pixels"},
                "normalization": {"obs_mean": [0.] * 5, "obs_std": [1.] * 5}}
    return env, (FixedPolicy(), config, metadata, {}, "cpu")


def test_rendering_defaults_to_two_success_and_two_failure_mp4s(rollout, monkeypatch, tmp_path):
    env, arguments = rollout
    writes = []

    def save_video(frames, path):
        writes.append(path.name)
        assert all(np.asarray(frame).shape == (64, 64, 3) for frame in frames)
        path.write_bytes(b"mock-mp4")

    monkeypatch.setattr(evaluator, "_save_video", save_video)
    result = evaluator.evaluate(*arguments, output=tmp_path, render=True)
    assert result["episodes"] == 8 and result["success_rate"] == .5
    assert writes == ["failure-seed1.mp4", "success-seed2.mp4",
                      "failure-seed3.mp4", "success-seed4.mp4"]
    assert env.render_calls == 6  # No frames once both media quotas are full.
    assert not list(tmp_path.glob("*.gif"))
    for seed in range(1, 9):
        episode = json.loads((tmp_path / f"episode-seed{seed}.json").read_text())
        assert ("video" in episode) == (seed <= 4)
        assert "gif" not in episode
        if "video" in episode:
            assert (tmp_path / episode["video"]).is_file()
    assert env.closed


@pytest.mark.parametrize("render,save_videos,save_gifs", [
    (False, True, True), (True, False, False),
])
def test_disabled_media_never_renders(rollout, monkeypatch, tmp_path, render, save_videos, save_gifs):
    env, arguments = rollout
    writer = Mock()
    monkeypatch.setattr(evaluator, "_save_video", writer)
    evaluator.evaluate(*arguments, output=tmp_path, render=render,
                       save_videos=save_videos, save_gifs=save_gifs)
    assert env.render_calls == 0
    writer.assert_not_called()
    assert not list(tmp_path.glob("*.mp4")) and not list(tmp_path.glob("*.gif"))


def test_optional_gifs_have_distinct_artifact_fields(rollout, monkeypatch, tmp_path):
    _, arguments = rollout
    monkeypatch.setattr(evaluator, "_save_video", lambda frames, path: path.write_bytes(b"mock-mp4"))
    evaluator.evaluate(*arguments, output=tmp_path, render=True, save_gifs=True, seeds=[1])
    episode = json.loads((tmp_path / "episode-seed1.json").read_text())
    assert episode["video"] == "failure-seed1.mp4"
    assert episode["gif"] == "failure-seed1.gif"
    assert (tmp_path / episode["gif"]).read_bytes().startswith(b"GIF")


def test_video_encoding_failure_is_visible_and_closes_environment(rollout, monkeypatch, tmp_path):
    env, arguments = rollout
    monkeypatch.setattr(evaluator, "_save_video", Mock(side_effect=RuntimeError("encoder failed")))
    with pytest.raises(RuntimeError, match="encoder failed"):
        evaluator.evaluate(*arguments, output=tmp_path, render=True, seeds=[1])
    assert env.closed


def test_real_mp4_round_trip(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    import imageio.v2 as imageio

    frames = [np.full((64, 64, 3), value, dtype=np.uint8) for value in (0, 80, 160)]
    path = tmp_path / "rollout.mp4"
    evaluator._save_video(frames, path)
    assert path.stat().st_size > 0
    with imageio.get_reader(path, format="FFMPEG") as reader:
        assert reader.get_meta_data()["fps"] == 10
        assert reader.count_frames() == 3
        assert reader.get_data(0).shape == frames[0].shape
        assert reader.get_data(2).mean() > reader.get_data(0).mean() + 100

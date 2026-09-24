"""Bounded CPU spawn checks and deterministic parallel rollout orchestration."""

from concurrent.futures import Future
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from action_bridge.configs.sb_pusht import get_config
from action_bridge.eval import revision_pusht as serial
from action_bridge.eval import revision_pusht_parallel as parallel
from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.completion import LearnedCompletion
from action_bridge.plan_revision.models import build_policy
from action_bridge.plan_revision.training import completion_config


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_state(method="ddim", *, weights=False):
    config = get_config(method) | {
        "horizon": 4, "execute": 2, "channels": [8, 16],
        "history_dim": 16, "hidden_dim": 16, "time_dim": 8,
        "reference_hidden_dim": 8, "num_inference_steps": 2,
        "max_episode_steps": 2, "evaluation_seeds": [31, 4, 23],
    }
    metadata = {
        "dataset_path": "/not/a/real/training/dataset.zarr",
        "codec": {"mean": [256., 256.], "std": [20., 20.],
                  "low": [0., 0.], "high": [512., 512.], "units": "pixels"},
        "normalization": {"obs_mean": [256., 256., 256., 256., np.pi],
                          "obs_std": [100., 100., 100., 100., np.pi]},
    }
    dependencies = {}
    ema = {}
    if weights:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12)
            ema = copy.deepcopy(build_policy(config).state_dict())
            if method != "ddim":
                reference_config = completion_config(config)
                reference = LearnedCompletion(**reference_config)
                dependencies = {
                    "completion_config": reference_config,
                    "completion_state": reference.state_dict(),
                    "innovation_variance": torch.ones(2),
                }
    return {"config": config, "metadata": metadata, "dependencies": dependencies,
            "ema": ema, "direction": "forward"}


def read_episode(directory, seed):
    return json.loads((Path(directory) / f"episode-seed{seed}.json").read_text())


@pytest.fixture
def scored_episode():
    return {
        "actions_raw": [[.5, .75]], "actions_preclipped_raw": [[.5, .75]],
        "states_raw": [[.5, .75, .25, .125, .625]],
        "rewards": [.5, .75], "coverage": [.5, .75],
        "max_coverage": .75, "final_coverage": .75,
        "success": True, "env_success": True, "terminated": True,
        "truncated": False, "episode_length": 2,
    }


@pytest.mark.parametrize("field", ["rewards", "coverage", "max_coverage", "final_coverage"])
def test_replay_accepts_score_roundoff_without_modifying_records(scored_episode, field):
    original = copy.deepcopy(scored_episode)
    replay = copy.deepcopy(original)
    replay[field] = (np.asarray(replay[field]) + 3.33e-16).tolist()
    assert replay[field] != original[field]
    expected_replay = copy.deepcopy(replay)

    parallel._validate_replay(original, replay, seed=31)

    assert original == scored_episode
    assert replay == expected_replay


@pytest.mark.parametrize("field", ["rewards", "coverage", "max_coverage", "final_coverage"])
@pytest.mark.parametrize("change", ["drift", "shape"])
def test_replay_rejects_score_drift_and_shape_changes(scored_episode, field, change):
    replay = copy.deepcopy(scored_episode)
    replay[field] = ((np.asarray(replay[field]) + .001).tolist() if change == "drift"
                     else [replay[field]])

    with pytest.raises(RuntimeError) as error:
        parallel._validate_replay(scored_episode, replay, seed=31)

    assert "31" in str(error.value) and field in str(error.value)


@pytest.mark.parametrize("field", ["rewards", "coverage", "max_coverage", "final_coverage"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["original", "replay", "both"])
def test_replay_rejects_nonfinite_scores(scored_episode, field, nonfinite, side):
    original, replay = copy.deepcopy(scored_episode), copy.deepcopy(scored_episode)
    for name, episode in (("original", original), ("replay", replay)):
        if side in (name, "both"):
            episode[field] = np.full_like(np.asarray(episode[field]), nonfinite).tolist()

    with pytest.raises(RuntimeError) as error:
        parallel._validate_replay(original, replay, seed=31)

    assert "31" in str(error.value) and field in str(error.value)


@pytest.mark.parametrize("field", ["actions_raw", "actions_preclipped_raw", "states_raw"])
def test_replay_rejects_one_ulp_trajectory_drift(scored_episode, field):
    replay = copy.deepcopy(scored_episode)
    replay[field][0][0] = np.nextafter(replay[field][0][0], np.inf).item()
    assert replay[field] != scored_episode[field]

    with pytest.raises(RuntimeError) as error:
        parallel._validate_replay(scored_episode, replay, seed=31)

    assert "31" in str(error.value) and field in str(error.value)


@pytest.mark.parametrize("field", ["success", "env_success", "terminated", "truncated",
                                   "episode_length"])
def test_replay_rejects_changed_outcomes(scored_episode, field):
    replay = copy.deepcopy(scored_episode)
    replay[field] = replay[field] + 1 if field == "episode_length" else not replay[field]

    with pytest.raises(RuntimeError) as error:
        parallel._validate_replay(scored_episode, replay, seed=31)

    assert "31" in str(error.value) and field in str(error.value)


@pytest.mark.parametrize("method", ["ddim", "fm_paired", "sb_kinetic"])
def test_spawn_matches_serial_for_uneven_seed_count_without_training_files(tmp_path, method):
    """Real tiny policies, real spawn, and at most two simulator steps per seed."""
    pytest.importorskip("gym_pusht")
    state = tiny_state(method, weights=True)
    identity = checkpoints.content_digest(state)
    config = state["config"]
    seeds = config["evaluation_seeds"]
    serial_output, parallel_output = tmp_path / "serial", tmp_path / "parallel"
    expected = serial.evaluate(
        checkpoints.restore_policy(state, "cpu"), config, state["metadata"],
        state["dependencies"], "cpu", output=serial_output, seeds=seeds,
        save_videos=False, progress=False,
    )
    actual = parallel.evaluate_parallel(
        state, config, "cpu", output=parallel_output, seeds=seeds,
        workers=2, threads=1, save_videos=False, progress=False,
    )

    assert actual.keys() == expected.keys()
    for key in expected:
        if key == "inference_seconds_per_replan":
            assert actual[key] > 0
        elif isinstance(expected[key], float):
            assert actual[key] == pytest.approx(expected[key], rel=1e-6, abs=1e-7)
        else:
            assert actual[key] == expected[key]
    assert actual["episodes"] == 3 and actual["seeds"] == [31, 4, 23]
    assert json.loads((parallel_output / "metrics.json").read_text()) == actual
    assert sorted(path.name for path in parallel_output.glob("episode-*.json")) == [
        "episode-seed23.json", "episode-seed31.json", "episode-seed4.json",
    ]
    for seed in seeds:
        expected_episode = read_episode(serial_output, seed)
        actual_episode = read_episode(parallel_output, seed)
        assert actual_episode["seed"] == seed
        for key in ("actions_raw", "actions_preclipped_raw", "states_raw", "rewards", "coverage"):
            np.testing.assert_allclose(actual_episode[key], expected_episode[key], rtol=1e-6, atol=1e-6)
        assert "video" not in actual_episode and "gif" not in actual_episode
    assert checkpoints.content_digest(state) == identity
    assert not list(parallel_output.glob("*.mp4"))
    assert not list(parallel_output.glob("*.gif"))


@pytest.mark.parametrize("arguments,match", [
    ({"workers": 0}, "workers"),
    ({"workers": -1}, "workers"),
    ({"threads": 0}, "threads"),
    ({"threads": -1}, "threads"),
    ({"device": "cuda:0"}, "[Cc][Pp][Uu]"),
    ({"device": "mps"}, "[Cc][Pp][Uu]"),
    ({"seeds": [9, 9]}, "[Ss]eed|unique|duplicate"),
    ({"seeds": []}, "[Ss]eed"),
])
def test_invalid_parallel_arguments_fail_before_launch(monkeypatch, tmp_path, arguments, match):
    executor = Mock(side_effect=AssertionError("invalid arguments started a worker"))
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", executor)
    state = tiny_state()
    options = dict(device="cpu", seeds=[2, 1], workers=2, threads=1,
                   output=tmp_path / "evaluation", save_videos=False, progress=False)
    options.update(arguments)
    with pytest.raises(ValueError, match=match):
        parallel.evaluate_parallel(state, state["config"], **options)
    executor.assert_not_called()
    assert not (options["output"] / "metrics.json").exists()


class FixedPolicy(torch.nn.Module):
    def sample(self, obs_hist, act_hist, *args, **kwargs):
        return torch.zeros((1, 4, 2)), {"nfe": 1}


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


@pytest.fixture
def inline_pool(monkeypatch):
    """One reusable fake worker, with results deliberately collected backwards."""
    observed = SimpleNamespace(pools=[], calls=[], scores={}, environments=[], writes=[],
                               fail_scoring_seed=None, fail_render_seed=None,
                               corrupt_replay_seed=None, roundoff_replay_seed=None)
    original_evaluate = serial.evaluate
    restore = Mock(return_value=FixedPolicy())
    monkeypatch.setattr(parallel.checkpoints, "restore_policy", restore)
    monkeypatch.setattr(parallel.torch, "set_num_interop_threads", Mock())
    observed.restore = restore

    def make_env(**kwargs):
        env = VideoEnv()
        observed.environments.append(env)
        return env

    def evaluate(*args, **kwargs):
        seed, = kwargs["seeds"]
        observed.calls.append((seed, dict(kwargs)))
        assert kwargs["write_metrics"] is False
        assert kwargs["progress"] is False
        output = Path(kwargs["output"])
        assert not (output / "metrics.json").exists()
        if not kwargs["render"] and seed == observed.fail_scoring_seed:
            raise RuntimeError("scoring failed")
        metrics = original_evaluate(*args, **kwargs)
        episode_path = output / f"episode-seed{seed}.json"
        episode = read_episode(output, seed)
        if kwargs["render"]:
            episode["inference_seconds_per_replan"] = 999.
            episode["render_only_marker"] = True
            if seed == observed.corrupt_replay_seed:
                episode["actions_raw"][0][0] += 1
            if seed == observed.roundoff_replay_seed:
                for field in ("rewards", "coverage", "max_coverage", "final_coverage"):
                    episode[field] = (np.asarray(episode[field]) + 3.33e-16).tolist()
        else:
            episode["scoring_only_marker"] = seed
            observed.scores[seed] = copy.deepcopy(episode)
        episode_path.write_text(json.dumps(episode))
        return metrics

    def save_video(frames, path):
        assert frames
        observed.writes.append(path.name)
        if path.stem.endswith(f"seed{observed.fail_render_seed}"):
            raise RuntimeError("encoder failed")
        path.write_bytes(b"mock-mp4")

    class InlineExecutor:
        def __init__(self, *, max_workers, mp_context, initializer, initargs):
            self.max_workers = max_workers
            self.context = mp_context
            self.submissions = []
            self.closed = False
            observed.pools.append(self)
            initializer(*initargs)

        def submit(self, function, *args, **kwargs):
            self.submissions.append((function, args, kwargs))
            future = Future()
            try:
                future.set_result(function(*args, **kwargs))
            except BaseException as error:
                future.set_exception(error)
            return future

        def shutdown(self, wait=True, *, cancel_futures=False):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.shutdown()

    monkeypatch.setattr(serial, "_make_pusht_env", make_env)
    monkeypatch.setattr(serial, "evaluate", evaluate)
    monkeypatch.setattr(serial, "_save_video", save_video)
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(parallel, "as_completed", lambda futures: reversed(list(futures)))
    return observed


@pytest.mark.parametrize("save_videos,save_gifs", [(True, False), (False, True), (True, True)])
def test_global_media_quota_seed_order_and_original_scoring_records(
        inline_pool, tmp_path, save_videos, save_gifs):
    state = tiny_state()
    seeds = [9, 2, 7, 6, 1, 4, 8, 3]
    result = parallel.evaluate_parallel(
        state, state["config"], "cpu", output=tmp_path, seeds=seeds,
        workers=3, threads=1, save_videos=save_videos, save_gifs=save_gifs, progress=False,
    )
    assert result["seeds"] == seeds and result["episodes"] == len(seeds)
    assert result["success_rate"] == .5
    assert result["inference_seconds_per_replan"] == pytest.approx(
        np.mean([episode["inference_seconds_per_replan"] for episode in inline_pool.scores.values()]))
    pool, = inline_pool.pools
    assert pool.max_workers == 3 and pool.context.get_start_method() == "spawn"
    assert pool.closed
    inline_pool.restore.assert_called_once()
    assert len(pool.submissions) == 12
    assert [seed for seed, kwargs in inline_pool.calls if not kwargs["render"]] == seeds
    assert [seed for seed, kwargs in inline_pool.calls if kwargs["render"]] == [9, 2, 7, 6]
    for _, kwargs in inline_pool.calls[:len(seeds)]:
        assert not kwargs["save_videos"] and not kwargs["save_gifs"]
    for seed in seeds:
        episode = read_episode(tmp_path, seed)
        artifacts = {key: episode.pop(key) for key in ("video", "gif") if key in episode}
        assert episode == inline_pool.scores[seed]
        if seed not in seeds[:4]:
            assert artifacts == {}
            continue
        bucket = "success" if seed % 2 == 0 else "failure"
        expected = {}
        if save_videos:
            expected["video"] = f"{bucket}-seed{seed}.mp4"
        if save_gifs:
            expected["gif"] = f"{bucket}-seed{seed}.gif"
        assert artifacts == expected
        assert all((tmp_path / filename).is_file() for filename in artifacts.values())
    assert sum(env.render_calls for env in inline_pool.environments) == 6
    assert all(env.closed for env in inline_pool.environments)
    assert json.loads((tmp_path / "metrics.json").read_text()) == result


def test_render_roundoff_preserves_scored_episode_and_metrics(inline_pool, tmp_path):
    state = tiny_state()
    seeds = [7]
    inline_pool.roundoff_replay_seed = 7

    result = parallel.evaluate_parallel(
        state, state["config"], "cpu", output=tmp_path, seeds=seeds,
        workers=1, threads=1, save_videos=True, progress=False,
    )

    episode = read_episode(tmp_path, 7)
    assert episode.pop("video") == "failure-seed7.mp4"
    assert episode == inline_pool.scores[7]
    expected = serial.summarize_episodes(
        [inline_pool.scores[7]], state["config"], state["metadata"],
        completion_id=2, seeds=seeds,
    )
    assert result == expected
    assert json.loads((tmp_path / "metrics.json").read_text()) == expected
    assert (tmp_path / "failure-seed7.mp4").read_bytes() == b"mock-mp4"
    assert [kwargs["render"] for _, kwargs in inline_pool.calls] == [False, True]


def test_disabled_media_has_only_nonrendering_scoring_jobs(inline_pool, tmp_path):
    state = tiny_state()
    result = parallel.evaluate_parallel(
        state, state["config"], "cpu", output=tmp_path, seeds=[9, 2, 7],
        workers=2, threads=1, save_videos=False, save_gifs=False, progress=False,
    )
    assert result["episodes"] == 3
    assert len(inline_pool.calls) == 3
    assert not any(kwargs["render"] for _, kwargs in inline_pool.calls)
    assert not inline_pool.writes
    assert not any(env.render_calls for env in inline_pool.environments)


@pytest.mark.parametrize("failure", ["scoring", "encoding", "changed_replay"])
def test_worker_failure_leaves_no_completion_metrics(inline_pool, tmp_path, failure):
    state = tiny_state()
    if failure == "scoring":
        inline_pool.fail_scoring_seed = 7
    elif failure == "encoding":
        inline_pool.fail_render_seed = 7
    else:
        inline_pool.corrupt_replay_seed = 7
    messages = {"scoring": "scoring failed", "encoding": "encoder failed",
                "changed_replay": "replay differs"}
    with pytest.raises(RuntimeError, match=messages[failure]):
        parallel.evaluate_parallel(
            state, state["config"], "cpu", output=tmp_path, seeds=[9, 2, 7, 6],
            workers=2, threads=1, save_videos=True, progress=False,
        )
    assert not (tmp_path / "metrics.json").exists()
    assert inline_pool.pools[0].closed
    assert all(env.closed for env in inline_pool.environments)
    if failure == "scoring":
        assert not any(kwargs["render"] for _, kwargs in inline_pool.calls)
    else:
        assert read_episode(tmp_path, 7) == inline_pool.scores[7]

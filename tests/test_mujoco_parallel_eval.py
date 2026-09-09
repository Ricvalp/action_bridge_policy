from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from action_bridge.eval.mujoco_online import parallel


def worker_result(seed, outcomes, *, offset=0):
    return {
        "summary": {
            "attempted_episodes": len(outcomes),
            "successful_episodes": sum(outcomes),
            "failure_counts": {"time_limit": len(outcomes) - sum(outcomes)},
        },
        "episodes": [
            {"episode_index": offset + index, "seed": seed + index, "success": outcome}
            for index, outcome in enumerate(outcomes)
        ],
        "checkpoint_identifier": "sha256:example",
        "actions_per_plan": 8,
    }


def arguments(tmp_path, *extra):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    return [
        "--checkpoint",
        str(checkpoint),
        "--trusted-checkpoint",
        "--run-dir",
        str(tmp_path / "run"),
        "--actions-per-plan",
        "8",
        *extra,
    ]


@pytest.mark.parametrize(
    "episodes,workers,sizes", [(10, 3, [4, 3, 3]), (2, 8, [1, 1]), (1, 1, [1])]
)
def test_shards_cover_same_disjoint_seeds(episodes, workers, sizes):
    shards = parallel.split_episodes(episodes, workers, 101)
    assert [shard.episodes for shard in shards] == sizes
    assert [
        seed
        for shard in shards
        for seed in range(shard.seed, shard.seed + shard.episodes)
    ] == list(range(101, 101 + episodes))
    assert [
        index
        for shard in shards
        for index in range(shard.episode_index, shard.episode_index + shard.episodes)
    ] == list(range(episodes))


@pytest.mark.parametrize(
    "episodes,workers,seed", [(0, 2, 0), (1, 0, 0), (1, 1, -1), (2, 1, 2**32 - 1)]
)
def test_shards_reject_invalid_capacity_or_seeds(episodes, workers, seed):
    with pytest.raises(ValueError):
        parallel.split_episodes(episodes, workers, seed)


def test_aggregation_weights_episode_counts_not_worker_rates():
    result = parallel.aggregate_results(
        [
            worker_result(103, [False], offset=3),
            worker_result(100, [True, True, False]),
        ],
        episodes=4,
        seed=100,
    )
    assert result["success_rate"] == 0.5  # Not mean([2 / 3, 0]).
    assert result["successful_episodes"] == 2
    assert result["attempted_episodes"] == 4
    assert result["failure_counts"] == {"time_limit": 2}
    assert [episode["seed"] for episode in result["episodes"]] == [100, 101, 102, 103]


def test_worker_exception_is_not_a_zero_success_result():
    result = worker_result(100, [False])
    result["summary"]["failure_counts"] = {"exception": 1}
    with pytest.raises(RuntimeError, match="exceptions"):
        parallel.aggregate_results([result], episodes=1, seed=100)


def test_aggregation_rejects_missing_seeds_and_changed_checkpoint():
    with pytest.raises(ValueError, match="seeds exactly once"):
        parallel.aggregate_results([worker_result(101, [True])], episodes=1, seed=100)
    first, second = worker_result(100, [True]), worker_result(101, [True], offset=1)
    second["checkpoint_identifier"] = "sha256:different"
    with pytest.raises(ValueError, match="checkpoint_identifier"):
        parallel.aggregate_results([first, second], episodes=2, seed=100)


def test_cpu_environment_does_not_hide_gpu_from_renderer_or_parent(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-allocated-by-slurm")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")
    scoring = parallel.worker_environment(1)
    assert scoring["CUDA_VISIBLE_DEVICES"] == ""
    assert scoring["MUJOCO_GL"] == "disable"
    assert "PYOPENGL_PLATFORM" not in scoring
    assert scoring["OMP_NUM_THREADS"] == scoring["OPENBLAS_NUM_THREADS"] == "1"
    renderer = parallel.worker_environment(2, video_backend="egl")
    assert renderer["CUDA_VISIBLE_DEVICES"] == "GPU-allocated-by-slurm"
    assert renderer["MUJOCO_GL"] == "egl"
    assert renderer["PYOPENGL_PLATFORM"] == "egl"
    assert renderer["MKL_NUM_THREADS"] == "2"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-allocated-by-slurm"
    assert os.environ["PYOPENGL_PLATFORM"] == "osmesa"
    assert parallel.worker_environment(1, video_backend="osmesa")["PYOPENGL_PLATFORM"] == "osmesa"


def test_real_spawn_worker_hides_cuda_and_keeps_parent_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-allocated-by-slurm")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=parallel.multiprocessing.get_context("spawn"),
        initializer=parallel._initialize_worker,
        initargs=(1,),
    ) as pool:
        visible = pool.submit(os.getenv, "CUDA_VISIBLE_DEVICES").result(timeout=60)
        backend = pool.submit(os.getenv, "MUJOCO_GL").result(timeout=60)
    assert visible == ""
    assert backend == "disable"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-allocated-by-slurm"


def test_worker_uses_cpu_adapter_and_single_action_backend(tmp_path, monkeypatch):
    import phi_mujoco.evaluation

    from action_bridge.eval.mujoco_online import torch_backend
    from action_bridge.training import mujoco_provenance

    args = parallel._parser().parse_args(arguments(tmp_path))
    args.run_dir.mkdir()
    loaded, configs = [], []
    adapter = SimpleNamespace(
        integration="test",
        checkpoint_identifier="sha256:test",
        actions_per_plan=8,
        metadata=SimpleNamespace(to_json_dict=dict),
        provenance={},
    )

    def load(checkpoint, **kwargs):
        loaded.append(kwargs)
        return adapter

    class Runner:
        def __init__(self, integration, policy, config):
            assert policy is adapter
            configs.append(config)

        def run(self):
            config = configs[-1]
            config.run_directory.mkdir()
            record = {"episode_index": 0, "seed": config.base_seed, "success": True}
            (config.run_directory / "episodes.jsonl").write_text(
                json.dumps(record) + "\n"
            )
            return SimpleNamespace(
                as_dict=lambda: worker_result(config.base_seed, [True])["summary"]
            )

    monkeypatch.setattr(torch_backend, "load_torch_policy_adapter", load)
    monkeypatch.setattr(phi_mujoco.evaluation, "EvaluationRunner", Runner)
    monkeypatch.setattr(mujoco_provenance, "training_provenance", dict)
    monkeypatch.setattr(parallel, "_worker_log", lambda path: nullcontext())
    result = parallel._score_worker(args, parallel.Shard(1, 5, 1, 105))
    assert loaded == [
        {"trusted_checkpoint": True, "device": "cpu", "actions_per_plan": 8}
    ]
    assert configs[0].actions_per_plan == 1
    assert configs[0].record_video is False
    assert result["episodes"][0]["episode_index"] == 5


def test_video_candidates_are_capped_by_outcome():
    records = worker_result(100, [False, True, True, False, True, False])["episodes"]
    selected = parallel.select_video_episodes(records, successes=2, failures=1)
    assert [record["seed"] for record in selected] == [100, 101, 102]
    assert parallel.select_video_episodes(records, successes=0, failures=0) == []


def test_rendered_actual_outcomes_control_labels_and_quotas(tmp_path, monkeypatch):
    args = parallel._parser().parse_args(
        arguments(tmp_path, "--success-videos", "1", "--failure-videos", "1")
    )
    records = worker_result(100, [True, False])["episodes"]

    def render(args, episode):
        return {
            "seed": episode["seed"],
            "success": False,
            "scoring_success": episode["success"],
            "path": "clip.mp4",
        }

    monkeypatch.setattr(parallel, "_render_episode", render)
    videos, errors = parallel.render_selected_videos(args, records)
    assert len(videos) == 1
    assert videos[0]["success"] is False
    assert videos[0]["scoring_success"] is True
    assert errors == []
    assert records[0]["success"] is True  # Rendering cannot change scored outcomes.


def test_video_error_does_not_discard_other_videos(tmp_path, monkeypatch):
    args = parallel._parser().parse_args(
        arguments(tmp_path, "--success-videos", "1", "--failure-videos", "1")
    )

    def render(args, episode):
        if episode["success"]:
            raise RuntimeError("EGL unavailable")
        return {"success": False, "path": "failure.mp4", "seed": episode["seed"]}

    monkeypatch.setattr(parallel, "_render_episode", render)
    videos, errors = parallel.render_selected_videos(
        args, worker_result(100, [True, False])["episodes"]
    )
    assert len(videos) == 1
    assert errors == [{"seed": 100, "message": "EGL unavailable"}]


def test_video_runs_fresh_cpu_inference_process_with_egl(tmp_path, monkeypatch):
    args = parallel._parser().parse_args(arguments(tmp_path, "--max-steps", "12"))
    args.run_dir.mkdir()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "allocated-gpu")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        directory = args.run_dir / "video-reruns" / "seed-100"
        directory.mkdir()
        (directory / "clip.mp4").write_bytes(b"fake video")
        (directory / "episodes.jsonl").write_text(
            json.dumps({"success": False, "video_path": "clip.mp4"})
        )
        (directory / "summary.json").write_text(
            json.dumps({"failure_counts": {"time_limit": 1}})
        )

    monkeypatch.setattr(parallel.subprocess, "run", run)
    video = parallel._render_episode(args, {"seed": 100, "success": True})
    command, kwargs = calls[0]
    assert command[:3] == [
        parallel.sys.executable,
        "-m",
        "action_bridge.eval.mujoco_online",
    ]
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--actions-per-plan") + 1] == "8"
    assert command[command.index("--max-steps") + 1] == "12"
    assert kwargs["env"]["MUJOCO_GL"] == "egl"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "allocated-gpu"
    assert kwargs["stderr"] == subprocess.STDOUT
    assert video == {
        "seed": 100,
        "success": False,
        "scoring_success": True,
        "path": "video-reruns/seed-100/clip.mp4",
    }


@pytest.mark.parametrize(
    "visible,allocated,selected",
    [
        ("", "0", ""),
        ("GPU-uuid", "0", "GPU-uuid"),
        ("0", "2", "0"),
        ("0", "", "0"),
        ("0", "0", "2"),
    ],
)
def test_slurm_egl_rejects_ambiguous_gpu_mapping(visible, allocated, selected):
    environment = {
        "MUJOCO_GL": "egl",
        "SLURM_JOB_ID": "123",
        "CUDA_VISIBLE_DEVICES": visible,
        "SLURM_JOB_GPUS": allocated,
        "MUJOCO_EGL_DEVICE_ID": selected,
    }
    with pytest.raises(RuntimeError, match="cannot confirm the allocated GPU"):
        parallel._check_slurm_egl(environment)
    environment["MUJOCO_GL"] = "osmesa"
    parallel._check_slurm_egl(environment)


def test_slurm_egl_accepts_matching_numeric_gpu_mapping():
    parallel._check_slurm_egl(
        {
            "MUJOCO_GL": "egl",
            "SLURM_JOB_ID": "123",
            "CUDA_VISIBLE_DEVICES": "2",
            "SLURM_JOB_GPUS": "2",
        }
    )


def test_cli_spawns_pool_and_saves_summary(tmp_path, monkeypatch):
    calls = []

    class Pool:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, args, shard):
            future = Future()
            future.set_result(
                worker_result(
                    shard.seed,
                    [shard.index == 0] * shard.episodes,
                    offset=shard.episode_index,
                )
            )
            return future

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", Pool)
    assert (
        parallel.main(
            arguments(
                tmp_path, "--episodes", "5", "--num-workers", "2", "--seed", "100"
            )
        )
        == 0
    )
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["success_rate"] == 3 / 5
    assert summary["selected_videos"] == []
    assert calls[0]["mp_context"].get_start_method() == "spawn"
    assert calls[0]["initializer"] is parallel._initialize_worker
    assert calls[0]["max_workers"] == 2


def test_cli_rejects_existing_directory_without_overwriting(tmp_path, capsys):
    args = arguments(tmp_path)
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "summary.json").write_text("existing")
    assert parallel.main(args) == 2
    assert json.loads(capsys.readouterr().err)["exception_type"] == "FileExistsError"
    assert (directory / "summary.json").read_text() == "existing"


def test_scoring_process_error_fails_cli_without_success_summary(
    tmp_path, monkeypatch, capsys
):
    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, *args):
            future = Future()
            future.set_exception(RuntimeError("simulator initialization failed"))
            return future

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", Pool)
    assert parallel.main(arguments(tmp_path, "--episodes", "1")) == 2
    assert not (tmp_path / "run" / "summary.json").exists()
    assert json.loads(capsys.readouterr().err)["stage"] == "scoring"

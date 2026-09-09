"""Evaluate a checkpoint in parallel CPU simulators, then render a few episodes.

This coordinator runs in its own process, not in the training process. Scoring
workers never use CUDA. Optional video subprocesses retain the job's GPU
visibility so EGL can render, while policy inference remains on the CPU.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Shard:
    index: int
    episode_index: int
    episodes: int
    seed: int


def split_episodes(episodes: int, num_workers: int, seed: int) -> list[Shard]:
    """Assign contiguous, non-overlapping seeds, independent of worker count."""
    if episodes < 1 or num_workers < 1:
        raise ValueError("episodes and num_workers must be positive")
    if seed < 0 or seed + episodes > 2**32:
        raise ValueError("episode seeds must fit in the uint32 range")
    workers = min(episodes, num_workers)
    count, remainder = divmod(episodes, workers)
    shards = []
    offset = 0
    for index in range(workers):
        size = count + (index < remainder)
        shards.append(Shard(index, offset, size, seed + offset))
        offset += size
    return shards


def worker_environment(
    threads: int, *, video_backend: str | None = None
) -> dict[str, str]:
    """Return child settings without modifying the coordinator's environment."""
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = str(threads)
    environment["MUJOCO_GL"] = video_backend or "disable"
    if video_backend is None:
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment.pop("PYOPENGL_PLATFORM", None)
    else:
        environment["PYOPENGL_PLATFORM"] = video_backend
    return environment


def _initialize_worker(threads: int) -> None:
    os.environ.update(worker_environment(threads))
    # Native libraries are imported only after the worker's CUDA/GL settings.
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


@contextmanager
def _worker_log(path: Path) -> Iterator[None]:
    """Capture Python and native-library output inside this worker only."""
    with path.open("x", encoding="utf-8") as stream:
        sys.stdout.flush()
        sys.stderr.flush()
        original = (os.dup(1), os.dup(2))
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            with redirect_stdout(stream), redirect_stderr(stream):
                yield
        finally:
            stream.flush()
            for target, saved in zip((1, 2), original):
                os.dup2(saved, target)
                os.close(saved)


def _read_episodes(directory: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (directory / "episodes.jsonl").read_text().splitlines()
        if line
    ]


def _score_worker(args: argparse.Namespace, shard: Shard) -> dict:
    directory = args.run_dir / f"worker-{shard.index:03d}"
    with _worker_log(directory.with_suffix(".log")):
        from phi_mujoco.evaluation import EvaluationConfig, EvaluationRunner

        from action_bridge.training.mujoco_provenance import training_provenance

        from .torch_backend import load_torch_policy_adapter

        adapter = load_torch_policy_adapter(
            args.checkpoint,
            trusted_checkpoint=args.trusted_checkpoint,
            device="cpu",
            actions_per_plan=args.actions_per_plan,
        )
        result = EvaluationRunner(
            adapter.integration,
            adapter,
            EvaluationConfig(
                episodes=shard.episodes,
                base_seed=shard.seed,
                max_steps=args.max_steps,
                # The adapter executes chunks and must see every observation.
                actions_per_plan=1,
                run_directory=directory,
                record_video=False,
            ),
        ).run()
        records = _read_episodes(directory)
        for record in records:
            record["episode_index"] += shard.episode_index
        return {
            "summary": result.as_dict(),
            "episodes": records,
            "checkpoint_identifier": adapter.checkpoint_identifier,
            "actions_per_plan": adapter.actions_per_plan,
            "online_evaluation": adapter.metadata.to_json_dict(),
            "training_provenance": adapter.provenance,
            "evaluation_provenance": training_provenance(),
        }


def aggregate_results(results: list[dict], *, episodes: int, seed: int) -> dict:
    """Combine episode counts, never average worker success rates."""
    if not results:
        raise ValueError("no evaluation workers returned results")
    records = []
    failures: Counter = Counter()
    successful = attempted = 0
    for result in results:
        summary = result["summary"]
        if summary["successful_episodes"] is None:
            raise ValueError("success-rate evaluation requires binary task success")
        if any(
            "exception" in name.lower() and count
            for name, count in summary["failure_counts"].items()
        ):
            raise RuntimeError("a simulator worker reported episode exceptions")
        worker_records = result["episodes"]
        if any(type(record["success"]) is not bool for record in worker_records):
            raise ValueError("every scored episode must have a binary success outcome")
        if (
            len(worker_records) != summary["attempted_episodes"]
            or sum(record["success"] for record in worker_records)
            != summary["successful_episodes"]
        ):
            raise ValueError("worker episode records disagree with its summary")
        for key in ("checkpoint_identifier", "actions_per_plan"):
            if result[key] != results[0][key]:
                raise ValueError(f"evaluation workers disagree on {key}")
        successful += summary["successful_episodes"]
        attempted += summary["attempted_episodes"]
        failures.update(summary["failure_counts"])
        records.extend(worker_records)
    records.sort(key=lambda record: record["episode_index"])
    if attempted != episodes or [record["seed"] for record in records] != list(
        range(seed, seed + episodes)
    ):
        raise ValueError("evaluation did not cover the requested seeds exactly once")
    return {
        "attempted_episodes": attempted,
        "successful_episodes": successful,
        "success_rate": successful / attempted,
        "episodes": records,
        "failure_counts": dict(failures),
        **{
            key: value
            for key, value in results[0].items()
            if key not in ("summary", "episodes")
        },
    }


def select_video_episodes(
    records: list[dict], successes: int, failures: int
) -> list[dict]:
    """Choose a bounded number of deterministic candidates of each outcome."""
    remaining = {True: successes, False: failures}
    selected = []
    for record in records:
        if remaining[record["success"]] > 0:
            selected.append(record)
            remaining[record["success"]] -= 1
    return selected


def _check_slurm_egl(environment: dict[str, str]) -> None:
    """Do not guess EGL device mappings on a shared GPU node."""
    if environment["MUJOCO_GL"] != "egl" or "SLURM_JOB_ID" not in environment:
        return
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    allocated = environment.get("SLURM_JOB_GPUS", "")
    selected = environment.get("MUJOCO_EGL_DEVICE_ID", visible)
    if (
        not visible.isdecimal()
        or not allocated.isdecimal()
        or int(allocated) != int(visible)
        or selected != visible
    ):
        raise RuntimeError(
            "EGL video rendering cannot confirm the allocated GPU: "
            f"CUDA_VISIBLE_DEVICES={visible!r}, SLURM_JOB_GPUS={allocated!r}, "
            f"MUJOCO_EGL_DEVICE_ID={selected!r}. The pinned Robosuite EGL renderer "
            "requires one numeric GPU ID with unambiguous Slurm device mapping. "
            "Run the EGL smoke check and ask the cluster administrators about "
            "device mapping, or use OSMesa. Success-rate scoring is unaffected."
        )


def _render_episode(args: argparse.Namespace, episode: dict) -> dict:
    directory = args.run_dir / "video-reruns" / f"seed-{episode['seed']}"
    directory.parent.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "action_bridge.eval.mujoco_online",
        "--checkpoint",
        str(args.checkpoint),
        "--trusted-checkpoint",
        "--run-dir",
        str(directory),
        "--device",
        "cpu",
        "--episodes",
        "1",
        "--seed",
        str(episode["seed"]),
        "--actions-per-plan",
        str(args.actions_per_plan),
        "--record-video",
        "--render-width",
        str(args.render_width),
        "--render-height",
        str(args.render_height),
        "--json",
    ]
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    environment = worker_environment(
        args.worker_threads, video_backend=args.video_backend
    )
    _check_slurm_egl(environment)
    # A fresh interpreter is required: MUJOCO_GL is resolved at import time.
    with directory.with_suffix(".log").open("x", encoding="utf-8") as log:
        subprocess.run(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    summary = json.loads((directory / "summary.json").read_text())
    if any(
        "exception" in name.lower() and count
        for name, count in summary["failure_counts"].items()
    ):
        raise RuntimeError("video rerun reported an episode exception")
    rendered = _read_episodes(directory)[0]
    if type(rendered["success"]) is not bool or not rendered["video_path"]:
        raise ValueError("video rerun produced no binary outcome or video")
    path = directory / rendered["video_path"]
    if not path.is_file():
        raise FileNotFoundError(f"video rerun did not save {path}")
    return {
        "seed": episode["seed"],
        "success": rendered["success"],
        "scoring_success": episode["success"],
        "path": path.relative_to(args.run_dir).as_posix(),
    }


def render_selected_videos(
    args: argparse.Namespace, records: list[dict]
) -> tuple[list[dict], list[dict]]:
    remaining = {True: args.success_videos, False: args.failure_videos}
    videos, errors = [], []
    for episode in select_video_episodes(
        records, args.success_videos, args.failure_videos
    ):
        try:
            video = _render_episode(args, episode)
            # Rerendering can change an outcome. Label and limit actual outcomes.
            if remaining[video["success"]] > 0:
                videos.append(video)
                remaining[video["success"]] -= 1
        except Exception as error:  # noqa: BLE001 - videos must not discard valid scores.
            errors.append({"seed": episode["seed"], "message": str(error)})
    return videos, errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trusted-checkpoint", action="store_true")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--actions-per-plan", type=int, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--success-videos", type=int, default=0)
    parser.add_argument("--failure-videos", type=int, default=0)
    parser.add_argument("--video-backend", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.monotonic()
    stage = "configuration"
    try:
        if not args.trusted_checkpoint:
            raise ValueError(
                "--trusted-checkpoint is required for trusted local PyTorch checkpoints"
            )
        if (
            min(
                args.worker_threads,
                args.actions_per_plan,
                args.render_width,
                args.render_height,
            )
            < 1
        ):
            raise ValueError(
                "worker threads, actions per plan, and render dimensions must be positive"
            )
        if min(args.success_videos, args.failure_videos) < 0:
            raise ValueError("video quotas must be non-negative")
        if args.max_steps is not None and args.max_steps < 1:
            raise ValueError("max_steps must be positive")
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
        shards = split_episodes(args.episodes, args.num_workers, args.seed)
        args.run_dir = args.run_dir.expanduser().resolve()
        args.run_dir.mkdir(parents=True, exist_ok=False)
        stage = "scoring"
        with ProcessPoolExecutor(
            max_workers=len(shards),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(args.worker_threads,),
        ) as pool:
            futures = [pool.submit(_score_worker, args, shard) for shard in shards]
            results = [future.result() for future in futures]
        summary = aggregate_results(results, episodes=args.episodes, seed=args.seed)
        summary["score_wall_seconds"] = time.monotonic() - started
        summary["selected_videos"], summary["video_errors"] = render_selected_videos(
            args, summary["episodes"]
        )
        summary.update(
            {
                "checkpoint": str(args.checkpoint),
                "run_directory": str(args.run_dir),
                "num_workers": len(shards),
                "worker_threads": args.worker_threads,
                "base_seed": args.seed,
                "wall_seconds": time.monotonic() - started,
            }
        )
        (args.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    key: summary[key]
                    for key in (
                        "success_rate",
                        "successful_episodes",
                        "attempted_episodes",
                        "run_directory",
                    )
                }
            )
        )
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # noqa: BLE001 - CLI reports simulator failures as JSON.
        print(
            json.dumps(
                {
                    "stage": stage,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

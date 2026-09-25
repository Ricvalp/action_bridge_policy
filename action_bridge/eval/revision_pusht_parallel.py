"""Spawned CPU workers for independent, seeded closed-loop Push-T episodes."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from action_bridge.eval import revision_pusht as serial
from action_bridge.plan_revision import checkpoints


_WORKER = {}
_REPLAY_KEYS = ("actions_raw", "actions_preclipped_raw", "states_raw",
                "success", "env_success", "terminated", "truncated", "episode_length")
_COVERAGE_KEYS = ("rewards", "coverage", "max_coverage", "final_coverage")


def _validate_replay(original, replay, seed):
    for key in _REPLAY_KEYS:
        if original[key] != replay[key]:
            raise RuntimeError(f"Rendered replay differs from scored episode for seed {seed}: {key}")
    # Polygon area calculations can differ at float64 roundoff across workers,
    # even when every action and physical state is identical. Keep trajectory
    # and success checks exact; allow only tiny absolute error in these scores.
    for key in _COVERAGE_KEYS:
        scored, rendered = np.asarray(original[key]), np.asarray(replay[key])
        if (scored.shape != rendered.shape
                or not np.isfinite(scored).all() or not np.isfinite(rendered).all()
                or not np.allclose(scored, rendered, rtol=0, atol=1e-12)):
            raise RuntimeError(f"Rendered replay differs from scored episode for seed {seed}: {key}")


@contextmanager
def _worker_environment(threads):
    settings = {name: str(threads) for name in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    settings.update(CUDA_VISIBLE_DEVICES="",
                    SDL_VIDEODRIVER=os.environ.get("SDL_VIDEODRIVER", "dummy"),
                    SDL_AUDIODRIVER=os.environ.get("SDL_AUDIODRIVER", "dummy"),
                    MPLBACKEND=os.environ.get("MPLBACKEND", "Agg"))
    previous = {name: os.environ.get(name) for name in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _initialize_worker(payload, output, completion_id, threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _WORKER.update(payload=payload, policy=checkpoints.restore_policy(payload, "cpu"),
                   output=Path(output), completion_id=completion_id)


def _evaluate_seed(seed, render=False, save_videos=False, save_gifs=False):
    payload = _WORKER["payload"]
    path = _WORKER["output"] / f"episode-seed{seed}.json"
    original = json.loads(path.read_text()) if render else None
    try:
        serial.evaluate(_WORKER["policy"], payload["config"], payload["metadata"],
                        payload["dependencies"], "cpu", output=_WORKER["output"],
                        completion_id=_WORKER["completion_id"], seeds=[seed],
                        render=render, save_videos=save_videos, save_gifs=save_gifs,
                        progress=False, write_metrics=False)
        if render:
            replay = json.loads(path.read_text())
            _validate_replay(original, replay, seed)
            original.update({key: replay[key] for key in ("video", "gif") if key in replay})
    finally:
        # Preserve scoring timings and diagnostics even if recording failed.
        if original is not None:
            path.write_text(json.dumps(original))
    return seed


def _run_seeds(pool, seeds, *, description, progress, render=False,
               save_videos=False, save_gifs=False):
    futures = []
    try:
        for seed in seeds:
            futures.append(pool.submit(_evaluate_seed, seed, render, save_videos, save_gifs))
        with tqdm(total=len(seeds), desc=description, unit="episode", disable=not progress) as bar:
            for future in as_completed(futures):
                future.result()
                bar.update(1)
    except BaseException:
        for future in futures:
            future.cancel()
        raise


def evaluate_parallel(state, config, device, *, output, completion_id=2, seeds,
                      workers, threads=1, save_videos=True, save_gifs=False, progress=True):
    """Score every seed before recording the first two successes and failures.

    Workers restore only inference state and own their environments and RNGs.
    ``metrics.json`` is published only after scoring and requested media succeed.
    The output directory must not contain results for this evaluation already.
    """
    if torch.device(device).type != "cpu":
        raise ValueError("Parallel Push-T evaluation supports CPU devices only")
    if workers < 1 or threads < 1:
        raise ValueError("workers and threads must be positive")
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Evaluation seeds must be nonnegative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be unique")
    serial.validate_evaluation_completion(config, completion_id)
    output = Path(output)
    targets = [output / "metrics.json", *(output / f"episode-seed{seed}.json" for seed in seeds)]
    if any(path.exists() for path in targets):
        raise FileExistsError("Parallel evaluation requires fresh episode and metrics paths")
    output.mkdir(parents=True, exist_ok=True)
    payload = {"config": dict(config), **{key: state[key] for key in ("metadata", "dependencies", "ema")}}
    with _worker_environment(threads):
        with ProcessPoolExecutor(max_workers=min(workers, len(seeds)),
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize_worker,
                                 initargs=(payload, output, completion_id, threads)) as pool:
            _run_seeds(pool, seeds, description=f"Evaluating {config['method']}", progress=progress)
            episodes = [json.loads((output / f"episode-seed{seed}.json").read_text()) for seed in seeds]
            if save_videos or save_gifs:
                selected, counts = [], {True: 0, False: 0}
                for episode in episodes:
                    bucket = bool(episode["success"])
                    if counts[bucket] < 2:
                        selected.append(episode["seed"])
                        counts[bucket] += 1
                _run_seeds(pool, selected, description="Recording videos", progress=progress,
                           render=True, save_videos=save_videos, save_gifs=save_gifs)
                episodes = [json.loads((output / f"episode-seed{seed}.json").read_text()) for seed in seeds]
    metrics = serial.summarize_episodes(episodes, config, state["metadata"],
                                       completion_id=completion_id, seeds=seeds)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics

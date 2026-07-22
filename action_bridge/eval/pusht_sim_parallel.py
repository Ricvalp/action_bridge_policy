"""Parallel checkpoint-based Push-T simulator evaluation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from action_bridge.config import save_config, to_plain_dict
from action_bridge.eval.pusht_sim import (
    _aggregate_rollout_diagnostics,
    _rollout_summary,
    plot_pusht_sim_contact_parameters,
    plot_pusht_sim_contact_reference,
    plot_pusht_sim_frames,
    plot_pusht_sim_geometric_reference,
    plot_pusht_sim_latent_chunk_samples,
    plot_pusht_sim_latent_spread,
    plot_pusht_sim_reference_field_diagnostics,
    plot_pusht_sim_reward_curves,
    plot_pusht_sim_rollouts,
    rollout_pusht_sim_episode,
    save_pusht_sim_gif,
    save_pusht_sim_rollout_artifacts,
    save_pusht_sim_video,
)
from action_bridge.training.common import build_model, resolve_device, save_json, seed_everything


def _episode_shards(episodes: int, num_workers: int) -> list[tuple[int, int]]:
    episodes = max(1, int(episodes))
    num_workers = max(1, min(int(num_workers), episodes))
    base = episodes // num_workers
    extra = episodes % num_workers
    shards = []
    start = 0
    for worker_idx in range(num_workers):
        count = base + (1 if worker_idx < extra else 0)
        if count > 0:
            shards.append((start, count))
            start += count
    return shards


def _load_checkpoint_model(checkpoint: Path, config: Dict[str, Any], device: torch.device):
    raw = torch.load(checkpoint, map_location="cpu")
    model = build_model(config).to(device)
    model.load_state_dict(raw["model_state"])
    model.eval()
    return model


def _run_worker(
    checkpoint: str,
    config: Dict[str, Any],
    device_name: str,
    output_dir: Optional[str],
    start_episode: int,
    episode_count: int,
    worker_idx: int,
    worker_threads: int,
) -> Dict[str, Any]:
    if worker_threads > 0:
        torch.set_num_threads(int(worker_threads))
        try:
            torch.set_num_interop_threads(max(1, min(int(worker_threads), 4)))
        except RuntimeError:
            pass
    config = copy.deepcopy(config)
    eval_cfg = config.setdefault("eval", {})
    seed0 = int(eval_cfg.get("sim_seed", config.get("seed", 0)))
    render_episodes = int(eval_cfg.get("sim_render_episodes", 0))
    device = resolve_device(device_name)
    seed_everything(seed0 + int(start_episode))
    model = _load_checkpoint_model(Path(checkpoint), config, device)

    rollouts = []
    gif_paths: list[str] = []
    video_paths: list[str] = []
    root = Path(output_dir) if output_dir else None
    worker_dir = root / "workers" / f"worker_{worker_idx:02d}" if root is not None else None
    if worker_dir is not None:
        worker_dir.mkdir(parents=True, exist_ok=True)

    for local_idx in range(int(episode_count)):
        episode_idx = int(start_episode) + local_idx
        rollout = rollout_pusht_sim_episode(
            model,
            config,
            device,
            seed=seed0 + episode_idx,
            collect_frames=episode_idx < render_episodes,
        )
        rollouts.append({"episode_index": episode_idx, "rollout": rollout})

        if root is not None:
            save_pusht_sim_rollout_artifacts(rollout, root, episode_idx)
            if bool(eval_cfg.get("sim_save_gifs", True)):
                gif_path = save_pusht_sim_gif(
                    rollout,
                    episode_idx,
                    root / "figures" / "pusht_sim_gifs",
                    fps=float(eval_cfg.get("sim_gif_fps", 10.0)),
                )
                if gif_path is not None:
                    gif_paths.append(gif_path)
            if bool(eval_cfg.get("sim_save_videos", False)):
                video_path = save_pusht_sim_video(
                    rollout,
                    episode_idx,
                    root / "figures" / "pusht_sim_videos",
                    fps=float(eval_cfg.get("sim_video_fps", 10.0)),
                )
                if video_path is not None:
                    video_paths.append(video_path)
            save_json(
                worker_dir / "partial.json" if worker_dir is not None else root / f"worker_{worker_idx:02d}_partial.json",
                {
                    "worker": worker_idx,
                    "start_episode": int(start_episode),
                    "completed_episodes": local_idx + 1,
                    "requested_episodes": int(episode_count),
                    "rollouts": [_rollout_summary(item["rollout"], item["episode_index"]) for item in rollouts],
                },
            )

    return {"worker": worker_idx, "rollouts": rollouts, "gif_paths": gif_paths, "video_paths": video_paths}


def _aggregate_metrics(rollouts: list[Dict[str, Any]]) -> Dict[str, float]:
    success = np.asarray([item["success"] for item in rollouts], dtype=np.float32)
    max_rewards = np.asarray([item["max_reward"] for item in rollouts], dtype=np.float32)
    final_rewards = np.asarray([item["final_reward"] for item in rollouts], dtype=np.float32)
    lengths = np.asarray([item["episode_length"] for item in rollouts], dtype=np.float32)
    replans = np.asarray([item["num_replans"] for item in rollouts], dtype=np.float32)
    path_kl = np.asarray([item["path_KL_energy"] for item in rollouts], dtype=np.float32)
    boundary = np.asarray([item["chunk_boundary_discontinuity"] for item in rollouts], dtype=np.float32)
    metrics = {
        "sim_success_rate": float(success.mean()) if success.size else 0.0,
        "sim_max_reward": float(max_rewards.mean()) if max_rewards.size else 0.0,
        "sim_final_reward": float(final_rewards.mean()) if final_rewards.size else 0.0,
        "sim_episode_length": float(lengths.mean()) if lengths.size else 0.0,
        "sim_num_replans": float(replans.mean()) if replans.size else 0.0,
        "sim_path_KL_energy": float(path_kl.mean()) if path_kl.size else 0.0,
        "sim_chunk_boundary_discontinuity": float(boundary.mean()) if boundary.size else 0.0,
        "sim_episodes": float(len(rollouts)),
    }
    metrics.update(_aggregate_rollout_diagnostics(rollouts))
    return metrics


def _write_aggregate_artifacts(
    rollouts: list[Dict[str, Any]],
    metrics: Dict[str, float],
    gif_paths: list[str],
    video_paths: list[str],
    config: Dict[str, Any],
    output_dir: Optional[Path],
) -> None:
    if output_dir is None:
        return
    eval_cfg = config.get("eval", {})
    figures = output_dir / "figures"
    plot_pusht_sim_rollouts(rollouts, figures / "pusht_sim_rollouts.png", max_episodes=int(eval_cfg.get("sim_plot_episodes", 8)))
    plot_pusht_sim_reward_curves(rollouts, figures / "pusht_sim_rewards.png")
    plot_pusht_sim_frames(
        rollouts,
        figures / "pusht_sim_frames.png",
        max_episodes=max(1, int(eval_cfg.get("sim_render_episodes", 0))),
        frames_per_episode=int(eval_cfg.get("sim_frames_per_episode", 6)),
    )
    plot_pusht_sim_contact_reference(
        rollouts,
        figures / "pusht_sim_contact_reference.png",
        config,
        max_panels=int(eval_cfg.get("sim_contact_panels", 6)),
        grid_size=int(eval_cfg.get("sim_contact_grid_size", 90)),
    )
    plot_pusht_sim_contact_parameters(
        rollouts,
        figures / "pusht_sim_contact_parameters.png",
        max_replans=eval_cfg.get("sim_contact_parameter_max_replans", None),
    )
    plot_pusht_sim_latent_chunk_samples(
        rollouts,
        figures / "pusht_sim_latent_chunk_samples.png",
        max_panels=int(eval_cfg.get("sim_latent_sample_panels", 6)),
    )
    plot_pusht_sim_latent_spread(rollouts, figures / "pusht_sim_latent_spread.png")
    plot_pusht_sim_geometric_reference(
        rollouts,
        figures / "pusht_sim_geometric_reference.png",
        max_panels=int(eval_cfg.get("sim_geometric_panels", 8)),
    )
    plot_pusht_sim_reference_field_diagnostics(rollouts, figures / "pusht_sim_reference_field_diagnostics.png")
    save_json(
        output_dir / "metrics" / "pusht_sim_metrics.json",
        {
            **metrics,
            "gif_paths": gif_paths,
            "video_paths": video_paths,
            "rollouts": [_rollout_summary(item, idx) for idx, item in enumerate(rollouts)],
        },
    )


def evaluate_pusht_sim_checkpoint_parallel(
    checkpoint: Path,
    config: Dict[str, Any],
    device_name: str,
    output_dir: Optional[Path] = None,
    num_workers: int = 1,
    worker_threads: int = 1,
) -> Dict[str, float]:
    """Evaluate a checkpoint by running Push-T episodes across CPU worker processes."""

    eval_cfg = config.get("eval", {})
    episodes = max(1, int(eval_cfg.get("sim_episodes", 50)))
    shards = _episode_shards(episodes, num_workers)
    config_plain = to_plain_dict(config)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config_plain, output_dir / "pusht_sim_config.json")

    rollouts_by_index: dict[int, Dict[str, Any]] = {}
    gif_paths: list[str] = []
    video_paths: list[str] = []
    with ProcessPoolExecutor(max_workers=len(shards)) as executor:
        futures = [
            executor.submit(
                _run_worker,
                str(checkpoint),
                config_plain,
                device_name,
                str(output_dir) if output_dir is not None else None,
                start,
                count,
                worker_idx,
                int(worker_threads),
            )
            for worker_idx, (start, count) in enumerate(shards)
        ]
        iterator = tqdm(as_completed(futures), total=len(futures), desc="Push-T sim workers", unit="worker", dynamic_ncols=True)
        for future in iterator:
            result = future.result()
            for item in result["rollouts"]:
                rollouts_by_index[int(item["episode_index"])] = item["rollout"]
            gif_paths.extend(result.get("gif_paths", []))
            video_paths.extend(result.get("video_paths", []))
            if output_dir is not None:
                ordered_partial = [rollouts_by_index[idx] for idx in sorted(rollouts_by_index)]
                save_json(
                    output_dir / "metrics" / "pusht_sim_partial.json",
                    {
                        "completed_episodes": len(ordered_partial),
                        "requested_episodes": episodes,
                        "rollouts": [_rollout_summary(item, idx) for idx, item in enumerate(ordered_partial)],
                    },
                )

    rollouts = [rollouts_by_index[idx] for idx in sorted(rollouts_by_index)]
    metrics = _aggregate_metrics(rollouts)
    metrics["sim_num_workers"] = float(len(shards))
    _write_aggregate_artifacts(rollouts, metrics, gif_paths, video_paths, config_plain, output_dir)
    return metrics

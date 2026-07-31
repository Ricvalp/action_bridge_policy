"""Generate interactive HTML views of RLBench episodes and training batches."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Dict, List, Sequence

import numpy as np
from torch.utils.data import DataLoader, Subset

from action_bridge.data.rlbench_cache import (
    RLBenchCacheStore,
    build_cache_keys,
    build_variation_keys,
    discover_tasks,
)
from action_bridge.data.rlbench_dataset import RLBenchDataset
from action_bridge.eval.rlbench_visualization import (
    episode_animation_figure,
    training_batch_figure,
    write_figure_html,
)


_PREFERRED_TASKS = (
    "open_fridge",
    "stack_cups",
    "basketball_in_hoop",
    "insert_onto_square_peg",
)


def _default_tasks(cache_root: Path, count: int) -> List[str]:
    available = discover_tasks(cache_root)
    preferred = [task for task in _PREFERRED_TASKS if task in available]
    if len(preferred) >= int(count):
        return preferred[: int(count)]
    remaining = [task for task in available if task not in preferred]
    if remaining:
        positions = np.linspace(0, len(remaining) - 1, int(count) - len(preferred))
        preferred.extend(remaining[int(round(position))] for position in positions)
    return preferred[: int(count)]


def _episode_visualizations(
    cache_root: Path,
    output_dir: Path,
    tasks: Sequence[str],
    *,
    variation: int,
    episodes_per_task: int,
    seed: int,
    frame_stride: int,
    max_animation_frames: int,
    animation_point_count: int,
    chunk_horizon: int,
    embed_plotly: bool,
) -> List[Path]:
    rng = np.random.default_rng(int(seed))
    outputs = []
    for task in tasks:
        task_keys = build_variation_keys(cache_root, task)
        matching = [key for key in task_keys if key.variation == int(variation)]
        selected_key = matching[0] if matching else task_keys[0]
        store = RLBenchCacheStore([selected_key], keep_open=True)
        episode_ids = store.list_episode_ids(0)
        count = min(int(episodes_per_task), len(episode_ids))
        selected_episodes = rng.choice(episode_ids, size=count, replace=False)
        for episode_id in selected_episodes:
            figure = episode_animation_figure(
                store,
                0,
                int(episode_id),
                frame_stride=frame_stride,
                max_frames=max_animation_frames,
                chunk_horizon=chunk_horizon,
                point_count=animation_point_count,
            )
            path = (
                output_dir
                / "episodes"
                / f"{task}_variation{selected_key.variation}_episode{int(episode_id)}.html"
            )
            write_figure_html(figure, path, embed_plotly=embed_plotly)
            outputs.append(path)
            print(f"Wrote {path}", flush=True)
        store.close()
    return outputs


def _representative_indices(
    dataset: RLBenchDataset,
    count: int,
    *,
    seed: int,
) -> List[int]:
    by_task: Dict[str, List[int]] = defaultdict(list)
    for index, window in enumerate(dataset.indices):
        task = dataset.keys[window.variation_index].task
        by_task[task].append(index)
    rng = np.random.default_rng(int(seed))
    selections: Dict[str, List[int]] = {}
    per_task = int(np.ceil(int(count) / max(1, len(by_task))))
    for task, indices in by_task.items():
        if len(indices) <= per_task:
            selections[task] = list(indices)
        else:
            positions = np.linspace(0, len(indices) - 1, per_task + 2)[1:-1]
            jitter = rng.integers(-max(1, len(indices) // 30), max(2, len(indices) // 30), size=per_task)
            selected_positions = np.clip(np.rint(positions).astype(int) + jitter, 0, len(indices) - 1)
            selections[task] = [indices[int(position)] for position in selected_positions]
    output = []
    tasks = sorted(selections)
    while len(output) < int(count) and any(selections.values()):
        for task in tasks:
            if selections[task] and len(output) < int(count):
                output.append(selections[task].pop(0))
    return output


def _batch_visualizations(
    cache_root: Path,
    output_dir: Path,
    tasks: Sequence[str],
    *,
    split: str,
    variation: int,
    obs_history: int,
    action_history: int,
    chunk_horizon: int,
    action_representation: str,
    point_count: int,
    batch_size: int,
    num_batches: int,
    seed: int,
    embed_plotly: bool,
) -> List[Path]:
    dataset = RLBenchDataset(
        str(cache_root),
        split=split,
        tasks=tasks,
        variation_ids=[int(variation)],
        obs_history=obs_history,
        action_history=action_history,
        chunk_horizon=chunk_horizon,
        action_offset=1,
        action_representation=action_representation,
        point_count=point_count,
        point_sampling="random",
        point_sampling_seed=seed,
        include_rgb=True,
        include_mask_id=True,
        pad_episode_starts=True,
    )
    count = min(len(dataset), int(batch_size) * int(num_batches))
    indices = _representative_indices(dataset, count, seed=seed)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )
    outputs = []
    offset = 0
    for batch_number, batch in enumerate(loader):
        batch_indices = indices[offset : offset + int(batch["future_actions"].shape[0])]
        offset += len(batch_indices)
        metadata = []
        for dataset_index in batch_indices:
            window = dataset.indices[dataset_index]
            key = dataset.keys[window.variation_index]
            metadata.append(
                {
                    "task": key.task,
                    "variation": key.variation,
                    "episode_id": window.episode_id,
                    "time_index": window.time_index,
                }
            )
        figure = training_batch_figure(
            batch,
            metadata,
            action_representation=action_representation,
        )
        path = (
            output_dir
            / "batches"
            / f"training_batch_{batch_number:02d}_{action_representation}.html"
        )
        write_figure_html(figure, path, embed_plotly=embed_plotly)
        outputs.append(path)
        print(f"Wrote {path}", flush=True)
    dataset.close()
    return outputs


def _write_index(output_dir: Path, files: Sequence[Path], config: Dict) -> None:
    relative_files = [str(path.relative_to(output_dir)) for path in files]
    (output_dir / "visualization_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    links = "\n".join(
        f'<li><a href="{path}">{path}</a></li>'
        for path in relative_files
    )
    (output_dir / "index.html").write_text(
        (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>RLBench visualizations</title>"
            "<style>body{font:16px system-ui;max-width:900px;margin:40px auto;"
            "padding:0 20px;color:#172033}li{margin:10px 0}</style></head>"
            f"<body><h1>RLBench visualizations</h1><ul>{links}</ul></body></html>"
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default="data/rlbench_cache")
    parser.add_argument("--output-dir")
    parser.add_argument("--tasks", nargs="*", default=())
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-animation-frames", type=int, default=70)
    parser.add_argument("--animation-point-count", type=int, default=1024)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--obs-history", type=int, default=2)
    parser.add_argument("--action-history", type=int, default=2)
    parser.add_argument("--chunk-horizon", type=int, default=16)
    parser.add_argument(
        "--action-representation",
        choices=("absolute", "delta_xyz"),
        default="absolute",
    )
    parser.add_argument("--batch-point-count", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--embed-plotly", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root).expanduser().resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(f"RLBench cache not found: {cache_root}")
    tasks = list(args.tasks) or _default_tasks(cache_root, args.num_tasks)
    if not tasks:
        raise RuntimeError(f"No RLBench tasks found under {cache_root}.")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs")
        / "rlbench_visualizations"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["cache_root"] = str(cache_root)
    config["output_dir"] = str(output_dir)
    config["tasks"] = tasks
    files = _episode_visualizations(
        cache_root,
        output_dir,
        tasks,
        variation=args.variation,
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
        frame_stride=args.frame_stride,
        max_animation_frames=args.max_animation_frames,
        animation_point_count=args.animation_point_count,
        chunk_horizon=args.chunk_horizon,
        embed_plotly=args.embed_plotly,
    )
    files.extend(
        _batch_visualizations(
            cache_root,
            output_dir,
            tasks,
            split=args.split,
            variation=args.variation,
            obs_history=args.obs_history,
            action_history=args.action_history,
            chunk_horizon=args.chunk_horizon,
            action_representation=args.action_representation,
            point_count=args.batch_point_count,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            seed=args.seed,
            embed_plotly=args.embed_plotly,
        )
    )
    _write_index(output_dir, files, config)
    print(f"RLBench visualization index: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()

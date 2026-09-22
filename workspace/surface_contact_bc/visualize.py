"""Inspect a collected demo or generate a fresh scripted-expert episode."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import new_run, sha256, write_json
from .data import load_episodes
from .env import Context, rollout_expert, sample_context
from .evaluation import episode_metrics
from .plotting import plot_episode, save_rollout_gif


def visualize_episode(output=None, *, dataset=None, split="train", episode=0,
                      seed=0, gif=False, fps=15):
    if fps < 1:
        raise ValueError("GIF fps must be positive.")
    if dataset is None:
        context = sample_context(seed)
        demo = rollout_expert(context)
        source = dict(kind="fresh_expert", context_seed=seed)
        title = f"Scripted expert, seed {seed}"
    else:
        episodes = load_episodes(dataset, split)
        if not 0 <= episode < len(episodes):
            raise ValueError(f"Episode index must be between 0 and {len(episodes) - 1}.")
        demo = episodes[episode]
        context = Context.from_dict(demo["context"])
        source = dict(kind="dataset_episode", dataset=str(Path(dataset).resolve()),
                      manifest_sha256=sha256(Path(dataset) / "manifest.json"),
                      split=split, episode=episode, episode_id=demo["id"])
        title = f"Expert demo: {split}, episode {episode}"
    output = Path(output) if output else new_run("expert-view")
    output.mkdir(parents=True, exist_ok=False)
    # Flatten simulator diagnostics to the same non-pickled format as eval traces.
    trace = dict(observations=demo["observations"], actions=demo["actions"], **demo["infos"])
    trace["replan"] = np.ones(len(trace["actions"]), dtype=bool)
    trace["impulse"] = np.zeros(len(trace["actions"]))
    metrics = episode_metrics(trace, context)
    np.savez_compressed(output / "episode.npz", **trace)
    write_json(output / "context.json", context.to_dict())
    write_json(output / "metrics.json", metrics)
    write_json(output / "source.json", source)
    plot_episode(trace, context, output, title=title)
    if gif:
        save_rollout_gif(trace, context, output / "rollout.gif", fps=fps)
    return output, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", type=Path, help="Read an existing collection")
    source.add_argument("--seed", type=int, default=0, help="Generate this expert context (default: 0)")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Fresh directory; default is a timestamped runs/ directory")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args(argv)
    output, metrics = visualize_episode(**vars(args))
    print(json.dumps(dict(output=str(output), task_success=metrics["task_success"],
                          clean_success=metrics["clean_success"],
                          peak_normal_force=metrics["peak_normal_force"]), indent=2))


if __name__ == "__main__":
    main()

"""Run the first action-bridge baseline comparison."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_integer("epochs", 5, "Training epochs per variant.")
flags.DEFINE_integer("num_trajectories", 1000, "Number of synthetic expert trajectories.")
flags.DEFINE_integer("trajectory_length", 72, "Expert trajectory length.")
flags.DEFINE_integer("context", 6, "State/action history length.")
flags.DEFINE_integer("horizon", 16, "Predicted action horizon.")
flags.DEFINE_integer("batch_size", 128, "Training batch size.")
flags.DEFINE_integer("rollout_episodes", 96, "Closed-loop rollout episodes.")
flags.DEFINE_integer("replan_every", 4, "Actions to execute before replanning.")
flags.DEFINE_integer("seed", 7, "Random seed.")
flags.DEFINE_string("device", "auto", "Torch device.")
flags.DEFINE_list(
    "configs",
    ["regression", "gaussian_chunk", "bridge_no_energy", "bridge_prev", "bridge_gaussian"],
    "Config module names under action_bridge/configs, without .py.",
)


def main(argv) -> None:
    if len(argv) > 1:
        raise app.UsageError(f"Unknown arguments: {argv[1:]}")

    root = Path(__file__).resolve().parents[1]
    rows = []
    for config_name in FLAGS.configs:
        config_path = root / "action_bridge" / "configs" / f"{config_name}.py"
        run_name = f"compare_{config_name}_n{FLAGS.num_trajectories}_h{FLAGS.horizon}_seed{FLAGS.seed}"
        cmd = [
            sys.executable,
            "-m",
            "action_bridge.train",
            f"--config={config_path}",
            f"--config.run_name={run_name}",
            f"--config.train.epochs={FLAGS.epochs}",
            f"--config.data.num_trajectories={FLAGS.num_trajectories}",
            f"--config.data.trajectory_length={FLAGS.trajectory_length}",
            f"--config.data.context={FLAGS.context}",
            f"--config.data.horizon={FLAGS.horizon}",
            f"--config.train.batch_size={FLAGS.batch_size}",
            f"--config.eval.rollout_episodes={FLAGS.rollout_episodes}",
            f"--config.eval.replan_every={FLAGS.replan_every}",
            f"--config.seed={FLAGS.seed}",
            f"--config.device={FLAGS.device}",
        ]
        subprocess.run(cmd, cwd=root, check=True)
        with (root / "runs" / run_name / "metrics.json").open("r", encoding="utf-8") as f:
            rows.append(json.load(f))

    print("\nSummary")
    for row in rows:
        print(
            f"{row['run_name']:>48}  "
            f"mse={row['action_mse']:.4f}  "
            f"succ={row['rollout_success_rate']:.3f}  "
            f"dist={row['rollout_final_distance']:.3f}  "
            f"disc={row['rollout_chunk_discontinuity']:.3f}  "
            f"jerk={row['rollout_jerk']:.3f}  "
            f"cross={row['rollout_obstacle_cross_rate']:.3f}  "
            f"wrong={row['rollout_wrong_side_fraction']:.3f}  "
            f"nfe={row['network_evals']:.0f}"
        )


if __name__ == "__main__":
    app.run(main)

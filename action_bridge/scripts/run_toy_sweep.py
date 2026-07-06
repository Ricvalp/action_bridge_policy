"""Run a compact delayed-branch comparison sweep."""

from __future__ import annotations

import argparse

from action_bridge.config import apply_overrides, load_config
from action_bridge.training.train_toy import train


SWEEP = [
    ("direct_bc", "toy_delayed_categorical", ["model.policy_type=direct_bc", "model.latent_type=none"]),
    ("autoregressive_bc", "toy_delayed_categorical", ["model.policy_type=autoregressive_bc", "model.latent_type=none"]),
    ("bc_smooth", "toy_delayed_categorical", ["model.policy_type=direct_bc", "model.latent_type=none", "loss.lambda_acc=0.01", "loss.lambda_jerk=0.001"]),
    ("pathkl_no_latent", "toy_delayed_categorical", ["model.latent_type=none"]),
    ("pathkl_categorical", "toy_delayed_categorical", []),
    ("pathkl_continuous", "toy_delayed_continuous", []),
    ("pathkl_categorical_tube", "toy_delayed_categorical", ["loss.tube_training=true"]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-contexts", type=int, default=512)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    for run_name, config_name, local_overrides in SWEEP:
        overrides = [
            f"run_id=sweep_{run_name}",
            f"optim.max_steps={args.max_steps}",
            f"device={args.device}",
            f"data.num_contexts={args.num_contexts}",
        ]
        overrides.extend(local_overrides)
        overrides.extend(args.overrides)
        config = apply_overrides(load_config(config_name), overrides)
        train(config)


if __name__ == "__main__":
    main()

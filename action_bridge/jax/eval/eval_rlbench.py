"""Evaluate a JAX RLBench checkpoint on cached demonstration windows."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import jax
import numpy as np
from ml_collections import ConfigDict

from action_bridge.config import apply_overrides, override_args
from action_bridge.jax.eval.visualization import (
    prediction_chunk_figure,
    write_prediction_chunk_html,
)
from action_bridge.jax.models.config import (
    loss_config_from_config,
    policy_config_from_config,
)
from action_bridge.jax.training.checkpoints import load_checkpoint
from action_bridge.jax.training.data import BatchSource
from action_bridge.jax.training.losses import bridge_loss, direct_bc_loss
from action_bridge.jax.training.train_rlbench import _build_dataset, _build_model


def evaluate(
    checkpoint_path: str | Path,
    *,
    split: str = "val",
    num_batches: int = 16,
    batch_size: int | None = None,
    output_dir: str | Path | None = None,
    overrides=(),
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    config = apply_overrides(ConfigDict(checkpoint["config"]), overrides)
    if batch_size is not None:
        config.optim.batch_size = int(batch_size)
    dataset = _build_dataset(config, split)
    model, policy_config = _build_model(config, dataset)
    loss_config = loss_config_from_config(config)
    params = jax.device_put(checkpoint["params"])
    source = BatchSource(
        dataset,
        batch_size=int(config.optim.batch_size),
        sampling_strategy=str(config.data.sampling_strategy),
        seed=int(config.seed) + 71237,
    )
    is_bridge = str(config.policy_type) == "action_bridge"

    @jax.jit
    def eval_step(batch):
        if is_bridge:
            posterior_output = model.apply(
                {"params": params},
                batch,
                train=False,
                use_posterior=True,
                deterministic_latent=True,
            )
            prior_output = model.apply(
                {"params": params},
                batch,
                train=False,
                use_posterior=False,
                deterministic_latent=True,
            )
            posterior = bridge_loss(
                posterior_output,
                batch,
                loss_config,
                policy_config.bridge,
                np.asarray(10**9, dtype=np.int32),
            )
            prior = bridge_loss(
                prior_output,
                batch,
                loss_config,
                policy_config.bridge,
                np.asarray(10**9, dtype=np.int32),
            )
            metrics = {f"posterior_{key}": value for key, value in posterior.items()}
            metrics.update({f"prior_{key}": value for key, value in prior.items()})
            return metrics, prior_output
        output = model.apply({"params": params}, batch, train=False)
        return direct_bc_loss(
            output, batch, loss_config, policy_config.bridge
        ), output

    collected = []
    last_batch = None
    last_output = None
    for _ in range(int(num_batches)):
        last_batch = jax.tree_util.tree_map(jax.device_put, source())
        metrics, last_output = eval_step(last_batch)
        collected.append(jax.device_get(metrics))
    means = {
        key: float(np.mean([np.asarray(item[key]) for item in collected]))
        for key in collected[0]
    }
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.output_dir) / "eval" / (
            f"{timestamp}_{checkpoint_path.parent.parent.name}_{checkpoint_path.stem}_rlbench_offline"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_step": int(checkpoint["step"]),
                "split": str(split),
                "num_batches": int(num_batches),
                "batch_size": int(config.optim.batch_size),
                "metrics": means,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    figure = prediction_chunk_figure(
        jax.device_get(last_batch),
        jax.device_get(last_output),
        num_examples=min(4, int(config.optim.batch_size)),
    )
    write_prediction_chunk_html(figure, output_dir / "predicted_chunks.html")
    dataset.close()
    print(f"RLBench offline evaluation written to: {output_dir}")
    print(means)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    arguments, unknown = parser.parse_known_args(argv)
    evaluate(
        arguments.checkpoint,
        split=arguments.split,
        num_batches=arguments.num_batches,
        batch_size=arguments.batch_size,
        output_dir=arguments.output_dir,
        overrides=override_args(unknown),
    )


if __name__ == "__main__":
    main()

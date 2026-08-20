"""Train Action Bridge policies on validated low-dimensional PHI Isaac Lab demos."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from action_bridge.config import apply_overrides, load_config, save_config
from action_bridge.eval.eval_isaaclab import evaluate_isaaclab_offline
from action_bridge.training.common import (
    append_csv,
    build_dataset,
    build_model,
    cycle,
    load_config_from_checkpoint,
    make_run_dir,
    move_to_device,
    resolve_device,
    restore_training_state,
    save_json,
    seed_everything,
    tensor_metrics_to_float,
    writable_numpy_collate,
)
from action_bridge.training.isaaclab_online_metadata import (
    configure_isaaclab_online_metadata,
)
from action_bridge.training.losses import model_loss
from action_bridge.training.train_toy import (
    attach_wandb_run_metadata,
    log_wandb_scalars,
    maybe_init_wandb,
    maybe_update_reference_ema,
    save_checkpoint,
    validation_loss,
)


def _offline_max_batches(config) -> int:
    value = int(config.get("eval", {}).get("offline_max_batches", 0))
    if value < 0:
        raise ValueError("eval.offline_max_batches must be non-negative")
    return value


def _first_close_sampling_weights(dataset: Any, weight: float) -> torch.Tensor | None:
    """Return deterministic per-window weights for rare open-to-close commands.

    The corrected Franka expert emits exactly one critical gripper transition per
    episode.  Uniform window sampling makes those starts less than one percent of
    the training stream, even though missing that single command prevents the
    entire lift.  Keep uniform sampling as the default and make the task-aware
    weighting explicit in the resolved experiment config.
    """

    if isinstance(weight, bool) or not math.isfinite(float(weight)) or weight < 1.0:
        raise ValueError("data.first_close_sampling_weight must be finite and >= 1")
    if weight == 1.0:
        return None
    if not hasattr(dataset, "indices") or not hasattr(dataset, "episodes"):
        raise TypeError("first-close sampling requires an Isaac Lab window dataset")

    episodes = {int(episode.episode_index): episode for episode in dataset.episodes}
    weights = torch.ones(len(dataset), dtype=torch.float64)
    transition_count = 0
    for dataset_index, key in enumerate(dataset.indices):
        episode_index = int(key.episode_index)
        time_index = int(key.time_index)
        try:
            actions = episodes[episode_index].arrays.actions
        except KeyError as exc:
            raise ValueError(
                f"window references unknown episode {episode_index}"
            ) from exc
        current_gripper = float(actions[time_index, -1])
        previous_gripper = (
            1.0 if time_index == 0 else float(actions[time_index - 1, -1])
        )
        if previous_gripper >= 0.0 and current_gripper < 0.0:
            weights[dataset_index] = float(weight)
            transition_count += 1
    if transition_count == 0:
        raise ValueError(
            "first-close sampling was requested, but the training split contains "
            "no open-to-close transition"
        )
    return weights


def _training_sampler(dataset: Any, config: Any) -> WeightedRandomSampler | None:
    weight = float(config.data.get("first_close_sampling_weight", 1.0))
    weights = _first_close_sampling_weights(dataset, weight)
    if weights is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 0)))
    return WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def _periodic_offline_eval(model, dataset, config, device, run_dir: Path, step: int, wandb_run):
    output_dir = run_dir / "eval" / f"offline_step_{step:06d}"
    metrics = evaluate_isaaclab_offline(
        model,
        dataset,
        config,
        device,
        output_dir=output_dir,
        max_batches=_offline_max_batches(config),
    )
    row = {"step": step, **metrics}
    append_csv(run_dir / "metrics" / "periodic_offline_metrics.csv", row)
    save_config(config, output_dir / "eval_config.json")
    log_wandb_scalars(wandb_run, metrics, step=step, prefix="offline_eval")
    model.train()
    return metrics


def train(config):
    """Run one reproducible offline Isaac Lab imitation-learning experiment."""

    seed_everything(int(config.get("seed", 0)))
    device = resolve_device(str(config.get("device", "cpu")))
    config.resolved_device = str(device)
    if str(config.get("benchmark")) != "isaaclab_franka_cube_lift":
        raise ValueError(
            "train_isaaclab requires benchmark='isaaclab_franka_cube_lift'"
        )

    train_set = build_dataset(config, split="train")
    normalization_stats = getattr(train_set, "normalization_stats", None)
    if normalization_stats is None:
        raise ValueError("Isaac Lab training requires train-derived normalization statistics")
    config.data.normalization_stats = normalization_stats
    config.data.normalization = train_set.normalization.to_json_dict()
    val_set = build_dataset(config, split="val")
    test_set = build_dataset(config, split="test")
    configure_isaaclab_online_metadata(config, train_set, val_set, test_set)

    run_dir = make_run_dir(config)
    save_config(config, run_dir / "config.json")
    batch_size = int(config.optim.batch_size)
    if batch_size < 1:
        raise ValueError("optim.batch_size must be positive")
    train_sampler = _training_sampler(train_set, config)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=writable_numpy_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=writable_numpy_collate,
    )
    batches = cycle(train_loader)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.optim.lr),
        weight_decay=float(config.optim.get("weight_decay", 0.0)),
    )
    max_steps = int(config.optim.max_steps)
    if max_steps < 1:
        raise ValueError("optim.max_steps must be positive")
    grad_clip = float(config.optim.get("grad_clip", 1.0))
    logging = config.logging
    log_every = int(logging.log_every_steps)
    eval_every = int(logging.eval_every_steps)
    checkpoint_every = int(logging.checkpoint_every_steps)
    full_eval_every = int(logging.get("full_eval_every_steps", 0))
    if log_every < 1 or eval_every < 1:
        raise ValueError("logging.log_every_steps and eval_every_steps must be positive")

    start_step = 1
    best_val = float("inf")
    resume_from = config.get("resume_from")
    if resume_from:
        start_step, best_val = restore_training_state(resume_from, model, optimizer, device)

    wandb_run = maybe_init_wandb(config, run_dir)
    attach_wandb_run_metadata(config, wandb_run)
    save_config(config, run_dir / "config.json")
    try:
        for step in range(start_step, max_steps + 1):
            model.train()
            batch = move_to_device(next(batches), device)
            output = model_loss(model, batch, config.loss, global_step=step)
            optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            maybe_update_reference_ema(model, config)

            if step == 1 or step % log_every == 0:
                row = {"step": step, **tensor_metrics_to_float(output)}
                append_csv(run_dir / "metrics" / "train_metrics.csv", row)
                log_wandb_scalars(wandb_run, row, step=step, prefix="train")
                print(
                    {
                        key: round(value, 6) if isinstance(value, float) else value
                        for key, value in row.items()
                        if key in {"step", "loss", "action_mse", "path_kl", "latent_kl"}
                    },
                    flush=True,
                )

            if step % eval_every == 0 or step == max_steps:
                val_loss = validation_loss(
                    model,
                    val_loader,
                    config,
                    device,
                    max_batches=max(1, len(val_loader)),
                )
                row = {"step": step, "val_loss": val_loss}
                append_csv(run_dir / "metrics" / "val_metrics.csv", row)
                log_wandb_scalars(wandb_run, row, step=step, prefix="val")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        run_dir / "checkpoints" / "best.pt",
                        model,
                        optimizer,
                        config,
                        step,
                        best_val,
                    )
                save_checkpoint(
                    run_dir / "checkpoints" / "latest.pt",
                    model,
                    optimizer,
                    config,
                    step,
                    best_val,
                )

            if full_eval_every > 0 and step % full_eval_every == 0 and step != max_steps:
                _periodic_offline_eval(
                    model,
                    val_set,
                    config,
                    device,
                    run_dir,
                    step,
                    wandb_run,
                )
            if checkpoint_every > 0 and step % checkpoint_every == 0:
                save_checkpoint(
                    run_dir / "checkpoints" / f"step_{step:06d}.pt",
                    model,
                    optimizer,
                    config,
                    step,
                    best_val,
                )

        final_step = max_steps if start_step <= max_steps else start_step - 1
        save_checkpoint(
            run_dir / "checkpoints" / "latest.pt",
            model,
            optimizer,
            config,
            final_step,
            best_val,
        )
        metrics = evaluate_isaaclab_offline(
            model,
            test_set,
            config,
            device,
            output_dir=run_dir,
            max_batches=_offline_max_batches(config),
        )
        save_json(run_dir / "metrics" / "test_metrics.json", metrics)
        log_wandb_scalars(wandb_run, metrics, step=final_step, prefix="test")
        print(f"Run directory: {run_dir}", flush=True)
        print(metrics, flush=True)
        return run_dir
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name", default="isaaclab_franka_cube_lift_continuous"
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--trusted-checkpoint",
        action="store_true",
        help="acknowledge that a resumed PyTorch checkpoint may execute code while loading",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    resume_from = args.resume_from
    if resume_from is None:
        for override in args.overrides:
            if override.startswith("resume_from="):
                resume_from = Path(override.split("=", 1)[1])
                break
    if resume_from is not None:
        if not args.trusted_checkpoint:
            parser.error("--resume-from requires --trusted-checkpoint")
        config = apply_overrides(load_config_from_checkpoint(resume_from), args.overrides)
        config.resume_from = str(resume_from)
        config.resume = True
    else:
        config = apply_overrides(load_config(args.config_name), args.overrides)
    train(config)


if __name__ == "__main__":
    main()


__all__ = ["main", "train"]

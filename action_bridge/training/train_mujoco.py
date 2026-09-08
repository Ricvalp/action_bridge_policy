"""Train low-dimensional Action Bridge policies on validated PHI MuJoCo demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from action_bridge.config import apply_overrides, load_config, save_config
from action_bridge.eval.eval_mujoco import evaluate_mujoco_offline
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
from action_bridge.training.losses import model_loss
from action_bridge.training.mujoco_online_metadata import (
    configure_mujoco_online_metadata,
)
from action_bridge.training.mujoco_provenance import training_provenance
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


def _periodic_offline_eval(
    model, dataset, config, device, run_dir: Path, step: int, wandb_run
):
    output_dir = run_dir / "eval" / f"offline_step_{step:06d}"
    metrics = evaluate_mujoco_offline(
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
    """Run one reproducible offline MuJoCo imitation-learning experiment."""

    seed_everything(int(config.get("seed", 0)))
    device = resolve_device(str(config.get("device", "cpu")))
    config.resolved_device = str(device)
    if str(config.get("benchmark")) != "mujoco":
        raise ValueError("train_mujoco requires benchmark='mujoco'")
    if config.get("resume_from"):
        previous_config = load_config_from_checkpoint(config.resume_from)
        previous_metadata = previous_config.get("online_evaluation")
        if previous_metadata is None:
            raise ValueError(
                "resumed checkpoint is missing MuJoCo online metadata; retrain it"
            )
        # Validate against the actual checkpoint even for programmatic train(config).
        config.online_evaluation = previous_metadata

    train_set = build_dataset(config, split="train")
    normalization_stats = getattr(train_set, "normalization_stats", None)
    if normalization_stats is None:
        raise ValueError(
            "MuJoCo training requires train-derived normalization statistics"
        )
    config.data.normalization_stats = normalization_stats
    config.data.normalization = train_set.normalization.to_dict()
    val_set = build_dataset(config, split="val")
    test_set = (
        build_dataset(config, split="test")
        if train_set.split_plan.test_episode_indices
        else None
    )
    configure_mujoco_online_metadata(config, train_set, val_set, test_set)
    config.provenance = training_provenance()

    run_dir = make_run_dir(config)
    save_config(config, run_dir / "config.json")
    save_json(run_dir / "provenance.json", config.provenance.to_dict())
    print(
        f"{train_set.spec.name}: {len(train_set.split_plan.train_episode_indices)} train / "
        f"{len(train_set.split_plan.val_episode_indices)} validation / "
        f"{len(train_set.split_plan.test_episode_indices)} test episodes; "
        f"state={train_set.obs_dim}, action={train_set.action_dim}, "
        f"horizon={config.chunk_horizon}; device={device}",
        flush=True,
    )
    batch_size = int(config.optim.batch_size)
    if batch_size < 1:
        raise ValueError("optim.batch_size must be positive")
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
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
    validation_max_batches = int(logging.get("validation_max_batches", 0))
    if log_every < 1 or eval_every < 1:
        raise ValueError(
            "logging.log_every_steps and eval_every_steps must be positive"
        )
    if validation_max_batches < 0:
        raise ValueError("logging.validation_max_batches must be non-negative")

    start_step = 1
    best_val = float("inf")
    resume_from = config.get("resume_from")
    if resume_from:
        start_step, best_val = restore_training_state(
            resume_from, model, optimizer, device
        )

    wandb_run = maybe_init_wandb(config, run_dir)
    attach_wandb_run_metadata(config, wandb_run)
    save_config(config, run_dir / "config.json")
    progress = tqdm(
        range(start_step, max_steps + 1),
        desc="Training",
        unit="step",
        disable=not bool(logging.get("progress", True)),
    )
    try:
        for step in progress:
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
                message = str(
                    {
                        key: round(value, 6) if isinstance(value, float) else value
                        for key, value in row.items()
                        if key in {"step", "loss", "action_mse", "path_kl", "latent_kl"}
                    }
                )
                tqdm.write(message)
                progress.set_postfix(loss=f"{row['loss']:.4f}")

            if step % eval_every == 0 or step == max_steps:
                val_loss = validation_loss(
                    model,
                    val_loader,
                    config,
                    device,
                    max_batches=validation_max_batches or len(val_loader),
                )
                row = {"step": step, "val_loss": val_loss}
                append_csv(run_dir / "metrics" / "val_metrics.csv", row)
                log_wandb_scalars(wandb_run, row, step=step, prefix="val")
                tqdm.write(f"step {step}: val_loss={val_loss:.6f}")
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

            if (
                full_eval_every > 0
                and step % full_eval_every == 0
                and step != max_steps
            ):
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
        final_split = "test" if test_set is not None else "val"
        metrics = evaluate_mujoco_offline(
            model,
            test_set if test_set is not None else val_set,
            config,
            device,
            output_dir=run_dir,
            max_batches=_offline_max_batches(config),
        )
        save_json(run_dir / "metrics" / f"{final_split}_metrics.json", metrics)
        log_wandb_scalars(wandb_run, metrics, step=final_step, prefix=final_split)
        print(f"Run directory: {run_dir}", flush=True)
        print(metrics, flush=True)
        return run_dir
    finally:
        progress.close()
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="mujoco_robomimic_square")
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
        config = apply_overrides(
            load_config_from_checkpoint(resume_from), args.overrides
        )
        config.resume_from = str(resume_from)
        config.resume = True
    else:
        config = apply_overrides(load_config(args.config_name), args.overrides)
    train(config)


if __name__ == "__main__":
    main()


__all__ = ["main", "train"]

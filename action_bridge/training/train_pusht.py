"""Train Push-T low-dimensional action bridge policies and baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from action_bridge.config import apply_overrides, load_config, save_config, to_config_dict
from action_bridge.eval.eval_pusht import evaluate_pusht_model
from action_bridge.eval.pusht_sim import evaluate_pusht_sim_model
from action_bridge.training.common import (
    append_csv,
    build_dataset,
    build_model,
    cycle,
    make_run_dir,
    move_to_device,
    resolve_device,
    save_json,
    seed_everything,
    tensor_metrics_to_float,
)
from action_bridge.training.losses import model_loss
from action_bridge.training.train_toy import (
    log_wandb_figures,
    log_wandb_scalars,
    maybe_init_wandb,
    periodic_eval_max_batches,
    save_checkpoint,
    validation_loss,
)


def sync_config_dims(config, dataset) -> None:
    obs_dim = getattr(dataset, "obs_dim", None)
    action_dim = getattr(dataset, "action_dim", None)
    if obs_dim is not None:
        config["obs_dim"] = int(obs_dim)
    if action_dim is not None:
        config["action_dim"] = int(action_dim)
    normalization_stats = getattr(dataset, "normalization_stats", None)
    if normalization_stats is not None:
        config["data"]["normalization_stats"] = normalization_stats


def periodic_pusht_eval_config(config, logging_cfg) -> ConfigDict:
    eval_config = to_config_dict(config)
    if "eval" not in eval_config:
        eval_config.eval = ConfigDict()
    if "inference" not in eval_config:
        eval_config.inference = ConfigDict()
    closed_loop_episodes = logging_cfg.get("full_eval_closed_loop_episodes", None)
    if closed_loop_episodes is not None:
        eval_config.eval.offline_rollout_episodes = int(closed_loop_episodes)
    num_samples = logging_cfg.get("full_eval_num_samples", None)
    if num_samples is not None:
        eval_config.inference.num_samples = int(num_samples)
    return eval_config


def periodic_pusht_sim_eval_config(config, logging_cfg) -> ConfigDict:
    eval_config = to_config_dict(config)
    if "eval" not in eval_config:
        eval_config.eval = ConfigDict()
    eval_config.eval.sim_closed_loop = True
    eval_config.eval.sim_episodes = int(logging_cfg.get("sim_eval_episodes", 20))
    eval_config.eval.sim_max_steps = int(logging_cfg.get("sim_eval_max_steps", 500))
    if logging_cfg.get("sim_eval_n_exec", None) is not None:
        eval_config.eval.sim_n_exec = int(logging_cfg.get("sim_eval_n_exec"))
    if logging_cfg.get("sim_eval_seed", None) is not None:
        eval_config.eval.sim_seed = int(logging_cfg.get("sim_eval_seed"))
    eval_config.eval.sim_render_episodes = int(logging_cfg.get("sim_eval_render_episodes", 0))
    eval_config.eval.sim_plot_episodes = int(logging_cfg.get("sim_eval_plot_episodes", 8))
    eval_config.eval.sim_save_gifs = bool(logging_cfg.get("sim_eval_save_gifs", False))
    eval_config.eval.sim_save_videos = bool(logging_cfg.get("sim_eval_save_videos", False))
    eval_config.eval.sim_collect_contact_diagnostics = bool(logging_cfg.get("sim_eval_collect_contact_diagnostics", False))
    return eval_config


@torch.no_grad()
def run_periodic_pusht_eval(model, datasets: dict, config, device, run_dir: Path, step: int, wandb_run) -> dict:
    logging_cfg = config.get("logging", {})
    split = str(logging_cfg.get("full_eval_split", "val"))
    if split not in datasets:
        raise ValueError(f"logging.full_eval_split must be one of {sorted(datasets)}, got {split!r}.")
    eval_config = periodic_pusht_eval_config(config, logging_cfg)
    output_dir = run_dir / "eval" / f"step_{step:06d}"
    metrics = evaluate_pusht_model(
        model,
        datasets[split],
        eval_config,
        device,
        output_dir=output_dir,
        max_batches=periodic_eval_max_batches(logging_cfg),
    )
    row = {"step": step, "split": split}
    row.update(metrics)
    append_csv(run_dir / "metrics" / "periodic_eval_metrics.csv", row)
    save_config(eval_config, output_dir / "eval_config.json")
    save_json(output_dir / "eval_metadata.json", {"step": step, "split": split, "run_id": config.get("run_id")})
    log_wandb_scalars(wandb_run, metrics, step=step, prefix=f"{split}_eval")
    log_wandb_figures(wandb_run, output_dir / "figures", step=step, prefix=f"{split}_eval")
    model.train()
    return metrics


@torch.no_grad()
def run_periodic_pusht_sim_eval(model, config, device, run_dir: Path, step: int, wandb_run) -> dict:
    logging_cfg = config.get("logging", {})
    eval_config = periodic_pusht_sim_eval_config(config, logging_cfg)
    output_dir = run_dir / "eval" / f"sim_step_{step:06d}"
    try:
        metrics = evaluate_pusht_sim_model(model, eval_config, device, output_dir=output_dir)
    except Exception as exc:
        if not bool(logging_cfg.get("sim_eval_continue_on_error", True)):
            raise
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"sim_eval_error": 1.0}
        save_json(output_dir / "metrics" / "pusht_sim_error.json", {"step": step, "error": repr(exc)})
        append_csv(run_dir / "metrics" / "periodic_sim_eval_errors.csv", {"step": step, "error": repr(exc)})
        print(f"Push-T sim eval failed at step {step}: {exc!r}")
    else:
        row = {"step": step}
        row.update(metrics)
        append_csv(run_dir / "metrics" / "periodic_sim_eval_metrics.csv", row)
    save_config(eval_config, output_dir / "pusht_sim_config.json")
    save_json(output_dir / "eval_metadata.json", {"step": step, "run_id": config.get("run_id"), "kind": "pusht_sim"})
    log_wandb_scalars(wandb_run, metrics, step=step, prefix="sim_eval")
    log_wandb_figures(wandb_run, output_dir / "figures", step=step, prefix="sim_eval")
    model.train()
    return metrics


def train(config):
    seed_everything(int(config.get("seed", 0)))
    device = resolve_device(str(config.get("device", "cpu")))
    config["resolved_device"] = str(device)

    train_set = build_dataset(config, split="train")
    sync_config_dims(config, train_set)
    val_set = build_dataset(config, split="val")
    test_set = build_dataset(config, split="test")
    datasets = {"train": train_set, "val": val_set, "test": test_set}

    run_dir = make_run_dir(config)
    save_config(config, run_dir / "config.json")

    batch_size = int(config.get("optim", {}).get("batch_size", 256))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    batches = cycle(train_loader)

    model = build_model(config).to(device)
    optim_cfg = config.get("optim", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 2e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )
    max_steps = int(optim_cfg.get("max_steps", 200000))
    grad_clip = float(optim_cfg.get("grad_clip", 1.0))
    log_every = int(config.get("logging", {}).get("log_every_steps", 25))
    eval_every = int(config.get("logging", {}).get("eval_every_steps", max(25, log_every)))
    full_eval_every = int(config.get("logging", {}).get("full_eval_every_steps", 0))
    checkpoint_every = int(config.get("logging", {}).get("checkpoint_every_steps", 10000))
    sim_eval_every = int(config.get("logging", {}).get("sim_eval_every_steps", 0))
    sim_eval_enabled = bool(config.get("logging", {}).get("sim_eval_enabled", False))
    best_val = float("inf")
    wandb_run = maybe_init_wandb(config, run_dir)

    try:
        for step in range(1, max_steps + 1):
            model.train()
            batch = move_to_device(next(batches), device)
            out = model_loss(model, batch, config.get("loss", {}), global_step=step)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if step % log_every == 0 or step == 1:
                row = {"step": step}
                row.update(tensor_metrics_to_float(out))
                append_csv(run_dir / "metrics" / "train_metrics.csv", row)
                log_wandb_scalars(wandb_run, row, step=step, prefix="train")
                print({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items() if k in {"step", "loss", "action_mse", "path_kl", "latent_kl"}})

            if step % eval_every == 0 or step == max_steps:
                val = validation_loss(model, val_loader, config, device)
                val_row = {"step": step, "val_loss": val}
                append_csv(run_dir / "metrics" / "val_metrics.csv", val_row)
                log_wandb_scalars(wandb_run, val_row, step=step, prefix="val")
                save_checkpoint(run_dir / "checkpoints" / "latest.pt", model, optimizer, config, step, best_val)
                if val < best_val:
                    best_val = val
                    save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, config, step, best_val)

            if full_eval_every > 0 and step % full_eval_every == 0 and step != max_steps:
                run_periodic_pusht_eval(model, datasets, config, device, run_dir, step, wandb_run)

            if sim_eval_enabled and sim_eval_every > 0 and step % sim_eval_every == 0:
                run_periodic_pusht_sim_eval(model, config, device, run_dir, step, wandb_run)

            if checkpoint_every > 0 and step % checkpoint_every == 0:
                save_checkpoint(run_dir / "checkpoints" / f"step_{step:06d}.pt", model, optimizer, config, step, best_val)

        save_checkpoint(run_dir / "checkpoints" / "latest.pt", model, optimizer, config, max_steps, best_val)
        metrics = evaluate_pusht_model(model, test_set, config, device, output_dir=run_dir)
        save_json(run_dir / "metrics" / "test_metrics.json", metrics)
        log_wandb_scalars(wandb_run, metrics, step=max_steps, prefix="test")
        log_wandb_figures(wandb_run, run_dir / "figures", step=max_steps, prefix="test")
        print(f"Run directory: {run_dir}")
        print(metrics)
        return run_dir
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="pusht_lowdim_continuous")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config_name), args.overrides)
    train(config)


if __name__ == "__main__":
    main()

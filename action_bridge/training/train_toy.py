"""Train toy action bridge policies and baselines."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from action_bridge.config import apply_overrides, flatten_dict, load_config, save_config, to_config_dict, to_plain_dict
from action_bridge.eval.eval_toy import evaluate_toy_model
from action_bridge.training.common import (
    append_csv,
    build_dataset,
    build_model,
    cycle,
    load_config_from_checkpoint,
    make_run_dir,
    move_to_device,
    restore_training_state,
    resolve_device,
    save_json,
    seed_everything,
    tensor_metrics_to_float,
)
from action_bridge.training.losses import model_loss


FIGURE_FILES = [
    "action_chunk_2d_with_t.png",
    "action_error_histograms.png",
    "closed_loop_rollouts.png",
    "continuous_latent_scatter.png",
    "energy_histograms.png",
    "generated_same_history.png",
    "pusht_sim_contact_parameters.png",
    "pusht_sim_contact_reference.png",
    "pusht_sim_frames.png",
    "pusht_sim_latent_chunk_samples.png",
    "pusht_sim_latent_spread.png",
    "pusht_sim_rewards.png",
    "pusht_sim_rollouts.png",
    "receding_horizon_action_rollout.png",
    "wrong_side_go_around_lateral_summary.png",
    "wrong_side_go_around_latent_chunks.png",
]


def save_checkpoint(path: Path, model, optimizer, config, step: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plain_config = to_plain_dict(config)
    wandb_cfg = plain_config.get("logging", {}).get("wandb", {})
    wandb_run_id = wandb_cfg.get("id") if isinstance(wandb_cfg, dict) else None
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": plain_config,
            "step": step,
            "best_metric": best_metric,
            "wandb_run_id": wandb_run_id,
        },
        path,
    )


@torch.no_grad()
def validation_loss(model, loader, config, device, max_batches: int = 4) -> float:
    model.eval()
    values = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        batch = move_to_device(batch, device)
        out = model_loss(model, batch, config.get("loss", {}), global_step=0)
        values.append(float(out["loss"].detach().cpu().item()))
    model.train()
    return float(sum(values) / max(1, len(values)))


def wandb_config(config) -> dict:
    logging_cfg = config.get("logging", {})
    raw = logging_cfg.get("wandb", {})
    if isinstance(raw, bool):
        return {"enabled": raw}
    return to_plain_dict(raw)


def wandb_log_images_enabled(config) -> bool:
    cfg = wandb_config(config)
    return bool(cfg.get("log_images", True))


def _infer_wandb_run_id(run_dir: Path) -> str | None:
    wandb_dir = run_dir / "wandb"
    latest = wandb_dir / "latest-run"
    candidates = []
    if latest.exists():
        try:
            candidates.append(latest.resolve())
        except OSError:
            candidates.append(latest)
    if wandb_dir.exists():
        candidates.extend(sorted(wandb_dir.glob("run-*"), key=lambda path: path.stat().st_mtime, reverse=True))
    for candidate in candidates:
        name = candidate.name
        if name.startswith("run-") and "-" in name:
            return name.rsplit("-", 1)[-1]
    return None


def _max_logged_step(run_dir: Path) -> int:
    max_step = 0
    for path in [
        run_dir / "metrics" / "train_metrics.csv",
        run_dir / "metrics" / "val_metrics.csv",
        run_dir / "metrics" / "periodic_eval_metrics.csv",
        run_dir / "metrics" / "periodic_sim_eval_metrics.csv",
    ]:
        if not path.exists():
            continue
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        max_step = max(max_step, int(float(row.get("step", 0))))
                    except (TypeError, ValueError):
                        continue
        except OSError:
            continue
    return max_step


def attach_wandb_run_metadata(config, wandb_run) -> None:
    if wandb_run is None:
        return
    logging_cfg = config.get("logging", None)
    if not isinstance(logging_cfg, (ConfigDict, dict)):
        logging_cfg = ConfigDict()
        config["logging"] = logging_cfg
    wandb_cfg = logging_cfg.get("wandb", None)
    if isinstance(wandb_cfg, bool):
        wandb_cfg = ConfigDict({"enabled": wandb_cfg})
        logging_cfg["wandb"] = wandb_cfg
    elif not isinstance(wandb_cfg, (ConfigDict, dict)):
        wandb_cfg = ConfigDict()
        logging_cfg["wandb"] = wandb_cfg
    wandb_cfg["id"] = str(wandb_run.id)


def maybe_init_wandb(config, run_dir: Path):
    cfg = wandb_config(config)
    if not bool(cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb logging is enabled, but wandb is not installed. Run `uv sync` from the sandbox.") from exc

    resume_same_run = bool(cfg.get("resume_same_run", True))
    wandb_id = cfg.get("id")
    if not wandb_id and config.get("resume_from") and resume_same_run:
        wandb_id = _infer_wandb_run_id(run_dir)
    resume_mode = None
    if config.get("resume_from") and resume_same_run and wandb_id:
        checkpoint_step = int(config.get("resume_checkpoint_step", 0))
        max_logged_step = _max_logged_step(run_dir)
        if max_logged_step > checkpoint_step and not bool(cfg.get("allow_older_checkpoint_same_run", False)):
            print(
                "W&B same-run resume disabled: checkpoint step "
                f"{checkpoint_step} is older than already logged step {max_logged_step}. "
                "Starting a fresh W&B run for this branch. Set "
                "logging.wandb.allow_older_checkpoint_same_run=true to force the old behavior."
            )
        else:
            resume_mode = cfg.get("resume_mode", "allow")

    init_kwargs = dict(
        project=cfg.get("project") or "action-bridge-policy",
        entity=cfg.get("entity"),
        name=cfg.get("name") or config.get("run_id"),
        group=cfg.get("group"),
        tags=cfg.get("tags") or None,
        mode=cfg.get("mode") or "online",
        dir=str(run_dir),
        config=flatten_dict(config),
    )
    if resume_mode:
        init_kwargs["id"] = wandb_id
        init_kwargs["resume"] = resume_mode
    return wandb.init(**init_kwargs)


def log_wandb_scalars(wandb_run, metrics: dict, step: int, prefix: str) -> None:
    if wandb_run is None:
        return
    payload = {
        f"{prefix}/{key}": value
        for key, value in metrics.items()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    }
    if payload:
        wandb_run.log(payload, step=step)


def log_wandb_figures(wandb_run, figures_dir: Path, step: int, prefix: str, log_images: bool = True) -> None:
    if wandb_run is None or not log_images:
        return
    import wandb

    payload = {}
    for filename in FIGURE_FILES:
        path = figures_dir / filename
        if path.exists():
            payload[f"{prefix}/{path.stem}"] = wandb.Image(str(path))
    if payload:
        wandb_run.log(payload, step=step)


def periodic_eval_config(config, logging_cfg) -> ConfigDict:
    eval_config = to_config_dict(config)
    if "eval" not in eval_config:
        eval_config.eval = ConfigDict()
    if "inference" not in eval_config:
        eval_config.inference = ConfigDict()
    closed_loop_episodes = logging_cfg.get("full_eval_closed_loop_episodes", None)
    if closed_loop_episodes is not None:
        eval_config.eval.closed_loop_episodes = int(closed_loop_episodes)
    num_samples = logging_cfg.get("full_eval_num_samples", None)
    if num_samples is not None:
        eval_config.inference.num_samples = int(num_samples)
    return eval_config


def periodic_eval_max_batches(logging_cfg):
    value = logging_cfg.get("full_eval_max_batches", None)
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


@torch.no_grad()
def run_periodic_eval(model, datasets: dict, config, device, run_dir: Path, step: int, wandb_run) -> dict:
    logging_cfg = config.get("logging", {})
    split = str(logging_cfg.get("full_eval_split", "val"))
    if split not in datasets:
        raise ValueError(f"logging.full_eval_split must be one of {sorted(datasets)}, got {split!r}.")
    eval_config = periodic_eval_config(config, logging_cfg)
    output_dir = run_dir / "eval" / f"step_{step:06d}"
    metrics = evaluate_toy_model(
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
    log_wandb_figures(wandb_run, output_dir / "figures", step=step, prefix=f"{split}_eval", log_images=wandb_log_images_enabled(config))
    model.train()
    return metrics


def train(config):
    seed_everything(int(config.get("seed", 0)))
    device = resolve_device(str(config.get("device", "cpu")))
    config["resolved_device"] = str(device)
    run_dir = make_run_dir(config)
    save_config(config, run_dir / "config.json")

    train_set = build_dataset(config, split="train")
    val_set = build_dataset(config, split="val")
    test_set = build_dataset(config, split="test")
    datasets = {"train": train_set, "val": val_set, "test": test_set}
    train_loader = DataLoader(
        train_set,
        batch_size=int(config.get("optim", {}).get("batch_size", 256)),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_set, batch_size=int(config.get("optim", {}).get("batch_size", 256)), shuffle=False)
    batches = cycle(train_loader)
    model = build_model(config).to(device)
    optim_cfg = config.get("optim", {})
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optim_cfg.get("lr", 3e-4)), weight_decay=float(optim_cfg.get("weight_decay", 0.0)))
    max_steps = int(optim_cfg.get("max_steps", 100000))
    grad_clip = float(optim_cfg.get("grad_clip", 1.0))
    log_every = int(config.get("logging", {}).get("log_every_steps", 25))
    eval_every = int(config.get("logging", {}).get("eval_every_steps", max(25, log_every)))
    full_eval_every = int(config.get("logging", {}).get("full_eval_every_steps", 0))
    checkpoint_every = int(config.get("logging", {}).get("checkpoint_every_steps", 10000))
    start_step = 1
    best_val = float("inf")
    resume_from = config.get("resume_from", None)
    if resume_from:
        start_step, best_val = restore_training_state(Path(resume_from), model, optimizer, device)
        print(f"Resumed training from {resume_from} at step {start_step - 1} with best_metric={best_val:.6g}")
    wandb_run = maybe_init_wandb(config, run_dir)
    attach_wandb_run_metadata(config, wandb_run)
    save_config(config, run_dir / "config.json")
    log_wandb_images = wandb_log_images_enabled(config)

    try:
        for step in range(start_step, max_steps + 1):
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
                if val < best_val:
                    best_val = val
                    save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, config, step, best_val)
                save_checkpoint(run_dir / "checkpoints" / "latest.pt", model, optimizer, config, step, best_val)

            if full_eval_every > 0 and step % full_eval_every == 0 and step != max_steps:
                run_periodic_eval(model, datasets, config, device, run_dir, step, wandb_run)

            if checkpoint_every > 0 and step % checkpoint_every == 0:
                save_checkpoint(run_dir / "checkpoints" / f"step_{step:06d}.pt", model, optimizer, config, step, best_val)

        final_step = max_steps if start_step <= max_steps else start_step - 1
        save_checkpoint(run_dir / "checkpoints" / "latest.pt", model, optimizer, config, final_step, best_val)
        metrics = evaluate_toy_model(model, test_set, config, device, output_dir=run_dir)
        save_json(run_dir / "metrics" / "test_metrics.json", metrics)
        log_wandb_scalars(wandb_run, metrics, step=final_step, prefix="test")
        log_wandb_figures(wandb_run, run_dir / "figures", step=final_step, prefix="test", log_images=log_wandb_images)
        print(f"Run directory: {run_dir}")
        print(metrics)
        return run_dir
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    resume_from = args.resume_from
    if resume_from is None:
        for item in args.overrides:
            if item.startswith("resume_from="):
                resume_from = item.split("=", 1)[1]
                break
    if resume_from:
        config = load_config_from_checkpoint(resume_from)
        config = apply_overrides(config, args.overrides)
        config["resume_from"] = resume_from
        config["resume"] = True
    else:
        if args.config_name is None:
            parser.error("--config-name is required unless --resume-from or resume_from=... is provided.")
        config = apply_overrides(load_config(args.config_name), args.overrides)
    train(config)


if __name__ == "__main__":
    main()

"""Train toy action bridge policies and baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from action_bridge.config import apply_overrides, load_config, save_config
from action_bridge.eval.eval_toy import evaluate_toy_model
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


def save_checkpoint(path: Path, model, optimizer, config, step: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "step": step,
            "best_metric": best_metric,
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


def train(config):
    seed_everything(int(config.get("seed", 0)))
    device = resolve_device(str(config.get("device", "cpu")))
    config["resolved_device"] = str(device)
    run_dir = make_run_dir(config)
    save_config(config, run_dir / "config.yaml")

    train_set = build_dataset(config, split="train")
    val_set = build_dataset(config, split="val")
    test_set = build_dataset(config, split="test")
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
    best_val = float("inf")

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
            print({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items() if k in {"step", "loss", "action_mse", "path_kl", "latent_kl"}})

        if step % eval_every == 0 or step == max_steps:
            val = validation_loss(model, val_loader, config, device)
            append_csv(run_dir / "metrics" / "val_metrics.csv", {"step": step, "val_loss": val})
            save_checkpoint(run_dir / "checkpoints" / "latest.pt", model, optimizer, config, step, best_val)
            if val < best_val:
                best_val = val
                save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, config, step, best_val)

    save_checkpoint(run_dir / "checkpoints" / "latest.pt", model, optimizer, config, max_steps, best_val)
    metrics = evaluate_toy_model(model, test_set, config, device, output_dir=run_dir)
    save_json(run_dir / "metrics" / "test_metrics.json", metrics)
    print(f"Run directory: {run_dir}")
    print(metrics)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config_name), args.overrides)
    train(config)


if __name__ == "__main__":
    main()

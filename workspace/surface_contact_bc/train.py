"""Train either generator on exactly the same offline demonstration windows."""

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import trange

from .artifacts import new_run, provenance, sha256, write_json
from .config import DEFAULTS, METHODS, method_config
from .data import EpisodeWindows, Normalizer
from .policies import build_policy


def tensor_windows(dataset, device):
    """These toy datasets fit in RAM; no workers or disk reads during training."""
    keys = ("obs_hist", "act_hist", "future_actions")
    rows = [dataset[index] for index in range(len(dataset))]
    return {key: torch.as_tensor(np.stack([row[key] for row in rows]),
                                 dtype=torch.float32, device=device) for key in keys}


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters(), strict=True):
        target.lerp_(source, 1 - decay)
    for target, source in zip(ema.buffers(), model.buffers(), strict=True):
        target.copy_(source)


@torch.inference_mode()
def validation_mse(policy, batch, action_scale, seed=10001):
    """Free-running predicted chunks at held-out expert histories, NOT a rollout.

    This is neither teacher forcing inside the chunk nor closed-loop simulation.
    Report world-target MSE in m², averaged over both coordinates and the horizon.
    A single fixed-seed DDIM noise draw per context is used; no best-of-N sampling.
    """
    policy.eval()
    generator = torch.Generator(device=batch["obs_hist"].device).manual_seed(seed)
    squared = 0.0
    count = len(batch["obs_hist"])
    for start in range(0, count, 64):
        part = {key: value[start:start + 64] for key, value in batch.items()}
        predicted, _ = policy.generate(part["obs_hist"], part["act_hist"],
                                       generator=generator)
        squared += (predicted - part["future_actions"]).square().sum().item()
    return squared / batch["future_actions"].numel() * float(action_scale) ** 2


def load_checkpoint(path, device="cpu", reference_only=False):
    payload = torch.load(path, map_location=device, weights_only=True)
    config = dict(payload["config"])
    if reference_only:
        if config["method"] != "action_bridge_contact_frame_full":
            raise ValueError("Reference-only evaluation uses the full bridge checkpoint")
        config["method"] = "reference_only"
    model = build_policy(config, payload["normalizer"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, Normalizer.from_dict(payload["normalizer"]), payload


def train(config, dataset, output=None, reference_checkpoint=None, progress=True):
    config = dict(config)
    if config["method"] == "action_bridge_contact_frame_no_kl":
        config["beta_kl"] = 0.0
    if config["method"] == "reference_only":
        raise ValueError("Do not train reference_only: evaluate the full bridge with --reference-only")
    if min(config[key] for key in ("steps", "batch_size", "val_every", "val_windows", "log_every")) < 1:
        raise ValueError("steps, batch_size, validation and logging intervals must be positive")
    if not 0 <= config["ema_decay"] < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    if not 1 <= config["n_exec"] <= config["horizon"]:
        raise ValueError("n_exec must be between one and horizon")
    torch.set_num_threads(config["threads"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    # Model initialization and diffusion noise must not change minibatch order.
    batch_generator = torch.Generator(device=device).manual_seed(config["seed"])
    output = Path(output) if output else new_run(config["method"])
    output.mkdir(parents=True, exist_ok=False)

    window_args = dict(obs_history=config["obs_history"],
                       action_history=config["action_history"], horizon=config["horizon"])
    training = EpisodeWindows(dataset, split="train", limit=config["train_episodes"], **window_args)
    validation = EpisodeWindows(dataset, split="val", **window_args)
    if not len(validation):
        raise ValueError("The collection needs held-out validation episodes for checkpoint selection")
    validation.normalizer = training.normalizer
    stats = training.normalizer.to_dict()
    train_batch = tensor_windows(training, device)
    val_batch = tensor_windows(validation, device)
    selection = torch.randperm(len(val_batch["obs_hist"]), generator=torch.Generator().manual_seed(10001))
    selection = selection[:config["val_windows"]].to(device)
    val_batch = {key: value[selection] for key, value in val_batch.items()}
    config["train_episodes"] = len(training.episodes)
    config["dataset"] = str(Path(dataset).resolve())
    reference = None
    if config["method"] == "diffusion_residual":
        if reference_checkpoint is None:
            raise ValueError("diffusion_residual needs --reference-checkpoint from the matched full bridge")
        reference, ref_stats, ref_payload = load_checkpoint(reference_checkpoint, device)
        if ref_payload["config"]["method"] != "action_bridge_contact_frame_full":
            raise ValueError("Use the full contact-frame bridge as the shared frozen reference")
        if ref_stats.to_dict() != stats:
            raise ValueError("Reference and residual diffusion must use identical train data/normalization")
        for field in ("horizon", "n_exec", "obs_history", "action_history", "seed", "train_episodes"):
            if ref_payload["config"][field] != config[field]:
                raise ValueError(f"Reference mismatch for {field}")
        for key, filename in (("dataset_manifest_sha256", "manifest.json"),
                              ("dataset_episodes_sha256", "episodes.npz")):
            if ref_payload["provenance"][key] != sha256(Path(dataset) / filename):
                raise ValueError("Reference and residual diffusion must use the exact same collection")
        config["reference_config"] = ref_payload["config"]
        config["reference_checkpoint_sha256"] = sha256(reference_checkpoint)
    model = build_policy(config, stats, reference_policy=reference).to(device)
    if config["method"] == "diffusion_residual":
        indices = torch.linspace(0, len(train_batch["obs_hist"]) - 1,
                                 min(4096, len(train_batch["obs_hist"])), device=device).long()
        model.fit_residual_normalization({key: value[indices] for key, value in train_batch.items()})
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=config["lr"], weight_decay=config["weight_decay"])
    metadata = provenance(dataset)
    metadata.update(parameter_count=sum(p.numel() for p in model.parameters()),
                    trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    train_windows=len(training), validation_windows=len(selection))
    if reference is not None:
        metadata.update(reference_pretraining_updates=ref_payload["config"]["steps"],
                        reference_selected_step=ref_payload["step"],
                        reference_pretraining_parameters=ref_payload["provenance"]["parameter_count"],
                        frozen_reference_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad))
    write_json(output / "config.json", config)
    write_json(output / "provenance.json", metadata)
    best = float("inf")
    started = time.perf_counter()
    progress_bar = trange(1, config["steps"] + 1, desc=config["method"], disable=not progress)
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "loss", "teacher_mse", "path_kl",
                                                   "unroll_mse", "noise_mse", "denoising_mse",
                                                   "action_denoising_mse", "val_action_mse",
                                                   "denoising_steps", "lr", "seconds"))
        writer.writeheader()
        for step in progress_bar:
            model.train()
            indices = torch.randint(len(train_batch["obs_hist"]), (config["batch_size"],),
                                    device=device, generator=batch_generator)
            batch = {key: value[indices] for key, value in train_batch.items()}
            # Same optimizer/schedule for all learned generators.
            lr = config["lr"] * (0.1 + 0.9 * (1 + math.cos(math.pi * (step - 1) / config["steps"])) / 2)
            for group in optimizer.param_groups:
                group["lr"] = lr
            losses = model.loss(batch)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            update_ema(ema, model, min(config["ema_decay"], (1 + step) / (10 + step)))
            validate = step == 1 or step % config["val_every"] == 0 or step == config["steps"]
            val_mse = None
            if validate:
                candidates = config.get("inference_candidates", [config["num_inference_steps"]])
                if not config["method"].startswith("diffusion"):
                    candidates = [None]
                scores = []
                for candidate in candidates:
                    if candidate is not None:
                        ema.set_inference_steps(candidate)
                    scores.append((validation_mse(ema, val_batch, stats["action_scale"]), candidate))
                val_mse, chosen_steps = min(scores, key=lambda item: item[0])
                if chosen_steps is not None:
                    ema.set_inference_steps(chosen_steps)
                    config["num_inference_steps"] = chosen_steps
                payload = dict(config=config, normalizer=stats, model=ema.state_dict(),
                               training_model=model.state_dict(), optimizer=optimizer.state_dict(),
                               step=step, val_action_mse=val_mse, provenance=metadata)
                torch.save(payload, output / "latest.pt")
                if val_mse < best:
                    best = val_mse
                    torch.save(payload, output / "best.pt")
            if validate or step % config["log_every"] == 0:
                row = {key: float(value.detach().mean()) for key, value in losses.items()
                       if key in writer.fieldnames}
                row.update(step=step, val_action_mse=val_mse,
                           denoising_steps=config["num_inference_steps"] if config["method"].startswith("diffusion") else 0,
                           lr=lr, seconds=time.perf_counter() - started)
                writer.writerow(row)
                stream.flush()
                progress_bar.set_postfix(loss=f"{float(losses['loss'].detach()):.4g}", best=f"{best:.3g}")
    summary = dict(method=config["method"], seed=config["seed"], updates=config["steps"],
                   train_episodes=config["train_episodes"], best_val_action_mse=best,
                   training_seconds=time.perf_counter() - started,
                   parameter_count=metadata["parameter_count"], output=str(output))
    write_json(output / "training_summary.json", summary)
    write_json(output / "config.json", config)
    return output, summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output", type=Path)
    result.add_argument("--method", choices=METHODS, default=METHODS[0])
    result.add_argument("--config", type=Path, help="Optional JSON overrides")
    result.add_argument("--reference-checkpoint", type=Path)
    result.add_argument("--skip-eval", action="store_true", help="Train only; evaluation has a separate entry point")
    result.add_argument("--eval-episodes", type=int, default=8)
    result.add_argument("--quiet", action="store_true")
    for key in ("steps", "batch_size", "horizon", "n_exec", "train_episodes", "seed", "threads",
                "hidden_dim", "h_emb_dim", "val_every", "val_windows", "num_inference_steps"):
        result.add_argument("--" + key.replace("_", "-"), type=int)
    for key in ("lr", "beta_kl", "lambda_unroll", "ema_decay"):
        result.add_argument("--" + key.replace("_", "-"), type=float)
    result.add_argument("--device")
    result.add_argument("--prediction-type", choices=("epsilon", "sample"),
                        help="DDIM denoising target: noise or clean action chunks")
    result.add_argument("--unet-channels", nargs="+", type=int)
    result.add_argument("--inference-candidates", nargs="+", type=int,
                        help="Select DDIM steps using validation MSE only (e.g. 10 20 50)")
    return result


def main():
    args = parser().parse_args()
    config = method_config(args.method)
    if args.config:
        config.update(json.loads(args.config.read_text()))
    config.update({key: value for key, value in vars(args).items()
                   if (key in DEFAULTS or key == "inference_candidates") and value is not None})
    config["method"] = args.method
    output, summary = train(config, args.dataset, args.output,
                            args.reference_checkpoint, progress=not args.quiet)
    print(json.dumps(summary, indent=2))
    if not args.skip_eval:
        from .evaluate import evaluate_checkpoint
        evaluate_checkpoint(output / "best.pt", output / "eval", suites=["id"],
                            episodes=args.eval_episodes, device=config["device"])


if __name__ == "__main__":
    main()

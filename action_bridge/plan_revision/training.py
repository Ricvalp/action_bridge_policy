"""Ordinary BC/FM and alternating DSBM, with no task or simulator imports.

The caller supplies common records, optional pairing/validation callbacks, and
optional tracking/visualization. All learned dependencies are checkpointed.
"""
from __future__ import annotations

import json
import shutil
import time
from numbers import Real
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.cache import draw_records, reference_for, refresh_coupling
from action_bridge.plan_revision.completion import LearnedCompletion
from action_bridge.plan_revision.contracts import take
from action_bridge.plan_revision.models import build_policy
from action_bridge.plan_revision.tracking import preserve_rng


def log(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def track_metrics(tracker, step, metrics, prefix):
    if tracker is not None:
        with preserve_rng():
            tracker.log(step, {f"{prefix}/{key}": value for key, value in metrics.items()
                               if key != "step" and isinstance(value, Real)})


def track_images(tracker, visualize, model, step, *, final=False):
    if tracker is None or visualize is None or not tracker.images_due(step, final=final):
        return
    # Neither diagnostic sampling nor the logger may affect the optimizer RNG.
    with preserve_rng(), torch.no_grad():
        training_states = [(module, module.training) for module in model.modules()]
        model.eval()
        try:
            tracker.images(step, visualize(model, step))
        finally:
            for module, training in training_states:
                module.training = training


def completion_config(config):
    return {key: config[key] for key in ("obs_dim", "action_dim", "obs_history", "action_history", "horizon", "robot_dt")} | {
        "hidden_dim": config["reference_hidden_dim"]}


def restore_completion(dependencies, device, *, encoder=None):
    if dependencies.get("completion_encoder_spec", {}).get("kind", "HistoryEncoder") != "HistoryEncoder" and encoder is None:
        raise ValueError("Supply the custom reference encoder specified by this checkpoint")
    model = LearnedCompletion(**dependencies["completion_config"], encoder=encoder).to(device)
    model.load_state_dict(dependencies["completion_state"])
    return model.eval().requires_grad_(False)


def fit_completion(records, validation, config, output, metadata, device, *, encoder=None,
                   tracker=None, visualize=None):
    """Standalone small reference fit; never optimize it through BC/DSBM."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config["seed"])
    model = LearnedCompletion(**completion_config(config), encoder=encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    first = 0
    latest = output / "latest.pt"
    if latest.exists():
        state = checkpoints.load(latest, device)
        if not checkpoints.same_training_config(state["config"], config) or state["metadata"] != metadata:
            raise ValueError("Reference checkpoint/config/dataset mismatch; use another run root")
        if state.get("complete"):
            return state
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first = state["step"]
        checkpoints.restore_rng(state["rng"])
    start = time.perf_counter()
    for step in tqdm(range(first, config["reference_updates"]), desc="reference", initial=first,
                     total=config["reference_updates"]):
        batch = draw_records(records, config["batch_size"], device, endpoint_std=0.)
        loss = model.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        final = step + 1 == config["reference_updates"]
        if (step + 1) % config["log_every"] == 0 or final:
            row = {"step": step + 1, "reference_mse": float(loss.detach()),
                   "grad_norm": float(grad_norm), "lr": optimizer.param_groups[0]["lr"]}
            log(output / "train.jsonl", row)
            track_metrics(tracker, step + 1, row, "train")
        track_images(tracker, visualize, model, step + 1, final=final)
        if (step + 1) % config["checkpoint_every"] == 0:
            checkpoints.save(latest, dict(config=config, metadata=metadata, model=model.state_dict(),
                             optimizer=optimizer.state_dict(), step=step + 1, complete=False))
    model.eval().requires_grad_(False)
    with torch.no_grad():
        # Fixed training-only calibration, no validation statistics in the prior.
        calibration = take(records, torch.arange(min(4096, len(records["future_actions"]))), device)
        residual = model.residuals(calibration)
        variance = residual.square().mean(dim=(0, 1)).clamp_min(config["innovation_floor"])
        prior = model.prior(calibration["obs_hist"], calibration["act_hist"], variance,
                            ridge=config["prior_ridge"], max_rate=config["max_rate"])
        raw_max = prior["raw_rate_max"]
        model.precision_scale.fill_(1. / max(float(torch.as_tensor(raw_max).median()), 1e-6))
        val = take(validation, torch.arange(min(1024, len(validation["future_actions"]))), device)
        diagnostics = {key: float(value.mean()) for key, value in model.diagnostics(val).items()}
    log(output / "validation.jsonl", {"step": config["reference_updates"], **diagnostics})
    track_metrics(tracker, config["reference_updates"], diagnostics, "val")
    payload = dict(config=config, metadata=metadata, model=model.state_dict(),
                   optimizer=optimizer.state_dict(), step=config["reference_updates"], complete=True,
                   completion_config=completion_config(config), completion_state=model.state_dict(),
                   completion_encoder_spec=config.get("reference_encoder_spec", {"kind": "HistoryEncoder"}),
                   innovation_variance=variance.cpu(), diagnostics=diagnostics,
                   training_seconds=time.perf_counter() - start)
    checkpoints.save(latest, payload)
    return checkpoints.load(latest)


def train(records, config, output, metadata, dependencies, device, *, pairer=None, validate=None,
          stop_after=None, encoder=None, completion_encoder=None, tracker=None, visualize=None,
          evaluation_factory=None):
    """Resume exact optimizer/EMA/phase/cache state; `stop_after` is for tests.

    DSBM is Algorithm 1 of Shi et al. (2023): reverse projection, refresh from
    real target endpoints, forward projection, refresh from real sources. The
    opposite EMA (including encoder) is frozen per phase. Repeated cache refresh
    draws new real contexts, not trajectories detached from their conditioning.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if validate is not None and evaluation_factory is not None:
        raise ValueError("Choose either synchronous validation or an asynchronous evaluation worker")
    if config["validation_every"] < 1:
        raise ValueError("validation_every must be positive")
    seed_all(config["seed"])
    policy = build_policy(config, encoder=encoder).to(device)
    ema = checkpoints.frozen_copy(policy)
    bridge = config["method"].startswith("sb_")
    if bridge and config["updates"] != 2 * config["rounds"] * config["phase_updates"]:
        raise ValueError("DSBM total updates must equal rounds * 2 * phase_updates")
    if config["method"] == "fm_local_ot" and pairer is None:
        raise ValueError("Local OT requires an explicit context-distance/compatibility callback")
    modules = {"forward": policy.forward_field, "reverse": policy.reverse_field} if bridge else {"forward": policy}
    optimizers = {key: torch.optim.AdamW(module.parameters(), lr=config["lr"],
                                        weight_decay=config["weight_decay"]) for key, module in modules.items()}
    latest = output / "latest.pt"
    step, current_phase, cache, snapshot = 0, -1, None, None
    best, elapsed, cache_seconds = float("-inf"), 0., 0.
    cache_provenance = None
    resume_rng = None
    if latest.exists():
        state = checkpoints.load(latest, device)
        if not checkpoints.same_training_config(state["config"], config) or state["metadata"] != metadata:
            raise ValueError("Checkpoint/config/cache provenance mismatch; use another run directory")
        if state.get("complete"):
            return state
        policy.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        for key in optimizers:
            optimizers[key].load_state_dict(state["optimizers"][key])
        step, current_phase = state["step"], state["phase"]
        cache, cache_provenance = state["coupling_cache"], state["cache_provenance"]
        if state["opposite_snapshot"] is not None:
            snapshot = checkpoints.frozen_copy(policy)
            snapshot.load_state_dict(state["opposite_snapshot"])
        best, elapsed, cache_seconds = state["best_validation"], state["training_seconds"], state["cache_seconds"]
        # The saved dependency weights/statistics are authoritative after the
        # config and provenance hashes match; do not silently substitute a
        # caller's new completer into an existing coupling cache.
        dependencies = state["dependencies"]
        resume_rng = state["rng"]
    best_report = output / "best_eval.json"
    if (output / "best.pt").exists() and best_report.exists():
        best = max(best, json.loads(best_report.read_text())["success_rate"])
    completion = restore_completion(dependencies, device, encoder=completion_encoder) if "completion_state" in dependencies else None
    if resume_rng is not None:
        # Constructing modules consumes random numbers, even when their weights
        # are immediately replaced. Restore RNG only after all reconstruction.
        checkpoints.restore_rng(resume_rng)
    start = time.perf_counter()

    def payload(complete=False):
        import diffusers
        return dict(config=config, metadata=metadata, dependencies=dependencies,
                    model=policy.state_dict(), ema=ema.state_dict(),
                    optimizers={key: optimizer.state_dict() for key, optimizer in optimizers.items()},
                    step=step, phase=current_phase, outer_round=current_phase // 2 if bridge else 0,
                    direction=("reverse" if current_phase % 2 == 0 else "forward") if bridge else "forward",
                    coupling_cache=cache, cache_provenance=cache_provenance,
                    opposite_snapshot=None if snapshot is None else snapshot.state_dict(),
                    best_validation=best, training_seconds=elapsed + time.perf_counter() - start,
                    cache_seconds=cache_seconds, complete=complete,
                    scheduler=dict(policy.policy.noise_scheduler.config) if config["method"] == "ddim" else None,
                    diffusers_version=diffusers.__version__,
                    parameters=sum(p.numel() for p in policy.parameters()))

    def forward_updates(at_step):
        if not bridge:
            return at_step
        completed_rounds, remainder = divmod(at_step, 2 * config["phase_updates"])
        return completed_rounds * config["phase_updates"] + max(0, remainder - config["phase_updates"])

    def evaluation_payload():
        state = payload(complete=step == config["updates"])
        # Immutable inference snapshot, not an optimizer/coupling-cache copy.
        keys = ("config", "metadata", "dependencies", "ema", "step", "phase", "outer_round",
                "direction", "training_seconds", "cache_seconds", "complete", "scheduler",
                "diffusers_version", "parameters")
        candidate = {key: state[key] for key in keys}
        # SB deploys its forward field even when the reverse field is training.
        candidate.update(training_direction=state["direction"], direction="forward",
                         forward_updates=forward_updates(step))
        return candidate

    def evaluation_finished(eval_step, metrics, checkpoint):
        nonlocal best
        row = {"step": eval_step, **metrics, "forward_updates": forward_updates(eval_step),
               "checkpoint_sha256": checkpoints.digest(checkpoint)}
        log(output / "validation.jsonl", row)
        if tracker is not None:
            tracker.log_evaluation(eval_step, {key: value for key, value in row.items()
                                              if key != "step" and isinstance(value, Real)})
        score = float(metrics["success_rate"])
        tqdm.write(f"Closed-loop step {eval_step}: success={score:.1%}")
        # A random, not-yet-trained SB forward field is diagnostic only.
        if forward_updates(eval_step) > 0 and score > best:
            best = score
            temporary = output / "best.pt.tmp"
            shutil.copyfile(checkpoint, temporary)
            temporary.replace(output / "best.pt")
            temporary_report = best_report.with_suffix(".json.tmp")
            temporary_report.write_text(json.dumps(row, indent=2) + "\n")
            temporary_report.replace(best_report)

    with preserve_rng():
        evaluator = evaluation_factory(evaluation_finished) if evaluation_factory is not None else None
    progress = tqdm(total=config["updates"], initial=step, desc=config["method"])
    try:
        while step < config["updates"]:
            phase = step // config["phase_updates"] if bridge else 0
            direction = "reverse" if bridge and phase % 2 == 0 else "forward"
            if phase != current_phase:
                current_phase = phase
                snapshot = checkpoints.frozen_copy(ema) if bridge and phase > 0 else None
                cache = None
            if bridge:
                if cache is None or step % config["coupling_refresh_every"] == 0:
                    cache, cache_provenance = refresh_coupling(records, config["coupling_records"], device,
                                                              config, completion, snapshot, direction)
                    cache_provenance.update(phase=phase, refresh_step=step,
                                            source_hash=metadata["source_hash"],
                                            snapshot_phase=phase - 1 if snapshot is not None else None)
                    cache_seconds += cache_provenance["cache_seconds"]
                indices = torch.randint(len(cache["x0"]), (config["batch_size"],), device="cpu")
                batch = take(cache, indices, device)
                losses = policy.loss(batch, reference_for(batch, config), direction=direction,
                                     x0=batch["x0"], x1=batch["x1"])
            else:
                batch = draw_records(records, config["batch_size"], device, completion=completion,
                                     executed=config["execute"], source_std=config["source_std"],
                                     endpoint_std=config["endpoint_std"])
                pairing_metrics = {}
                if pairer is not None:
                    batch, pairing_metrics = pairer(batch, completion, device)
                losses = policy.loss(batch)
                losses.update(pairing_metrics)
            optimizer = optimizers[direction]
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(modules[direction].parameters(), config["grad_clip"])
            optimizer.step()
            if bridge:
                checkpoints.update_ema(getattr(ema, direction + "_field"), modules[direction], config["ema_decay"])
            else:
                checkpoints.update_ema(ema, policy, config["ema_decay"])
            step += 1
            progress.update()
            if step % config["log_every"] == 0 or step == config["updates"]:
                row = {key: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                       for key, value in losses.items()}
                row.update(step=step, phase=current_phase, direction=direction,
                           reverse_phase=int(direction == "reverse"),
                           grad_norm=float(grad_norm), lr=optimizer.param_groups[0]["lr"],
                           elapsed_seconds=elapsed + time.perf_counter() - start,
                           cache_seconds=cache_seconds)
                log(output / "train.jsonl", row)
                track_metrics(tracker, step, row, "train")
                progress.set_postfix(loss=f"{row['loss']:.4g}", phase=current_phase)
            if direction == "forward":
                final = step == config["updates"] or (bridge and step % config["phase_updates"] == 0)
                track_images(tracker, visualize, ema, step, final=final)
            milestone = step % config["validation_every"] == 0 or step == config["updates"]
            if evaluator is not None:
                with preserve_rng():
                    if milestone or step % min(100, config["log_every"]) == 0:
                        evaluator.poll()
                    if milestone:
                        if evaluator.busy:
                            log(output / "validation.jsonl", {"step": step, "status": "skipped_busy"})
                        else:
                            evaluator.submit(evaluation_payload())
            elif validate is not None and milestone:
                # Optional synchronous callback for other task attachments.
                if not bridge or (direction == "forward" and step % config["phase_updates"] == 0):
                    with preserve_rng():
                        metrics = validate(ema, step)
                    log(output / "validation.jsonl", {"step": step, **metrics})
                    track_metrics(tracker, step, metrics, "val")
                    score = metrics["success_rate"]
                    if score > best:
                        best = score
                        checkpoints.save(output / "best.pt", payload())
            stopping = stop_after is not None and step >= stop_after
            if step % config["checkpoint_every"] == 0 or milestone or stopping:
                checkpoints.save(latest, payload(complete=step == config["updates"] and evaluator is None))
            if stopping:
                break
        if evaluator is not None:
            with preserve_rng():
                evaluator.finish()
                # A busy worker may have skipped the last interval. Evaluate
                # final weights after optimization, without a training backlog.
                if step == config["updates"] and evaluator.last_submitted_step != step:
                    evaluator.submit(evaluation_payload())
                    evaluator.finish()
            checkpoints.save(latest, payload(complete=step == config["updates"]))
    finally:
        progress.close()
        if evaluator is not None:
            with preserve_rng():
                evaluator.close()
    return checkpoints.load(latest)

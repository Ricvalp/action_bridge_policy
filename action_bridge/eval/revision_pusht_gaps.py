"""Measure state/old-plan consistency in pixels, before revision or source noise."""
from __future__ import annotations

import numpy as np
import torch

from action_bridge.plan_revision.contracts import ActionCodec, take
from action_bridge.plan_revision.completion import COMPLETION_NAMES
from action_bridge.plan_revision.data import build_self_sources, immutable_snapshot
from action_bridge.plan_revision.tracking import preserve_rng
from action_bridge.plan_revision.training import restore_completion


GAP_KEYS = (
    "retained_pusher_gap_px",
    "previous_command_mismatch_px",
    "command_tracking_gap_px",
    "retained_plan_jump_px",
)


def gap_record(pusher_xy, last_command_xy, old_plan_xy, executed):
    """Compare clean absolute targets with the observed pusher and actual history.

    old[K] is the first *unexecuted* target; old[K-1] is the final target from
    the supposedly executed prefix. Online, clipping can make that latter
    target differ from the command actually sent to the simulator.
    """
    pusher = np.asarray(pusher_xy, dtype=np.float64)
    command = np.asarray(last_command_xy, dtype=np.float64)
    old = np.asarray(old_plan_xy, dtype=np.float64)
    if pusher.shape != (2,) or command.shape != (2,) or old.ndim != 2 or old.shape[1] != 2:
        raise ValueError("Plan gaps require pixel xy positions and an [H,2] old plan")
    if not 0 < executed < len(old):
        raise ValueError("A retained-plan gap requires 0 < executed < horizon")
    differences = (old[executed] - pusher, old[executed - 1] - command,
                   command - pusher, old[executed] - old[executed - 1])
    values = [float(np.linalg.norm(delta)) for delta in differences]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite state/plan gap")
    return dict(zip(GAP_KEYS, values))


def summarize_gaps(rows):
    """Pool replan events, not episode averages; no retained plan is not zero gap."""
    result = {"gap_samples": len(rows)}
    for key in GAP_KEYS if rows else ():
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result.update({f"{key}_mean": float(values.mean()),
                       f"{key}_median": float(np.median(values)),
                       f"{key}_p95": float(np.quantile(values, .95)),
                       f"{key}_max": float(values.max())})
    return result


def _validation_prefixes(payload, config, metadata, episodes, replans, seed):
    """Use matching held-out coordinates/splits, but allow moving dataset files."""
    if episodes < 1 or replans < 2:
        raise ValueError("Use at least one validation episode and two replans (including startup)")
    saved = payload["metadata"]
    for key in ("normalization", "codec", "splits", "dataset_sha256", "normalizer_id",
                "observation_profile", "action_profile", "train_subset"):
        if saved.get(key) != metadata.get(key):
            raise ValueError(f"Window cache and checkpoint differ in {key}")
    spec = payload.get("data_spec", {})
    for key in ("horizon", "execute", "obs_history", "action_history"):
        if key in spec and spec[key] != config[key]:
            raise ValueError(f"Window cache and evaluation differ in {key}")
    records = payload["records"]["val"]
    if "startup_actions" not in records:
        raise ValueError("Window cache predates self-source replay; prepare fresh windows including episode startup")
    if records["future_actions"].shape[1:] != (config["horizon"], 2):
        raise ValueError("Validation chunks do not match the checkpoint horizon")
    ids = records["episode_id"].unique(sorted=True)
    generator = torch.Generator().manual_seed(seed)
    selected = ids[torch.randperm(len(ids), generator=generator)[:episodes]]
    indices = []
    for episode in selected:
        rows = (records["episode_id"] == episode).nonzero().flatten()
        rows = rows[records["time_index"][rows].argsort()][:replans]
        times = records["time_index"][rows]
        if int(times[0]) != 0 or bool(((times[1:] - times[:-1]) != config["execute"]).any()):
            raise ValueError("Validation prefixes must start at episode time zero on the checkpoint's execute grid")
        indices.extend(rows.tolist())
    if not indices:
        raise ValueError("No held-out windows available for source-gap diagnostics")
    return take(records, torch.tensor(indices)), selected.tolist()


def _decoded_row(record, old, config, metadata, codec):
    stats = metadata["normalization"]
    state = record["obs_hist"][-1]
    state = state * state.new_tensor(stats["obs_std"]) + state.new_tensor(stats["obs_mean"])
    pusher = state[:2].cpu().numpy()
    command = codec.decode(record["act_hist"][-1]).cpu().numpy()
    plan = codec.decode(old).cpu().numpy()
    executed = config["execute"]
    return {"episode_id": int(record["episode_id"]), "robot_step": int(record["time_index"]),
            "pusher_xy": pusher.tolist(), "last_command_xy": command.tolist(),
            "previous_command_xy": plan[executed - 1].tolist(),
            "retained_first_xy": plan[executed].tolist(),
            **gap_record(pusher, command, plan, executed)}


@torch.no_grad()
def diagnose_offline_sources(policy, config, metadata, dependencies, windows_payload, device,
                             *, episodes=8, replans=16, seed=7301, completion_id=2):
    """Compare generated versus expert old plans on identical held-out histories.

    This measures a fixed checkpoint's all-self offline replay, NOT its original
    mixed, frozen training-source cache. Expert old plans need no generation:
    the measured retained boundary precedes tail completion and source noise.
    Physical states and past executed commands remain recorded in both panels.
    No simulation, training, parameter mutation, or learner-RNG consumption.
    """
    if config["method"] == "ddim":
        raise ValueError("Offline source diagnostics require an FM/SB reviser, not DDIM")
    if config["obs_dim"] != 5 or config["action_dim"] != 2 or metadata["codec"]["units"] != "pixels":
        raise ValueError("These diagnostics require the low-dimensional Push-T pixel interface")
    if not 0 < config["execute"] <= config["horizon"]:
        raise ValueError("Require 0 < execute <= horizon")
    if completion_id not in range(len(COMPLETION_NAMES)) or completion_id not in config.get("training_completion_modes", (0, 1, 2)):
        raise ValueError("The requested completion mode was not trained in this checkpoint")
    with preserve_rng():
        windows, selected = _validation_prefixes(windows_payload, config, metadata, episodes, replans, seed)
        rows = {"self": [], "expert": []}
        if config["execute"] < config["horizon"]:
            completion = restore_completion(dependencies, device)
            replay, _ = build_self_sources(
                windows, immutable_snapshot(policy).to(device), completion,
                torch.as_tensor(dependencies["innovation_variance"]), config, device,
                block=0, seed=seed, p_self=1., modes=(completion_id,),
            )
            codec = ActionCodec(**metadata["codec"])
            previous = {(int(episode), int(step)): index for index, (episode, step) in enumerate(
                zip(windows["episode_id"], windows["time_index"]))}
            for index in replay["has_previous_plan"].nonzero().flatten().tolist():
                record = {key: value[index] for key, value in replay.items()}
                rows["self"].append(_decoded_row(record, record["old_actions"], config, metadata, codec))
                key = (int(record["episode_id"]), int(record["time_index"]) - config["execute"])
                expert_old = windows["future_actions"][previous[key]]
                rows["expert"].append(_decoded_row(record, expert_old, config, metadata, codec))
        metrics = {f"offline_{panel}_{key}": value for panel, values in rows.items()
                   for key, value in summarize_gaps(values).items()}
    return {"metrics": metrics, "records": rows,
            "selection": {"split": "val", "episode_ids": selected, "seed": seed,
                          "replans_per_episode_limit": replans, "includes_startup_in_limit": True,
                          "windows": len(windows["episode_id"]), "completion_id": completion_id,
                          "horizon": config["horizon"], "execute": config["execute"],
                          "self_probability": 1., "units": "pixels", "aggregation": "replan_event",
                          "boundary": "clean retained old target; before completion noise/revision",
                          "startup_excluded": True, "no_retained_plan": config["execute"] == config["horizon"]}}

"""Detached sequence replay: recorded histories and generated plans stay separate."""
from __future__ import annotations

import copy
import time

import torch

from action_bridge.plan_revision.cache import reference_for
from action_bridge.plan_revision.completion import complete_plan
from action_bridge.plan_revision.contracts import take, tree_map


def immutable_snapshot(model):
    """Independent parameters *and buffers*, including an injected encoder."""
    return copy.deepcopy(model).eval().requires_grad_(False)


def _concatenate(pieces):
    if isinstance(pieces[0], dict):
        return {key: _concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    return torch.cat(pieces)


class SourceReplayError(RuntimeError):
    """Numerical failures stop replay instead of inventing an expert fallback."""

    def __init__(self, stage, indices, diagnostics):
        self.diagnostics = dict(diagnostics)
        super().__init__(f"Nonfinite {stage} in self-source replay at window indices {indices}; "
                         f"numerical_failures={diagnostics['numerical_failures']}")


@torch.no_grad()
def build_self_sources(windows, snapshot, completion, innovation_variance, config, device,
                       *, block, seed, p_self=None, modes=(0, 1, 2)):
    """Replay each logged episode once per mode using one frozen EMA policy.

    Windows are complete-horizon records on the replan grid, including time zero.
    The adapter supplies the startup action; no observation layout or simulator
    is assumed here. Every sequence carries its own generated plan. Sources
    (including their one noise draw), targets and generated outputs stay separate.
    ``source_origin`` is audit-only: startup=0, expert-old=1, generated-old=2.
    """
    if config["method"] == "ddim":
        raise ValueError("DDIM is an independent baseline, not a self-source reviser")
    if snapshot.training or any(parameter.requires_grad for parameter in snapshot.parameters()):
        raise ValueError("source producer must be an immutable, frozen eval snapshot")
    if completion.training or any(parameter.requires_grad for parameter in completion.parameters()):
        raise ValueError("the shared completer must be frozen before source replay")
    if p_self is None:
        p_self = config.get("source_self_probabilities", (.1, .5, 1., 1.))[block]
    if not 0 <= p_self <= 1:
        raise ValueError("p_self must lie in [0,1]")
    if not modes or len(set(modes)) != len(modes) or any(mode not in (0, 1, 2) for mode in modes):
        raise ValueError("modes must be distinct completion IDs from [0,1,2]")
    required = {"episode_id", "time_index", "obs_hist", "act_hist", "future_actions",
                "valid_mask", "startup_actions"}
    if required - windows.keys():
        raise ValueError(f"sequence replay is missing adapter fields: {sorted(required - windows.keys())}")
    if not len(windows["future_actions"]) or not bool(windows["valid_mask"].all()):
        raise ValueError("source replay requires nonempty, complete valid expert horizons")
    if windows["startup_actions"].shape != windows["future_actions"].shape:
        raise ValueError("startup_actions must match future_actions shape")
    horizon, action_dim = windows["future_actions"].shape[1:]
    executed = config["execute"]
    if not 0 < executed <= horizon:
        raise ValueError("self-source replay requires 0 < execute <= horizon")

    # Sorting rows is harmless; skipping a replan is not: the old plan must be
    # from the immediately preceding replan, not an isolated earlier query.
    sequences = []
    for episode in windows["episode_id"].unique(sorted=True):
        rows = torch.where(windows["episode_id"] == episode)[0]
        rows = rows[windows["time_index"][rows].argsort()]
        times = windows["time_index"][rows]
        if int(times[0]) != 0 or bool(((times[1:] - times[:-1]) != executed).any()):
            raise ValueError("each replay episode must start at time zero and use the execute-sized grid")
        sequences.append(rows.tolist())

    generator = torch.Generator(device=device).manual_seed(seed)
    start = time.perf_counter()
    diagnostics = {"protocol": "self_source_v1", "block": int(block), "seed": int(seed),
                   "p_self": float(p_self), "numerical_failures": 0,
                   "startup_records": 0, "expert_records": 0, "self_records": 0,
                   "replay_episodes": len(sequences) * len(modes)}

    def check_finite(values, stage, indices):
        failed = ~torch.isfinite(values).flatten(1).all(1)
        if bool(failed.any()):
            diagnostics["numerical_failures"] += int(failed.sum())
            raise SourceReplayError(stage, indices[failed.cpu()].tolist(), diagnostics)

    pieces = []
    for mode in modes:
        previous_predictions = {}
        for position in range(max(map(len, sequences))):
            active = [episode for episode, sequence in enumerate(sequences) if position < len(sequence)]
            for offset in range(0, len(active), config["batch_size"]):
                episodes = active[offset:offset + config["batch_size"]]
                indices = torch.tensor([sequences[episode][position] for episode in episodes])
                batch = take(windows, indices, device)
                count = len(indices)
                mode_ids = torch.full((count,), mode, dtype=torch.long, device=device)
                # K=H exhausts the old plan. Reinitialize from the adapter's
                # current observed command anchor, not an external generator.
                available = position > 0 and executed < horizon
                has_previous = torch.full((count,), available, dtype=torch.bool, device=device)
                if not available:
                    old = batch["startup_actions"].clone()
                    source = old.clone()
                    origin = torch.zeros(count, dtype=torch.long, device=device)
                else:
                    old = torch.stack([previous_predictions[episode] for episode in episodes]).to(device)
                    self_choice = torch.rand(count, device=device, generator=generator) < p_self
                    # Future labels are read only for the declared previous-
                    # expert branch, and never passed to the snapshot sampler.
                    teacher_rows = (~self_choice).nonzero(as_tuple=True)[0]
                    if len(teacher_rows):
                        previous_rows = torch.tensor([
                            sequences[episodes[row]][position - 1] for row in teacher_rows.tolist()
                        ])
                        old[teacher_rows] = windows["future_actions"][previous_rows].to(device)
                    origin = torch.where(self_choice, 2, 1)
                    source = complete_plan(old, executed, batch["obs_hist"], batch["act_hist"],
                                           mode_ids, completion, robot_dt=config["robot_dt"])
                source = source + config["source_std"] * torch.randn(
                    source.shape, device=device, dtype=source.dtype, generator=generator)
                check_finite(source, "source", indices)
                prior = completion.prior(
                    batch["obs_hist"], batch["act_hist"], innovation_variance.to(device),
                    ridge=config["prior_ridge"], max_rate=config["max_rate"],
                    mobility_smoothing=config["mobility_smoothing"],
                )
                batch.update(source_actions=source, old_actions=old, completion_id=mode_ids,
                             has_previous_plan=has_previous, source_origin=origin,
                             p_self=torch.full((count,), p_self, dtype=source.dtype, device=device),
                             block=torch.full((count,), block, dtype=torch.long, device=device),
                             precision=prior["precision"], prior_mean=prior["mean"])
                if config["mobility_smoothing"]:
                    batch["mobility"] = prior["mobility"].expand(count, -1, -1).clone()
                reference = reference_for(batch, config) if config["method"].startswith("sb_") else None
                generated, _ = snapshot.sample(
                    batch["obs_hist"], batch["act_hist"], mode_ids, source_actions=source,
                    reference=reference, generator=generator, has_previous_plan=has_previous,
                )
                if generated.shape != (count, horizon, action_dim):
                    raise ValueError("source snapshot returned a chunk with the wrong shape")
                check_finite(generated, "generated plan", indices)
                batch["generated_actions"] = generated
                for episode, prediction in zip(episodes, generated.detach().cpu()):
                    previous_predictions[episode] = prediction.clone()
                for code, name in enumerate(("startup_records", "expert_records", "self_records")):
                    diagnostics[name] += int((origin == code).sum())
                pieces.append(tree_map(lambda value: value.detach().cpu(), batch))

    diagnostics.update(source_seconds=time.perf_counter() - start,
                       source_records=sum(len(piece["future_actions"]) for piece in pieces),
                       completion_modes=list(modes))
    return _concatenate(pieces), diagnostics

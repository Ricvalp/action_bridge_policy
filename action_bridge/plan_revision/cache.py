"""Fixed robot-time source records and separate revision-time coupling caches."""
from __future__ import annotations

import time

import torch

from action_bridge.plan_revision.contracts import take, tree_map
from action_bridge.plan_revision.completion import complete_plan
from action_bridge.plan_revision.gaussian import GaussianReference


def reference_kind_for(config):
    """Validate the reference ablation independently of chunk completion.

    ``learned`` keeps the existing OU/kinetic process. The two first-order
    ablations share the OU policy architecture and use identity mobility so
    that changing the potential does not also change the noise geometry.
    """
    kind = config.get("reference_kind", "learned")
    if kind not in {"learned", "brownian", "isotropic_ou"}:
        raise ValueError("reference_kind must be learned, brownian or isotropic_ou")
    if kind != "learned":
        if config["method"] != "sb_ou":
            raise ValueError(f"reference_kind={kind} requires method=sb_ou")
        if config.get("mobility_smoothing", 0.) != 0:
            raise ValueError(f"reference_kind={kind} requires mobility_smoothing=0")
    return kind


def reference_for(batch, config):
    """Construct the same reference for training, coupling, replay and inference.

    The input precision is already stabilized by the learned completer.
    Isotropic OU retains its contextual center and mean eigenvalue; Brownian
    removes its attraction. Neither variant changes the completed source.
    """
    variant = reference_kind_for(config)
    precision, mean = batch["precision"], batch["prior_mean"]
    mobility = batch.get("mobility")
    kind = "kinetic" if config["method"] == "sb_kinetic" else "ou"
    if variant != "learned":
        n = mean.shape[-1]
        eye = torch.eye(n, device=mean.device, dtype=precision.dtype)
        if mobility is not None:
            if mobility.shape not in {(n, n), precision.shape} or not torch.equal(
                mobility, eye.to(mobility).expand_as(mobility)
            ):
                raise ValueError(f"reference_kind={variant} requires identity mobility")
        mobility = None
        if variant == "brownian":
            kind = "brownian"
            precision = torch.zeros_like(precision)
        else:
            # trace(P)/n matches mean attraction strength, removing only the
            # anisotropy/correlations of the frozen learned potential.
            rate = precision.to(torch.float64).diagonal(dim1=-2, dim2=-1).mean(-1)
            precision = rate[:, None, None] * eye.to(torch.float64)
    return GaussianReference(precision, mean, kind=kind,
                             temperature=config["temperature"], gamma=config["revision_gamma"],
                             mobility=mobility)


def draw_records(records, count, device, *, completion=None, executed=None,
                 source_std=0.01, endpoint_std=0.001):
    """Sample the prescribed source population; only targets receive new noise.

    Sequence replay already selected the mode, completed the old plan and
    perturbed the source. Redrawing any of these changes the prescribed law.
    """
    indices = torch.randint(len(records["future_actions"]), (count,))
    batch = take(records, indices, device)
    batch["record_id"] = indices.to(device)
    if "source_actions" in batch:
        if "completion_id" not in batch or "has_previous_plan" not in batch:
            raise ValueError("cached sources require completion_id and has_previous_plan")
    elif "old_actions" in batch:
        # Generic, manually provided old-plan batches remain useful outside the
        # versioned self_source_v1 trainer. Its caches always take the branch above.
        batch["completion_id"] = torch.randint(3, (count,), device=device)
        with torch.no_grad():
            source = complete_plan(batch["old_actions"], executed, batch["obs_hist"],
                                   batch["act_hist"], batch["completion_id"], completion,
                                   robot_dt=completion.robot_dt if completion is not None else 1.)
        batch["source_actions"] = source + source_std * torch.randn_like(source)
    else:
        batch.setdefault("completion_id", torch.zeros(count, dtype=torch.long, device=device))
    batch["future_actions"] = (batch["future_actions"] +
                               endpoint_std * torch.randn_like(batch["future_actions"]))
    return batch


@torch.no_grad()
def refresh_coupling(records, count, device, config, completion, snapshot, direction):
    """Reverse fit uses forward-generated pairs; forward fit uses reverse pairs.

    snapshot includes its history encoder and is frozen for the entire phase.
    Context/record/completion identifiers stay attached to both endpoints.
    """
    start = time.perf_counter()
    pieces = []
    for offset in range(0, count, config["batch_size"]):
        batch = draw_records(records, min(config["batch_size"], count - offset), device,
                             completion=completion, executed=config["execute"],
                             source_std=config["source_std"], endpoint_std=config["endpoint_std"])
        reference = reference_for(batch, config)
        x0 = reference.augment(batch["source_actions"].flatten(1))
        x1 = reference.augment(batch["future_actions"].flatten(1))
        if snapshot is not None:
            reverse = direction == "forward"
            generated, _ = snapshot.rollout(x1 if reverse else x0, batch["obs_hist"],
                                            batch["act_hist"], batch["completion_id"],
                                            reference, reverse=reverse,
                                            steps=config["coupling_steps"],
                                            **({"has_previous_plan": batch["has_previous_plan"]}
                                               if "has_previous_plan" in batch else {}))
            if reverse:
                x0 = generated
            else:
                x1 = generated
        batch["x0"], batch["x1"] = x0.detach(), x1.detach()
        pieces.append(tree_map(lambda x: x.detach().cpu(), batch))
    def concatenate(items):
        if isinstance(items[0], dict):
            return {key: concatenate([item[key] for item in items]) for key in items[0]}
        return torch.cat(items)
    return concatenate(pieces), {"cache_seconds": time.perf_counter() - start,
                                 "cache_records": count,
                                 "generated_endpoint": "none" if snapshot is None else
                                 ("source" if direction == "forward" else "target")}

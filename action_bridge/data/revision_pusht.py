"""Push-T's only data-specific pieces: windows, coordinates and OT distance."""
from __future__ import annotations

import torch

from action_bridge.data.pusht_adapter import PushTLowDimDataset
from action_bridge.plan_revision.checkpoints import object_digest
from action_bridge.plan_revision.contracts import ActionCodec


def load_windows(path, config):
    """Split complete episodes before any fitted quantity or eligible window."""
    result, split_ids = {}, {}
    stats = None
    for split in ("train", "val", "test"):
        dataset = PushTLowDimDataset(
            str(path), split=split, obs_key="data/state" if str(path).endswith(".zarr") else None,
            obs_history=config["obs_history"], action_history=config["action_history"],
            chunk_horizon=config["horizon"], normalize=True, normalization_stats=stats,
            pad_episode_starts=True)
        if dataset.obs_dim != 5 or dataset.action_dim != 2:
            raise ValueError("Push-T requires state=[pusher xy, block xy, block angle] and absolute xy actions")
        stats = dataset.normalization_stats
        split_ids[split] = dataset.episode_ids
        records = []
        for episode, t in dataset.indices:
            if t < config["execute"]:
                continue
            now = dataset.item_from_episode_time(episode, t)
            earlier = dataset.item_from_episode_time(episode, t - config["execute"])
            records.append({"obs_hist": now["obs_hist"], "act_hist": now["act_hist"],
                            "future_actions": now["future_actions"],
                            "earlier_obs_hist": earlier["obs_hist"],
                            "earlier_act_hist": earlier["act_hist"],
                            "valid_mask": torch.ones(config["horizon"], dtype=torch.bool),
                            "episode_id": torch.tensor(episode), "time_index": torch.tensor(t),
                            "earlier_time_index": torch.tensor(t - config["execute"]),
                            "obs_history_mask": torch.arange(t - config["obs_history"] + 1, t + 1) >= 0,
                            "act_history_mask": torch.arange(t - config["action_history"], t) >= 0,
                            "earlier_obs_history_mask": torch.arange(t - config["execute"] -
                                config["obs_history"] + 1, t - config["execute"] + 1) >= 0,
                            "earlier_action_history_mask": torch.arange(t - config["execute"] -
                                config["action_history"], t - config["execute"]) >= 0})
        if not records:
            raise ValueError(f"No complete causal windows in {split}")
        result[split] = {key: torch.stack([item[key] for item in records]) for key in records[0]}
    if any(set(split_ids[a]) & set(split_ids[b]) for a, b in
           (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Need disjoint train/val/test episodes (at least three episodes)")
    codec = ActionCodec(stats["action_mean"], stats["action_std"], [0., 0.], [512., 512.], "pixels")
    metadata = {"dataset_path": str(path), "splits": split_ids, "normalization": stats,
                "normalizer_id": object_digest(stats), "codec": codec.specification(),
                "observation_profile": "pusht/state/pusher_xy-block_xy-angle/v1",
                "action_profile": "pusht/absolute_pusher_target_xy/pixels/v1",
                "source_noise_pixels": [x * config["source_std"] for x in stats["action_std"]],
                "endpoint_noise_pixels": [x * config["endpoint_std"] for x in stats["action_std"]],
                "padding": "earlier observation repeats initial observation; missing executed commands use initial pusher xy",
                "windows": {key: len(value["future_actions"]) for key, value in result.items()}}
    return result, metadata


def context_features(records, stats):
    """Observed geometry + actual recent commands, never expert futures."""
    state = records["obs_hist"][:, -1]
    angle = state[:, 4] * stats["obs_std"][4] + stats["obs_mean"][4]
    return torch.cat((state[:, :4], angle.sin()[:, None], angle.cos()[:, None],
                      records["act_hist"].flatten(1)), dim=1)


def context_distances(features):
    return torch.cdist(features, features) / features.shape[-1] ** .5


def neighbor_blocks(records, stats, size=8):
    """Train-only neighbour indices and a conservative median nearest radius.

    Uniformly sampled anchors are the targets in BOTH FM variants. Neighbours
    only supply the local transport block, not an oversampled target pool.
    """
    features = context_features(records, stats)
    neighbors, distances = [], []
    k = min(size, len(features))
    for start in range(0, len(features), 256):
        distance = torch.cdist(features[start:start + 256], features) / features.shape[-1] ** .5
        values, indices = distance.topk(k, largest=False)
        neighbors.append(indices)
        distances.append(values)
    neighbors = torch.cat(neighbors)
    distances = torch.cat(distances)
    # Two proposals can share a context exactly; use the first positive neighbour.
    positive = distances.masked_fill(distances <= 1e-5, float("inf")).min(dim=1).values
    positive = positive[positive.isfinite()]
    radius = float(positive.median()) if len(positive) else 0.
    return {"neighbors": neighbors, "features": features, "radius": radius,
            "distance_spec": "rms standardized pusher/block xy, sin/cos angle, recent executed targets"}


def make_local_pairer(records, neighborhood, config):
    """Uniform anchor targets with batched, observation-local transport blocks."""
    from action_bridge.plan_revision.completion import complete_plan
    from action_bridge.plan_revision.contracts import take
    from action_bridge.plan_revision.ot import batched_local_ot_pair

    # Cache the index arrangement once. Tied/duplicate *contexts* can make topk
    # omit the anchor; it must still occur exactly once in column zero.
    neighbors = neighborhood["neighbors"].cpu()
    anchors = torch.arange(len(neighbors))
    size = min(config["ot_block_size"], neighbors.shape[1], len(neighbors))
    if size < 1:
        raise ValueError("local OT requires a nonempty neighbourhood")
    if bool((neighbors.sort(dim=1).values.diff(dim=1) == 0).any()):
        raise ValueError("neighbour indices must be unique within each block")
    order = (neighbors == anchors[:, None]).long().argsort(dim=1, stable=True)
    blocks = torch.cat((anchors[:, None], neighbors.gather(1, order)[:, :size - 1]), dim=1)
    fields = {key: records[key] for key in ("old_actions", "obs_hist", "act_hist", "future_actions")}

    @torch.no_grad()
    def pair(batch, completion, device):
        ids = blocks[batch["record_id"].cpu()]
        count = len(ids)
        block = take(fields, ids.flatten(), device)
        mode = batch["completion_id"][:, None].expand(-1, size)
        source = complete_plan(block["old_actions"], config["execute"], block["obs_hist"],
                               block["act_hist"], mode.flatten(), completion,
                               robot_dt=config.get("robot_dt", 1.))
        source += config["source_std"] * torch.randn_like(source)
        source = source.reshape(count, size, *source.shape[1:])
        targets = block["future_actions"] + config["endpoint_std"] * torch.randn_like(block["future_actions"])
        targets = targets.reshape(count, size, *targets.shape[1:])
        # Keep the original uniform target's exact dequantized endpoint/history.
        targets[:, 0] = batch["future_actions"]
        features = neighborhood["features"][ids].to(device)
        distances = torch.cdist(features, features) / features.shape[-1] ** .5
        selected, _, metrics = batched_local_ot_pair(
            source, targets, distances, distances <= neighborhood["radius"],
            completion_ids=mode, radius=neighborhood["radius"],
            entropy=config["ot_entropy"], context_weight=config["ot_context_weight"],
            iterations=config.get("ot_iterations", 500), target_indices=[0])
        batch["source_actions"] = source[torch.arange(count, device=device), selected[:, 0]]
        # Scalar tensors synchronize only when the trainer actually writes a log.
        return batch, {"ot/" + key: value.float().mean() for key, value in metrics.items()}
    return pair

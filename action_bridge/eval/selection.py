"""Dataset index selection helpers for diagnostic plots."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Iterable, Optional


def _clamp_fraction(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _unique_in_order(values: Iterable[int]) -> list[int]:
    seen = set()
    out: list[int] = []
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _linspace_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count == 1:
        return [length // 2]
    return _unique_in_order(round(i * (length - 1) / (count - 1)) for i in range(count))


def _group_dataset_indices(dataset) -> Optional[OrderedDict[int, list[tuple[int, int]]]]:
    pairs = getattr(dataset, "indices", None)
    if not pairs:
        return None
    groups: OrderedDict[int, list[tuple[int, int]]] = OrderedDict()
    for dataset_idx, pair in enumerate(pairs):
        if len(pair) < 2:
            return None
        traj_id = int(pair[0])
        time_index = int(pair[1])
        groups.setdefault(traj_id, []).append((dataset_idx, time_index))
    for entries in groups.values():
        entries.sort(key=lambda item: item[1])
    return groups


def _select_from_entries(entries: list[tuple[int, int]], time_fraction: float) -> int:
    if not entries:
        raise ValueError("Cannot select from an empty trajectory.")
    if len(entries) == 1:
        return entries[0][0]
    fraction = _clamp_fraction(time_fraction)
    first_t = entries[0][1]
    last_t = entries[-1][1]
    target_t = first_t + fraction * (last_t - first_t)
    return min(entries, key=lambda item: (abs(item[1] - target_t), item[0]))[0]


def select_history_index(dataset, trajectory_fraction: float = 0.5, time_fraction: float = 0.5) -> int:
    """Select one dataset index by trajectory position and within-trajectory time."""

    if len(dataset) <= 0:
        raise ValueError("Cannot select a history from an empty dataset.")
    groups = _group_dataset_indices(dataset)
    if groups:
        traj_ids = list(groups.keys())
        traj_pos = round(_clamp_fraction(trajectory_fraction) * (len(traj_ids) - 1))
        return _select_from_entries(groups[traj_ids[int(traj_pos)]], time_fraction)
    return round(_clamp_fraction(time_fraction) * (len(dataset) - 1))


def representative_history_indices(
    dataset,
    count: int,
    time_fractions: Iterable[float] = (0.2, 0.5, 0.8),
    trajectory_fractions: Optional[Iterable[float]] = None,
) -> list[int]:
    """Select diagnostic indices spread across trajectories and chunk start times."""

    count = min(int(count), len(dataset))
    if count <= 0:
        return []
    groups = _group_dataset_indices(dataset)
    if not groups:
        positions = _linspace_indices(len(dataset), count)
        return positions[:count]

    traj_ids = list(groups.keys())
    times = [_clamp_fraction(value) for value in time_fractions]
    if not times:
        times = [0.5]

    if trajectory_fractions is None:
        num_traj = max(1, math.ceil(count / len(times)))
        traj_positions = _linspace_indices(len(traj_ids), min(num_traj, len(traj_ids)))
    else:
        traj_positions = _unique_in_order(round(_clamp_fraction(value) * (len(traj_ids) - 1)) for value in trajectory_fractions)
        if not traj_positions:
            traj_positions = [len(traj_ids) // 2]

    selected: list[int] = []
    used = set()
    for traj_pos in traj_positions:
        entries = groups[traj_ids[int(traj_pos)]]
        for time_fraction in times:
            idx = _select_from_entries(entries, time_fraction)
            if idx in used:
                continue
            selected.append(idx)
            used.add(idx)
            if len(selected) >= count:
                return selected

    for idx in _linspace_indices(len(dataset), count * 2):
        if idx in used:
            continue
        selected.append(idx)
        used.add(idx)
        if len(selected) >= count:
            return selected
    return selected[:count]


def history_index_metadata(dataset, indices: Iterable[int]) -> list[dict[str, int]]:
    pairs = getattr(dataset, "indices", None)
    metadata = []
    for idx in indices:
        item = {"dataset_index": int(idx)}
        if pairs:
            pair = pairs[int(idx)]
            if len(pair) >= 2:
                item["trajectory_id"] = int(pair[0])
                item["time_index"] = int(pair[1])
        metadata.append(item)
    return metadata

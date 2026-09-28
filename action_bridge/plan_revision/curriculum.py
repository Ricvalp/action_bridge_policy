"""Block-based robot-time K schedules; revision time and model width stay fixed."""
from __future__ import annotations

import torch

from action_bridge.plan_revision.contracts import take


CURRICULUM_PROTOCOL = "self_source_k_curriculum_v1"
K_MODEL_SCHEMA = "k_conditioned_v1"


def is_curriculum(config):
    return config.get("protocol") == CURRICULUM_PROTOCOL


def validate_curriculum(config):
    """Reject ambiguous schedules, rather than clipping intervals or budgets."""
    if not is_curriculum(config):
        return
    horizon, deployment = config["horizon"], config["execute"]
    if type(horizon) is not int or horizon < 2:
        raise ValueError("curriculum horizon must be an integer >= 2")
    values = config.get("k_values_by_block")
    blocks = config.get("training_blocks")
    if type(blocks) is not int or blocks < 1 or not isinstance(values, (tuple, list)) or len(values) != blocks:
        raise ValueError("k_values_by_block must have one interval per training block")
    if any(type(k) is not int or not 1 <= k <= horizon for k in values):
        raise ValueError("all scheduled K values must be integers in [1, H]")
    if list(values) != sorted(values):
        raise ValueError("K curriculum must be nondecreasing")
    if type(deployment) is not int or values[-1] != deployment:
        raise ValueError("the final K block must equal deployment execute")
    if config.get("condition_on_k") is not True or config.get("model_schema") != K_MODEL_SCHEMA:
        raise ValueError("K curriculum requires condition_on_k and model_schema=k_conditioned_v1")
    updates = config.get("updates")
    if type(updates) is not int or updates < blocks or updates % blocks:
        raise ValueError("curriculum updates must split into equal nonempty training blocks")
    if config.get("method", "").startswith("sb_"):
        rounds, phase_updates = config.get("rounds"), config.get("phase_updates")
        if (type(rounds) is not int or rounds != blocks or type(phase_updates) is not int
                or phase_updates < 1 or 2 * phase_updates * blocks != updates):
            raise ValueError("SB curriculum needs one complete reverse/forward round per block")
    probabilities = config.get("source_self_probabilities")
    if not isinstance(probabilities, (list, tuple)) or len(probabilities) != blocks or any(
            not 0 <= probability <= 1 for probability in probabilities):
        raise ValueError("source_self_probabilities must specify a probability per block")
    if 3 in config.get("training_completion_modes", [0, 1, 2]) or config.get("completion_id", 2) == 3:
        raise ValueError("direct_mlp has fixed-K outputs and is not supported by the K curriculum")
    fraction = config.get("startup_sampling_fraction", .1)
    if isinstance(fraction, bool) or not 0 < fraction < 1:
        raise ValueError("startup_sampling_fraction must be strictly between zero and one")
    if config.get("curriculum_mode", "scheduled") != "scheduled":
        raise ValueError("only the deterministic scheduled K curriculum is implemented")


def scheduled_k(config, block):
    if not is_curriculum(config):
        return config["execute"]
    validate_curriculum(config)
    if type(block) is not int or not 0 <= block < config["training_blocks"]:
        raise ValueError("curriculum block lies outside the configured schedule")
    return config["k_values_by_block"][block]


def replay_k(config):
    """Active replay interval; execute remains the deployment interval."""
    return config.get("active_k", config["execute"]) if is_curriculum(config) else config["execute"]


def stage_config(config, block):
    result = dict(config)
    if is_curriculum(config):
        result.update(active_k=scheduled_k(config, block), curriculum_block=block)
    return result


def stage_windows(windows, config):
    """Select the stage grid from contiguous recorded-time adapter windows.

    Histories themselves remain contiguous at the original control frequency.
    We always replay from time zero so every retained source has its actual
    generated ancestors, including decisions not ultimately sampled by SGD.
    """
    if not is_curriculum(config):
        return windows
    selected = []
    for episode in windows["episode_id"].unique(sorted=True):
        rows = torch.where(windows["episode_id"] == episode)[0]
        rows = rows[windows["time_index"][rows].argsort()]
        times = windows["time_index"][rows]
        if int(times[0]) != 0 or bool((times.diff() != 1).any()):
            raise ValueError("K curriculum requires dense contiguous recorded windows starting at time zero")
        selected.append(rows[times.remainder(replay_k(config)) == 0])
    if not selected:
        raise ValueError("K curriculum needs at least one complete recorded episode")
    return take(windows, torch.cat(selected))

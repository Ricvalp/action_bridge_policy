"""Fixed validation action chunks in Push-T's pixel coordinates (no simulator)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from action_bridge.eval.visualization import _draw_tee, _import_pyplot
from action_bridge.plan_revision.contracts import ActionCodec, take
from action_bridge.plan_revision.data import build_self_sources, immutable_snapshot
from action_bridge.plan_revision.training import restore_completion


def _example_indices(records, count):
    # Cached sources contain multiple proposals per history. Show each history
    # only once, spread across validation rather than adjacent timesteps.
    unique = {}
    for index, pair in enumerate(zip(records["episode_id"].tolist(), records["time_index"].tolist())):
        unique.setdefault(tuple(pair), index)
    indices = list(unique.values())
    count = min(count, len(indices))
    return torch.tensor([indices[int((i + .5) * len(indices) / count)] for i in range(count)],
                        dtype=torch.long)


def _preview_replays(records, count, replans=3):
    """A few short episode prefixes, including startup, keep logging cheap."""
    episodes = records["episode_id"].unique(sorted=True).tolist()[:count]
    indices, selected = [], []
    for episode in episodes:
        candidates = (records["episode_id"] == episode).nonzero().flatten()
        candidates = candidates[records["time_index"][candidates].argsort()][:replans]
        indices.extend(candidates.tolist())
        selected.append(len(indices) - 1)
    return take(records, torch.tensor(indices, dtype=torch.long)), torch.tensor(selected, dtype=torch.long)


def _reference_rollout(model, batch):
    """Autonomous command dynamics, not teacher-observed next-step predictions."""
    attractor, stiffness, damping = model.coefficients(batch["obs_hist"], batch["act_hist"])
    dt = model.robot_dt
    q = batch["act_hist"][:, -1]
    velocity = (q - batch["act_hist"][:, -2]) / dt
    actions = []
    for index in range(model.horizon):
        velocity = velocity + dt * (
            -stiffness[:, index] * (q - attractor[:, index]) - damping[:, index] * velocity)
        q = q + dt * velocity
        actions.append(q)
    return torch.stack(actions, dim=1)


def _plot_chunk(path, state, expert, predicted, *, source=None, title=""):
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        _draw_tee(ax, np.array([256., 256., math.pi / 4]), "tab:green", .8,
                  label="Goal T", linestyle="--")
        _draw_tee(ax, state[2:5], "0.45", .45, label="Current T")
        ax.scatter(*state[:2], s=85, color="tab:blue", edgecolors="black", label="Current pusher", zorder=5)
        if source is not None:
            ax.plot(*source.T, "--", color="0.55", linewidth=1, label="Completed old plan")
        ax.plot(*expert.T, "o-", color="tab:blue", markersize=3, linewidth=1.5, label="GT commands")
        ax.plot(*predicted.T, "o-", color="tab:orange", markersize=3, linewidth=1.5, label="Generated commands")
        ax.set(xlim=(0, 512), ylim=(512, 0), xlabel="x (pixels)", ylabel="y (pixels)", title=title)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)


def make_action_chunk_plotter(records, config, metadata, output, device, *,
                             dependencies=None, count=3, reference=False):
    """Return ``plot(model, step) -> PNG paths`` for fixed held-out histories.

    The current T and goal are scene context, not predicted object trajectories.
    Both curves are absolute pusher target commands, decoded exactly once.
    Generated chunks use inference-time sampling, never future expert actions.
    """
    if count < 0:
        raise ValueError("Action chunk image count cannot be negative")
    reviser = not reference and config["method"] != "ddim"
    if reviser and count:
        replay_windows, indices = _preview_replays(records, count)
        batch = take(replay_windows, indices, device)
    else:
        indices = _example_indices(records, count)
        batch = take(records, indices, device)
    if not len(indices):
        return lambda model, step: []
    codec = ActionCodec(**metadata["codec"])
    stats = metadata["normalization"]
    state = batch["obs_hist"][:, -1]
    state = state * state.new_tensor(stats["obs_std"]) + state.new_tensor(stats["obs_mean"])
    states = state.detach().cpu().numpy()
    expert = codec.decode(batch["future_actions"]).detach().cpu().numpy()
    completion = None
    if reviser:
        # Loading frozen weights initializes a temporary CPU module first. Keep
        # that initialization from consuming the trainer's random stream.
        with torch.random.fork_rng(devices=[]):
            completion = restore_completion(dependencies, device)
    output = Path(output)

    @torch.no_grad()
    def plot(model, step):
        training_states = [(module, module.training) for module in model.modules()]
        model.eval()
        try:
            # The same sampling noise at each step makes visual changes useful.
            generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 7301)
            source = None
            if reference:
                prediction = _reference_rollout(model, batch)
            elif reviser:
                # Replay recorded contexts with this model's own previous
                # predictions. Labels are drawn only for the blue GT curve.
                replay, _ = build_self_sources(
                    replay_windows, immutable_snapshot(model), completion,
                    torch.as_tensor(dependencies["innovation_variance"], device=device),
                    config, device, block=0, seed=int(config["seed"]) + 7301,
                    p_self=1., modes=(config.get("completion_id", 2),))
                positions = []
                for episode, timestamp in zip(batch["episode_id"].tolist(), batch["time_index"].tolist()):
                    match = ((replay["episode_id"] == episode) & (replay["time_index"] == timestamp)).nonzero().flatten()
                    if len(match) != 1:
                        raise ValueError("Preview replay did not preserve its selected history key")
                    positions.append(int(match[0]))
                selected = take(replay, torch.tensor(positions), device)
                source, prediction = selected["source_actions"], selected["generated_actions"]
            else:
                prediction, _ = model.sample(batch["obs_hist"], batch["act_hist"], config.get("completion_id", 2),
                                             source_actions=None, reference=None, generator=generator)
            predicted = codec.decode(prediction).detach().cpu().numpy()
            source_pixels = None if source is None else codec.decode(source).detach().cpu().numpy()
            paths = []
            method = "Reference rollout" if reference else config["method"]
            for index in range(len(indices)):
                path = output / "figures" / f"step_{step:08d}_example_{index:02d}.png"
                episode, time_index = int(batch["episode_id"][index]), int(batch["time_index"][index])
                _plot_chunk(path, states[index], expert[index], predicted[index],
                            source=None if source_pixels is None else source_pixels[index],
                            title=f"{method} | step {step:,}\nValidation episode {episode}, t={time_index}")
                paths.append(path)
            return paths
        finally:
            for module, training in training_states:
                module.training = training

    return plot

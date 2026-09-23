"""Fixed validation action chunks in Push-T's pixel coordinates (no simulator)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from action_bridge.eval.visualization import _draw_tee, _import_pyplot
from action_bridge.plan_revision.cache import reference_for
from action_bridge.plan_revision.completion import complete_plan
from action_bridge.plan_revision.contracts import ActionCodec, take
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
    indices = _example_indices(records, count)
    if not len(indices):
        return lambda model, step: []
    batch = take(records, indices, device)
    codec = ActionCodec(**metadata["codec"])
    stats = metadata["normalization"]
    state = batch["obs_hist"][:, -1]
    state = state * state.new_tensor(stats["obs_std"]) + state.new_tensor(stats["obs_mean"])
    states = state.detach().cpu().numpy()
    expert = codec.decode(batch["future_actions"]).detach().cpu().numpy()
    completion = None
    if not reference and "old_actions" in batch and config.get("completion_id", 2) == 2:
        # Loading frozen weights initializes a temporary CPU module first. Keep
        # that initialization from consuming the trainer's random stream.
        with torch.random.fork_rng(devices=[]):
            completion = restore_completion(dependencies, device)
    gaussian = reference_for(batch, config) if not reference and config["method"].startswith("sb_") else None
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
            else:
                if "old_actions" in batch:
                    source = complete_plan(batch["old_actions"], config["execute"], batch["obs_hist"],
                                           batch["act_hist"], config.get("completion_id", 2), completion,
                                           robot_dt=config.get("robot_dt", 1.))
                    noise = torch.randn(source.shape, device=source.device, dtype=source.dtype,
                                        generator=generator)
                    source = source + config["source_std"] * noise
                prediction, _ = model.sample(batch["obs_hist"], batch["act_hist"], config.get("completion_id", 2),
                                             source_actions=source, reference=gaussian, generator=generator)
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

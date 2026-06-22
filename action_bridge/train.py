"""Train and evaluate execution-time action bridge policies."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random

from absl import app
from ml_collections import config_dict, config_flags
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    ActionChunkDataset,
    PointObstacleConfig,
    clip_action,
    generate_dataset,
    load_npz_arrays,
    segment_crosses_obstacle,
    step_state,
)
from .losses import (
    action_batch_metrics,
    action_jerk_loss,
    bridge_path_energy,
    particle_diversity,
    particle_path_diversity,
    sinkhorn_bridge_energy,
    sinkhorn_marginal_matching,
    sinkhorn_path_matching,
)
from .models import (
    ChunkMLPPolicy,
    LatentSinkhornActionBridgePolicy,
    ResidualActionBridgePolicy,
    SinkhornActionBridgePolicy,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "bridge_prev.py"
_CONFIG = config_flags.DEFINE_config_file(
    "config",
    str(DEFAULT_CONFIG),
    "Path to an ml_collections config file.",
)


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cfg_to_dict(value):
    if isinstance(value, config_dict.ConfigDict):
        return {key: cfg_to_dict(value[key]) for key in value}
    if isinstance(value, dict):
        return {key: cfg_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cfg_to_dict(item) for item in value]
    return value


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_run_name(cfg: config_dict.ConfigDict) -> str:
    return f"{cfg.model.type}_{cfg.model.get('init_type', 'none')}_n{cfg.data.num_trajectories}_h{cfg.data.horizon}_s{cfg.seed}"


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def dataset_path(root: Path, cfg: config_dict.ConfigDict) -> Path:
    if cfg.data.path:
        return resolve_path(root, cfg.data.path)
    prefix_steps = int(cfg.data.get("shared_prefix_steps", 0))
    if prefix_steps > 0:
        return (
            root
            / "data"
            / (
                f"point_modes_prefix{prefix_steps}_"
                f"n{cfg.data.num_trajectories}_t{cfg.data.trajectory_length}_s{cfg.seed}.npz"
            )
        )
    return (
        root
        / "data"
        / f"point_modes_n{cfg.data.num_trajectories}_t{cfg.data.trajectory_length}_s{cfg.seed}.npz"
    )


def make_env_config(cfg: config_dict.ConfigDict) -> PointObstacleConfig:
    return PointObstacleConfig(
        shared_prefix_steps=int(cfg.data.get("shared_prefix_steps", 0)),
        shared_prefix_speed=float(cfg.data.get("shared_prefix_speed", 0.55)),
        shared_prefix_target_x=float(cfg.data.get("shared_prefix_target_x", 0.30)),
    )


def make_loaders(root: Path, cfg: config_dict.ConfigDict, env_cfg: PointObstacleConfig):
    path = dataset_path(root, cfg)
    if cfg.data.force_regenerate or not path.exists():
        generate_dataset(
            path=path,
            num_trajectories=cfg.data.num_trajectories,
            trajectory_length=cfg.data.trajectory_length,
            seed=cfg.seed,
            cfg=env_cfg,
            paired_modes=cfg.data.paired_modes,
        )

    train_ds = ActionChunkDataset(
        path,
        context=cfg.data.context,
        horizon=cfg.data.horizon,
        split="train",
        train_fraction=cfg.data.train_fraction,
    )
    test_ds = ActionChunkDataset(
        path,
        context=cfg.data.context,
        horizon=cfg.data.horizon,
        split="test",
        train_fraction=cfg.data.train_fraction,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.train.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.train.num_workers,
    )
    return train_loader, test_loader, path


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def make_model(cfg: config_dict.ConfigDict) -> torch.nn.Module:
    common = dict(
        context=cfg.data.context,
        horizon=cfg.data.horizon,
        state_dim=cfg.model.state_dim,
        action_dim=cfg.model.action_dim,
        history_dim=cfg.model.history_dim,
        hidden_dim=cfg.model.hidden_dim,
        noise_dim=cfg.model.noise_dim,
        noise_scale=cfg.model.noise_scale,
        action_limit=cfg.model.action_limit,
        use_context_actions=cfg.model.get("use_context_actions", True),
    )
    if cfg.model.type == "chunk":
        return ChunkMLPPolicy(**common)
    if cfg.model.type == "bridge":
        return ResidualActionBridgePolicy(
            **common,
            tau=cfg.model.tau,
            init_type=cfg.model.init_type,
            init_noise_scale=cfg.model.init_noise_scale,
        )
    if cfg.model.type == "sinkhorn_bridge":
        return SinkhornActionBridgePolicy(
            context=cfg.data.context,
            horizon=cfg.data.horizon,
            state_dim=cfg.model.state_dim,
            action_dim=cfg.model.action_dim,
            history_dim=cfg.model.history_dim,
            hidden_dim=cfg.model.hidden_dim,
            tau=cfg.model.tau,
            init_type=cfg.model.init_type,
            init_noise_scale=cfg.model.init_noise_scale,
            particles=cfg.model.get("particles", 8),
            action_limit=cfg.model.action_limit,
            use_context_actions=cfg.model.get("use_context_actions", True),
        )
    if cfg.model.type == "latent_sinkhorn_bridge":
        return LatentSinkhornActionBridgePolicy(
            context=cfg.data.context,
            horizon=cfg.data.horizon,
            state_dim=cfg.model.state_dim,
            action_dim=cfg.model.action_dim,
            history_dim=cfg.model.history_dim,
            hidden_dim=cfg.model.hidden_dim,
            tau=cfg.model.tau,
            init_type=cfg.model.init_type,
            init_noise_scale=cfg.model.init_noise_scale,
            particles=cfg.model.get("particles", 16),
            latent_dim=cfg.model.get("latent_dim", 8),
            latent_init_scale=cfg.model.get("latent_init_scale", 1.0),
            latent_limit=cfg.model.get("latent_limit", 2.0),
            action_limit=cfg.model.action_limit,
            use_context_actions=cfg.model.get("use_context_actions", True),
        )
    raise ValueError(f"Unknown model type: {cfg.model.type}")


def path_context_features(
    batch: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    cfg: config_dict.ConfigDict,
) -> torch.Tensor:
    mode = cfg.loss.get("path_context", "state")
    if mode == "state":
        return batch["context_states"][:, -1]
    if mode == "state_action":
        return torch.cat(
            [
                batch["context_states"].reshape(batch["context_states"].shape[0], -1),
                batch["context_actions"].reshape(batch["context_actions"].shape[0], -1),
            ],
            dim=-1,
        )
    if mode == "history":
        return out["history"]
    if mode == "none":
        return batch["context_states"].new_zeros((batch["context_states"].shape[0], 1))
    raise ValueError(f"Unknown path_context: {mode}")


def compute_loss(batch: dict[str, torch.Tensor], model: torch.nn.Module, cfg: config_dict.ConfigDict):
    out = model(batch, deterministic=False)
    pred = out["actions"]
    target = batch["future_actions"]
    if cfg.model.type in ("sinkhorn_bridge", "latent_sinkhorn_bridge"):
        mean_action_loss = F.mse_loss(pred, target)
        endpoint_loss = F.mse_loss(pred[:, -1], target[:, -1])
        first_loss = F.mse_loss(pred[:, 0], target[:, 0])
        sinkhorn = target.new_zeros(())
        if cfg.loss.get("sinkhorn_weight", 0.0) != 0.0:
            sinkhorn = sinkhorn_marginal_matching(
                out["particles"],
                target,
                out["history"],
                epsilon=cfg.loss.sinkhorn_epsilon,
                iterations=cfg.loss.sinkhorn_iterations,
                context_weight=cfg.loss.sinkhorn_context_weight,
                intermediate_weight=cfg.loss.sinkhorn_intermediate_weight,
                endpoint_weight=cfg.loss.sinkhorn_endpoint_weight,
            )
        path_sinkhorn = target.new_zeros(())
        if cfg.loss.get("path_sinkhorn_weight", 0.0) != 0.0:
            path_sinkhorn = sinkhorn_path_matching(
                out["particles"],
                target,
                path_context_features(batch, out, cfg),
                epsilon=cfg.loss.get("path_sinkhorn_epsilon", cfg.loss.sinkhorn_epsilon),
                iterations=cfg.loss.get("path_sinkhorn_iterations", cfg.loss.sinkhorn_iterations),
                context_weight=cfg.loss.get("path_context_weight", 1.0),
            )
        bridge = target.new_zeros(())
        if cfg.loss.get("bridge_weight", 0.0) != 0.0:
            bridge = sinkhorn_bridge_energy(
                out["init_particles"],
                out["particles"],
                out["history"],
                phi_final=cfg.loss.phi_final,
                epsilon=cfg.loss.sinkhorn_epsilon,
                iterations=cfg.loss.sinkhorn_iterations,
                context_weight=cfg.loss.sinkhorn_context_weight,
            )
        jerk = action_jerk_loss(pred, batch["context_actions"])
        diversity = particle_diversity(out["particles"])
        path_diversity = particle_path_diversity(out["particles"])
        total = (
            cfg.loss.sinkhorn_weight * sinkhorn
            + cfg.loss.get("path_sinkhorn_weight", 0.0) * path_sinkhorn
            + cfg.loss.bridge_weight * bridge
            + cfg.loss.mean_action_weight * mean_action_loss
            + cfg.loss.endpoint_weight * endpoint_loss
            + cfg.loss.first_action_weight * first_loss
            + cfg.loss.jerk_weight * jerk
            - cfg.loss.diversity_weight * diversity
        )
        parts = {
            "loss": total.detach(),
            "sinkhorn": sinkhorn.detach(),
            "path_sinkhorn": path_sinkhorn.detach(),
            "action": mean_action_loss.detach(),
            "endpoint": endpoint_loss.detach(),
            "first": first_loss.detach(),
            "bridge": bridge.detach(),
            "jerk": jerk.detach(),
            "diversity": diversity.detach(),
            "path_diversity": path_diversity.detach(),
        }
        return total, parts

    action_loss = F.mse_loss(pred, target)
    endpoint_loss = F.mse_loss(pred[:, -1], target[:, -1])
    first_loss = F.mse_loss(pred[:, 0], target[:, 0])
    bridge = bridge_path_energy(out["init_action"], pred, cfg.loss.phi_final)
    jerk = action_jerk_loss(pred, batch["context_actions"])
    total = (
        cfg.loss.action_weight * action_loss
        + cfg.loss.endpoint_weight * endpoint_loss
        + cfg.loss.first_action_weight * first_loss
        + cfg.loss.bridge_weight * bridge
        + cfg.loss.jerk_weight * jerk
    )
    parts = {
        "loss": total.detach(),
        "action": action_loss.detach(),
        "endpoint": endpoint_loss.detach(),
        "first": first_loss.detach(),
        "bridge": bridge.detach(),
        "jerk": jerk.detach(),
    }
    return total, parts


@torch.no_grad()
def evaluate_actions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: config_dict.ConfigDict,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    total = 0
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch, deterministic=cfg.eval.deterministic)
        metrics = action_batch_metrics(
            out["actions"],
            batch["future_actions"],
            out["init_action"],
            batch["context_actions"],
        )
        if "particles" in out:
            if cfg.loss.get("sinkhorn_weight", 0.0) != 0.0:
                metrics["marginal_sinkhorn"] = sinkhorn_marginal_matching(
                    out["particles"],
                    batch["future_actions"],
                    out["history"],
                    epsilon=cfg.loss.sinkhorn_epsilon,
                    iterations=cfg.loss.sinkhorn_iterations,
                    context_weight=cfg.loss.sinkhorn_context_weight,
                    intermediate_weight=cfg.loss.sinkhorn_intermediate_weight,
                    endpoint_weight=cfg.loss.sinkhorn_endpoint_weight,
                )
            if cfg.loss.get("bridge_weight", 0.0) != 0.0:
                metrics["bridge_energy"] = sinkhorn_bridge_energy(
                    out["init_particles"],
                    out["particles"],
                    out["history"],
                    phi_final=cfg.loss.phi_final,
                    epsilon=cfg.loss.sinkhorn_epsilon,
                    iterations=cfg.loss.sinkhorn_iterations,
                    context_weight=cfg.loss.sinkhorn_context_weight,
                )
            else:
                metrics["bridge_energy"] = torch.zeros((), device=device)
            metrics["particle_diversity"] = particle_diversity(out["particles"])
            metrics["particle_path_diversity"] = particle_path_diversity(out["particles"])
            if cfg.loss.get("path_sinkhorn_weight", 0.0) != 0.0:
                metrics["path_sinkhorn"] = sinkhorn_path_matching(
                    out["particles"],
                    batch["future_actions"],
                    path_context_features(batch, out, cfg),
                    epsilon=cfg.loss.get("path_sinkhorn_epsilon", cfg.loss.sinkhorn_epsilon),
                    iterations=cfg.loss.get("path_sinkhorn_iterations", cfg.loss.sinkhorn_iterations),
                    context_weight=cfg.loss.get("path_context_weight", 1.0),
                )
        else:
            metrics["bridge_energy"] = bridge_path_energy(out["init_action"], out["actions"], cfg.loss.phi_final)
        metrics["network_evals"] = torch.tensor(float(model.network_evals), device=device)
        batch_size = batch["future_actions"].shape[0]
        total += batch_size
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * batch_size
    return {key: value / max(1, total) for key, value in sums.items()}


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mode_stats(positions: np.ndarray, mode: int, env_cfg: PointObstacleConfig) -> tuple[float, float]:
    center_x = env_cfg.obstacle_center_x
    center_y = env_cfg.obstacle_center_y
    band = np.abs(positions[:, 0] - center_x) < 0.24
    signs = np.sign(positions[band, 1] - center_y)
    signs = signs[signs != 0]
    if signs.size == 0:
        return 0.0, 0.0
    switches = float(np.sum(signs[1:] * signs[:-1] < 0))
    wrong = float(np.mean(signs != mode))
    return switches, wrong


def action_jerk_np(actions: np.ndarray) -> float:
    if len(actions) < 3:
        return 0.0
    jerk = actions[2:] - 2.0 * actions[1:-1] + actions[:-2]
    return float(np.linalg.norm(jerk, axis=-1).mean())


@torch.no_grad()
def predict_chunk(
    model: torch.nn.Module,
    context_states: np.ndarray,
    context_actions: np.ndarray,
    device: torch.device,
    deterministic: bool,
    sample_strategy: str = "mean",
) -> np.ndarray:
    batch = {
        "context_states": torch.from_numpy(context_states[None]).to(device=device, dtype=torch.float32),
        "context_actions": torch.from_numpy(context_actions[None]).to(device=device, dtype=torch.float32),
    }
    out = model(batch, deterministic=deterministic)
    if sample_strategy == "sample" and "particles" in out:
        particle_idx = torch.randint(out["particles"].shape[1], (1,), device=device).item()
        return out["particles"][0, particle_idx].detach().cpu().numpy()
    return out["actions"][0].detach().cpu().numpy()


def rollout_policy(
    model: torch.nn.Module,
    data_file: Path,
    device: torch.device,
    cfg: config_dict.ConfigDict,
    env_cfg: PointObstacleConfig,
    run_dir: Path,
) -> dict[str, float]:
    arrays = load_npz_arrays(data_file)
    states = arrays["states"]
    actions = arrays["actions"]
    modes = arrays["modes"]
    n_train = int(cfg.data.train_fraction * len(states))
    test_ids = list(range(n_train, len(states)))
    if cfg.eval.rollout_episodes > 0:
        test_ids = test_ids[: cfg.eval.rollout_episodes]

    model.eval()
    episode_rows = []
    plotted = []
    for traj_id in test_ids:
        c = cfg.data.context
        max_steps = cfg.data.trajectory_length - c
        current_state = states[traj_id, c].copy()
        context_states = states[traj_id, c - c + 1 : c + 1].copy()
        context_actions = actions[traj_id, c - c : c].copy()
        mode = int(modes[traj_id])

        executed_actions = []
        positions = [current_state[:2].copy()]
        obstacle_crosses = []
        obstacle_contacts = []
        wall_contacts = []
        chunk_discontinuities = []

        t = 0
        while t < max_steps:
            chunk = predict_chunk(
                model,
                context_states,
                context_actions,
                device,
                cfg.eval.deterministic,
                sample_strategy=cfg.eval.get("policy_sample", "mean"),
            )
            steps = min(cfg.eval.replan_every, len(chunk), max_steps - t)
            for local_idx in range(steps):
                raw_action = chunk[local_idx]
                action = clip_action(raw_action, env_cfg)
                prev_action = context_actions[-1]
                if local_idx == 0:
                    chunk_discontinuities.append(float(np.linalg.norm(action - prev_action)))
                prev_pos = current_state[:2].copy()
                current_state, contacts = step_state(current_state, action, env_cfg)
                crossed = segment_crosses_obstacle(prev_pos, current_state[:2], env_cfg)
                executed_actions.append(action.copy())
                positions.append(current_state[:2].copy())
                obstacle_crosses.append(float(crossed))
                wall_contacts.append(float(contacts[0]))
                obstacle_contacts.append(float(contacts[1]))

                context_states = np.concatenate([context_states[1:], current_state[None]], axis=0)
                context_actions = np.concatenate([context_actions[1:], action[None]], axis=0)
                t += 1
                if t >= max_steps:
                    break

        executed = np.asarray(executed_actions, dtype=np.float32)
        pos = np.asarray(positions, dtype=np.float32)
        goal = current_state[2:]
        final_distance = float(np.linalg.norm(current_state[:2] - goal))
        success = float(final_distance <= env_cfg.success_radius)
        path_length = float(np.linalg.norm(pos[1:] - pos[:-1], axis=-1).sum()) if len(pos) > 1 else 0.0
        switches, wrong_side = mode_stats(pos, mode, env_cfg)
        row = {
            "success_rate": success,
            "final_distance": final_distance,
            "path_length": path_length,
            "jerk": action_jerk_np(executed),
            "chunk_discontinuity": float(np.mean(chunk_discontinuities)) if chunk_discontinuities else 0.0,
            "obstacle_cross_rate": float(np.mean(obstacle_crosses)) if obstacle_crosses else 0.0,
            "obstacle_contact_rate": float(np.mean(obstacle_contacts)) if obstacle_contacts else 0.0,
            "wall_contact_rate": float(np.mean(wall_contacts)) if wall_contacts else 0.0,
            "mode_switches": switches,
            "wrong_side_fraction": wrong_side,
        }
        episode_rows.append(row)
        if len(plotted) < cfg.eval.plot_examples:
            plotted.append({"positions": pos, "goal": goal.copy(), "mode": mode, "success": success})

    plot_rollouts(plotted, env_cfg, run_dir / "rollouts.png")
    return {
        f"rollout_{key}": float(np.mean([row[key] for row in episode_rows])) if episode_rows else 0.0
        for key in episode_rows[0].keys()
    }


def plot_rollouts(examples: list[dict], env_cfg: PointObstacleConfig, path: Path) -> None:
    import matplotlib.pyplot as plt

    if not examples:
        return
    cols = 4
    rows = int(np.ceil(len(examples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.asarray(axes).reshape(-1)
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    ox = env_cfg.obstacle_center_x + env_cfg.obstacle_radius * np.cos(theta)
    oy = env_cfg.obstacle_center_y + env_cfg.obstacle_radius * np.sin(theta)
    for idx, ax in enumerate(axes):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= len(examples):
            ax.axis("off")
            continue
        ax.plot(ox, oy, color="black", linewidth=1)
        ex = examples[idx]
        pos = ex["positions"]
        ax.plot(pos[:, 0], pos[:, 1], "-o", markersize=2, linewidth=1.5)
        ax.scatter([pos[0, 0]], [pos[0, 1]], color="tab:blue", s=18)
        ax.scatter([ex["goal"][0]], [ex["goal"][1]], color="tab:green", marker="*", s=60)
        side = "top" if ex["mode"] > 0 else "bottom"
        status = "ok" if ex["success"] else "fail"
        ax.set_title(f"{side} / {status}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def path_side(positions: np.ndarray, env_cfg: PointObstacleConfig) -> int:
    """Classify a rolled-out path as top (+1), bottom (-1), or unresolved (0)."""

    center_y = env_cfg.obstacle_center_y
    band = np.abs(positions[:, 0] - env_cfg.obstacle_center_x) < 0.26
    signs = np.sign(positions[band, 1] - center_y)
    signs = signs[signs != 0]
    if signs.size == 0:
        final_sign = float(np.sign(positions[-1, 1] - center_y))
        return int(final_sign) if final_sign != 0 else 0
    vote = float(np.sign(np.mean(signs)))
    return int(vote) if vote != 0 else 0


def path_has_side_switch(positions: np.ndarray, env_cfg: PointObstacleConfig) -> bool:
    center_y = env_cfg.obstacle_center_y
    relevant = (positions[:, 0] > env_cfg.shared_prefix_target_x) & (np.abs(positions[:, 1] - center_y) > 0.03)
    signs = np.sign(positions[relevant, 1] - center_y)
    signs = signs[np.abs(signs) > 0]
    if signs.size < 2:
        return False
    return bool(np.any(signs[1:] * signs[:-1] < 0))


@torch.no_grad()
def predict_particle_chunks(
    model: torch.nn.Module,
    context_states: np.ndarray,
    context_actions: np.ndarray,
    device: torch.device,
    deterministic: bool,
    fallback_samples: int,
) -> np.ndarray:
    batch = {
        "context_states": torch.from_numpy(context_states[None]).to(device=device, dtype=torch.float32),
        "context_actions": torch.from_numpy(context_actions[None]).to(device=device, dtype=torch.float32),
    }
    out = model(batch, deterministic=deterministic)
    if "particles" in out:
        return out["particles"][0].detach().cpu().numpy()
    samples = []
    for _ in range(fallback_samples):
        out = model(batch, deterministic=deterministic)
        samples.append(out["actions"][0].detach().cpu().numpy())
    return np.stack(samples, axis=0)


def rollout_open_loop(
    state: np.ndarray,
    chunk: np.ndarray,
    env_cfg: PointObstacleConfig,
) -> tuple[np.ndarray, float]:
    current = state.copy()
    positions = [current[:2].copy()]
    contacts = []
    for raw_action in chunk:
        current, contact = step_state(current, clip_action(raw_action, env_cfg), env_cfg)
        positions.append(current[:2].copy())
        contacts.append(float(contact[1]))
    return np.asarray(positions, dtype=np.float32), float(np.mean(contacts)) if contacts else 0.0


def mode_entropy(top_count: int, bottom_count: int) -> float:
    total = top_count + bottom_count
    if total == 0:
        return 0.0
    probs = np.array([top_count / total, bottom_count / total], dtype=np.float64)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def plot_multimodal_samples(
    examples: list[dict],
    env_cfg: PointObstacleConfig,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if not examples:
        return
    cols = 4
    rows = int(np.ceil(len(examples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    ox = env_cfg.obstacle_center_x + env_cfg.obstacle_radius * np.cos(theta)
    oy = env_cfg.obstacle_center_y + env_cfg.obstacle_radius * np.sin(theta)
    for idx, ax in enumerate(axes):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= len(examples):
            ax.axis("off")
            continue
        ax.plot(ox, oy, color="black", linewidth=1)
        ex = examples[idx]
        for positions, side in zip(ex["positions"], ex["sides"], strict=True):
            color = "tab:blue" if side > 0 else "tab:orange" if side < 0 else "0.55"
            ax.plot(positions[:, 0], positions[:, 1], color=color, alpha=0.35, linewidth=1.0)
        start = ex["current_state"][:2]
        goal = ex["current_state"][2:]
        ax.scatter([start[0]], [start[1]], color="black", s=18)
        ax.scatter([goal[0]], [goal[1]], color="tab:green", marker="*", s=60)
        ax.set_title(
            f"top {ex['top']} / bottom {ex['bottom']} / H {ex['entropy']:.2f}",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def evaluate_multimodality(
    model: torch.nn.Module,
    data_file: Path,
    device: torch.device,
    cfg: config_dict.ConfigDict,
    env_cfg: PointObstacleConfig,
    run_dir: Path,
) -> dict[str, float]:
    examples_to_plot = int(cfg.eval.get("multimodal_examples", 0))
    if examples_to_plot <= 0:
        return {}

    arrays = load_npz_arrays(data_file)
    states = arrays["states"]
    actions = arrays["actions"]
    n_train = int(cfg.data.train_fraction * len(states))
    test_ids = list(range(n_train, len(states)))
    t = max(int(cfg.data.context), int(cfg.data.get("shared_prefix_steps", cfg.data.context)))
    if t >= actions.shape[1]:
        return {}

    model.eval()
    seen_contexts = set()
    rows = []
    plotted = []
    fallback_samples = int(cfg.eval.get("multimodal_samples", cfg.model.get("particles", 16)))
    for traj_id in test_ids:
        if len(plotted) >= examples_to_plot:
            break
        c = cfg.data.context
        context_key = tuple(np.round(states[traj_id, t], 3))
        if context_key in seen_contexts:
            continue
        seen_contexts.add(context_key)
        context_states = states[traj_id, t - c + 1 : t + 1].copy()
        context_actions = actions[traj_id, t - c : t].copy()
        chunks = predict_particle_chunks(
            model,
            context_states,
            context_actions,
            device,
            deterministic=False,
            fallback_samples=fallback_samples,
        )
        sample_limit = int(cfg.eval.get("multimodal_samples", chunks.shape[0]))
        chunks = chunks[:sample_limit]

        positions = []
        sides = []
        switch_flags = []
        contact_rates = []
        for chunk in chunks:
            pos, contact_rate = rollout_open_loop(states[traj_id, t], chunk, env_cfg)
            side = path_side(pos, env_cfg)
            positions.append(pos)
            sides.append(side)
            switch_flags.append(float(path_has_side_switch(pos, env_cfg)))
            contact_rates.append(contact_rate)

        top = int(np.sum(np.asarray(sides) > 0))
        bottom = int(np.sum(np.asarray(sides) < 0))
        unresolved = int(np.sum(np.asarray(sides) == 0))
        flat_chunks = chunks.reshape(chunks.shape[0], -1)
        if len(flat_chunks) > 1:
            diffs = flat_chunks[:, None, :] - flat_chunks[None, :, :]
            pairwise = np.sqrt(np.maximum((diffs * diffs).sum(axis=-1), 0.0))
            diversity = float(pairwise[np.triu_indices(len(flat_chunks), k=1)].mean())
        else:
            diversity = 0.0
        entropy = mode_entropy(top, bottom)
        row = {
            "top_fraction": top / max(1, top + bottom + unresolved),
            "bottom_fraction": bottom / max(1, top + bottom + unresolved),
            "unresolved_fraction": unresolved / max(1, top + bottom + unresolved),
            "mode_entropy": entropy,
            "mode_switch_rate": float(np.mean(switch_flags)) if switch_flags else 0.0,
            "obstacle_contact_rate": float(np.mean(contact_rates)) if contact_rates else 0.0,
            "sample_path_diversity": diversity,
        }
        rows.append(row)
        plotted.append(
            {
                "positions": positions,
                "sides": sides,
                "current_state": states[traj_id, t].copy(),
                "top": top,
                "bottom": bottom,
                "entropy": entropy,
            }
        )

    plot_multimodal_samples(plotted, env_cfg, run_dir / "multimodal_samples.png")
    if not rows:
        return {}
    return {
        f"multimodal_{key}": float(np.mean([row[key] for row in rows]))
        for key in rows[0].keys()
    }


def paired_partner_id(
    traj_id: int,
    states: np.ndarray,
    modes: np.ndarray,
    t: int,
) -> int | None:
    candidates = []
    adjacent = traj_id + 1 if traj_id % 2 == 0 else traj_id - 1
    if 0 <= adjacent < len(states):
        candidates.append(adjacent)
    same_context = np.max(np.abs(states[:, t] - states[traj_id, t]), axis=1) < 1e-5
    candidates.extend(np.flatnonzero(same_context).tolist())
    for candidate in candidates:
        if candidate == traj_id:
            continue
        if int(modes[candidate]) != -int(modes[traj_id]):
            continue
        if np.max(np.abs(states[candidate, t] - states[traj_id, t])) < 1e-5:
            return int(candidate)
    return None


def plot_position_marginals(
    examples: list[dict],
    env_cfg: PointObstacleConfig,
    path: Path,
    time_indices: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    if not examples:
        return
    rows = len(examples)
    cols = len(time_indices)
    fig, axes = plt.subplots(rows, cols, figsize=(2.45 * cols, 2.45 * rows), squeeze=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 120)
    ox = env_cfg.obstacle_center_x + env_cfg.obstacle_radius * np.cos(theta)
    oy = env_cfg.obstacle_center_y + env_cfg.obstacle_radius * np.sin(theta)
    for row_idx, ex in enumerate(examples):
        generated = ex["generated_positions"]
        expert = ex["expert_positions"]
        for col_idx, step in enumerate(time_indices):
            ax = axes[row_idx, col_idx]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.plot(ox, oy, color="black", linewidth=0.8)
            gen_pos = generated[:, step]
            ax.scatter(gen_pos[:, 0], gen_pos[:, 1], color="0.35", alpha=0.55, s=10)
            ax.scatter(expert[0, step, 0], expert[0, step, 1], color="tab:blue", marker="x", s=45)
            ax.scatter(expert[1, step, 0], expert[1, step, 1], color="tab:orange", marker="x", s=45)
            ax.scatter(ex["start"][0], ex["start"][1], color="black", s=12)
            if row_idx == 0:
                ax.set_title(f"k={int(step)}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def evaluate_position_marginals(
    model: torch.nn.Module,
    data_file: Path,
    device: torch.device,
    cfg: config_dict.ConfigDict,
    env_cfg: PointObstacleConfig,
    run_dir: Path,
) -> dict[str, float]:
    examples_to_plot = int(cfg.eval.get("marginal_examples", 0))
    if examples_to_plot <= 0:
        return {}

    arrays = load_npz_arrays(data_file)
    states = arrays["states"]
    actions = arrays["actions"]
    modes = arrays["modes"]
    n_train = int(cfg.data.train_fraction * len(states))
    test_ids = list(range(n_train, len(states)))
    t = max(int(cfg.data.context), int(cfg.data.get("shared_prefix_steps", cfg.data.context)))
    horizon = int(cfg.data.horizon)
    if t + horizon >= states.shape[1]:
        return {}

    model.eval()
    rows = []
    plotted = []
    seen_contexts = set()
    sample_limit = int(cfg.eval.get("marginal_samples", cfg.model.get("particles", 16)))
    time_slices = int(cfg.eval.get("marginal_time_slices", 6))
    time_indices = np.unique(np.linspace(0, horizon, min(horizon + 1, time_slices), dtype=np.int64))
    for traj_id in test_ids:
        if len(plotted) >= examples_to_plot:
            break
        partner = paired_partner_id(traj_id, states, modes, t)
        if partner is None:
            continue
        context_key = tuple(np.round(states[traj_id, t], 4))
        if context_key in seen_contexts:
            continue
        seen_contexts.add(context_key)
        top_id = traj_id if int(modes[traj_id]) > 0 else partner
        bottom_id = partner if top_id == traj_id else traj_id
        c = cfg.data.context
        context_states = states[traj_id, t - c + 1 : t + 1].copy()
        context_actions = actions[traj_id, t - c : t].copy()
        chunks = predict_particle_chunks(
            model,
            context_states,
            context_actions,
            device,
            deterministic=False,
            fallback_samples=sample_limit,
        )[:sample_limit]

        generated_positions = []
        for chunk in chunks:
            positions, _ = rollout_open_loop(states[traj_id, t], chunk, env_cfg)
            generated_positions.append(positions)
        generated_positions = np.asarray(generated_positions, dtype=np.float32)
        expert_positions = np.stack(
            [
                states[top_id, t : t + horizon + 1, :2],
                states[bottom_id, t : t + horizon + 1, :2],
            ],
            axis=0,
        )

        nearest_distances = []
        entropies = []
        unresolved = []
        for step in range(horizon + 1):
            gen = generated_positions[:, step]
            gt = expert_positions[:, step]
            dist = np.linalg.norm(gen[:, None, :] - gt[None, :, :], axis=-1)
            nearest_distances.append(float(dist.min(axis=1).mean()))
            signs = np.sign(gen[:, 1] - env_cfg.obstacle_center_y)
            top = int(np.sum(signs > 0))
            bottom = int(np.sum(signs < 0))
            zero = int(np.sum(signs == 0))
            entropies.append(mode_entropy(top, bottom))
            unresolved.append(zero / max(1, len(signs)))

        rows.append(
            {
                "nearest_gt_distance": float(np.mean(nearest_distances)),
                "mode_entropy": float(np.mean(entropies)),
                "unresolved_fraction": float(np.mean(unresolved)),
            }
        )
        plotted.append(
            {
                "generated_positions": generated_positions,
                "expert_positions": expert_positions,
                "start": states[traj_id, t, :2].copy(),
            }
        )

    plot_position_marginals(plotted, env_cfg, run_dir / "position_marginals.png", time_indices)
    if not rows:
        return {}
    return {
        f"position_marginal_{key}": float(np.mean([row[key] for row in rows]))
        for key in rows[0].keys()
    }


def selection_score(metrics: dict[str, float]) -> float:
    if "path_sinkhorn" in metrics:
        return metrics["path_sinkhorn"]
    if "marginal_sinkhorn" in metrics:
        return metrics["marginal_sinkhorn"]
    return metrics["action_mse"] + 0.15 * metrics["action_endpoint_mse"]


def train_from_config(cfg: config_dict.ConfigDict) -> dict[str, float]:
    set_seed(cfg.seed)
    root = experiment_root()
    run_name = cfg.run_name or default_run_name(cfg)
    run_dir = root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = make_env_config(cfg)
    train_loader, test_loader, data_file = make_loaders(root, cfg, env_cfg)
    device = get_device(cfg.device)
    model = make_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    history = []
    best_score = float("inf")
    best_state = None
    best_epoch = -1
    use_batch_bar = cfg.model.type in ("sinkhorn_bridge", "latent_sinkhorn_bridge")
    epoch_iter = range(cfg.train.epochs) if use_batch_bar else tqdm(
        range(cfg.train.epochs),
        desc="epochs",
        dynamic_ncols=True,
    )
    for epoch in epoch_iter:
        model.train()
        sums: dict[str, float] = {}
        count = 0
        batch_iter = train_loader
        if use_batch_bar:
            batch_iter = tqdm(
                train_loader,
                desc=f"epoch {epoch + 1}/{cfg.train.epochs}",
                dynamic_ncols=True,
                leave=True,
                mininterval=1.0,
            )
        for batch in batch_iter:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            loss, parts = compute_loss(batch, model, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            count += 1
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())
            if use_batch_bar:
                batch_iter.set_postfix(
                    loss=f"{sums.get('loss', 0.0) / max(1, count):.4f}",
                    sinkhorn=f"{sums.get('sinkhorn', 0.0) / max(1, count):.4f}",
                    path=f"{sums.get('path_sinkhorn', 0.0) / max(1, count):.4f}",
                    bridge=f"{sums.get('bridge', 0.0) / max(1, count):.4f}",
                    div=f"{sums.get('diversity', 0.0) / max(1, count):.4f}",
                    refresh=False,
                )

        row = {key: value / max(1, count) for key, value in sums.items()}
        eval_metrics = evaluate_actions(model, test_loader, device, cfg)
        row.update({f"test_{key}": value for key, value in eval_metrics.items()})
        row["epoch"] = epoch
        score = selection_score(eval_metrics)
        row["selection_score"] = score
        history.append(row)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if use_batch_bar:
            tqdm.write(
                f"epoch={epoch:03d} loss={row['loss']:.4f} "
                f"action_mse={eval_metrics['action_mse']:.4f} "
                f"disc={eval_metrics['action_chunk_discontinuity']:.4f} "
                f"jerk={eval_metrics['action_jerk']:.4f}"
            )
        else:
            epoch_iter.set_postfix(
                loss=f"{row['loss']:.4f}",
                action_mse=f"{eval_metrics['action_mse']:.4f}",
                disc=f"{eval_metrics['action_chunk_discontinuity']:.4f}",
                jerk=f"{eval_metrics['action_jerk']:.4f}",
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    action_metrics = evaluate_actions(model, test_loader, device, cfg)
    rollout_metrics = rollout_policy(model, data_file, device, cfg, env_cfg, run_dir)
    multimodal_metrics = evaluate_multimodality(model, data_file, device, cfg, env_cfg, run_dir)
    position_marginal_metrics = evaluate_position_marginals(model, data_file, device, cfg, env_cfg, run_dir)
    metrics = {
        **action_metrics,
        **rollout_metrics,
        **multimodal_metrics,
        **position_marginal_metrics,
        "run_name": run_name,
        "model_type": cfg.model.type,
        "init_type": cfg.model.get("init_type", "none"),
        "noise_dim": int(cfg.model.noise_dim),
        "bridge_weight": float(cfg.loss.bridge_weight),
        "jerk_weight": float(cfg.loss.jerk_weight),
        "dataset": str(data_file.relative_to(root)),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
    }

    torch.save({"model": model.state_dict(), "config": cfg_to_dict(cfg)}, run_dir / "model.pt")
    write_csv(run_dir / "losses.csv", history)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg_to_dict(cfg), f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


def main(argv) -> None:
    if len(argv) > 1:
        raise app.UsageError(f"Unknown arguments: {argv[1:]}")
    train_from_config(_CONFIG.value)


if __name__ == "__main__":
    app.run(main)

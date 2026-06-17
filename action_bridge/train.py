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
    sinkhorn_bridge_energy,
    sinkhorn_marginal_matching,
)
from .models import ChunkMLPPolicy, ResidualActionBridgePolicy, SinkhornActionBridgePolicy


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
    return (
        root
        / "data"
        / f"point_modes_n{cfg.data.num_trajectories}_t{cfg.data.trajectory_length}_s{cfg.seed}.npz"
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
    raise ValueError(f"Unknown model type: {cfg.model.type}")


def compute_loss(batch: dict[str, torch.Tensor], model: torch.nn.Module, cfg: config_dict.ConfigDict):
    out = model(batch, deterministic=False)
    pred = out["actions"]
    target = batch["future_actions"]
    if cfg.model.type == "sinkhorn_bridge":
        mean_action_loss = F.mse_loss(pred, target)
        endpoint_loss = F.mse_loss(pred[:, -1], target[:, -1])
        first_loss = F.mse_loss(pred[:, 0], target[:, 0])
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
        total = (
            cfg.loss.sinkhorn_weight * sinkhorn
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
            "action": mean_action_loss.detach(),
            "endpoint": endpoint_loss.detach(),
            "first": first_loss.detach(),
            "bridge": bridge.detach(),
            "jerk": jerk.detach(),
            "diversity": diversity.detach(),
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
            metrics["bridge_energy"] = sinkhorn_bridge_energy(
                out["init_particles"],
                out["particles"],
                out["history"],
                phi_final=cfg.loss.phi_final,
                epsilon=cfg.loss.sinkhorn_epsilon,
                iterations=cfg.loss.sinkhorn_iterations,
                context_weight=cfg.loss.sinkhorn_context_weight,
            )
            metrics["particle_diversity"] = particle_diversity(out["particles"])
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
        ax.plot(ox, oy, color="black", linewidth=1)
        if idx >= len(examples):
            ax.axis("off")
            continue
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


def selection_score(metrics: dict[str, float]) -> float:
    return metrics["action_mse"] + 0.15 * metrics["action_endpoint_mse"]


def train_from_config(cfg: config_dict.ConfigDict) -> dict[str, float]:
    set_seed(cfg.seed)
    root = experiment_root()
    run_name = cfg.run_name or default_run_name(cfg)
    run_dir = root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = PointObstacleConfig()
    train_loader, test_loader, data_file = make_loaders(root, cfg, env_cfg)
    device = get_device(cfg.device)
    model = make_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    history = []
    best_score = float("inf")
    best_state = None
    best_epoch = -1
    epoch_iter = tqdm(range(cfg.train.epochs), desc="epochs", dynamic_ncols=True)
    for epoch in epoch_iter:
        model.train()
        sums: dict[str, float] = {}
        count = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            loss, parts = compute_loss(batch, model, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            count += 1
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())

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
    metrics = {
        **action_metrics,
        **rollout_metrics,
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

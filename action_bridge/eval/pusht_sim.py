"""Gym Push-T simulator closed-loop evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from action_bridge.data.pusht_adapter import denormalize_actions_np, normalize_actions_np, normalize_observations_np
from action_bridge.eval.rollout import generate_chunk, predict_actions
from action_bridge.models.action_bridge_policy import ActionBridgePolicy
from action_bridge.training.common import save_json


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _normalization_stats(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data_cfg = config.get("data", {})
    stats = data_cfg.get("normalization_stats")
    if bool(data_cfg.get("normalize", False)) and stats is not None:
        return stats
    return None


def _normalize_action_grid(grid: torch.Tensor, stats: Optional[Dict[str, Any]]) -> torch.Tensor:
    if stats is None:
        return grid
    mean = torch.as_tensor(stats["action_mean"], dtype=grid.dtype, device=grid.device)
    std = torch.as_tensor(stats["action_std"], dtype=grid.dtype, device=grid.device)
    return (grid - mean) / std


def _denormalize_action_tensor(tensor: torch.Tensor, stats: Optional[Dict[str, Any]]) -> torch.Tensor:
    if stats is None:
        return tensor
    mean = torch.as_tensor(stats["action_mean"], dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(stats["action_std"], dtype=tensor.dtype, device=tensor.device)
    return tensor * std + mean


def _reference_rollout_q_seq(model, batch: Dict[str, torch.Tensor], steps: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not isinstance(model, ActionBridgePolicy) or not bool(getattr(model, "uses_contact_langevin", False)):
        return None, None
    h = model.encode_history(batch["obs_hist"], batch["act_hist"])
    q, p = model.coordinate_adapter.init_qp_from_history(batch)
    q_values = [q]
    for k in range(max(0, int(steps))):
        q, p, _ = model.reference_process.reference_step(q, p, h, k % model.chunk_horizon)
        q_values.append(q)
    q_seq = torch.stack(q_values, dim=1)
    return q_seq, h


def _contact_replan_record(
    model,
    batch: Dict[str, torch.Tensor],
    pred: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    state: np.ndarray,
    act_hist_raw: np.ndarray,
    time_index: int,
    replan_index: int,
) -> Optional[Dict[str, Any]]:
    if not isinstance(model, ActionBridgePolicy) or not bool(getattr(model, "uses_contact_langevin", False)):
        return None
    if "q_seq" not in pred:
        return None
    stats = _normalization_stats(config)
    eval_cfg = config.get("eval", {})
    reference_steps = int(eval_cfg.get("sim_reference_steps", max(48, int(config.get("chunk_horizon", 16)) * 4)))
    default_local_step = int(eval_cfg.get("sim_n_exec", config.get("inference", {}).get("n_exec", 4))) - 1
    local_step = min(int(eval_cfg.get("sim_contact_potential_step", default_local_step)), int(config.get("chunk_horizon", 16)) - 1)
    local_step = max(0, local_step)

    h = model.encode_history(batch["obs_hist"], batch["act_hist"])
    q_seq = pred["q_seq"].detach().cpu()[0]
    q_seq_raw = _denormalize_action_tensor(q_seq, stats).numpy()

    aux_values = []
    for k in range(model.chunk_horizon):
        _, aux = model.reference_process.force(pred["q_seq"][:, k], pred["p_seq"][:, k], h, k)
        aux_values.append(aux)

    def stack_aux(key: str) -> Optional[torch.Tensor]:
        values = [item.get(key) for item in aux_values]
        if any(value is None for value in values):
            return None
        return torch.stack(values, dim=1).detach().cpu()[0]

    m = stack_aux("m")
    k_diag = stack_aux("k_diag")
    gamma = stack_aux("gamma")
    if m is None or k_diag is None or gamma is None:
        return None
    m_raw = _denormalize_action_tensor(m, stats).numpy()

    ref_q_seq, _ = _reference_rollout_q_seq(model, batch, reference_steps)
    ref_q_raw = _denormalize_action_tensor(ref_q_seq.detach().cpu()[0], stats).numpy() if ref_q_seq is not None else None
    return {
        "time_index": int(time_index),
        "replan_index": int(replan_index),
        "local_step": int(min(local_step, m.shape[0] - 1)),
        "state": np.asarray(state, dtype=np.float32).copy(),
        "act_hist": np.asarray(act_hist_raw, dtype=np.float32).copy(),
        "q_seq": q_seq_raw.astype(np.float32),
        "reference_q_seq": ref_q_raw.astype(np.float32) if ref_q_raw is not None else None,
        "m": m.numpy().astype(np.float32),
        "m_path": m_raw.astype(np.float32),
        "k_diag": k_diag.numpy().astype(np.float32),
        "gamma": gamma.numpy().astype(np.float32),
        "reference_steps": int(reference_steps),
    }


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _patch_pymunk_collision_handler() -> None:
    """Make gym-pusht 0.1.x work with pymunk 7.x.

    gym-pusht still calls the pymunk 6 API:
        handler = space.add_collision_handler(a, b)
        handler.post_solve = callback

    pymunk 7 removed that method in favor of:
        space.on_collision(a, b, post_solve=callback)
    """

    try:
        import pymunk
    except ImportError:
        return
    if hasattr(pymunk.Space, "add_collision_handler") or not hasattr(pymunk.Space, "on_collision"):
        return

    class _CollisionHandlerCompat:
        def __init__(self, space, collision_type_a: int, collision_type_b: int):
            object.__setattr__(self, "_space", space)
            object.__setattr__(self, "_collision_type_a", collision_type_a)
            object.__setattr__(self, "_collision_type_b", collision_type_b)
            object.__setattr__(self, "begin", None)
            object.__setattr__(self, "pre_solve", None)
            object.__setattr__(self, "post_solve", None)
            object.__setattr__(self, "separate", None)

        def __setattr__(self, name: str, value) -> None:
            object.__setattr__(self, name, value)
            if name in {"begin", "pre_solve", "post_solve", "separate"}:
                self._refresh()

        def _refresh(self) -> None:
            self._space.on_collision(
                self._collision_type_a,
                self._collision_type_b,
                begin=self.begin,
                pre_solve=self.pre_solve,
                post_solve=self.post_solve,
                separate=self.separate,
            )

    def add_collision_handler(self, collision_type_a: int, collision_type_b: int):
        return _CollisionHandlerCompat(self, collision_type_a, collision_type_b)

    pymunk.Space.add_collision_handler = add_collision_handler


def _make_pusht_env(render_mode: str = "rgb_array", obs_type: str = "state"):
    try:
        _patch_pymunk_collision_handler()
        import gymnasium as gym
        import gym_pusht  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Push-T simulator evaluation requires gym-pusht. Install it in this environment with "
            "`uv pip install gym-pusht`, then rerun with eval.sim_closed_loop=true."
        ) from exc
    return gym.make("gym_pusht/PushT-v0", obs_type=obs_type, render_mode=render_mode)


def _obs_to_state(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        if "state" in obs:
            obs = obs["state"]
        elif "agent_pos" in obs and "environment_state" in obs:
            raise ValueError("Push-T sim eval currently expects obs_type='state', not keypoint observations.")
        else:
            raise ValueError(f"Unsupported Push-T observation dict keys: {sorted(obs.keys())}.")
    state = np.asarray(obs, dtype=np.float32)
    if state.shape[-1] != 5:
        raise ValueError(f"Expected Push-T state observation with shape [5], got {state.shape}.")
    return state


def _reset_env(env, seed: int, reset_to_state: Optional[np.ndarray] = None) -> tuple[np.ndarray, Dict[str, Any]]:
    options = None
    if reset_to_state is not None:
        options = {"reset_to_state": np.asarray(reset_to_state, dtype=np.float32)}
    obs, info = env.reset(seed=int(seed), options=options)
    return _obs_to_state(obs), info


def _info_success(info: Dict[str, Any]) -> bool:
    for key in ["is_success", "success"]:
        if key not in info:
            continue
        value = np.asarray(info[key])
        return bool(value.reshape(-1)[0])
    return False


def _action_bounds(env) -> tuple[np.ndarray, np.ndarray]:
    space = getattr(env, "action_space", None)
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is None or high is None:
        return np.array([0.0, 0.0], dtype=np.float32), np.array([512.0, 512.0], dtype=np.float32)
    return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)


def _generate_chunk_with_commitment(
    model,
    batch: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    z: Optional[torch.Tensor],
    z_emb: Optional[torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    deterministic = bool(config.get("inference", {}).get("deterministic", True))
    if not isinstance(model, ActionBridgePolicy):
        return predict_actions(model, batch, deterministic=deterministic), z, z_emb

    commitment = str(config.get("inference", {}).get("latent_commitment", "chunk"))
    if commitment == "episode" and z_emb is not None:
        pred = generate_chunk(model, batch["obs_hist"], batch["act_hist"], deterministic=deterministic, z=z, z_emb=z_emb)
    elif commitment == "sticky":
        pred = generate_chunk(
            model,
            batch["obs_hist"],
            batch["act_hist"],
            deterministic=deterministic,
            z=z,
            sticky=True,
            kappa=float(config.get("eval", {}).get("sticky_kappa", 2.0)),
            rho_z=float(config.get("eval", {}).get("rho_z", 1.0)),
        )
        z = pred.get("z")
    else:
        pred = generate_chunk(model, batch["obs_hist"], batch["act_hist"], deterministic=deterministic)
    if commitment == "episode" and z_emb is None:
        z = pred.get("z")
        z_emb = pred.get("z_emb")
    return pred, z, z_emb


@torch.no_grad()
def rollout_pusht_sim_episode(model, config: Dict[str, Any], device: torch.device, seed: int) -> Dict[str, Any]:
    eval_cfg = config.get("eval", {})
    obs_history = int(config.get("obs_history", 2))
    action_history = int(config.get("action_history", 2))
    chunk_horizon = int(config.get("chunk_horizon", 16))
    n_exec = int(eval_cfg.get("sim_n_exec", eval_cfg.get("n_exec", config.get("inference", {}).get("n_exec", chunk_horizon))))
    n_exec = max(1, min(n_exec, chunk_horizon))
    max_steps = int(eval_cfg.get("sim_max_steps", 300))
    success_threshold = float(eval_cfg.get("sim_success_threshold", 0.95))
    render_mode = str(eval_cfg.get("sim_render_mode", "rgb_array"))
    obs_type = str(eval_cfg.get("sim_obs_type", "state"))

    env = _make_pusht_env(render_mode=render_mode, obs_type=obs_type)
    low, high = _action_bounds(env)
    try:
        state, _ = _reset_env(env, seed)
        obs_hist = np.repeat(state[None], obs_history, axis=0).astype(np.float32)
        initial_action = state[:2].astype(np.float32)
        act_hist = np.repeat(initial_action[None], action_history, axis=0).astype(np.float32)

        states = [state.copy()]
        actions = []
        rewards = []
        infos = []
        frames = []
        path_kl_values = []
        boundary_values = []
        contact_records = []
        z = None
        z_emb = None
        terminated = False
        truncated = False
        max_reward = -float("inf")
        num_replans = 0

        for _ in range(0, max_steps, n_exec):
            stats = _normalization_stats(config)
            model_obs_hist = normalize_observations_np(obs_hist, stats) if stats is not None else obs_hist
            model_act_hist = normalize_actions_np(act_hist, stats) if stats is not None else act_hist
            batch = {
                "obs_hist": torch.from_numpy(model_obs_hist[None]).to(device=device, dtype=torch.float32),
                "act_hist": torch.from_numpy(model_act_hist[None]).to(device=device, dtype=torch.float32),
            }
            pred, z, z_emb = _generate_chunk_with_commitment(model, batch, config, z, z_emb)
            record = _contact_replan_record(
                model,
                batch,
                pred,
                config,
                state=state,
                act_hist_raw=act_hist,
                time_index=len(actions),
                replan_index=num_replans,
            )
            if record is not None:
                contact_records.append(record)
            pred_actions = pred["actions"][0].detach().cpu().numpy().astype(np.float32)
            if stats is not None:
                pred_actions = denormalize_actions_np(pred_actions, stats)
            execute = min(n_exec, pred_actions.shape[0], max_steps - len(actions))
            if execute <= 0:
                break
            boundary_values.append(float(np.linalg.norm(pred_actions[0] - act_hist[-1])))
            path_kl_steps = pred.get("path_kl_steps")
            if path_kl_steps is not None:
                path_kl_values.append(_as_float(path_kl_steps[0, :execute].sum()))
            num_replans += 1

            for action in pred_actions[:execute]:
                clipped = np.clip(action, low, high).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(clipped)
                state = _obs_to_state(obs)
                actions.append(clipped.copy())
                rewards.append(float(reward))
                infos.append(dict(info))
                states.append(state.copy())
                max_reward = max(max_reward, float(reward))
                if render_mode == "rgb_array":
                    try:
                        frame = env.render()
                        if frame is not None:
                            frames.append(np.asarray(frame))
                    except Exception:
                        pass
                obs_hist = np.concatenate([obs_hist[1:], state[None]], axis=0)
                act_hist = np.concatenate([act_hist[1:], clipped[None]], axis=0)
                if terminated or truncated:
                    break
            if terminated or truncated:
                break

        final_reward = rewards[-1] if rewards else 0.0
        success = bool(terminated) or max_reward >= success_threshold or (bool(infos) and _info_success(infos[-1]))
        return {
            "seed": int(seed),
            "states": np.asarray(states, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "frames": frames,
            "success": success,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "max_reward": float(max_reward if rewards else 0.0),
            "final_reward": float(final_reward),
            "episode_length": float(len(actions)),
            "num_replans": float(num_replans),
            "path_KL_energy": float(np.sum(path_kl_values)) if path_kl_values else 0.0,
            "chunk_boundary_discontinuity": float(np.mean(boundary_values)) if boundary_values else 0.0,
            "contact_records": contact_records,
        }
    finally:
        env.close()


def _tee_polygons(pose: np.ndarray, scale: float = 30.0):
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    length = 4.0
    local_polys = [
        np.array([[-length * scale / 2, scale], [length * scale / 2, scale], [length * scale / 2, 0.0], [-length * scale / 2, 0.0]], dtype=np.float32),
        np.array([[-scale / 2, scale], [-scale / 2, length * scale], [scale / 2, length * scale], [scale / 2, scale]], dtype=np.float32),
    ]
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
    return [poly @ rotation.T + np.array([x, y], dtype=np.float32) for poly in local_polys]


def _draw_tee(ax, pose: np.ndarray, color: str, alpha: float, label: Optional[str] = None, linestyle: str = "-") -> None:
    from matplotlib.patches import Polygon

    for idx, poly in enumerate(_tee_polygons(pose)):
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor=color if linestyle == "-" else "none",
                edgecolor=color,
                linewidth=1.3,
                linestyle=linestyle,
                alpha=alpha,
                label=label if idx == 0 else None,
            )
        )


def plot_pusht_sim_rollouts(rollouts: list[Dict[str, Any]], path: Path, max_episodes: int = 8) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(max_episodes, len(rollouts))
    cols = min(4, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 4.1 * rows), squeeze=False)
    goal_pose = np.array([256.0, 256.0, math.pi / 4], dtype=np.float32)
    for flat_idx, ax in enumerate(axes.ravel()):
        if flat_idx >= count:
            ax.axis("off")
            continue
        rollout = rollouts[flat_idx]
        states = rollout["states"]
        actions = rollout["actions"]
        agent = states[:, :2]
        block = states[:, 2:5]
        _draw_tee(ax, goal_pose, color="tab:green", alpha=0.24, linestyle="--", label="goal T")
        _draw_tee(ax, block[0], color="0.55", alpha=0.25, label="initial T")
        _draw_tee(ax, block[-1], color="tab:orange", alpha=0.45, label="final T")
        ax.plot(agent[:, 0], agent[:, 1], color="tab:purple", linewidth=1.5, label="agent")
        ax.plot(block[:, 0], block[:, 1], color="tab:blue", linewidth=1.2, label="block center")
        if actions.size:
            ax.scatter(actions[:, 0], actions[:, 1], color="black", s=4, alpha=0.4, label="actions")
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_aspect("equal", adjustable="box")
        status = "success" if rollout["success"] else "fail"
        ax.set_title(f"seed={rollout['seed']} | {status} | maxR={rollout['max_reward']:.2f}", fontsize=9)
        if flat_idx == 0:
            ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_pusht_sim_reward_curves(rollouts: list[Dict[str, Any]], path: Path) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for rollout in rollouts:
        rewards = rollout["rewards"]
        if rewards.size:
            ax.plot(np.arange(1, rewards.shape[0] + 1), rewards, alpha=0.45)
    ax.set_xlabel("sim step")
    ax.set_ylabel("coverage reward")
    ax.set_title("Push-T simulator rewards")
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_pusht_sim_frames(rollouts: list[Dict[str, Any]], path: Path, max_episodes: int = 2, frames_per_episode: int = 6) -> None:
    selected = [rollout for rollout in rollouts if rollout.get("frames")]
    if not selected:
        return
    plt = _import_pyplot()
    selected = selected[:max_episodes]
    rows = len(selected)
    cols = int(frames_per_episode)
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows), squeeze=False)
    for row, rollout in enumerate(selected):
        frames = rollout["frames"]
        positions = np.linspace(0, len(frames) - 1, num=min(cols, len(frames)), dtype=int)
        for col in range(cols):
            ax = axes[row, col]
            if col >= len(positions):
                ax.axis("off")
                continue
            idx = int(positions[col])
            ax.imshow(frames[idx])
            ax.set_title(f"seed={rollout['seed']} t={idx + 1}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _select_contact_records(records: list[Dict[str, Any]], max_panels: int) -> list[Dict[str, Any]]:
    if max_panels <= 0 or len(records) <= max_panels:
        return records
    positions = np.linspace(0, len(records) - 1, num=max_panels).round().astype(int)
    selected = []
    seen = set()
    for pos in positions:
        pos = int(pos)
        if pos in seen:
            continue
        selected.append(records[pos])
        seen.add(pos)
    return selected


def plot_pusht_sim_contact_reference(
    rollouts: list[Dict[str, Any]],
    path: Path,
    config: Dict[str, Any],
    max_panels: int = 6,
    grid_size: int = 90,
) -> None:
    selected_rollout = next((rollout for rollout in rollouts if rollout.get("contact_records")), None)
    if selected_rollout is None:
        return
    records = _select_contact_records(list(selected_rollout["contact_records"]), int(max_panels))
    if not records:
        return

    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = _normalization_stats(config)
    xs = torch.linspace(0.0, 512.0, int(grid_size))
    ys = torch.linspace(0.0, 512.0, int(grid_size))
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    raw_grid = torch.stack([xx, yy], dim=-1)
    model_grid = _normalize_action_grid(raw_grid, stats)

    potentials = []
    for record in records:
        local_step = int(record["local_step"])
        m = torch.as_tensor(record["m"][local_step], dtype=torch.float32)
        stiffness = torch.as_tensor(record["k_diag"][local_step], dtype=torch.float32)
        diff = model_grid - m
        potentials.append(0.5 * (stiffness * diff.pow(2)).sum(dim=-1))
    max_potential = float(torch.stack(potentials).max().item())
    levels = torch.linspace(0.0, max(max_potential, 1e-8), 25).numpy()

    cols = len(records)
    fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 4.2), squeeze=False, layout="constrained")
    rollout_states = np.asarray(selected_rollout["states"], dtype=np.float32)
    rollout_actions = np.asarray(selected_rollout["actions"], dtype=np.float32)
    agent_path = rollout_states[:, :2]
    block_path = rollout_states[:, 2:5]
    goal_pose = np.array([256.0, 256.0, math.pi / 4], dtype=np.float32)

    for col, (ax, record, potential) in enumerate(zip(axes.ravel(), records, potentials)):
        image = ax.contourf(xx.numpy(), yy.numpy(), potential.numpy(), levels=levels, cmap="viridis")
        t = int(record["time_index"])
        local_step = int(record["local_step"])
        state = np.asarray(record["state"], dtype=np.float32)
        q_seq = np.asarray(record["q_seq"], dtype=np.float32)
        m_path = np.asarray(record["m_path"], dtype=np.float32)
        ref_q = record.get("reference_q_seq")
        stiffness = np.asarray(record["k_diag"][local_step], dtype=np.float32)
        gamma = np.asarray(record["gamma"][local_step], dtype=np.float32).mean()

        _draw_tee(ax, goal_pose, color="tab:green", alpha=0.24, linestyle="--", label="goal T" if col == 0 else None)
        _draw_tee(ax, state[2:5], color="0.45", alpha=0.35, label="current T" if col == 0 else None)
        ax.plot(block_path[: t + 1, 0], block_path[: t + 1, 1], color="0.4", linewidth=1.1, alpha=0.65, label="block so far" if col == 0 else None)
        ax.plot(agent_path[: t + 1, 0], agent_path[: t + 1, 1], color="tab:purple", linewidth=1.5, label="agent so far" if col == 0 else None)
        if rollout_actions.size:
            ax.scatter(rollout_actions[:t, 0], rollout_actions[:t, 1], color="white", s=6, alpha=0.6, label="executed actions" if col == 0 else None)
        ax.plot(q_seq[:, 0], q_seq[:, 1], color="tab:orange", linestyle="--", linewidth=1.5, label="planned chunk" if col == 0 else None)
        if ref_q is not None:
            ref_q = np.asarray(ref_q, dtype=np.float32)
            ax.plot(ref_q[:, 0], ref_q[:, 1], color="tab:green", linestyle="-.", linewidth=1.7, label=f"reference only ({int(record['reference_steps'])})" if col == 0 else None)
            ax.scatter(ref_q[-1:, 0], ref_q[-1:, 1], color="tab:green", marker="s", s=28, zorder=8, label="reference end" if col == 0 else None)
        ax.plot(m_path[:, 0], m_path[:, 1], color="tab:red", marker="x", markersize=3, linewidth=1.0, alpha=0.75, label="m path" if col == 0 else None)
        ax.scatter([float(m_path[local_step, 0])], [float(m_path[local_step, 1])], color="red", marker="x", s=60, linewidths=2.0, zorder=8, label="m local" if col == 0 else None)
        ax.scatter([float(state[0])], [float(state[1])], color="black", marker="o", s=28, zorder=8, label="agent now" if col == 0 else None)
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"replan={record['replan_index']} | t={t} | k={local_step}\n"
            f"stiff=[{stiffness[0]:.2g},{stiffness[1]:.2g}] | gamma={gamma:.3g}",
            fontsize=8.5,
        )
    axes[0, 0].legend(fontsize=6, loc="upper right")
    fig.colorbar(image, ax=axes[0, :], fraction=0.02, pad=0.015, label="V(q,h,k)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise ValueError(f"Expected frame shape [H,W,3] or [H,W,4], got {array.shape}.")
    return array


def save_pusht_sim_gifs(rollouts: list[Dict[str, Any]], directory: Path, fps: float = 10.0) -> list[str]:
    """Save one animated GIF per rollout that has collected RGB frames."""

    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
    written = []
    for idx, rollout in enumerate(rollouts):
        frames = rollout.get("frames") or []
        if not frames:
            continue
        pil_frames = [Image.fromarray(_frame_to_uint8(frame)) for frame in frames]
        status = "success" if rollout["success"] else "fail"
        path = directory / f"episode_{idx:03d}_seed_{int(rollout['seed'])}_{status}.gif"
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        written.append(str(path))
    return written


def save_pusht_sim_videos(rollouts: list[Dict[str, Any]], directory: Path, fps: float = 10.0) -> list[str]:
    """Save MP4 videos when imageio is available."""

    try:
        import imageio.v3 as iio
    except ImportError:
        return []

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for idx, rollout in enumerate(rollouts):
        frames = rollout.get("frames") or []
        if not frames:
            continue
        array = np.stack([_frame_to_uint8(frame) for frame in frames], axis=0)
        status = "success" if rollout["success"] else "fail"
        path = directory / f"episode_{idx:03d}_seed_{int(rollout['seed'])}_{status}.mp4"
        iio.imwrite(path, array, fps=float(fps))
        written.append(str(path))
    return written


@torch.no_grad()
def evaluate_pusht_sim_model(model, config: Dict[str, Any], device: torch.device, output_dir: Optional[Path] = None) -> Dict[str, float]:
    eval_cfg = config.get("eval", {})
    episodes = int(eval_cfg.get("sim_episodes", 50))
    seed0 = int(eval_cfg.get("sim_seed", config.get("seed", 0)))
    render_episodes = int(eval_cfg.get("sim_render_episodes", 0))
    model.eval()
    rollouts = []
    for episode_idx in range(max(1, episodes)):
        rollout = rollout_pusht_sim_episode(model, config, device, seed=seed0 + episode_idx)
        if episode_idx >= render_episodes:
            rollout = dict(rollout)
            rollout["frames"] = []
        rollouts.append(rollout)

    success = np.asarray([item["success"] for item in rollouts], dtype=np.float32)
    max_rewards = np.asarray([item["max_reward"] for item in rollouts], dtype=np.float32)
    final_rewards = np.asarray([item["final_reward"] for item in rollouts], dtype=np.float32)
    lengths = np.asarray([item["episode_length"] for item in rollouts], dtype=np.float32)
    replans = np.asarray([item["num_replans"] for item in rollouts], dtype=np.float32)
    path_kl = np.asarray([item["path_KL_energy"] for item in rollouts], dtype=np.float32)
    boundary = np.asarray([item["chunk_boundary_discontinuity"] for item in rollouts], dtype=np.float32)
    metrics = {
        "sim_success_rate": float(success.mean()),
        "sim_max_reward": float(max_rewards.mean()),
        "sim_final_reward": float(final_rewards.mean()),
        "sim_episode_length": float(lengths.mean()),
        "sim_num_replans": float(replans.mean()),
        "sim_path_KL_energy": float(path_kl.mean()),
        "sim_chunk_boundary_discontinuity": float(boundary.mean()),
        "sim_episodes": float(len(rollouts)),
    }

    if output_dir is not None:
        figures = output_dir / "figures"
        plot_pusht_sim_rollouts(rollouts, figures / "pusht_sim_rollouts.png", max_episodes=int(eval_cfg.get("sim_plot_episodes", 8)))
        plot_pusht_sim_reward_curves(rollouts, figures / "pusht_sim_rewards.png")
        plot_pusht_sim_frames(
            rollouts,
            figures / "pusht_sim_frames.png",
            max_episodes=max(1, render_episodes),
            frames_per_episode=int(eval_cfg.get("sim_frames_per_episode", 6)),
        )
        plot_pusht_sim_contact_reference(
            rollouts,
            figures / "pusht_sim_contact_reference.png",
            config,
            max_panels=int(eval_cfg.get("sim_contact_panels", 6)),
            grid_size=int(eval_cfg.get("sim_contact_grid_size", 90)),
        )
        gif_paths = []
        video_paths = []
        if bool(eval_cfg.get("sim_save_gifs", True)):
            gif_paths = save_pusht_sim_gifs(
                rollouts,
                figures / "pusht_sim_gifs",
                fps=float(eval_cfg.get("sim_gif_fps", 10.0)),
            )
        if bool(eval_cfg.get("sim_save_videos", False)):
            video_paths = save_pusht_sim_videos(
                rollouts,
                figures / "pusht_sim_videos",
                fps=float(eval_cfg.get("sim_video_fps", 10.0)),
            )
        save_json(
            output_dir / "metrics" / "pusht_sim_metrics.json",
            {
                **metrics,
                "gif_paths": gif_paths,
                "video_paths": video_paths,
                "rollouts": [
                    {
                        "seed": item["seed"],
                        "success": item["success"],
                        "terminated": item["terminated"],
                        "truncated": item["truncated"],
                        "max_reward": item["max_reward"],
                        "final_reward": item["final_reward"],
                        "episode_length": item["episode_length"],
                        "num_replans": item["num_replans"],
                        "path_KL_energy": item["path_KL_energy"],
                        "chunk_boundary_discontinuity": item["chunk_boundary_discontinuity"],
                    }
                    for item in rollouts
                ],
            },
        )
    return metrics

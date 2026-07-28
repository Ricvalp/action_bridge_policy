"""Gym Push-T simulator closed-loop evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

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
    obs_state = batch["obs_hist"][:, -1]
    q_values = [q]
    for k in range(max(0, int(steps))):
        q, p, _ = model.reference_process.reference_step(q, p, h, k % model.chunk_horizon, obs_state=obs_state)
        q_values.append(q)
    q_seq = torch.stack(q_values, dim=1)
    return q_seq, h


def _sample_contact_latent_chunks(
    model,
    batch: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    num_samples: int,
) -> Optional[np.ndarray]:
    if num_samples <= 0:
        return None
    if not isinstance(model, ActionBridgePolicy) or not bool(getattr(model, "uses_contact_langevin", False)):
        return None

    stats = _normalization_stats(config)
    deterministic = bool(config.get("inference", {}).get("deterministic", True))
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        samples = []
        for _ in range(int(num_samples)):
            sample = generate_chunk(model, batch["obs_hist"], batch["act_hist"], deterministic=deterministic)
            q_seq = sample.get("q_seq")
            if q_seq is None:
                return None
            q_seq_raw = _denormalize_action_tensor(q_seq.detach().cpu()[0], stats).numpy()
            samples.append(q_seq_raw.astype(np.float32))
        return np.stack(samples, axis=0)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


def _path_smoothness_stats(paths: list[np.ndarray], prefix: str) -> Dict[str, float]:
    speeds = []
    accels = []
    jerks = []
    for path in paths:
        arr = np.asarray(path, dtype=np.float32)
        if arr.shape[0] >= 2:
            speeds.append(np.linalg.norm(np.diff(arr, axis=0), axis=-1))
        if arr.shape[0] >= 3:
            accels.append(np.linalg.norm(np.diff(arr, n=2, axis=0), axis=-1))
        if arr.shape[0] >= 4:
            jerks.append(np.linalg.norm(np.diff(arr, n=3, axis=0), axis=-1))

    def summarize(name: str, values: list[np.ndarray]) -> Dict[str, float]:
        if not values:
            return {
                f"{prefix}_{name}_mean": 0.0,
                f"{prefix}_{name}_p95": 0.0,
                f"{prefix}_{name}_max": 0.0,
            }
        flat = np.concatenate([np.asarray(item, dtype=np.float32).reshape(-1) for item in values])
        if flat.size == 0:
            return {
                f"{prefix}_{name}_mean": 0.0,
                f"{prefix}_{name}_p95": 0.0,
                f"{prefix}_{name}_max": 0.0,
            }
        return {
            f"{prefix}_{name}_mean": float(np.mean(flat)),
            f"{prefix}_{name}_p95": float(np.percentile(flat, 95)),
            f"{prefix}_{name}_max": float(np.max(flat)),
        }

    return {
        **summarize("velocity", speeds),
        **summarize("acceleration", accels),
        **summarize("jerk", jerks),
    }


def _aggregate_rollout_diagnostics(rollouts: list[Dict[str, Any]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    action_paths = [np.asarray(item["actions"], dtype=np.float32) for item in rollouts if len(item.get("actions", [])) > 0]
    agent_paths = [np.asarray(item["states"], dtype=np.float32)[:, :2] for item in rollouts if len(item.get("states", [])) > 0]
    metrics.update(_path_smoothness_stats(action_paths, "action"))
    metrics.update(_path_smoothness_stats(agent_paths, "agent"))

    mismatches = []
    for rollout in rollouts:
        actions = np.asarray(rollout.get("actions", []), dtype=np.float32)
        states = np.asarray(rollout.get("states", []), dtype=np.float32)
        if actions.size == 0 or states.shape[0] < 2:
            continue
        count = min(actions.shape[0], states.shape[0] - 1)
        mismatches.append(np.linalg.norm(actions[:count] - states[1 : count + 1, :2], axis=-1))
    if mismatches:
        mismatch = np.concatenate(mismatches)
        metrics.update(
            {
                "pusher_action_mismatch_mean": float(np.mean(mismatch)),
                "pusher_action_mismatch_p95": float(np.percentile(mismatch, 95)),
                "pusher_action_mismatch_max": float(np.max(mismatch)),
            }
        )
    else:
        metrics.update({"pusher_action_mismatch_mean": 0.0, "pusher_action_mismatch_p95": 0.0, "pusher_action_mismatch_max": 0.0})

    gamma_values = []
    k_values = []
    q_to_m_values = []
    f_ref_norms = []
    control_norms = []
    ratio_values = []
    damping_power_values = []
    for rollout in rollouts:
        for record in rollout.get("contact_records") or []:
            gamma = record.get("gamma")
            if gamma is not None:
                gamma_values.append(np.asarray(gamma, dtype=np.float32).reshape(-1))
            k_diag = record.get("k_diag")
            if k_diag is not None:
                k_values.append(np.asarray(k_diag, dtype=np.float32).reshape(-1))
            q_to_m = record.get("q_to_m")
            if q_to_m is not None:
                q_to_m_values.append(np.asarray(q_to_m, dtype=np.float32).reshape(-1))
            f_ref_norm = record.get("f_ref_norm")
            if f_ref_norm is not None:
                f_ref_norms.append(np.asarray(f_ref_norm, dtype=np.float32).reshape(-1))
            control_norm = record.get("control_accel_norm")
            if control_norm is not None:
                control_norms.append(np.asarray(control_norm, dtype=np.float32).reshape(-1))
            ratio = record.get("reference_control_ratio")
            if ratio is not None:
                ratio_values.append(np.asarray(ratio, dtype=np.float32).reshape(-1))
            damping_power = record.get("damping_power")
            if damping_power is not None:
                damping_power_values.append(np.asarray(damping_power, dtype=np.float32).reshape(-1))

    def add_distribution(name: str, values: list[np.ndarray]) -> None:
        if not values:
            metrics[f"{name}_mean"] = 0.0
            metrics[f"{name}_std"] = 0.0
            metrics[f"{name}_p05"] = 0.0
            metrics[f"{name}_p50"] = 0.0
            metrics[f"{name}_p95"] = 0.0
            return
        flat = np.concatenate(values).astype(np.float32)
        if flat.size == 0:
            metrics[f"{name}_mean"] = 0.0
            metrics[f"{name}_std"] = 0.0
            metrics[f"{name}_p05"] = 0.0
            metrics[f"{name}_p50"] = 0.0
            metrics[f"{name}_p95"] = 0.0
            return
        metrics[f"{name}_mean"] = float(np.mean(flat))
        metrics[f"{name}_std"] = float(np.std(flat))
        metrics[f"{name}_p05"] = float(np.percentile(flat, 5))
        metrics[f"{name}_p50"] = float(np.percentile(flat, 50))
        metrics[f"{name}_p95"] = float(np.percentile(flat, 95))

    add_distribution("gamma", gamma_values)
    add_distribution("k_diag", k_values)
    add_distribution("q_to_m", q_to_m_values)
    add_distribution("f_ref_norm", f_ref_norms)
    add_distribution("control_accel_norm", control_norms)
    add_distribution("reference_control_ratio", ratio_values)
    add_distribution("damping_power", damping_power_values)

    overpush = np.asarray([float(item.get("overpush", False)) for item in rollouts], dtype=np.float32)
    contact_losses = np.asarray([float(item.get("contact_loss_count", 0.0)) for item in rollouts], dtype=np.float32)
    dist_first_contact = np.asarray(
        [float(item["distance_to_t_before_first_contact"]) for item in rollouts if item.get("distance_to_t_before_first_contact") is not None],
        dtype=np.float32,
    )
    dist_after_final_contact = np.asarray(
        [float(item["distance_to_target_after_final_contact"]) for item in rollouts if item.get("distance_to_target_after_final_contact") is not None],
        dtype=np.float32,
    )
    metrics["overpush_rate"] = float(overpush.mean()) if overpush.size else 0.0
    metrics["contact_loss_count_mean"] = float(contact_losses.mean()) if contact_losses.size else 0.0
    metrics["contact_loss_count_max"] = float(contact_losses.max()) if contact_losses.size else 0.0
    metrics["distance_to_t_before_first_contact_mean"] = float(dist_first_contact.mean()) if dist_first_contact.size else 0.0
    metrics["distance_to_target_after_final_contact_mean"] = float(dist_after_final_contact.mean()) if dist_after_final_contact.size else 0.0
    return metrics


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
    latent_samples = int(eval_cfg.get("sim_latent_samples", 0))
    default_local_step = int(eval_cfg.get("sim_n_exec", config.get("inference", {}).get("n_exec", 4))) - 1
    local_step = min(int(eval_cfg.get("sim_contact_potential_step", default_local_step)), int(config.get("chunk_horizon", 16)) - 1)
    local_step = max(0, local_step)

    h = model.encode_history(batch["obs_hist"], batch["act_hist"])
    obs_state = batch["obs_hist"][:, -1]
    q_seq = pred["q_seq"].detach().cpu()[0]
    q_seq_raw = _denormalize_action_tensor(q_seq, stats).numpy()

    aux_values = []
    f_ref_values = []
    control_accel_values = []
    damping_power_values = []
    q_to_m_values = []
    for k in range(model.chunk_horizon):
        q_k = pred["q_seq"][:, k]
        p_k = pred["p_seq"][:, k]
        f_ref, aux = model.reference_process.force(q_k, p_k, h, k, obs_state=obs_state)
        aux_values.append(aux)
        f_ref_values.append(f_ref)
        controls = pred.get("controls")
        if controls is not None:
            u = controls[:, k]
            sigma = model.reference_process.sigma_like(q_k)
            control_accel = sigma * u if model.reference_process.control_is_whitened else u
        else:
            control_accel = torch.zeros_like(q_k)
        control_accel_values.append(control_accel)
        gamma_k = aux.get("gamma")
        if gamma_k is not None:
            damping_power_values.append((gamma_k * p_k.pow(2)).sum(dim=-1))
        else:
            damping_power_values.append(torch.zeros(q_k.shape[0], dtype=q_k.dtype, device=q_k.device))
        m_k = aux.get("m")
        if m_k is not None:
            q_to_m_values.append(torch.linalg.norm(q_k - m_k, dim=-1))
        else:
            q_to_m_values.append(torch.zeros(q_k.shape[0], dtype=q_k.dtype, device=q_k.device))

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
    b_star_px = stack_aux("b_star_px")
    n_star = stack_aux("n_star")
    m_pre_px = stack_aux("m_pre_px")
    m_push_px = stack_aux("m_push_px")
    m_geo_px = stack_aux("m_geo_px")
    rho_contact = stack_aux("rho_contact")
    rho_goal = stack_aux("rho_goal")
    d_contact = stack_aux("d_contact")
    goal_err = stack_aux("goal_err")
    delta_push = stack_aux("delta_push")
    f_ref = torch.stack(f_ref_values, dim=1).detach().cpu()[0]
    control_accel = torch.stack(control_accel_values, dim=1).detach().cpu()[0]
    f_ref_norm = torch.linalg.norm(f_ref, dim=-1)
    control_accel_norm = torch.linalg.norm(control_accel, dim=-1)
    reference_control_ratio = f_ref_norm / (f_ref_norm + control_accel_norm).clamp_min(1e-8)
    damping_power = torch.stack(damping_power_values, dim=1).detach().cpu()[0]
    q_to_m = torch.stack(q_to_m_values, dim=1).detach().cpu()[0]
    m_raw = _denormalize_action_tensor(m, stats).numpy()

    ref_q_seq, _ = _reference_rollout_q_seq(model, batch, reference_steps)
    ref_q_raw = _denormalize_action_tensor(ref_q_seq.detach().cpu()[0], stats).numpy() if ref_q_seq is not None else None
    sampled_q = _sample_contact_latent_chunks(model, batch, config, latent_samples)
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
        "f_ref": f_ref.numpy().astype(np.float32),
        "control_accel": control_accel.numpy().astype(np.float32),
        "f_ref_norm": f_ref_norm.numpy().astype(np.float32),
        "control_accel_norm": control_accel_norm.numpy().astype(np.float32),
        "reference_control_ratio": reference_control_ratio.numpy().astype(np.float32),
        "damping_power": damping_power.numpy().astype(np.float32),
        "q_to_m": q_to_m.numpy().astype(np.float32),
        "reference_steps": int(reference_steps),
        "latent_sample_q_seq": sampled_q,
        "b_star_px": b_star_px.numpy().astype(np.float32) if b_star_px is not None else None,
        "n_star": n_star.numpy().astype(np.float32) if n_star is not None else None,
        "m_pre_px": m_pre_px.numpy().astype(np.float32) if m_pre_px is not None else None,
        "m_push_px": m_push_px.numpy().astype(np.float32) if m_push_px is not None else None,
        "m_geo_px": m_geo_px.numpy().astype(np.float32) if m_geo_px is not None else None,
        "rho_contact": rho_contact.numpy().astype(np.float32) if rho_contact is not None else None,
        "rho_goal": rho_goal.numpy().astype(np.float32) if rho_goal is not None else None,
        "d_contact": d_contact.numpy().astype(np.float32) if d_contact is not None else None,
        "goal_err": goal_err.numpy().astype(np.float32) if goal_err is not None else None,
        "delta_push": delta_push.numpy().astype(np.float32) if delta_push is not None else None,
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


def _sample_eval_latent(
    model: ActionBridgePolicy,
    batch: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    z: Optional[torch.Tensor],
    z_emb: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    h_emb = model.encode_history(batch["obs_hist"], batch["act_hist"])
    latent_mode = str(config.get("eval", {}).get("sim_latent_mode", "default"))
    if model.latent_type == "categorical":
        if latent_mode in {"prior_mean", "argmax"}:
            return model.sample_prior_z(h_emb, mode="argmax")
        if latent_mode in {"episode_sample", "episode_sticky"} and z_emb is not None:
            return z, z_emb
        return model.sample_prior_z(h_emb, mode="sample")
    if model.latent_type == "continuous":
        if latent_mode == "prior_mean":
            return model.sample_prior_z(h_emb, deterministic_continuous=True)
        if latent_mode in {"episode_sample", "episode_sticky"} and z_emb is not None:
            return z, z_emb
        return model.sample_prior_z(h_emb, deterministic_continuous=False)
    return None, model.zero_z_embedding(h_emb.shape[0], h_emb.device, h_emb.dtype)


@torch.no_grad()
def _generate_contact_chunk_with_intervention(
    policy: ActionBridgePolicy,
    batch: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    z: Optional[torch.Tensor],
    z_emb: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if not bool(getattr(policy, "uses_contact_langevin", False)):
        raise RuntimeError("Contact interventions require a contact_langevin ActionBridgePolicy.")
    intervention = str(config.get("eval", {}).get("sim_intervention", "full_policy"))
    deterministic = bool(config.get("inference", {}).get("deterministic", True))
    h_emb = policy.encode_history(batch["obs_hist"], batch["act_hist"])
    obs_state = batch["obs_hist"][:, -1]
    if z_emb is None:
        z, z_emb = policy.sample_prior_z(h_emb, deterministic_continuous=False)

    adapter = policy.coordinate_adapter
    q, p = adapter.init_qp_from_history({"obs_hist": batch["obs_hist"], "act_hist": batch["act_hist"]})
    q_list = [q]
    p_list = [p]
    controls = []
    path_kl_steps = []
    path_kl = torch.zeros(batch["obs_hist"].shape[0], device=batch["obs_hist"].device, dtype=batch["obs_hist"].dtype)
    f_ref_list = []
    control_accel_list = []

    for k in range(policy.chunk_horizon):
        f_ref, aux = policy.reference_process.force(q, p, h_emb, k, obs_state=obs_state)
        u = policy.contact_control(q, p, h_emb, k, z_emb)
        sigma = policy.reference_process.sigma_like(q)
        control_accel = sigma * u if policy.reference_process.control_is_whitened else u
        u_for_kl = u

        if intervention == "reference_only":
            control_accel = torch.zeros_like(control_accel)
            u_for_kl = torch.zeros_like(u)
        elif intervention == "potential_only":
            control_accel = torch.zeros_like(control_accel)
            u_for_kl = torch.zeros_like(u)
            grad_v = aux.get("grad_v")
            f_ref = -grad_v if grad_v is not None else torch.zeros_like(f_ref)
        elif intervention == "control_only":
            f_ref = torch.zeros_like(f_ref)
        elif intervention == "no_damping":
            grad_v = aux.get("grad_v")
            f_ref = -grad_v if grad_v is not None else torch.zeros_like(f_ref)
        elif intervention == "no_potential":
            gamma = aux.get("gamma")
            f_ref = -gamma * p if gamma is not None else torch.zeros_like(f_ref)
        elif intervention in {"full_policy", "full"}:
            pass
        else:
            raise ValueError(f"Unknown sim_intervention {intervention!r}.")

        if policy.reference_process.control_is_whitened:
            step_path_kl = 0.5 * policy.reference_process.dt * u_for_kl.pow(2).sum(dim=-1)
        else:
            step_path_kl = 0.5 * policy.reference_process.dt * (u_for_kl / sigma).pow(2).sum(dim=-1)
        path_kl = path_kl + step_path_kl
        path_kl_steps.append(step_path_kl)
        controls.append(u_for_kl)
        f_ref_list.append(f_ref)
        control_accel_list.append(control_accel)

        if deterministic or policy.reference_process.deterministic_inference:
            noise = torch.zeros_like(q)
        else:
            noise = (policy.reference_process.dt**0.5) * sigma * torch.randn_like(q)
        p = p + policy.reference_process.dt * (f_ref + control_accel) + noise
        if hasattr(policy.reference_process, "_denorm_action_delta") and getattr(policy.reference_process, "max_step_norm", 0.0) > 0:
            p_px = policy.reference_process._denorm_action_delta(p)
            step_norm = torch.linalg.norm(p_px, dim=-1, keepdim=True)
            scale = (float(policy.reference_process.max_step_norm) / step_norm.clamp_min(1e-8)).clamp_max(1.0)
            p = policy.reference_process._norm_action_delta(p_px * scale)
        q = q + policy.reference_process.dt * p
        if hasattr(policy.reference_process, "_denorm_action") and bool(getattr(policy.reference_process, "is_geometric_pusht", False)):
            q = policy.reference_process._norm_action(policy.reference_process._denorm_action(q).clamp(0.0, 512.0))
        q_list.append(q)
        p_list.append(p)

    q_seq = torch.stack(q_list, dim=1)
    actions = adapter.decode_raw_actions(q_seq)
    return {
        "actions": actions,
        "means": actions,
        "controls": torch.stack(controls, dim=1),
        "q_seq": q_seq,
        "p_seq": torch.stack(p_list, dim=1),
        "z": z,
        "z_emb": z_emb,
        "path_kl_energy": path_kl,
        "path_kl_steps": torch.stack(path_kl_steps, dim=1),
        "f_ref": torch.stack(f_ref_list, dim=1),
        "control_accel": torch.stack(control_accel_list, dim=1),
    }


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

    eval_cfg = config.get("eval", {})
    intervention = str(eval_cfg.get("sim_intervention", "full_policy"))
    latent_mode = str(eval_cfg.get("sim_latent_mode", "default"))
    uses_intervention = intervention not in {"full_policy", "full"}

    def run_with_latent(z_value: Optional[torch.Tensor], z_emb_value: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if uses_intervention:
            return _generate_contact_chunk_with_intervention(model, batch, config, z_value, z_emb_value)
        return generate_chunk(
            model,
            batch["obs_hist"],
            batch["act_hist"],
            deterministic=deterministic,
            z=z_value,
            z_emb=z_emb_value,
        )

    if latent_mode != "default":
        z, z_emb = _sample_eval_latent(model, batch, config, z, z_emb)
        pred = run_with_latent(z, z_emb)
        return pred, pred.get("z"), pred.get("z_emb")

    commitment = str(config.get("inference", {}).get("latent_commitment", "chunk"))
    if commitment == "episode" and z_emb is not None:
        pred = run_with_latent(z, z_emb)
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
        z_emb = pred.get("z_emb")
        if uses_intervention:
            pred = run_with_latent(z, z_emb)
    else:
        if uses_intervention:
            z, z_emb = _sample_eval_latent(model, batch, config, z, z_emb)
            pred = run_with_latent(z, z_emb)
        else:
            pred = generate_chunk(model, batch["obs_hist"], batch["act_hist"], deterministic=deterministic)
    if commitment == "episode" and z_emb is None:
        z = pred.get("z")
        z_emb = pred.get("z_emb")
    return pred, z, z_emb


@torch.no_grad()
def rollout_pusht_sim_episode(
    model,
    config: Dict[str, Any],
    device: torch.device,
    seed: int,
    collect_frames: bool = True,
) -> Dict[str, Any]:
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
    collect_contact = bool(eval_cfg.get("sim_collect_contact_diagnostics", True))

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
            if collect_contact:
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
                if collect_frames and render_mode == "rgb_array":
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
        contacts = np.asarray([float(info.get("n_contacts", 0.0)) > 0.0 for info in infos], dtype=bool)
        contact_loss_count = 0
        distance_to_t_before_first_contact = None
        distance_to_target_after_final_contact = None
        states_np = np.asarray(states, dtype=np.float32)
        if contacts.size:
            contact_loss_count = int(np.logical_and(contacts[:-1], ~contacts[1:]).sum()) if contacts.size > 1 else 0
            contact_indices = np.nonzero(contacts)[0]
            if contact_indices.size:
                first = int(contact_indices[0])
                final = int(contact_indices[-1])
                distance_to_t_before_first_contact = float(np.linalg.norm(states_np[first, :2] - states_np[first, 2:4]))
                goal_xy = np.array([256.0, 256.0], dtype=np.float32)
                distance_to_target_after_final_contact = float(np.linalg.norm(states_np[min(final + 1, states_np.shape[0] - 1), 2:4] - goal_xy))
        overpush = bool(max_reward >= 0.90 and final_reward <= 0.85)
        return {
            "seed": int(seed),
            "states": states_np,
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
            "overpush": overpush,
            "contact_loss_count": float(contact_loss_count),
            "distance_to_t_before_first_contact": distance_to_t_before_first_contact,
            "distance_to_target_after_final_contact": distance_to_target_after_final_contact,
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


def plot_pusht_sim_contact_parameters(
    rollouts: list[Dict[str, Any]],
    path: Path,
    max_replans: Optional[int] = None,
) -> None:
    selected_rollout = next((rollout for rollout in rollouts if rollout.get("contact_records")), None)
    if selected_rollout is None:
        return
    records = list(selected_rollout["contact_records"])
    if max_replans is not None and len(records) > int(max_replans):
        records = _select_contact_records(records, int(max_replans))
    if not records:
        return

    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)

    times = np.asarray([record["time_index"] for record in records], dtype=np.float32)
    local_steps = np.asarray([record["local_step"] for record in records], dtype=np.int64)
    k_diag = np.stack([np.asarray(record["k_diag"], dtype=np.float32) for record in records], axis=0)
    gamma = np.stack([np.asarray(record["gamma"], dtype=np.float32) for record in records], axis=0)
    m_path = np.stack([np.asarray(record["m_path"], dtype=np.float32) for record in records], axis=0)
    q_seq = np.stack([np.asarray(record["q_seq"], dtype=np.float32) for record in records], axis=0)

    gamma_mean = gamma.mean(axis=-1)
    k_mean = k_diag.mean(axis=-1)
    k_anisotropy = np.abs(k_diag[..., 0] - k_diag[..., 1])
    m_motion = np.linalg.norm(np.diff(m_path, axis=1), axis=-1)
    q_to_m = np.linalg.norm(q_seq[:, 1:] - m_path, axis=-1)
    executed_gamma = gamma_mean[np.arange(len(records)), local_steps]
    executed_k0 = k_diag[np.arange(len(records)), local_steps, 0]
    executed_k1 = k_diag[np.arange(len(records)), local_steps, 1]
    executed_k_mean = k_mean[np.arange(len(records)), local_steps]
    executed_q_to_m = q_to_m[np.arange(len(records)), local_steps]

    fig, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), layout="constrained")
    panels = [
        ("gamma over chunk", gamma_mean, "gamma"),
        ("stiffness kx over chunk", k_diag[..., 0], "kx"),
        ("stiffness ky over chunk", k_diag[..., 1], "ky"),
        ("stiffness anisotropy |kx-ky|", k_anisotropy, "|kx-ky|"),
    ]
    for ax, (title, values, label) in zip(axes[:2].ravel(), panels):
        image = ax.imshow(values, aspect="auto", interpolation="nearest", origin="lower")
        ax.set_title(title)
        ax.set_ylabel("replan index")
        ax.set_xlabel("chunk step k")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label=label)

    ax = axes[2, 0]
    ax.plot(times, executed_gamma, color="tab:blue", linewidth=1.6, label="gamma at executed k")
    ax.plot(times, executed_k0, color="tab:orange", linewidth=1.2, label="kx at executed k")
    ax.plot(times, executed_k1, color="tab:green", linewidth=1.2, label="ky at executed k")
    ax.plot(times, executed_k_mean, color="black", linewidth=1.4, alpha=0.7, label="mean k at executed k")
    ax.set_title("Parameters actually used for the executed action")
    ax.set_xlabel("sim step")
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.plot(times, executed_q_to_m, color="tab:red", linewidth=1.5, label="|planned q - m| at executed k")
    ax.plot(times, m_motion.mean(axis=-1), color="tab:purple", linewidth=1.3, label="mean |delta m| over chunk")
    ax.set_title("Attractor use")
    ax.set_xlabel("sim step")
    ax.set_ylabel("pixels")
    ax.legend(fontsize=8)

    fig.savefig(path, dpi=170)
    plt.close(fig)


def _records_with_latent_samples(rollouts: list[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], list[Dict[str, Any]]]:
    selected_rollout = next((rollout for rollout in rollouts if rollout.get("contact_records")), None)
    if selected_rollout is None:
        return None, []
    records = [
        record
        for record in selected_rollout["contact_records"]
        if record.get("latent_sample_q_seq") is not None and len(record.get("latent_sample_q_seq", [])) > 0
    ]
    return selected_rollout, records


def plot_pusht_sim_latent_chunk_samples(
    rollouts: list[Dict[str, Any]],
    path: Path,
    max_panels: int = 6,
) -> None:
    selected_rollout, records = _records_with_latent_samples(rollouts)
    if selected_rollout is None or not records:
        return
    records = _select_contact_records(records, int(max_panels))
    if not records:
        return

    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = len(records)
    fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 4.2), squeeze=False, layout="constrained")
    rollout_states = np.asarray(selected_rollout["states"], dtype=np.float32)
    rollout_actions = np.asarray(selected_rollout["actions"], dtype=np.float32)
    agent_path = rollout_states[:, :2]
    block_path = rollout_states[:, 2:5]
    goal_pose = np.array([256.0, 256.0, math.pi / 4], dtype=np.float32)

    for col, (ax, record) in enumerate(zip(axes.ravel(), records)):
        t = int(record["time_index"])
        local_step = int(record["local_step"])
        state = np.asarray(record["state"], dtype=np.float32)
        actual_q = np.asarray(record["q_seq"], dtype=np.float32)
        samples = np.asarray(record["latent_sample_q_seq"], dtype=np.float32)
        ref_q = record.get("reference_q_seq")

        _draw_tee(ax, goal_pose, color="tab:green", alpha=0.20, linestyle="--", label="goal T" if col == 0 else None)
        _draw_tee(ax, state[2:5], color="0.45", alpha=0.32, label="current T" if col == 0 else None)
        ax.plot(block_path[: t + 1, 0], block_path[: t + 1, 1], color="0.45", linewidth=1.0, alpha=0.6, label="block so far" if col == 0 else None)
        ax.plot(agent_path[: t + 1, 0], agent_path[: t + 1, 1], color="tab:purple", linewidth=1.4, label="agent so far" if col == 0 else None)
        if rollout_actions.size:
            ax.scatter(rollout_actions[:t, 0], rollout_actions[:t, 1], color="0.15", s=6, alpha=0.45, label="executed actions" if col == 0 else None)
        for sample_idx, sample_q in enumerate(samples):
            ax.plot(
                sample_q[:, 0],
                sample_q[:, 1],
                color="tab:blue",
                linewidth=0.9,
                alpha=0.22,
                label="sampled z chunks" if col == 0 and sample_idx == 0 else None,
            )
            ax.scatter(sample_q[-1:, 0], sample_q[-1:, 1], color="tab:blue", s=8, alpha=0.22)
        if ref_q is not None:
            ref_q = np.asarray(ref_q, dtype=np.float32)
            ax.plot(ref_q[:, 0], ref_q[:, 1], color="tab:green", linestyle="-.", linewidth=1.4, alpha=0.8, label="reference only" if col == 0 else None)
        ax.plot(actual_q[:, 0], actual_q[:, 1], color="tab:orange", linestyle="--", linewidth=2.0, label="executed sampled chunk" if col == 0 else None)
        ax.scatter([float(state[0])], [float(state[1])], color="black", marker="o", s=32, zorder=8, label="agent now" if col == 0 else None)
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"replan={record['replan_index']} | t={t} | k={local_step}\n"
            f"{samples.shape[0]} sampled latents",
            fontsize=8.5,
        )
    axes[0, 0].legend(fontsize=6, loc="upper right")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_pusht_sim_latent_spread(
    rollouts: list[Dict[str, Any]],
    path: Path,
) -> None:
    _, records = _records_with_latent_samples(rollouts)
    if not records:
        return
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)

    times = np.asarray([record["time_index"] for record in records], dtype=np.float32)
    local_steps = np.asarray([record["local_step"] for record in records], dtype=np.int64)
    endpoint_spread = []
    executed_step_spread = []
    mean_path_spread = []
    for record, local_step in zip(records, local_steps):
        samples = np.asarray(record["latent_sample_q_seq"], dtype=np.float32)
        mean_path = samples.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(samples - mean_path, axis=-1)
        mean_path_spread.append(float(distances.mean()))
        endpoint_spread.append(float(distances[:, -1].mean()))
        executed_step_spread.append(float(distances[:, int(local_step)].mean()))

    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    ax.plot(times, executed_step_spread, label="spread at executed k", linewidth=1.5)
    ax.plot(times, endpoint_spread, label="final-step spread", linewidth=1.5)
    ax.plot(times, mean_path_spread, label="mean path spread", linewidth=1.5)
    ax.set_xlabel("sim step")
    ax.set_ylabel("mean distance to sample mean, pixels")
    ax.set_title("Diversity from sampling z at each replan")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_pusht_sim_geometric_reference(
    rollouts: list[Dict[str, Any]],
    path: Path,
    max_panels: int = 8,
) -> None:
    selected_rollout = next(
        (
            rollout
            for rollout in rollouts
            if any(record.get("b_star_px") is not None for record in rollout.get("contact_records") or [])
        ),
        None,
    )
    if selected_rollout is None:
        return
    records = [
        record
        for record in selected_rollout.get("contact_records") or []
        if record.get("b_star_px") is not None and record.get("m_pre_px") is not None and record.get("m_push_px") is not None
    ]
    records = _select_contact_records(records, int(max_panels))
    if not records:
        return

    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = min(4, len(records))
    rows = int(math.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows), squeeze=False, layout="constrained")
    rollout_states = np.asarray(selected_rollout["states"], dtype=np.float32)
    rollout_actions = np.asarray(selected_rollout["actions"], dtype=np.float32)
    agent_path = rollout_states[:, :2]
    block_path = rollout_states[:, 2:5]
    goal_pose = np.array([256.0, 256.0, math.pi / 4], dtype=np.float32)

    for ax, record in zip(axes.ravel(), records):
        t = int(record["time_index"])
        local_step = int(record["local_step"])
        state = np.asarray(record["state"], dtype=np.float32)
        q_seq = np.asarray(record["q_seq"], dtype=np.float32)
        ref_q = record.get("reference_q_seq")
        b_star = np.asarray(record["b_star_px"][local_step], dtype=np.float32)
        n_star = np.asarray(record["n_star"][local_step], dtype=np.float32)
        m_pre = np.asarray(record["m_pre_px"][local_step], dtype=np.float32)
        m_push = np.asarray(record["m_push_px"][local_step], dtype=np.float32)
        m_geo = np.asarray(record["m_geo_px"][local_step], dtype=np.float32)
        rho_contact = float(np.asarray(record["rho_contact"], dtype=np.float32)[local_step])
        rho_goal = float(np.asarray(record["rho_goal"], dtype=np.float32)[local_step])
        d_contact = float(np.asarray(record["d_contact"], dtype=np.float32)[local_step])
        delta_push = float(np.asarray(record["delta_push"], dtype=np.float32)[local_step])
        stiffness = np.asarray(record["k_diag"][local_step], dtype=np.float32)
        gamma = float(np.asarray(record["gamma"][local_step], dtype=np.float32).mean())

        _draw_tee(ax, goal_pose, color="tab:green", alpha=0.22, linestyle="--", label="goal T")
        _draw_tee(ax, state[2:5], color="0.45", alpha=0.32, label="current T")
        ax.plot(block_path[: t + 1, 0], block_path[: t + 1, 1], color="0.45", linewidth=1.0, alpha=0.6, label="block so far")
        ax.plot(agent_path[: t + 1, 0], agent_path[: t + 1, 1], color="tab:purple", linewidth=1.4, label="agent so far")
        if rollout_actions.size:
            ax.scatter(rollout_actions[:t, 0], rollout_actions[:t, 1], color="0.15", s=7, alpha=0.45, label="commanded targets")
        ax.plot(q_seq[:, 0], q_seq[:, 1], color="tab:orange", linestyle="--", linewidth=1.6, label="planned chunk")
        if ref_q is not None:
            ref_q = np.asarray(ref_q, dtype=np.float32)
            ax.plot(ref_q[:, 0], ref_q[:, 1], color="tab:green", linestyle="-.", linewidth=1.5, label="reference only")

        ax.scatter([b_star[0]], [b_star[1]], color="tab:red", s=42, marker="o", zorder=8, label="b*")
        ax.arrow(
            float(b_star[0]),
            float(b_star[1]),
            float(28.0 * n_star[0]),
            float(28.0 * n_star[1]),
            color="tab:red",
            width=0.7,
            head_width=6.0,
            length_includes_head=True,
            zorder=8,
        )
        ax.scatter([m_pre[0]], [m_pre[1]], color="tab:blue", marker="s", s=46, zorder=8, label="m_pre")
        ax.scatter([m_push[0]], [m_push[1]], color="tab:orange", marker="D", s=44, zorder=8, label="m_push")
        ax.scatter([m_geo[0]], [m_geo[1]], color="black", marker="x", s=55, linewidths=2.0, zorder=9, label="m_geo")
        ax.scatter([float(state[0])], [float(state[1])], color="black", marker="o", s=28, zorder=9, label="pusher now")
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"replan={record['replan_index']} | t={t} | k={local_step}\n"
            f"rho_c={rho_contact:.2f}, rho_g={rho_goal:.2f}, d={d_contact:.1f}, dp={delta_push:.1f}\n"
            f"K={float(np.mean(stiffness)):.3f}, gamma={gamma:.3f}",
            fontsize=8.2,
        )

    for ax in axes.ravel()[len(records) :]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=5.8, loc="upper right")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_pusht_sim_reference_field_diagnostics(rollouts: list[Dict[str, Any]], path: Path) -> None:
    rows = []
    goal_xy = np.array([256.0, 256.0], dtype=np.float32)
    for rollout in rollouts:
        success = bool(rollout.get("success", False))
        for record in rollout.get("contact_records") or []:
            state = np.asarray(record["state"], dtype=np.float32)
            pusher_block_dist = float(np.linalg.norm(state[:2] - state[2:4]))
            block_goal_dist = float(np.linalg.norm(state[2:4] - goal_xy))
            gamma = np.asarray(record.get("gamma"), dtype=np.float32)
            k_diag = np.asarray(record.get("k_diag"), dtype=np.float32)
            f_ref_norm = np.asarray(record.get("f_ref_norm"), dtype=np.float32)
            control_norm = np.asarray(record.get("control_accel_norm"), dtype=np.float32)
            ratio = np.asarray(record.get("reference_control_ratio"), dtype=np.float32)
            damping_power = np.asarray(record.get("damping_power"), dtype=np.float32)
            q_to_m = np.asarray(record.get("q_to_m"), dtype=np.float32)
            if gamma.size == 0 or k_diag.size == 0:
                continue
            horizon = gamma.shape[0]
            for k in range(horizon):
                rows.append(
                    {
                        "k": int(k),
                        "success": success,
                        "pusher_block_dist": pusher_block_dist,
                        "block_goal_dist": block_goal_dist,
                        "gamma": float(np.mean(gamma[k])),
                        "k_mean": float(np.mean(k_diag[k])),
                        "f_ref_norm": float(f_ref_norm[k]) if f_ref_norm.size > k else 0.0,
                        "control_norm": float(control_norm[k]) if control_norm.size > k else 0.0,
                        "ratio": float(ratio[k]) if ratio.size > k else 0.0,
                        "damping_power": float(damping_power[k]) if damping_power.size > k else 0.0,
                        "q_to_m": float(q_to_m[k]) if q_to_m.size > k else 0.0,
                    }
                )
    if not rows:
        return

    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    k_values = np.asarray([row["k"] for row in rows], dtype=np.int64)
    max_k = int(k_values.max())

    def by_k(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means = []
        lows = []
        highs = []
        for k in range(max_k + 1):
            values = np.asarray([row[name] for row in rows if row["k"] == k], dtype=np.float32)
            if values.size == 0:
                means.append(0.0)
                lows.append(0.0)
                highs.append(0.0)
            else:
                means.append(float(np.mean(values)))
                lows.append(float(np.percentile(values, 10)))
                highs.append(float(np.percentile(values, 90)))
        return np.arange(max_k + 1), np.asarray(means), np.asarray([lows, highs])

    fig, axes = plt.subplots(3, 2, figsize=(13.0, 11.0), layout="constrained")
    panels = [
        ("gamma vs chunk step", "gamma", "gamma"),
        ("mean stiffness K vs chunk step", "k_mean", "K"),
        ("reference/control norm ratio vs chunk step", "ratio", "ratio"),
        ("distance to attractor vs chunk step", "q_to_m", "|q - m|"),
    ]
    for ax, (title, key, ylabel) in zip(axes[:2].ravel(), panels):
        xs, mean, band = by_k(key)
        ax.plot(xs, mean, linewidth=1.8)
        ax.fill_between(xs, band[0], band[1], alpha=0.22)
        ax.set_title(title)
        ax.set_xlabel("chunk step k")
        ax.set_ylabel(ylabel)

    ax = axes[2, 0]
    xs, f_mean, f_band = by_k("f_ref_norm")
    _, c_mean, c_band = by_k("control_norm")
    ax.plot(xs, f_mean, linewidth=1.7, label="||f_R||")
    ax.fill_between(xs, f_band[0], f_band[1], alpha=0.18)
    ax.plot(xs, c_mean, linewidth=1.7, label="||sigma u||")
    ax.fill_between(xs, c_band[0], c_band[1], alpha=0.18)
    ax.set_title("reference and control magnitudes")
    ax.set_xlabel("chunk step k")
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    for success, color, label in [(False, "tab:red", "failure"), (True, "tab:green", "success")]:
        subset = [row for row in rows if bool(row["success"]) == success]
        if not subset:
            continue
        ax.scatter(
            [row["pusher_block_dist"] for row in subset],
            [row["ratio"] for row in subset],
            s=9,
            alpha=0.35,
            color=color,
            label=label,
        )
    ax.set_title("reference/control ratio vs pusher-block distance")
    ax.set_xlabel("pusher-block distance, pixels")
    ax.set_ylabel("ratio")
    ax.legend(fontsize=8)

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


def _rollout_status(rollout: Dict[str, Any]) -> str:
    return "success" if bool(rollout["success"]) else "fail"


def _rollout_artifact_stem(rollout: Dict[str, Any], episode_idx: int) -> str:
    return f"episode_{episode_idx:03d}_seed_{int(rollout['seed'])}_{_rollout_status(rollout)}"


def _rollout_summary(rollout: Dict[str, Any], episode_idx: int) -> Dict[str, Any]:
    return {
        "episode_index": int(episode_idx),
        "seed": int(rollout["seed"]),
        "success": bool(rollout["success"]),
        "terminated": bool(rollout["terminated"]),
        "truncated": bool(rollout["truncated"]),
        "max_reward": float(rollout["max_reward"]),
        "final_reward": float(rollout["final_reward"]),
        "episode_length": float(rollout["episode_length"]),
        "num_replans": float(rollout["num_replans"]),
        "path_KL_energy": float(rollout["path_KL_energy"]),
        "chunk_boundary_discontinuity": float(rollout["chunk_boundary_discontinuity"]),
        "overpush": bool(rollout.get("overpush", False)),
        "contact_loss_count": float(rollout.get("contact_loss_count", 0.0)),
        "distance_to_t_before_first_contact": rollout.get("distance_to_t_before_first_contact"),
        "distance_to_target_after_final_contact": rollout.get("distance_to_target_after_final_contact"),
        "num_contact_records": int(len(rollout.get("contact_records") or [])),
        "num_frames": int(len(rollout.get("frames") or [])),
    }


def _save_contact_records_npz(records: list[Dict[str, Any]], path: Path) -> None:
    if not records:
        return
    arrays: Dict[str, np.ndarray] = {
        "time_index": np.asarray([int(record["time_index"]) for record in records], dtype=np.int64),
        "replan_index": np.asarray([int(record["replan_index"]) for record in records], dtype=np.int64),
        "local_step": np.asarray([int(record["local_step"]) for record in records], dtype=np.int64),
        "reference_steps": np.asarray([int(record.get("reference_steps", 0)) for record in records], dtype=np.int64),
    }
    array_keys = [
        "state",
        "act_hist",
        "q_seq",
        "reference_q_seq",
        "m",
        "m_path",
        "k_diag",
        "gamma",
        "f_ref",
        "control_accel",
        "f_ref_norm",
        "control_accel_norm",
        "reference_control_ratio",
        "damping_power",
        "q_to_m",
        "b_star_px",
        "n_star",
        "m_pre_px",
        "m_push_px",
        "m_geo_px",
        "rho_contact",
        "rho_goal",
        "d_contact",
        "goal_err",
        "delta_push",
        "latent_sample_q_seq",
    ]
    for idx, record in enumerate(records):
        for key in array_keys:
            value = record.get(key)
            if value is not None:
                arrays[f"record_{idx:03d}_{key}"] = np.asarray(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_pusht_sim_rollout_artifacts(rollout: Dict[str, Any], output_dir: Path, episode_idx: int) -> Path:
    """Write raw artifacts for one rollout as soon as the episode finishes."""

    episode_dir = output_dir / "rollouts" / _rollout_artifact_stem(rollout, episode_idx)
    episode_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        episode_dir / "trajectory.npz",
        states=np.asarray(rollout["states"], dtype=np.float32),
        actions=np.asarray(rollout["actions"], dtype=np.float32),
        rewards=np.asarray(rollout["rewards"], dtype=np.float32),
    )
    save_json(episode_dir / "metadata.json", _rollout_summary(rollout, episode_idx))
    _save_contact_records_npz(rollout.get("contact_records") or [], episode_dir / "contact_records.npz")
    return episode_dir


def save_pusht_sim_gif(rollout: Dict[str, Any], episode_idx: int, directory: Path, fps: float = 10.0) -> Optional[str]:
    """Save one animated GIF for one rollout if it has collected RGB frames."""

    frames = rollout.get("frames") or []
    if not frames:
        return None
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
    pil_frames = [Image.fromarray(_frame_to_uint8(frame)) for frame in frames]
    path = directory / f"{_rollout_artifact_stem(rollout, episode_idx)}.gif"
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return str(path)


def save_pusht_sim_video(rollout: Dict[str, Any], episode_idx: int, directory: Path, fps: float = 10.0) -> Optional[str]:
    """Save one MP4 video for one rollout when imageio is available."""

    frames = rollout.get("frames") or []
    if not frames:
        return None
    try:
        import imageio.v3 as iio
    except ImportError:
        return None

    directory.mkdir(parents=True, exist_ok=True)
    array = np.stack([_frame_to_uint8(frame) for frame in frames], axis=0)
    path = directory / f"{_rollout_artifact_stem(rollout, episode_idx)}.mp4"
    iio.imwrite(path, array, fps=float(fps))
    return str(path)


def save_pusht_sim_gifs(rollouts: list[Dict[str, Any]], directory: Path, fps: float = 10.0) -> list[str]:
    """Save one animated GIF per rollout that has collected RGB frames."""

    written = []
    for idx, rollout in enumerate(tqdm(rollouts, desc="Saving Push-T GIFs", unit="gif", dynamic_ncols=True)):
        path = save_pusht_sim_gif(rollout, idx, directory, fps=fps)
        if path is not None:
            written.append(path)
    return written


def save_pusht_sim_videos(rollouts: list[Dict[str, Any]], directory: Path, fps: float = 10.0) -> list[str]:
    """Save MP4 videos when imageio is available."""

    written = []
    for idx, rollout in enumerate(tqdm(rollouts, desc="Saving Push-T videos", unit="video", dynamic_ncols=True)):
        path = save_pusht_sim_video(rollout, idx, directory, fps=fps)
        if path is not None:
            written.append(path)
    return written


@torch.no_grad()
def evaluate_pusht_sim_model(model, config: Dict[str, Any], device: torch.device, output_dir: Optional[Path] = None) -> Dict[str, float]:
    eval_cfg = config.get("eval", {})
    episodes = int(eval_cfg.get("sim_episodes", 50))
    seed0 = int(eval_cfg.get("sim_seed", config.get("seed", 0)))
    render_episodes = int(eval_cfg.get("sim_render_episodes", 0))
    model.eval()
    rollouts = []
    gif_paths: list[str] = []
    video_paths: list[str] = []
    episode_iter = tqdm(
        range(max(1, episodes)),
        desc="Push-T sim rollouts",
        unit="episode",
        dynamic_ncols=True,
    )
    for episode_idx in episode_iter:
        rollout = rollout_pusht_sim_episode(
            model,
            config,
            device,
            seed=seed0 + episode_idx,
            collect_frames=episode_idx < render_episodes,
        )
        rollouts.append(rollout)
        if output_dir is not None:
            save_pusht_sim_rollout_artifacts(rollout, output_dir, episode_idx)
            if bool(eval_cfg.get("sim_save_gifs", True)):
                gif_path = save_pusht_sim_gif(
                    rollout,
                    episode_idx,
                    output_dir / "figures" / "pusht_sim_gifs",
                    fps=float(eval_cfg.get("sim_gif_fps", 10.0)),
                )
                if gif_path is not None:
                    gif_paths.append(gif_path)
            if bool(eval_cfg.get("sim_save_videos", False)):
                video_path = save_pusht_sim_video(
                    rollout,
                    episode_idx,
                    output_dir / "figures" / "pusht_sim_videos",
                    fps=float(eval_cfg.get("sim_video_fps", 10.0)),
                )
                if video_path is not None:
                    video_paths.append(video_path)
            save_json(
                output_dir / "metrics" / "pusht_sim_partial.json",
                {
                    "completed_episodes": len(rollouts),
                    "requested_episodes": max(1, episodes),
                    "rollouts": [_rollout_summary(item, idx) for idx, item in enumerate(rollouts)],
                },
            )
        episode_iter.set_postfix(
            success=int(bool(rollout["success"])),
            max_reward=f"{float(rollout['max_reward']):.3f}",
            length=int(rollout["episode_length"]),
        )

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
    metrics.update(_aggregate_rollout_diagnostics(rollouts))

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
        plot_pusht_sim_contact_parameters(
            rollouts,
            figures / "pusht_sim_contact_parameters.png",
            max_replans=eval_cfg.get("sim_contact_parameter_max_replans", None),
        )
        plot_pusht_sim_latent_chunk_samples(
            rollouts,
            figures / "pusht_sim_latent_chunk_samples.png",
            max_panels=int(eval_cfg.get("sim_latent_sample_panels", 6)),
        )
        plot_pusht_sim_latent_spread(
            rollouts,
            figures / "pusht_sim_latent_spread.png",
        )
        plot_pusht_sim_geometric_reference(
            rollouts,
            figures / "pusht_sim_geometric_reference.png",
            max_panels=int(eval_cfg.get("sim_geometric_panels", 8)),
        )
        plot_pusht_sim_reference_field_diagnostics(
            rollouts,
            figures / "pusht_sim_reference_field_diagnostics.png",
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
                        "overpush": item.get("overpush", False),
                        "contact_loss_count": item.get("contact_loss_count", 0.0),
                        "distance_to_t_before_first_contact": item.get("distance_to_t_before_first_contact"),
                        "distance_to_target_after_final_contact": item.get("distance_to_target_after_final_contact"),
                    }
                    for item in rollouts
                ],
            },
        )
    return metrics

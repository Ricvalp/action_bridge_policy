"""Push-T chunk collection for ContactBridgeSAC."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from action_bridge.data.pusht_adapter import denormalize_actions_np, normalize_actions_np, normalize_observations_np
from action_bridge.eval.pusht_sim import _action_bounds, _info_success, _make_pusht_env, _normalization_stats, _obs_to_state, _reset_env
from action_bridge.rl.costs import compute_bc_cost, compute_ref_cost, compute_ref_cost_mean
from action_bridge.rl.replay import ChunkReplayBuffer


def _model_histories(obs_hist_raw: np.ndarray, act_hist_raw: np.ndarray, stats: Optional[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if stats is None:
        return obs_hist_raw.astype(np.float32), act_hist_raw.astype(np.float32)
    return (
        normalize_observations_np(obs_hist_raw, stats).astype(np.float32),
        normalize_actions_np(act_hist_raw, stats).astype(np.float32),
    )


def _normalize_action_chunk(actions_raw: np.ndarray, stats: Optional[Dict[str, Any]]) -> np.ndarray:
    if stats is None:
        return actions_raw.astype(np.float32)
    return normalize_actions_np(actions_raw, stats).astype(np.float32)


@torch.no_grad()
def policy_chunk(
    policy,
    obs_hist: np.ndarray,
    act_hist: np.ndarray,
    stats: Optional[Dict[str, Any]],
    device: torch.device,
    stochastic_latent: bool,
    deterministic_eval: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    obs_model, act_model = _model_histories(obs_hist, act_hist, stats)
    obs_t = torch.from_numpy(obs_model[None]).to(device=device, dtype=torch.float32)
    act_t = torch.from_numpy(act_model[None]).to(device=device, dtype=torch.float32)
    actions, info = policy.forward_rl(
        obs_t,
        act_t,
        deterministic=bool(deterministic_eval),
        sample_latent=bool(stochastic_latent),
        sample_dynamics_noise=False,
        return_info=True,
    )
    actions_model = actions[0].detach().cpu().numpy().astype(np.float32)
    actions_raw = denormalize_actions_np(actions_model, stats).astype(np.float32) if stats is not None else actions_model
    return actions_model, actions_raw, info


@torch.no_grad()
def collect_pusht_episode(
    policy,
    bc_policy,
    config: Dict[str, Any],
    device: torch.device,
    replay: Optional[ChunkReplayBuffer],
    seed: int,
    n_exec: int,
    gamma_rl: float,
    stochastic_latent: bool = True,
    max_steps: int = 500,
    success_bonus: float = 0.0,
    render: bool = False,
) -> Dict[str, float]:
    stats = _normalization_stats(config)
    obs_history = int(config.get("obs_history", 2))
    action_history = int(config.get("action_history", 2))
    chunk_horizon = int(config.get("chunk_horizon", 16))
    n_exec = max(1, min(int(n_exec), chunk_horizon))
    env = _make_pusht_env(render_mode="rgb_array" if render else "rgb_array", obs_type=str(config.get("eval", {}).get("sim_obs_type", "state")))
    low, high = _action_bounds(env)
    try:
        state, _ = _reset_env(env, int(seed))
        obs_hist = np.repeat(state[None], obs_history, axis=0).astype(np.float32)
        initial_action = state[:2].astype(np.float32)
        act_hist = np.repeat(initial_action[None], action_history, axis=0).astype(np.float32)
        rewards = []
        path_kl_values = []
        bc_cost_values = []
        max_reward = -float("inf")
        terminated = False
        truncated = False
        num_transitions = 0
        frames = []

        for _ in range(0, int(max_steps), n_exec):
            obs_model, act_model = _model_histories(obs_hist, act_hist, stats)
            actions_model, actions_raw, info = policy_chunk(
                policy,
                obs_hist,
                act_hist,
                stats,
                device,
                stochastic_latent=stochastic_latent,
                deterministic_eval=not stochastic_latent,
            )
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_model[None]).to(device=device, dtype=torch.float32)
                act_t = torch.from_numpy(act_model[None]).to(device=device, dtype=torch.float32)
                bc_actions, _ = bc_policy.forward_rl(obs_t, act_t, deterministic=True, sample_latent=False, return_info=True)
                actor_actions_t = torch.from_numpy(actions_model[None]).to(device=device, dtype=torch.float32)
                bc_cost = float(compute_bc_cost(actor_actions_t, bc_actions).detach().cpu().item())
            path_kl = float(compute_ref_cost(info, upto=n_exec).detach().cpu().item())
            path_kl_mean = float(compute_ref_cost_mean(info, upto=n_exec).detach().cpu().item())

            reward_m = 0.0
            discount = 1.0
            coverage_t = rewards[-1] if rewards else 0.0
            steps_executed = 0
            prev_obs_model = obs_model.copy()
            prev_act_model = act_model.copy()
            planned_actions_model = actions_model.copy()
            exec_actions_raw = []

            for action_raw in actions_raw[:n_exec]:
                clipped = np.clip(action_raw, low, high).astype(np.float32)
                obs, reward, terminated, truncated, info_env = env.step(clipped)
                state = _obs_to_state(obs)
                reward = float(reward)
                reward_m += discount * reward
                discount *= float(gamma_rl)
                rewards.append(reward)
                max_reward = max(max_reward, reward)
                exec_actions_raw.append(clipped.copy())
                obs_hist = np.concatenate([obs_hist[1:], state[None]], axis=0)
                act_hist = np.concatenate([act_hist[1:], clipped[None]], axis=0)
                steps_executed += 1
                if render:
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.asarray(frame))
                if terminated or truncated:
                    break

            if steps_executed == 0:
                break

            success = bool(terminated) or max_reward >= float(config.get("eval", {}).get("sim_success_threshold", 0.95))
            if success and success_bonus != 0.0:
                reward_m += float(success_bonus)
            next_obs_model, next_act_model = _model_histories(obs_hist, act_hist, stats)
            exec_actions_raw_arr = np.asarray(exec_actions_raw, dtype=np.float32)
            if exec_actions_raw_arr.shape[0] < n_exec:
                pad = np.repeat(exec_actions_raw_arr[-1][None], n_exec - exec_actions_raw_arr.shape[0], axis=0)
                exec_actions_raw_arr = np.concatenate([exec_actions_raw_arr, pad], axis=0)
            exec_actions_model = _normalize_action_chunk(exec_actions_raw_arr, stats)
            if replay is not None:
                replay.add(
                    obs_hist=prev_obs_model.astype(np.float32),
                    act_hist=prev_act_model.astype(np.float32),
                    exec_actions=exec_actions_model.astype(np.float32),
                    planned_actions=planned_actions_model.astype(np.float32),
                    reward_m=np.asarray(reward_m, dtype=np.float32),
                    next_obs_hist=next_obs_model.astype(np.float32),
                    next_act_hist=next_act_model.astype(np.float32),
                    done=np.asarray(float(terminated or truncated), dtype=np.float32),
                    discount_m=np.asarray(float(gamma_rl) ** steps_executed, dtype=np.float32),
                    path_kl=np.asarray(path_kl, dtype=np.float32),
                    bc_cost=np.asarray(bc_cost, dtype=np.float32),
                    success=np.asarray(float(success), dtype=np.float32),
                    coverage_t=np.asarray(coverage_t, dtype=np.float32),
                    coverage_tp=np.asarray(rewards[-1] if rewards else 0.0, dtype=np.float32),
                )
            num_transitions += 1
            path_kl_values.append(path_kl_mean)
            bc_cost_values.append(bc_cost)
            if terminated or truncated:
                break

        return {
            "episode_return": float(sum(rewards)),
            "episode_length": float(len(rewards)),
            "max_reward": float(max_reward if rewards else 0.0),
            "final_reward": float(rewards[-1] if rewards else 0.0),
            "success": float(bool(terminated) or max_reward >= float(config.get("eval", {}).get("sim_success_threshold", 0.95))),
            "num_transitions": float(num_transitions),
            "path_kl_executed_mean": float(np.mean(path_kl_values)) if path_kl_values else 0.0,
            "bc_cost_mean": float(np.mean(bc_cost_values)) if bc_cost_values else 0.0,
        }
    finally:
        env.close()

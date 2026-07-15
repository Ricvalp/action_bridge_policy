"""Push-T chunk collection for ContactBridgeSAC."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

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


def _summarize_episodes(episodes: list[Dict[str, float]], prefix: str = "") -> Dict[str, float]:
    if not episodes:
        return {
            f"{prefix}episodes": 0.0,
            f"{prefix}episode_return_mean": 0.0,
            f"{prefix}episode_length_mean": 0.0,
            f"{prefix}max_reward_mean": 0.0,
            f"{prefix}final_reward_mean": 0.0,
            f"{prefix}success_rate": 0.0,
        }

    def mean(key: str) -> float:
        return float(np.mean([float(item.get(key, 0.0)) for item in episodes]))

    return {
        f"{prefix}episodes": float(len(episodes)),
        f"{prefix}episode_return_mean": mean("episode_return"),
        f"{prefix}episode_length_mean": mean("episode_length"),
        f"{prefix}max_reward_mean": mean("max_reward"),
        f"{prefix}final_reward_mean": mean("final_reward"),
        f"{prefix}success_rate": mean("success"),
    }


def _reset_worker_state(env, seed: int, obs_history: int, action_history: int) -> Dict[str, Any]:
    state, _ = _reset_env(env, int(seed))
    obs_hist = np.repeat(state[None], obs_history, axis=0).astype(np.float32)
    initial_action = state[:2].astype(np.float32)
    act_hist = np.repeat(initial_action[None], action_history, axis=0).astype(np.float32)
    return {
        "state": state,
        "obs_hist": obs_hist,
        "act_hist": act_hist,
        "rewards": [],
        "max_reward": -float("inf"),
        "episode_steps": 0,
        "episode_return": 0.0,
    }


def _finish_worker_episode(worker: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
    rewards = worker["rewards"]
    max_reward = float(worker["max_reward"] if rewards else 0.0)
    return {
        "episode_return": float(worker["episode_return"]),
        "episode_length": float(worker["episode_steps"]),
        "max_reward": max_reward,
        "final_reward": float(rewards[-1] if rewards else 0.0),
        "success": float(max_reward >= float(config.get("eval", {}).get("sim_success_threshold", 0.95))),
    }


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


@torch.no_grad()
def collect_pusht_vector(
    policy,
    bc_policy,
    config: Dict[str, Any],
    device: torch.device,
    replay: Optional[ChunkReplayBuffer],
    seed: int,
    num_envs: int,
    n_exec: int,
    gamma_rl: float,
    stochastic_latent: bool = True,
    max_episode_steps: int = 500,
    target_env_steps: Optional[int] = None,
    target_episodes: Optional[int] = None,
    success_bonus: float = 0.0,
    show_progress: bool = False,
    progress_desc: str = "Vector collection",
) -> Dict[str, float]:
    """Collect chunk transitions from multiple live Push-T envs.

    This is synchronous vector collection: policy inference is batched on the
    requested device, while the lightweight Gym environments are stepped in a
    Python loop. It is deliberately simple and avoids multiprocessing state
    headaches in the first RL implementation.
    """

    stats = _normalization_stats(config)
    obs_history = int(config.get("obs_history", 2))
    action_history = int(config.get("action_history", 2))
    chunk_horizon = int(config.get("chunk_horizon", 16))
    n_exec = max(1, min(int(n_exec), chunk_horizon))
    num_envs = max(1, int(num_envs))
    target_env_steps = int(target_env_steps) if target_env_steps is not None else None
    target_episodes = int(target_episodes) if target_episodes is not None else None
    if target_env_steps is None and target_episodes is None:
        target_env_steps = num_envs * n_exec

    envs = [_make_pusht_env(render_mode="rgb_array", obs_type=str(config.get("eval", {}).get("sim_obs_type", "state"))) for _ in range(num_envs)]
    lows_highs = [_action_bounds(env) for env in envs]
    workers = [_reset_worker_state(env, int(seed) + idx, obs_history, action_history) for idx, env in enumerate(envs)]
    next_seed = int(seed) + num_envs
    env_steps = 0
    transitions = 0
    path_kl_values = []
    bc_cost_values = []
    completed_episodes: list[Dict[str, float]] = []
    pbar_total = target_episodes if target_episodes is not None else target_env_steps
    pbar = tqdm(total=pbar_total, desc=progress_desc, unit="episode" if target_episodes is not None else "env_step", disable=not show_progress)

    def done_collecting() -> bool:
        if target_episodes is not None and len(completed_episodes) >= target_episodes:
            return True
        if target_env_steps is not None and env_steps >= target_env_steps:
            return True
        return False

    try:
        while not done_collecting():
            obs_raw = np.stack([worker["obs_hist"] for worker in workers], axis=0).astype(np.float32)
            act_raw = np.stack([worker["act_hist"] for worker in workers], axis=0).astype(np.float32)
            obs_model, act_model = _model_histories(obs_raw, act_raw, stats)
            obs_t = torch.from_numpy(obs_model).to(device=device, dtype=torch.float32)
            act_t = torch.from_numpy(act_model).to(device=device, dtype=torch.float32)
            actions, info = policy.forward_rl(
                obs_t,
                act_t,
                deterministic=not bool(stochastic_latent),
                sample_latent=bool(stochastic_latent),
                sample_dynamics_noise=False,
                return_info=True,
            )
            with torch.no_grad():
                bc_actions, _ = bc_policy.forward_rl(obs_t, act_t, deterministic=True, sample_latent=False, return_info=True)
                bc_cost_batch = compute_bc_cost(actions, bc_actions).detach().cpu().numpy().astype(np.float32)
                path_kl_batch = compute_ref_cost(info, upto=n_exec).detach().cpu().numpy().astype(np.float32)
                path_kl_mean_batch = compute_ref_cost_mean(info, upto=n_exec).detach().cpu().numpy().astype(np.float32)
            actions_model = actions.detach().cpu().numpy().astype(np.float32)
            actions_raw = denormalize_actions_np(actions_model, stats).astype(np.float32) if stats is not None else actions_model

            for env_idx, env in enumerate(envs):
                if done_collecting():
                    break
                worker = workers[env_idx]
                low, high = lows_highs[env_idx]
                prev_obs_model = obs_model[env_idx].copy()
                prev_act_model = act_model[env_idx].copy()
                planned_actions_model = actions_model[env_idx].copy()
                reward_m = 0.0
                discount = 1.0
                coverage_t = worker["rewards"][-1] if worker["rewards"] else 0.0
                exec_actions_raw = []
                steps_executed = 0
                terminated = False
                truncated = False

                for action_raw in actions_raw[env_idx, :n_exec]:
                    clipped = np.clip(action_raw, low, high).astype(np.float32)
                    obs, reward, terminated, truncated, _ = env.step(clipped)
                    state = _obs_to_state(obs)
                    reward = float(reward)
                    reward_m += discount * reward
                    discount *= float(gamma_rl)
                    worker["episode_return"] += reward
                    worker["rewards"].append(reward)
                    worker["max_reward"] = max(float(worker["max_reward"]), reward)
                    worker["episode_steps"] += 1
                    env_steps += 1
                    steps_executed += 1
                    exec_actions_raw.append(clipped.copy())
                    worker["state"] = state
                    worker["obs_hist"] = np.concatenate([worker["obs_hist"][1:], state[None]], axis=0)
                    worker["act_hist"] = np.concatenate([worker["act_hist"][1:], clipped[None]], axis=0)
                    if target_env_steps is not None and show_progress:
                        pbar.update(1)
                    if terminated or truncated or worker["episode_steps"] >= int(max_episode_steps):
                        break
                    if target_env_steps is not None and env_steps >= target_env_steps:
                        break

                if steps_executed == 0:
                    continue
                success = bool(terminated) or float(worker["max_reward"]) >= float(config.get("eval", {}).get("sim_success_threshold", 0.95))
                if success and success_bonus != 0.0:
                    reward_m += float(success_bonus)
                next_obs_model, next_act_model = _model_histories(worker["obs_hist"], worker["act_hist"], stats)
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
                        path_kl=np.asarray(path_kl_batch[env_idx], dtype=np.float32),
                        bc_cost=np.asarray(bc_cost_batch[env_idx], dtype=np.float32),
                        success=np.asarray(float(success), dtype=np.float32),
                        coverage_t=np.asarray(coverage_t, dtype=np.float32),
                        coverage_tp=np.asarray(worker["rewards"][-1] if worker["rewards"] else 0.0, dtype=np.float32),
                    )
                transitions += 1
                path_kl_values.append(float(path_kl_mean_batch[env_idx]))
                bc_cost_values.append(float(bc_cost_batch[env_idx]))

                if terminated or truncated or worker["episode_steps"] >= int(max_episode_steps):
                    completed_episodes.append(_finish_worker_episode(worker, config))
                    if target_episodes is not None and show_progress:
                        pbar.update(1)
                    workers[env_idx] = _reset_worker_state(env, next_seed, obs_history, action_history)
                    next_seed += 1

        metrics = _summarize_episodes(completed_episodes)
        metrics.update(
            {
                "env_steps": float(env_steps),
                "num_transitions": float(transitions),
                "num_envs": float(num_envs),
                "path_kl_executed_mean": float(np.mean(path_kl_values)) if path_kl_values else 0.0,
                "bc_cost_mean": float(np.mean(bc_cost_values)) if bc_cost_values else 0.0,
            }
        )
        return metrics
    finally:
        if pbar is not None:
            pbar.close()
        for env in envs:
            env.close()

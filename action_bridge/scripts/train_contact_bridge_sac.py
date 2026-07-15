"""Conservative ContactBridgeSAC fine-tuning for pretrained Push-T bridge actors."""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from ml_collections import ConfigDict
from tqdm.auto import tqdm

from action_bridge.config import apply_overrides, load_config, save_config, to_plain_dict
from action_bridge.eval.pusht_sim import evaluate_pusht_sim_model
from action_bridge.rl.collection import collect_pusht_episode
from action_bridge.rl.costs import compute_bc_cost, compute_ref_cost_mean, linear_schedule
from action_bridge.rl.critics import DoubleChunkQ, soft_update
from action_bridge.rl.replay import ChunkReplayBuffer, ReplayBatch
from action_bridge.training.common import append_csv, build_model, make_run_dir, resolve_device, save_json, seed_everything, tensor_metrics_to_float
from action_bridge.training.train_toy import log_wandb_figures, log_wandb_scalars, maybe_init_wandb


def load_bc_actor(checkpoint: Path, config) -> tuple[torch.nn.Module, Dict]:
    raw = torch.load(checkpoint, map_location="cpu")
    ckpt_config = apply_overrides(raw["config"], [])
    for key, value in config.items():
        if key in {"rl", "logging", "run_id", "output_dir", "resume", "device", "resolved_device", "seed"}:
            ckpt_config[key] = value
    model = build_model(ckpt_config)
    model.load_state_dict(raw["model_state"])
    return model, ckpt_config


def freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def configure_actor_trainable(policy, rl_cfg) -> None:
    for param in policy.parameters():
        param.requires_grad_(False)
    if hasattr(policy, "control_net"):
        for param in policy.control_net.parameters():
            param.requires_grad_(True)
    if getattr(policy, "latent", None) is not None:
        for param in policy.latent.parameters():
            param.requires_grad_(True)
    if bool(rl_cfg.get("train_history_encoder", False)):
        for param in policy.history_encoder.parameters():
            param.requires_grad_(True)
    if not bool(rl_cfg.get("freeze_reference", True)):
        for param in policy.reference_process.parameters():
            param.requires_grad_(True)


def actor_parameters(policy):
    return [param for param in policy.parameters() if param.requires_grad]


def save_rl_checkpoint(path: Path, policy, bc_policy, critics, critic_target, actor_optim, critic_optim, alpha_optim, log_alpha, config, env_steps: int, update_steps: int) -> None:
    del bc_policy
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": policy.state_dict(),
            "critic_state": critics.state_dict(),
            "critic_target_state": critic_target.state_dict(),
            "actor_optimizer_state": actor_optim.state_dict(),
            "critic_optimizer_state": critic_optim.state_dict(),
            "alpha_optimizer_state": alpha_optim.state_dict(),
            "log_alpha": log_alpha.detach().cpu(),
            "config": to_plain_dict(config),
            "env_steps": int(env_steps),
            "update_steps": int(update_steps),
        },
        path,
    )


def batch_to_dict(batch: ReplayBatch) -> Dict[str, torch.Tensor]:
    return {
        "obs_hist": batch.obs_hist,
        "act_hist": batch.act_hist,
        "exec_actions": batch.exec_actions,
        "planned_actions": batch.planned_actions,
        "reward_m": batch.reward_m,
        "next_obs_hist": batch.next_obs_hist,
        "next_act_hist": batch.next_act_hist,
        "done": batch.done,
        "discount_m": batch.discount_m,
        "path_kl": batch.path_kl,
        "bc_cost": batch.bc_cost,
        "success": batch.success,
        "coverage_t": batch.coverage_t,
        "coverage_tp": batch.coverage_tp,
    }


def critic_update(
    policy,
    bc_policy,
    critics: DoubleChunkQ,
    critic_target: DoubleChunkQ,
    batch: ReplayBatch,
    critic_optim,
    alpha: torch.Tensor,
    lambda_bc: float,
    n_exec: int,
    grad_clip: float,
    use_bc_target_actor: bool,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        target_actor = bc_policy if use_bc_target_actor else policy
        next_actions, next_info = target_actor.forward_rl(
            batch.next_obs_hist,
            batch.next_act_hist,
            deterministic=bool(use_bc_target_actor),
            sample_latent=not bool(use_bc_target_actor),
            sample_dynamics_noise=False,
            return_info=True,
        )
        next_exec = next_actions[:, :n_exec]
        next_ref_cost = compute_ref_cost_mean(next_info, upto=n_exec)
        bc_next_actions, _ = bc_policy.forward_rl(batch.next_obs_hist, batch.next_act_hist, deterministic=True, sample_latent=False, return_info=True)
        next_bc_cost = compute_bc_cost(next_actions, bc_next_actions)
        next_q = critic_target.min_q(batch.next_obs_hist, batch.next_act_hist, next_exec)
        if use_bc_target_actor:
            target = batch.reward_m + batch.discount_m * (1.0 - batch.done) * next_q
        else:
            target = batch.reward_m + batch.discount_m * (1.0 - batch.done) * (next_q - alpha.detach() * next_ref_cost - float(lambda_bc) * next_bc_cost)

    q1, q2 = critics(batch.obs_hist, batch.act_hist, batch.exec_actions)
    q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    critic_optim.zero_grad(set_to_none=True)
    q_loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(critics.parameters(), float(grad_clip))
    critic_optim.step()
    td_error = 0.5 * ((q1.detach() - target).abs() + (q2.detach() - target).abs())
    return {
        "critic_loss": q_loss.detach(),
        "q_data_mean": 0.5 * (q1.detach().mean() + q2.detach().mean()),
        "q_target_mean": target.detach().mean(),
        "td_error_mean": td_error.mean(),
        "td_error_p95": torch.quantile(td_error, 0.95),
    }


def actor_alpha_update(
    policy,
    bc_policy,
    critics: DoubleChunkQ,
    batch: ReplayBatch,
    actor_optim,
    alpha_optim,
    log_alpha: torch.Tensor,
    lambda_bc: float,
    target_ref_cost: float,
    n_exec: int,
    actor_grad_clip: float,
    alpha_min: float,
    alpha_max: float,
) -> Dict[str, torch.Tensor]:
    actions, info = policy.forward_rl(
        batch.obs_hist,
        batch.act_hist,
        deterministic=False,
        sample_latent=True,
        sample_dynamics_noise=False,
        return_info=True,
    )
    exec_actions = actions[:, :n_exec]
    ref_cost = compute_ref_cost_mean(info, upto=n_exec)
    with torch.no_grad():
        bc_actions, _ = bc_policy.forward_rl(batch.obs_hist, batch.act_hist, deterministic=True, sample_latent=False, return_info=True)
    bc_cost = compute_bc_cost(actions, bc_actions)
    q_pi = critics.min_q(batch.obs_hist, batch.act_hist, exec_actions)
    alpha = log_alpha.exp().clamp(float(alpha_min), float(alpha_max))
    actor_loss = (alpha.detach() * ref_cost + float(lambda_bc) * bc_cost - q_pi).mean()
    actor_optim.zero_grad(set_to_none=True)
    actor_loss.backward()
    if actor_grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(actor_parameters(policy), float(actor_grad_clip))
    actor_optim.step()

    alpha_loss = -(log_alpha * (ref_cost.detach() - float(target_ref_cost))).mean()
    alpha_optim.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_optim.step()
    with torch.no_grad():
        log_alpha.clamp_(math.log(float(alpha_min)), math.log(float(alpha_max)))
    return {
        "actor_loss": actor_loss.detach(),
        "alpha_loss": alpha_loss.detach(),
        "alpha": log_alpha.exp().detach().clamp(float(alpha_min), float(alpha_max)),
        "ref_cost_mean": ref_cost.detach().mean(),
        "bc_cost_mean": bc_cost.detach().mean(),
        "q_policy_mean": q_pi.detach().mean(),
    }


def run_eval(policy, config, device, run_dir: Path, env_steps: int, wandb_run) -> Dict[str, float]:
    eval_config = deepcopy(config)
    rl_cfg = eval_config.get("rl", {})
    if "eval" not in eval_config:
        eval_config.eval = ConfigDict()
    eval_config.eval.sim_closed_loop = True
    eval_config.eval.sim_episodes = int(rl_cfg.get("eval_episodes", 20))
    eval_config.eval.sim_max_steps = int(rl_cfg.get("eval_max_steps", 500))
    eval_config.eval.sim_n_exec = int(rl_cfg.get("n_exec", 8))
    eval_config.eval.sim_render_episodes = int(rl_cfg.get("eval_render_episodes", 0))
    eval_config.eval.sim_save_gifs = bool(rl_cfg.get("eval_save_gifs", False))
    eval_config.eval.sim_collect_contact_diagnostics = True
    output_dir = run_dir / "eval" / f"env_step_{env_steps:08d}"
    try:
        metrics = evaluate_pusht_sim_model(policy, eval_config, device, output_dir=output_dir)
    except Exception as exc:
        if not bool(rl_cfg.get("continue_eval_on_error", True)):
            raise
        metrics = {"sim_eval_error": 1.0}
        save_json(output_dir / "metrics" / "pusht_sim_error.json", {"env_steps": env_steps, "error": repr(exc)})
    append_csv(run_dir / "metrics" / "sim_eval_metrics.csv", {"env_steps": env_steps, **metrics})
    log_wandb_scalars(wandb_run, metrics, step=env_steps, prefix="sim_eval")
    log_wandb_figures(wandb_run, output_dir / "figures", step=env_steps, prefix="sim_eval")
    policy.train()
    return metrics


def train(config) -> Path:
    rl_cfg = config.get("rl", {})
    checkpoint = rl_cfg.get("checkpoint")
    if checkpoint is None:
        raise ValueError("rl.checkpoint must point to a pretrained ActionBridgePolicy checkpoint.")
    seed_everything(int(config.get("seed", 0)))
    device = resolve_device(str(config.get("device", "cpu")))
    policy, config = load_bc_actor(Path(checkpoint), config)
    config["resolved_device"] = str(device)
    policy = policy.to(device)
    bc_policy = deepcopy(policy).to(device).eval()
    freeze_module(bc_policy)
    configure_actor_trainable(policy, rl_cfg)
    trainable = actor_parameters(policy)
    if not trainable:
        raise RuntimeError("No trainable actor parameters. Check rl.freeze_reference/train modules settings.")

    n_exec = int(rl_cfg.get("n_exec", config.get("inference", {}).get("n_exec", 8)))
    critics = DoubleChunkQ(
        obs_history=int(config.get("obs_history", 2)),
        obs_dim=int(config.get("obs_dim", 5)),
        action_history=int(config.get("action_history", 2)),
        action_dim=int(config.get("action_dim", 2)),
        n_exec=n_exec,
        hidden_dim=int(rl_cfg.get("critic_hidden_dim", 512)),
        depth=int(rl_cfg.get("critic_depth", 3)),
    ).to(device)
    critic_target = critics.make_target().to(device)

    actor_optim = torch.optim.AdamW(trainable, lr=float(rl_cfg.get("actor_lr", 1.0e-5)))
    critic_optim = torch.optim.AdamW(critics.parameters(), lr=float(rl_cfg.get("critic_lr", 3.0e-4)))
    log_alpha = torch.tensor(float(rl_cfg.get("alpha_init", 0.05))).log().to(device).requires_grad_(True)
    alpha_optim = torch.optim.AdamW([log_alpha], lr=float(rl_cfg.get("alpha_lr", 1.0e-4)))
    replay = ChunkReplayBuffer(capacity=int(rl_cfg.get("replay_size", 1000000)), seed=int(config.get("seed", 0)))

    run_dir = make_run_dir(config)
    save_config(config, run_dir / "config.json")
    wandb_run = maybe_init_wandb(config, run_dir)

    try:
        prefill_episodes = int(rl_cfg.get("prefill_bc_episodes", 200))
        gamma_rl = float(rl_cfg.get("gamma", 0.99))
        max_steps = int(rl_cfg.get("eval_max_steps", 500))
        for ep in tqdm(range(prefill_episodes), desc="BC replay prefill", unit="episode"):
            metrics = collect_pusht_episode(
                bc_policy,
                bc_policy,
                config,
                device,
                replay,
                seed=int(config.get("seed", 0)) + ep,
                n_exec=n_exec,
                gamma_rl=gamma_rl,
                stochastic_latent=True,
                max_steps=max_steps,
                success_bonus=float(rl_cfg.get("success_bonus", 0.0)),
            )
            if ep % max(1, int(rl_cfg.get("log_every_prefill_episodes", 10))) == 0:
                append_csv(run_dir / "metrics" / "prefill_metrics.csv", {"episode": ep, "replay_size": len(replay), **metrics})

        critic_pretrain_steps = int(rl_cfg.get("critic_pretrain_steps", 50000))
        batch_size = int(rl_cfg.get("batch_size", 256))
        for step in tqdm(range(1, critic_pretrain_steps + 1), desc="Critic pretrain", unit="update"):
            batch = replay.sample(batch_size, device)
            out = critic_update(
                policy,
                bc_policy,
                critics,
                critic_target,
                batch,
                critic_optim,
                alpha=log_alpha.exp().detach(),
                lambda_bc=0.0,
                n_exec=n_exec,
                grad_clip=float(rl_cfg.get("critic_grad_clip", 1.0)),
                use_bc_target_actor=True,
            )
            soft_update(critics, critic_target, float(rl_cfg.get("target_tau", 0.005)))
            if step % int(rl_cfg.get("log_every_updates", 100)) == 0 or step == 1:
                row = {"phase": "critic_pretrain", "update": step, "replay_size": len(replay)}
                row.update(tensor_metrics_to_float(out))
                append_csv(run_dir / "metrics" / "rl_train_metrics.csv", row)
                log_wandb_scalars(wandb_run, row, step=step, prefix="rl")

        env_steps = 0
        update_steps = 0
        last_eval = -1
        last_ckpt = -1
        total_env_steps = int(rl_cfg.get("total_env_steps", 200000))
        pbar = tqdm(total=total_env_steps, desc="Online ContactBridgeSAC", unit="env_step")
        episode_idx = 0
        while env_steps < total_env_steps:
            collect_metrics = collect_pusht_episode(
                policy,
                bc_policy,
                config,
                device,
                replay,
                seed=int(config.get("seed", 0)) + 100000 + episode_idx,
                n_exec=n_exec,
                gamma_rl=gamma_rl,
                stochastic_latent=bool(rl_cfg.get("stochastic_latent_collection", True)),
                max_steps=max_steps,
                success_bonus=float(rl_cfg.get("success_bonus", 0.0)),
            )
            episode_idx += 1
            steps_this_episode = int(collect_metrics.get("episode_length", 0.0))
            env_steps += steps_this_episode
            pbar.update(max(0, steps_this_episode))
            append_csv(run_dir / "metrics" / "online_collection_metrics.csv", {"env_steps": env_steps, "episode": episode_idx, "replay_size": len(replay), **collect_metrics})

            updates = max(1, int(round(float(rl_cfg.get("updates_per_env_step", 1.0)) * max(1, steps_this_episode))))
            for _ in range(updates):
                update_steps += 1
                batch = replay.sample(batch_size, device)
                lambda_bc = linear_schedule(
                    env_steps,
                    start=float(rl_cfg.get("lambda_bc_start", 10.0)),
                    end=float(rl_cfg.get("lambda_bc_end", 0.5)),
                    duration=int(rl_cfg.get("lambda_bc_anneal_steps", 100000)),
                )
                alpha = log_alpha.exp().detach().clamp(float(rl_cfg.get("alpha_min", 1e-4)), float(rl_cfg.get("alpha_max", 10.0)))
                critic_out = critic_update(
                    policy,
                    bc_policy,
                    critics,
                    critic_target,
                    batch,
                    critic_optim,
                    alpha=alpha,
                    lambda_bc=lambda_bc,
                    n_exec=n_exec,
                    grad_clip=float(rl_cfg.get("critic_grad_clip", 1.0)),
                    use_bc_target_actor=False,
                )
                actor_out = {}
                if update_steps % int(rl_cfg.get("actor_update_delay", 2)) == 0:
                    actor_out = actor_alpha_update(
                        policy,
                        bc_policy,
                        critics,
                        batch,
                        actor_optim,
                        alpha_optim,
                        log_alpha,
                        lambda_bc=lambda_bc,
                        target_ref_cost=float(rl_cfg.get("target_ref_cost", 0.05)),
                        n_exec=n_exec,
                        actor_grad_clip=float(rl_cfg.get("actor_grad_clip", 1.0)),
                        alpha_min=float(rl_cfg.get("alpha_min", 1e-4)),
                        alpha_max=float(rl_cfg.get("alpha_max", 10.0)),
                    )
                soft_update(critics, critic_target, float(rl_cfg.get("target_tau", 0.005)))
                if update_steps % int(rl_cfg.get("log_every_updates", 100)) == 0:
                    row = {"phase": "online", "env_steps": env_steps, "update": update_steps, "replay_size": len(replay), "lambda_bc": lambda_bc}
                    row.update(tensor_metrics_to_float(critic_out))
                    row.update(tensor_metrics_to_float(actor_out))
                    append_csv(run_dir / "metrics" / "rl_train_metrics.csv", row)
                    log_wandb_scalars(wandb_run, row, step=env_steps, prefix="rl")

            if env_steps - last_eval >= int(rl_cfg.get("eval_every_env_steps", 5000)):
                last_eval = env_steps
                run_eval(policy, config, device, run_dir, env_steps, wandb_run)
            if env_steps - last_ckpt >= int(rl_cfg.get("checkpoint_every_env_steps", 10000)):
                last_ckpt = env_steps
                save_rl_checkpoint(run_dir / "checkpoints" / f"env_step_{env_steps:08d}.pt", policy, bc_policy, critics, critic_target, actor_optim, critic_optim, alpha_optim, log_alpha, config, env_steps, update_steps)

        pbar.close()
        save_rl_checkpoint(run_dir / "checkpoints" / "latest.pt", policy, bc_policy, critics, critic_target, actor_optim, critic_optim, alpha_optim, log_alpha, config, env_steps, update_steps)
        replay.save_npz(run_dir / "replay_final.npz")
        run_eval(policy, config, device, run_dir, env_steps, wandb_run)
        print(f"Run directory: {run_dir}")
        return run_dir
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pusht_contact_bridge_sac")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config_name), args.overrides)
    train(config)


if __name__ == "__main__":
    main()

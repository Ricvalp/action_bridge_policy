"""Training losses for path-KL action bridge policies and baselines."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from action_bridge.models.action_bridge_policy import ActionBridgePolicy
from action_bridge.models.baselines import AutoregressiveBCPolicy, DirectChunkBCPolicy
from action_bridge.models.latents import categorical_entropy, categorical_kl, gaussian_kl
from action_bridge.training.schedules import linear_warmup


def teacher_forced_prev_actions(act_hist: torch.Tensor, future_actions: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if k == 0:
        return act_hist[:, -2], act_hist[:, -1]
    if k == 1:
        return act_hist[:, -1], future_actions[:, 0]
    return future_actions[:, k - 2], future_actions[:, k - 1]


def smoothness_losses(actions: torch.Tensor, act_hist: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    if act_hist is not None and act_hist.shape[1] >= 2:
        path = torch.cat([act_hist[:, -2:], actions], dim=1)
    elif act_hist is not None and act_hist.shape[1] >= 1:
        path = torch.cat([act_hist[:, -1:], actions], dim=1)
    else:
        path = actions
    if path.shape[1] >= 2:
        accel = path[:, 1:] - path[:, :-1]
        accel_energy = accel.pow(2).sum(dim=-1).mean()
    else:
        accel_energy = actions.new_zeros(())
    if path.shape[1] >= 4:
        jerk = path[:, 3:] - 3.0 * path[:, 2:-1] + 3.0 * path[:, 1:-2] - path[:, :-3]
        jerk_energy = jerk.pow(2).sum(dim=-1).mean()
    else:
        jerk_energy = actions.new_zeros(())
    return {"acceleration_energy": accel_energy, "jerk_energy": jerk_energy}


def bridge_path_losses_conditioned(
    policy: ActionBridgePolicy,
    h_emb: torch.Tensor,
    act_hist: torch.Tensor,
    future_actions: torch.Tensor,
    z_emb: Optional[torch.Tensor],
    tube_noise_std: float = 0.0,
) -> Dict[str, torch.Tensor]:
    nll = torch.zeros(future_actions.shape[0], device=future_actions.device, dtype=future_actions.dtype)
    path_kl = torch.zeros_like(nll)
    mse = torch.zeros_like(nll)
    for k in range(future_actions.shape[1]):
        a_prevprev, a_prev = teacher_forced_prev_actions(act_hist, future_actions, k)
        if tube_noise_std > 0:
            a_prevprev = a_prevprev + tube_noise_std * torch.randn_like(a_prevprev)
            a_prev = a_prev + tube_noise_std * torch.randn_like(a_prev)
        mu_r, log_sigma = policy.reference_process(a_prev, a_prevprev, h_emb, k)
        u = policy.control(a_prev, a_prevprev, h_emb, k, z_emb)
        mu = mu_r + u
        target = future_actions[:, k]
        sigma = log_sigma.exp().clamp_min(1e-6)
        step_nll = 0.5 * ((target - mu) / sigma).pow(2) + log_sigma
        step_kl = 0.5 * (u / sigma).pow(2)
        nll = nll + step_nll.sum(dim=-1)
        path_kl = path_kl + step_kl.sum(dim=-1)
        mse = mse + (target - mu).pow(2).sum(dim=-1)
    return {"nll": nll, "path_kl": path_kl, "mse": mse / max(1, future_actions.shape[1])}


def bridge_loss(
    policy: ActionBridgePolicy,
    batch: Dict[str, torch.Tensor],
    loss_config: Dict,
    global_step: int = 0,
) -> Dict[str, torch.Tensor]:
    obs_hist = batch["obs_hist"]
    act_hist = batch["act_hist"]
    future_actions = batch["future_actions"]
    h_emb = policy.encode_history(obs_hist, act_hist)
    beta_r = float(loss_config.get("beta_R", 0.01))
    beta_z = linear_warmup(
        global_step,
        float(loss_config.get("beta_z_start", 0.0)),
        float(loss_config.get("beta_z_end", 0.01)),
        int(loss_config.get("beta_z_warmup_steps", 10000)),
    )
    tube_std = 0.0
    if bool(loss_config.get("tube_training", False)):
        tube_std = linear_warmup(
            global_step,
            float(loss_config.get("tube_noise_std_start", 0.0)),
            float(loss_config.get("tube_noise_std_end", 0.02)),
            int(loss_config.get("tube_noise_warmup_steps", 5000)),
        )
    free_nats = float(loss_config.get("free_nats", 0.0))

    if policy.latent_type == "categorical":
        p_logits = policy.latent.prior_logits(h_emb)
        q_logits = policy.latent.posterior_logits(h_emb, future_actions)
        q_probs = torch.softmax(q_logits, dim=-1)
        per_z_nll = []
        per_z_kl = []
        per_z_mse = []
        for z_id in range(policy.latent.num_categories):
            ids = torch.full((future_actions.shape[0],), z_id, device=future_actions.device, dtype=torch.long)
            z_emb = policy.latent.embed_ids(ids)
            out = bridge_path_losses_conditioned(policy, h_emb, act_hist, future_actions, z_emb, tube_noise_std=tube_std)
            per_z_nll.append(out["nll"])
            per_z_kl.append(out["path_kl"])
            per_z_mse.append(out["mse"])
        nll_z = torch.stack(per_z_nll, dim=-1)
        path_kl_z = torch.stack(per_z_kl, dim=-1)
        mse_z = torch.stack(per_z_mse, dim=-1)
        nll = (q_probs * nll_z).sum(dim=-1).mean()
        path_kl = (q_probs * path_kl_z).sum(dim=-1).mean()
        action_mse = (q_probs * mse_z).sum(dim=-1).mean()
        raw_latent_kl = categorical_kl(q_logits, p_logits)
        latent_kl_loss = raw_latent_kl.clamp_min(free_nats).mean() if free_nats > 0 else raw_latent_kl.mean()
        prior_entropy = categorical_entropy(p_logits).mean()
        posterior_entropy = categorical_entropy(q_logits).mean()
        aux = future_actions.new_zeros(())
        beta_aux = float(loss_config.get("beta_aux_mode_ce", 0.0))
        if beta_aux > 0 and "mode_label" in batch:
            aux = F.cross_entropy(q_logits, batch["mode_label"].long())
        loss = nll + beta_r * path_kl + beta_z * latent_kl_loss + beta_aux * aux
        return {
            "loss": loss,
            "nll": nll,
            "path_kl": path_kl,
            "action_mse": action_mse,
            "latent_kl": raw_latent_kl.mean(),
            "latent_kl_loss": latent_kl_loss,
            "prior_entropy": prior_entropy,
            "posterior_entropy": posterior_entropy,
            "beta_z": future_actions.new_tensor(beta_z),
            "tube_noise_std": future_actions.new_tensor(tube_std),
            "aux_mode_ce": aux,
        }

    if policy.latent_type == "continuous":
        mu_p, logvar_p = policy.latent.prior_params(h_emb)
        mu_q, logvar_q = policy.latent.posterior_params(h_emb, future_actions)
        raw_latent_kl = gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
        latent_kl_loss = raw_latent_kl.clamp_min(free_nats).mean() if free_nats > 0 else raw_latent_kl.mean()
        n_samples = int(loss_config.get("num_z_samples_train", 1))
        nll_total = future_actions.new_zeros(())
        path_kl_total = future_actions.new_zeros(())
        mse_total = future_actions.new_zeros(())
        for _ in range(max(1, n_samples)):
            z = policy.latent.reparameterize(mu_q, logvar_q)
            z_emb = policy.latent.embed(z)
            out = bridge_path_losses_conditioned(policy, h_emb, act_hist, future_actions, z_emb, tube_noise_std=tube_std)
            nll_total = nll_total + out["nll"].mean()
            path_kl_total = path_kl_total + out["path_kl"].mean()
            mse_total = mse_total + out["mse"].mean()
        denom = float(max(1, n_samples))
        nll = nll_total / denom
        path_kl = path_kl_total / denom
        action_mse = mse_total / denom
        loss = nll + beta_r * path_kl + beta_z * latent_kl_loss
        posterior_entropy = 0.5 * (1.0 + torch.log(torch.tensor(2.0 * torch.pi, device=future_actions.device)) + logvar_q).sum(dim=-1).mean()
        prior_entropy = 0.5 * (1.0 + torch.log(torch.tensor(2.0 * torch.pi, device=future_actions.device)) + logvar_p).sum(dim=-1).mean()
        return {
            "loss": loss,
            "nll": nll,
            "path_kl": path_kl,
            "action_mse": action_mse,
            "latent_kl": raw_latent_kl.mean(),
            "latent_kl_loss": latent_kl_loss,
            "prior_entropy": prior_entropy,
            "posterior_entropy": posterior_entropy,
            "beta_z": future_actions.new_tensor(beta_z),
            "tube_noise_std": future_actions.new_tensor(tube_std),
        }

    out = bridge_path_losses_conditioned(policy, h_emb, act_hist, future_actions, None, tube_noise_std=tube_std)
    nll = out["nll"].mean()
    path_kl = out["path_kl"].mean()
    action_mse = out["mse"].mean()
    loss = nll + beta_r * path_kl
    return {
        "loss": loss,
        "nll": nll,
        "path_kl": path_kl,
        "action_mse": action_mse,
        "latent_kl": future_actions.new_zeros(()),
        "latent_kl_loss": future_actions.new_zeros(()),
        "prior_entropy": future_actions.new_zeros(()),
        "posterior_entropy": future_actions.new_zeros(()),
        "beta_z": future_actions.new_tensor(beta_z),
        "tube_noise_std": future_actions.new_tensor(tube_std),
    }


def direct_bc_loss(model: DirectChunkBCPolicy, batch: Dict[str, torch.Tensor], loss_config: Dict) -> Dict[str, torch.Tensor]:
    pred = model(batch["obs_hist"], batch["act_hist"])
    mse = F.mse_loss(pred, batch["future_actions"])
    smooth = smoothness_losses(pred, batch["act_hist"])
    loss = mse
    loss = loss + float(loss_config.get("lambda_acc", 0.0)) * smooth["acceleration_energy"]
    loss = loss + float(loss_config.get("lambda_jerk", 0.0)) * smooth["jerk_energy"]
    return {
        "loss": loss,
        "action_mse": mse,
        "nll": mse,
        "path_kl": pred.new_zeros(()),
        "latent_kl": pred.new_zeros(()),
        "acceleration_energy": smooth["acceleration_energy"],
        "jerk_energy": smooth["jerk_energy"],
    }


def autoregressive_bc_loss(model: AutoregressiveBCPolicy, batch: Dict[str, torch.Tensor], loss_config: Dict) -> Dict[str, torch.Tensor]:
    obs_hist = batch["obs_hist"]
    act_hist = batch["act_hist"]
    future = batch["future_actions"]
    h = model.encode_history(obs_hist, act_hist)
    preds = []
    for k in range(future.shape[1]):
        a_prevprev, a_prev = teacher_forced_prev_actions(act_hist, future, k)
        preds.append(model.predict_step(a_prev, a_prevprev, h, k))
    pred = torch.stack(preds, dim=1)
    mse = F.mse_loss(pred, future)
    smooth = smoothness_losses(pred, act_hist)
    loss = mse
    loss = loss + float(loss_config.get("lambda_acc", 0.0)) * smooth["acceleration_energy"]
    loss = loss + float(loss_config.get("lambda_jerk", 0.0)) * smooth["jerk_energy"]
    return {
        "loss": loss,
        "action_mse": mse,
        "nll": mse,
        "path_kl": pred.new_zeros(()),
        "latent_kl": pred.new_zeros(()),
        "acceleration_energy": smooth["acceleration_energy"],
        "jerk_energy": smooth["jerk_energy"],
    }


def model_loss(model, batch: Dict[str, torch.Tensor], loss_config: Dict, global_step: int = 0) -> Dict[str, torch.Tensor]:
    if isinstance(model, ActionBridgePolicy):
        return bridge_loss(model, batch, loss_config, global_step=global_step)
    if isinstance(model, DirectChunkBCPolicy):
        return direct_bc_loss(model, batch, loss_config)
    if isinstance(model, AutoregressiveBCPolicy):
        return autoregressive_bc_loss(model, batch, loss_config)
    raise TypeError(f"Unsupported model type {type(model)!r}.")

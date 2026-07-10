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


def scheduled_loss_weight(loss_config: Dict, key: str, global_step: int) -> float:
    target = float(loss_config.get(key, 0.0))
    if target <= 0.0:
        return 0.0
    warmup = int(loss_config.get(f"{key}_warmup_steps", 0))
    if warmup <= 0:
        return target
    return linear_warmup(global_step, 0.0, target, warmup)


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


def bridge_unrolled_mse_conditioned(
    policy: ActionBridgePolicy,
    h_emb: torch.Tensor,
    act_hist: torch.Tensor,
    future_actions: torch.Tensor,
    z_emb: Optional[torch.Tensor],
) -> torch.Tensor:
    a_prevprev = act_hist[:, -2]
    a_prev = act_hist[:, -1]
    mse = torch.zeros(future_actions.shape[0], device=future_actions.device, dtype=future_actions.dtype)
    for k in range(future_actions.shape[1]):
        action, _, _, _ = policy.step(a_prev, a_prevprev, h_emb, k, z_emb, deterministic=True)
        mse = mse + (future_actions[:, k] - action).pow(2).sum(dim=-1)
        a_prevprev, a_prev = a_prev, action
    return mse / max(1, future_actions.shape[1])


def contact_path_losses_conditioned(
    policy: ActionBridgePolicy,
    h_emb: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    z_emb: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    future_actions = batch["future_actions"]
    adapter = policy.coordinate_adapter
    ref = policy.reference_process
    q_seq = adapter.build_q_sequence(batch)
    p_seq = adapter.build_p_sequence(q_seq, batch)
    loss_p = torch.zeros(future_actions.shape[0], device=future_actions.device, dtype=future_actions.dtype)
    loss_q = torch.zeros_like(loss_p)
    path_kl = torch.zeros_like(loss_p)
    mse = torch.zeros_like(loss_p)
    ref_reg_terms = []
    gamma_values = []
    k_values = []
    m_values = []
    for k in range(future_actions.shape[1]):
        q = q_seq[:, k]
        p = p_seq[:, k]
        q_pred, p_pred, u, aux = policy.contact_step(q, p, h_emb, k, z_emb, deterministic=True)
        sigma = ref.sigma_like(q)
        loss_p = loss_p + 0.5 * ((p_seq[:, k + 1] - p_pred) / sigma).pow(2).sum(dim=-1)
        loss_q = loss_q + 0.5 * ((q_seq[:, k + 1] - q_pred) / sigma).pow(2).sum(dim=-1)
        if ref.control_is_whitened:
            step_kl = 0.5 * ref.dt * u.pow(2).sum(dim=-1)
        else:
            step_kl = 0.5 * ref.dt * (u / sigma).pow(2).sum(dim=-1)
        path_kl = path_kl + step_kl
        raw_pred = adapter.decode_step(q, q_pred)
        mse = mse + (future_actions[:, k] - raw_pred).pow(2).sum(dim=-1)

        gamma = aux.get("gamma")
        if gamma is not None:
            gamma_values.append(gamma.detach())
            ref_reg_terms.append(gamma.pow(2).mean())
        k_diag = aux.get("k_diag")
        if k_diag is not None:
            k_values.append(k_diag.detach())
            ref_reg_terms.append(k_diag.pow(2).mean())
        m = aux.get("m")
        if m is not None:
            m_values.append(m)

    ref_reg = torch.stack(ref_reg_terms).mean() if ref_reg_terms else future_actions.new_zeros(())
    if len(m_values) > 1:
        m_seq = torch.stack(m_values, dim=1)
        m_smooth = (m_seq[:, 1:] - m_seq[:, :-1]).pow(2).mean()
    else:
        m_smooth = future_actions.new_zeros(())
    if gamma_values:
        gamma_cat = torch.cat([g.reshape(-1) for g in gamma_values])
        gamma_mean = gamma_cat.mean()
        gamma_min = gamma_cat.min()
        gamma_max = gamma_cat.max()
    else:
        gamma_mean = gamma_min = gamma_max = future_actions.new_zeros(())
    if k_values:
        k_cat = torch.cat([v.reshape(-1) for v in k_values])
        k_diag_mean = k_cat.mean()
        k_diag_max = k_cat.max()
    else:
        k_diag_mean = k_diag_max = future_actions.new_zeros(())

    velocity_energy = (q_seq[:, 1:] - q_seq[:, :-1]).pow(2).sum(dim=-1).mean()
    if q_seq.shape[1] >= 3:
        q_accel = q_seq[:, 2:] - 2.0 * q_seq[:, 1:-1] + q_seq[:, :-2]
        q_accel_energy = q_accel.pow(2).sum(dim=-1).mean()
    else:
        q_accel_energy = future_actions.new_zeros(())
    data_loss = loss_p + ref.lambda_q * loss_q
    return {
        "nll": data_loss,
        "loss_p": loss_p,
        "loss_q": loss_q,
        "path_kl": path_kl,
        "mse": mse / max(1, future_actions.shape[1]),
        "ref_reg": ref_reg,
        "m_smooth": m_smooth,
        "gamma_mean": gamma_mean,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "k_diag_mean": k_diag_mean,
        "k_diag_max": k_diag_max,
        "velocity_energy": velocity_energy,
        "q_acceleration_energy": q_accel_energy,
    }


def contact_unrolled_mse_conditioned(
    policy: ActionBridgePolicy,
    h_emb: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    z_emb: Optional[torch.Tensor],
) -> torch.Tensor:
    future_actions = batch["future_actions"]
    adapter = policy.coordinate_adapter
    q, p = adapter.init_qp_from_history(batch)
    mse = torch.zeros(future_actions.shape[0], device=future_actions.device, dtype=future_actions.dtype)
    for k in range(future_actions.shape[1]):
        q_next, p_next, _, _ = policy.contact_step(q, p, h_emb, k, z_emb, deterministic=True)
        raw_pred = adapter.decode_step(q, q_next)
        mse = mse + (future_actions[:, k] - raw_pred).pow(2).sum(dim=-1)
        q, p = q_next, p_next
    return mse / max(1, future_actions.shape[1])


def _contact_metric_template(policy: ActionBridgePolicy, value: torch.Tensor, metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    ref = policy.reference_process
    ref_reg_loss = ref.lambda_ref_reg * metrics["ref_reg"]
    m_smooth_loss = ref.lambda_m_smooth * metrics["m_smooth"]
    zero = value.new_zeros(())
    return {
        "nll": value,
        "path_kl": metrics["path_kl"],
        "action_mse": metrics["action_mse"],
        "unroll_mse": metrics.get("unroll_mse", zero),
        "lambda_unroll": metrics.get("lambda_unroll", zero),
        "loss_p": metrics["loss_p"],
        "loss_q": metrics["loss_q"],
        "control_energy": metrics["path_kl"],
        "reference_reg": ref_reg_loss,
        "m_smooth_loss": m_smooth_loss,
        "gamma_mean": metrics["gamma_mean"],
        "gamma_min": metrics["gamma_min"],
        "gamma_max": metrics["gamma_max"],
        "k_diag_mean": metrics["k_diag_mean"],
        "k_diag_max": metrics["k_diag_max"],
        "velocity_energy": metrics["velocity_energy"],
        "q_acceleration_energy": metrics["q_acceleration_energy"],
    }


def contact_bridge_loss(
    policy: ActionBridgePolicy,
    batch: Dict[str, torch.Tensor],
    loss_config: Dict,
    global_step: int = 0,
) -> Dict[str, torch.Tensor]:
    obs_hist = batch["obs_hist"]
    act_hist = batch["act_hist"]
    future_actions = batch["future_actions"]
    h_emb = policy.encode_history(obs_hist, act_hist)
    ref = policy.reference_process
    beta_z = linear_warmup(
        global_step,
        float(loss_config.get("beta_z_start", 0.0)),
        float(loss_config.get("beta_z_end", 0.01)),
        int(loss_config.get("beta_z_warmup_steps", 10000)),
    )
    lambda_unroll = scheduled_loss_weight(loss_config, "lambda_unroll", global_step)
    free_nats = float(loss_config.get("free_nats", 0.0))

    def aggregate(out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "nll": out["nll"].mean(),
            "path_kl": out["path_kl"].mean(),
            "action_mse": out["mse"].mean(),
            "loss_p": out["loss_p"].mean(),
            "loss_q": out["loss_q"].mean(),
            "ref_reg": out["ref_reg"],
            "m_smooth": out["m_smooth"],
            "gamma_mean": out["gamma_mean"],
            "gamma_min": out["gamma_min"],
            "gamma_max": out["gamma_max"],
            "k_diag_mean": out["k_diag_mean"],
            "k_diag_max": out["k_diag_max"],
            "velocity_energy": out["velocity_energy"],
            "q_acceleration_energy": out["q_acceleration_energy"],
        }

    if policy.latent_type == "categorical":
        p_logits = policy.latent.prior_logits(h_emb)
        q_logits = policy.latent.posterior_logits(h_emb, future_actions)
        q_probs = torch.softmax(q_logits, dim=-1)
        per_z = []
        per_z_unroll = []
        for z_id in range(policy.latent.num_categories):
            ids = torch.full((future_actions.shape[0],), z_id, device=future_actions.device, dtype=torch.long)
            z_emb = policy.latent.embed_ids(ids)
            per_z.append(contact_path_losses_conditioned(policy, h_emb, batch, z_emb))
            if lambda_unroll > 0.0:
                per_z_unroll.append(contact_unrolled_mse_conditioned(policy, h_emb, batch, z_emb))
        nll_z = torch.stack([item["nll"] for item in per_z], dim=-1)
        path_kl_z = torch.stack([item["path_kl"] for item in per_z], dim=-1)
        mse_z = torch.stack([item["mse"] for item in per_z], dim=-1)
        loss_p_z = torch.stack([item["loss_p"] for item in per_z], dim=-1)
        loss_q_z = torch.stack([item["loss_q"] for item in per_z], dim=-1)
        if per_z_unroll:
            unroll_z = torch.stack(per_z_unroll, dim=-1)
            unroll_mse = (q_probs * unroll_z).sum(dim=-1).mean()
        else:
            unroll_mse = future_actions.new_zeros(())
        metrics = {
            "nll": (q_probs * nll_z).sum(dim=-1).mean(),
            "path_kl": (q_probs * path_kl_z).sum(dim=-1).mean(),
            "action_mse": (q_probs * mse_z).sum(dim=-1).mean(),
            "loss_p": (q_probs * loss_p_z).sum(dim=-1).mean(),
            "loss_q": (q_probs * loss_q_z).sum(dim=-1).mean(),
            "unroll_mse": unroll_mse,
            "lambda_unroll": future_actions.new_tensor(lambda_unroll),
            **{key: per_z[0][key] for key in ["ref_reg", "m_smooth", "gamma_mean", "gamma_min", "gamma_max", "k_diag_mean", "k_diag_max", "velocity_energy", "q_acceleration_energy"]},
        }
        raw_latent_kl = categorical_kl(q_logits, p_logits)
        latent_kl_loss = raw_latent_kl.clamp_min(free_nats).mean() if free_nats > 0 else raw_latent_kl.mean()
        prior_entropy = categorical_entropy(p_logits).mean()
        posterior_entropy = categorical_entropy(q_logits).mean()
        aux = future_actions.new_zeros(())
        beta_aux = float(loss_config.get("beta_aux_mode_ce", 0.0))
        if beta_aux > 0 and "mode_label" in batch:
            aux = F.cross_entropy(q_logits, batch["mode_label"].long())
        base_loss = metrics["nll"] + ref.beta_kl * metrics["path_kl"]
        ref_reg_loss = ref.lambda_ref_reg * metrics["ref_reg"]
        m_smooth_loss = ref.lambda_m_smooth * metrics["m_smooth"]
        loss = base_loss + lambda_unroll * unroll_mse + beta_z * latent_kl_loss + beta_aux * aux + ref_reg_loss + m_smooth_loss
        return {
            "loss": loss,
            **_contact_metric_template(policy, metrics["nll"], metrics),
            "latent_kl": raw_latent_kl.mean(),
            "latent_kl_loss": latent_kl_loss,
            "prior_entropy": prior_entropy,
            "posterior_entropy": posterior_entropy,
            "beta_z": future_actions.new_tensor(beta_z),
            "beta_kl": future_actions.new_tensor(ref.beta_kl),
            "aux_mode_ce": aux,
        }

    if policy.latent_type == "continuous":
        mu_p, logvar_p = policy.latent.prior_params(h_emb)
        mu_q, logvar_q = policy.latent.posterior_params(h_emb, future_actions)
        raw_latent_kl = gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
        latent_kl_loss = raw_latent_kl.clamp_min(free_nats).mean() if free_nats > 0 else raw_latent_kl.mean()
        n_samples = int(loss_config.get("num_z_samples_train", 1))
        totals = []
        unroll_total = future_actions.new_zeros(())
        for _ in range(max(1, n_samples)):
            z = policy.latent.reparameterize(mu_q, logvar_q)
            z_emb = policy.latent.embed(z)
            totals.append(aggregate(contact_path_losses_conditioned(policy, h_emb, batch, z_emb)))
            if lambda_unroll > 0.0:
                unroll_total = unroll_total + contact_unrolled_mse_conditioned(policy, h_emb, batch, z_emb).mean()
        denom = float(max(1, n_samples))
        metrics = {}
        for key in totals[0]:
            metrics[key] = sum(item[key] for item in totals) / denom
        metrics["unroll_mse"] = unroll_total / denom if lambda_unroll > 0.0 else future_actions.new_zeros(())
        metrics["lambda_unroll"] = future_actions.new_tensor(lambda_unroll)
        base_loss = metrics["nll"] + ref.beta_kl * metrics["path_kl"]
        ref_reg_loss = ref.lambda_ref_reg * metrics["ref_reg"]
        m_smooth_loss = ref.lambda_m_smooth * metrics["m_smooth"]
        loss = base_loss + lambda_unroll * metrics["unroll_mse"] + beta_z * latent_kl_loss + ref_reg_loss + m_smooth_loss
        posterior_entropy = 0.5 * (1.0 + torch.log(torch.tensor(2.0 * torch.pi, device=future_actions.device)) + logvar_q).sum(dim=-1).mean()
        prior_entropy = 0.5 * (1.0 + torch.log(torch.tensor(2.0 * torch.pi, device=future_actions.device)) + logvar_p).sum(dim=-1).mean()
        return {
            "loss": loss,
            **_contact_metric_template(policy, metrics["nll"], metrics),
            "latent_kl": raw_latent_kl.mean(),
            "latent_kl_loss": latent_kl_loss,
            "prior_entropy": prior_entropy,
            "posterior_entropy": posterior_entropy,
            "beta_z": future_actions.new_tensor(beta_z),
            "beta_kl": future_actions.new_tensor(ref.beta_kl),
        }

    metrics = aggregate(contact_path_losses_conditioned(policy, h_emb, batch, None))
    metrics["unroll_mse"] = (
        contact_unrolled_mse_conditioned(policy, h_emb, batch, None).mean()
        if lambda_unroll > 0.0
        else future_actions.new_zeros(())
    )
    metrics["lambda_unroll"] = future_actions.new_tensor(lambda_unroll)
    base_loss = metrics["nll"] + ref.beta_kl * metrics["path_kl"]
    ref_reg_loss = ref.lambda_ref_reg * metrics["ref_reg"]
    m_smooth_loss = ref.lambda_m_smooth * metrics["m_smooth"]
    loss = base_loss + lambda_unroll * metrics["unroll_mse"] + ref_reg_loss + m_smooth_loss
    return {
        "loss": loss,
        **_contact_metric_template(policy, metrics["nll"], metrics),
        "latent_kl": future_actions.new_zeros(()),
        "latent_kl_loss": future_actions.new_zeros(()),
        "prior_entropy": future_actions.new_zeros(()),
        "posterior_entropy": future_actions.new_zeros(()),
        "beta_z": future_actions.new_tensor(beta_z),
        "beta_kl": future_actions.new_tensor(ref.beta_kl),
    }


def bridge_loss(
    policy: ActionBridgePolicy,
    batch: Dict[str, torch.Tensor],
    loss_config: Dict,
    global_step: int = 0,
) -> Dict[str, torch.Tensor]:
    if bool(getattr(policy, "uses_contact_langevin", False)):
        return contact_bridge_loss(policy, batch, loss_config, global_step=global_step)

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
    lambda_unroll = scheduled_loss_weight(loss_config, "lambda_unroll", global_step)
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
        per_z_unroll = []
        for z_id in range(policy.latent.num_categories):
            ids = torch.full((future_actions.shape[0],), z_id, device=future_actions.device, dtype=torch.long)
            z_emb = policy.latent.embed_ids(ids)
            out = bridge_path_losses_conditioned(policy, h_emb, act_hist, future_actions, z_emb, tube_noise_std=tube_std)
            per_z_nll.append(out["nll"])
            per_z_kl.append(out["path_kl"])
            per_z_mse.append(out["mse"])
            if lambda_unroll > 0.0:
                per_z_unroll.append(bridge_unrolled_mse_conditioned(policy, h_emb, act_hist, future_actions, z_emb))
        nll_z = torch.stack(per_z_nll, dim=-1)
        path_kl_z = torch.stack(per_z_kl, dim=-1)
        mse_z = torch.stack(per_z_mse, dim=-1)
        if per_z_unroll:
            unroll_z = torch.stack(per_z_unroll, dim=-1)
            unroll_mse = (q_probs * unroll_z).sum(dim=-1).mean()
        else:
            unroll_mse = future_actions.new_zeros(())
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
        loss = nll + beta_r * path_kl + lambda_unroll * unroll_mse + beta_z * latent_kl_loss + beta_aux * aux
        return {
            "loss": loss,
            "nll": nll,
            "path_kl": path_kl,
            "action_mse": action_mse,
            "unroll_mse": unroll_mse,
            "lambda_unroll": future_actions.new_tensor(lambda_unroll),
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
        unroll_total = future_actions.new_zeros(())
        for _ in range(max(1, n_samples)):
            z = policy.latent.reparameterize(mu_q, logvar_q)
            z_emb = policy.latent.embed(z)
            out = bridge_path_losses_conditioned(policy, h_emb, act_hist, future_actions, z_emb, tube_noise_std=tube_std)
            nll_total = nll_total + out["nll"].mean()
            path_kl_total = path_kl_total + out["path_kl"].mean()
            mse_total = mse_total + out["mse"].mean()
            if lambda_unroll > 0.0:
                unroll_total = unroll_total + bridge_unrolled_mse_conditioned(policy, h_emb, act_hist, future_actions, z_emb).mean()
        denom = float(max(1, n_samples))
        nll = nll_total / denom
        path_kl = path_kl_total / denom
        action_mse = mse_total / denom
        unroll_mse = unroll_total / denom if lambda_unroll > 0.0 else future_actions.new_zeros(())
        loss = nll + beta_r * path_kl + lambda_unroll * unroll_mse + beta_z * latent_kl_loss
        posterior_entropy = 0.5 * (1.0 + torch.log(torch.tensor(2.0 * torch.pi, device=future_actions.device)) + logvar_q).sum(dim=-1).mean()
        prior_entropy = 0.5 * (1.0 + torch.log(torch.tensor(2.0 * torch.pi, device=future_actions.device)) + logvar_p).sum(dim=-1).mean()
        return {
            "loss": loss,
            "nll": nll,
            "path_kl": path_kl,
            "action_mse": action_mse,
            "unroll_mse": unroll_mse,
            "lambda_unroll": future_actions.new_tensor(lambda_unroll),
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
    unroll_mse = (
        bridge_unrolled_mse_conditioned(policy, h_emb, act_hist, future_actions, None).mean()
        if lambda_unroll > 0.0
        else future_actions.new_zeros(())
    )
    loss = nll + beta_r * path_kl + lambda_unroll * unroll_mse
    return {
        "loss": loss,
        "nll": nll,
        "path_kl": path_kl,
        "action_mse": action_mse,
        "unroll_mse": unroll_mse,
        "lambda_unroll": future_actions.new_tensor(lambda_unroll),
        "latent_kl": future_actions.new_zeros(()),
        "latent_kl_loss": future_actions.new_zeros(()),
        "prior_entropy": future_actions.new_zeros(()),
        "posterior_entropy": future_actions.new_zeros(()),
        "beta_z": future_actions.new_tensor(beta_z),
        "tube_noise_std": future_actions.new_tensor(tube_std),
    }


def direct_bc_loss(
    model: DirectChunkBCPolicy,
    batch: Dict[str, torch.Tensor],
    loss_config: Dict,
    global_step: int = 0,
) -> Dict[str, torch.Tensor]:
    del global_step
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
        "unroll_mse": mse,
        "lambda_unroll": pred.new_zeros(()),
        "acceleration_energy": smooth["acceleration_energy"],
        "jerk_energy": smooth["jerk_energy"],
    }


def autoregressive_bc_loss(
    model: AutoregressiveBCPolicy,
    batch: Dict[str, torch.Tensor],
    loss_config: Dict,
    global_step: int = 0,
) -> Dict[str, torch.Tensor]:
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
    free_pred = model.generate(obs_hist, act_hist, deterministic=True)
    unroll_mse = F.mse_loss(free_pred, future)
    lambda_unroll = scheduled_loss_weight(loss_config, "lambda_unroll", global_step)
    smooth = smoothness_losses(pred, act_hist)
    loss = mse
    loss = loss + lambda_unroll * unroll_mse
    loss = loss + float(loss_config.get("lambda_acc", 0.0)) * smooth["acceleration_energy"]
    loss = loss + float(loss_config.get("lambda_jerk", 0.0)) * smooth["jerk_energy"]
    return {
        "loss": loss,
        "action_mse": mse,
        "nll": mse,
        "path_kl": pred.new_zeros(()),
        "latent_kl": pred.new_zeros(()),
        "unroll_mse": unroll_mse,
        "lambda_unroll": pred.new_tensor(lambda_unroll),
        "acceleration_energy": smooth["acceleration_energy"],
        "jerk_energy": smooth["jerk_energy"],
    }


def model_loss(model, batch: Dict[str, torch.Tensor], loss_config: Dict, global_step: int = 0) -> Dict[str, torch.Tensor]:
    if isinstance(model, ActionBridgePolicy):
        return bridge_loss(model, batch, loss_config, global_step=global_step)
    if isinstance(model, DirectChunkBCPolicy):
        return direct_bc_loss(model, batch, loss_config, global_step=global_step)
    if isinstance(model, AutoregressiveBCPolicy):
        return autoregressive_bc_loss(model, batch, loss_config, global_step=global_step)
    raise TypeError(f"Unsupported model type {type(model)!r}.")

"""Open-loop and simple receding-horizon generation."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from action_bridge.models.action_bridge_policy import ActionBridgePolicy
from action_bridge.models.baselines import AutoregressiveBCPolicy, DirectChunkBCPolicy


@torch.no_grad()
def generate_chunk(
    policy: ActionBridgePolicy,
    obs_hist: torch.Tensor,
    act_hist: torch.Tensor,
    mode: str = "sample",
    deterministic: bool = True,
    z: Optional[torch.Tensor] = None,
    z_emb: Optional[torch.Tensor] = None,
    sticky: bool = False,
    kappa: float = 2.0,
    rho_z: float = 1.0,
) -> Dict[str, torch.Tensor]:
    h_emb = policy.encode_history(obs_hist, act_hist)
    if z_emb is None:
        z, z_emb = policy.sample_prior_z(
            h_emb,
            mode=mode,
            deterministic_continuous=False,
            z_prev=z,
            sticky=sticky,
            kappa=kappa,
            rho_z=rho_z,
        )
    if bool(getattr(policy, "uses_contact_langevin", False)):
        adapter = policy.coordinate_adapter
        q, p = adapter.init_qp_from_history({"obs_hist": obs_hist, "act_hist": act_hist})
        obs_state = obs_hist[:, -1]
        q_list = [q]
        p_list = [p]
        controls = []
        path_kl_steps = []
        path_kl = torch.zeros(obs_hist.shape[0], device=obs_hist.device, dtype=obs_hist.dtype)
        for k in range(policy.chunk_horizon):
            q, p, u, _ = policy.contact_step(q, p, h_emb, k, z_emb, obs_state=obs_state, deterministic=deterministic)
            sigma = policy.reference_process.sigma_like(q)
            if policy.reference_process.control_is_whitened:
                step_path_kl = 0.5 * policy.reference_process.dt * u.pow(2).sum(dim=-1)
            else:
                step_path_kl = 0.5 * policy.reference_process.dt * (u / sigma).pow(2).sum(dim=-1)
            path_kl = path_kl + step_path_kl
            path_kl_steps.append(step_path_kl)
            controls.append(u)
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
        }

    a_prevprev = act_hist[:, -2]
    a_prev = act_hist[:, -1]
    actions = []
    means = []
    controls = []
    path_kl_steps = []
    path_kl = torch.zeros(obs_hist.shape[0], device=obs_hist.device, dtype=obs_hist.dtype)
    for k in range(policy.chunk_horizon):
        mu_r, log_sigma = policy.reference_process(a_prev, a_prevprev, h_emb, k)
        u = policy.control(a_prev, a_prevprev, h_emb, k, z_emb)
        mu = mu_r + u
        if deterministic:
            action = mu
        else:
            action = mu + log_sigma.exp() * torch.randn_like(mu)
        step_path_kl = 0.5 * (u / log_sigma.exp().clamp_min(1e-6)).pow(2).sum(dim=-1)
        path_kl = path_kl + step_path_kl
        path_kl_steps.append(step_path_kl)
        actions.append(action)
        means.append(mu)
        controls.append(u)
        a_prevprev, a_prev = a_prev, action
    return {
        "actions": torch.stack(actions, dim=1),
        "means": torch.stack(means, dim=1),
        "controls": torch.stack(controls, dim=1),
        "z": z,
        "z_emb": z_emb,
        "path_kl_energy": path_kl,
        "path_kl_steps": torch.stack(path_kl_steps, dim=1),
    }


@torch.no_grad()
def predict_actions(model, batch: Dict[str, torch.Tensor], deterministic: bool = True, mode: str = "sample") -> Dict[str, torch.Tensor]:
    if isinstance(model, ActionBridgePolicy):
        return generate_chunk(model, batch["obs_hist"], batch["act_hist"], mode=mode, deterministic=deterministic)
    if isinstance(model, DirectChunkBCPolicy):
        actions = model(batch["obs_hist"], batch["act_hist"])
    elif isinstance(model, AutoregressiveBCPolicy):
        actions = model.generate(batch["obs_hist"], batch["act_hist"], deterministic=deterministic)
    else:
        actions = model(batch["obs_hist"], batch["act_hist"])
    return {
        "actions": actions,
        "path_kl_energy": torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype),
        "path_kl_steps": torch.zeros(actions.shape[0], actions.shape[1], device=actions.device, dtype=actions.dtype),
        "z": None,
    }


def actions_to_positions(initial_pos: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    increments = torch.cumsum(actions, dim=1)
    return torch.cat([initial_pos[:, None], initial_pos[:, None] + increments], dim=1)

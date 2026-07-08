"""Latent controlled action bridge policy."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from action_bridge.data.action_coordinates import ActionCoordinateAdapter
from action_bridge.models.encoders import HistoryEncoder, SinusoidalTimeEmbedding, make_mlp, timestep_tensor
from action_bridge.models.latents import CategoricalLatent, ContinuousLatent
from action_bridge.models.references import ReferenceProcess, build_reference


class ControlNet(nn.Module):
    def __init__(
        self,
        action_dim: int,
        h_emb_dim: int,
        time_emb_dim: int,
        z_embed_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        control_scale: float = 0.05,
    ):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_emb_dim)
        self.control_scale = float(control_scale)
        in_dim = 2 * action_dim + h_emb_dim + time_emb_dim + z_embed_dim
        self.net = make_mlp(in_dim, action_dim, hidden_dim, depth)

    def forward(
        self,
        a_prev: torch.Tensor,
        a_prevprev: torch.Tensor,
        h_emb: torch.Tensor,
        k: int,
        z_emb: torch.Tensor,
    ) -> torch.Tensor:
        tk = timestep_tensor(k, a_prev.shape[0], a_prev.device)
        t_emb = self.time(tk).to(dtype=a_prev.dtype)
        return self.control_scale * torch.tanh(self.net(torch.cat([a_prev, a_prevprev, h_emb, t_emb, z_emb], dim=-1)))


class ContactControlNet(nn.Module):
    def __init__(
        self,
        action_dim: int,
        h_emb_dim: int,
        time_emb_dim: int,
        z_embed_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        control_scale: float = 1.0,
    ):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_emb_dim)
        self.control_scale = float(control_scale)
        in_dim = 2 * action_dim + h_emb_dim + time_emb_dim + z_embed_dim
        self.net = make_mlp(in_dim, action_dim, hidden_dim, depth)

    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        h_emb: torch.Tensor,
        k: int,
        z_emb: torch.Tensor,
    ) -> torch.Tensor:
        tk = timestep_tensor(k, q.shape[0], q.device)
        t_emb = self.time(tk).to(dtype=q.dtype)
        return self.control_scale * self.net(torch.cat([q, p, h_emb, t_emb, z_emb], dim=-1))


class ActionBridgePolicy(nn.Module):
    """Reference process plus learned control residual.

    This is an amortized path-KL controlled process, not an exact
    Schrodinger Bridge solver.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_history: int,
        action_history: int,
        chunk_horizon: int,
        model_config: Dict,
        reference_config: Dict,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs_history = int(obs_history)
        self.action_history = int(action_history)
        self.chunk_horizon = int(chunk_horizon)
        self.latent_type = model_config.get("latent_type", "categorical")
        hidden_dim = int(model_config.get("hidden_dim", 256))
        h_emb_dim = int(model_config.get("h_emb_dim", hidden_dim))
        time_emb_dim = int(model_config.get("time_emb_dim", 32))
        self.z_embed_dim = int(model_config.get("z_embed_dim", 32))
        if self.latent_type in {"none", None}:
            self.z_embed_dim = 0

        self.history_encoder = HistoryEncoder(
            obs_history=obs_history,
            action_history=action_history,
            obs_dim=obs_dim,
            action_dim=action_dim,
            h_emb_dim=h_emb_dim,
            hidden_dim=hidden_dim,
            depth=int(model_config.get("encoder_depth", 3)),
            layer_norm=bool(model_config.get("layer_norm", False)),
        )
        self.reference_process: ReferenceProcess = build_reference(reference_config, action_dim=action_dim, h_emb_dim=h_emb_dim)
        self.uses_contact_langevin = bool(getattr(self.reference_process, "is_contact_langevin", False))
        self.coordinate_adapter = ActionCoordinateAdapter(
            coordinate_mode=str(reference_config.get("coordinate_mode", "raw_action")),
            dt=float(reference_config.get("dt", 1.0)),
            action_dim=action_dim,
        )

        if self.latent_type == "categorical":
            self.latent = CategoricalLatent(
                h_emb_dim=h_emb_dim,
                chunk_horizon=chunk_horizon,
                action_dim=action_dim,
                num_categories=int(model_config.get("num_categories", 2)),
                z_embed_dim=self.z_embed_dim,
                hidden_dim=hidden_dim,
            )
        elif self.latent_type == "continuous":
            self.latent = ContinuousLatent(
                h_emb_dim=h_emb_dim,
                chunk_horizon=chunk_horizon,
                action_dim=action_dim,
                z_dim=int(model_config.get("z_dim", 4)),
                z_embed_dim=self.z_embed_dim,
                hidden_dim=hidden_dim,
                prior_type=model_config.get("continuous_prior", "learned_conditional_gaussian"),
            )
        elif self.latent_type in {"none", None}:
            self.latent = None
        else:
            raise ValueError(f"Unknown latent_type {self.latent_type!r}.")

        if self.uses_contact_langevin:
            self.control_net = ContactControlNet(
                action_dim=action_dim,
                h_emb_dim=h_emb_dim,
                time_emb_dim=time_emb_dim,
                z_embed_dim=self.z_embed_dim,
                hidden_dim=hidden_dim,
                depth=int(model_config.get("control_depth", 4)),
                control_scale=float(model_config.get("control_scale", 1.0)),
            )
        else:
            self.control_net = ControlNet(
                action_dim=action_dim,
                h_emb_dim=h_emb_dim,
                time_emb_dim=time_emb_dim,
                z_embed_dim=self.z_embed_dim,
                hidden_dim=hidden_dim,
                depth=int(model_config.get("control_depth", 4)),
                control_scale=float(model_config.get("control_scale", 0.05)),
            )

    def encode_history(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(obs_hist, act_hist)

    def zero_z_embedding(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.z_embed_dim, device=device, dtype=dtype)

    def control(
        self,
        a_prev: torch.Tensor,
        a_prevprev: torch.Tensor,
        h_emb: torch.Tensor,
        k: int,
        z_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z_emb is None:
            z_emb = self.zero_z_embedding(a_prev.shape[0], a_prev.device, a_prev.dtype)
        return self.control_net(a_prev, a_prevprev, h_emb, k, z_emb)

    def contact_control(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        h_emb: torch.Tensor,
        k: int,
        z_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.uses_contact_langevin:
            raise RuntimeError("contact_control is only available for contact_langevin references.")
        if z_emb is None:
            z_emb = self.zero_z_embedding(q.shape[0], q.device, q.dtype)
        return self.control_net(q, p, h_emb, k, z_emb)

    def contact_step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        h_emb: torch.Tensor,
        k: int,
        z_emb: Optional[torch.Tensor],
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if not self.uses_contact_langevin:
            raise RuntimeError("contact_step is only available for contact_langevin references.")
        f_ref, aux = self.reference_process.force(q, p, h_emb, k)
        u = self.contact_control(q, p, h_emb, k, z_emb)
        sigma = self.reference_process.sigma_like(q)
        if self.reference_process.control_is_whitened:
            control_accel = sigma * u
        else:
            control_accel = u
        if deterministic or self.reference_process.deterministic_inference:
            noise = torch.zeros_like(q)
        else:
            noise = (self.reference_process.dt**0.5) * sigma * torch.randn_like(q)
        p_next = p + self.reference_process.dt * (f_ref + control_accel) + noise
        q_next = q + self.reference_process.dt * p_next
        return q_next, p_next, u, aux

    def prior_logits(self, h_emb: torch.Tensor) -> torch.Tensor:
        if self.latent_type != "categorical":
            raise RuntimeError("prior_logits is only available for categorical latents.")
        return self.latent.prior_logits(h_emb)

    def posterior_logits(self, h_emb: torch.Tensor, future_actions: torch.Tensor) -> torch.Tensor:
        if self.latent_type != "categorical":
            raise RuntimeError("posterior_logits is only available for categorical latents.")
        return self.latent.posterior_logits(h_emb, future_actions)

    def sample_prior_z(
        self,
        h_emb: torch.Tensor,
        mode: str = "sample",
        deterministic_continuous: bool = False,
        z_prev: Optional[torch.Tensor] = None,
        sticky: bool = False,
        kappa: float = 2.0,
        rho_z: float = 1.0,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if self.latent_type == "categorical":
            if sticky:
                return self.latent.sticky_sample(h_emb, z_prev=z_prev, kappa=kappa, mode=mode)
            return self.latent.sample_prior(h_emb, mode=mode)
        if self.latent_type == "continuous":
            if sticky:
                return self.latent.sticky_sample(h_emb, z_prev=z_prev, rho_z=rho_z, deterministic=deterministic_continuous)
            return self.latent.sample_prior(h_emb, deterministic=deterministic_continuous)
        return None, self.zero_z_embedding(h_emb.shape[0], h_emb.device, h_emb.dtype)

    def step(
        self,
        a_prev: torch.Tensor,
        a_prevprev: torch.Tensor,
        h_emb: torch.Tensor,
        k: int,
        z_emb: Optional[torch.Tensor],
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_r, log_sigma = self.reference_process(a_prev, a_prevprev, h_emb, k)
        u = self.control(a_prev, a_prevprev, h_emb, k, z_emb)
        mu = mu_r + u
        if deterministic:
            action = mu
        else:
            action = mu + log_sigma.exp() * torch.randn_like(mu)
        return action, mu, mu_r, u

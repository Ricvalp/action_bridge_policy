"""Categorical and continuous latent modules."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from action_bridge.models.encoders import FutureActionEncoder, make_mlp


def categorical_kl(q_logits: torch.Tensor, p_logits: torch.Tensor) -> torch.Tensor:
    q_log = F.log_softmax(q_logits, dim=-1)
    p_log = F.log_softmax(p_logits, dim=-1)
    q = q_log.exp()
    return (q * (q_log - p_log)).sum(dim=-1)


def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def gaussian_kl(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        logvar_p
        - logvar_q
        + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp().clamp_min(1e-8)
        - 1.0
    ).sum(dim=-1)


class CategoricalLatent(nn.Module):
    def __init__(
        self,
        h_emb_dim: int,
        chunk_horizon: int,
        action_dim: int,
        num_categories: int = 2,
        z_embed_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_categories = int(num_categories)
        self.z_embed_dim = int(z_embed_dim)
        self.future_encoder = FutureActionEncoder(chunk_horizon, action_dim, output_dim=h_emb_dim, hidden_dim=hidden_dim, depth=2)
        self.prior_net = make_mlp(h_emb_dim, self.num_categories, hidden_dim, depth=2)
        self.posterior_net = make_mlp(2 * h_emb_dim, self.num_categories, hidden_dim, depth=2)
        self.embedding = nn.Embedding(self.num_categories, self.z_embed_dim)

    def prior_logits(self, h_emb: torch.Tensor) -> torch.Tensor:
        return self.prior_net(h_emb)

    def posterior_logits(self, h_emb: torch.Tensor, future_actions: torch.Tensor) -> torch.Tensor:
        future = self.future_encoder(future_actions)
        return self.posterior_net(torch.cat([h_emb, future], dim=-1))

    def embed_ids(self, z_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(z_ids.long())

    def sample_prior(self, h_emb: torch.Tensor, mode: str = "sample") -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.prior_logits(h_emb)
        if mode == "argmax":
            z = logits.argmax(dim=-1)
        else:
            z = torch.distributions.Categorical(logits=logits).sample()
        return z, self.embed_ids(z)

    def sticky_sample(self, h_emb: torch.Tensor, z_prev: Optional[torch.Tensor], kappa: float = 2.0, mode: str = "sample") -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.prior_logits(h_emb)
        if z_prev is not None:
            penalty = torch.ones_like(logits) * (-float(kappa))
            penalty.scatter_(1, z_prev.long().view(-1, 1), 0.0)
            logits = logits + penalty
        if mode == "argmax":
            z = logits.argmax(dim=-1)
        else:
            z = torch.distributions.Categorical(logits=logits).sample()
        return z, self.embed_ids(z)


class ContinuousLatent(nn.Module):
    def __init__(
        self,
        h_emb_dim: int,
        chunk_horizon: int,
        action_dim: int,
        z_dim: int = 4,
        z_embed_dim: int = 32,
        hidden_dim: int = 256,
        prior_type: str = "learned_conditional_gaussian",
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.z_embed_dim = int(z_embed_dim)
        self.prior_type = prior_type
        self.future_encoder = FutureActionEncoder(chunk_horizon, action_dim, output_dim=h_emb_dim, hidden_dim=hidden_dim, depth=2)
        if prior_type == "learned_conditional_gaussian":
            self.prior_net = make_mlp(h_emb_dim, 2 * self.z_dim, hidden_dim, depth=2)
        elif prior_type == "standard_normal":
            self.prior_net = None
        else:
            raise ValueError(f"Unknown continuous prior type {prior_type!r}.")
        self.posterior_net = make_mlp(2 * h_emb_dim, 2 * self.z_dim, hidden_dim, depth=2)
        self.z_embed = make_mlp(self.z_dim, self.z_embed_dim, hidden_dim, depth=2)

    def prior_params(self, h_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prior_net is None:
            return torch.zeros(h_emb.shape[0], self.z_dim, device=h_emb.device, dtype=h_emb.dtype), torch.zeros(
                h_emb.shape[0], self.z_dim, device=h_emb.device, dtype=h_emb.dtype
            )
        mu, logvar = self.prior_net(h_emb).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 6.0)

    def posterior_params(self, h_emb: torch.Tensor, future_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        future = self.future_encoder(future_actions)
        mu, logvar = self.posterior_net(torch.cat([h_emb, future], dim=-1)).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 6.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def embed(self, z: torch.Tensor) -> torch.Tensor:
        return self.z_embed(z)

    def sample_prior(self, h_emb: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.prior_params(h_emb)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        return z, self.embed(z)

    def sticky_sample(
        self,
        h_emb: torch.Tensor,
        z_prev: Optional[torch.Tensor],
        rho_z: float = 1.0,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if z_prev is not None and rho_z >= 1.0:
            return z_prev, self.embed(z_prev)
        z_new, _ = self.sample_prior(h_emb, deterministic=deterministic)
        if z_prev is not None:
            rho = float(rho_z)
            z_new = rho * z_prev + max(0.0, 1.0 - rho * rho) ** 0.5 * z_new
        return z_new, self.embed(z_new)

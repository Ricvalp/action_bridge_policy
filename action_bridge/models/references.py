"""Reference action processes used during training and inference."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from action_bridge.models.encoders import SinusoidalTimeEmbedding, make_mlp, timestep_tensor


class ReferenceProcess(nn.Module):
    def forward(self, a_k: torch.Tensor, a_k_minus_1: torch.Tensor, h_emb: torch.Tensor, k: int, extra: Optional[dict] = None):
        raise NotImplementedError


class BrownianReference(ReferenceProcess):
    """Raw-action random-walk reference: mu_R = a_k."""

    def __init__(self, action_dim: int, sigma: float = 0.05, learn_sigma: bool = False):
        super().__init__()
        self.action_dim = action_dim
        init = torch.full((action_dim,), float(sigma)).log()
        self.log_sigma = nn.Parameter(init, requires_grad=learn_sigma)

    def forward(self, a_k: torch.Tensor, a_k_minus_1: torch.Tensor, h_emb: torch.Tensor, k: int, extra: Optional[dict] = None):
        del a_k_minus_1, h_emb, k, extra
        log_sigma = self.log_sigma.clamp(-8.0, 2.0).expand_as(a_k)
        return a_k, log_sigma


class ContinuationReference(ReferenceProcess):
    """Low-acceleration continuation: mu_R = a_k + alpha(a_k - a_{k-1})."""

    def __init__(
        self,
        action_dim: int,
        sigma: float = 0.05,
        alpha: float = 0.8,
        learn_alpha: bool = False,
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        alpha = min(0.999, max(0.001, float(alpha)))
        alpha_logit = torch.logit(torch.tensor(alpha))
        self.alpha_logit = nn.Parameter(alpha_logit, requires_grad=learn_alpha)
        init = torch.full((action_dim,), float(sigma)).log()
        self.log_sigma = nn.Parameter(init, requires_grad=learn_sigma)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, a_k: torch.Tensor, a_k_minus_1: torch.Tensor, h_emb: torch.Tensor, k: int, extra: Optional[dict] = None):
        del h_emb, k, extra
        mu = a_k + self.alpha * (a_k - a_k_minus_1)
        log_sigma = self.log_sigma.clamp(-8.0, 2.0).expand_as(a_k)
        return mu, log_sigma


class LowAccelerationReference(ReferenceProcess):
    """Continuation with phase/context-dependent damping alpha_k."""

    def __init__(
        self,
        action_dim: int,
        h_emb_dim: int,
        time_emb_dim: int = 32,
        hidden_dim: int = 128,
        sigma: float = 0.05,
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_emb_dim)
        self.alpha_net = make_mlp(h_emb_dim + time_emb_dim, 1, hidden_dim, depth=2)
        self.log_sigma = nn.Parameter(torch.full((action_dim,), float(sigma)).log(), requires_grad=learn_sigma)

    def forward(self, a_k: torch.Tensor, a_k_minus_1: torch.Tensor, h_emb: torch.Tensor, k: int, extra: Optional[dict] = None):
        del extra
        tk = timestep_tensor(k, a_k.shape[0], a_k.device)
        t_emb = self.time(tk).to(dtype=a_k.dtype)
        alpha = torch.sigmoid(self.alpha_net(torch.cat([h_emb, t_emb], dim=-1)))
        mu = a_k + alpha * (a_k - a_k_minus_1)
        log_sigma = self.log_sigma.clamp(-8.0, 2.0).expand_as(a_k)
        return mu, log_sigma


class LowJerkReference(ReferenceProcess):
    """Optional second-order reference; falls back to continuation without a_{k-2}."""

    def __init__(self, action_dim: int, sigma: float = 0.05, rho: float = 0.2, learn_sigma: bool = False):
        super().__init__()
        self.rho = float(rho)
        self.log_sigma = nn.Parameter(torch.full((action_dim,), float(sigma)).log(), requires_grad=learn_sigma)

    def forward(self, a_k: torch.Tensor, a_k_minus_1: torch.Tensor, h_emb: torch.Tensor, k: int, extra: Optional[dict] = None):
        del h_emb, k
        velocity = a_k - a_k_minus_1
        if extra is not None and "a_k_minus_2" in extra:
            prev_velocity = a_k_minus_1 - extra["a_k_minus_2"]
            accel = velocity - prev_velocity
        else:
            accel = torch.zeros_like(velocity)
        mu = a_k + velocity + self.rho * accel
        log_sigma = self.log_sigma.clamp(-8.0, 2.0).expand_as(a_k)
        return mu, log_sigma


def build_reference(config: Dict, action_dim: int, h_emb_dim: int) -> ReferenceProcess:
    kind = config.get("type", "continuation")
    sigma = float(config.get("sigma", 0.05))
    learn_sigma = bool(config.get("learn_sigma", False))
    if kind in {"brownian", "raw_action"}:
        return BrownianReference(action_dim=action_dim, sigma=sigma, learn_sigma=learn_sigma)
    if kind == "continuation":
        return ContinuationReference(
            action_dim=action_dim,
            sigma=sigma,
            alpha=float(config.get("alpha", 0.8)),
            learn_alpha=bool(config.get("learn_alpha", False)),
            learn_sigma=learn_sigma,
        )
    if kind in {"low_acceleration", "learned_alpha"}:
        return LowAccelerationReference(
            action_dim=action_dim,
            h_emb_dim=h_emb_dim,
            time_emb_dim=int(config.get("time_emb_dim", 32)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            sigma=sigma,
            learn_sigma=learn_sigma,
        )
    if kind == "low_jerk":
        return LowJerkReference(action_dim=action_dim, sigma=sigma, rho=float(config.get("rho", 0.2)), learn_sigma=learn_sigma)
    raise ValueError(f"Unknown reference type {kind!r}.")

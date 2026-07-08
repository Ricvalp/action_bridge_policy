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


def bounded_positive(raw: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return float(min_val) + (float(max_val) - float(min_val)) * torch.sigmoid(raw)


class ContactLangevinReference(ReferenceProcess):
    """Underdamped Langevin/contact-Hamiltonian reference over q and p."""

    is_contact_langevin = True

    def __init__(
        self,
        action_dim: int,
        h_emb_dim: int,
        sigma: float = 0.05,
        dt: float = 1.0,
        control_is_whitened: bool = True,
        gamma_mode: str = "constant",
        gamma_const: float = 0.2,
        gamma_min: float = 0.0,
        gamma_max: float = 0.95,
        potential_type: str = "none",
        stiffness_mode: str = "learned_diag",
        k_const: float = 0.0,
        k_min: float = 0.0,
        k_max: float = 2.0,
        attractor_mode: str = "learned",
        time_emb_dim: int = 32,
        hidden_dim: int = 128,
        beta_kl: float = 1.0,
        lambda_q: float = 1.0,
        lambda_ref_reg: float = 1e-4,
        lambda_m_smooth: float = 1e-3,
        deterministic_inference: bool = True,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.dt = float(dt)
        self.control_is_whitened = bool(control_is_whitened)
        self.gamma_mode = str(gamma_mode)
        self.gamma_const = float(gamma_const)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.potential_type = str(potential_type)
        self.stiffness_mode = str(stiffness_mode)
        self.k_const = float(k_const)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.attractor_mode = str(attractor_mode)
        self.beta_kl = float(beta_kl)
        self.lambda_q = float(lambda_q)
        self.lambda_ref_reg = float(lambda_ref_reg)
        self.lambda_m_smooth = float(lambda_m_smooth)
        self.deterministic_inference = bool(deterministic_inference)

        if self.potential_type not in {"none", "quadratic"}:
            raise ValueError(f"Unknown potential_type {self.potential_type!r}.")
        if self.gamma_mode not in {"constant", "learned_scalar", "learned_diag"}:
            raise ValueError(f"Unknown gamma_mode {self.gamma_mode!r}.")
        if self.stiffness_mode not in {"constant", "learned_diag"}:
            raise ValueError(f"Unknown stiffness_mode {self.stiffness_mode!r}.")
        if self.attractor_mode not in {"learned", "previous_q", "zero"}:
            raise ValueError(f"Unknown attractor_mode {self.attractor_mode!r}.")

        self.time = SinusoidalTimeEmbedding(time_emb_dim)
        in_dim = int(h_emb_dim) + int(time_emb_dim)
        self.log_sigma = nn.Parameter(torch.full((self.action_dim,), float(sigma)).log(), requires_grad=False)

        if self.potential_type == "quadratic" and self.attractor_mode == "learned":
            self.m_net = make_mlp(in_dim, self.action_dim, hidden_dim, depth=2)
        else:
            self.m_net = None
        if self.potential_type == "quadratic" and self.stiffness_mode == "learned_diag":
            self.k_net = make_mlp(in_dim, self.action_dim, hidden_dim, depth=2)
        else:
            self.k_net = None

        if self.gamma_mode == "learned_scalar":
            self.gamma_net = make_mlp(in_dim, 1, hidden_dim, depth=2)
        elif self.gamma_mode == "learned_diag":
            self.gamma_net = make_mlp(in_dim, self.action_dim, hidden_dim, depth=2)
        else:
            self.gamma_net = None

    def sigma_like(self, q: torch.Tensor) -> torch.Tensor:
        return self.log_sigma.exp().clamp_min(1e-6).to(dtype=q.dtype, device=q.device).expand_as(q)

    def time_embedding(self, k: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tk = timestep_tensor(k, batch_size, device)
        return self.time(tk).to(dtype=dtype)

    def params(self, q: torch.Tensor, h_emb: torch.Tensor, k: int):
        t_emb = self.time_embedding(k, q.shape[0], q.device, q.dtype)
        hk = torch.cat([h_emb, t_emb], dim=-1)

        m = None
        k_diag = None
        if self.potential_type == "quadratic":
            if self.attractor_mode == "learned":
                m = self.m_net(hk)
            elif self.attractor_mode == "previous_q":
                m = q
            else:
                m = torch.zeros_like(q)

            if self.stiffness_mode == "learned_diag":
                k_diag = bounded_positive(self.k_net(hk), self.k_min, self.k_max)
            else:
                k_diag = torch.full_like(q, self.k_const)

        if self.gamma_mode == "constant":
            gamma = torch.full((q.shape[0], 1), self.gamma_const, device=q.device, dtype=q.dtype)
        else:
            gamma = bounded_positive(self.gamma_net(hk), self.gamma_min, self.gamma_max)
        return m, k_diag, gamma

    def force(self, q: torch.Tensor, p: torch.Tensor, h_emb: torch.Tensor, k: int):
        m, k_diag, gamma = self.params(q, h_emb, k)
        grad_v = k_diag * (q - m) if k_diag is not None and m is not None else torch.zeros_like(q)
        damping = gamma * p
        f_ref = -grad_v - damping
        aux = {"m": m, "k_diag": k_diag, "gamma": gamma, "grad_v": grad_v}
        return f_ref, aux

    def reference_step(self, q: torch.Tensor, p: torch.Tensor, h_emb: torch.Tensor, k: int):
        f_ref, aux = self.force(q, p, h_emb, k)
        p_next = p + self.dt * f_ref
        q_next = q + self.dt * p_next
        return q_next, p_next, aux


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
    if kind == "contact_langevin":
        return ContactLangevinReference(
            action_dim=action_dim,
            h_emb_dim=h_emb_dim,
            sigma=sigma,
            dt=float(config.get("dt", 1.0)),
            control_is_whitened=bool(config.get("control_is_whitened", True)),
            gamma_mode=str(config.get("gamma_mode", "constant")),
            gamma_const=float(config.get("gamma_const", 0.2)),
            gamma_min=float(config.get("gamma_min", 0.0)),
            gamma_max=float(config.get("gamma_max", 0.95)),
            potential_type=str(config.get("potential_type", "none")),
            stiffness_mode=str(config.get("stiffness_mode", "learned_diag")),
            k_const=float(config.get("k_const", 0.0)),
            k_min=float(config.get("k_min", 0.0)),
            k_max=float(config.get("k_max", 2.0)),
            attractor_mode=str(config.get("attractor_mode", "learned")),
            time_emb_dim=int(config.get("time_emb_dim", config.get("time_embed_dim", 32))),
            hidden_dim=int(config.get("hidden_dim", 128)),
            beta_kl=float(config.get("beta_kl", config.get("beta_R", 1.0))),
            lambda_q=float(config.get("lambda_q", 1.0)),
            lambda_ref_reg=float(config.get("lambda_ref_reg", 1e-4)),
            lambda_m_smooth=float(config.get("lambda_m_smooth", 1e-3)),
            deterministic_inference=bool(config.get("deterministic_inference", True)),
        )
    raise ValueError(f"Unknown reference type {kind!r}.")

"""Reference action processes used during training and inference."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from action_bridge.models.encoders import SinusoidalTimeEmbedding, make_mlp, timestep_tensor
from action_bridge.models.geometric_pusht import (
    default_t_polygon_local,
    safe_unit_torch,
    sample_polygon_boundary_local,
    sigmoid_torch,
    target_pose_tensor,
    transform_boundary_batch,
    wrap_angle_torch,
)


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

    def force(self, q: torch.Tensor, p: torch.Tensor, h_emb: torch.Tensor, k: int, obs_state: Optional[torch.Tensor] = None):
        del obs_state
        m, k_diag, gamma = self.params(q, h_emb, k)
        grad_v = k_diag * (q - m) if k_diag is not None and m is not None else torch.zeros_like(q)
        damping = gamma * p
        f_ref = -grad_v - damping
        aux = {"m": m, "k_diag": k_diag, "gamma": gamma, "grad_v": grad_v}
        return f_ref, aux

    def reference_step(self, q: torch.Tensor, p: torch.Tensor, h_emb: torch.Tensor, k: int, obs_state: Optional[torch.Tensor] = None):
        f_ref, aux = self.force(q, p, h_emb, k, obs_state=obs_state)
        p_next = p + self.dt * f_ref
        q_next = q + self.dt * p_next
        return q_next, p_next, aux


class GeometricPushTReference(ReferenceProcess):
    """Frozen geometric Push-T reference in denormalized pixel coordinates."""

    is_contact_langevin = True
    is_geometric_pusht = True

    def __init__(
        self,
        action_dim: int,
        sigma: float = 3.0,
        dt: float = 1.0,
        control_is_whitened: bool = True,
        target_pose: Optional[list[float]] = None,
        normalization_stats: Optional[Dict[str, Any]] = None,
        n_per_edge: int = 16,
        pusher_radius: float = 5.0,
        delta_pre: float = 8.0,
        delta_push_far: float = 7.0,
        delta_push_near: float = 2.0,
        lambda_tau: float = 0.35,
        lambda_travel: float = 0.0005,
        k_trans: float = 1.0,
        k_rot: float = 0.35,
        soft_contact_selection: bool = True,
        contact_softmax_temp: float = 0.05,
        contact_gate_d0: float = 8.0,
        tau_contact: float = 4.0,
        tau_goal: float = 35.0,
        lambda_theta: float = 25.0,
        K_free: float = 0.04,
        K_contact: float = 0.12,
        K_goal_gain: float = 1.5,
        K_min: float = 0.0,
        K_max: float = 0.35,
        gamma_free: float = 0.08,
        gamma_contact: float = 0.18,
        gamma_goal: float = 0.35,
        gamma_min: float = 0.02,
        gamma_max: float = 0.85,
        max_step_norm: float = 12.0,
        beta_kl: float = 0.003,
        lambda_q: float = 1.0,
        deterministic_inference: bool = True,
    ):
        super().__init__()
        if int(action_dim) != 2:
            raise ValueError("GeometricPushTReference requires action_dim=2.")
        self.action_dim = int(action_dim)
        self.dt = float(dt)
        self.control_is_whitened = bool(control_is_whitened)
        self.beta_kl = float(beta_kl)
        self.lambda_q = float(lambda_q)
        self.lambda_ref_reg = 0.0
        self.lambda_m_smooth = 0.0
        self.deterministic_inference = bool(deterministic_inference)

        self.n_per_edge = int(n_per_edge)
        self.pusher_radius = float(pusher_radius)
        self.delta_pre = float(delta_pre)
        self.delta_push_far = float(delta_push_far)
        self.delta_push_near = float(delta_push_near)
        self.lambda_tau = float(lambda_tau)
        self.lambda_travel = float(lambda_travel)
        self.k_trans = float(k_trans)
        self.k_rot = float(k_rot)
        self.soft_contact_selection = bool(soft_contact_selection)
        self.contact_softmax_temp = float(contact_softmax_temp)
        self.contact_gate_d0 = float(contact_gate_d0)
        self.tau_contact = float(tau_contact)
        self.tau_goal = float(tau_goal)
        self.lambda_theta = float(lambda_theta)
        self.K_free = float(K_free)
        self.K_contact = float(K_contact)
        self.K_goal_gain = float(K_goal_gain)
        self.K_min = float(K_min)
        self.K_max = float(K_max)
        self.gamma_free = float(gamma_free)
        self.gamma_contact = float(gamma_contact)
        self.gamma_goal = float(gamma_goal)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.max_step_norm = float(max_step_norm)

        self.log_sigma = nn.Parameter(torch.full((self.action_dim,), float(sigma)).log(), requires_grad=False)

        poly = default_t_polygon_local()
        boundary_local, normals_local = sample_polygon_boundary_local(poly, self.n_per_edge)
        self.register_buffer("t_polygon_local", poly)
        self.register_buffer("boundary_local", boundary_local)
        self.register_buffer("normals_local", normals_local)

        target = target_pose_tensor(torch.device("cpu"), torch.float32, target_pose)
        self.register_buffer("target_pose", target)

        stats = normalization_stats or {}
        action_mean = torch.as_tensor(stats.get("action_mean", [0.0, 0.0]), dtype=torch.float32)
        action_std = torch.as_tensor(stats.get("action_std", [1.0, 1.0]), dtype=torch.float32).clamp_min(1e-6)
        obs_mean = torch.as_tensor(stats.get("obs_mean", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32)
        obs_std = torch.as_tensor(stats.get("obs_std", [1.0, 1.0, 1.0, 1.0, 1.0]), dtype=torch.float32).clamp_min(1e-6)
        self.register_buffer("action_mean", action_mean[:2].clone())
        self.register_buffer("action_std", action_std[:2].clone())
        self.register_buffer("obs_mean", obs_mean.clone())
        self.register_buffer("obs_std", obs_std.clone())

    def sigma_like(self, q: torch.Tensor) -> torch.Tensor:
        return self.log_sigma.exp().clamp_min(1e-6).to(dtype=q.dtype, device=q.device).expand_as(q)

    def _denorm_action(self, q: torch.Tensor) -> torch.Tensor:
        mean = self.action_mean.to(device=q.device, dtype=q.dtype)
        std = self.action_std.to(device=q.device, dtype=q.dtype)
        return q * std + mean

    def _norm_action(self, q_px: torch.Tensor) -> torch.Tensor:
        mean = self.action_mean.to(device=q_px.device, dtype=q_px.dtype)
        std = self.action_std.to(device=q_px.device, dtype=q_px.dtype)
        return (q_px - mean) / std

    def _denorm_action_delta(self, p: torch.Tensor) -> torch.Tensor:
        std = self.action_std.to(device=p.device, dtype=p.dtype)
        return p * std

    def _norm_action_delta(self, p_px: torch.Tensor) -> torch.Tensor:
        std = self.action_std.to(device=p_px.device, dtype=p_px.dtype)
        return p_px / std

    def _denorm_obs_state(self, obs_state: torch.Tensor) -> torch.Tensor:
        if obs_state is None:
            raise ValueError("GeometricPushTReference requires obs_state=[x_p,y_p,x_T,y_T,theta_T].")
        if obs_state.shape[-1] < 5:
            raise ValueError(f"GeometricPushTReference requires obs_state with at least 5 dims, got {obs_state.shape}.")
        obs5 = obs_state[..., :5]
        mean = self.obs_mean[:5].to(device=obs_state.device, dtype=obs_state.dtype)
        std = self.obs_std[:5].to(device=obs_state.device, dtype=obs_state.dtype)
        return obs5 * std + mean

    def geometric_params(self, q: torch.Tensor, p: torch.Tensor, obs_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        q_px = self._denorm_action(q)
        p_px = self._denorm_action_delta(p)
        obs_px = self._denorm_obs_state(obs_state)
        pusher_pos = obs_px[:, :2]
        block_pos = obs_px[:, 2:4]
        block_theta = obs_px[:, 4]
        target = self.target_pose.to(device=q.device, dtype=q.dtype)
        target_pos = target[:2]
        target_theta = target[2]

        boundary_pts, normals = transform_boundary_batch(
            self.boundary_local.to(device=q.device, dtype=q.dtype),
            self.normals_local.to(device=q.device, dtype=q.dtype),
            block_pos,
            block_theta,
        )
        e_pos = target_pos[None, :] - block_pos
        e_theta = wrap_angle_torch(target_theta - block_theta)
        w_des = torch.cat([self.k_trans * e_pos, self.k_rot * e_theta[:, None]], dim=-1)
        w_des = safe_unit_torch(w_des)

        push_force = -normals
        r = boundary_pts - block_pos[:, None, :]
        tau = r[..., 0] * push_force[..., 1] - r[..., 1] * push_force[..., 0]
        w = torch.cat([push_force, self.lambda_tau * tau[..., None]], dim=-1)
        w = safe_unit_torch(w)

        m_pre_all = boundary_pts + (self.pusher_radius + self.delta_pre) * normals
        travel = (pusher_pos[:, None, :] - m_pre_all).pow(2).sum(dim=-1)
        scores = (w * w_des[:, None, :]).sum(dim=-1) - self.lambda_travel * travel

        if self.soft_contact_selection:
            weights = torch.softmax(scores / max(self.contact_softmax_temp, 1e-6), dim=-1)
            b_star = (weights[..., None] * boundary_pts).sum(dim=1)
            n_star = safe_unit_torch((weights[..., None] * normals).sum(dim=1))
            m_pre = (weights[..., None] * m_pre_all).sum(dim=1)
        else:
            idx = scores.argmax(dim=-1)
            batch_idx = torch.arange(q.shape[0], device=q.device)
            b_star = boundary_pts[batch_idx, idx]
            n_star = normals[batch_idx, idx]
            m_pre = m_pre_all[batch_idx, idx]

        d_contact = torch.linalg.norm(q_px - b_star, dim=-1) - self.pusher_radius
        rho_contact = sigmoid_torch((self.contact_gate_d0 - d_contact) / max(self.tau_contact, 1e-6))
        e_pos_norm = torch.linalg.norm(block_pos - target_pos[None, :], dim=-1)
        e_theta_goal = wrap_angle_torch(block_theta - target_theta)
        goal_err = e_pos_norm.pow(2) + self.lambda_theta * e_theta_goal.pow(2)
        rho_goal = torch.exp(-goal_err / max(self.tau_goal * self.tau_goal, 1e-6))

        delta_push = self.delta_push_far * (1.0 - rho_goal) + self.delta_push_near * rho_goal
        m_push = b_star - delta_push[:, None] * n_star
        m_geo = (1.0 - rho_contact[:, None]) * m_pre + rho_contact[:, None] * m_push

        K = self.K_free * (1.0 - rho_contact) + self.K_contact * rho_contact
        K = K * (1.0 + self.K_goal_gain * rho_goal)
        K = K.clamp(self.K_min, self.K_max)
        gamma = self.gamma_free + self.gamma_contact * rho_contact + self.gamma_goal * rho_goal
        gamma = gamma.clamp(self.gamma_min, self.gamma_max)

        grad_v_px = K[:, None] * (q_px - m_geo)
        f_ref_px = -grad_v_px - gamma[:, None] * p_px
        f_ref = self._norm_action_delta(f_ref_px)
        m_norm = self._norm_action(m_geo)
        m_pre_norm = self._norm_action(m_pre)
        m_push_norm = self._norm_action(m_push)
        b_star_norm = self._norm_action(b_star)
        grad_v = self._norm_action_delta(grad_v_px)
        return {
            "f_ref": f_ref,
            "m": m_norm,
            "k_diag": K[:, None].expand_as(q),
            "gamma": gamma[:, None],
            "grad_v": grad_v,
            "boundary_pts_px": boundary_pts,
            "normals": normals,
            "b_star_px": b_star,
            "n_star": n_star,
            "m_pre_px": m_pre,
            "m_push_px": m_push,
            "m_geo_px": m_geo,
            "m_pre": m_pre_norm,
            "m_push": m_push_norm,
            "b_star": b_star_norm,
            "rho_contact": rho_contact,
            "rho_goal": rho_goal,
            "d_contact": d_contact,
            "goal_err": goal_err,
            "delta_push": delta_push,
            "scores": scores,
        }

    def force(self, q: torch.Tensor, p: torch.Tensor, h_emb: torch.Tensor, k: int, obs_state: Optional[torch.Tensor] = None):
        del h_emb, k
        params = self.geometric_params(q, p, obs_state)
        f_ref = params.pop("f_ref")
        return f_ref, params

    def reference_step(self, q: torch.Tensor, p: torch.Tensor, h_emb: torch.Tensor, k: int, obs_state: Optional[torch.Tensor] = None):
        f_ref, aux = self.force(q, p, h_emb, k, obs_state=obs_state)
        p_next = p + self.dt * f_ref
        step_norm = torch.linalg.norm(self._denorm_action_delta(p_next), dim=-1, keepdim=True)
        if self.max_step_norm > 0:
            scale = (self.max_step_norm / step_norm.clamp_min(1e-8)).clamp_max(1.0)
            p_next = self._norm_action_delta(self._denorm_action_delta(p_next) * scale)
        q_next = q + self.dt * p_next
        q_next_px = self._denorm_action(q_next).clamp(0.0, 512.0)
        q_next = self._norm_action(q_next_px)
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
    if kind == "geometric_pusht":
        return GeometricPushTReference(
            action_dim=action_dim,
            sigma=float(config.get("sigma", 3.0)),
            dt=float(config.get("dt", 1.0)),
            control_is_whitened=bool(config.get("control_is_whitened", True)),
            target_pose=config.get("target_pose", None),
            normalization_stats=config.get("normalization_stats", None),
            n_per_edge=int(config.get("n_per_edge", 16)),
            pusher_radius=float(config.get("pusher_radius", 5.0)),
            delta_pre=float(config.get("delta_pre", 8.0)),
            delta_push_far=float(config.get("delta_push_far", 7.0)),
            delta_push_near=float(config.get("delta_push_near", 2.0)),
            lambda_tau=float(config.get("lambda_tau", 0.35)),
            lambda_travel=float(config.get("lambda_travel", 0.0005)),
            k_trans=float(config.get("k_trans", 1.0)),
            k_rot=float(config.get("k_rot", 0.35)),
            soft_contact_selection=bool(config.get("soft_contact_selection", True)),
            contact_softmax_temp=float(config.get("contact_softmax_temp", 0.05)),
            contact_gate_d0=float(config.get("contact_gate_d0", 8.0)),
            tau_contact=float(config.get("tau_contact", 4.0)),
            tau_goal=float(config.get("tau_goal", 35.0)),
            lambda_theta=float(config.get("lambda_theta", 25.0)),
            K_free=float(config.get("K_free", 0.04)),
            K_contact=float(config.get("K_contact", 0.12)),
            K_goal_gain=float(config.get("K_goal_gain", 1.5)),
            K_min=float(config.get("K_min", 0.0)),
            K_max=float(config.get("K_max", 0.35)),
            gamma_free=float(config.get("gamma_free", 0.08)),
            gamma_contact=float(config.get("gamma_contact", 0.18)),
            gamma_goal=float(config.get("gamma_goal", 0.35)),
            gamma_min=float(config.get("gamma_min", 0.02)),
            gamma_max=float(config.get("gamma_max", 0.85)),
            max_step_norm=float(config.get("max_step_norm", 12.0)),
            beta_kl=float(config.get("beta_kl", config.get("beta_R", 0.003))),
            lambda_q=float(config.get("lambda_q", 1.0)),
            deterministic_inference=bool(config.get("deterministic_inference", True)),
        )
    raise ValueError(f"Unknown reference type {kind!r}.")

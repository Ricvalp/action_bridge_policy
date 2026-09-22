"""A learned normal spring and contact-frame damping of *target* motion.

The physical point is not the reference state. Here q is a normalized Cartesian
target and p is its difference per command index (dt=1), not physical velocity.
"""

from __future__ import annotations

import torch
from torch import nn

from action_bridge.models.encoders import make_mlp
from action_bridge.models.references import ContactLangevinReference, bounded_positive


def contact_frame(normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    normal = torch.nn.functional.normalize(normal, dim=-1)
    tangent = torch.stack((normal[..., 1], -normal[..., 0]), dim=-1)
    return normal, tangent


def contact_projectors(normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    normal, _ = contact_frame(normal)
    pn = normal.unsqueeze(-1) * normal.unsqueeze(-2)
    eye = torch.eye(2, dtype=normal.dtype, device=normal.device)
    return pn, eye - pn


class ContactFrameReference(ContactLangevinReference):
    """Positive, bounded eigenvalues; no ordering or tangential attractor.

    Normal target offset is learned in physical metres. Surface geometry is
    recovered from the common observation normalizer, then transformed exactly
    to the isotropically normalized action coordinates.
    """

    def __init__(self, config: dict, stats: dict):
        super().__init__(
            action_dim=2,
            h_emb_dim=int(config.get("h_emb_dim", 128)),
            sigma=float(config.get("sigma", 1.0)),
            dt=1.0,
            control_is_whitened=True,
            gamma_mode="constant",
            potential_type="none",
            time_emb_dim=int(config.get("time_emb_dim", 32)),
            beta_kl=float(config.get("beta_kl", 0.001)),
            lambda_q=float(config.get("lambda_q", 1.0)),
            lambda_ref_reg=float(config.get("lambda_ref_reg", 0.0)),
            lambda_m_smooth=0.0,
        )
        self.isotropic = config["method"] == "action_bridge_isotropic"
        self.gamma_min = float(config.get("gamma_min", 0.0))
        self.gamma_max = float(config.get("gamma_max", 0.8))
        self.k_min = float(config.get("stiffness_min", 0.0))
        self.k_max = float(config.get("stiffness_max", 0.5))
        self.offset_max = float(config.get("offset_max", 0.04))
        if not (0 <= self.gamma_min < self.gamma_max < 2):
            raise ValueError("Require 0 <= gamma_min < gamma_max < 2 at command dt=1")
        if not (0 <= self.k_min < self.k_max < 2 * (2 - self.gamma_max)):
            raise ValueError("Stiffness bounds must satisfy the semi-implicit stability bound")
        self.register_buffer("obs_mean", torch.tensor(stats["obs_mean"], dtype=torch.float32))
        self.register_buffer("obs_scale", torch.tensor(stats["obs_scale"], dtype=torch.float32))
        self.register_buffer("action_mean", torch.tensor(stats["action_mean"], dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(stats["action_scale"], dtype=torch.float32))
        h_dim = int(config.get("h_emb_dim", 128))
        t_dim = int(config.get("time_emb_dim", 32))
        n_gamma = 1 if self.isotropic else 2
        self.parameter_net = make_mlp(h_dim + t_dim, n_gamma + 2, int(config.get("hidden_dim", 128)), depth=2)
        # A weak, symmetric starting reference does not prescribe the answer.
        head = self.parameter_net[-1]
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        with torch.no_grad():
            gamma_fraction = (0.04 - self.gamma_min) / (self.gamma_max - self.gamma_min)
            stiffness_fraction = (0.01 - self.k_min) / (self.k_max - self.k_min)
            head.bias[:n_gamma] = torch.logit(torch.tensor(gamma_fraction).clamp(0.01, 0.99))
            head.bias[n_gamma] = torch.logit(torch.tensor(stiffness_fraction).clamp(0.01, 0.99))

    def geometry(self, obs_state: torch.Tensor):
        raw = obs_state * self.obs_scale + self.obs_mean
        normal, tangent = contact_frame(raw[..., 4:6])
        offset = (raw[..., 6] - (normal * self.action_mean).sum(-1)) / self.action_scale
        return normal, tangent, offset

    def parameters_at(self, h_emb: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
        phase = self.time_embedding(k, h_emb.shape[0], h_emb.device, h_emb.dtype)
        raw = self.parameter_net(torch.cat((h_emb, phase), dim=-1))
        n_gamma = 1 if self.isotropic else 2
        gamma = bounded_positive(raw[..., :n_gamma], self.gamma_min, self.gamma_max)
        return {
            "gamma_normal": gamma[..., 0],
            "gamma_tangent": gamma[..., 0 if self.isotropic else 1],
            "stiffness": bounded_positive(raw[..., n_gamma], self.k_min, self.k_max),
            "desired_offset": self.offset_max * torch.tanh(raw[..., n_gamma + 1]),
        }

    def force(self, q, p, h_emb, k, obs_state=None):
        if obs_state is None:
            raise ValueError("Contact-frame reference requires the current observation geometry")
        normal, tangent, surface_offset = self.geometry(obs_state)
        values = self.parameters_at(h_emb, k)
        target_offset = surface_offset + values["desired_offset"] / self.action_scale
        error = (q * normal).sum(-1) - target_offset
        grad_v = (values["stiffness"] * error)[..., None] * normal
        normal_damping = values["gamma_normal"] * (p * normal).sum(-1)
        tangent_damping = values["gamma_tangent"] * (p * tangent).sum(-1)
        force = -grad_v - normal_damping[..., None] * normal - tangent_damping[..., None] * tangent
        return force, {
            **values,
            "normal": normal,
            "tangent": tangent,
            "grad_v": grad_v,
            "gamma": torch.stack((values["gamma_normal"], values["gamma_tangent"]), -1),
            "k_diag": values["stiffness"][..., None] * normal.square(),
            "m": target_offset[..., None] * normal,
        }


def path_diagnostics(reference, force, control, aux):
    """Force projections share normalized target-acceleration units; u is whitened."""
    normal, tangent = aux["normal"], aux["tangent"]
    residual = reference.sigma_like(force) * control
    return {
        **{key: aux[key] for key in ("gamma_normal", "gamma_tangent", "stiffness", "desired_offset")},
        "ref_normal": (force * normal).sum(-1),
        "ref_tangent": (force * tangent).sum(-1),
        "residual_normal": (residual * normal).sum(-1),
        "residual_tangent": (residual * tangent).sum(-1),
        "control_normal": (control * normal).sum(-1),
        "control_tangent": (control * tangent).sum(-1),
        "control_ref_ratio": residual.norm(dim=-1) / force.norm(dim=-1).clamp_min(1e-8),
        "path_kl": 0.5 * reference.dt * control.square().sum(-1),
    }

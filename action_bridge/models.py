"""Policy models for execution-time action bridge experiments."""

from __future__ import annotations

import torch
from torch import nn


def mlp(dims: list[int], activation: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ActionContextEncoder(nn.Module):
    """Encode recent states and actions into a compact history vector."""

    def __init__(
        self,
        context: int,
        state_dim: int,
        action_dim: int,
        history_dim: int,
        hidden_dim: int,
        use_context_actions: bool = True,
    ):
        super().__init__()
        self.context = context
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_context_actions = use_context_actions
        in_dim = context * (state_dim + action_dim)
        self.net = mlp([in_dim, hidden_dim, hidden_dim, history_dim])

    def forward(self, context_states: torch.Tensor, context_actions: torch.Tensor) -> torch.Tensor:
        if not self.use_context_actions:
            context_actions = torch.zeros_like(context_actions)
        flat = torch.cat(
            [
                context_states.reshape(context_states.shape[0], -1),
                context_actions.reshape(context_actions.shape[0], -1),
            ],
            dim=-1,
        )
        return self.net(flat)


class ChunkMLPPolicy(nn.Module):
    """One-shot action chunk predictor, optionally conditioned on Gaussian noise."""

    def __init__(
        self,
        context: int,
        horizon: int,
        state_dim: int = 4,
        action_dim: int = 2,
        history_dim: int = 96,
        hidden_dim: int = 192,
        noise_dim: int = 0,
        noise_scale: float = 1.0,
        action_limit: float = 1.0,
        use_context_actions: bool = True,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.noise_dim = noise_dim
        self.noise_scale = noise_scale
        self.action_limit = action_limit
        self.encoder = ActionContextEncoder(
            context, state_dim, action_dim, history_dim, hidden_dim, use_context_actions
        )
        self.head = mlp([history_dim + noise_dim, hidden_dim, hidden_dim, horizon * action_dim])

    def sample_noise(self, batch_size: int, device: torch.device, dtype: torch.dtype, deterministic: bool) -> torch.Tensor:
        if self.noise_dim <= 0:
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)
        if deterministic:
            return torch.zeros(batch_size, self.noise_dim, device=device, dtype=dtype)
        return self.noise_scale * torch.randn(batch_size, self.noise_dim, device=device, dtype=dtype)

    def forward(self, batch: dict[str, torch.Tensor], deterministic: bool = False) -> dict[str, torch.Tensor]:
        context_states = batch["context_states"]
        context_actions = batch["context_actions"]
        h = self.encoder(context_states, context_actions)
        noise = self.sample_noise(h.shape[0], h.device, h.dtype, deterministic)
        pred = self.head(torch.cat([h, noise], dim=-1))
        pred = pred.reshape(h.shape[0], self.horizon, self.action_dim)
        pred = self.action_limit * torch.tanh(pred)
        init_action = context_actions[:, -1]
        return {
            "actions": pred,
            "init_action": init_action,
            "history": h,
            "noise": noise,
        }

    @property
    def network_evals(self) -> int:
        return 1


class ResidualActionBridgePolicy(nn.Module):
    """Execution-time residual action bridge.

    The recurrent outputs are the actual future actions, not artificial sampler
    states. The first bridge state is initialized from the previous action or
    from a Gaussian ablation.
    """

    def __init__(
        self,
        context: int,
        horizon: int,
        state_dim: int = 4,
        action_dim: int = 2,
        history_dim: int = 96,
        hidden_dim: int = 192,
        tau: float = 0.45,
        init_type: str = "prev_action",
        init_noise_scale: float = 0.5,
        noise_dim: int = 0,
        noise_scale: float = 1.0,
        action_limit: float = 1.0,
        use_context_actions: bool = True,
    ):
        super().__init__()
        if init_type not in ("prev_action", "zero", "gaussian"):
            raise ValueError(f"Unknown init_type: {init_type}")
        self.horizon = horizon
        self.action_dim = action_dim
        self.tau = tau
        self.init_type = init_type
        self.init_noise_scale = init_noise_scale
        self.noise_dim = noise_dim
        self.noise_scale = noise_scale
        self.action_limit = action_limit
        self.encoder = ActionContextEncoder(
            context, state_dim, action_dim, history_dim, hidden_dim, use_context_actions
        )
        block_dim = action_dim + history_dim + noise_dim + 1
        self.blocks = nn.ModuleList(
            [mlp([block_dim, hidden_dim, hidden_dim, action_dim]) for _ in range(horizon)]
        )

    def sample_noise(self, batch_size: int, device: torch.device, dtype: torch.dtype, deterministic: bool) -> torch.Tensor:
        if self.noise_dim <= 0:
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)
        if deterministic:
            return torch.zeros(batch_size, self.noise_dim, device=device, dtype=dtype)
        return self.noise_scale * torch.randn(batch_size, self.noise_dim, device=device, dtype=dtype)

    def initial_action(self, context_actions: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if self.init_type == "prev_action":
            return context_actions[:, -1]
        if self.init_type == "zero":
            return torch.zeros_like(context_actions[:, -1])
        if deterministic:
            return torch.zeros_like(context_actions[:, -1])
        return self.init_noise_scale * torch.randn_like(context_actions[:, -1])

    def forward(self, batch: dict[str, torch.Tensor], deterministic: bool = False) -> dict[str, torch.Tensor]:
        context_states = batch["context_states"]
        context_actions = batch["context_actions"]
        h = self.encoder(context_states, context_actions)
        noise = self.sample_noise(h.shape[0], h.device, h.dtype, deterministic)
        action = self.initial_action(context_actions, deterministic=deterministic)
        init_action = action
        outputs = []
        for k, block in enumerate(self.blocks):
            time = torch.full(
                (h.shape[0], 1),
                k / max(1, self.horizon - 1),
                device=h.device,
                dtype=h.dtype,
            )
            delta = block(torch.cat([action, h, noise, time], dim=-1))
            action = self.action_limit * torch.tanh(action + self.tau * delta)
            outputs.append(action)
        return {
            "actions": torch.stack(outputs, dim=1),
            "init_action": init_action,
            "history": h,
            "noise": noise,
        }

    @property
    def network_evals(self) -> int:
        return self.horizon


class SinkhornActionBridgePolicy(nn.Module):
    """Conditional particle bridge over execution-time action marginals.

    This is the CWG-like probabilistic variant. It samples a cloud of source
    actions, then applies a stack of residual maps:

        z_{k+1} = z_k + tau * f_k(z_k, h_t)

    Each residual block corresponds to one future action time. The returned
    `particles` tensor represents conditional action marginals along the
    bridge path.
    """

    def __init__(
        self,
        context: int,
        horizon: int,
        state_dim: int = 4,
        action_dim: int = 2,
        history_dim: int = 96,
        hidden_dim: int = 192,
        tau: float = 0.35,
        init_type: str = "prev_action",
        init_noise_scale: float = 0.35,
        particles: int = 8,
        action_limit: float = 1.0,
        use_context_actions: bool = True,
    ):
        super().__init__()
        if init_type not in ("prev_action", "zero", "gaussian"):
            raise ValueError(f"Unknown init_type: {init_type}")
        self.horizon = horizon
        self.action_dim = action_dim
        self.tau = tau
        self.init_type = init_type
        self.init_noise_scale = init_noise_scale
        self.particles = particles
        self.action_limit = action_limit
        self.encoder = ActionContextEncoder(
            context, state_dim, action_dim, history_dim, hidden_dim, use_context_actions
        )
        self.blocks = nn.ModuleList(
            [mlp([action_dim + history_dim, hidden_dim, hidden_dim, action_dim]) for _ in range(horizon)]
        )

    def initial_particles(self, context_actions: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        batch_size = context_actions.shape[0]
        if self.init_type == "prev_action":
            base = context_actions[:, -1, None, :].expand(batch_size, self.particles, self.action_dim)
            if deterministic:
                noise = torch.zeros_like(base)
            else:
                noise = self.init_noise_scale * torch.randn_like(base)
            return self.action_limit * torch.tanh(base + noise)
        if self.init_type == "zero":
            base = torch.zeros(batch_size, self.particles, self.action_dim, device=context_actions.device, dtype=context_actions.dtype)
            if deterministic:
                return base
            return self.action_limit * torch.tanh(base + self.init_noise_scale * torch.randn_like(base))
        if deterministic:
            return torch.zeros(batch_size, self.particles, self.action_dim, device=context_actions.device, dtype=context_actions.dtype)
        return self.action_limit * torch.tanh(
            self.init_noise_scale
            * torch.randn(batch_size, self.particles, self.action_dim, device=context_actions.device, dtype=context_actions.dtype)
        )

    def forward(self, batch: dict[str, torch.Tensor], deterministic: bool = False) -> dict[str, torch.Tensor]:
        context_states = batch["context_states"]
        context_actions = batch["context_actions"]
        h = self.encoder(context_states, context_actions)
        z = self.initial_particles(context_actions, deterministic=deterministic)
        init_particles = z
        h_particles = h[:, None, :].expand(-1, self.particles, -1)
        outputs = []
        for block in self.blocks:
            flat = torch.cat([z, h_particles], dim=-1).reshape(z.shape[0] * self.particles, -1)
            delta = block(flat).reshape(z.shape[0], self.particles, self.action_dim)
            z = self.action_limit * torch.tanh(z + self.tau * delta)
            outputs.append(z)
        particles = torch.stack(outputs, dim=2)
        return {
            "actions": particles.mean(dim=1),
            "particles": particles,
            "init_particles": init_particles,
            "init_action": init_particles.mean(dim=1),
            "history": h,
        }

    @property
    def network_evals(self) -> int:
        return self.horizon

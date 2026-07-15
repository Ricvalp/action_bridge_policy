"""Chunk-level critics for ContactBridgeSAC."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


def make_mlp(in_dim: int, out_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    layers = []
    current = int(in_dim)
    for _ in range(max(1, int(depth))):
        layers.append(nn.Linear(current, int(hidden_dim)))
        layers.append(nn.SiLU())
        current = int(hidden_dim)
    layers.append(nn.Linear(current, int(out_dim)))
    return nn.Sequential(*layers)


class ChunkQNetwork(nn.Module):
    """Q(h, A_exec) over flattened histories and executed action chunks."""

    def __init__(
        self,
        obs_history: int,
        obs_dim: int,
        action_history: int,
        action_dim: int,
        n_exec: int,
        hidden_dim: int = 512,
        depth: int = 3,
    ):
        super().__init__()
        self.obs_history = int(obs_history)
        self.obs_dim = int(obs_dim)
        self.action_history = int(action_history)
        self.action_dim = int(action_dim)
        self.n_exec = int(n_exec)
        in_dim = self.obs_history * self.obs_dim + self.action_history * self.action_dim + self.n_exec * self.action_dim
        self.net = make_mlp(in_dim, 1, hidden_dim=int(hidden_dim), depth=int(depth))

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor, exec_actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_hist.flatten(1), act_hist.flatten(1), exec_actions.flatten(1)], dim=-1)
        return self.net(x).squeeze(-1)


class DoubleChunkQ(nn.Module):
    def __init__(
        self,
        obs_history: int,
        obs_dim: int,
        action_history: int,
        action_dim: int,
        n_exec: int,
        hidden_dim: int = 512,
        depth: int = 3,
    ):
        super().__init__()
        kwargs = dict(
            obs_history=obs_history,
            obs_dim=obs_dim,
            action_history=action_history,
            action_dim=action_dim,
            n_exec=n_exec,
            hidden_dim=hidden_dim,
            depth=depth,
        )
        self.q1 = ChunkQNetwork(**kwargs)
        self.q2 = ChunkQNetwork(**kwargs)

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor, exec_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs_hist, act_hist, exec_actions), self.q2(obs_hist, act_hist, exec_actions)

    def min_q(self, obs_hist: torch.Tensor, act_hist: torch.Tensor, exec_actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(obs_hist, act_hist, exec_actions)
        return torch.minimum(q1, q2)

    def make_target(self) -> "DoubleChunkQ":
        target = deepcopy(self)
        for param in target.parameters():
            param.requires_grad_(False)
        return target


@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    tau = float(tau)
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.mul_(1.0 - tau).add_(tau * param.data)

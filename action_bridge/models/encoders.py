"""Encoders and MLP utilities."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Type

import torch
from torch import nn
import torch.nn.functional as F


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    activation: Type[nn.Module] = nn.SiLU,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    dim = input_dim
    for _ in range(max(0, depth - 1)):
        layers.append(nn.Linear(dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation())
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        obs_history: int,
        action_history: int,
        obs_dim: int,
        action_dim: int,
        h_emb_dim: int = 256,
        hidden_dim: int = 256,
        depth: int = 3,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.obs_history = obs_history
        self.action_history = action_history
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        in_dim = obs_history * obs_dim + action_history * action_dim
        self.net = make_mlp(in_dim, h_emb_dim, hidden_dim, depth, layer_norm=layer_norm)

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        flat = torch.cat([obs_hist.reshape(obs_hist.shape[0], -1), act_hist.reshape(act_hist.shape[0], -1)], dim=-1)
        return self.net(flat)


class FutureActionEncoder(nn.Module):
    def __init__(
        self,
        chunk_horizon: int,
        action_dim: int,
        output_dim: int = 256,
        hidden_dim: int = 256,
        depth: int = 2,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.net = make_mlp(chunk_horizon * action_dim, output_dim, hidden_dim, depth, layer_norm=layer_norm)

    def forward(self, future_actions: torch.Tensor) -> torch.Tensor:
        return self.net(future_actions.reshape(future_actions.shape[0], -1))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        if k.ndim == 0:
            k = k[None]
        half = self.dim // 2
        if half <= 0:
            return k.float()[:, None]
        scale = math.log(10000.0) / max(1, half - 1)
        freqs = torch.exp(-scale * torch.arange(half, device=k.device, dtype=torch.float32))
        args = k.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


def timestep_tensor(k: int, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.full((batch_size,), int(k), device=device, dtype=torch.long)

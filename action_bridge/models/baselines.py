"""BC baselines that separate path-KL control from ordinary smoothing."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from action_bridge.models.encoders import HistoryEncoder, SinusoidalTimeEmbedding, make_mlp, timestep_tensor


class DirectChunkBCPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_history: int,
        action_history: int,
        chunk_horizon: int,
        model_config: Dict,
    ):
        super().__init__()
        hidden_dim = int(model_config.get("hidden_dim", 256))
        h_emb_dim = int(model_config.get("h_emb_dim", hidden_dim))
        self.action_dim = action_dim
        self.chunk_horizon = chunk_horizon
        self.history_encoder = HistoryEncoder(obs_history, action_history, obs_dim, action_dim, h_emb_dim, hidden_dim)
        self.head = make_mlp(h_emb_dim, chunk_horizon * action_dim, hidden_dim, depth=int(model_config.get("depth", 3)))

    def encode_history(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(obs_hist, act_hist)

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        h = self.encode_history(obs_hist, act_hist)
        return self.head(h).reshape(obs_hist.shape[0], self.chunk_horizon, self.action_dim)


class AutoregressiveBCPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_history: int,
        action_history: int,
        chunk_horizon: int,
        model_config: Dict,
    ):
        super().__init__()
        hidden_dim = int(model_config.get("hidden_dim", 256))
        h_emb_dim = int(model_config.get("h_emb_dim", hidden_dim))
        time_dim = int(model_config.get("time_emb_dim", 32))
        self.action_dim = action_dim
        self.chunk_horizon = chunk_horizon
        self.history_encoder = HistoryEncoder(obs_history, action_history, obs_dim, action_dim, h_emb_dim, hidden_dim)
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.net = make_mlp(2 * action_dim + h_emb_dim + time_dim, action_dim, hidden_dim, depth=int(model_config.get("depth", 4)))

    def encode_history(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(obs_hist, act_hist)

    def predict_step(self, a_prev: torch.Tensor, a_prevprev: torch.Tensor, h_emb: torch.Tensor, k: int) -> torch.Tensor:
        tk = timestep_tensor(k, a_prev.shape[0], a_prev.device)
        t_emb = self.time(tk).to(dtype=a_prev.dtype)
        return self.net(torch.cat([a_prev, a_prevprev, h_emb, t_emb], dim=-1))

    def generate(self, obs_hist: torch.Tensor, act_hist: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        h = self.encode_history(obs_hist, act_hist)
        a_prevprev = act_hist[:, -2]
        a_prev = act_hist[:, -1]
        actions = []
        for k in range(self.chunk_horizon):
            action = self.predict_step(a_prev, a_prevprev, h, k)
            actions.append(action)
            a_prevprev, a_prev = a_prev, action
        return torch.stack(actions, dim=1)

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        return self.generate(obs_hist, act_hist)


class ReferenceOnlyPolicy(nn.Module):
    def __init__(self, reference, chunk_horizon: int):
        super().__init__()
        self.reference_process = reference
        self.chunk_horizon = chunk_horizon

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(obs_hist.shape[0], 1, device=obs_hist.device, dtype=obs_hist.dtype)
        a_prevprev = act_hist[:, -2]
        a_prev = act_hist[:, -1]
        actions = []
        for k in range(self.chunk_horizon):
            action, _ = self.reference_process(a_prev, a_prevprev, h, k)
            actions.append(action)
            a_prevprev, a_prev = a_prev, action
        return torch.stack(actions, dim=1)

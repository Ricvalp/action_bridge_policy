"""Robot-index tail completion and its frozen quadratic whole-plan prior.

These operations use absolute target commands in Euclidean model coordinates.
The robot-index step ``robot_dt`` is unrelated to the bridge's revision time.
Neither coefficients nor completion can inspect a future expert endpoint.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from action_bridge.models.encoders import HistoryEncoder


COMPLETION_NAMES = ("repeat", "fixed_damped", "learned_dissipative")


def _absolute_targets(semantics: str) -> None:
    if semantics != "absolute_target":
        raise ValueError(
            "damped completion requires continuous Euclidean absolute_target actions; "
            "deltas, torques, quaternions and categorical channels need another codec/completer"
        )


def complete_plan(
    old: torch.Tensor,
    executed: int,
    obs_hist: torch.Tensor | Mapping[str, torch.Tensor],
    act_hist: torch.Tensor,
    mode: int | torch.Tensor,
    learned: LearnedCompletion | None = None,
    *,
    robot_dt: float = 1.0,
    action_semantics: str = "absolute_target",
) -> torch.Tensor:
    """Align ``old[executed:]`` exactly and append the chosen causal tail.

    Mode IDs are 0=repeat, 1=fixed damping (0.8), 2=learned damping.
    A fully executed plan requires an external bootstrap: there is no retained
    tail to initialize here. Mixed mode IDs are supported in one batch.
    """
    _absolute_targets(action_semantics)
    if old.ndim != 3 or old.shape[1] < 2:
        raise ValueError("old must have shape [batch, horizon >= 2, action_dim]")
    batch, horizon, _ = old.shape
    if not isinstance(executed, int) or not 0 <= executed < horizon:
        raise ValueError("executed must satisfy 0 <= executed < horizon; K=H requires bootstrap")
    if robot_dt <= 0:
        raise ValueError("robot_dt must be positive")
    modes = torch.as_tensor(mode, device=old.device)
    if modes.is_floating_point() and not bool((modes == modes.round()).all()):
        raise ValueError("completion mode must contain integer IDs")
    modes = modes.long()
    if modes.ndim == 0:
        modes = modes.expand(batch)
    if modes.shape != (batch,) or bool(((modes < 0) | (modes > 2)).any()):
        raise ValueError("completion mode must be an integer ID in [0, 2], scalar or [batch]")
    if executed == 0:
        return old.clone()
    use_learned = bool((modes == 2).any())
    if use_learned:
        if learned is None:
            raise ValueError("learned_dissipative completion requires a fitted LearnedCompletion")
        if learned.horizon != horizon or learned.action_dim != old.shape[2]:
            raise ValueError("old plan shape does not match the learned completion model")
        if learned.robot_dt != robot_dt:
            raise ValueError("completion robot_dt differs from the learned model")
        attractor, stiffness, damping = learned.coefficients(obs_hist, act_hist)
    # The end of the retained overlap is the end of the old plan. Even K=H-1
    # has a valid previous old command for its velocity boundary.
    q = old[:, -1]
    p = (old[:, -1] - old[:, -2]) / robot_dt
    tail = []
    for index in range(horizon - executed, horizon):
        next_p = 0.8 * p
        if use_learned:
            learned_p = p + robot_dt * (
                -stiffness[:, index] * (q - attractor[:, index]) - damping[:, index] * p
            )
            next_p = torch.where((modes == 2)[:, None], learned_p, next_p)
        next_p = torch.where((modes == 0)[:, None], torch.zeros_like(next_p), next_p)
        q = q + robot_dt * next_p
        p = next_p
        tail.append(q)
    return torch.cat([old[:, executed:], torch.stack(tail, dim=1)], dim=1)


class LearnedCompletion(nn.Module):
    """Small standalone supervised damped-command reference (no controller).

    A replacement encoder takes ``(obs_hist, act_hist)`` and returns [B,hidden_dim].
    The default is the repository's state-history encoder. ``precision_scale``
    is fitted once from *training* contexts by the caller, then saved/frozen with
    the weights. It scales revision rates, not robot-index gains.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_history: int,
        action_history: int,
        horizon: int,
        hidden_dim: int = 64,
        robot_dt: float = 1.0,
        *,
        encoder: nn.Module | None = None,
        action_semantics: str = "absolute_target",
        attractor_limit: float = 4.0,
    ) -> None:
        super().__init__()
        _absolute_targets(action_semantics)
        if min(obs_dim, action_dim, obs_history, hidden_dim) < 1 or action_history < 2 or horizon < 2:
            raise ValueError("positive dimensions, horizon >= 2 and two executed commands are required")
        if robot_dt <= 0 or attractor_limit <= 0:
            raise ValueError("robot_dt and attractor_limit must be positive")
        self.horizon = horizon
        self.action_dim = action_dim
        self.action_history = action_history
        self.robot_dt = float(robot_dt)
        self.action_semantics = action_semantics
        self.attractor_limit = float(attractor_limit)
        self.encoder = encoder if encoder is not None else HistoryEncoder(
            obs_history, action_history, obs_dim, action_dim,
            h_emb_dim=hidden_dim, hidden_dim=hidden_dim, depth=2,
        )
        # A shared small head sees bounded robot-index features, not H independent
        # full-capacity action predictors or a future-action encoder.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3 * action_dim)
        )
        index = torch.linspace(0.0, 1.0, horizon)
        self.register_buffer("index_features", torch.stack([index, torch.sin(torch.pi * index),
                                                           torch.cos(torch.pi * index)], dim=-1))
        self.register_buffer("precision_scale", torch.tensor(1.0))

    def coefficients(
        self, obs_hist: torch.Tensor | Mapping[str, torch.Tensor], act_hist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check_history(act_hist)
        history = self.encoder(obs_hist, act_hist)
        features = torch.cat([
            history[:, None].expand(-1, self.horizon, -1),
            self.index_features[None].expand(history.shape[0], -1, -1),
        ], dim=-1)
        raw_m, raw_k, raw_gamma = self.head(features).chunk(3, dim=-1)
        attractor = self.attractor_limit * raw_m.tanh()
        # Symplectic-Euler scalar stability: 0 < dt*gamma < 2 and
        # dt^2*K < 4 - 2*dt*gamma. These conservative bounds satisfy it.
        stiffness = (0.01 + 0.49 * raw_k.sigmoid()) / self.robot_dt**2
        damping = (0.15 + 0.80 * raw_gamma.sigmoid()) / self.robot_dt
        return attractor, stiffness, damping

    def _check_history(self, act_hist: torch.Tensor) -> None:
        if act_hist.ndim != 3 or act_hist.shape[1:] != (self.action_history, self.action_dim):
            raise ValueError("executed action history does not match configured history/action dimensions")

    def _check_batch(self, batch: Mapping[str, torch.Tensor]) -> None:
        actions = batch["future_actions"]
        if actions.ndim != 3 or actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("future_actions does not match configured horizon/action dimensions")
        mask = batch.get("valid_mask")
        if mask is None or mask.shape != actions.shape[:2] or not bool((mask == 1).all()):
            raise ValueError("the whole-plan prior requires an explicit fully valid horizon mask")
        self._check_history(batch["act_hist"])

    def residuals(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Train-label innovations e_k, used for supervision and train-only S."""
        self._check_batch(batch)
        target = batch["future_actions"]
        m, k, gamma = self.coefficients(batch["obs_hist"], batch["act_hist"])
        history = batch["act_hist"]
        q = torch.cat([history[:, -1:], target[:, :-1]], dim=1)
        previous = torch.cat([history[:, -2:], target[:, :-2]], dim=1)
        p = (q - previous) / self.robot_dt
        next_p = (target - q) / self.robot_dt
        return next_p - p + self.robot_dt * (k * (q - m) + gamma * p)

    def loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Mean squared next-target error under teacher-observed transitions."""
        return (self.robot_dt * self.residuals(batch)).square().mean()

    def complete(self, old, executed, obs_hist, act_hist, mode):
        return complete_plan(old, executed, obs_hist, act_hist, mode, self,
                             robot_dt=self.robot_dt, action_semantics=self.action_semantics)

    @torch.no_grad()
    def diagnostics(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._check_batch(batch)
        m, k, gamma = self.coefficients(batch["obs_hist"], batch["act_hist"])
        q = batch["act_hist"][:, -1]
        p = (q - batch["act_hist"][:, -2]) / self.robot_dt
        sequence, velocities, accelerations = [], [], []
        for index in range(self.horizon):
            next_p = p + self.robot_dt * (-k[:, index] * (q - m[:, index]) - gamma[:, index] * p)
            accelerations.append((next_p - p) / self.robot_dt)
            p = next_p
            q = q + self.robot_dt * p
            sequence.append(q)
            velocities.append(p)
        return {
            "reference_mse": self.loss(batch),
            "rollout_mse": (torch.stack(sequence, 1) - batch["future_actions"]).square().mean(),
            "reference_rollout_speed": torch.stack(velocities, 1).square().sum(-1).mean().sqrt(),
            "reference_rollout_acceleration": torch.stack(accelerations, 1).square().sum(-1).mean().sqrt(),
        }

    def prior(
        self,
        obs_hist: torch.Tensor | Mapping[str, torch.Tensor],
        act_hist: torch.Tensor,
        innovation_variance: torch.Tensor,
        ridge: float = 0.05,
        max_rate: float = 4.0,
        mobility_smoothing: float = 0.0,
        *,
        min_rate: float = 1e-3,
    ) -> dict[str, torch.Tensor]:
        """Build e=B A-d, its Gaussian prior, and explicit spectral stabilization.

        ``innovation_variance`` is a positive train-residual variance [d] (or
        [H,d]); the caller estimates/floors it on the training split. The raw
        quadratic mean is retained when precision rates are scaled/clamped.
        A mobility spectral bound keeps eigenvalues of D P below max_rate.
        Diagnostics expose every stabilization rather than silently changing S.
        """
        if not 0 < min_rate < max_rate or ridge <= 0 or mobility_smoothing < 0:
            raise ValueError("require 0 < min_rate < max_rate, ridge > 0, mobility_smoothing >= 0")
        if not bool(torch.isfinite(self.precision_scale)) or not bool(self.precision_scale > 0):
            raise ValueError("precision_scale must be a finite positive train-fitted constant")
        m, stiffness, damping = self.coefficients(obs_hist, act_hist)
        dtype, device = m.dtype, m.device
        # Build/solve the potentially ill-conditioned raw precision in float64;
        # the stabilized precision has a bounded condition number in model dtype.
        m, stiffness, damping = m.double(), stiffness.double(), damping.double()
        history = act_hist.double()
        batch, horizon, dim = m.shape
        n = horizon * dim
        variance = torch.as_tensor(innovation_variance, device=device, dtype=torch.float64)
        if variance.shape == (dim,):
            variance = variance.expand(horizon, dim)
        if variance.shape != (horizon, dim) or not bool(torch.isfinite(variance).all()) or not bool((variance > 0).all()):
            raise ValueError("innovation_variance must be finite positive [action_dim] or [horizon,action_dim]")
        dt = self.robot_dt
        middle = dt * stiffness - 2.0 / dt + damping
        previous = 1.0 / dt - damping
        identity = torch.eye(n, device=device, dtype=torch.float64)
        matrix = identity[None].expand(batch, -1, -1) / dt
        matrix = matrix + torch.diag_embed(middle[:, 1:].flatten(1), offset=-dim)
        if horizon > 2:
            matrix = matrix + torch.diag_embed(previous[:, 2:].flatten(1), offset=-2 * dim)
        boundary = dt * stiffness * m
        boundary = boundary.clone()
        boundary[:, 0] -= middle[:, 0] * history[:, -1] + previous[:, 0] * history[:, -2]
        boundary[:, 1] -= previous[:, 1] * history[:, -1]
        inv_variance = variance.flatten().reciprocal()
        raw_precision = matrix.transpose(-1, -2) @ (inv_variance[None, :, None] * matrix) + ridge * identity
        anchor = history[:, -1:].expand(-1, horizon, -1).reshape(batch, n)
        rhs = (matrix.transpose(-1, -2) @ (inv_variance[None] * boundary.flatten(1))[..., None]).squeeze(-1)
        rhs = rhs + ridge * anchor
        mean = torch.cholesky_solve(rhs[..., None], torch.linalg.cholesky(raw_precision)).squeeze(-1)
        mobility = identity.clone()
        if mobility_smoothing and horizon > 2:
            second = torch.zeros(horizon - 2, horizon, device=device, dtype=torch.float64)
            index = torch.arange(horizon - 2, device=device)
            second[index, index] = 1
            second[index, index + 1] = -2
            second[index, index + 2] = 1
            mobility = identity + mobility_smoothing * torch.kron(
                second.T @ second, torch.eye(dim, device=device, dtype=torch.float64)
            )
        mobility_max = torch.linalg.eigvalsh(mobility)[-1]
        eigenvalues, eigenvectors = torch.linalg.eigh(raw_precision)
        scaled = eigenvalues * self.precision_scale.double()
        stabilized = scaled.clamp(min=min_rate / mobility_max, max=max_rate / mobility_max)
        precision = (eigenvectors * stabilized[:, None]) @ eigenvectors.transpose(-1, -2)
        precision = (precision + precision.transpose(-1, -2)) * 0.5
        return {
            "precision": precision.to(dtype), "mean": mean.to(dtype), "mobility": mobility.to(dtype),
            "raw_rate_min": eigenvalues[:, 0].to(dtype), "raw_rate_max": eigenvalues[:, -1].to(dtype),
            "rate_min": stabilized[:, 0].to(dtype), "rate_max": stabilized[:, -1].to(dtype),
            "precision_scale": self.precision_scale.detach().clone(),
            "stabilized_fraction": (scaled != stabilized).to(dtype).mean(-1),
        }

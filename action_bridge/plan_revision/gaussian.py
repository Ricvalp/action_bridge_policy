"""Exact affine Gaussian references in *revision* time, independent of a task.

``precision`` and ``mean`` describe a frozen whole-plan quadratic potential.
Kinetic states are packed as ``[position, revision_velocity]``; their white
noise and instantaneous controls act on velocity only. Reverse integration
uses increasing reverse time and does **not** flip velocity parity.

For identity mobility we diagonalize the precision once. All training bridge
algebra then uses scalar OU modes or independent two-dimensional kinetic
modes. Small Van Loan exponentials give exact transition covariances, including
at short durations where subtracting stationary covariances loses precision.
Nonidentity mobility uses the same identities with dense matrices. The formulas
are Gaussian conditioning for controllable linear SDEs; see Chen, Georgiou and
Pavon, https://arxiv.org/abs/1502.01265. There is no inverse of instantaneous
``G G.T`` (which is singular for the kinetic reference).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _mv(matrix: Tensor, vector: Tensor) -> Tensor:
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def _symmetric(matrix: Tensor) -> Tensor:
    return (matrix + matrix.transpose(-1, -2)) * 0.5


def _solve_cov(covariance: Tensor, rhs: Tensor) -> Tensor:
    """SPD covariance solve; no inverse or eigenvalue clipping."""
    return torch.cholesky_solve(rhs, torch.linalg.cholesky(_symmetric(covariance)))


def _exprel(value: Tensor) -> Tensor:
    """expm1(x)/x, with the correct continuous limit at zero."""
    small = value.abs() < 1e-5
    denominator = torch.where(small, torch.ones_like(value), value)
    series = 1 + value * (0.5 + value * (1 / 6 + value / 24))
    return torch.where(small, series, torch.expm1(value) / denominator)


class GaussianReference:
    """A batch of context-frozen OU, kinetic, or Brownian references.

    Args:
        precision: Positive definite ``[B,n,n]`` plan precision. Ignored by
            Brownian dynamics (zero precision is allowed for that special case).
        mean: ``[B,n]`` plan center, in the model's action coordinates.
        mobility: Fixed SPD ``[n,n]`` or contextual ``[B,n,n]`` matrix; default I.

    Public state/control tensors are ``[B,state_dim]`` / ``[B,n]``. Durations
    and physical bridge times can be scalars or ``[B]`` tensors. Moment methods
    return full-state covariances. Prefer :meth:`sample_bridge` for training:
    it avoids materializing and factoring a full kinetic covariance. Algebra
    uses float64; outputs preserve the input state dtype. Coefficients should
    be frozen before constructing this object; this is not a learned module.
    """

    def __init__(
        self,
        precision: Tensor,
        mean: Tensor,
        kind: str = "ou",
        temperature: float = 0.05,
        gamma: float = 2.0,
        mobility: Tensor | None = None,
    ) -> None:
        if kind not in {"ou", "kinetic", "brownian"}:
            raise ValueError(f"unknown Gaussian reference kind: {kind}")
        if mean.ndim != 2 or precision.shape != (*mean.shape, mean.shape[-1]):
            raise ValueError("expected precision [B,n,n] and mean [B,n]")
        if not mean.is_floating_point() or not precision.is_floating_point():
            raise ValueError("reference coefficients must be floating-point tensors")
        if precision.device != mean.device:
            raise ValueError("precision and mean must be on the same device")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        self.kind = kind
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.batch_size, self.n = mean.shape
        self.state_dim = self.n * (2 if kind == "kinetic" else 1)
        self.precision = precision.to(torch.float64)
        self.mean = mean.to(torch.float64)
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.precision).all():
            raise ValueError("reference coefficients must be finite")
        precision_tolerance = 10 * torch.finfo(precision.dtype).eps * self.precision.abs().amax().clamp_min(1)
        if (self.precision - self.precision.transpose(-1, -2)).abs().amax() > precision_tolerance:
            raise ValueError("precision must be symmetric")
        self.precision = _symmetric(self.precision)
        eye = torch.eye(self.n, device=mean.device, dtype=torch.float64)
        self.mobility = eye.expand(self.batch_size, -1, -1)
        if mobility is not None:
            if mobility.shape not in {(self.n, self.n), precision.shape}:
                raise ValueError("mobility must have shape [n,n] or [B,n,n]")
            self.mobility = mobility.to(device=mean.device, dtype=torch.float64).expand_as(precision)
            mobility_tolerance = 10 * torch.finfo(mobility.dtype).eps * self.mobility.abs().amax().clamp_min(1)
            if not torch.isfinite(self.mobility).all() or (
                self.mobility - self.mobility.transpose(-1, -2)
            ).abs().amax() > mobility_tolerance:
                raise ValueError("mobility must be finite and symmetric")
            self.mobility = _symmetric(self.mobility)
        try:
            mobility_root = torch.linalg.cholesky(self.mobility)
            if kind != "brownian":
                torch.linalg.cholesky(self.precision)
        except torch.linalg.LinAlgError as error:
            raise ValueError("precision and mobility must be positive definite") from error
        self.center = (
            torch.cat((self.mean, torch.zeros_like(self.mean)), dim=-1)
            if kind == "kinetic"
            else self.mean
        )
        self._modal = torch.equal(self.mobility, eye.expand_as(self.mobility))
        self._sigma = math.sqrt(2 * temperature * (gamma if kind == "kinetic" else 1))
        self._cache: dict[tuple[float, bool], tuple[Tensor, Tensor, Tensor]] = {}
        if self._modal:
            if kind == "brownian":
                self._rates = torch.zeros_like(self.mean)
                self._basis = eye.expand_as(self.precision)
            else:
                self._rates, self._basis = torch.linalg.eigh(self.precision)
        else:
            zero = torch.zeros_like(self.precision)
            if kind == "kinetic":
                self._F = torch.cat(
                    (
                        torch.cat((zero, eye.expand_as(zero)), dim=-1),
                        torch.cat((-self.precision, -gamma * self.mobility), dim=-1),
                    ),
                    dim=-2,
                )
                self._G = torch.cat((zero, self._sigma * mobility_root), dim=-2)
            else:
                self._F = zero if kind == "brownian" else -self.mobility @ self.precision
                self._G = self._sigma * mobility_root

    def _state(self, state: Tensor) -> Tensor:
        if state.shape != (self.batch_size, self.state_dim):
            raise ValueError(f"state must have shape {(self.batch_size, self.state_dim)}")
        if state.device != self.mean.device:
            raise ValueError("state and reference must be on the same device")
        return state.to(torch.float64)

    def _time(self, time: float | Tensor, *, bridge: bool = False) -> Tensor:
        result = torch.as_tensor(time, device=self.mean.device, dtype=torch.float64)
        if result.ndim == 0:
            result = result.expand(self.batch_size)
        if result.shape != (self.batch_size,):
            raise ValueError("time must be a scalar or a [B] tensor")
        if not torch.isfinite(result).all() or (result < 0).any():
            raise ValueError("time must be finite and nonnegative")
        if bridge and (result > 1).any():
            raise ValueError("bridge time must lie in [0,1]")
        return result

    def _to_modes(self, state: Tensor, *, centered: bool = True) -> Tensor:
        state = state - self.center if centered else state
        position = _mv(self._basis.transpose(-1, -2), state[..., : self.n])
        if self.kind != "kinetic":
            return position.unsqueeze(-1)
        velocity = _mv(self._basis.transpose(-1, -2), state[..., self.n :])
        return torch.stack((position, velocity), dim=-1)

    def _from_modes(self, modes: Tensor, *, centered: bool = True) -> Tensor:
        position = _mv(self._basis, modes[..., 0])
        result = position
        if self.kind == "kinetic":
            result = torch.cat((position, _mv(self._basis, modes[..., 1])), dim=-1)
        return result + self.center if centered else result

    def _full_covariance(self, modal_cov: Tensor) -> Tensor:
        def rotate(values: Tensor) -> Tensor:
            return (self._basis * values.unsqueeze(-2)) @ self._basis.transpose(-1, -2)

        if self.kind != "kinetic":
            return rotate(modal_cov[..., 0, 0])
        return torch.cat(
            (
                torch.cat((rotate(modal_cov[..., 0, 0]), rotate(modal_cov[..., 0, 1])), dim=-1),
                torch.cat((rotate(modal_cov[..., 1, 0]), rotate(modal_cov[..., 1, 1])), dim=-1),
            ),
            dim=-2,
        )

    def _moments(self, dt: float | Tensor, reverse: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """Return transition Phi, covariance Q, and held-control gain J."""
        key = (float(dt), reverse) if isinstance(dt, (float, int)) else None
        if key is not None and key in self._cache:
            return self._cache[key]
        duration = self._time(dt)
        if self._modal and self.kind != "kinetic":
            rate = self._rates if reverse else -self._rates
            scaled = rate * duration[:, None]
            phi = scaled.exp()[..., None, None]
            variance = self._sigma**2 * duration[:, None] * _exprel(2 * scaled)
            covariance = variance[..., None, None]
            gain = (self._sigma * duration[:, None] * _exprel(scaled))[..., None]
        else:
            if self._modal:
                shape = (*self._rates.shape, 2, 2)
                dynamics = self.mean.new_zeros(shape)
                dynamics[..., 0, 1] = 1
                dynamics[..., 1, 0] = -self._rates
                dynamics[..., 1, 1] = -self.gamma
                noise = self.mean.new_zeros((*self._rates.shape, 2, 1))
                noise[..., 1, 0] = self._sigma
                duration = duration[:, None, None, None]
            else:
                dynamics, noise = self._F, self._G
                duration = duration[:, None, None]
            if reverse:
                dynamics = -dynamics
            # exp([[F,GG'],[0,-F']]*dt) = [[Phi,Q Phi^-T],[0,Phi^-T]].
            d = dynamics.shape[-1]
            zero = torch.zeros_like(dynamics)
            block = torch.cat(
                (
                    torch.cat((dynamics, noise @ noise.transpose(-1, -2)), dim=-1),
                    torch.cat((zero, -dynamics.transpose(-1, -2)), dim=-1),
                ),
                dim=-2,
            )
            exponential = torch.matrix_exp(block * duration)
            phi = exponential[..., :d, :d]
            covariance = _symmetric(exponential[..., :d, d:] @ phi.transpose(-1, -2))
            identity = torch.eye(d, device=phi.device, dtype=phi.dtype)
            if self.kind == "brownian":
                gain = duration * noise
            else:
                gain = torch.linalg.solve(dynamics, (phi - identity) @ noise)
            if self._modal:
                gain = gain.squeeze(-1)
        if not torch.isfinite(phi).all() or not torch.isfinite(covariance).all():
            raise ValueError("nonfinite Gaussian transition: check reference rates and durations")
        result = phi, covariance, gain
        # The typical sampler has 32 fixed durations. Bound memory when callers
        # query many different times; per-example random training times are not cached.
        if key is not None and len(self._cache) < 128:
            self._cache[key] = result
        return result

    def drift(self, x: Tensor) -> Tensor:
        state = self._state(x)
        if self.kind == "brownian":
            result = torch.zeros_like(state)
        elif self.kind == "ou":
            result = -_mv(self.mobility, _mv(self.precision, state - self.mean))
        else:
            position, velocity = state[..., : self.n], state[..., self.n :]
            force = -_mv(self.precision, position - self.mean) - self.gamma * _mv(
                self.mobility, velocity
            )
            result = torch.cat((velocity, force), dim=-1)
        return result.to(x.dtype)

    def noise_control(self, u: Tensor) -> Tensor:
        """Instantaneous ``G u``: kinetic positions receive no direct control."""
        if u.shape != (self.batch_size, self.n):
            raise ValueError(f"control must have shape {(self.batch_size, self.n)}")
        if self._modal:
            result = self._sigma * u
            if self.kind == "kinetic":
                result = torch.cat((torch.zeros_like(result), result), dim=-1)
        else:
            result = _mv(self._G, u.to(torch.float64)).to(u.dtype)
        return result

    def transition(
        self, x: Tensor, dt: float | Tensor, reverse: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Exact uncontrolled mean/covariance over a nonnegative duration."""
        state = self._state(x)
        phi, covariance, _ = self._moments(dt, reverse)
        if self._modal:
            mean = self._from_modes(_mv(phi, self._to_modes(state)))
            covariance = self._full_covariance(covariance)
        else:
            mean = self.center + _mv(phi, state - self.center)
        return mean.to(x.dtype), covariance.to(x.dtype)

    def _bridge_moments(
        self, x0: Tensor, x1: Tensor, t: float | Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        time = self._time(t, bridge=True)
        boundary = (time == 0) | (time == 1)
        safe_time = torch.where(boundary, 0.5, time)
        phi0, q0, _ = self._moments(safe_time)
        phir, qr, _ = self._moments(1 - safe_time)
        phi1, q1, _ = self._moments(1.0)
        initial, final = self._state(x0), self._state(x1)
        if self._modal:
            initial, final = self._to_modes(initial), self._to_modes(final)
        else:
            initial, final = initial - self.center, final - self.center
        cross = q0 @ phir.transpose(-1, -2)
        gain = _solve_cov(q1, cross.transpose(-1, -2)).transpose(-1, -2)
        mean = _mv(phi0, initial) + _mv(gain, final - _mv(phi1, initial))
        # Joseph form of the same conditional covariance avoids cancellation
        # of Q0 - C Q1^-1 C.T near the terminal endpoint (especially kinetic
        # position variance, which shrinks cubically with remaining time).
        identity = torch.eye(q0.shape[-1], device=q0.device, dtype=q0.dtype)
        residual = identity - gain @ phir
        covariance = _symmetric(
            residual @ q0 @ residual.transpose(-1, -2)
            + gain @ qr @ gain.transpose(-1, -2)
        )
        shape = (self.batch_size,) + (1,) * (mean.ndim - 1)
        mean = torch.where((time == 0).view(shape), initial, mean)
        mean = torch.where((time == 1).view(shape), final, mean)
        covariance = torch.where(boundary.view(*shape, 1), 0, covariance)
        return mean, covariance, boundary

    def bridge(self, x0: Tensor, x1: Tensor, t: float | Tensor) -> tuple[Tensor, Tensor]:
        """Exact full-state bridge moments, including deterministic endpoints."""
        mean, covariance, _ = self._bridge_moments(x0, x1, t)
        if self._modal:
            mean = self._from_modes(mean)
            covariance = self._full_covariance(covariance)
        else:
            mean = mean + self.center
        return mean.to(x0.dtype), covariance.to(x0.dtype)

    def sample_bridge(
        self,
        x0: Tensor,
        x1: Tensor,
        t: float | Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw an exact conditional bridge state, without dense modal factoring."""
        mean, covariance, boundary = self._bridge_moments(x0, x1, t)
        d = covariance.shape[-1]
        shape = (self.batch_size,) + (1,) * (covariance.ndim - 1)
        identity = torch.eye(d, device=mean.device, dtype=mean.dtype)
        safe_covariance = torch.where(boundary.view(shape), identity, covariance)
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        perturbation = _mv(torch.linalg.cholesky(safe_covariance), noise)
        perturbation = torch.where(boundary.view(shape[:-1]), 0, perturbation)
        state = mean + perturbation
        state = self._from_modes(state) if self._modal else state + self.center
        return state.to(x0.dtype)

    def controls(
        self, x: Tensor, x0: Tensor, x1: Tensor, t: float | Tensor
    ) -> tuple[Tensor, Tensor]:
        """Whitened forward/reverse endpoint scores ``G.T s_plus/minus``.

        ``t`` is physical forward time for **both** scores, strictly inside
        (0,1). Reverse drift in increasing reverse time is ``-drift(x)+G u``.
        """
        time = self._time(t, bridge=True)
        if ((time == 0) | (time == 1)).any():
            raise ValueError("score targets require bridge times strictly inside (0,1)")
        state, initial, final = self._state(x), self._state(x0), self._state(x1)
        if self._modal:
            state, initial, final = map(self._to_modes, (state, initial, final))
        else:
            state, initial, final = (value - self.center for value in (state, initial, final))
        phi0, q0, _ = self._moments(time)
        phir, qr, _ = self._moments(1 - time)
        score_plus = _mv(
            phir.transpose(-1, -2),
            _solve_cov(qr, (final - _mv(phir, state)).unsqueeze(-1)).squeeze(-1),
        )
        score_minus = -_solve_cov(q0, (state - _mv(phi0, initial)).unsqueeze(-1)).squeeze(-1)
        if self._modal:
            channel = 1 if self.kind == "kinetic" else 0
            plus = self._sigma * _mv(self._basis, score_plus[..., channel])
            minus = self._sigma * _mv(self._basis, score_minus[..., channel])
        else:
            plus = _mv(self._G.transpose(-1, -2), score_plus)
            minus = _mv(self._G.transpose(-1, -2), score_minus)
        return plus.to(x.dtype), minus.to(x.dtype)

    def step(
        self,
        x: Tensor,
        u: Tensor,
        dt: float | Tensor,
        reverse: bool = False,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Exact linear-reference step with a held neural noise-channel control.

        Integrated kinetic noise is correlated in position and velocity, even
        though instantaneous noise/control acts on velocity only. This is a
        stochastic step, not a drift-only approximation or bridge endpoint clamp.
        """
        state = self._state(x)
        if u.shape != (self.batch_size, self.n):
            raise ValueError(f"control must have shape {(self.batch_size, self.n)}")
        phi, covariance, gain = self._moments(dt, reverse)
        zero_duration = self._time(dt) == 0
        if self._modal:
            control = _mv(self._basis.transpose(-1, -2), u.to(torch.float64))
            mean = _mv(phi, self._to_modes(state)) + gain * control.unsqueeze(-1)
        else:
            mean = _mv(phi, state - self.center) + _mv(gain, u.to(torch.float64))
        shape = (self.batch_size,) + (1,) * (covariance.ndim - 1)
        identity = torch.eye(covariance.shape[-1], device=state.device, dtype=state.dtype)
        safe_covariance = torch.where(zero_duration.view(shape), identity, covariance)
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        perturbation = _mv(torch.linalg.cholesky(safe_covariance), noise)
        perturbation = torch.where(zero_duration.view(shape[:-1]), 0, perturbation)
        result = mean + perturbation
        result = self._from_modes(result) if self._modal else result + self.center
        return result.to(x.dtype)

    def augment(self, actions_flat: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Draw fresh independent endpoint velocity N(0,T I) for kinetic SB."""
        if actions_flat.shape != (self.batch_size, self.n):
            raise ValueError(f"actions must have shape {(self.batch_size, self.n)}")
        if self.kind != "kinetic":
            return actions_flat
        velocity = torch.randn(
            actions_flat.shape,
            device=actions_flat.device,
            dtype=actions_flat.dtype,
            generator=generator,
        ) * math.sqrt(self.temperature)
        return torch.cat((actions_flat, velocity), dim=-1)

    def positions(self, state: Tensor) -> Tensor:
        self._state(state)
        return state[..., : self.n]

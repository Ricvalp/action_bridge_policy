"""Dataset-independent DDIM, flow matching and directional bridge policies.

All actions here are encoded model coordinates. The caller owns normalization,
source perturbation, endpoint dequantization, and the stateful execution cache.
An injected observation encoder has signature ``encoder(obs_hist, act_hist)``
and returns ``[B, history_dim]``; ``obs_hist`` may therefore be a mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn
import torch.nn.functional as F

from action_bridge.models.diffusion_policy import ConditionalUnet1D, DiffusionPolicy
from action_bridge.models.encoders import HistoryEncoder, SinusoidalTimeEmbedding


def validate_full_horizon(batch, horizon: int, action_dim: int) -> torch.Tensor:
    """Partial Gaussian endpoints are not implemented; never silently mask them."""
    actions = batch["future_actions"]
    if actions.ndim != 3 or actions.shape[1:] != (horizon, action_dim):
        raise ValueError(f"future_actions must have shape [B,{horizon},{action_dim}]")
    mask = batch.get("valid_mask")
    if mask is None or mask.shape != actions.shape[:2]:
        raise ValueError("valid_mask [B,H] is required by the chunk batch contract")
    if not bool(mask.all()):
        raise ValueError("plan revision currently requires fully valid endpoints")
    return actions


def get_completion_ids(batch) -> torch.Tensor:
    mode = batch.get("completion_id")
    if mode is None:
        context = batch.get("context", {})
        mode = context.get("completion_id") if isinstance(context, Mapping) else None
    if mode is None:
        raise ValueError("revision batches require completion_id in [0,2]")
    return mode


def _normal_noise(tensor, generator):
    return torch.randn(tensor.shape, device=tensor.device, dtype=tensor.dtype, generator=generator)


def _history_encoder(config, encoder=None):
    if encoder is not None:
        return encoder
    return HistoryEncoder(
        int(config["obs_history"]), int(config["action_history"]),
        int(config["obs_dim"]), int(config["action_dim"]),
        int(config.get("history_dim", 256)), int(config.get("hidden_dim", 256)),
    )


class FieldNet(nn.Module):
    """Whole-plan vector field or noise-channel control, never endpoint-conditioned.

    Flattened kinetic states are packed as ``[all Y coordinates, all V coordinates]``.
    The temporal backbone sees concatenated Y/V channels at each robot index.
    """

    def __init__(self, config: Mapping, *, kinetic=False, encoder=None):
        super().__init__()
        self.horizon = int(config["horizon"])
        self.action_dim = int(config["action_dim"])
        self.n = self.horizon * self.action_dim
        self.kinetic = kinetic
        self.state_dim = self.n * (2 if kinetic else 1)
        history_dim = int(config.get("history_dim", 256))
        time_dim = int(config.get("time_dim", 64))
        mode_dim = int(config.get("mode_dim", 16))
        channels = tuple(int(x) for x in config.get("channels", (80, 160, 320)))
        if not channels or any(x < 8 or x % 8 for x in channels):
            raise ValueError("channels must be nonempty positive multiples of eight")
        self.history_encoder = _history_encoder(config, encoder)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, 4 * time_dim),
            nn.SiLU(), nn.Linear(4 * time_dim, time_dim),
        )
        self.mode_embedding = nn.Embedding(3, mode_dim)
        self.unet = ConditionalUnet1D(
            self.action_dim * (2 if kinetic else 1), channels,
            history_dim + time_dim + mode_dim + 1, output_dim=self.action_dim,
        )

    def conditioning(self, obs_hist, act_hist, mode, has_previous_plan=None):
        history = self.history_encoder(obs_hist, act_hist)
        mode = torch.as_tensor(mode, device=history.device, dtype=torch.long).expand(history.shape[0])
        if bool(((mode < 0) | (mode > 2)).any()):
            raise ValueError("completion_id must be repeat=0, fixed_damped=1 or learned_dissipative=2")
        available = torch.as_tensor(
            True if has_previous_plan is None else has_previous_plan,
            device=history.device, dtype=history.dtype,
        ).expand(history.shape[0])
        if not bool(((available == 0) | (available == 1)).all()):
            raise ValueError("has_previous_plan must be a boolean scalar or [batch]")
        return torch.cat((history, self.mode_embedding(mode), available[:, None]), dim=-1)

    def conditioned(self, state, tau, conditioning):
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"state must have shape [B,{self.state_dim}]")
        if self.kinetic:
            position, velocity = state.split(self.n, dim=-1)
            chunk = torch.cat((position.reshape(-1, self.horizon, self.action_dim),
                               velocity.reshape(-1, self.horizon, self.action_dim)), dim=-1)
        else:
            chunk = state.reshape(-1, self.horizon, self.action_dim)
        tau = torch.as_tensor(tau, device=state.device, dtype=state.dtype).expand(state.shape[0])
        # Match the denoiser's useful frequency range while retaining tau in [0,1].
        condition = torch.cat((conditioning, self.time_embedding(100 * tau)), dim=-1)
        return self.unet(chunk, condition).reshape(-1, self.n)

    def forward(self, state, tau, obs_hist, act_hist, mode, has_previous_plan=None):
        return self.conditioned(state, tau, self.conditioning(obs_hist, act_hist, mode, has_previous_plan))


class DDIMChunkPolicy(nn.Module):
    """The existing diffusion policy behind the shared chunk-generator boundary."""

    def __init__(self, config: Mapping, *, encoder=None):
        super().__init__()
        self.config = dict(config)
        self.horizon = int(config["horizon"])
        self.action_dim = int(config["action_dim"])
        self.policy = DiffusionPolicy(
            int(config["obs_dim"]), self.action_dim, int(config["obs_history"]),
            int(config["action_history"]), self.horizon,
            {
                "unet_channels": config.get("channels", (80, 160, 320)),
                "h_emb_dim": config.get("history_dim", 256),
                "hidden_dim": config.get("hidden_dim", 256),
                "time_emb_dim": config.get("time_dim", 64),
                "num_train_timesteps": config.get("num_train_timesteps", 100),
                "num_inference_steps": config.get("num_inference_steps", 32),
                "beta_schedule": config.get("beta_schedule", "squaredcos_cap_v2"),
                "timestep_spacing": config.get("timestep_spacing", "leading"),
            }, history_encoder=encoder,
        )

    def scheduler_metadata(self):
        import diffusers
        return {"config": dict(self.policy.noise_scheduler.config), "diffusers_version": diffusers.__version__}

    def loss(self, batch, *, generator=None):
        actions = validate_full_horizon(batch, self.horizon, self.action_dim)
        noise = _normal_noise(actions, generator)
        timesteps = torch.randint(
            self.policy.num_train_timesteps, (len(actions),), device=actions.device, generator=generator,
        )
        noisy = self.policy.noise_scheduler.add_noise(actions, noise, timesteps)
        history = self.policy.encode_history(batch["obs_hist"], batch["act_hist"])
        prediction = self.policy.predict_noise(noisy, timesteps, history)
        loss = F.mse_loss(prediction, noise)
        return {"loss": loss, "noise_mse": loss.detach()}

    @torch.no_grad()
    def sample(self, obs_hist, act_hist, mode=None, *, source_actions=None, reference=None,
               generator=None, has_previous_plan=None):
        del mode, source_actions, reference, has_previous_plan
        actions = self.policy.generate(obs_hist, act_hist, generator=generator)
        return actions, {"nfe": self.policy.num_inference_steps}


class FlowMatchingPolicy(nn.Module):
    def __init__(self, config: Mapping, *, encoder=None):
        super().__init__()
        self.config = dict(config)
        self.horizon = int(config["horizon"])
        self.action_dim = int(config["action_dim"])
        self.field = FieldNet(config, encoder=encoder)
        self.num_inference_steps = int(config.get("num_inference_steps", 32))
        if self.num_inference_steps < 2 or self.num_inference_steps % 2:
            raise ValueError("midpoint FM requires a positive even inference NFE budget")

    def loss(self, batch, *, generator=None):
        target = validate_full_horizon(batch, self.horizon, self.action_dim)
        source = batch["source_actions"]
        if source.shape != target.shape:
            raise ValueError("source_actions must match future_actions shape")
        tau = torch.rand(len(target), device=target.device, dtype=target.dtype, generator=generator)
        state = torch.lerp(source, target, tau[:, None, None])
        prediction = self.field(state.flatten(1), tau, batch["obs_hist"], batch["act_hist"],
                                get_completion_ids(batch), batch.get("has_previous_plan"))
        loss = F.mse_loss(prediction, (target - source).flatten(1))
        return {"loss": loss, "velocity_mse": loss.detach()}

    @torch.no_grad()
    def sample(self, obs_hist, act_hist, mode=None, *, source_actions=None, reference=None,
               generator=None, has_previous_plan=None):
        del reference, generator
        if source_actions is None or source_actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("FM requires an aligned/completed source_actions chunk")
        conditioning = self.field.conditioning(obs_hist, act_hist, mode, has_previous_plan)
        state = source_actions.flatten(1)
        steps = self.num_inference_steps // 2
        dt = 1.0 / steps
        for index in range(steps):
            tau = index * dt
            midpoint = state + (dt / 2) * self.field.conditioned(state, tau, conditioning)
            state = state + dt * self.field.conditioned(midpoint, tau + dt / 2, conditioning)
        return state.reshape_as(source_actions), {"nfe": 2 * steps}


class BridgePolicy(nn.Module):
    """Separate forward/reverse controls, sampled with stochastic exact-reference steps.

    Training samples tau in [cutoff, 1-cutoff]. Positive directional time weights
    are distance-to-endpoint (OU) or its cube (kinetic), compensating the stronger
    kinetic endpoint score scaling. These are weighted *control* regressions,
    not interpolation or an additional control-energy regularizer.
    """

    def __init__(self, config: Mapping, *, encoder=None):
        super().__init__()
        import copy
        self.config = dict(config)
        self.horizon = int(config["horizon"])
        self.action_dim = int(config["action_dim"])
        self.kinetic = config["method"] == "sb_kinetic"
        self.forward_field = FieldNet(config, kinetic=self.kinetic, encoder=encoder)
        self.reverse_field = FieldNet(config, kinetic=self.kinetic, encoder=copy.deepcopy(encoder))
        self.num_inference_steps = int(config.get("num_inference_steps", 32))
        self.time_cutoff = float(config.get("time_cutoff", .01))
        if not 0 < self.time_cutoff < .5:
            raise ValueError("time_cutoff must lie strictly between zero and one half")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")

    def loss(self, batch, reference, *, direction="forward", x0=None, x1=None, generator=None):
        target = validate_full_horizon(batch, self.horizon, self.action_dim)
        if direction not in ("forward", "reverse"):
            raise ValueError("direction must be forward or reverse")
        if x0 is None:
            source = batch["source_actions"]
            if source.shape != target.shape:
                raise ValueError("source_actions must match future_actions shape")
            x0 = reference.augment(source.flatten(1), generator=generator)
        if x1 is None:
            x1 = reference.augment(target.flatten(1), generator=generator)
        tau = self.time_cutoff + (1 - 2 * self.time_cutoff) * torch.rand(
            len(target), device=target.device, dtype=target.dtype, generator=generator,
        )
        # Coupling endpoints and Gaussian target construction are not learned inputs.
        with torch.no_grad():
            state = reference.sample_bridge(x0.detach(), x1.detach(), tau, generator=generator)
            forward, reverse = reference.controls(state, x0.detach(), x1.detach(), tau)
            regression_target = forward if direction == "forward" else reverse
            distance = 1 - tau if direction == "forward" else tau
            weights = distance.pow(3 if self.kinetic else 1)
        field = self.forward_field if direction == "forward" else self.reverse_field
        prediction = field(state, tau, batch["obs_hist"], batch["act_hist"],
                           get_completion_ids(batch), batch.get("has_previous_plan"))
        error = (prediction - regression_target).square().mean(-1)
        loss = (weights * error).mean()
        return {"loss": loss, "control_mse": error.mean().detach(),
                "weighted_control_mse": loss.detach()}

    @torch.no_grad()
    def rollout(self, state, obs_hist, act_hist, mode, reference, *, reverse=False,
                generator=None, steps=None, has_previous_plan=None):
        """Reverse time is increasing clock s; the field still receives tau=1-s."""
        count = self.num_inference_steps if steps is None else int(steps)
        if count < 1:
            raise ValueError("rollout steps must be positive")
        field = self.reverse_field if reverse else self.forward_field
        conditioning = field.conditioning(obs_hist, act_hist, mode, has_previous_plan)
        grid = .5 - .5 * torch.cos(torch.linspace(0, math.pi, count + 1, device=state.device, dtype=state.dtype))
        energy = torch.zeros(len(state), device=state.device, dtype=state.dtype)
        # Save a few candidate plans, not physical trajectories.
        snapshots = [reference.positions(state).detach().clone()]
        for index in range(count):
            dt = grid[index + 1] - grid[index]
            tau = 1 - grid[index] if reverse else grid[index]
            # No training at singular endpoints; use the nearest supported time.
            tau = tau.clamp(self.time_cutoff, 1 - self.time_cutoff)
            control = field.conditioned(state, tau, conditioning)
            energy += .5 * dt * control.square().sum(-1)
            state = reference.step(state, control, dt, reverse=reverse, generator=generator)
            if index in (count // 3, 2 * count // 3, count - 1):
                snapshots.append(reference.positions(state).detach().clone())
        return state, {"nfe": count, "control_energy": energy,
                       "revision_states": torch.stack(snapshots, dim=1)}

    @torch.no_grad()
    def sample(self, obs_hist, act_hist, mode=None, *, source_actions=None, reference=None,
               generator=None, has_previous_plan=None):
        if source_actions is None or source_actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("SB requires an aligned/completed source_actions chunk")
        if reference is None:
            raise ValueError("SB requires frozen context-dependent Gaussian reference coefficients")
        state = reference.augment(source_actions.flatten(1), generator=generator)
        state, metrics = self.rollout(state, obs_hist, act_hist, mode, reference,
                                      generator=generator, has_previous_plan=has_previous_plan)
        return reference.positions(state).reshape_as(source_actions), metrics


def build_policy(config: Mapping, *, encoder: nn.Module | None = None) -> nn.Module:
    """Rebuild a generator from its saved shape/config, without data or simulation."""
    encoder_spec = config.get("encoder_spec", {"kind": "HistoryEncoder", "observations": "flat_tensor"})
    if encoder is None and (
        not isinstance(encoder_spec, Mapping)
        or encoder_spec.get("kind") != "HistoryEncoder"
        or encoder_spec.get("observations", "flat_tensor") != "flat_tensor"
    ):
        raise ValueError("unknown encoder_spec: supply its encoder explicitly through a caller-provided factory")
    for key in ("horizon", "action_dim", "obs_history", "action_history", "obs_dim"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    method = config["method"]
    if method == "ddim":
        return DDIMChunkPolicy(config, encoder=encoder)
    if method in ("fm_paired", "fm_local_ot"):
        return FlowMatchingPolicy(config, encoder=encoder)
    if method in ("sb_ou", "sb_kinetic"):
        return BridgePolicy(config, encoder=encoder)
    raise ValueError(f"unknown plan revision method: {method}")


__all__ = ["FieldNet", "DDIMChunkPolicy", "FlowMatchingPolicy", "BridgePolicy",
           "build_policy", "validate_full_horizon", "get_completion_ids"]

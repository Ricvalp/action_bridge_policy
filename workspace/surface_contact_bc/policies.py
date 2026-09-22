"""Thin toy adapters around the repository's Action Bridge and DDIM policies.

Every wrapper takes the same normalized history and returns normalized world
targets. Losses use demonstrations only. Simulator/contact metrics never enter
the optimizer. A normal/tangent coordinate change is the sole frame-DP change.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from action_bridge.models.action_bridge_policy import ActionBridgePolicy
from action_bridge.models.diffusion_policy import DiffusionPolicy
from action_bridge.training.losses import contact_path_losses_conditioned, contact_unrolled_mse_conditioned

from .config import METHODS
from .reference import ContactFrameReference, contact_frame, path_diagnostics


class ToyPolicy(nn.Module):
    def __init__(self, config: dict, stats: dict):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.stats = copy.deepcopy(stats)
        self.obs_history = int(config.get("obs_history", 2))
        self.action_history = int(config.get("action_history", 2))
        self.chunk_horizon = int(config.get("horizon", 8))


class BridgePolicy(ToyPolicy):
    def __init__(self, config: dict, stats: dict):
        super().__init__(config, stats)
        model_config = {
            "hidden_dim": int(config.get("hidden_dim", 128)),
            "h_emb_dim": int(config.get("h_emb_dim", 128)),
            "time_emb_dim": int(config.get("time_emb_dim", 32)),
            "latent_type": "none",
            "control_scale": float(config.get("control_scale", 1.0)),
        }
        self.model = ActionBridgePolicy(
            obs_dim=11, action_dim=2, obs_history=self.obs_history,
            action_history=self.action_history, chunk_horizon=self.chunk_horizon,
            model_config=model_config,
            reference_config={"type": "contact_langevin", "coordinate_mode": "absolute_action", "dt": 1.0},
        )
        # Specialize our own policy, without changing existing experiments.
        self.model.reference_process = ContactFrameReference(config, stats)
        self.reference_only = config["method"] == "reference_only"
        self.beta_kl = 0.0 if config["method"] == "action_bridge_contact_frame_no_kl" else float(config.get("beta_kl", 0.001))
        # A small initial residual avoids arbitrary high-acceleration free paths.
        head = self.model.control_net.net[-1]
        nn.init.normal_(head.weight, std=0.001)
        nn.init.zeros_(head.bias)

    def loss(self, batch):
        if self.reference_only:
            raise ValueError("Reference-only is an ablation of a trained bridge, not a separate training objective")
        h = self.model.encode_history(batch["obs_hist"], batch["act_hist"])
        terms = contact_path_losses_conditioned(self.model, h, batch, None)
        # Average path sums over horizon and action dimensions, keeping the
        # fitting/KL ratio unchanged and loss scales comparable across horizons.
        path_size = batch["future_actions"].shape[1] * 2
        fitting = terms["nll"].mean() / path_size
        path_kl = terms["path_kl"].mean() / path_size
        lambda_unroll = float(self.config.get("lambda_unroll", 1.0))
        unroll = (
            contact_unrolled_mse_conditioned(self.model, h, batch, None).mean() / 2
            if lambda_unroll else fitting.new_zeros(())
        )
        loss = fitting + self.beta_kl * path_kl + lambda_unroll * unroll
        loss = loss + self.model.reference_process.lambda_ref_reg * terms["ref_reg"]
        return {
            "loss": loss,
            "fitting": fitting.detach(),
            "teacher_mse": terms["mse"].mean().detach() / 2,
            "path_kl": path_kl.detach(),
            "unroll_mse": unroll.detach(),
            "gamma_mean": terms["gamma_mean"].detach(),
        }

    @torch.no_grad()
    def generate(self, obs_hist, act_hist, generator=None, diagnostics=False):
        del generator  # The primary bridge path is deterministic.
        model, reference = self.model, self.model.reference_process
        h = model.encode_history(obs_hist, act_hist)
        q, p = model.coordinate_adapter.init_qp_from_history({"act_hist": act_hist})
        actions, records = [], []
        for k in range(self.chunk_horizon):
            force, aux = reference.force(q, p, h, k, obs_state=obs_hist[:, -1])
            control = torch.zeros_like(q) if self.reference_only else model.contact_control(q, p, h, k)
            p = p + reference.dt * (force + reference.sigma_like(q) * control)
            q = q + reference.dt * p
            actions.append(q)
            if diagnostics:
                records.append(path_diagnostics(reference, force, control, aux))
        return torch.stack(actions, 1), stack_diagnostics(records)


def stack_diagnostics(records):
    return {key: torch.stack([record[key] for record in records], 1) for key in records[0]} if records else {}


class DiffusionToyPolicy(ToyPolicy):
    def __init__(self, config, stats):
        super().__init__(config, stats)
        self.prediction_type = str(config.get("prediction_type", "epsilon"))
        if self.prediction_type not in {"epsilon", "sample"}:
            raise ValueError("prediction_type must be 'epsilon' (noise) or 'sample' (clean target)")
        self.config["prediction_type"] = self.prediction_type
        self.model = DiffusionPolicy(
            obs_dim=11, action_dim=2, obs_history=self.obs_history,
            action_history=self.action_history, chunk_horizon=self.chunk_horizon,
            model_config={
                "hidden_dim": int(config.get("hidden_dim", 128)),
                "h_emb_dim": int(config.get("h_emb_dim", 128)),
                "time_emb_dim": int(config.get("time_emb_dim", 32)),
                "unet_channels": config.get("unet_channels", [32, 64]),
                "num_train_timesteps": int(config.get("num_train_timesteps", 100)),
                "num_inference_steps": int(config.get("num_inference_steps", 20)),
            },
        )
        # Diffusers' DDIM step interprets "sample" as the predicted clean x0.
        # The shared core defaults to epsilon; keep this optional setting local.
        self.model.noise_scheduler.register_to_config(prediction_type=self.prediction_type)
        self.local_frame = config["method"] == "diffusion_contact_frame"
        self.register_buffer("obs_mean", torch.tensor(stats["obs_mean"], dtype=torch.float32))
        self.register_buffer("obs_scale", torch.tensor(stats["obs_scale"], dtype=torch.float32))
        self.register_buffer("action_mean", torch.tensor(stats["action_mean"], dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(stats["action_scale"], dtype=torch.float32))

    def set_inference_steps(self, steps: int):
        if not 1 <= steps <= self.model.num_train_timesteps:
            raise ValueError("DDIM steps must be between 1 and num_train_timesteps")
        self.model.num_inference_steps = int(steps)
        self.config["num_inference_steps"] = int(steps)

    def frame(self, obs_hist):
        raw = obs_hist[:, -1] * self.obs_scale + self.obs_mean
        n, t = contact_frame(raw[:, 4:6])
        rotation = torch.stack((n, t), dim=-2)
        surface_origin = raw[:, 6:7] * n
        origin = (surface_origin - self.action_mean) / self.action_scale
        return rotation, origin

    def encode_frame(self, obs_hist, act_hist, future_actions=None):
        rotation, origin = self.frame(obs_hist)

        def local_targets(targets):
            return torch.einsum("bij,bhj->bhi", rotation, targets - origin[:, None])

        # An invertible coordinate change using the SAME train-only statistics.
        # Retain normal, c, goal, force and time: no context is privileged/lost.
        raw = obs_hist * self.obs_scale + self.obs_mean
        local_obs = obs_hist.clone()
        position = (raw[..., :2] - self.action_mean) / self.action_scale
        local_obs[..., :2] = local_targets(position)
        velocity_scale = self.obs_scale[2:4].square().mean().sqrt()
        local_obs[..., 2:4] = torch.einsum("bij,bhj->bhi", rotation, raw[..., 2:4] / velocity_scale)
        future = None if future_actions is None else local_targets(future_actions)
        return local_obs, local_targets(act_hist), future

    def decode_frame(self, actions, obs_hist):
        rotation, origin = self.frame(obs_hist)
        return torch.einsum("bji,bhj->bhi", rotation, actions) + origin[:, None]

    def _diffusion_loss(self, batch):
        """Ordinary denoising BC, predicting either epsilon or the clean chunk.

        Both corruption and DDIM sampling remain Diffusers operations. The core
        network method is historically named predict_noise; for sample mode its
        identical U-Net output is trained to predict x0 instead.
        """
        if self.prediction_type == "epsilon":
            losses = self.model.diffusion_loss(batch)
            return {**losses, "denoising_mse": losses["noise_mse"]}
        actions = batch["future_actions"]
        noise = torch.randn_like(actions)
        timesteps = torch.randint(self.model.num_train_timesteps, (len(actions),), device=actions.device)
        corrupted = self.model.noise_scheduler.add_noise(actions, noise, timesteps)
        history = self.model.encode_history(batch["obs_hist"], batch["act_hist"])
        prediction = self.model.predict_noise(corrupted, timesteps, history)
        loss = torch.nn.functional.mse_loss(prediction, actions)
        return {"loss": loss, "denoising_mse": loss.detach(), "action_denoising_mse": loss.detach()}

    def loss(self, batch):
        if not self.local_frame:
            return self._diffusion_loss(batch)
        obs, history, future = self.encode_frame(batch["obs_hist"], batch["act_hist"], batch["future_actions"])
        return self._diffusion_loss({"obs_hist": obs, "act_hist": history, "future_actions": future})

    @torch.no_grad()
    def generate(self, obs_hist, act_hist, generator=None, diagnostics=False):
        del diagnostics
        if self.local_frame:
            obs, history, _ = self.encode_frame(obs_hist, act_hist)
            actions = self.model.generate(obs, history, generator=generator)
            return self.decode_frame(actions, obs_hist), {}
        return self.model.generate(obs_hist, act_hist, generator=generator), {}


class FrozenReferenceDecoder(nn.Module):
    """Only the learned reference and its history encoder, not bridge control."""

    def __init__(self, bridge: BridgePolicy):
        super().__init__()
        self.reference = copy.deepcopy(bridge.model.reference_process)
        self.encoder = copy.deepcopy(bridge.model.history_encoder)
        self.adapter = bridge.model.coordinate_adapter
        self.requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def inverse_controls(self, batch):
        q = self.adapter.build_q_sequence(batch)
        p = self.adapter.build_p_sequence(q, batch)
        h = self.encoder(batch["obs_hist"], batch["act_hist"])
        controls = []
        for k in range(q.shape[1] - 1):
            force, _ = self.reference.force(q[:, k], p[:, k], h, k, obs_state=batch["obs_hist"][:, -1])
            controls.append(((p[:, k + 1] - p[:, k]) / self.reference.dt - force) / self.reference.sigma_like(q[:, k]))
        return torch.stack(controls, 1)

    @torch.no_grad()
    def decode(self, controls, obs_hist, act_hist, diagnostics):
        q, p = self.adapter.init_qp_from_history({"act_hist": act_hist})
        h = self.encoder(obs_hist, act_hist)
        actions, records = [], []
        for k in range(controls.shape[1]):
            force, aux = self.reference.force(q, p, h, k, obs_state=obs_hist[:, -1])
            control = controls[:, k]
            p = p + self.reference.dt * (force + self.reference.sigma_like(q) * control)
            q = q + self.reference.dt * p
            actions.append(q)
            if diagnostics:
                records.append(path_diagnostics(self.reference, force, control, aux))
        return torch.stack(actions, 1), stack_diagnostics(records)


class ResidualDiffusionPolicy(DiffusionToyPolicy):
    def __init__(self, config, stats, reference_policy=None):
        super().__init__(config, stats)
        if reference_policy is None:
            if "reference_config" not in config:
                raise ValueError("Residual diffusion requires a trained reference checkpoint")
            # State is restored from our own checkpoint after construction.
            reference_policy = BridgePolicy(config["reference_config"], stats)
        if reference_policy.chunk_horizon != self.chunk_horizon:
            raise ValueError("Residual diffusion and frozen reference must use the same horizon")
        if reference_policy.stats != stats:
            raise ValueError("Frozen reference and residual diffusion must share dataset normalization")
        self.config["reference_config"] = copy.deepcopy(reference_policy.config)
        self.decoder = FrozenReferenceDecoder(reference_policy)
        self.register_buffer("residual_mean", torch.zeros(2))
        self.register_buffer("residual_scale", torch.ones(()))

    @torch.no_grad()
    def fit_residual_normalization(self, train_batch):
        controls = self.decoder.inverse_controls(train_batch)
        self.residual_mean.copy_(controls.mean((0, 1)))
        self.residual_scale.copy_((controls - self.residual_mean).square().mean().sqrt().clamp_min(1e-4))

    def loss(self, batch):
        controls = self.decoder.inverse_controls(batch)
        targets = (controls - self.residual_mean) / self.residual_scale
        return self._diffusion_loss({**batch, "future_actions": targets})

    @torch.no_grad()
    def generate(self, obs_hist, act_hist, generator=None, diagnostics=False):
        controls = self.model.generate(obs_hist, act_hist, generator=generator)
        controls = controls * self.residual_scale + self.residual_mean
        return self.decoder.decode(controls, obs_hist, act_hist, diagnostics)


def build_policy(config: dict, stats: dict, reference_policy=None) -> ToyPolicy:
    method = config["method"]
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if method == "diffusion_residual":
        return ResidualDiffusionPolicy(config, stats, reference_policy)
    if method.startswith("diffusion_"):
        return DiffusionToyPolicy(config, stats)
    return BridgePolicy(config, stats)

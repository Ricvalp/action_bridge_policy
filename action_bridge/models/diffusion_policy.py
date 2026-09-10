"""A small conditional temporal U-Net with Diffusers DDIM sampling.

The network predicts noise on normalized action chunks. Conditioning uses the
same observation/action history encoder as our other policies; no future expert
actions enter the conditioning. There is no latent posterior or reference model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from action_bridge.models.encoders import HistoryEncoder, SinusoidalTimeEmbedding


class _ConditionalResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, condition_dim: int):
        super().__init__()
        groups = min(8, output_channels // 2)
        self.first = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )
        self.condition = nn.Sequential(
            nn.SiLU(), nn.Linear(condition_dim, 2 * output_channels)
        )
        self.second = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )
        self.residual = (
            nn.Conv1d(input_channels, output_channels, kernel_size=1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, actions: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, bias = self.condition(condition).unsqueeze(-1).chunk(2, dim=1)
        hidden = self.first(actions)
        hidden = (1.0 + scale) * hidden + bias
        return self.second(hidden) + self.residual(actions)


class _ConditionalUnet1D(nn.Module):
    def __init__(self, action_dim: int, channels: Sequence[int], condition_dim: int):
        super().__init__()
        self.padding_multiple = 2 ** (len(channels) - 1)
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        previous = action_dim
        for index, width in enumerate(channels):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        _ConditionalResidualBlock(previous, width, condition_dim),
                        _ConditionalResidualBlock(width, width, condition_dim),
                    ]
                )
            )
            if index < len(channels) - 1:
                self.downsample.append(nn.Conv1d(width, width, 4, stride=2, padding=1))
            previous = width

        self.middle = nn.ModuleList(
            [
                _ConditionalResidualBlock(channels[-1], channels[-1], condition_dim),
                _ConditionalResidualBlock(channels[-1], channels[-1], condition_dim),
            ]
        )
        self.upsample = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for width in reversed(channels[:-1]):
            self.upsample.append(
                nn.ConvTranspose1d(previous, width, 4, stride=2, padding=1)
            )
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        _ConditionalResidualBlock(2 * width, width, condition_dim),
                        _ConditionalResidualBlock(width, width, condition_dim),
                    ]
                )
            )
            previous = width
        self.output = nn.Sequential(
            nn.GroupNorm(min(8, channels[0] // 2), channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], action_dim, kernel_size=1),
        )

    def forward(self, actions: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        horizon = actions.shape[1]
        hidden = actions.transpose(1, 2)
        # Padding/cropping supports short or odd horizons without special cases
        # in the U-Net. Only the original action timesteps enter the loss.
        padding = (-horizon) % self.padding_multiple
        if padding:
            hidden = F.pad(hidden, (0, padding), mode="replicate")
        skips = []
        for index, blocks in enumerate(self.down_blocks):
            for block in blocks:
                hidden = block(hidden, condition)
            if index < len(self.downsample):
                skips.append(hidden)
                hidden = self.downsample[index](hidden)
        for block in self.middle:
            hidden = block(hidden, condition)
        for upsample, blocks, skip in zip(
            self.upsample, self.up_blocks, reversed(skips), strict=True
        ):
            hidden = torch.cat((upsample(hidden), skip), dim=1)
            for block in blocks:
                hidden = block(hidden, condition)
        return self.output(hidden).transpose(1, 2)[:, :horizon]


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_history: int,
        action_history: int,
        chunk_horizon: int,
        model_config: Mapping,
    ):
        super().__init__()
        # Import lazily so non-diffusion policies do not require Diffusers.
        try:
            from diffusers import DDIMScheduler
        except ImportError as error:
            raise RuntimeError(
                "DiffusionPolicy requires diffusers==0.39.0; add --extra diffusion "
                "to the environment setup command alongside your existing extras."
            ) from error

        channels = tuple(
            int(width) for width in model_config.get("unet_channels", (80, 160, 320))
        )
        if not channels or any(width < 8 or width % 8 for width in channels):
            raise ValueError(
                "unet_channels must be nonempty positive multiples of eight"
            )
        history_dim = int(model_config.get("h_emb_dim", 256))
        hidden_dim = int(model_config.get("hidden_dim", 256))
        time_dim = int(model_config.get("time_emb_dim", 64))
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.num_train_timesteps = int(model_config.get("num_train_timesteps", 100))
        self.num_inference_steps = int(model_config.get("num_inference_steps", 20))
        if self.chunk_horizon < 1:
            raise ValueError("chunk_horizon must be positive")
        if not 1 <= self.num_inference_steps <= self.num_train_timesteps:
            raise ValueError(
                "num_inference_steps must be between one and num_train_timesteps"
            )
        self.history_encoder = HistoryEncoder(
            obs_history, action_history, obs_dim, action_dim, history_dim, hidden_dim
        )
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, 4 * time_dim),
            nn.SiLU(),
            nn.Linear(4 * time_dim, time_dim),
        )
        self.unet = _ConditionalUnet1D(action_dim, channels, history_dim + time_dim)
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule=model_config.get("beta_schedule", "squaredcos_cap_v2"),
            prediction_type="epsilon",
            clip_sample=False,
            # With unclipped epsilon predictions, starting at the nearly zero-
            # SNR final cosine step amplifies small noise errors dramatically.
            timestep_spacing=model_config.get("timestep_spacing", "leading"),
        )

    def encode_history(
        self, obs_hist: torch.Tensor, act_hist: torch.Tensor
    ) -> torch.Tensor:
        return self.history_encoder(obs_hist, act_hist)

    def predict_noise(
        self, actions: torch.Tensor, timesteps: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        timesteps = timesteps.expand(actions.shape[0])
        condition = torch.cat((history, self.time_embedding(timesteps)), dim=-1)
        return self.unet(actions, condition)

    def diffusion_loss(
        self, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        actions = batch["future_actions"]
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            self.num_train_timesteps, (actions.shape[0],), device=actions.device
        )
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        history = self.encode_history(batch["obs_hist"], batch["act_hist"])
        prediction = self.predict_noise(noisy_actions, timesteps, history)
        loss = F.mse_loss(prediction, noise)
        return {"loss": loss, "noise_mse": loss.detach()}

    @torch.inference_mode()
    def generate(
        self,
        obs_hist: torch.Tensor,
        act_hist: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        history = self.encode_history(obs_hist, act_hist)
        actions = torch.randn(
            (obs_hist.shape[0], self.chunk_horizon, self.action_dim),
            device=obs_hist.device,
            dtype=obs_hist.dtype,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(
            self.num_inference_steps, device=actions.device
        )
        for timestep in self.noise_scheduler.timesteps:
            noise = self.predict_noise(actions, timestep, history)
            actions = self.noise_scheduler.step(
                noise, timestep, actions, eta=0.0, generator=generator
            ).prev_sample
        return actions

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        return self.generate(obs_hist, act_hist)


__all__ = ["DiffusionPolicy"]

"""Query-only RLBench point-cloud and history encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from action_bridge.jax.models.attention import (
    CrossAttentionBlock,
    LatentPerceiver,
    SelfAttentionBlock,
    SelfAttentionStack,
    TransformerConfig,
)


@dataclass(frozen=True)
class EncoderConfig:
    encoder_type: str = "supernode"
    d_model: int = 256
    n_heads: int = 4
    mlp_mult: int = 4
    dropout: float = 0.0
    frame_num_latents: int = 8
    frame_layers: int = 2
    supernodes: int = 64
    supernode_temperature: float = 0.005
    supernode_center_sampling: str = "linspace"
    supernode_layers: int = 2
    history_layers: int = 1
    query_num_latents: int = 64
    query_layers: int = 1
    max_obs_history: int = 16
    max_action_history: int = 32
    mask_id_vocab: int = 256
    use_rgb: bool = True
    use_mask_id: bool = False
    use_task_tokens: bool = True

    def transformer(self) -> TransformerConfig:
        return TransformerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mlp_mult=self.mlp_mult,
            dropout=self.dropout,
        )


class PointFeatureEmbed(nn.Module):
    cfg: EncoderConfig

    @nn.compact
    def __call__(
        self,
        xyz: jnp.ndarray,
        *,
        rgb: Optional[jnp.ndarray] = None,
        mask_id: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        pieces = [xyz.astype(jnp.float32)]
        if bool(self.cfg.use_rgb) and rgb is not None:
            pieces.append(rgb.astype(jnp.float32))
        x = nn.Dense(int(self.cfg.d_model), name="xyz_rgb_proj")(jnp.concatenate(pieces, -1))
        if bool(self.cfg.use_mask_id) and mask_id is not None:
            ids = jnp.clip(mask_id.astype(jnp.int32), 0, int(self.cfg.mask_id_vocab) - 1)
            x = x + nn.Embed(
                int(self.cfg.mask_id_vocab), int(self.cfg.d_model), name="mask_embed"
            )(ids)
        return x


def _linspace_centers(batch_size: int, num_points: int, num_centers: int) -> jnp.ndarray:
    indices = jnp.linspace(0, max(num_points - 1, 0), num_centers).round().astype(jnp.int32)
    return jnp.broadcast_to(indices[None], (batch_size, num_centers))


def _mask_balanced_centers(
    valid: jnp.ndarray,
    mask_id: jnp.ndarray,
    num_centers: int,
) -> jnp.ndarray:
    batch, num_points = valid.shape

    def one_row(row_valid, row_mask):
        invalid_id = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
        safe_mask = jnp.where(row_valid, row_mask.astype(jnp.int32), invalid_id)
        order = jnp.argsort(safe_mask, stable=True)
        sorted_valid = row_valid[order]
        sorted_mask = safe_mask[order]
        previous = jnp.concatenate([jnp.asarray([-1], jnp.int32), sorted_mask[:-1]])
        starts = (sorted_mask != previous) & sorted_valid
        group_id = jnp.cumsum(starts.astype(jnp.int32)) - 1
        safe_group = jnp.maximum(group_id, 0)
        positions = jnp.arange(num_points, dtype=jnp.int32)
        start_positions = jnp.where(starts, positions, 0)
        group_start = jax.lax.associative_scan(jnp.maximum, start_positions)
        rank = positions - group_start
        counts = jnp.bincount(
            safe_group,
            weights=sorted_valid.astype(jnp.float32),
            length=num_points,
        )
        fraction = rank.astype(jnp.float32) / jnp.maximum(counts[safe_group], 1.0)
        key = fraction + 1e-3 * safe_group.astype(jnp.float32) / max(num_points, 1)
        key = jnp.where(sorted_valid, key, 2.0 + positions / max(num_points, 1))
        balanced_order = jnp.argsort(key, stable=True)
        take = jnp.arange(num_centers, dtype=jnp.int32) % max(num_points, 1)
        return order[balanced_order[take]]

    return jax.vmap(one_row)(valid.astype(jnp.bool_), mask_id).reshape(batch, num_centers)


class SupernodeFrameTokenizer(nn.Module):
    cfg: EncoderConfig

    @nn.compact
    def __call__(
        self,
        xyz: jnp.ndarray,
        state: jnp.ndarray,
        valid: jnp.ndarray,
        *,
        rgb: Optional[jnp.ndarray] = None,
        mask_id: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch, num_points = xyz.shape[:2]
        if self.cfg.supernode_center_sampling == "mask_balanced" and mask_id is not None:
            indices = _mask_balanced_centers(valid, mask_id, int(self.cfg.supernodes))
        elif self.cfg.supernode_center_sampling == "linspace":
            indices = _linspace_centers(batch, num_points, int(self.cfg.supernodes))
        else:
            raise ValueError("supernode_center_sampling must be linspace or mask_balanced.")
        points = PointFeatureEmbed(self.cfg, name="point_embed")(
            xyz, rgb=rgb, mask_id=mask_id
        )
        centers = jnp.take_along_axis(xyz.astype(jnp.float32), indices[:, :, None], axis=1)
        distance2 = jnp.sum((centers[:, :, None] - xyz[:, None].astype(jnp.float32)) ** 2, -1)
        logits = -distance2 / max(float(self.cfg.supernode_temperature), 1e-6)
        logits = jnp.where(valid[:, None].astype(jnp.bool_), logits, -1e9)
        weights = nn.softmax(logits, axis=-1)
        tokens = jnp.einsum("bmn,bnd->bmd", weights, points)
        state_token = nn.Dense(int(self.cfg.d_model), name="state_proj")(
            state.astype(jnp.float32)
        )[:, None]
        tokens = jnp.concatenate([tokens, state_token], axis=1)
        mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)
        tokens = SelfAttentionStack(
            self.cfg.transformer(), int(self.cfg.supernode_layers), name="refine"
        )(tokens, mask=mask, train=train)
        return tokens, mask


class PerceiverFrameTokenizer(nn.Module):
    cfg: EncoderConfig

    @nn.compact
    def __call__(
        self,
        xyz: jnp.ndarray,
        state: jnp.ndarray,
        valid: jnp.ndarray,
        *,
        rgb: Optional[jnp.ndarray] = None,
        mask_id: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        points = PointFeatureEmbed(self.cfg, name="point_embed")(
            xyz, rgb=rgb, mask_id=mask_id
        )
        state_token = nn.Dense(int(self.cfg.d_model), name="state_proj")(
            state.astype(jnp.float32)
        )[:, None]
        tokens = jnp.concatenate([points, state_token], axis=1)
        mask = jnp.concatenate(
            [valid.astype(jnp.bool_), jnp.ones((valid.shape[0], 1), dtype=jnp.bool_)],
            axis=1,
        )
        tokens = LatentPerceiver(
            self.cfg.transformer(),
            int(self.cfg.frame_num_latents),
            int(self.cfg.frame_layers),
            name="frame_perceiver",
        )(tokens, token_mask=mask, train=train)
        return tokens, jnp.ones(tokens.shape[:2], dtype=jnp.bool_)


class RLBenchHistoryEncoder(nn.Module):
    cfg: EncoderConfig
    num_tasks: int
    num_task_variations: int

    @nn.compact
    def __call__(
        self,
        batch: Dict[str, jnp.ndarray],
        *,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        xyz = batch["point_cloud_hist"]
        state = batch["obs_hist"]
        valid = batch["point_valid_hist"]
        batch_size, obs_history, num_points = xyz.shape[:3]
        if obs_history > int(self.cfg.max_obs_history):
            raise ValueError("Observation history exceeds encoder.max_obs_history.")
        rgb = batch.get("rgb_hist")
        mask_id = batch.get("mask_id_hist")
        tokenizer_cls = {
            "supernode": SupernodeFrameTokenizer,
            "perceiver": PerceiverFrameTokenizer,
        }.get(str(self.cfg.encoder_type))
        if tokenizer_cls is None:
            raise ValueError("encoder_type must be supernode or perceiver.")
        frame_tokens, frame_mask = tokenizer_cls(self.cfg, name="frame_tokenizer")(
            xyz.reshape(batch_size * obs_history, num_points, 3),
            state.reshape(batch_size * obs_history, state.shape[-1]),
            valid.reshape(batch_size * obs_history, num_points),
            rgb=None if rgb is None else rgb.reshape(batch_size * obs_history, num_points, 3),
            mask_id=None if mask_id is None else mask_id.reshape(batch_size * obs_history, num_points),
            train=train,
        )
        tokens_per_frame = int(frame_tokens.shape[1])
        frame_tokens = frame_tokens.reshape(
            batch_size, obs_history, tokens_per_frame, int(self.cfg.d_model)
        )
        frame_positions = self.param(
            "frame_positions",
            nn.initializers.normal(stddev=0.02),
            (int(self.cfg.max_obs_history), int(self.cfg.d_model)),
        )[:obs_history]
        frame_tokens = frame_tokens + frame_positions[None, :, None]
        frame_tokens = frame_tokens.reshape(batch_size, obs_history * tokens_per_frame, -1)
        obs_mask = batch["obs_history_mask"].astype(jnp.bool_)
        frame_mask = frame_mask.reshape(batch_size, obs_history, tokens_per_frame)
        frame_mask = (frame_mask & obs_mask[:, :, None]).reshape(batch_size, -1)

        action_history = batch["act_hist"].astype(jnp.float32)
        history_length = int(action_history.shape[1])
        if history_length > int(self.cfg.max_action_history):
            raise ValueError("Action history exceeds encoder.max_action_history.")
        action_tokens = nn.Dense(int(self.cfg.d_model), name="action_history_proj")(action_history)
        action_positions = self.param(
            "action_history_positions",
            nn.initializers.normal(stddev=0.02),
            (int(self.cfg.max_action_history), int(self.cfg.d_model)),
        )[:history_length]
        action_tokens = action_tokens + action_positions[None]
        action_mask = batch["action_history_mask"].astype(jnp.bool_)
        if int(self.cfg.history_layers) > 0:
            action_tokens = SelfAttentionStack(
                self.cfg.transformer(), int(self.cfg.history_layers), name="action_history_refine"
            )(action_tokens, mask=action_mask, train=train)

        context = jnp.concatenate([frame_tokens, action_tokens], axis=1)
        context_mask = jnp.concatenate([frame_mask, action_mask], axis=1)
        if bool(self.cfg.use_task_tokens):
            task_table = self.param(
                "task_tokens",
                nn.initializers.normal(stddev=0.02),
                (max(1, int(self.num_tasks)), int(self.cfg.d_model)),
            )
            variation_table = self.param(
                "task_variation_tokens",
                nn.initializers.normal(stddev=0.02),
                (max(1, int(self.num_task_variations)), int(self.cfg.d_model)),
            )
            task_ids = jnp.clip(batch["task_id"].astype(jnp.int32), 0, task_table.shape[0] - 1)
            variation_ids = jnp.clip(
                batch["task_variation_id"].astype(jnp.int32),
                0,
                variation_table.shape[0] - 1,
            )
            class_tokens = jnp.stack(
                [task_table[task_ids], variation_table[variation_ids]], axis=1
            )
            context = jnp.concatenate([context, class_tokens], axis=1)
            context_mask = jnp.concatenate(
                [context_mask, jnp.ones((batch_size, 2), dtype=jnp.bool_)], axis=1
            )

        if int(self.cfg.query_layers) > 0:
            context = SelfAttentionStack(
                self.cfg.transformer(), int(self.cfg.query_layers), name="context_refine"
            )(context, mask=context_mask, train=train)
        if int(self.cfg.query_num_latents) > 0:
            context = LatentPerceiver(
                self.cfg.transformer(),
                int(self.cfg.query_num_latents),
                1,
                name="context_compressor",
            )(context, token_mask=context_mask, train=train)
            context_mask = jnp.ones(context.shape[:2], dtype=jnp.bool_)
        weights = context_mask.astype(jnp.float32)
        global_embedding = jnp.sum(context * weights[:, :, None], axis=1)
        global_embedding = global_embedding / jnp.maximum(weights.sum(1, keepdims=True), 1.0)
        return context, context_mask, global_embedding


class ActionQueryDecoder(nn.Module):
    d_model: int
    n_heads: int
    mlp_mult: int
    dropout: float
    num_layers: int
    horizon: int

    @nn.compact
    def __call__(
        self,
        context: jnp.ndarray,
        context_mask: jnp.ndarray,
        *,
        train: bool = False,
    ) -> jnp.ndarray:
        cfg = TransformerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mlp_mult=self.mlp_mult,
            dropout=self.dropout,
        )
        queries = self.param(
            "action_queries",
            nn.initializers.normal(stddev=0.02),
            (int(self.horizon), int(self.d_model)),
        )
        x = jnp.broadcast_to(
            queries[None], (context.shape[0], int(self.horizon), int(self.d_model))
        )
        for index in range(int(self.num_layers)):
            x = SelfAttentionBlock(cfg, name=f"self_{index}")(x, train=train)
            x = CrossAttentionBlock(cfg, name=f"cross_{index}")(
                x, context, context_mask=context_mask, train=train
            )
        return nn.LayerNorm(name="output_norm")(x)


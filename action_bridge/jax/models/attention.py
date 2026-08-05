"""Small Flax transformer building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 256
    n_heads: int = 4
    mlp_mult: int = 4
    dropout: float = 0.0


class MultiHeadAttention(nn.Module):
    cfg: TransformerConfig

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        *,
        context_mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        context = query if context is None else context
        d_model = int(self.cfg.d_model)
        n_heads = int(self.cfg.n_heads)
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        head_dim = d_model // n_heads
        q = nn.Dense(d_model, use_bias=False, name="q")(query)
        k = nn.Dense(d_model, use_bias=False, name="k")(context)
        v = nn.Dense(d_model, use_bias=False, name="v")(context)
        batch, query_len = q.shape[:2]
        context_len = int(k.shape[1])
        q = q.reshape(batch, query_len, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, context_len, n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, context_len, n_heads, head_dim).transpose(0, 2, 1, 3)
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(jnp.float32)
        logits = logits / jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32))
        if context_mask is not None:
            logits = jnp.where(
                context_mask.astype(jnp.bool_)[:, None, None, :],
                logits,
                jnp.asarray(-1e9, dtype=logits.dtype),
            )
        weights = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        if float(self.cfg.dropout) > 0.0:
            weights = nn.Dropout(rate=float(self.cfg.dropout))(
                weights, deterministic=not train
            )
        output = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        output = output.transpose(0, 2, 1, 3).reshape(batch, query_len, d_model)
        return nn.Dense(d_model, use_bias=False, name="out")(output)


class FeedForward(nn.Module):
    cfg: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        hidden = int(self.cfg.mlp_mult) * int(self.cfg.d_model)
        x = nn.Dense(hidden, name="fc1")(x)
        x = nn.gelu(x)
        if float(self.cfg.dropout) > 0.0:
            x = nn.Dropout(rate=float(self.cfg.dropout))(x, deterministic=not train)
        return nn.Dense(int(self.cfg.d_model), name="fc2")(x)


class SelfAttentionBlock(nn.Module):
    cfg: TransformerConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        y = nn.LayerNorm(name="ln1")(x)
        x = x + MultiHeadAttention(self.cfg, name="self_attn")(
            y, context_mask=mask, train=train
        )
        y = nn.LayerNorm(name="ln2")(x)
        return x + FeedForward(self.cfg, name="mlp")(y, train=train)


class CrossAttentionBlock(nn.Module):
    cfg: TransformerConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        context: jnp.ndarray,
        *,
        context_mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        y = nn.LayerNorm(name="ln_cross")(x)
        x = x + MultiHeadAttention(self.cfg, name="cross_attn")(
            y,
            context,
            context_mask=context_mask,
            train=train,
        )
        y = nn.LayerNorm(name="ln_mlp")(x)
        return x + FeedForward(self.cfg, name="mlp")(y, train=train)


class SelfAttentionStack(nn.Module):
    cfg: TransformerConfig
    num_layers: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        for index in range(int(self.num_layers)):
            x = SelfAttentionBlock(self.cfg, name=f"block_{index}")(
                x, mask=mask, train=train
            )
        return x


class LatentPerceiver(nn.Module):
    cfg: TransformerConfig
    num_latents: int
    num_layers: int

    @nn.compact
    def __call__(
        self,
        tokens: jnp.ndarray,
        *,
        token_mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        batch = int(tokens.shape[0])
        latents = self.param(
            "latents",
            nn.initializers.normal(stddev=0.02),
            (int(self.num_latents), int(self.cfg.d_model)),
        )
        x = jnp.broadcast_to(
            latents[None],
            (batch, int(self.num_latents), int(self.cfg.d_model)),
        )
        for index in range(int(self.num_layers)):
            x = CrossAttentionBlock(self.cfg, name=f"cross_{index}")(
                x, tokens, context_mask=token_mask, train=train
            )
            x = SelfAttentionBlock(self.cfg, name=f"self_{index}")(x, train=train)
        return x


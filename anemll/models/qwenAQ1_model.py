"""Qwen 3 model implementation with ANEMLL-QUANT-1 quantization.

This module provides a Qwen 3 implementation using AnemllConv2d layers
for low-rank scale quantization. Weights are stored as LUT[idx] values
in [-1, 1] multiplied by low-rank scales (scale_A @ scale_B).

Based on qwen_model.py with quantization support added.
"""

from __future__ import annotations

import os
import json
import math
from typing import Dict, Optional

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import quantization components
from .anemll_quant import (
    AnemllConv2d,
    AnemllQuantConfig,
    load_anemll_weights,
    convert_conv2d_to_anemll,
    DEFAULT_MLP_SCALE_RANK,
    DEFAULT_ATTN_SCALE_RANK,
)

# ---------------------------------------------------------------------------
# Qwen 3 ANEMLL-QUANT-1 model implementation
# ---------------------------------------------------------------------------

MODEL_DTYPE = torch.float16
TEST_DEVICE = "cpu"
CONTEXT_LENGTH = 256

# Cache configuration constants
FORCE_UNIFIED_CACHE = True
ENABLE_UNIFIED_CACHE = True
STATE_LENGTH = 256
DISABLE_KV_CACHE = False

# LM head configuration constants
ENABLE_CONV2D = bool(1)
ENABLE_VACAB_SPLIT = bool(1)
ENABLE_VACAB_SPLIT8 = bool(0)
ENABLE_VACAB_SPLIT16 = bool(1)
ENABLE_LOGITS2 = bool(1)
ENABLE_COREML = bool(0)


class QwenConfig:
    """Configuration for Qwen model with ANEMLL quantization support."""

    def __init__(self, **kwargs):
        self.architectures = kwargs.get("architectures", ["QwenForCausalLM"])
        self.attention_bias = kwargs.get("attention_bias", False)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.bos_token_id = kwargs.get("bos_token_id", 128000)
        self.eos_token_id = kwargs.get("eos_token_id", 128001)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.hidden_size = kwargs.get("hidden_size", 4096)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.intermediate_size = kwargs.get("intermediate_size", 14336)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 8192)
        self.model_type = kwargs.get("model_type", "qwen3")
        self.num_attention_heads = kwargs.get("num_attention_heads", 32)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)
        self.head_dim = kwargs.get(
            "head_dim",
            self.hidden_size // max(1, self.num_attention_heads),
        )
        self.pretraining_tp = kwargs.get("pretraining_tp", 1)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-05)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        if self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling.get("rope_type", "qwen3")
        self.rope_theta = kwargs.get("rope_theta", 500000.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)
        self.torch_required = kwargs.get("torch_dtype", "bfloat16")
        self.transformers_version = kwargs.get("transformers_version", "4.40.0.dev0")
        self.use_cache = kwargs.get("use_cache", True)
        self.vocab_size = kwargs.get("vocab_size", 128257)
        self.context_length = kwargs.get("context_length", CONTEXT_LENGTH)
        self.state_length = kwargs.get("state_length", STATE_LENGTH)

        # ANEMLL quantization config
        self.mlp_scale_rank = kwargs.get("mlp_scale_rank", DEFAULT_MLP_SCALE_RANK)
        self.attn_scale_rank = kwargs.get("attn_scale_rank", DEFAULT_ATTN_SCALE_RANK)
        self.lut_bits = kwargs.get("lut_bits", None)

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def get_quant_config(self) -> AnemllQuantConfig:
        """Get ANEMLL quantization config from model config."""
        return AnemllQuantConfig(
            mlp_scale_rank=self.mlp_scale_rank,
            attn_scale_rank=self.attn_scale_rank,
            lut_bits=self.lut_bits,
        )


def get_kv_cache_idx(layer_idx, num_layers, num_groups=1):
    """Helper function to get KV cache indices."""
    layers_per_group = num_layers // num_groups
    group_idx = layer_idx // layers_per_group
    layer_in_group_idx = layer_idx % layers_per_group
    return group_idx, layer_in_group_idx, layers_per_group


# -----------------------------------------------------------------------------
# Qwen building blocks with ANEMLL quantization
# -----------------------------------------------------------------------------


class QwenRMSNorm(nn.Module):
    """ANE optimized RMSNorm implementation."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        doubled = torch.cat([x, -x], dim=-1)
        hidden_size = hidden_states.shape[-1]
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * hidden_size,),
            weight=None,
            bias=None,
            eps=float(self.variance_epsilon)
        )
        normed = normed[..., :hidden_size]
        return (normed * self.weight
                       .to(normed.dtype, copy=False)
                       .to(normed.device, copy=False))


class QwenHeadNorm(nn.Module):
    """ANE optimized RMSNorm for attention heads."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        doubled = torch.cat([x, -x], dim=-1)
        hidden_size = hidden_states.shape[-1]
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * hidden_size,),
            weight=None,
            bias=None,
            eps=float(self.variance_epsilon)
        )
        normed = normed[..., :hidden_size]
        return (normed * self.weight
                       .to(normed.dtype, copy=False)
                       .to(normed.device, copy=False))


class QwenRotaryEmbedding(nn.Module):
    """Simple rotary positional embedding."""

    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.dim, 2).float().to(TEST_DEVICE) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max(config.context_length, config.state_length) * 2, device=TEST_DEVICE).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().unsqueeze(0)
        self.sin_cached = emb.sin().unsqueeze(0)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor | None = None):
        if position_ids is not None:
            if position_ids.dim() == 1:
                pos_ids = position_ids
            else:
                pos_ids = position_ids.squeeze(0)
            cos = self.cos_cached[:, pos_ids].to(x.dtype)
            sin = self.sin_cached[:, pos_ids].to(x.dtype)
            return cos, sin
        else:
            seq_len = x.shape[1]
            return (
                self.cos_cached[:, :seq_len].to(x.dtype),
                self.sin_cached[:, :seq_len].to(x.dtype),
            )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_prefill(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_single(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, n_kv, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].repeat(1, 1, n_rep, 1, 1)
    return hidden_states.view(bsz, n_kv * n_rep, seq_len, head_dim)


class QwenMLP(nn.Module):
    """Qwen MLP with AnemllConv2d for quantization support."""

    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.scale_rank = config.mlp_scale_rank

        # Use AnemllConv2d layers for quantization
        self.gate_proj = AnemllConv2d(
            self.hidden_size, self.intermediate_size,
            scale_rank=self.scale_rank, dtype=MODEL_DTYPE
        )
        self.up_proj = AnemllConv2d(
            self.hidden_size, self.intermediate_size,
            scale_rank=self.scale_rank, dtype=MODEL_DTYPE
        )
        self.down_proj = AnemllConv2d(
            self.intermediate_size, self.hidden_size,
            scale_rank=self.scale_rank, dtype=MODEL_DTYPE
        )

        self.act_fn = F.silu

    def forward(self, x):
        x = x.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)

        a = self.gate_proj(x)
        b = self.up_proj(x)
        c = self.act_fn(a)
        d = c * b
        e = self.down_proj(d)

        return e.squeeze(2).permute(0, 2, 1)


class QwenAttention(nn.Module):
    """Qwen Attention with AnemllConv2d for quantization support."""

    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.scale_rank = config.attn_scale_rank

        if not hasattr(QwenAttention, '_config_printed'):
            print(f"QwenAttention (AQ1) using head_dim={self.head_dim}, scale_rank={self.scale_rank}")
            QwenAttention._config_printed = True

        self.rotary_emb = QwenRotaryEmbedding(config)

        q_proj_dim = self.num_heads * self.head_dim
        kv_proj_dim = self.num_kv_heads * self.head_dim

        # Use AnemllConv2d layers for quantization
        self.q_proj = AnemllConv2d(
            self.hidden_size, q_proj_dim,
            scale_rank=self.scale_rank, dtype=MODEL_DTYPE
        )
        self.k_proj = AnemllConv2d(
            self.hidden_size, kv_proj_dim,
            scale_rank=self.scale_rank, dtype=MODEL_DTYPE
        )
        self.v_proj = AnemllConv2d(
            self.hidden_size, kv_proj_dim,
            scale_rank=self.scale_rank, dtype=MODEL_DTYPE
        )
        self.o_proj = AnemllConv2d(
            q_proj_dim, self.hidden_size,
            scale_rank=self.scale_rank, dtype=MODEL_DTYPE
        )

        self.q_norm = QwenHeadNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = QwenHeadNorm(self.head_dim, eps=config.rms_norm_eps)
        self.scale = 1 / math.sqrt(self.head_dim)

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = x.repeat(1, n_rep, 1, 1)
        x = x.view(1, -1, x.size(-2), x.size(-1))
        return x

    def get_new_kv_cache(self, hidden_states, current_pos, rotary_emb):
        """Get new key-value cache entries for single token generation."""
        bsz, q_len, _ = hidden_states.shape

        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

        query_states = self.q_proj(hidden_states).view(1, self.num_heads, 1, self.head_dim).to(MODEL_DTYPE)
        key_states = self.k_proj(hidden_states).view(1, self.num_kv_heads, 1, self.head_dim).to(MODEL_DTYPE)
        value_states = self.v_proj(hidden_states).view(1, self.num_kv_heads, 1, self.head_dim).to(MODEL_DTYPE)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = rotary_emb
        query_states, key_states = apply_rotary_pos_emb_single(query_states, key_states, cos, sin)

        return query_states, key_states, value_states

    def get_new_kv_cache_prefill(self, hidden_states, current_pos, rotary_emb):
        """Get new key-value cache entries for prefilling."""
        bsz, seq_len, _ = hidden_states.shape

        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(1, self.num_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)
        key_states = key_states.view(1, self.num_kv_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)
        value_states = value_states.view(1, self.num_kv_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = rotary_emb
        cos = cos.permute(0, 2, 1, 3)
        sin = sin.permute(0, 2, 1, 3)

        query_states, key_states = apply_rotary_pos_emb_prefill(query_states, key_states, cos, sin)

        return query_states.to(MODEL_DTYPE), key_states.to(MODEL_DTYPE), value_states.to(MODEL_DTYPE)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        hs = hidden_states.permute(0, 2, 1).unsqueeze(2)

        query_states = (
            self.q_proj(hs)
            .view(bsz, self.num_heads, self.head_dim, seq_len)
            .permute(0, 1, 3, 2)
        )
        key_states = (
            self.k_proj(hs)
            .view(bsz, self.num_kv_heads, self.head_dim, seq_len)
            .permute(0, 1, 3, 2)
        )
        value_states = (
            self.v_proj(hs)
            .view(bsz, self.num_kv_heads, self.head_dim, seq_len)
            .permute(0, 1, 3, 2)
        )

        n_rep = self.num_heads // self.num_kv_heads
        key_states = self.repeat_kv(key_states.squeeze(0), n_rep)
        value_states = self.repeat_kv(value_states.squeeze(0), n_rep)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        )
        if causal_mask is not None:
            causal_mask_slice = causal_mask[:, :, :seq_len, :seq_len]
            attn_weights = attn_weights + causal_mask_slice.to(attn_weights.dtype)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (
            attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, -1)
        )
        out = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return out.squeeze(2).permute(0, 2, 1)

    def forward_regular(self, hidden_states, query_states, kv_cache_layer=None, causal_mask=None, current_pos=None):
        """Forward pass for single token generation."""
        bsz, q_len, _ = hidden_states.shape

        K_layer_cache, V_layer_cache = kv_cache_layer
        K_layer_cache = K_layer_cache[..., :self.config.state_length, :]
        V_layer_cache = V_layer_cache[..., :self.config.state_length, :]

        n_rep = self.num_heads // self.num_kv_heads
        key_states = self.repeat_kv(K_layer_cache, n_rep)
        value_states = self.repeat_kv(V_layer_cache, n_rep)

        attn_weights = torch.matmul(query_states.to(MODEL_DTYPE), key_states.transpose(-1, -2).to(MODEL_DTYPE)) * self.scale

        if causal_mask is not None:
            q_seq_len = query_states.shape[-2]
            k_seq_len = key_states.shape[-2]
            attn_weights = attn_weights + causal_mask.to(MODEL_DTYPE)[:, :, :q_seq_len, :k_seq_len]

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states.to(MODEL_DTYPE))

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return attn_output.squeeze(2).permute(0, 2, 1)

    def forward_prefill(self, hidden_states, query_states, kv_cache_layer=None, causal_mask=None):
        """Forward pass for prefill mode."""
        bsz, q_len, _ = hidden_states.shape

        K_layer_cache, V_layer_cache = kv_cache_layer
        K_layer_cache = K_layer_cache[..., :self.config.state_length, :]
        V_layer_cache = V_layer_cache[..., :self.config.state_length, :]

        n_rep = self.num_heads // self.num_kv_heads
        key_states = self.repeat_kv(K_layer_cache, n_rep)
        value_states = self.repeat_kv(V_layer_cache, n_rep)

        attn_weights = torch.einsum('bhqd,bhkd->bhqk', query_states.to(MODEL_DTYPE), key_states.to(MODEL_DTYPE)) * self.scale

        if causal_mask is not None:
            q_seq_len = query_states.shape[2]
            k_seq_len = min(key_states.shape[2], self.config.context_length)
            mask_slice = causal_mask.to(MODEL_DTYPE)[:, :, :q_seq_len, :k_seq_len]
            attn_weights = attn_weights + mask_slice

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, value_states.to(MODEL_DTYPE))

        attn_output = attn_output.transpose(1, 2).contiguous()
        actual_bsz, actual_seq_len, num_heads, head_dim = attn_output.shape
        attn_output = attn_output.reshape(actual_bsz, actual_seq_len, num_heads * head_dim)

        attn_output = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return attn_output.squeeze(2).permute(0, 2, 1)


class QwenDecoderLayer(nn.Module):
    """Qwen decoder layer with ANEMLL quantization."""

    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.self_attn = QwenAttention(config)
        self.mlp = QwenMLP(config)
        self.input_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_mask, position_ids, current_pos)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QwenModel(nn.Module):
    """Qwen model with ANEMLL quantization support."""

    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        self.disable_kv_cache = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size).to(TEST_DEVICE)
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        if not hasattr(QwenModel, '_config_printed'):
            print(f"QwenModel (AQ1) using head_dim={self.head_dim}")
            QwenModel._config_printed = True

        if FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
            cache_size = (
                2 * config.num_hidden_layers,
                config.num_key_value_heads,
                config.state_length,
                self.head_dim
            )
            self.register_buffer("kv_cache_0", torch.zeros(cache_size, dtype=MODEL_DTYPE, device=TEST_DEVICE))
            if not hasattr(QwenModel, '_cache_init_printed'):
                print(f"Initialized unified KV kv_cache_0 with shape: {self.kv_cache_0.shape}")
                QwenModel._cache_init_printed = True

    def get_rotary_embeddings_s(self, current_pos):
        """Get rotary embeddings for the current position."""
        sin = self.layers[0].self_attn.rotary_emb.sin_cached[:, current_pos].view(1, 1, 1, -1)
        cos = self.layers[0].self_attn.rotary_emb.cos_cached[:, current_pos].view(1, 1, 1, -1)
        return cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE)

    def get_rotary_embedding_prefill(self, positions):
        """Get rotary embeddings for a sequence of positions."""
        rotary_emb = self.layers[0].self_attn.rotary_emb
        seq_len = positions.size(0)
        cos = rotary_emb.cos_cached[:, positions].view(1, seq_len, 1, rotary_emb.dim)
        sin = rotary_emb.sin_cached[:, positions].view(1, seq_len, 1, rotary_emb.dim)
        return cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE)

    def process_layer_prefill(self, layer_idx, hidden_states, position_ids, causal_mask, current_pos, rotary_emb, layer_offset):
        """Process a single transformer layer in prefill mode."""
        layer = self.layers[layer_idx]

        normalized_states = layer.input_layernorm(hidden_states)

        query_states, key_states, value_states = layer.self_attn.get_new_kv_cache_prefill(
            normalized_states, current_pos, rotary_emb
        )

        group_idx, layer_in_group_idx, layers_per_group = get_kv_cache_idx(layer_idx, self.config.num_hidden_layers)

        if FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
            kv_cache = getattr(self, "kv_cache_0")
        else:
            kv_cache = getattr(self, f"kv_cache_{group_idx}")

        key_idx = layer_in_group_idx
        value_idx = layer_in_group_idx + layers_per_group

        seq_length = key_states.shape[2]
        kv_cache[key_idx:key_idx + 1, :, current_pos:current_pos + seq_length, :] = key_states
        kv_cache[value_idx:value_idx + 1, :, current_pos:current_pos + seq_length, :] = value_states

        key_cache = kv_cache[key_idx:key_idx + 1].squeeze(0)
        value_cache = kv_cache[value_idx:value_idx + 1].squeeze(0)

        attn_output = layer.self_attn.forward_prefill(
            hidden_states=normalized_states,
            query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask,
        )

        hidden_states = hidden_states + attn_output

        post_attn = layer.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + layer.mlp(post_attn)

        return hidden_states

    def process_layer_regular(self, layer_idx, hidden_states, position_ids, causal_mask, current_pos, rotary_emb, layer_offset):
        """Process a single transformer layer in regular mode."""
        layer = self.layers[layer_idx]
        seq_len = hidden_states.shape[1]

        normalized_states = layer.input_layernorm(hidden_states)

        if seq_len == 1:
            query_states, key_states, value_states = layer.self_attn.get_new_kv_cache(
                normalized_states, current_pos, rotary_emb
            )
        else:
            query_states, key_states, value_states = layer.self_attn.get_new_kv_cache_prefill(
                normalized_states, current_pos, rotary_emb
            )

        if not self.disable_kv_cache:
            group_idx, layer_in_group_idx, layers_per_group = get_kv_cache_idx(layer_idx, self.config.num_hidden_layers)

            if FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
                kv_cache = getattr(self, "kv_cache_0")
            else:
                kv_cache = getattr(self, f"kv_cache_{group_idx}")

            key_idx = layer_in_group_idx
            value_idx = layer_in_group_idx + layers_per_group

            if seq_len == 1:
                pos = current_pos
                kv_cache[key_idx:key_idx + 1, :, pos:pos + 1, :] = key_states
                kv_cache[value_idx:value_idx + 1, :, pos:pos + 1, :] = value_states
            else:
                pos = current_pos.item() if isinstance(current_pos, torch.Tensor) else current_pos
                kv_cache[key_idx:key_idx + 1, :, pos:pos + seq_len, :] = key_states
                kv_cache[value_idx:value_idx + 1, :, pos:pos + seq_len, :] = value_states

            key_cache = kv_cache[key_idx:key_idx + 1].squeeze(0)
            value_cache = kv_cache[value_idx:value_idx + 1].squeeze(0)

            if seq_len == 1:
                attn_output = layer.self_attn.forward_regular(
                    hidden_states=normalized_states,
                    query_states=query_states,
                    kv_cache_layer=(key_cache, value_cache),
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                )
            else:
                cache_len = self.config.state_length
                adjusted_causal_mask = torch.zeros((1, 1, seq_len, cache_len), dtype=MODEL_DTYPE, device=TEST_DEVICE)
                pos = current_pos.item() if isinstance(current_pos, torch.Tensor) else current_pos
                for i in range(seq_len):
                    for j in range(pos + i + 1, pos + seq_len):
                        if j < cache_len:
                            adjusted_causal_mask[0, 0, i, j] = float('-inf')

                attn_output = layer.self_attn.forward_prefill(
                    hidden_states=normalized_states,
                    query_states=query_states,
                    kv_cache_layer=(key_cache, value_cache),
                    causal_mask=adjusted_causal_mask,
                )
        else:
            n_rep = layer.self_attn.num_heads // layer.self_attn.num_kv_heads
            fake_key_cache = torch.zeros(
                (layer.self_attn.num_kv_heads, self.config.state_length, layer.self_attn.head_dim),
                dtype=MODEL_DTYPE, device=TEST_DEVICE
            )
            fake_value_cache = torch.zeros(
                (layer.self_attn.num_kv_heads, self.config.state_length, layer.self_attn.head_dim),
                dtype=MODEL_DTYPE, device=TEST_DEVICE
            )

            pos = current_pos.item() if isinstance(current_pos, torch.Tensor) else current_pos
            if seq_len == 1:
                fake_key_cache[:, pos:pos + 1, :] = key_states.squeeze(0)
                fake_value_cache[:, pos:pos + 1, :] = value_states.squeeze(0)
            else:
                fake_key_cache[:, pos:pos + seq_len, :] = key_states.squeeze(0)
                fake_value_cache[:, pos:pos + seq_len, :] = value_states.squeeze(0)

            if seq_len == 1:
                adjusted_causal_mask = causal_mask
            else:
                cache_len = self.config.state_length
                adjusted_causal_mask = torch.zeros((1, 1, seq_len, cache_len), dtype=MODEL_DTYPE, device=TEST_DEVICE)
                for i in range(seq_len):
                    for j in range(pos + i + 1, pos + seq_len):
                        if j < cache_len:
                            adjusted_causal_mask[0, 0, i, j] = float('-inf')

            if seq_len == 1:
                attn_output = layer.self_attn.forward_regular(
                    hidden_states=normalized_states,
                    query_states=query_states,
                    kv_cache_layer=(fake_key_cache, fake_value_cache),
                    causal_mask=adjusted_causal_mask,
                    current_pos=current_pos,
                )
            else:
                attn_output = layer.self_attn.forward_prefill(
                    hidden_states=normalized_states,
                    query_states=query_states,
                    kv_cache_layer=(fake_key_cache, fake_value_cache),
                    causal_mask=adjusted_causal_mask,
                )

        hidden_states = hidden_states + attn_output
        post_attn = layer.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + layer.mlp(post_attn)

        return hidden_states

    def process_layer(self, layer_idx, hidden_states, position_ids, causal_mask, current_pos, rotary_emb, layer_offset, IN_PREFILL=False):
        """Process a single transformer layer."""
        if IN_PREFILL:
            return self.process_layer_prefill(layer_idx, hidden_states, position_ids, causal_mask, current_pos, rotary_emb, layer_offset)
        else:
            return self.process_layer_regular(layer_idx, hidden_states, position_ids, causal_mask, current_pos, rotary_emb, layer_offset)

    def process_layers(self, hidden_states, position_ids, causal_mask, current_pos, rotary_emb, start_layer=0, end_layer=None, IN_PREFILL=False):
        """Process a range of transformer layers."""
        if end_layer is None:
            end_layer = len(self.layers)

        layer_offset = 0 if ENABLE_UNIFIED_CACHE else start_layer

        for i in range(start_layer, end_layer):
            hidden_states = self.process_layer(
                i, hidden_states, position_ids,
                causal_mask, current_pos, rotary_emb, layer_offset, IN_PREFILL
            )
        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
        IN_PREFILL: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the transformer layers."""
        hidden_states = self.embed_tokens(input_ids)

        if IN_PREFILL:
            rotary_emb = self.get_rotary_embedding_prefill(position_ids)
        else:
            rotary_emb = self.get_rotary_embeddings_s(current_pos)

        hidden_states = self.process_layers(
            hidden_states, position_ids, causal_mask,
            current_pos, rotary_emb, start_layer=0, end_layer=None, IN_PREFILL=IN_PREFILL,
        )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def forward_prefill(self, hidden_states, position_ids=None, causal_mask=None, current_pos=None, start_layer=None, end_layer=None):
        """Forward pass for prefilling KV cache."""
        rotary_emb = self.get_rotary_embedding_prefill(position_ids)

        if start_layer is not None and end_layer is not None:
            hidden_states = self.process_layers(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                rotary_emb=rotary_emb,
                start_layer=start_layer,
                end_layer=end_layer,
                IN_PREFILL=True
            )
        else:
            hidden_states = self.process_layers(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                rotary_emb=rotary_emb,
                IN_PREFILL=True
            )

        if end_layer is None or end_layer == len(self.layers):
            hidden_states = self.norm(hidden_states)

        return hidden_states

    def load_pretrained_weights(self, model_path: str) -> bool:
        """Load pretrained weights from standard HuggingFace format.

        This converts standard Conv2d weights to AnemllConv2d format
        with identity scales.
        """
        if not os.path.isdir(model_path):
            raise FileNotFoundError(model_path)

        state_dict: Dict[str, torch.Tensor] = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(
                    safetensors.torch.load_file(os.path.join(model_path, file))
                )

        conv_state = {}
        for k, v in state_dict.items():
            new_k = k.replace("model.", "") if k.startswith("model.") else k
            if "lm_head.weight" in new_k:
                continue
            if any(
                proj in new_k
                for proj in [
                    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
                ]
            ):
                # Reshape for Conv2d format
                conv_state[new_k] = v.view(v.shape[0], v.shape[1], 1, 1)
            else:
                conv_state[new_k] = v

        # Initialize AnemllConv2d scale factors to identity
        for name, module in self.named_modules():
            if isinstance(module, AnemllConv2d):
                from .anemll_quant import _init_scale_identity
                _init_scale_identity(module)

        missing, unexpected = self.load_state_dict(conv_state, strict=False)

        # Filter expected missing keys
        expected_missing = ['kv_cache_0']
        anemll_params = ['scale_A', 'scale_B']
        missing = [m for m in missing if m not in expected_missing and not any(p in m for p in anemll_params)]
        missing = [m for m in missing if "rotary_emb.inv_freq" not in m]

        if missing or unexpected:
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        return not missing and not unexpected

    def load_anemll_checkpoint(self, checkpoint_path: str, verbose: bool = True) -> bool:
        """Load ANEMLL quantized checkpoint with scale factors.

        This loads checkpoints that contain weight + scale_A + scale_B.
        """
        quant_config = self.config.get_quant_config()
        missing, unexpected = load_anemll_weights(
            self, checkpoint_path, quant_config, verbose=verbose
        )
        return len(missing) == 0


class QwenForCausalLM(nn.Module):
    """Qwen Causal LM with ANEMLL-QUANT-1 support."""

    config_class = QwenConfig

    def __init__(self, config: QwenConfig, enable_coreml=False, disable_kv_cache=False, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.enable_coreml = enable_coreml
        self.disable_kv_cache = disable_kv_cache or DISABLE_KV_CACHE

        if enable_coreml:
            global ENABLE_COREML
            ENABLE_COREML = True
            print(f"Set global ENABLE_COREML = {ENABLE_COREML}")

        self.model = QwenModel(config)
        self.model.disable_kv_cache = self.disable_kv_cache

        # Initialize lm_head (not quantized - uses standard Conv2d)
        if ENABLE_CONV2D:
            if ENABLE_VACAB_SPLIT16:
                vocab_split = config.vocab_size // 16
                vocab_remainder = config.vocab_size % 16
                for i in range(16):
                    split_size = vocab_split + (1 if i < vocab_remainder else 0)
                    setattr(self, f"lm_head16_{i + 1}",
                            nn.Conv2d(config.hidden_size, split_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE))
                if not hasattr(QwenForCausalLM, '_lm_head_printed'):
                    print("Created lm_head16_1 through lm_head16_16")
                    QwenForCausalLM._lm_head_printed = True
            elif ENABLE_VACAB_SPLIT8:
                vocab_split = config.vocab_size // 8
                vocab_remainder = config.vocab_size % 8
                for i in range(8):
                    split_size = vocab_split + (1 if i < vocab_remainder else 0)
                    setattr(self, f"lm_head8_{i + 1}",
                            nn.Conv2d(config.hidden_size, split_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE))
                print("Created lm_head8_1 through lm_head8_8")
            elif ENABLE_VACAB_SPLIT:
                self.lm_head2_1 = nn.Conv2d(config.hidden_size, config.vocab_size // 2, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head2_2 = nn.Conv2d(config.hidden_size, config.vocab_size // 2, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                print("Created lm_head2_1 and lm_head2_2")
            else:
                self.lm_head1 = nn.Conv2d(config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                print("Created lm_head1")
        else:
            self.lm_head = nn.Conv2d(config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
            print("Created linear lm_head")

    def forward(
        self,
        input_ids: torch.LongTensor,
        update_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        current_pos: torch.LongTensor,
        IN_PREFILL: bool = False,
    ) -> torch.Tensor:
        assert len(input_ids.shape) == 2, "input_ids must be 2D"

        hidden_states = self.model(
            input_ids, causal_mask, position_ids, current_pos, IN_PREFILL=IN_PREFILL
        )

        if not IN_PREFILL and current_pos is not None:
            seq_len = hidden_states.shape[1]
            if seq_len == 1:
                pos_tensor = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
            else:
                if isinstance(current_pos, torch.Tensor):
                    pos_tensor = current_pos if current_pos.dim() > 0 else current_pos.unsqueeze(0)
                else:
                    pos_tensor = torch.tensor([current_pos], device=hidden_states.device, dtype=torch.long)
            hidden_states = torch.index_select(hidden_states, dim=1, index=pos_tensor)

        if ENABLE_CONV2D:
            hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

            if ENABLE_VACAB_SPLIT16:
                logits_list = []
                for i in range(16):
                    logits_list.append(getattr(self, f"lm_head16_{i + 1}")(hidden_states).squeeze(2).transpose(1, 2))

                if self.enable_coreml and ENABLE_LOGITS2:
                    return tuple(logits_list)
                else:
                    logits = torch.cat(logits_list, dim=2)

            elif ENABLE_VACAB_SPLIT8:
                logits_list = []
                for i in range(8):
                    logits_list.append(getattr(self, f"lm_head8_{i + 1}")(hidden_states).squeeze(2).transpose(1, 2))

                if self.enable_coreml and ENABLE_LOGITS2:
                    return tuple(logits_list)
                else:
                    logits = torch.cat(logits_list, dim=2)

            elif ENABLE_VACAB_SPLIT:
                logits1 = self.lm_head2_1(hidden_states).squeeze(2).transpose(1, 2)
                logits2 = self.lm_head2_2(hidden_states).squeeze(2).transpose(1, 2)

                if self.enable_coreml and ENABLE_LOGITS2:
                    return logits1, logits2

                logits = torch.cat([logits1, logits2], dim=2)
            else:
                logits = self.lm_head1(hidden_states).squeeze(2).transpose(1, 2)
        else:
            logits = self.lm_head(hidden_states.permute(0, 2, 1).unsqueeze(2))
            logits = logits.squeeze(2).permute(0, 2, 1)

        return logits

    def prefill_kv_cache(self, input_ids, position_ids, start_pos, causal_mask):
        """Pre-fills KV cache for a batch of tokens."""
        batch_size, seq_length = input_ids.shape

        hidden_states = self.model.embed_tokens(input_ids)
        hidden_states = hidden_states.to(MODEL_DTYPE)

        if causal_mask is not None:
            causal_mask_prefill = causal_mask[:, :, :seq_length, :]
        else:
            causal_mask_prefill = None

        with torch.no_grad():
            self.model.forward_prefill(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask_prefill,
                current_pos=start_pos
            )

    def load_pretrained_weights(self, model_path: str) -> bool:
        """Load pretrained weights from standard HuggingFace format."""
        if not self.model.load_pretrained_weights(model_path):
            return False

        state_dict: Dict[str, torch.Tensor] = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(
                    safetensors.torch.load_file(os.path.join(model_path, file))
                )

        lm_head_present = False
        embed_tokens_key = None
        for k, v in state_dict.items():
            if k == "lm_head.weight":
                lm_head_present = True
            if "embed_tokens.weight" in k:
                embed_tokens_key = k

        if not lm_head_present:
            print("lm_head.weight not found in the model file dictionary")
            if embed_tokens_key:
                print(f"Using {embed_tokens_key} for lm_head.weight")
                state_dict['lm_head.weight'] = state_dict[embed_tokens_key].clone()
            else:
                print("embed_tokens.weight not found. Unable to set lm_head.weight")
                return False

        lm_head_weight = state_dict.get("lm_head.weight")

        if lm_head_weight is not None:
            if ENABLE_CONV2D:
                reshaped_weight = lm_head_weight.view(lm_head_weight.shape[0], lm_head_weight.shape[1], 1, 1)
                if ENABLE_VACAB_SPLIT16:
                    vocab_split = self.config.vocab_size // 16
                    vocab_remainder = self.config.vocab_size % 16
                    split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(16)]
                    splits = torch.split(reshaped_weight, split_sizes)
                    for i, split in enumerate(splits):
                        getattr(self, f"lm_head16_{i + 1}").weight.data.copy_(split)
                        print(f"Loaded lm_head16_{i + 1}.weight with shape {split.shape}")
                elif ENABLE_VACAB_SPLIT8:
                    vocab_split = self.config.vocab_size // 8
                    vocab_remainder = self.config.vocab_size % 8
                    split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(8)]
                    splits = torch.split(reshaped_weight, split_sizes)
                    for i, split in enumerate(splits):
                        getattr(self, f"lm_head8_{i + 1}").weight.data.copy_(split)
                        print(f"Loaded lm_head8_{i + 1}.weight with shape {split.shape}")
                elif ENABLE_VACAB_SPLIT:
                    vocab_split = self.config.vocab_size // 2
                    split1, split2 = torch.split(reshaped_weight, [vocab_split, self.config.vocab_size - vocab_split])
                    self.lm_head2_1.weight.data.copy_(split1)
                    self.lm_head2_2.weight.data.copy_(split2)
                    print("Loaded lm_head2_1.weight and lm_head2_2.weight")
                else:
                    self.lm_head1.weight.data.copy_(reshaped_weight)
                    print("Loaded lm_head1.weight")
            else:
                self.lm_head.weight.data.copy_(lm_head_weight.view(lm_head_weight.shape[0], lm_head_weight.shape[1], 1, 1))
        else:
            print("Warning: lm_head.weight not found in model weights")
            return False

        return True

    def load_anemll_checkpoint(self, checkpoint_path: str, verbose: bool = True) -> bool:
        """Load ANEMLL quantized checkpoint with scale factors.

        This loads checkpoints that contain weight + scale_A + scale_B
        for the quantized layers.
        """
        return self.model.load_anemll_checkpoint(checkpoint_path, verbose=verbose)

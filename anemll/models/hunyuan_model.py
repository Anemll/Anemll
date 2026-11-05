"""HY-MT2 (HunYuanDenseV1) model with ANE-friendly Conv2d projections.

The architecture uses alpha-NTK rotary embeddings followed by Q/K RMSNorm.
State updates use a position mask and static slices for Core ML/ANE lowering.
"""

from __future__ import annotations

import json
import math
import os

import safetensors.torch
import torch
from torch import nn
from torch.nn import functional as F

MODEL_DTYPE = torch.float16
TEST_DEVICE = "cpu"
CONTEXT_LENGTH = 256
STATE_LENGTH = 256


class HunyuanConfig:
    """Configuration for the dense HY-MT2 checkpoint family."""

    def __init__(self, **kwargs):
        self.architectures = kwargs.get("architectures", ["HunYuanDenseV1ForCausalLM"])
        self.model_type = kwargs.get("model_type", "hunyuan_v1_dense")
        self.hidden_size = kwargs.get("hidden_size", 2048)
        self.intermediate_size = kwargs.get("intermediate_size", 6144)
        self.num_attention_heads = kwargs.get("num_attention_heads", 16)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.head_dim = kwargs.get(
            "attention_head_dim",
            kwargs.get("head_dim", self.hidden_size // self.num_attention_heads),
        )
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-05)
        self.vocab_size = kwargs.get("vocab_size", 120818)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.context_length = kwargs.get("context_length", CONTEXT_LENGTH)
        self.state_length = kwargs.get("state_length", self.context_length)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 8192)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.attention_bias = kwargs.get("attention_bias", False)
        self.use_cla = kwargs.get("use_cla", False)
        self.use_qk_norm = kwargs.get("use_qk_norm", True)
        self.mlp_bias = kwargs.get("mlp_bias", False)
        self.norm_type = kwargs.get("norm_type", "rms")
        self.use_rotary_pos_emb = kwargs.get("use_rotary_pos_emb", True)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.use_cache = kwargs.get("use_cache", True)
        self.rope_scaling = dict(
            kwargs.get("rope_scaling") or kwargs.get("rope_parameters") or {}
        )
        self.rope_theta = kwargs.get(
            "rope_theta", self.rope_scaling.get("rope_theta", 10000.0)
        )
        if self.model_type != "hunyuan_v1_dense":
            raise ValueError(
                "HunyuanConfig supports model_type='hunyuan_v1_dense' only"
            )
        if (
            self.use_cla
            or self.attention_bias
            or self.hidden_act != "silu"
            or not self.use_qk_norm
            or self.mlp_bias
            or not self.use_rotary_pos_emb
            or self.norm_type != "rms"
        ):
            raise ValueError(
                "Unsupported HY-MT2 architecture flags: requires use_cla=false, attention_bias=false, hidden_act='silu', use_qk_norm=true, mlp_bias=false, use_rotary_pos_emb=true, norm_type='rms'"
            )
        if (
            self.head_dim <= 2
            or self.head_dim % 2
            or self.num_attention_heads % self.num_key_value_heads
        ):
            raise ValueError(
                "head_dim must be even and >2; attention heads must be divisible by KV heads"
            )
        if self.context_length > self.state_length:
            raise ValueError("state_length must be at least context_length")
        rope_type = self.rope_scaling.get(
            "type", self.rope_scaling.get("rope_type", "default")
        )
        if rope_type not in ("default", "dynamic"):
            raise ValueError(f"Unsupported HY-MT2 RoPE type: {rope_type}")
        if rope_type == "dynamic" and self.rope_scaling.get("alpha") is None:
            raise ValueError(
                "HY-MT2 dynamic RoPE requires alpha (fixed alpha-NTK scaling)"
            )

    @classmethod
    def from_json(cls, json_file):
        with open(json_file) as f:
            return cls(**json.load(f))


class HunyuanRMSNorm(nn.Module):
    """ANE optimized RMSNorm implementation. We use layer_norm and avoid the mean subtraction.
    The mirrored input makes the LayerNorm mean exactly zero, preserving RMS statistics.
    """

    def __init__(self, hidden_size, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        doubled = torch.cat([x, -x], dim=-1)
        hidden_size = self.hidden_size
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * hidden_size,),
            weight=None,
            bias=None,
            eps=float(self.variance_epsilon),
        )
        normed = normed[..., :hidden_size]
        return normed * self.weight.to(normed.dtype, copy=False).to(
            normed.device, copy=False
        )


class HunyuanRotaryEmbedding(nn.Module):
    """Simple rotary positional embedding."""

    def __init__(self, config: HunyuanConfig) -> None:
        super().__init__()
        self.dim = config.head_dim
        rope_scaling = config.rope_scaling
        base = config.rope_theta
        if (
            rope_scaling.get("type", rope_scaling.get("rope_type")) == "dynamic"
            and rope_scaling.get("alpha") is not None
        ):
            alpha = rope_scaling["alpha"]
            base = config.rope_theta * alpha ** (self.dim / (self.dim - 2))
        inv_freq = 1.0 / base ** (
            torch.arange(0, self.dim, 2).float().to(TEST_DEVICE) / self.dim
        )
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(
            max(config.context_length, config.state_length) * 2, device=TEST_DEVICE
        ).type_as(self.inv_freq)
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
            return (cos, sin)
        else:
            seq_len = x.shape[1]
            return (
                self.cos_cached[:, :seq_len].to(x.dtype),
                self.sin_cached[:, :seq_len].to(x.dtype),
            )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return (q_embed, k_embed)


def apply_rotary_pos_emb_prefill(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return (q_embed, k_embed)


def apply_rotary_pos_emb_single(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return (q_embed, k_embed)


class HunyuanMLP(nn.Module):

    def __init__(self, config: HunyuanConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Conv2d(
            self.hidden_size,
            self.intermediate_size,
            kernel_size=1,
            bias=False,
            dtype=MODEL_DTYPE,
        )
        self.up_proj = nn.Conv2d(
            self.hidden_size,
            self.intermediate_size,
            kernel_size=1,
            bias=False,
            dtype=MODEL_DTYPE,
        )
        self.down_proj = nn.Conv2d(
            self.intermediate_size,
            self.hidden_size,
            kernel_size=1,
            bias=False,
            dtype=MODEL_DTYPE,
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


class HunyuanAttention(nn.Module):

    def __init__(self, config: HunyuanConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_emb = HunyuanRotaryEmbedding(config)
        q_proj_dim = self.num_heads * self.head_dim
        kv_proj_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Conv2d(
            self.hidden_size, q_proj_dim, 1, bias=False, dtype=MODEL_DTYPE
        ).to(TEST_DEVICE)
        self.k_proj = nn.Conv2d(
            self.hidden_size, kv_proj_dim, 1, bias=False, dtype=MODEL_DTYPE
        ).to(TEST_DEVICE)
        self.v_proj = nn.Conv2d(
            self.hidden_size, kv_proj_dim, 1, bias=False, dtype=MODEL_DTYPE
        ).to(TEST_DEVICE)
        self.o_proj = nn.Conv2d(
            q_proj_dim, self.hidden_size, 1, bias=False, dtype=MODEL_DTYPE
        ).to(TEST_DEVICE)
        self.q_norm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.scale = 1 / math.sqrt(self.head_dim)

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key/value heads n_rep times, while keeping the batch dimension intact.
        Input shape: (num_key_value_heads, seqlen, head_dim)
        Output shape: (batch=1, num_attention_heads, seqlen, head_dim)
        """
        x = x.unsqueeze(1)
        x = x.repeat(1, n_rep, 1, 1)
        x = x.flatten(0, 1).unsqueeze(0)
        return x

    def get_new_kv_cache(self, hidden_states, current_pos, rotary_emb):
        """Get new key-value cache entries for single token generation."""
        _bsz, _q_len, _ = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        query_states = (
            self.q_proj(hidden_states)
            .view(1, self.num_heads, 1, self.head_dim)
            .to(MODEL_DTYPE)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(1, self.num_kv_heads, 1, self.head_dim)
            .to(MODEL_DTYPE)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(1, self.num_kv_heads, 1, self.head_dim)
            .to(MODEL_DTYPE)
        )
        cos, sin = rotary_emb
        query_states, key_states = apply_rotary_pos_emb_single(
            query_states, key_states, cos, sin
        )
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        return (query_states, key_states, value_states)

    def get_new_kv_cache_prefill(self, hidden_states, current_pos, rotary_emb):
        """Get new key-value cache entries optimized for prefilling with batched tokens."""
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(1, self.num_heads, self.head_dim, -1).permute(
            0, 1, 3, 2
        )
        key_states = key_states.view(1, self.num_kv_heads, self.head_dim, -1).permute(
            0, 1, 3, 2
        )
        value_states = value_states.view(
            1, self.num_kv_heads, self.head_dim, -1
        ).permute(0, 1, 3, 2)
        cos, sin = rotary_emb
        cos = cos.permute(0, 2, 1, 3)
        sin = sin.permute(0, 2, 1, 3)
        query_states, key_states = apply_rotary_pos_emb_prefill(
            query_states, key_states, cos, sin
        )
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        return (
            query_states.to(MODEL_DTYPE),
            key_states.to(MODEL_DTYPE),
            value_states.to(MODEL_DTYPE),
        )

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
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
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

    def forward_regular(
        self,
        hidden_states,
        query_states,
        kv_cache_layer=None,
        causal_mask=None,
        current_pos=None,
    ):
        """Forward pass for single token generation."""
        _bsz, _q_len, _ = hidden_states.shape
        K_layer_cache, V_layer_cache = kv_cache_layer
        n_rep = self.num_heads // self.num_kv_heads
        q_grouped = query_states.to(MODEL_DTYPE).reshape(
            self.num_kv_heads, n_rep, self.head_dim
        )
        if causal_mask is not None:
            ane_zero = causal_mask.to(MODEL_DTYPE)[0, 0, 0, 0]
            K_mat = K_layer_cache.to(MODEL_DTYPE) + ane_zero
            V_mat = V_layer_cache.to(MODEL_DTYPE) + ane_zero
        else:
            K_mat = K_layer_cache.to(MODEL_DTYPE)
            V_mat = V_layer_cache.to(MODEL_DTYPE)
        attn_weights = torch.matmul(q_grouped, K_mat.transpose(-1, -2)) * self.scale
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask.to(MODEL_DTYPE)[0]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V_mat)
        attn_output = attn_output.reshape(1, 1, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return attn_output.squeeze(2).permute(0, 2, 1)

    def forward_prefill(
        self, hidden_states, query_states, kv_cache_layer=None, causal_mask=None
    ):
        """Forward pass for prefill mode"""
        K_layer_cache, V_layer_cache = kv_cache_layer
        n_rep = self.num_heads // self.num_kv_heads
        key_states = self.repeat_kv(K_layer_cache, n_rep)
        value_states = self.repeat_kv(V_layer_cache, n_rep)
        attn_weights = (
            torch.einsum(
                "bhqd,bhkd->bhqk",
                query_states.to(MODEL_DTYPE),
                key_states.to(MODEL_DTYPE),
            )
            * self.scale
        )
        if causal_mask is not None:
            k_seq_len = min(self.config.state_length, self.config.context_length)
            mask_slice = causal_mask.to(MODEL_DTYPE)[:, :, :, :k_seq_len]
            attn_weights = attn_weights + mask_slice
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum(
            "bhqk,bhkd->bhqd", attn_weights, value_states.to(MODEL_DTYPE)
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(1, -1, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return attn_output.squeeze(2).permute(0, 2, 1)


class HunyuanDecoderLayer(nn.Module):

    def __init__(self, config: HunyuanConfig) -> None:
        super().__init__()
        self.self_attn = HunyuanAttention(config)
        self.mlp = HunyuanMLP(config)
        self.input_layernorm = HunyuanRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = HunyuanRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, causal_mask, position_ids, current_pos
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HunyuanModel(nn.Module):
    """Dense transformer with a unified, stateful KV cache."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=MODEL_DTYPE
        )
        self.layers = nn.ModuleList(
            [HunyuanDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head_dim = config.head_dim
        self.register_buffer(
            "kv_cache_0",
            torch.zeros(
                2 * config.num_hidden_layers,
                config.num_key_value_heads,
                config.state_length,
                config.head_dim,
                dtype=MODEL_DTYPE,
            ),
        )

    def get_rotary_embeddings_s(self, current_pos):
        rotary = self.layers[0].self_attn.rotary_emb
        cos = rotary.cos_cached[:, current_pos].view(1, 1, 1, -1)
        sin = rotary.sin_cached[:, current_pos].view(1, 1, 1, -1)
        return (cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE))

    def get_rotary_embedding_prefill(self, positions):
        rotary = self.layers[0].self_attn.rotary_emb
        return (
            rotary.cos_cached[:, positions].unsqueeze(2).to(MODEL_DTYPE),
            rotary.sin_cached[:, positions].unsqueeze(2).to(MODEL_DTYPE),
        )

    def process_layer(
        self,
        layer_idx,
        hidden_states,
        position_ids,
        causal_mask,
        current_pos,
        rotary_emb,
        layer_offset=0,
        IN_PREFILL=False,
    ):
        layer = self.layers[layer_idx]
        normalized = layer.input_layernorm(hidden_states)
        if IN_PREFILL:
            query, key, value = layer.self_attn.get_new_kv_cache_prefill(
                normalized, current_pos, rotary_emb
            )
            length = (
                self.config.prefill_batch_size
                if torch.jit.is_tracing()
                else hidden_states.shape[1]
            )
            if not torch.jit.is_tracing():
                start = int(current_pos)
                if start < 0 or start + length > self.config.context_length:
                    raise ValueError(
                        "Prefill batch exceeds the configured context window"
                    )
            # Static position-mask writes preserve the runtime start offset.
            # The matrix [context, batch] places each incoming KV row without
            # a dynamic state slice or a rank-5 intermediate.
            # A runtime position mask avoids data-dependent state slice bounds.
            # Only the context window is written: --state-pad leaves the final
            # row outside every state write for the ANE compiler workaround.
            positions = torch.arange(
                self.config.context_length, device=hidden_states.device
            )
            incoming = torch.arange(length, device=hidden_states.device) + current_pos
            placement = (positions.unsqueeze(1) == incoming.unsqueeze(0)).to(
                MODEL_DTYPE
            )
            mask = placement.sum(dim=1).view(1, 1, -1, 1)
            key_rows = torch.matmul(placement, key.squeeze(0)).unsqueeze(0)
            value_rows = torch.matmul(placement, value.squeeze(0)).unsqueeze(0)
            value_idx = layer_idx + self.config.num_hidden_layers
            old_key = self.kv_cache_0[
                layer_idx : layer_idx + 1, :, : self.config.context_length, :
            ]
            key_cache = old_key * (1 - mask) + key_rows
            self.kv_cache_0[
                layer_idx : layer_idx + 1, :, : self.config.context_length, :
            ] = key_cache
            old_value = self.kv_cache_0[
                value_idx : value_idx + 1, :, : self.config.context_length, :
            ]
            value_cache = old_value * (1 - mask) + value_rows
            self.kv_cache_0[
                value_idx : value_idx + 1, :, : self.config.context_length, :
            ] = value_cache
            key_cache = key_cache.squeeze(0)
            value_cache = value_cache.squeeze(0)
            attention = layer.self_attn.forward_prefill(
                normalized, query, (key_cache, value_cache), causal_mask
            )
        else:
            query, key, value = layer.self_attn.get_new_kv_cache(
                normalized, current_pos, rotary_emb
            )
            # A runtime position mask avoids data-dependent state slice bounds.
            # Only the context window is written: --state-pad leaves the final
            # row outside every state write for the ANE compiler workaround.
            positions = torch.arange(
                self.config.context_length, device=hidden_states.device
            )
            mask = (positions == current_pos).to(MODEL_DTYPE).view(1, 1, -1, 1)
            value_idx = layer_idx + self.config.num_hidden_layers
            old_key = self.kv_cache_0[
                layer_idx : layer_idx + 1, :, : self.config.context_length, :
            ]
            key_cache = old_key * (1 - mask) + key * mask
            self.kv_cache_0[
                layer_idx : layer_idx + 1, :, : self.config.context_length, :
            ] = key_cache
            old_value = self.kv_cache_0[
                value_idx : value_idx + 1, :, : self.config.context_length, :
            ]
            value_cache = old_value * (1 - mask) + value * mask
            self.kv_cache_0[
                value_idx : value_idx + 1, :, : self.config.context_length, :
            ] = value_cache
            attention = layer.self_attn.forward_regular(
                normalized,
                query,
                (key_cache.squeeze(0), value_cache.squeeze(0)),
                causal_mask,
                current_pos,
            )
        hidden_states = hidden_states + attention
        return hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

    def process_layers(
        self,
        hidden_states,
        position_ids,
        causal_mask,
        current_pos,
        rotary_emb,
        start_layer=0,
        end_layer=None,
        IN_PREFILL=False,
    ):
        end_layer = len(self.layers) if end_layer is None else end_layer
        for index in range(start_layer, end_layer):
            hidden_states = self.process_layer(
                index,
                hidden_states,
                position_ids,
                causal_mask,
                current_pos,
                rotary_emb,
                IN_PREFILL=IN_PREFILL,
            )
        return hidden_states

    def forward(
        self, input_ids, causal_mask, position_ids, current_pos, IN_PREFILL=False
    ):
        hidden_states = self.embed_tokens(input_ids)
        rotary = (
            self.get_rotary_embedding_prefill(position_ids)
            if IN_PREFILL
            else self.get_rotary_embeddings_s(current_pos)
        )
        return self.norm(
            self.process_layers(
                hidden_states,
                position_ids,
                causal_mask,
                current_pos,
                rotary,
                IN_PREFILL=IN_PREFILL,
            )
        )

    def forward_prefill(
        self,
        hidden_states,
        position_ids=None,
        causal_mask=None,
        current_pos=None,
        start_layer=None,
        end_layer=None,
    ):
        rotary = self.get_rotary_embedding_prefill(position_ids)
        hidden_states = self.process_layers(
            hidden_states,
            position_ids,
            causal_mask,
            current_pos,
            rotary,
            start_layer=start_layer or 0,
            end_layer=end_layer,
            IN_PREFILL=True,
        )
        if end_layer is None or end_layer == len(self.layers):
            hidden_states = self.norm(hidden_states)
        return hidden_states


class HunyuanForCausalLM(nn.Module):
    config_class = HunyuanConfig

    def __init__(self, config, enable_coreml=False, **kwargs):
        super().__init__()
        self.config = config
        self.enable_coreml = enable_coreml
        self.model = HunyuanModel(config)
        for index in range(16):
            size = config.vocab_size // 16 + (index < config.vocab_size % 16)
            setattr(
                self,
                f"lm_head16_{index + 1}",
                nn.Conv2d(config.hidden_size, size, 1, bias=False, dtype=MODEL_DTYPE),
            )

    def forward(
        self,
        input_ids,
        update_mask,
        position_ids,
        causal_mask,
        current_pos,
        IN_PREFILL=False,
    ):
        hidden_states = self.model(
            input_ids, causal_mask, position_ids, current_pos, IN_PREFILL
        )
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)
        logits = tuple(
            getattr(self, f"lm_head16_{index + 1}")(hidden_states)
            .squeeze(2)
            .transpose(1, 2)
            for index in range(16)
        )
        return logits if self.enable_coreml else torch.cat(logits, dim=-1)

    def load_pretrained_weights(self, model_path):
        state = {}
        for path in sorted(os.listdir(model_path)):
            if path.endswith(".safetensors"):
                state.update(
                    safetensors.torch.load_file(os.path.join(model_path, path))
                )
        if not state:
            raise ValueError(f"No safetensors weights found in {model_path}")
        head = state.pop("lm_head.weight", None)
        if head is None:
            if not self.config.tie_word_embeddings:
                raise ValueError("Missing lm_head.weight for an untied checkpoint")
            head = state["model.embed_tokens.weight"]
        converted = {}
        for name, weight in state.items():
            name = name.replace(".query_layernorm.", ".q_norm.").replace(
                ".key_layernorm.", ".k_norm."
            )
            if name.endswith(
                (
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                )
            ):
                weight = weight.unsqueeze(-1).unsqueeze(-1)
            converted[name] = weight
        split_sizes = [
            getattr(self, f"lm_head16_{index + 1}").out_channels for index in range(16)
        ]
        for index, weight in enumerate(head.split(split_sizes)):
            converted[f"lm_head16_{index + 1}.weight"] = weight.unsqueeze(-1).unsqueeze(
                -1
            )
        missing, unexpected = self.load_state_dict(converted, strict=False)
        missing = [
            name
            for name in missing
            if name != "model.kv_cache_0" and (not name.endswith("rotary_emb.inv_freq"))
        ]
        unexpected = [
            name for name in unexpected if not name.endswith("rotary_emb.inv_freq")
        ]
        if missing or unexpected:
            raise ValueError(
                f"Checkpoint weight mismatch: missing={missing}, unexpected={unexpected}"
            )
        return True

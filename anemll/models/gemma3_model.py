"""Gemma3 model implementation for ANEMLL.

This module provides an ANE-optimized implementation of the Gemma3 architecture
for Apple Neural Engine. All dense layers are expressed as ``nn.Conv2d`` with
``kernel_size=1`` and weights are loaded from Hugging Face checkpoints with
correct reshaping.

Key Gemma3 architecture features:
- Interleaved sliding window (512) and full attention at layers 6, 12, 18
- Dual RoPE bases: rope_theta (1e6) for global, rope_local_base_freq (10k) for local
- Per-head Q/K normalization before attention
- GEGLU activation (GELU with tanh approximation)
- Large vocabulary (262,144 tokens) with 16-way LM head splitting
- Gemma-style RMSNorm with (1 + weight) scaling
- Embedding scaling by sqrt(hidden_size)
"""

from __future__ import annotations

import os
import json
import math
from typing import Dict

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Gemma3 270M model implementation adapted from llama_model.py
# ---------------------------------------------------------------------------

MODEL_DTYPE = torch.float16  # Match HF model dtype
TEST_DEVICE = "cpu"
CONTEXT_LENGTH = 256

# Cache configuration constants (following llama_model.py pattern)
FORCE_UNIFIED_CACHE = False  # Deprecated: Use ENABLE_SPLIT_CACHE instead
ENABLE_UNIFIED_CACHE = False  # Deprecated: Use ENABLE_SPLIT_CACHE instead
ENABLE_SPLIT_CACHE = True  # Use separate caches for local/global attention layers
STATE_LENGTH = CONTEXT_LENGTH   # match config.state_length by default
DISABLE_KV_CACHE = False  # Disable KV cache for simple testing

# Layer type constants for Gemma3 split cache
# NOTE: These defaults are for Gemma3-270M. Larger models will have different patterns.
# The actual layer types are now read from config.layer_types at runtime.
# These constants are kept for backwards compatibility but should not be used directly.
_DEFAULT_GLOBAL_ATTENTION_LAYER_INDICES = [5, 11, 17]  # 0-indexed layers with full attention (270M model)
_DEFAULT_NUM_GLOBAL_LAYERS = 3
_DEFAULT_NUM_LOCAL_LAYERS = 15  # 18 total - 3 global = 15 local

# Backward compatibility aliases (deprecated - use config.layer_types instead)
GLOBAL_ATTENTION_LAYER_INDICES = _DEFAULT_GLOBAL_ATTENTION_LAYER_INDICES
NUM_GLOBAL_LAYERS = _DEFAULT_NUM_GLOBAL_LAYERS
NUM_LOCAL_LAYERS = _DEFAULT_NUM_LOCAL_LAYERS

# LM head configuration constants (following llama_model.py pattern)
ENABLE_CONV2D = bool(1)      # Use Conv2d for LM head
ENABLE_VACAB_SPLIT = bool(1)  # Split vocab into 2 parts
ENABLE_VACAB_SPLIT8 = bool(0)  # Split vocab into 8 parts
ENABLE_VACAB_SPLIT16 = bool(1)  # Split vocab into 16 parts
ENABLE_LOGITS2 = bool(1)    # Return separate logits arrays for CoreML
ENABLE_COREML = bool(0)     # CoreML-specific returns


class Gemma3Config:
    def __init__(self, **kwargs):
        self.architectures = kwargs.get("architectures", ["Gemma3ForCausalLM"])
        self.attention_bias = kwargs.get("attention_bias", False)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)

        # Tokenizer / specials
        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.bos_token_id = kwargs.get("bos_token_id", 2)
        self.eos_token_id = kwargs.get("eos_token_id", 1)
        self.eos_token_ids = kwargs.get("eos_token_ids", [1, 106])

        # Geometry
        self.hidden_act = kwargs.get("hidden_act", "gelu_pytorch_tanh")
        self.hidden_size = kwargs.get("hidden_size", 640)
        self.num_attention_heads = kwargs.get("num_attention_heads", 4)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 18)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 1)
        self.head_dim = kwargs.get("head_dim", 256)
        self.intermediate_size = kwargs.get("intermediate_size", 2048)

        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1_000_000.0)
        self.rope_local_base_freq = kwargs.get("rope_local_base_freq", 10_000.0)
        self.query_pre_attn_scalar = kwargs.get("query_pre_attn_scalar", 256)

        self.model_type = kwargs.get("model_type", "gemma3")
        self.transformers_version = kwargs.get("transformers_version", "4.55.2")

        # Context / cache
        self.context_length = kwargs.get("context_length", 256)
        self.state_length = kwargs.get("state_length", self.context_length)

        # FP16 residual clamping (OFF by default - use weight scaling instead)
        # See docs/GEMMA3_FP16_SCALING.md for details
        self.enable_residual_clamp = kwargs.get("enable_residual_clamp", False)
        self.residual_clamp_value = kwargs.get("residual_clamp_value", 65504.0)

        # Interleaved attention (Gemma3 architectural feature)
        # layer_types should be read from HuggingFace config.json which defines
        # the exact pattern for each model size (270M, 1B, 4B, 12B, 27B)
        self.sliding_window = kwargs.get("sliding_window", 512)

        if "layer_types" in kwargs:
            # Use layer_types from config (preferred - supports all model sizes)
            self.layer_types = kwargs["layer_types"]
        else:
            # Fallback: compute default pattern based on model architecture
            # Gemma3 pattern: global attention every 6 layers, starting at layer 5 (0-indexed)
            # This works for 270M (18 layers) and should approximate larger models
            default_layer_types = ["sliding_attention"] * self.num_hidden_layers
            global_interval = 6  # Global attention every 6 layers
            first_global_layer = 5  # First global layer is at index 5 (0-indexed)
            for i in range(first_global_layer, self.num_hidden_layers, global_interval):
                default_layer_types[i] = "full_attention"
            self.layer_types = default_layer_types
            # Warn that we're using computed defaults
            print(f"WARNING: layer_types not in config, using computed defaults for {self.num_hidden_layers} layers")
            print(f"  Global attention at: {[i for i, t in enumerate(default_layer_types) if t == 'full_attention']}")

        # Attention window for ANE optimization (separate from sliding_window)
        # Controls: KV cache size (state_length), causal_mask dimension, KV slice
        self.attention_size = kwargs.get("attention_size", self.state_length)

        # Split cache configuration for local/global attention
        self.use_split_cache = kwargs.get("use_split_cache", ENABLE_SPLIT_CACHE)
        # Force unified cache (single KV state) for testing / compatibility
        self.single_cache = kwargs.get("single_cache", False)
        if self.single_cache:
            self.use_split_cache = False

        # Cache fill direction for local cache
        # False (default): Fill from left (standard), requires conditional for rotation
        # True: Fill from right (ANE-friendly), always shifts, but needs adjusted mask
        self.use_right_fill_cache = kwargs.get("use_right_fill_cache", False)

        # Force rotation mode for local cache (used for CoreML tracing)
        # None (default): Use conditional logic (pos < sliding_window -> fill, else -> rotate)
        # False: Always use fill mode (for infer function, pos < sliding_window)
        # True: Always use rotate mode (for infer_rotate function, pos >= sliding_window)
        self.force_rotation_mode = kwargs.get("force_rotation_mode", None)

        # Vocab
        self.vocab_size = kwargs.get("vocab_size", 262_144)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)

        self.use_cache = kwargs.get("use_cache", True)
        self.context_length = kwargs.get("context_length", 256)

        # Batch size for prefill (used in rotation operations)
        self.batch_size = kwargs.get("batch_size", 64)

        # Experimental: use dynamic slicing for prefill KV writes instead of update_mask.
        # This is not ANE-safe; intended only for testing.
        self.prefill_dynamic_slice = kwargs.get("prefill_dynamic_slice", False)

    def get_global_layer_indices(self) -> list:
        """Get indices of global (full attention) layers from config.layer_types."""
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    def get_local_layer_indices(self) -> list:
        """Get indices of local (sliding attention) layers from config.layer_types."""
        return [i for i, t in enumerate(self.layer_types) if t == "sliding_attention"]

    def get_num_global_layers(self) -> int:
        """Get count of global (full attention) layers."""
        return sum(1 for t in self.layer_types if t == "full_attention")

    def get_num_local_layers(self) -> int:
        """Get count of local (sliding attention) layers."""
        return sum(1 for t in self.layer_types if t == "sliding_attention")

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        # Newer Gemma3 configs (e.g., multimodal) nest text params under text_config
        if "text_config" in config_dict:
            text_cfg = dict(config_dict.get("text_config", {}))
            # Map HF field names to this implementation
            if "hidden_activation" in text_cfg and "hidden_act" not in text_cfg:
                text_cfg["hidden_act"] = text_cfg["hidden_activation"]
            # Carry over token IDs if defined at top-level
            for key in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id", "eos_token_ids"):
                if key in config_dict and key not in text_cfg:
                    text_cfg[key] = config_dict[key]
            config_dict = text_cfg
        return cls(**config_dict)


def get_kv_cache_idx(layer_idx, num_layers, num_groups=1):
    """Helper function to get KV cache indices."""
    layers_per_group = num_layers // num_groups
    group_idx = layer_idx // layers_per_group
    layer_in_group_idx = layer_idx % layers_per_group
    return group_idx, layer_in_group_idx, layers_per_group


def get_layer_cache_mapping(layer_idx: int, layer_types: list) -> tuple:
    """
    Map layer index to cache type and index within that cache.

    For split cache architecture:
    - Local (sliding window) layers: map to kv_cache_local
    - Global (full attention) layers: map to kv_cache_global

    Args:
        layer_idx: The original layer index (0-17 for Gemma3)
        layer_types: List of layer types from config.layer_types

    Returns:
        Tuple of (cache_type, cache_index) where:
        - cache_type is 'global' or 'local'
        - cache_index is the layer's position within that cache (0-based)
    """
    if layer_types[layer_idx] == "full_attention":
        # Global layers: count full_attention layers before this one
        # This works for any model size with any global layer pattern
        global_idx = sum(1 for i in range(layer_idx)
                        if layer_types[i] == "full_attention")
        return 'global', global_idx
    else:
        # Local layers: count sliding_attention layers before this one
        local_idx = sum(1 for i in range(layer_idx)
                       if layer_types[i] == "sliding_attention")
        return 'local', local_idx


# -----------------------------------------------------------------------------
# Gemma3 building blocks
# -----------------------------------------------------------------------------


class Gemma3RMSNorm(nn.Module):
    """ANE optimized RMSNorm implementation. We use layer_norm and avoid the mean subtraction.
    This give us the best quality for Boolq and other benchmarks."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # Gemma-style: learned offset around 1 → initialize to 0 and add in forward
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    
        # ──────────────────────────────────────────────────────────────────────
        # Compatibility path for PyTorch 1.x / 2.0–2.3                           .
        # We build a tensor whose mean is *exactly* zero so that LayerNorm's
        # mean‑subtraction becomes a no‑op and we recover RMS statistics:
        #
        #     concat([x, ‑x])  →  μ = 0,
        #                        σ² = ½(‖x‖²) = mean(x²)
        # ──────────────────────────────────────────────────────────────────────
        x = hidden_states

        # ❶ Make the last‑dimension mean zero.
        doubled = torch.cat([x, -x], dim=-1)

        hidden_size = hidden_states.shape[-1]
        # ❷ Run the highly‑optimised LayerNorm kernel on the doubled tensor.
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * hidden_size,),
            weight=None,          # no affine factors here
            bias=None,
            eps=float(self.variance_epsilon)
        )

        # ❸ Drop the mirror half → correct RMS‑normed activations.
        normed = normed[..., : hidden_size]

        # ❹ Apply Gemma-style gain: (1 + w)
        return (normed * (1.0 + self.weight
                       .to(normed.dtype, copy=False)
                       .to(normed.device, copy=False)))

class Gemma3HeadNorm(nn.Module):

    """ANE optimized RMSNorm implementation. We use layer_norm and avoid the mean subtraction.
    This give us the best quality for Boolq and other benchmarks."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # Gemma-style: learned offset around 1 → initialize to 0 and add in forward
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    
        # ──────────────────────────────────────────────────────────────────────
        # Compatibility path for PyTorch 1.x / 2.0–2.3                           .
        # We build a tensor whose mean is *exactly* zero so that LayerNorm's
        # mean‑subtraction becomes a no‑op and we recover RMS statistics:
        #
        #     concat([x, ‑x])  →  μ = 0,
        #                        σ² = ½(‖x‖²) = mean(x²)
        # ──────────────────────────────────────────────────────────────────────
        x = hidden_states

        # ❶ Make the last‑dimension mean zero.
        doubled = torch.cat([x, -x], dim=-1)

        hidden_size = hidden_states.shape[-1]
        # ❷ Run the highly‑optimised LayerNorm kernel on the doubled tensor.
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * hidden_size,),
            weight=None,          # no affine factors here
            bias=None,
            eps=float(self.variance_epsilon)
        )

        # ❸ Drop the mirror half → correct RMS‑normed activations.
        normed = normed[..., : hidden_size]

        # ❹ Apply Gemma-style gain: (1 + w)
        return (normed * (1.0 + self.weight
                       .to(normed.dtype, copy=False)
                       .to(normed.device, copy=False)))

class Gemma3RotaryEmbedding(nn.Module):
    """Simple rotary positional embedding."""

    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.dim = config.head_dim
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.dim, 2).float().to(TEST_DEVICE) / self.dim)
        )
        #inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(TEST_DEVICE) / self.dim))

        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max(config.context_length, config.state_length)*2, device=TEST_DEVICE).type_as(self.inv_freq)
      
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().unsqueeze(0)
        self.sin_cached = emb.sin().unsqueeze(0)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor | None = None):
        if position_ids is not None:
            # Handle both 1D and 2D position_ids
            if position_ids.dim() == 1:
                pos_ids = position_ids
            else:
                pos_ids = position_ids.squeeze(0)  # Remove batch dimension if present
            
            # Use actual position IDs for correct rotary embeddings
            cos = self.cos_cached[:, pos_ids].to(x.dtype)  # [1, seq_len, head_dim]
            sin = self.sin_cached[:, pos_ids].to(x.dtype)  # [1, seq_len, head_dim]
            return cos, sin
        else:
            # Fallback to sequential positions from 0
            seq_len = x.shape[1]
            return (
                self.cos_cached[:, :seq_len].to(x.dtype),
                self.sin_cached[:, :seq_len].to(x.dtype),
            )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
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
    # For prefill, cos/sin already have the correct shape [1, 1, seq_len, head_dim]
    # No need to unsqueeze - just apply directly
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_single(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # For single token generation, cos/sin already have shape [1, 1, 1, head_dim]
    # No need to unsqueeze - just apply directly 
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, n_kv, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].repeat(1, 1, n_rep, 1, 1)
    return hidden_states.view(bsz, n_kv * n_rep, seq_len, head_dim)


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Use single Conv2d layers (no splitting for Gemma3 for now)
        self.gate_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, kernel_size=1, bias=False, dtype=MODEL_DTYPE)
        self.up_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, kernel_size=1, bias=False, dtype=MODEL_DTYPE)
        self.down_proj = nn.Conv2d(self.intermediate_size, self.hidden_size, kernel_size=1, bias=False, dtype=MODEL_DTYPE)

        # GEGLU: gate uses GELU(tanh); keep the same three-projection GLU wiring
        def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))
        self.act_fn = gelu_tanh

    def forward(self, x):
        # Use identical step-by-step computation to LlamaMLP to prevent numerical explosion
        # Weights stay FP16, computation dtype follows input (Conv2d handles conversion)
        x = x.permute(0, 2, 1).unsqueeze(2)  # Reshape for Conv2d: [bsz, hidden, 1, seq]

        # Step-by-step computation for numerical stability (like LlamaMLP)
        a = self.gate_proj(x)          # gate projection
        b = self.up_proj(x)            # up projection
        d = self.act_fn(a) * b         # GEGLU: GELU(tanh)(gate) ⊙ up
        e = self.down_proj(d)          # down projection

        return e.squeeze(2).permute(0, 2, 1)  # Final output shape: [bsz, seq_len, hidden_size]


class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        if not hasattr(Gemma3Attention, '_config_printed'):
            print(f"Gemma3Attention using head_dim={self.head_dim} (from config: {getattr(config, 'head_dim', 'not set')})")
            print(f"Gemma3Attention projection dims: Q={self.num_heads * self.head_dim}, K/V={self.num_kv_heads * self.head_dim}")
            Gemma3Attention._config_printed = True
            
        # Two RoPE bases: global uses rope_theta (default 1e6), local uses rope_local_base_freq (10k)
        # We'll store the layer type on the attention module for easy access
        self.layer_type = None  # Will be set by DecoderLayer

        # Calculate correct projection dimensions
        q_proj_dim = self.num_heads * self.head_dim  # 16 * 128 = 2048
        kv_proj_dim = self.num_kv_heads * self.head_dim  # 8 * 128 = 1024
        
        self.q_proj = nn.Conv2d(
            self.hidden_size,
            q_proj_dim,
            1,
            bias=False,
            dtype=MODEL_DTYPE,
        ).to(TEST_DEVICE)
        self.k_proj = nn.Conv2d(
            self.hidden_size,
            kv_proj_dim,
            1,
            bias=False,
            dtype=MODEL_DTYPE,
        ).to(TEST_DEVICE)
        self.v_proj = nn.Conv2d(
            self.hidden_size,
            kv_proj_dim,
            1,
            bias=False,
            dtype=MODEL_DTYPE,
        ).to(TEST_DEVICE)
        self.o_proj = nn.Conv2d(
            q_proj_dim,
            self.hidden_size,
            1,
            bias=False,
            dtype=MODEL_DTYPE,
        ).to(TEST_DEVICE)
        self.q_norm = Gemma3HeadNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3HeadNorm(self.head_dim, eps=config.rms_norm_eps)
        self.scale = 1 / math.sqrt(getattr(config, "query_pre_attn_scalar", self.head_dim))

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key/value heads n_rep times, while keeping the batch dimension intact.
        Input shape: (num_key_value_heads, seqlen, head_dim)
        Output shape: (batch=1, num_attention_heads, seqlen, head_dim)
        """
        x = x.unsqueeze(1)  # Shape: (num_kv_heads, 1, seq_len, head_dim)
        x = x.repeat(1, n_rep, 1, 1)  # Shape: (num_kv_heads, n_rep, seq_len, head_dim)
        x = x.view(1, -1, x.size(-2), x.size(-1))  # Shape: (1, num_kv_heads * n_rep, seq_len, head_dim)
        return x

    def get_new_kv_cache(self, hidden_states, current_pos, rotary_emb):
        """Get new key-value cache entries for single token generation."""
        bsz, q_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Project QKV and ensure MODEL_DTYPE
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        
        # Perform projections with fixed dimensions
        query_states = self.q_proj(hidden_states).view(1, self.num_heads, 1, self.head_dim).to(MODEL_DTYPE)
        key_states = self.k_proj(hidden_states).view(1, self.num_kv_heads, 1, self.head_dim).to(MODEL_DTYPE)
        value_states = self.v_proj(hidden_states).view(1, self.num_kv_heads, 1, self.head_dim).to(MODEL_DTYPE)
        
        # Apply query and key normalization (critical for Gemma3!)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        # Use provided rotary embeddings (single token version)
        cos, sin = rotary_emb
        query_states, key_states = apply_rotary_pos_emb_single(query_states, key_states, cos, sin)
        
        return query_states, key_states, value_states

    def get_new_kv_cache_prefill(self, hidden_states, current_pos, rotary_emb):
        """Get new key-value cache entries optimized for prefilling with batched tokens."""
        bsz, seq_len, _ = hidden_states.shape # [1, seq_len, hidden_size]
        device = hidden_states.device
        
        # Project QKV and ensure MODEL_DTYPE - optimized for batch processing
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)  # [1, hidden_size, 1, seq_len]

        # Project all tokens at once using Conv2d
        query_states = self.q_proj(hidden_states)  # [1, num_heads * head_dim, 1, seq_len]
        key_states = self.k_proj(hidden_states)    # [1, num_kv_heads * head_dim, 1, seq_len]
        value_states = self.v_proj(hidden_states)  # [1, num_kv_heads * head_dim, 1, seq_len]

        # Reshape to final dimensions
        query_states = query_states.view(1, self.num_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)  # [1, num_heads, seq_len, head_dim]
        key_states = key_states.view(1, self.num_kv_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)  # [1, num_kv_heads, seq_len, head_dim]
        value_states = value_states.view(1, self.num_kv_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)  # [1, num_kv_heads, seq_len, head_dim]

        # Apply query and key normalization (critical for Gemma3!)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Get rotary embeddings for all positions at once
        cos, sin = rotary_emb
        cos = cos.permute(0, 2, 1, 3)  # [1, 1, seq_len, head_dim]
        sin = sin.permute(0, 2, 1, 3)  # [1, 1, seq_len, head_dim]

        # Apply rotary embeddings to all positions at once (cos/sin already have correct dims for prefill)
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
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        )
        if causal_mask is not None:
            # Slice causal mask to match seq_len x seq_len for attention weights
            causal_mask_slice = causal_mask[:, :, :seq_len, :seq_len]
            attn_weights = attn_weights + causal_mask_slice.to(attn_weights.dtype)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (
            attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, -1)
        )
        out = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return out.squeeze(2).permute(0, 2, 1)

    def forward_regular(self, hidden_states, query_states, kv_cache_layer=None, causal_mask=None, current_pos=None, layer_idx=None):
        """Forward pass for single token generation."""
        bsz, q_len, _ = hidden_states.shape

        # Get KV cache
        K_layer_cache, V_layer_cache = kv_cache_layer

        # Determine window size based on layer type and cache mode
        use_split_cache = getattr(self.config, 'use_split_cache', ENABLE_SPLIT_CACHE)
        if use_split_cache and layer_idx is not None:
            if self.config.layer_types[layer_idx] == "full_attention":
                # Global layer - use full attention_size (or state_length)
                window_size = self.config.attention_size
            else:
                # Local layer - use sliding_window (cache is already rotated)
                window_size = self.config.sliding_window
        else:
            # Legacy mode - use attention_size
            window_size = self.config.attention_size

        K_window = K_layer_cache[..., :window_size, :]
        V_window = V_layer_cache[..., :window_size, :]
        
        # Repeat KV for multi-head attention
        n_rep = self.num_heads // self.num_kv_heads
        key_states = self.repeat_kv(K_window, n_rep)
        value_states = self.repeat_kv(V_window, n_rep)

        # Compute attention using optimized path for batch_size=1
        attn_weights = torch.matmul(query_states.to(MODEL_DTYPE), key_states.transpose(-1, -2).to(MODEL_DTYPE)) * self.scale
        
        if causal_mask is not None:
            # Match the causal mask to the actual dimensions being used
            q_seq_len = query_states.shape[-2]  # Query sequence length (usually 1 for single token)
            k_seq_len = key_states.shape[-2]   # Key sequence length (actual filled cache length)
            attn_weights = attn_weights + causal_mask.to(MODEL_DTYPE)[:, :, :q_seq_len, :k_seq_len]

        # Optimized softmax for batch_size=1
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Compute attention output directly without einsum
        attn_output = torch.matmul(attn_weights, value_states.to(MODEL_DTYPE))
        
        # Reshape before projecting: [1, heads, q_len, head_dim] -> [1, q_len, heads*head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        
        # Project output (this will reshape from num_heads*head_dim back to hidden_size)
        attn_output = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return attn_output.squeeze(2).permute(0, 2, 1)

    def forward_prefill(self, hidden_states, query_states, kv_cache_layer=None, causal_mask=None, layer_idx=None):
        """Forward pass for prefill mode"""
        bsz, q_len, _ = hidden_states.shape

        # Get KV cache
        K_layer_cache, V_layer_cache = kv_cache_layer

        # Determine window size based on layer type and cache mode
        use_split_cache = getattr(self.config, 'use_split_cache', ENABLE_SPLIT_CACHE)
        if use_split_cache and layer_idx is not None:
            if self.config.layer_types[layer_idx] == "full_attention":
                # Global layer - use full attention_size (or state_length)
                window_size = self.config.attention_size
            else:
                # Local layer - use sliding_window (cache is already rotated)
                window_size = self.config.sliding_window
        else:
            # Legacy mode - use attention_size
            window_size = self.config.attention_size

        K_window = K_layer_cache[:, :window_size, :]
        V_window = V_layer_cache[:, :window_size, :]
        
        # Repeat KV for multi-head attention
        n_rep = self.num_heads // self.num_kv_heads
        key_states = self.repeat_kv(K_window, n_rep)
        value_states = self.repeat_kv(V_window, n_rep)
        
        # Compute scaled dot-product attention
        attn_weights = torch.einsum('bhqd,bhkd->bhqk', query_states.to(MODEL_DTYPE), key_states.to(MODEL_DTYPE)) * self.scale
        
        if causal_mask is not None:
            # Slice causal mask to match actual query and key sequence lengths
            q_seq_len = query_states.shape[2]  # Query sequence length
            k_seq_len = key_states.shape[2]  # Key sequence length matches window_size
            mask_slice = causal_mask.to(MODEL_DTYPE)[:, :, :q_seq_len, :k_seq_len]
            attn_weights = attn_weights + mask_slice
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, value_states.to(MODEL_DTYPE))
        
        # Reshape before projecting: [batch, heads, actual_seq_len, head_dim] -> [batch, actual_seq_len, heads*head_dim]
        # Use actual tensor dimensions instead of input q_len
        attn_output = attn_output.transpose(1, 2).contiguous()
        actual_bsz, actual_seq_len, num_heads, head_dim = attn_output.shape
        attn_output = attn_output.reshape(actual_bsz, actual_seq_len, num_heads * head_dim)
        
        # Project output (this will reshape from num_heads*head_dim back to hidden_size)
        attn_output = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return attn_output.squeeze(2).permute(0, 2, 1)


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.config = config  # Store config for clamping settings
        self.self_attn = Gemma3Attention(config)
        self.mlp = Gemma3MLP(config)
        # Gemma3 uses 4 norms per block:
        # 1. input_layernorm - before attention
        # 2. post_attention_layernorm - after attention (before residual add)
        # 3. pre_feedforward_layernorm - before MLP
        # 4. post_feedforward_layernorm - after MLP (before residual add)
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
    ) -> torch.Tensor:
        # Gemma3 architecture: pre-norm + post-norm for both attention and MLP
        # Note: Clamping is required for FP16/ANE to prevent overflow (Gemma3 was trained in bfloat16)

        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, causal_mask, position_ids, current_pos
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        # Optional FP16 clamping (OFF by default - use weight scaling instead)
        if self.config.enable_residual_clamp:
            hidden_states = torch.clamp(hidden_states, min=-self.config.residual_clamp_value, max=self.config.residual_clamp_value)

        # MLP block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        # Optional FP16 clamping (OFF by default - use weight scaling instead)
        if self.config.enable_residual_clamp:
            hidden_states = torch.clamp(hidden_states, min=-self.config.residual_clamp_value, max=self.config.residual_clamp_value)
        return hidden_states


class Gemma3Model(nn.Module):
    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.config = config
        self.disable_kv_cache = False  # Will be set by parent model
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=MODEL_DTYPE)
        # Gemma3 scales embeddings by sqrt(hidden_size)
        self.embedding_scale = config.hidden_size ** 0.5
        
        # Create dual RoPE embeddings for local vs global layers
        self.rotary_emb_global = Gemma3RotaryEmbedding(config)  # Uses rope_theta (1e6)
        # Create a config copy for local RoPE with different theta
        local_config = type(config)(**config.__dict__)
        local_config.rope_theta = getattr(config, 'rope_local_base_freq', 10000.0)
        self.rotary_emb_local = Gemma3RotaryEmbedding(local_config)  # Uses rope_local_base_freq (10k)
        
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize KV cache with MODEL_DTYPE (following llama_model.py pattern)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        if not hasattr(Gemma3Model, '_config_printed'):
            print(f"Gemma3Model using head_dim={self.head_dim} for KV cache (config has: {getattr(config, 'head_dim', 'not set')})")
            Gemma3Model._config_printed = True

        # Determine cache mode: split (local/global) or unified
        use_split_cache = getattr(config, 'use_split_cache', ENABLE_SPLIT_CACHE)
        use_single_cache = getattr(config, 'single_cache', False)
        if use_single_cache:
            use_split_cache = False

        if use_split_cache:
            # Split cache architecture for Gemma3 local/global attention
            # Count layers by type
            num_global_layers = sum(1 for t in config.layer_types if t == "full_attention")
            num_local_layers = config.num_hidden_layers - num_global_layers

            # Build layer index mappings
            self._local_layer_indices = [i for i in range(config.num_hidden_layers)
                                        if config.layer_types[i] == "sliding_attention"]
            self._global_layer_indices = [i for i in range(config.num_hidden_layers)
                                         if config.layer_types[i] == "full_attention"]

            # Local cache: for sliding window attention layers
            # Shape: [2*num_local_layers, num_kv_heads, sliding_window, head_dim]
            local_cache_size = (
                2 * num_local_layers,  # K and V for each local layer
                config.num_key_value_heads,
                config.sliding_window,  # 512 - only need sliding window size
                self.head_dim
            )
            self.register_buffer("kv_cache_local",
                               torch.zeros(local_cache_size, dtype=MODEL_DTYPE, device=TEST_DEVICE))

            # Global cache: for full attention layers
            # Shape: [2*num_global_layers, num_kv_heads, state_length, head_dim]
            global_cache_size = (
                2 * num_global_layers,  # K and V for each global layer
                config.num_key_value_heads,
                config.state_length,  # Full context length
                self.head_dim
            )
            self.register_buffer("kv_cache_global",
                               torch.zeros(global_cache_size, dtype=MODEL_DTYPE, device=TEST_DEVICE))

            if not hasattr(Gemma3Model, '_cache_init_printed'):
                print(f"Initialized SPLIT KV caches:")
                print(f"  kv_cache_local: {self.kv_cache_local.shape} for {num_local_layers} sliding window layers {self._local_layer_indices}")
                print(f"  kv_cache_global: {self.kv_cache_global.shape} for {num_global_layers} full attention layers {self._global_layer_indices}")
                Gemma3Model._cache_init_printed = True
        elif use_single_cache or FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
            # Legacy unified cache mode
            cache_size = (
                2 * config.num_hidden_layers,
                config.num_key_value_heads,
                config.state_length,
                self.head_dim
            )
            self.register_buffer("kv_cache_0", torch.zeros(cache_size, dtype=MODEL_DTYPE, device=TEST_DEVICE))
            if not hasattr(Gemma3Model, '_cache_init_printed'):
                print(f"Initialized unified KV kv_cache_0 with shape: {self.kv_cache_0.shape}")
                Gemma3Model._cache_init_printed = True
        else:
            # Per-layer cache mode (legacy)
            layers_per_group = config.num_hidden_layers
            for i in range(config.num_hidden_layers):
                cache_size = (
                    2 * layers_per_group,
                    config.num_key_value_heads,
                    config.state_length,
                    self.head_dim
                )
                self.register_buffer(
                    f"kv_cache_{i}",
                    torch.zeros(cache_size, dtype=MODEL_DTYPE, device=TEST_DEVICE)
                )

    def get_rotary_embeddings_s(self, current_pos, layer_idx=0):
        """Get rotary embeddings for the current position"""
        # Select RoPE based on layer type
        if hasattr(self.config, "layer_types") and self.config.layer_types[layer_idx] == "full_attention":
            rotary_emb = self.rotary_emb_global
        else:
            rotary_emb = self.rotary_emb_local
            
        sin = rotary_emb.sin_cached[:, current_pos].view(1, 1, 1, -1)
        cos = rotary_emb.cos_cached[:, current_pos].view(1, 1, 1, -1)
        return cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE)

    def get_rotary_embedding_prefill(self, positions, layer_idx=0):
        """Get rotary embeddings for a sequence of positions.
        Args:
            positions: Tensor of position indices
            layer_idx: Layer index to determine local vs global RoPE
        Returns:
            Tuple of (cos, sin) tensors with shape [1, seq_len, 1, head_dim]
        """
        # Select RoPE based on layer type
        if hasattr(self.config, "layer_types") and self.config.layer_types[layer_idx] == "full_attention":
            rotary_emb = self.rotary_emb_global
        else:
            rotary_emb = self.rotary_emb_local
        
        # Get embeddings for the range of positions directly
        # positions should be [batch_size, seq_len] or [seq_len]
        if positions.dim() == 2:
            seq_len = positions.size(1)  # [batch_size, seq_len]
            pos_indices = positions.squeeze(0)  # Remove batch dimension
        else:
            seq_len = positions.size(0)  # [seq_len]
            pos_indices = positions
            
        cos = rotary_emb.cos_cached[:, pos_indices].view(1, seq_len, 1, rotary_emb.dim)
        sin = rotary_emb.sin_cached[:, pos_indices].view(1, seq_len, 1, rotary_emb.dim)
        
        return cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE)

    def _kv_slice(self, layer_idx: int, current_pos: int, q_len: int) -> tuple[int, int]:
        """Return [start, end) key range to attend to for this layer / step.

        For split cache architecture:
        - Global layers: attend to full range [0, min(current_pos + q_len, state_length))
        - Local layers: attend to full local cache [0, min(current_pos + q_len, sliding_window))
                       (cache is already rotated to contain only valid positions)
        """
        if self.config.layer_types[layer_idx] == "full_attention":
            end = min(current_pos + q_len, self.config.state_length)
            return 0, end
        # sliding layer - for split cache, we attend to the full local cache
        # which contains the last sliding_window positions (already rotated)
        if getattr(self.config, 'use_split_cache', ENABLE_SPLIT_CACHE):
            effective_len = min(current_pos + q_len, self.config.sliding_window)
            return 0, effective_len
        else:
            # Legacy unified cache behavior
            end = min(current_pos + q_len, self.config.state_length)
            start = max(0, end - self.config.sliding_window)
            return start, end

    # -------------------------------------------------------------------------
    # Split cache storage and retrieval methods
    # -------------------------------------------------------------------------

    def _apply_update_mask(self, cache_slice: torch.Tensor, new_states: torch.Tensor,
                           update_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply update_mask to write new_states into cache_slice without dynamic slicing.

        Args:
            cache_slice: [1, num_heads, cache_len, head_dim]
            new_states:  [1, num_heads, 1, head_dim] (single token)
            update_mask: [1, 1, cache_len, 1] (float mask with 1 at write position)
        """
        mask = update_mask.to(dtype=cache_slice.dtype)
        expanded = new_states.expand(cache_slice.shape[0], new_states.shape[1], cache_slice.shape[2], new_states.shape[3])
        return cache_slice * (1.0 - mask) + expanded * mask

    def _apply_update_mask_batch(
        self,
        cache_slice: torch.Tensor,
        new_states: torch.Tensor,
        update_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply update_mask to write a batch of new_states into cache_slice.

        Args:
            cache_slice: [1, num_heads, cache_len, head_dim]
            new_states:  [1, num_heads, seq_len, head_dim]
            update_mask: [1, 1, cache_len, seq_len] (float mask with 1 at write positions)
        """
        mask = update_mask.to(dtype=cache_slice.dtype)
        mask = mask.expand(cache_slice.shape[0], cache_slice.shape[1], cache_slice.shape[2], mask.shape[-1])
        # Sum over seq_len: place each token at its target position.
        updates = torch.einsum("bhkd,bhlk->bhld", new_states, mask)
        mask_sum = torch.clamp(mask.sum(dim=-1, keepdim=True), max=1.0)
        return cache_slice * (1.0 - mask_sum) + updates

    def _fill_kv_local(self, layer_idx: int, key_states: torch.Tensor,
                      value_states: torch.Tensor, update_mask_local: torch.Tensor) -> None:
        """
        Fill KV states in local cache (for single-token inference).

        Left-fill using update_mask to avoid dynamic slicing.

        Args:
            layer_idx: Original layer index (0-17)
            key_states: Key tensor to store, shape [1, num_kv_heads, 1, head_dim]
            value_states: Value tensor to store, same shape as key_states
            update_mask_local: [1, 1, sliding_window, 1]
        """
        cache_idx = self._local_layer_indices.index(layer_idx)
        num_local = len(self._local_layer_indices)
        key_cache_idx = cache_idx
        value_cache_idx = cache_idx + num_local
        key_slice = self.kv_cache_local[key_cache_idx:key_cache_idx + 1]
        value_slice = self.kv_cache_local[value_cache_idx:value_cache_idx + 1]

        self.kv_cache_local[key_cache_idx:key_cache_idx + 1, :, :, :] = self._apply_update_mask(
            key_slice, key_states, update_mask_local
        )
        self.kv_cache_local[value_cache_idx:value_cache_idx + 1, :, :, :] = self._apply_update_mask(
            value_slice, value_states, update_mask_local
        )

    def _store_kv_local_prefill_masked(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        update_mask_local: torch.Tensor,
    ) -> None:
        """
        Store KV states in local cache during prefill (multi-token) using update_mask.

        Args:
            layer_idx: Original layer index (0-17)
            key_states: Key tensor, shape [1, num_kv_heads, seq_len, head_dim]
            value_states: Value tensor, same shape as key_states
            update_mask_local: [1, 1, sliding_window, seq_len]
        """
        cache_idx = self._local_layer_indices.index(layer_idx)
        num_local = len(self._local_layer_indices)
        key_cache_idx = cache_idx
        value_cache_idx = cache_idx + num_local
        key_slice = self.kv_cache_local[key_cache_idx:key_cache_idx + 1]
        value_slice = self.kv_cache_local[value_cache_idx:value_cache_idx + 1]

        self.kv_cache_local[key_cache_idx:key_cache_idx + 1, :, :, :] = self._apply_update_mask_batch(
            key_slice, key_states, update_mask_local
        )
        self.kv_cache_local[value_cache_idx:value_cache_idx + 1, :, :, :] = self._apply_update_mask_batch(
            value_slice, value_states, update_mask_local
        )

    def _update_kv_local(self, layer_idx: int, key_states: torch.Tensor,
                        value_states: torch.Tensor) -> None:
        """
        Update KV states in local cache (for generation, after cache is full).

        Always shifts left and stores at the end:
        1. cache[0:LEN-1] = cache[1:LEN]  (shift left, discard oldest)
        2. cache[LEN-1] = new_value  (store new at end)

        Use for positions >= sliding_window. No conditionals.

        Note: Uses torch.narrow + torch.cat for ANE compatibility.
        - torch.narrow explicitly specifies positive start/length (avoids negative offset errors)
        - torch.cat creates the new shifted tensor
        The original approach (cache[:-1] = cache[1:].clone()) fails during torch.jit.trace,
        and slice notation (cache[:, :, 1:, :]) causes "offset (-1)" errors on ANE.

        Args:
            layer_idx: Original layer index (0-17)
            key_states: Key tensor to store, shape [1, num_kv_heads, 1, head_dim]
            value_states: Value tensor to store, same shape as key_states
        """
        cache_idx = self._local_layer_indices.index(layer_idx)
        num_local = len(self._local_layer_indices)
        key_cache_idx = cache_idx
        value_cache_idx = cache_idx + num_local

        # Get sliding window size for narrow operation
        sw = self.config.sliding_window  # 512

        # Create shifted key cache: drop first token (use narrow to avoid negative offset)
        # torch.narrow(tensor, dim, start, length) - all positive values
        key_slice = self.kv_cache_local[key_cache_idx:key_cache_idx + 1]
        key_tail = torch.narrow(key_slice, 2, 1, sw - 1)  # start=1, length=511
        shifted_key = torch.cat([key_tail, key_states], dim=2)
        self.kv_cache_local[key_cache_idx:key_cache_idx + 1, :, :, :] = shifted_key

        # Create shifted value cache: drop first token (use narrow to avoid negative offset)
        value_slice = self.kv_cache_local[value_cache_idx:value_cache_idx + 1]
        value_tail = torch.narrow(value_slice, 2, 1, sw - 1)  # start=1, length=511
        shifted_value = torch.cat([value_tail, value_states], dim=2)
        self.kv_cache_local[value_cache_idx:value_cache_idx + 1, :, :, :] = shifted_value

    def _store_kv_local(self, layer_idx: int, key_states: torch.Tensor,
                       value_states: torch.Tensor, current_pos: int,
                       update_mask: torch.Tensor | None = None) -> None:
        """
        Store KV states in local cache (auto-selects fill or update).

        Behavior depends on config.force_rotation_mode:
        - None (default): Conditional logic based on current_pos
          - current_pos < sliding_window: fill mode (direct store)
          - current_pos >= sliding_window: rotate mode (shift + store)
        - False: Always use fill mode (for 'infer' function, pos < sliding_window)
        - True: Always use rotate mode (for 'infer_rotate' function, pos >= sliding_window)

        For CoreML export, set force_rotation_mode before tracing:
        - infer function: force_rotation_mode=False
        - infer_rotate function: force_rotation_mode=True

        Args:
            layer_idx: Original layer index (0-17)
            key_states: Key tensor to store
            value_states: Value tensor to store
            current_pos: Current absolute position in the sequence
        """
        force_rotation = getattr(self.config, 'force_rotation_mode', None)

        cache_idx = self._local_layer_indices.index(layer_idx)
        num_local = len(self._local_layer_indices)
        key_cache_idx = cache_idx
        value_cache_idx = cache_idx + num_local

        if force_rotation is True:
            # Always rotate (for infer_rotate function)
            self._update_kv_local(layer_idx, key_states, value_states)
        elif force_rotation is False:
            # Always fill (for infer function)
            if update_mask is None:
                # Fallback to direct write if update_mask is missing
                self.kv_cache_local[key_cache_idx:key_cache_idx + 1, :, current_pos:current_pos + 1, :] = key_states
                self.kv_cache_local[value_cache_idx:value_cache_idx + 1, :, current_pos:current_pos + 1, :] = value_states
            else:
                update_mask_local = torch.narrow(update_mask, 2, 0, self.config.sliding_window)
                self._fill_kv_local(layer_idx, key_states, value_states, update_mask_local)
        else:
            # Default: conditional based on position
            sliding_window = self.config.sliding_window
            if current_pos < sliding_window:
                if update_mask is None:
                    self.kv_cache_local[key_cache_idx:key_cache_idx + 1, :, current_pos:current_pos + 1, :] = key_states
                    self.kv_cache_local[value_cache_idx:value_cache_idx + 1, :, current_pos:current_pos + 1, :] = value_states
                else:
                    update_mask_local = torch.narrow(update_mask, 2, 0, self.config.sliding_window)
                    self._fill_kv_local(layer_idx, key_states, value_states, update_mask_local)
            else:
                self._update_kv_local(layer_idx, key_states, value_states)

    def _store_kv_local_prefill(self, layer_idx: int, key_states: torch.Tensor,
                               value_states: torch.Tensor, current_pos: int, seq_len: int) -> None:
        """
        Store KV states in local cache during prefill (multi-token).

        Standard left-fill: store at [current_pos : current_pos + seq_len).

        For prefill that exceeds sliding_window, only the last sliding_window tokens
        are kept.

        Args:
            layer_idx: Original layer index (0-17)
            key_states: Key tensor, shape [1, seq_len, num_kv_heads, head_dim] or similar
            value_states: Value tensor, same shape as key_states
            current_pos: Starting position (unused, kept for API compatibility)
            seq_len: Number of tokens being prefilled (fixed at compile time)
        """
        cache_idx = self._local_layer_indices.index(layer_idx)
        num_local = len(self._local_layer_indices)
        key_cache_idx = cache_idx
        value_cache_idx = cache_idx + num_local

        sliding_window = self.config.sliding_window

        # For prefill, if seq_len > sliding_window, only keep last sliding_window tokens
        if seq_len > sliding_window:
            start = seq_len - sliding_window
            key_states = torch.narrow(key_states, 1, start, sliding_window)
            value_states = torch.narrow(value_states, 1, start, sliding_window)
            seq_len = sliding_window

        if getattr(self.config, "prefill_dynamic_slice", False):
            # Dynamic append based on current_pos (experimental; may be illegal for ANE)
            start_pos = current_pos
            end_pos = current_pos + seq_len
            self.kv_cache_local[key_cache_idx:key_cache_idx + 1, :, start_pos:end_pos, :] = key_states[:, :seq_len, :]
            self.kv_cache_local[value_cache_idx:value_cache_idx + 1, :, start_pos:end_pos, :] = value_states[:, :seq_len, :]
            return

        # Prefill fill-mode uses fixed left-fill starting at 0 for ANE (static slicing)
        start_pos = 0
        end_pos = min(seq_len, sliding_window)
        self.kv_cache_local[key_cache_idx:key_cache_idx + 1, :, start_pos:end_pos, :] = key_states[:, :end_pos, :]
        self.kv_cache_local[value_cache_idx:value_cache_idx + 1, :, start_pos:end_pos, :] = value_states[:, :end_pos, :]

    def _update_kv_local_prefill(self, layer_idx: int, key_states: torch.Tensor,
                                 value_states: torch.Tensor, seq_len: int) -> None:
        """
        Update KV states in local cache during prefill with rotation (multi-token).

        Always performs shift-then-store for the entire batch:
        1. Shift cache left by batch_size positions
        2. Store new tokens at the end

        Use for prefill when current_pos >= sliding_window.

        Note: Uses config.batch_size instead of dynamic seq_len for CoreML tracing compatibility.
        The traced model expects fixed batch_size (64) for prefill operations.

        Args:
            layer_idx: Original layer index (0-17)
            key_states: Key tensor, shape [1, num_kv_heads, batch_size, head_dim]
            value_states: Value tensor, same shape as key_states
            seq_len: Ignored - uses config.batch_size for tracing compatibility
        """
        cache_idx = self._local_layer_indices.index(layer_idx)
        num_local = len(self._local_layer_indices)
        key_cache_idx = cache_idx
        value_cache_idx = cache_idx + num_local

        sw = self.config.sliding_window  # 512
        # Use config batch_size for tracing (must be concrete value for CoreML)
        batch_size = self.config.batch_size  # 64

        # During prefill_rotate, we always shift by batch_size and store batch_size tokens
        # batch_size (64) < sliding_window (512), so we always use the shift path
        # Shift left by batch_size, store new at end
        # Use positive slice indices for ANE compatibility (no negative offsets)
        key_slice = self.kv_cache_local[key_cache_idx:key_cache_idx + 1]
        key_tail = key_slice[:, :, batch_size:sw, :]  # Keep positions batch_size to sw-1
        shifted_key = torch.cat([key_tail, key_states[:, :, :batch_size, :]], dim=2)
        self.kv_cache_local[key_cache_idx:key_cache_idx + 1, :, :, :] = shifted_key

        value_slice = self.kv_cache_local[value_cache_idx:value_cache_idx + 1]
        value_tail = value_slice[:, :, batch_size:sw, :]  # Keep positions batch_size to sw-1
        shifted_value = torch.cat([value_tail, value_states[:, :, :batch_size, :]], dim=2)
        self.kv_cache_local[value_cache_idx:value_cache_idx + 1, :, :, :] = shifted_value

    def _store_kv_global(self, layer_idx: int, key_states: torch.Tensor,
                        value_states: torch.Tensor, current_pos: int,
                        update_mask: torch.Tensor | None = None) -> None:
        """
        Store KV states in global cache using left-fill masked update.

        Args:
            layer_idx: Original layer index (0-17)
            key_states: Key tensor to store
            value_states: Value tensor to store
            current_pos: Current position (unused, kept for API compatibility)
        """
        cache_idx = self._global_layer_indices.index(layer_idx)
        num_global = len(self._global_layer_indices)
        key_cache_idx = cache_idx
        value_cache_idx = cache_idx + num_global

        if update_mask is None:
            pos = min(current_pos, self.config.state_length - 1)
            self.kv_cache_global[key_cache_idx:key_cache_idx + 1, :, pos:pos + 1, :] = key_states
            self.kv_cache_global[value_cache_idx:value_cache_idx + 1, :, pos:pos + 1, :] = value_states
        else:
            update_mask_global = torch.narrow(update_mask, 2, 0, self.config.state_length)
            key_slice = self.kv_cache_global[key_cache_idx:key_cache_idx + 1]
            value_slice = self.kv_cache_global[value_cache_idx:value_cache_idx + 1]
            self.kv_cache_global[key_cache_idx:key_cache_idx + 1, :, :, :] = self._apply_update_mask(
                key_slice, key_states, update_mask_global
            )
            self.kv_cache_global[value_cache_idx:value_cache_idx + 1, :, :, :] = self._apply_update_mask(
                value_slice, value_states, update_mask_global
            )

    def _store_kv_global_prefill_masked(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        update_mask_global: torch.Tensor,
    ) -> None:
        """
        Store KV states in global cache during prefill (multi-token) using update_mask.

        Args:
            layer_idx: Original layer index (0-17)
            key_states: Key tensor, shape [1, num_kv_heads, seq_len, head_dim]
            value_states: Value tensor, same shape as key_states
            update_mask_global: [1, 1, state_length, seq_len]
        """
        cache_idx = self._global_layer_indices.index(layer_idx)
        num_global = len(self._global_layer_indices)
        key_cache_idx = cache_idx
        value_cache_idx = cache_idx + num_global
        key_slice = self.kv_cache_global[key_cache_idx:key_cache_idx + 1]
        value_slice = self.kv_cache_global[value_cache_idx:value_cache_idx + 1]

        self.kv_cache_global[key_cache_idx:key_cache_idx + 1, :, :, :] = self._apply_update_mask_batch(
            key_slice, key_states, update_mask_global
        )
        self.kv_cache_global[value_cache_idx:value_cache_idx + 1, :, :, :] = self._apply_update_mask_batch(
            value_slice, value_states, update_mask_global
        )

    def _update_kv_global(self, layer_idx: int, key_states: torch.Tensor,
                          value_states: torch.Tensor) -> None:
        """
        Update KV states in global cache with rotation (single token).

        Shifts left and stores at end, similar to _update_kv_local.
        Use when current_pos >= state_length.

        Args:
            layer_idx: Original layer index (0-17)
            key_states: Key tensor to store
            value_states: Value tensor to store
        """
        cache_idx = self._global_layer_indices.index(layer_idx)
        num_global = len(self._global_layer_indices)
        key_cache_idx = cache_idx
        value_cache_idx = cache_idx + num_global

        sl = self.config.state_length  # 1024

        # Shift left by 1, store new at end
        # Use torch.narrow for ANE compatibility (positive indices only)
        key_slice = self.kv_cache_global[key_cache_idx:key_cache_idx + 1]
        key_tail = torch.narrow(key_slice, 2, 1, sl - 1)
        shifted_key = torch.cat([key_tail, key_states], dim=2)
        self.kv_cache_global[key_cache_idx:key_cache_idx + 1, :, :, :] = shifted_key

        value_slice = self.kv_cache_global[value_cache_idx:value_cache_idx + 1]
        value_tail = torch.narrow(value_slice, 2, 1, sl - 1)
        shifted_value = torch.cat([value_tail, value_states], dim=2)
        self.kv_cache_global[value_cache_idx:value_cache_idx + 1, :, :, :] = shifted_value

    def _store_kv_global_prefill(self, layer_idx: int, key_states: torch.Tensor,
                                value_states: torch.Tensor, current_pos: int, seq_len: int) -> None:
        """
        Store KV states in global cache during prefill (multi-token).

        Standard left-fill: store at [current_pos : current_pos + seq_len).

        Args:
            layer_idx: Original layer index (0-17)
            key_states: Key tensor, shape [1, seq_len, num_kv_heads, head_dim] or similar
            value_states: Value tensor, same shape as key_states
            current_pos: Starting position (unused, kept for API compatibility)
            seq_len: Number of tokens being prefilled (fixed at compile time)
        """
        cache_idx = self._global_layer_indices.index(layer_idx)
        num_global = len(self._global_layer_indices)
        key_cache_idx = cache_idx
        value_cache_idx = cache_idx + num_global

        sl = self.config.state_length

        if seq_len > sl:
            start = seq_len - sl
            key_states = torch.narrow(key_states, 2, start, sl)
            value_states = torch.narrow(value_states, 2, start, sl)
            seq_len = sl

        if getattr(self.config, "prefill_dynamic_slice", False):
            # Dynamic append based on current_pos (experimental; may be illegal for ANE)
            start_pos = current_pos
            end_pos = current_pos + seq_len
            self.kv_cache_global[key_cache_idx:key_cache_idx + 1, :, start_pos:end_pos, :] = key_states[:, :seq_len, :]
            self.kv_cache_global[value_cache_idx:value_cache_idx + 1, :, start_pos:end_pos, :] = value_states[:, :seq_len, :]
            return

        # Prefill fill-mode uses fixed left-fill starting at 0 for ANE (static slicing)
        start_pos = 0
        end_pos = min(seq_len, self.config.state_length)
        self.kv_cache_global[key_cache_idx:key_cache_idx + 1, :, start_pos:end_pos, :] = key_states[:, :end_pos, :]
        self.kv_cache_global[value_cache_idx:value_cache_idx + 1, :, start_pos:end_pos, :] = value_states[:, :end_pos, :]

    def _update_kv_global_prefill(self, layer_idx: int, key_states: torch.Tensor,
                                  value_states: torch.Tensor, seq_len: int) -> None:
        """
        Update KV states in global cache during prefill with rotation (multi-token).

        Always performs shift-then-store for the entire batch:
        1. Shift cache left by seq_len positions
        2. Store new tokens at the end

        Use for prefill when current_pos >= state_length.

        Args:
            layer_idx: Original layer index (0-17)
            key_states: Key tensor, shape [1, num_kv_heads, seq_len, head_dim]
            value_states: Value tensor, same shape as key_states
            seq_len: Number of tokens being prefilled
        """
        cache_idx = self._global_layer_indices.index(layer_idx)
        num_global = len(self._global_layer_indices)
        key_cache_idx = cache_idx
        value_cache_idx = cache_idx + num_global

        sl = self.config.state_length  # 1024

        # Clamp seq_len to not exceed state_length
        actual_store_len = min(seq_len, sl)

        if seq_len >= sl:
            # If batch >= state_length, just store last state_length tokens
            self.kv_cache_global[key_cache_idx:key_cache_idx + 1, :, :, :] = key_states[:, :, -sl:, :]
            self.kv_cache_global[value_cache_idx:value_cache_idx + 1, :, :, :] = value_states[:, :, -sl:, :]
        else:
            # Shift left by seq_len, store new at end
            # Use torch.narrow for ANE compatibility (positive indices only)
            key_slice = self.kv_cache_global[key_cache_idx:key_cache_idx + 1]
            key_tail = torch.narrow(key_slice, 2, seq_len, sl - seq_len)
            shifted_key = torch.cat([key_tail, key_states[:, :, :actual_store_len, :]], dim=2)
            self.kv_cache_global[key_cache_idx:key_cache_idx + 1, :, :, :] = shifted_key

            value_slice = self.kv_cache_global[value_cache_idx:value_cache_idx + 1]
            value_tail = torch.narrow(value_slice, 2, seq_len, sl - seq_len)
            shifted_value = torch.cat([value_tail, value_states[:, :, :actual_store_len, :]], dim=2)
            self.kv_cache_global[value_cache_idx:value_cache_idx + 1, :, :, :] = shifted_value

    def _get_kv_cache_for_layer(self, layer_idx: int, current_pos: int) -> tuple:
        """
        Get the key and value cache tensors for a specific layer.

        Returns appropriately sliced cache based on layer type.

        Args:
            layer_idx: Original layer index (0-17)
            current_pos: Current position (used for determining valid range)

        Returns:
            Tuple of (key_cache, value_cache) tensors
        """
        if self.config.layer_types[layer_idx] == "full_attention":
            # Global layer - return from global cache
            cache_idx = self._global_layer_indices.index(layer_idx)
            num_global = len(self._global_layer_indices)
            key_cache_idx = cache_idx
            value_cache_idx = cache_idx + num_global

            key_cache = self.kv_cache_global[key_cache_idx:key_cache_idx + 1].squeeze(0)
            value_cache = self.kv_cache_global[value_cache_idx:value_cache_idx + 1].squeeze(0)
            return key_cache, value_cache
        else:
            # Local layer - return from local cache
            cache_idx = self._local_layer_indices.index(layer_idx)
            num_local = len(self._local_layer_indices)
            key_cache_idx = cache_idx
            value_cache_idx = cache_idx + num_local

            key_cache = self.kv_cache_local[key_cache_idx:key_cache_idx + 1].squeeze(0)
            value_cache = self.kv_cache_local[value_cache_idx:value_cache_idx + 1].squeeze(0)
            return key_cache, value_cache

    # -------------------------------------------------------------------------

    def process_layer_prefill(self, layer_idx, hidden_states, position_ids, causal_mask, current_pos, layer_offset,
                              update_mask=None):
        """Process a single transformer layer in prefill mode"""
        layer = self.layers[layer_idx]

        # Get layer-specific RoPE
        rotary_emb = self.get_rotary_embedding_prefill(position_ids, layer_idx)

        normalized_states = layer.input_layernorm(hidden_states)

        # Get query, key and value states for prefill
        query_states, key_states, value_states = layer.self_attn.get_new_kv_cache_prefill(
            normalized_states,
            current_pos,
            rotary_emb
        )

        seq_length = key_states.shape[2]
        use_split_cache = getattr(self.config, 'use_split_cache', ENABLE_SPLIT_CACHE)

        if use_split_cache:
            # Split cache path - store in appropriate cache based on layer type
            if update_mask is not None:
                if self.config.layer_types[layer_idx] == "full_attention":
                    update_mask_global = torch.narrow(update_mask, 2, 0, self.config.state_length)
                    self._store_kv_global_prefill_masked(layer_idx, key_states, value_states, update_mask_global)
                else:
                    update_mask_local = torch.narrow(update_mask, 2, 0, self.config.sliding_window)
                    self._store_kv_local_prefill_masked(layer_idx, key_states, value_states, update_mask_local)
            else:
                if self.config.layer_types[layer_idx] == "full_attention":
                    self._store_kv_global_prefill(layer_idx, key_states, value_states, current_pos, seq_length)
                else:
                    self._store_kv_local_prefill(layer_idx, key_states, value_states, current_pos, seq_length)

            # Get the KV cache for this layer
            key_cache, value_cache = self._get_kv_cache_for_layer(layer_idx, current_pos)
        else:
            # Legacy unified cache path
            # Get group indices
            group_idx, layer_in_group_idx, layers_per_group = get_kv_cache_idx(layer_idx, self.config.num_hidden_layers)

            # Get the combined KV cache for this group
            if FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
                kv_cache = getattr(self, "kv_cache_0")
            else:
                kv_cache = getattr(self, f"kv_cache_{group_idx}")

            key_idx = layer_in_group_idx
            value_idx = layer_in_group_idx + layers_per_group

            if update_mask is not None:
                update_mask_unified = torch.narrow(update_mask, 2, 0, self.config.state_length)
                key_slice = kv_cache[key_idx:key_idx + 1]
                value_slice = kv_cache[value_idx:value_idx + 1]
                kv_cache[key_idx:key_idx + 1, :, :, :] = self._apply_update_mask_batch(
                    key_slice, key_states, update_mask_unified
                )
                kv_cache[value_idx:value_idx + 1, :, :, :] = self._apply_update_mask_batch(
                    value_slice, value_states, update_mask_unified
                )
            else:
                # Store KV - direct indexing (model only used for positions < attention_size)
                end = min(current_pos + seq_length, self.config.state_length)
                kv_cache[key_idx:key_idx + 1, :, current_pos:end, :] = key_states[:, :end-current_pos, :]
                kv_cache[value_idx:value_idx + 1, :, current_pos:end, :] = value_states[:, :end-current_pos, :]

            # Get the key and value states for this layer from the merged cache
            key_cache = kv_cache[key_idx:key_idx + 1].squeeze(0)
            value_cache = kv_cache[value_idx:value_idx + 1].squeeze(0)

        # Run attention with the updated KV cache
        attn_output = layer.self_attn.forward_prefill(
            hidden_states=normalized_states,
            query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask,
            layer_idx=layer_idx,
        )

        # Apply post_attention_layernorm before residual add
        attn_output = layer.post_attention_layernorm(attn_output)
        hidden_states = hidden_states + attn_output
        # Optional FP16 clamping (OFF by default - use weight scaling instead)
        if self.config.enable_residual_clamp:
            hidden_states = torch.clamp(hidden_states, min=-self.config.residual_clamp_value, max=self.config.residual_clamp_value)

        # Apply MLP with proper normalization (Gemma3: pre + post feedforward norms)
        residual = hidden_states
        hidden_states = layer.pre_feedforward_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = layer.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        # Optional FP16 clamping (OFF by default - use weight scaling instead)
        if self.config.enable_residual_clamp:
            hidden_states = torch.clamp(hidden_states, min=-self.config.residual_clamp_value, max=self.config.residual_clamp_value)

        return hidden_states

    def process_layer_prefill_rotate(self, layer_idx, hidden_states, position_ids, causal_mask, current_pos, layer_offset,
                                     update_mask=None):
        """Process a single transformer layer in prefill mode WITH ROTATION.

        Use when prefilling at positions >= sliding_window.
        Local layers use rotation (shift + store), global layers still use fill mode.
        Global cache never rotates - it can store up to state_length tokens directly.
        """
        layer = self.layers[layer_idx]

        # Get layer-specific RoPE
        rotary_emb = self.get_rotary_embedding_prefill(position_ids, layer_idx)

        normalized_states = layer.input_layernorm(hidden_states)

        # Get query, key and value states for prefill
        query_states, key_states, value_states = layer.self_attn.get_new_kv_cache_prefill(
            normalized_states,
            current_pos,
            rotary_emb
        )

        seq_length = key_states.shape[2]
        use_split_cache = getattr(self.config, 'use_split_cache', ENABLE_SPLIT_CACHE)

        if use_split_cache:
            # Split cache path
            if self.config.layer_types[layer_idx] == "full_attention":
                # Global layer - always use fill mode (global cache never rotates)
                # Global attention can see up to state_length tokens without rotation
                self._store_kv_global_prefill(layer_idx, key_states, value_states, current_pos, seq_length)
            else:
                # Local layer - use rotation (shift + store)
                self._update_kv_local_prefill(layer_idx, key_states, value_states, seq_length)

            # Get the KV cache for this layer
            key_cache, value_cache = self._get_kv_cache_for_layer(layer_idx, current_pos)
        else:
            # Legacy unified cache path - not supported for rotation
            raise NotImplementedError("Rotation prefill not supported with unified cache")

        # Run attention with the updated KV cache
        attn_output = layer.self_attn.forward_prefill(
            hidden_states=normalized_states,
            query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask,
            layer_idx=layer_idx,
        )

        # Apply post_attention_layernorm before residual add
        attn_output = layer.post_attention_layernorm(attn_output)
        hidden_states = hidden_states + attn_output
        # Optional FP16 clamping (OFF by default - use weight scaling instead)
        if self.config.enable_residual_clamp:
            hidden_states = torch.clamp(hidden_states, min=-self.config.residual_clamp_value, max=self.config.residual_clamp_value)

        # Apply MLP with proper normalization (Gemma3: pre + post feedforward norms)
        residual = hidden_states
        hidden_states = layer.pre_feedforward_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = layer.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        # Optional FP16 clamping (OFF by default - use weight scaling instead)
        if self.config.enable_residual_clamp:
            hidden_states = torch.clamp(hidden_states, min=-self.config.residual_clamp_value, max=self.config.residual_clamp_value)

        return hidden_states

    def process_layer_regular(self, layer_idx, hidden_states, position_ids, causal_mask, current_pos, layer_offset,
                              update_mask=None):
        """Process a single transformer layer in regular (non-prefill) mode"""
        layer = self.layers[layer_idx]
        batch_size = position_ids.shape[0]
        seq_len = hidden_states.shape[1]
        
        # Get layer-specific RoPE
        if seq_len == 1:
            rotary_emb = self.get_rotary_embeddings_s(current_pos, layer_idx)
        else:
            rotary_emb = self.get_rotary_embedding_prefill(position_ids, layer_idx)

        normalized_states = layer.input_layernorm(hidden_states)
        
        # Choose appropriate method based on sequence length
        if seq_len == 1:
            # Single token - use the single token method
            query_states, key_states, value_states = layer.self_attn.get_new_kv_cache(
                normalized_states,
                current_pos,
                rotary_emb
            )
        else:
            # Multi-token - use the prefill method 
            query_states, key_states, value_states = layer.self_attn.get_new_kv_cache_prefill(
                normalized_states,
                current_pos,
                rotary_emb
            )

        if not self.disable_kv_cache:
            # Standard KV cache path
            use_split_cache = getattr(self.config, 'use_split_cache', ENABLE_SPLIT_CACHE)

            if use_split_cache:
                # Split cache path - store in appropriate cache based on layer type
                if seq_len == 1:
                    # Single token storage with rotation for local cache
                    if self.config.layer_types[layer_idx] == "full_attention":
                        self._store_kv_global(layer_idx, key_states, value_states, current_pos, update_mask)
                    else:
                        self._store_kv_local(layer_idx, key_states, value_states, current_pos, update_mask)
                else:
                    # Multi-token storage
                    if self.config.layer_types[layer_idx] == "full_attention":
                        self._store_kv_global_prefill(layer_idx, key_states, value_states, current_pos, seq_len)
                    else:
                        self._store_kv_local_prefill(layer_idx, key_states, value_states, current_pos, seq_len)

                # Get the KV cache for this layer
                key_cache, value_cache = self._get_kv_cache_for_layer(layer_idx, current_pos)
            else:
                # Legacy unified cache path
                group_idx, layer_in_group_idx, layers_per_group = get_kv_cache_idx(layer_idx, self.config.num_hidden_layers)

                # Get the combined KV cache for this group
                if FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
                    kv_cache = getattr(self, "kv_cache_0")
                else:
                    kv_cache = getattr(self, f"kv_cache_{group_idx}")

                key_idx = layer_in_group_idx
                value_idx = layer_in_group_idx + layers_per_group

                if seq_len == 1:
                    # Single token storage - direct indexing
                    pos = current_pos
                    kv_cache[key_idx:key_idx + 1, :, pos:pos + 1, :] = key_states
                    kv_cache[value_idx:value_idx + 1, :, pos:pos + 1, :] = value_states
                else:
                    # Multi-token storage (prefill) - direct indexing
                    pos = current_pos.item() if isinstance(current_pos, torch.Tensor) else current_pos
                    end = min(pos + seq_len, self.config.state_length)
                    kv_cache[key_idx:key_idx + 1, :, pos:end, :] = key_states[:, :end-pos, :]
                    kv_cache[value_idx:value_idx + 1, :, pos:end, :] = value_states[:, :end-pos, :]

                # Get the key and value states for this layer from the merged cache
                key_cache = kv_cache[key_idx:key_idx + 1].squeeze(0)
                value_cache = kv_cache[value_idx:value_idx + 1].squeeze(0)

            # Determine cache length for causal mask based on cache type
            if use_split_cache:
                if self.config.layer_types[layer_idx] == "full_attention":
                    cache_len = self.config.state_length
                else:
                    cache_len = self.config.sliding_window
            else:
                cache_len = self.config.state_length

            # Run attention with the updated KV cache
            if seq_len == 1:
                attn_output = layer.self_attn.forward_regular(
                    hidden_states=normalized_states,
                    query_states=query_states,
                    kv_cache_layer=(key_cache, value_cache),
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    layer_idx=layer_idx,
                )
            else:
                # For multi-token sequences, adjust causal mask to match cache dimensions
                adjusted_causal_mask = torch.zeros((1, 1, seq_len, cache_len), dtype=MODEL_DTYPE, device=TEST_DEVICE)

                # Apply causal mask only to the positions we're using (pos:pos+seq_len)
                pos = current_pos.item() if isinstance(current_pos, torch.Tensor) else current_pos
                for i in range(seq_len):
                    for j in range(pos + i + 1, pos + seq_len):
                        if j < cache_len:  # Make sure we don't go out of bounds
                            adjusted_causal_mask[0, 0, i, j] = float('-inf')

                attn_output = layer.self_attn.forward_prefill(
                    hidden_states=normalized_states,
                    query_states=query_states,
                    kv_cache_layer=(key_cache, value_cache),
                    causal_mask=adjusted_causal_mask,
                    layer_idx=layer_idx,
                )
        else:
            # No KV cache path - use the computed K/V directly without cache
            # Repeat KV for multi-head attention (same logic as forward_regular)
            n_rep = layer.self_attn.num_heads // layer.self_attn.num_kv_heads
            
            # Create fake cache with just current K/V for attention computation
            fake_key_cache = torch.zeros((layer.self_attn.num_kv_heads, self.config.state_length, layer.self_attn.head_dim), 
                                       dtype=MODEL_DTYPE, device=TEST_DEVICE)
            fake_value_cache = torch.zeros((layer.self_attn.num_kv_heads, self.config.state_length, layer.self_attn.head_dim), 
                                         dtype=MODEL_DTYPE, device=TEST_DEVICE)
            
            # Place current K/V at the correct position
            pos = current_pos.item() if isinstance(current_pos, torch.Tensor) else current_pos
            if seq_len == 1:
                fake_key_cache[:, pos:pos+1, :] = key_states.squeeze(0)
                fake_value_cache[:, pos:pos+1, :] = value_states.squeeze(0)
            else:
                fake_key_cache[:, pos:pos+seq_len, :] = key_states.squeeze(0)
                fake_value_cache[:, pos:pos+seq_len, :] = value_states.squeeze(0)
            
            # For disable KV cache, create a causal mask that only covers the actual sequence
            # This avoids dimension mismatches with the full cache size
            if seq_len == 1:
                # Single token - use original causal mask logic
                adjusted_causal_mask = causal_mask
            else:
                # Multi-token - create a causal mask that covers the full cache length
                # but only applies causal restriction to the actual sequence positions
                cache_len = self.config.state_length
                adjusted_causal_mask = torch.zeros((1, 1, seq_len, cache_len), dtype=MODEL_DTYPE, device=TEST_DEVICE)
                
                # Apply causal mask only to the positions we're using (pos:pos+seq_len)
                for i in range(seq_len):
                    for j in range(pos + i + 1, pos + seq_len):
                        if j < cache_len:  # Make sure we don't go out of bounds
                            adjusted_causal_mask[0, 0, i, j] = float('-inf')
            
            # Run attention with fake cache (same computation as forward_regular/prefill)
            if seq_len == 1:
                attn_output = layer.self_attn.forward_regular(
                    hidden_states=normalized_states,
                    query_states=query_states,
                    kv_cache_layer=(fake_key_cache, fake_value_cache),
                    causal_mask=adjusted_causal_mask,
                    current_pos=current_pos,
                    layer_idx=layer_idx,
                )
            else:
                attn_output = layer.self_attn.forward_prefill(
                    hidden_states=normalized_states,
                    query_states=query_states,
                    kv_cache_layer=(fake_key_cache, fake_value_cache),
                    causal_mask=adjusted_causal_mask,
                    layer_idx=layer_idx,
                )

        # Apply post_attention_layernorm before residual add
        attn_output = layer.post_attention_layernorm(attn_output)
        hidden_states = hidden_states + attn_output
        # Optional FP16 clamping (OFF by default - use weight scaling instead)
        if self.config.enable_residual_clamp:
            hidden_states = torch.clamp(hidden_states, min=-self.config.residual_clamp_value, max=self.config.residual_clamp_value)

        # Apply MLP with proper normalization (Gemma3: pre + post feedforward norms)
        residual = hidden_states
        hidden_states = layer.pre_feedforward_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = layer.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        # Optional FP16 clamping (OFF by default - use weight scaling instead)
        if self.config.enable_residual_clamp:
            hidden_states = torch.clamp(hidden_states, min=-self.config.residual_clamp_value, max=self.config.residual_clamp_value)

        return hidden_states

    def process_layer(self, layer_idx, hidden_states, position_ids, causal_mask, current_pos, layer_offset,
                      IN_PREFILL=False, IN_PREFILL_ROTATE=False, update_mask=None):
        """Process a single transformer layer, delegating to the appropriate mode-specific implementation"""
        if IN_PREFILL_ROTATE:
            return self.process_layer_prefill_rotate(
                layer_idx, hidden_states, position_ids, causal_mask, current_pos, layer_offset, update_mask
            )
        elif IN_PREFILL:
            return self.process_layer_prefill(
                layer_idx, hidden_states, position_ids, causal_mask, current_pos, layer_offset, update_mask
            )
        else:
            return self.process_layer_regular(layer_idx, hidden_states, position_ids, causal_mask, current_pos, layer_offset, update_mask)

    def process_layers(self, hidden_states, position_ids, causal_mask, current_pos, start_layer=0, end_layer=None,
                       IN_PREFILL=False, IN_PREFILL_ROTATE=False, update_mask=None):
        """Process a range of transformer layers"""
        if end_layer is None:
            end_layer = len(self.layers)

        layer_offset = 0
        if not ENABLE_UNIFIED_CACHE:
            layer_offset = start_layer

        for i in range(start_layer, end_layer):
            hidden_states = self.process_layer(
                i, hidden_states, position_ids,
                causal_mask, current_pos, layer_offset, IN_PREFILL, IN_PREFILL_ROTATE, update_mask
            )
        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
        update_mask: torch.Tensor | None = None,
        IN_PREFILL: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the transformer layers with KV cache support."""
        hidden_states = self.embed_tokens(input_ids)
        # Apply Gemma3 embedding scaling
        hidden_states = hidden_states * self.embedding_scale
        
        # Process layers (rotary embeddings are now retrieved per-layer based on layer type)
        hidden_states = self.process_layers(
            hidden_states, position_ids, causal_mask,
            current_pos, start_layer=0, end_layer=None, IN_PREFILL=IN_PREFILL, update_mask=update_mask,
        )

        # Always apply final normalization - critical for correct model output
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def forward_prefill(
        self,
        hidden_states,
        position_ids=None,
        causal_mask=None,
        current_pos=None,
        start_layer=None,
        end_layer=None,
        update_mask=None,
    ):
        """
        Forward pass for prefilling KV cache
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Note: rotary embeddings are now retrieved per-layer based on layer type
        
        # Process layers within the specified range if provided
        if start_layer is not None and end_layer is not None:
            hidden_states = self.process_layers(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                start_layer=start_layer,
                end_layer=end_layer,
                IN_PREFILL=True,
                update_mask=update_mask,
            )
        else:
            # Process all layers for non-split mode
            hidden_states = self.process_layers(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=True,
                update_mask=update_mask,
            )
        
        # Apply final normalization if this is the last block
        if end_layer is None or end_layer == len(self.layers):
            hidden_states = self.norm(hidden_states)

        return hidden_states

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_pretrained_weights(self, model_path: str) -> bool:
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
            # Handle HF checkpoints that wrap weights under language_model.model.*
            if k.startswith("language_model."):
                k = k[len("language_model."):]
            elif not k.startswith("model."):
                # Skip non-language weights (e.g., vision_tower.*)
                continue
            if k.startswith("model."):
                k = k[len("model."):]
            new_k = k
            if "lm_head.weight" in new_k:
                continue
            if any(
                proj in new_k
                for proj in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                ]
            ):
                conv_state[new_k] = v.view(v.shape[0], v.shape[1], 1, 1)
            else:
                conv_state[new_k] = v

        missing, unexpected = self.load_state_dict(conv_state, strict=False)
        # Filter out expected missing keys
        # - rotary embeddings are computed, not loaded
        # - KV cache buffer is initialized separately
        missing = [m for m in missing if "rotary_emb" not in m and "kv_cache" not in m]

        # Filter out unexpected keys that are actually expected from HF format differences
        # (none expected currently - all layer norms should now be present)

        allow_missing = os.environ.get("ANEMLL_ALLOW_MISSING_WEIGHTS", "").lower() in ("1", "true", "yes")
        if missing:
            print("Missing keys", missing)
            if unexpected:
                print("Unexpected keys", unexpected)
            # Highlight actionable TODO in red for conversion logs
            print("\033[91mTODO: Weights not found or renamed. Check checkpoint prefixes (e.g., language_model.*) and config.\033[0m")
            print("Hint: set ANEMLL_ALLOW_MISSING_WEIGHTS=1 (or --allow-missing-weights in convert scripts) to continue anyway.")
            if allow_missing:
                print("Continuing despite missing weights (ANEMLL_ALLOW_MISSING_WEIGHTS=1).")
                return True
            raise RuntimeError("Failed to load Gemma3 weights: missing keys.")
        if unexpected:
            print("Unexpected keys", unexpected)
        return True


class Gemma3ForCausalLM(nn.Module):
    config_class = Gemma3Config

    def __init__(self, config: Gemma3Config, enable_coreml=False, disable_kv_cache=False, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.enable_coreml = enable_coreml
        self.disable_kv_cache = disable_kv_cache or DISABLE_KV_CACHE
        
        # Update global ENABLE_COREML flag when instance is created with enable_coreml=True
        if enable_coreml:
            global ENABLE_COREML
            ENABLE_COREML = True
            print(f"Set global ENABLE_COREML = {ENABLE_COREML} for CoreML conversion")
        
        self.model = Gemma3Model(config)
        # Set the disable_kv_cache flag on the model
        self.model.disable_kv_cache = self.disable_kv_cache
        
        # Initialize lm_head as Conv2d for ANE optimization following llama_model.py pattern
        if ENABLE_CONV2D:
            if ENABLE_VACAB_SPLIT16:
                vocab_split = config.vocab_size // 16
                vocab_remainder = config.vocab_size % 16
                # Create 16 heads, with the first ones handling any remainder
                for i in range(16):
                    split_size = vocab_split + (1 if i < vocab_remainder else 0)
                    setattr(self, f"lm_head16_{i+1}", 
                           nn.Conv2d(config.hidden_size, split_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE))
                if not hasattr(Gemma3ForCausalLM, '_lm_head_printed'):
                    print("Created lm_head16_1 through lm_head16_16")
                    Gemma3ForCausalLM._lm_head_printed = True
            elif ENABLE_VACAB_SPLIT8:
                vocab_split = config.vocab_size // 8
                vocab_remainder = config.vocab_size % 8
                # Create 8 heads, with the last one handling any remainder
                for i in range(8):
                    split_size = vocab_split + (1 if i < vocab_remainder else 0)
                    setattr(self, f"lm_head8_{i+1}", 
                           nn.Conv2d(config.hidden_size, split_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE))
                print("Created lm_head8_1 through lm_head8_8")
            elif ENABLE_VACAB_SPLIT:
                self.lm_head2_1 = nn.Conv2d(config.hidden_size, config.vocab_size//2, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head2_2 = nn.Conv2d(config.hidden_size, config.vocab_size//2, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                print("Created lm_head2_1 and lm_head2_2")
            else:
                self.lm_head1 = nn.Conv2d(config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                print("Created lm_head1")
        else:
            # Use linear head
            self.lm_head = nn.Conv2d(
                config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE
            ).to(TEST_DEVICE)
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
        if not ENABLE_COREML:
            if not IN_PREFILL:
                assert position_ids.ndim in (1, 2), "position_ids must be 1D or 2D"
            else:
                assert (
                    position_ids.shape[-1] == input_ids.shape[-1]
                ), "position_ids length must match input_ids in prefill"

        if self.disable_kv_cache:
            # Use the same forward path as KV cache, but without cache operations
            # This ensures identical attention computation
            hidden_states = self.model(
                input_ids,
                causal_mask,
                position_ids,
                current_pos,
                update_mask,
                IN_PREFILL=IN_PREFILL,
            )
        else:
            # Standard KV cache path
            hidden_states = self.model(
                input_ids,
                causal_mask,
                position_ids,
                current_pos,
                update_mask,
                IN_PREFILL=IN_PREFILL,
            )
        
        # Extract hidden states at current position right before LM head
        if not IN_PREFILL and current_pos is not None:
            # For single token generation, extract the last (and only) position from hidden_states
            # hidden_states has shape [batch, 1, hidden_size] for single token generation
            seq_len = hidden_states.shape[1]
            if seq_len == 1:
                # Single token case - use position 0 (the only position available)
                pos_tensor = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
            else:
                # Multi-token case - use the actual current_pos (for compatibility)
                if isinstance(current_pos, torch.Tensor):
                    pos_tensor = current_pos if current_pos.dim() > 0 else current_pos.unsqueeze(0)
                else:
                    pos_tensor = torch.tensor([current_pos], device=hidden_states.device, dtype=torch.long)
            
            # Use index_select for position extraction
            hidden_states = torch.index_select(hidden_states, dim=1, index=pos_tensor)  # [batch, 1, hidden_size]
        
        # Project to vocabulary using appropriate head
        if ENABLE_CONV2D:
            # Reshape for Conv2d and ensure float16
            hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
            
            if ENABLE_VACAB_SPLIT16:
                # Use 16-way split head
                logits1 = self.lm_head16_1(hidden_states).squeeze(2).transpose(1, 2)
                logits2 = self.lm_head16_2(hidden_states).squeeze(2).transpose(1, 2)
                logits3 = self.lm_head16_3(hidden_states).squeeze(2).transpose(1, 2)
                logits4 = self.lm_head16_4(hidden_states).squeeze(2).transpose(1, 2)
                logits5 = self.lm_head16_5(hidden_states).squeeze(2).transpose(1, 2)
                logits6 = self.lm_head16_6(hidden_states).squeeze(2).transpose(1, 2)
                logits7 = self.lm_head16_7(hidden_states).squeeze(2).transpose(1, 2)
                logits8 = self.lm_head16_8(hidden_states).squeeze(2).transpose(1, 2)
                logits9 = self.lm_head16_9(hidden_states).squeeze(2).transpose(1, 2)
                logits10 = self.lm_head16_10(hidden_states).squeeze(2).transpose(1, 2)
                logits11 = self.lm_head16_11(hidden_states).squeeze(2).transpose(1, 2)
                logits12 = self.lm_head16_12(hidden_states).squeeze(2).transpose(1, 2)
                logits13 = self.lm_head16_13(hidden_states).squeeze(2).transpose(1, 2)
                logits14 = self.lm_head16_14(hidden_states).squeeze(2).transpose(1, 2)
                logits15 = self.lm_head16_15(hidden_states).squeeze(2).transpose(1, 2)
                logits16 = self.lm_head16_16(hidden_states).squeeze(2).transpose(1, 2)
                
                if self.enable_coreml and ENABLE_LOGITS2:
                    return logits1, logits2, logits3, logits4, logits5, logits6, logits7, logits8, logits9, logits10, logits11, logits12, logits13, logits14, logits15, logits16
                else:
                    logits = torch.cat([logits1, logits2, logits3, logits4, logits5, logits6, logits7, logits8, logits9, logits10, logits11, logits12, logits13, logits14, logits15, logits16], dim=2)
            
            elif ENABLE_VACAB_SPLIT8:
                # Use 8-way split head
                logits1 = self.lm_head8_1(hidden_states).squeeze(2).transpose(1, 2)
                logits2 = self.lm_head8_2(hidden_states).squeeze(2).transpose(1, 2)
                logits3 = self.lm_head8_3(hidden_states).squeeze(2).transpose(1, 2)
                logits4 = self.lm_head8_4(hidden_states).squeeze(2).transpose(1, 2)
                logits5 = self.lm_head8_5(hidden_states).squeeze(2).transpose(1, 2)
                logits6 = self.lm_head8_6(hidden_states).squeeze(2).transpose(1, 2)
                logits7 = self.lm_head8_7(hidden_states).squeeze(2).transpose(1, 2)
                logits8 = self.lm_head8_8(hidden_states).squeeze(2).transpose(1, 2)
                
                if self.enable_coreml and ENABLE_LOGITS2:
                    return logits1, logits2, logits3, logits4, logits5, logits6, logits7, logits8
                else:
                    logits = torch.cat([logits1, logits2, logits3, logits4, logits5, logits6, logits7, logits8], dim=2)
            
            elif ENABLE_VACAB_SPLIT:
                # Use 2-way split head
                logits1 = self.lm_head2_1(hidden_states).squeeze(2).transpose(1, 2)
                logits2 = self.lm_head2_2(hidden_states).squeeze(2).transpose(1, 2)
                
                if self.enable_coreml and ENABLE_LOGITS2:
                    return logits1, logits2
                
                logits = torch.cat([logits1, logits2], dim=2)
            
            else:
                # Use single head
                logits = self.lm_head1(hidden_states).squeeze(2).transpose(1, 2)
        else:
            # Use linear head (fallback)
            logits = self.lm_head(hidden_states.permute(0, 2, 1).unsqueeze(2))
            logits = logits.squeeze(2).permute(0, 2, 1)
        
        return logits

    def prefill_kv_cache(self, input_ids, position_ids, start_pos, causal_mask):
        """
        Pre-fills KV cache for a batch of tokens starting from start_pos.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_length]
            position_ids: Position IDs for the sequence
            start_pos: Starting position in the KV cache
            causal_mask: Causal attention mask
            
        Returns:
            None (updates KV cache in-place)
        """
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings and run through model
        hidden_states = self.model.embed_tokens(input_ids)
        # Apply Gemma3 embedding scaling
        hidden_states = hidden_states * self.model.embedding_scale
        hidden_states = hidden_states.to(MODEL_DTYPE)

        # Get correct causal mask for the sequence
        # For prefill, each token should attend to all previous tokens in the sequence
        if causal_mask is not None:
            # Take the full sequence slice of causal mask
            causal_mask_prefill = causal_mask[:, :, :seq_length, :]
        else:
            causal_mask_prefill = None
        
        # Build update mask for batch prefill writes
        mask_len = max(
            self.model.config.state_length,
            getattr(self.model.config, "sliding_window", 0) or 0,
        )
        update_mask = torch.zeros(
            (1, 1, mask_len, seq_length),
            dtype=MODEL_DTYPE,
            device=TEST_DEVICE,
        )
        for i in range(seq_length):
            pos = start_pos + i
            if pos < mask_len:
                update_mask[0, 0, pos, i] = 1.0

        # Process through model to update KV cache
        with torch.no_grad():
            self.model.forward_prefill(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask_prefill,
                current_pos=start_pos,
                update_mask=update_mask,
            )

    def load_pretrained_weights(self, model_path: str) -> bool:
        if not self.model.load_pretrained_weights(model_path):
            return False
        
        # Load lm_head weights with splitting support
        state_dict: Dict[str, torch.Tensor] = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(
                    safetensors.torch.load_file(os.path.join(model_path, file))
                )
        
        # Handle lm_head weight (following llama_model.py pattern)
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
        
        # Handle lm_head weight loading and splitting
        lm_head_weight = None
        for k, v in state_dict.items():
            if k == "lm_head.weight":
                lm_head_weight = v
                break
        
        if lm_head_weight is not None:
            if ENABLE_CONV2D:
                reshaped_weight = lm_head_weight.view(lm_head_weight.shape[0], lm_head_weight.shape[1], 1, 1)
                if ENABLE_VACAB_SPLIT16:
                    vocab_split = self.config.vocab_size // 16
                    vocab_remainder = self.config.vocab_size % 16
                    # Create splits with proper sizes, distributing remainder among first splits
                    split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(16)]
                    splits = torch.split(reshaped_weight, split_sizes)
                    for i, split in enumerate(splits):
                        getattr(self, f"lm_head16_{i+1}").weight.data.copy_(split)
                        print(f"Loaded lm_head16_{i+1}.weight with shape {split.shape}")
                elif ENABLE_VACAB_SPLIT8:
                    vocab_split = self.config.vocab_size // 8
                    vocab_remainder = self.config.vocab_size % 8
                    # Create splits with proper sizes, distributing remainder among first splits
                    split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(8)]
                    splits = torch.split(reshaped_weight, split_sizes)
                    for i, split in enumerate(splits):
                        getattr(self, f"lm_head8_{i+1}").weight.data.copy_(split)
                        print(f"Loaded lm_head8_{i+1}.weight with shape {split.shape}")
                elif ENABLE_VACAB_SPLIT:
                    vocab_split = self.config.vocab_size // 2
                    split1, split2 = torch.split(reshaped_weight, [vocab_split, self.config.vocab_size - vocab_split])
                    self.lm_head2_1.weight.data.copy_(split1)
                    self.lm_head2_2.weight.data.copy_(split2)
                    print(f"Loaded lm_head2_1.weight and lm_head2_2.weight")
                else:
                    self.lm_head1.weight.data.copy_(reshaped_weight)
                    print(f"Loaded lm_head1.weight")
            else:
                self.lm_head.weight.data.copy_(lm_head_weight.view(lm_head_weight.shape[0], lm_head_weight.shape[1], 1, 1))
        else:
            print("Warning: lm_head.weight not found in model weights")
            return False
        
        return True

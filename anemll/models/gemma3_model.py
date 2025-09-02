"""Gemma 3 model implementation for ANEMLL.

This module provides a lightweight implementation of the Gemma 3 architecture
adapted to the Apple Neural Engine restrictions.  All dense layers are expressed
as ``nn.Conv2d`` with ``kernel_size=1`` and weights are loaded from Hugging Face
checkpoints with the correct reshaping.  Only the pieces required for the unit
 tests are implemented.
"""

from __future__ import annotations
from linecache import cache
import os
import json
import math
from typing import Dict, Optional
import copy

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gemma 3 model implementation adapted from qwen_model.py
# ---------------------------------------------------------------------------

def gelu_pytorch_tanh(x):
    """
    A fast GELU implementation approximation from the original Gemma implementation.
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

ACT2FN = {
    "gelu_pytorch_tanh": gelu_pytorch_tanh,
    "gelu": F.gelu,
    "silu": F.silu,
}

MODEL_DTYPE = torch.float16
TEST_DEVICE = "cpu"
CONTEXT_LENGTH = 1024

# Cache configuration constants (following llama_model.py pattern)
FORCE_UNIFIED_CACHE = True  # Force using a single unified KV cache
ENABLE_UNIFIED_CACHE = True  # Enable unified KV cache by default
STATE_LENGTH = 512   # KV cache state length
DISABLE_KV_CACHE = True  # Disable KV cache for simple testing

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
        self.bos_token_id = kwargs.get("bos_token_id", 2)
        self.eos_token_id = kwargs.get("eos_token_id", 1)
        self.hidden_act = kwargs.get("hidden_act", "gelu_pytorch_tanh")
        #self.hidden_size = kwargs.get("hidden_size", 2304)
        self.hidden_size = kwargs.get("hidden_size", 2048)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.intermediate_size = kwargs.get("intermediate_size", 9216)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 131072)
        self.model_type = kwargs.get("model_type", "gemma3")
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 26)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get(
            "head_dim",
            256
        )
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-06)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        if self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling.get("rope_type", "gemma3")
        self.rope_theta = kwargs.get("rope_theta", 1000000.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.torch_required = kwargs.get("torch_dtype", "bfloat16")
        self.transformers_version = kwargs.get("transformers_version", "4.40.0.dev0")
        self.use_cache = kwargs.get("use_cache", True)
        self.vocab_size = kwargs.get("vocab_size", 262208)
        self.context_length = kwargs.get("context_length", CONTEXT_LENGTH)
        self.state_length = kwargs.get("state_length", STATE_LENGTH)
        self.pad_token_id = kwargs.get("pad_token_id",0)
        self.query_pre_attn_scalar = kwargs.get("query_pre_attn_scalar",256)
        self.sliding_window = kwargs.get("sliding_window",4096)
        self.final_logit_softcapping = kwargs.get("final_logit_softcapping",None)
        self.attn_logit_softcapping = kwargs.get("attn_logit_softcapping",None)
        self.cache_implementation = kwargs.get("cache_implementation","hybrid")
        self.rope_local_base_freq = kwargs.get("rope_local_base_freq",10000.0)
        self.sliding_window_pattern = kwargs.get("sliding_window_pattern",6)
        self.layer_types: list[str] = kwargs.get(
            "layer_types",
            ["attention"] * self.num_hidden_layers  # Default to standard attention if not provided
        )

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def get_kv_cache_idx(layer_idx, num_layers, num_groups=1):
    layers_per_group = num_layers // num_groups
    group_idx = layer_idx // layers_per_group
    layer_in_group_idx = layer_idx % layers_per_group
    return group_idx, layer_in_group_idx, layers_per_group


# -----------------------------------------------------------------------------
# Gemma3 building blocks
# -----------------------------------------------------------------------------

def _rope_init_default(config, device=None):
    """
    Default RoPE initialization function.
    """
    dim = config.head_dim
    base = config.rope_theta
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    attention_scaling = 1.0
    return inv_freq, attention_scaling

ROPE_INIT_FUNCTIONS = {
    "default": _rope_init_default,
    "gemma3": _rope_init_default,
}


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, 1, bias=False)
        self.up_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, 1, bias=False)
        self.down_proj = nn.Conv2d(self.intermediate_size, self.hidden_size, 1, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # Reshape for Conv2d: [batch, seq_len, hidden_size] -> [batch, hidden_size, seq_len, 1]
        x = x.transpose(1, 2).unsqueeze(-1)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # Reshape back: [batch, hidden_size, seq_len, 1] -> [batch, seq_len, hidden_size]
        return down_proj.squeeze(-1).transpose(1, 2)

class Gemma3RMSNorm(nn.Module):
    """Manual RMSNorm implementation with explicit dtype handling for JIT compatibility."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # Cast to float32 for high-precision calculation
        hidden_states = hidden_states.to(torch.float32)
        
        # Manual RMSNorm calculation in float32
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        # Apply weight (also in float32) and cast back to original dtype
        return (hidden_states * self.weight.to(torch.float32)).to(input_dtype)


class Gemma3HeadNorm(nn.Module):
    """Manual RMSNorm implementation with explicit dtype handling for JIT compatibility."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # Cast to float32 for high-precision calculation
        hidden_states = hidden_states.to(torch.float32)
        
        # Manual RMSNorm calculation in float32
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        # Apply weight (also in float32) and cast back to original dtype
        return (hidden_states * self.weight.to(torch.float32)).to(input_dtype)


class Gemma3RotaryEmbedding(nn.Module):
    """Simple rotary positional embedding."""

    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Gemma3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    #@dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # Ensure position_ids is 2D for consistent processing
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then set unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights



class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types#[layer_idx]
        # pass layer_idx so Gemma3Attention can pick per-layer settings
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

   
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask:torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: [Cache] = None, # type: ignore
        output_attentions: [bool] = False,# type: ignore
        use_cache: [bool] = False, # type: ignore
        cache_position: [torch.LongTensor] = None, # type: ignore
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        hidden_states = hidden_states.to(MODEL_DTYPE)
        
       
        position_embeddings_global, position_embeddings_local = position_embeddings
        position_embeddings_global = position_embeddings_global.to(MODEL_DTYPE)
        position_embeddings_local = position_embeddings_local.to(MODEL_DTYPE)
        position_embeddings = (position_embeddings_global, position_embeddings_local)

        if attention_mask is not None:
            attention_mask = attention_mask.to(MODEL_DTYPE)

        # 1. Self-Attention block
        residual = hidden_states
        normed_hidden_states = self.input_layernorm(hidden_states)
        attn_outputs, self_attn_weights = self.self_attn(
            hidden_states=normed_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + attn_outputs

        # 2. MLP (Feed-Forward) block
        residual = hidden_states
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(normed_hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
        
class Gemma3TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)

class Gemma3TextModel(nn.Module):
    config: Gemma3Config

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        # create ModuleList with per-layer indices so Gemma3DecoderLayer gets layer_idx
        self.layers = nn.ModuleList([Gemma3DecoderLayer(config, i) for i in range(self.config.num_hidden_layers)])
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # TODO: raushan fix this after RoPE refactor. For now we hack it by reassigning thetas
        # when we want to create a local RoPE layer. Config defaults should hold values for global RoPE
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        # self.post_init() # This was causing the error and is not needed

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
            new_k = k.replace("model.", "") if k.startswith("model.") else k
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
                # Reshape linear weights to Conv2d weights
                conv_state[new_k] = v.view(v.shape[0], v.shape[1], 1, 1)
            else:
                conv_state[new_k] = v

        missing, unexpected = self.load_state_dict(conv_state, strict=False)
        # Filter out expected missing keys
        missing = [m for m in missing if "rotary_emb.inv_freq" not in m]
        if missing or unexpected:
            print("Missing keys in model:", missing)
            print("Unexpected keys in model:", unexpected)
        return not missing and not unexpected

    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
        IN_PREFILL: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        # create position embeddings to be shared across the decoder layers
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        for decoder_layer in self.layers:
            # Determine which position embeddings to use for the layer
            if decoder_layer.self_attn.is_sliding:
                position_embeddings = position_embeddings_local
            else:
                position_embeddings = position_embeddings_global

            # The forward pass of Gemma3DecoderLayer expects a single position_embeddings argument
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Conv2d(
            config.hidden_size, config.num_attention_heads * self.head_dim, 1, bias=config.attention_bias
        )
        self.k_proj = nn.Conv2d(
            config.hidden_size, config.num_key_value_heads * self.head_dim, 1, bias=config.attention_bias
        )
        self.v_proj = nn.Conv2d(
            config.hidden_size, config.num_key_value_heads * self.head_dim, 1, bias=config.attention_bias
        )
        self.o_proj = nn.Conv2d(
            config.num_attention_heads * self.head_dim, config.hidden_size, 1, bias=config.attention_bias
        )
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_norm = Gemma3RMSNorm(hidden_size=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(hidden_size=config.head_dim, eps=config.rms_norm_eps)
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[cache] = None, # type: ignore
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        
        # Ensure all inputs are on the correct dtype
        hidden_states = hidden_states.to(MODEL_DTYPE)
        cos, sin = position_embeddings
        cos = cos.to(MODEL_DTYPE)
        sin = sin.to(MODEL_DTYPE)
        position_embeddings = (cos, sin)

        if attention_mask is not None:
            attention_mask = attention_mask.to(MODEL_DTYPE)

        bsz, q_len, _ = hidden_states.size()

        # Reshape for Conv2d: [batch, seq_len, hidden_size] -> [batch, hidden_size, seq_len, 1]
        hidden_states_conv = hidden_states.transpose(1, 2).unsqueeze(-1)
        
        
        query_states = self.q_proj(hidden_states_conv).squeeze(-1).transpose(1, 2).to(MODEL_DTYPE)
        key_states = self.k_proj(hidden_states_conv).squeeze(-1).transpose(1, 2).to(MODEL_DTYPE)
        value_states = self.v_proj(hidden_states_conv).squeeze(-1).transpose(1, 2).to(MODEL_DTYPE)

        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply rotary embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat KV heads if using GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if self.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping

        if attention_mask is not None:
            if attention_mask.size() != attn_weights.size():
                attention_mask = attention_mask[:, :, :q_len, :key_states.shape[-2]]
            attn_weights = attn_weights + attention_mask

        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=False)
        
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.config.num_attention_heads * self.head_dim)

        # Reshape for o_proj Conv2d
        attn_output_conv = attn_output.transpose(1, 2).unsqueeze(-1)
        attn_output = self.o_proj(attn_output_conv).squeeze(-1).transpose(1, 2)

        return attn_output, attn_weights
    

           


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
        
        self.model = Gemma3TextModel(config)
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
                IN_PREFILL=IN_PREFILL,
            )
        else:
            # Standard KV cache path
            hidden_states = self.model(
                input_ids,
                causal_mask,
                position_ids,
                current_pos,
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
        hidden_states = hidden_states.to(MODEL_DTYPE)

        # Get correct causal mask for the sequence
        # For prefill, each token should attend to all previous tokens in the sequence
        if causal_mask is not None:
            # Take the full sequence slice of causal mask
            causal_mask_prefill = causal_mask[:, :, :seq_length, :]
        else:
            causal_mask_prefill = None
        
        # Process through model to update KV cache
        with torch.no_grad():
            self.model.forward_prefill(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask_prefill,
                current_pos=start_pos
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

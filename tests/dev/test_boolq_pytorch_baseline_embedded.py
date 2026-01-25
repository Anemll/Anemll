#!/usr/bin/env python3
"""
Standalone test using ANEMLL PyTorch Qwen2.5 model (with embedded code) to debug
probabilities for BoolQ prompt tokens ' no' and ' yes'
"""

import os
import sys
import json
import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import safetensors.torch

# Set offline mode to prevent network calls
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

# Performance optimizations
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# PyTorch performance optimizations
torch.set_num_threads(4)
if torch.backends.mkldnn.is_available():
    torch.backends.mkldnn.enabled = True

# Embedded ANEMLL model constants and configuration
MODEL_DTYPE = torch.float16
TEST_DEVICE = "mps"
CONTEXT_LENGTH = 1024

# Cache configuration constants
FORCE_UNIFIED_CACHE = True
ENABLE_UNIFIED_CACHE = True
STATE_LENGTH = CONTEXT_LENGTH
DISABLE_KV_CACHE = False

# LM head configuration constants
ENABLE_CONV2D = bool(1)
ENABLE_VACAB_SPLIT = bool(0)  # Disable 2-way split
ENABLE_VACAB_SPLIT8 = bool(0)
ENABLE_VACAB_SPLIT16 = bool(1)  # Enable 16-way split for MPS compatibility
ENABLE_LOGITS2 = bool(0)  # Disable for evaluation simplicity
ENABLE_COREML = bool(0)



def debug_print_tensor(tensor: torch.Tensor, name: str, desc: str, id: int = -1) -> None:
    return
    """Debug print helper for tensor stats with optional layer ID"""
    print(f"\n--- Debug: {name} ({desc}) {'[Layer ' + str(id) + ']' if id >= 0 else ''} ---")
    print(f"Shape: {tensor.shape}")
    print(f"Norm: {torch.norm(tensor).item():.4f}")
    print(f"First 5: {tensor.flatten()[:5].tolist()}")


def ane_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Custom softmax implementation optimized for fp16 numerical stability."""
    if True:  # True - softmax, False - experimental softmax-SM
        return F.softmax(x, dim=dim)
    
    # Convert to fp32 for numerical stability during computation
    x_fp32 = x.float()
    
    # Stable softmax: subtract max before exponential
    x_max = torch.max(x_fp32, dim=dim, keepdim=True)[0]
    x_shifted = x_fp32 - x_max
    
    # Clamp shifted values to prevent extreme exponentials
    x_shifted = x_shifted.clamp(min=-18)
    
    # Compute exponential and normalize
    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    softmax = exp_x / (sum_exp)
    
    # Convert back to model dtype
    return softmax.to(MODEL_DTYPE)

class Qwen25Config:
    def __init__(self, **kwargs):
        self.architectures = kwargs.get("architectures", ["Qwen2ForCausalLM"])
        self.attention_bias = kwargs.get("attention_bias", True)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.bos_token_id = kwargs.get("bos_token_id", 151643)
        self.eos_token_id = kwargs.get("eos_token_id", 151645)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.hidden_size = kwargs.get("hidden_size", 896)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.intermediate_size = kwargs.get("intermediate_size", 4864)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.model_type = kwargs.get("model_type", "qwen2")
        self.num_attention_heads = kwargs.get("num_attention_heads", 14)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 24)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 2)
        self.head_dim = kwargs.get(
            "head_dim",
            self.hidden_size // max(1, self.num_attention_heads),
        )
        self.pretraining_tp = kwargs.get("pretraining_tp", 1)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-06)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        if self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling.get("rope_type", "default")
        self.base = kwargs.get("rope_theta", 1000000.0)
        if self.rope_scaling and 'factor' in self.rope_scaling:
            self.base = self.base * self.rope_scaling['factor']
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.torch_required = kwargs.get("torch_dtype", "bfloat16")
        self.transformers_version = kwargs.get("transformers_version", "4.37.0")
        self.use_cache = kwargs.get("use_cache", True)
        self.vocab_size = kwargs.get("vocab_size", 151936)
        self.context_length = kwargs.get("context_length", CONTEXT_LENGTH)
        self.state_length = kwargs.get("state_length", STATE_LENGTH)
        self.use_sliding_window = kwargs.get("use_sliding_window", False)
        self.sliding_window = kwargs.get("sliding_window", 32768)
        self.max_window_layers = kwargs.get("max_window_layers", 28)

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

def get_kv_cache_idx(layer_idx, num_layers, num_groups=1):
    """Helper function to get KV cache indices."""
    layers_per_group = num_layers // num_groups
    group_idx = layer_idx // layers_per_group
    layer_in_group_idx = layer_idx % layers_per_group
    return group_idx, layer_in_group_idx, layers_per_group

class Qwen25RMSNorm(nn.Module):
    """RMSNorm used in Qwen 2.5 models - ANE-aware implementation with mean subtraction."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(MODEL_DTYPE)

class Qwen25RotaryEmbedding(nn.Module):
    """Simple rotary positional embedding for Qwen 2.5."""

    def __init__(self, config: Qwen25Config) -> None:
        super().__init__()
        self.dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        
        # Apply rope_scaling factor if present
        self.base = getattr(config, 'rope_theta', 1000000.0)
        if hasattr(config, 'rope_scaling') and config.rope_scaling and 'factor' in config.rope_scaling:
            self.base = getattr(config, 'rope_theta', 1000000.0) * config.rope_scaling['factor']
        
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim).to(torch.float32)
        )

        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(config.max_position_embeddings, device=TEST_DEVICE).type_as(self.inv_freq)
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

# Complete model classes
class Qwen25MLP(nn.Module):
    def __init__(self, config: Qwen25Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Use single Conv2d layers (no splitting for Qwen 2.5 for now)
        self.gate_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, kernel_size=1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.up_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, kernel_size=1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.down_proj = nn.Conv2d(self.intermediate_size, self.hidden_size, kernel_size=1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)

        self.act_fn = F.silu

    def forward(self, x):
        # Use identical step-by-step computation to QwenMLP to prevent numerical explosion
        x = x.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)  # Ensure proper dtype and shape
        
        # Step-by-step computation for numerical stability (like QwenMLP)
        a = self.gate_proj(x)      # gate projection
        b = self.up_proj(x)        # up projection
        c = self.act_fn(a)         # activation on gate
        d = c * b                  # multiply gate * up
        e = self.down_proj(d)      # down projection
        
        return e.squeeze(2).permute(0, 2, 1)  # Final output shape: [bsz, seq_len, hidden_size]

class Qwen25Attention(nn.Module):
    def __init__(self, config: Qwen25Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.rotary_emb = Qwen25RotaryEmbedding(config)

        # Calculate correct projection dimensions
        q_proj_dim = self.num_heads * self.head_dim  # 14 * 64 = 896
        kv_proj_dim = self.num_kv_heads * self.head_dim  # 2 * 64 = 128
        
        self.q_proj = nn.Conv2d(self.hidden_size, q_proj_dim, 1, bias=True, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.k_proj = nn.Conv2d(self.hidden_size, kv_proj_dim, 1, bias=True, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.v_proj = nn.Conv2d(self.hidden_size, kv_proj_dim, 1, bias=True, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.o_proj = nn.Conv2d(q_proj_dim, self.hidden_size, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.scale = 1 / math.sqrt(self.head_dim)

    def forward(self, hidden_states: torch.Tensor, causal_mask: torch.Tensor, 
                position_ids: torch.LongTensor, current_pos: torch.LongTensor) -> torch.Tensor:
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
        key_states = repeat_kv(key_states, n_rep)
        value_states = repeat_kv(value_states, n_rep)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights = (torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale)
        if causal_mask is not None:
            causal_mask_slice = causal_mask[:, :, :seq_len, :seq_len]
            attn_weights = attn_weights + causal_mask_slice.to(attn_weights.dtype)
        attn_weights = ane_softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, -1))
        out = self.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        return out.squeeze(2).permute(0, 2, 1)

class Qwen25DecoderLayer(nn.Module):
    def __init__(self, config: Qwen25Config) -> None:
        super().__init__()
        self.self_attn = Qwen25Attention(config)
        self.mlp = Qwen25MLP(config)
        self.input_layernorm = Qwen25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, causal_mask: torch.Tensor,
                position_ids: torch.LongTensor, current_pos: torch.LongTensor, layer_id: int = 0) -> torch.Tensor:
        residual = hidden_states
        debug_print_tensor(hidden_states, "hidden_states", "Before input_layernorm", layer_id)

        hidden_states = self.input_layernorm(hidden_states)
        debug_print_tensor(hidden_states, "hidden_states", "After input_layernorm", layer_id)
        hidden_states = self.self_attn(hidden_states, causal_mask, position_ids, current_pos)
        debug_print_tensor(hidden_states, "hidden_states", "After self_attn", layer_id)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        debug_print_tensor(hidden_states, "hidden_states", "After post_attention_layernorm", layer_id)
        hidden_states = self.mlp(hidden_states)
        debug_print_tensor(hidden_states, "hidden_states", "After mlp", layer_id)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen25Model(nn.Module):
    def __init__(self, config: Qwen25Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen25DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Move entire model to device
        device = torch.device(TEST_DEVICE)
        self.to(device)

    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, causal_mask, position_ids, current_pos, layer_id=i)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_pretrained_weights(self, model_path: str) -> bool:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(model_path)
        state_dict: Dict[str, torch.Tensor] = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(safetensors.torch.load_file(os.path.join(model_path, file)))

        conv_state = {}
        device = torch.device(TEST_DEVICE)
        for k, v in state_dict.items():
            new_k = k.replace("model.", "") if k.startswith("model.") else k
            if "lm_head.weight" in new_k:
                continue
            # Ensure all tensors are moved to the correct device and dtype
            v = v.to(device=device, dtype=MODEL_DTYPE)
            if any(proj in new_k for proj in ["q_proj.weight", "k_proj.weight", "v_proj.weight", 
                                              "o_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight"]):
                conv_state[new_k] = v.view(v.shape[0], v.shape[1], 1, 1)
            elif any(proj in new_k for proj in ["q_proj.bias", "k_proj.bias", "v_proj.bias", "o_proj.bias"]):
                conv_state[new_k] = v
            else:
                conv_state[new_k] = v

        missing, unexpected = self.load_state_dict(conv_state, strict=False)
        missing = [m for m in missing if "rotary_emb.inv_freq" not in m]
        return not missing and not unexpected

class Qwen25ForCausalLM(nn.Module):
    def __init__(self, config: Qwen25Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen25Model(config)
        
        # Initialize lm_head with 16-way split for MPS compatibility
        device = torch.device(TEST_DEVICE)
        if ENABLE_CONV2D and ENABLE_VACAB_SPLIT16:
            vocab_split = config.vocab_size // 16
            vocab_remainder = config.vocab_size % 16
            # Create 16 heads, with the first ones handling any remainder
            for i in range(16):
                split_size = vocab_split + (1 if i < vocab_remainder else 0)
                setattr(self, f"lm_head16_{i+1}", 
                       nn.Conv2d(config.hidden_size, split_size, 1, bias=False, dtype=MODEL_DTYPE))
            print("Created lm_head16_1 through lm_head16_16 for MPS compatibility")
        else:
            # Fallback to single LM head
            self.lm_head = nn.Conv2d(config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE)
        
        # Move entire model to device
        self.to(device)

    def forward(
        self,
        input_ids: torch.LongTensor,
        update_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        current_pos: torch.LongTensor,
        IN_PREFILL: bool = False,
    ) -> torch.Tensor:
        if IN_PREFILL:
            return self.forward_batched(input_ids, causal_mask, position_ids, current_pos)
        else:
            return self.forward_single_token(input_ids, causal_mask, position_ids, current_pos)

    def forward_batched(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
    ) -> torch.Tensor:
        return self.model(input_ids, causal_mask, position_ids, current_pos)

    def forward_single_token(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        current_pos: torch.LongTensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, causal_mask, position_ids, current_pos)
        
        print("\n--- Debug: Hidden States Before LM Head (Embedded) ---")
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Hidden states norm: {torch.norm(hidden_states).item():.4f}")
        print(f"Hidden states first 5: {hidden_states.flatten()[:5].tolist()}")

        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        
        # Use 16-way split LM head if enabled
        if ENABLE_CONV2D and ENABLE_VACAB_SPLIT16:
            # Apply all 16 LM heads and concatenate results
            logits_parts = []
            for i in range(16):
                logits_part = getattr(self, f"lm_head16_{i+1}")(hidden_states).squeeze(2).transpose(1, 2)
                logits_parts.append(logits_part)
            logits = torch.cat(logits_parts, dim=2)
        else:
            # Use single LM head
            logits = self.lm_head(hidden_states).squeeze(2).transpose(1, 2)
        
        print("\n--- Debug: Raw Logits from LM Head (Embedded) ---")
        print(f"Raw logits shape: {logits.shape}")
        print(f"Raw logits norm: {torch.norm(logits).item():.4f}")
        print(f"Raw logits first 5: {logits.flatten()[:5].tolist()}")
        
        return logits

    def load_pretrained_weights(self, model_path: str) -> bool:
        if not self.model.load_pretrained_weights(model_path):
            return False
        
        # Load lm_head weights
        state_dict: Dict[str, torch.Tensor] = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(safetensors.torch.load_file(os.path.join(model_path, file)))
        
        # Handle lm_head weight (check for tied embeddings)
        lm_head_weight = None
        if "lm_head.weight" in state_dict:
            lm_head_weight = state_dict["lm_head.weight"]
        elif self.config.tie_word_embeddings and "model.embed_tokens.weight" in state_dict:
            print("Using tied embeddings for lm_head")
            lm_head_weight = state_dict["model.embed_tokens.weight"]
        
        if lm_head_weight is not None:
            # Ensure LM head weight is on correct device and dtype
            device = torch.device(TEST_DEVICE)
            lm_head_weight = lm_head_weight.to(device=device, dtype=MODEL_DTYPE)
            
            # Handle weight loading for 16-way split or single LM head
            if ENABLE_CONV2D and ENABLE_VACAB_SPLIT16:
                reshaped_weight = lm_head_weight.view(lm_head_weight.shape[0], lm_head_weight.shape[1], 1, 1)
                vocab_split = self.config.vocab_size // 16
                vocab_remainder = self.config.vocab_size % 16
                # Create splits with proper sizes, distributing remainder among first splits
                split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(16)]
                splits = torch.split(reshaped_weight, split_sizes)
                for i, split in enumerate(splits):
                    getattr(self, f"lm_head16_{i+1}").weight.data.copy_(split)
                    print(f"Loaded lm_head16_{i+1}.weight with shape {split.shape}")
            else:
                # Single LM head
                self.lm_head.weight.data.copy_(lm_head_weight.view(lm_head_weight.shape[0], lm_head_weight.shape[1], 1, 1))
        else:
            print("Warning: lm_head.weight not found")
            return False
        
        return True

# Sample 31 from BoolQ (Benson & Hedges) - exact same as our evaluation
#context = 'Benson & Hedges -- Benson & Hedges is a British brand of cigarettes owned by either Philip Morris International, British American Tobacco, or Japan Tobacco, depending on the region. In the UK, they are registered in Old Bond Street in London, and are manufactured in Lisnafillan, Ballymena, Northern Ireland.\nQuestion: do they still make benson & hedges cigarettes?\nAnswer:'

context = """While some 19th-century experiments suggested that the underlying premise is true if the heating is sufficiently gradual, according to contemporary biologists the premise is false: a frog that is gradually heated will jump out. Indeed, thermoregulation by changing location is a fundamentally necessary survival strategy for frogs and other ectotherms.
Question: does a frog jump out of boiling water
Answer:"""



def test_boolq_pytorch_baseline_embedded(model):
    """Complete test with embedded ANEMLL PyTorch Qwen2.5 model for BoolQ baseline comparison"""
    
    
    
    # Tokenize context
    context_tokens = tokenizer.encode(context, add_special_tokens=True)
    no_token_id = tokenizer.encode(" no", add_special_tokens=False)[0]
    yes_token_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
    
    print(f"\nContext: {repr(context)}")
    print(f"Context tokens length: {len(context_tokens)}")
    print(f"First 5 context tokens: {context_tokens[:5]}")
    print(f"Last 5 context tokens: {context_tokens[-5:]}")
    print(f"Token IDs: ' no'={no_token_id}, ' yes'={yes_token_id}")
    
    # Create simple causal mask
    seq_len = len(context_tokens)
    device = torch.device(TEST_DEVICE)
    causal_mask = torch.full((1, 1, seq_len, seq_len), float('-inf')).to(device)
    for i in range(seq_len):
        causal_mask[0, 0, i, :i+1] = 0
    
    # Run inference - ensure all tensors are on the correct device
    input_ids = torch.tensor([context_tokens], dtype=torch.long).to(device)
    position_ids = torch.arange(seq_len, dtype=torch.long).to(device)
    current_pos = torch.tensor(seq_len - 1, dtype=torch.long).to(device)
    update_mask = torch.zeros((1, seq_len), dtype=torch.float16).to(device)  # Dummy update mask
    
    debug = True
    if debug:
            print(f"\n[DEBUG] Context: {repr(context)}")
            #print(f"[DEBUG] Continuations: {repr(cont1)} | {repr(cont2)}")
            print(f"[DEBUG] Context tokens length: {len(context_tokens)}")
            print(f"[DEBUG] First 5 context tokens: {context_tokens[:5]}")
            print(f"[DEBUG] Last 5 context tokens: {context_tokens[-5:]}")
            print(f"[DEBUG] Continuation token IDs: {no_token_id} | {yes_token_id}")

    with torch.no_grad():
        print(f"\nRunning inference...")
        outputs = model(input_ids, update_mask, position_ids, causal_mask, current_pos, IN_PREFILL=False)
        logits = outputs[0, -1, :]  # Last token logits
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract specific token probabilities
        no_prob = probs[no_token_id].item()
        yes_prob = probs[yes_token_id].item()
        no_log_prob = log_probs[no_token_id].item()
        yes_log_prob = log_probs[yes_token_id].item()
        
        # Get top predicted token
        top_token_id = torch.argmax(logits).item()
        top_token_text = tokenizer.decode([top_token_id])
        top_prob = probs[top_token_id].item()
        
        print(f"\nResults:")
        print(f"  ' no' probability: {no_prob:.6f}")
        print(f"  ' yes' probability: {yes_prob:.6f}")
        print(f"  ' no' log probability: {no_log_prob:.4f}")
        print(f"  ' yes' log probability: {yes_log_prob:.4f}")
        print(f"  Predicted: '{top_token_text}' (ID: {top_token_id}, prob: {top_prob:.6f})")
        
        # Determine which is higher
        if yes_log_prob > no_log_prob:
            print(f"  Model predicts: YES (score diff: {yes_log_prob - no_log_prob:.4f})")
        else:
            print(f"  Model predicts: NO (score diff: {no_log_prob - yes_log_prob:.4f})")
    
    print("=" * 80)
    print("Comparison with baseline and evaluation results:")
    print("  HF Transformers: no=-3.3420, yes=-2.8594 → Predicts YES ✅")
    print("  MLX:             no=-3.2088, yes=-2.7088 → Predicts YES ✅")
    # Show actual results from this test
    prediction_emoji = "✅" if yes_log_prob > no_log_prob else "❌"
    prediction_text = "YES" if yes_log_prob > no_log_prob else "NO"
    print(f"  PyTorch (embedded): no={no_log_prob:.4f}, yes={yes_log_prob:.4f} → Predicts {prediction_text} {prediction_emoji}")
    print("=" * 80)


def make_causal_mask(length, start):
    """Create causal attention mask."""
    mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
    row_indices = np.arange(length).reshape(length, 1)
    col_indices = np.arange(length).reshape(1, length)
    mask[:, :, col_indices <= (row_indices + start)] = 0
    return mask

def test_batch_prefill_single_token(model):
    """Test BoolQ sample #31 with batch prefill + single token workflow like ANEMLL baseline."""
    
    
    print("=" * 80)
    print("ANEMLL PyTorch Qwen2.5 BoolQ Batch Prefill + Single Token Test (Embedded Code)")
    print("=" * 80)
    
    # Tokenize context
    context_tokens = tokenizer.encode(context, add_special_tokens=True)
    no_token_id = tokenizer.encode(" no", add_special_tokens=False)[0]
    yes_token_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
    
    print(f"\nContext: {repr(context)}")
    print(f"Context tokens length: {len(context_tokens)}")
    print(f"First 5 context tokens: {context_tokens[:5]}")
    print(f"Last 5 context tokens: {context_tokens[-5:]}")
    print(f"Token IDs: ' no'={no_token_id}, ' yes'={yes_token_id}")
    
    # Proper batch prefill workflow like ANEMLL baseline
    context_length = CONTEXT_LENGTH
    if len(context_tokens) > context_length - 1:
        context_tokens = context_tokens[-(context_length - 1):]


    #---------------------------------
    device = "cpu" #next(self.model.parameters()).device
    #with torch.no_grad():
    prompt_length = len(context_tokens)

        
    causal_mask_data = make_causal_mask(CONTEXT_LENGTH, 0)
    causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
    
    # Batch prefill
    input_ids = torch.tensor([context_tokens], dtype=torch.long)   
    # Run batched prefill
    batch_pos = 0
    batch_size = 128
    while batch_pos < prompt_length:
        batch_end = min(batch_pos + batch_size, prompt_length)
        current_batch_size = batch_end - batch_pos
        
        # Get current batch
        batch_input = input_ids[:, batch_pos:batch_end]
        
        # Pad to batch size
        import torch.nn.functional as F
        batch_input = F.pad(batch_input, (0, batch_size - current_batch_size), value=0)
        
        # Position IDs and masks
        position_ids = torch.arange(batch_pos, batch_pos + batch_size, dtype=torch.long, device=device)
        batch_causal_mask = causal_mask[:, :, batch_pos:batch_pos + batch_size, :]

        debug = True
        if debug:
            print(f"\n[DEBUG] Context: {repr(context)}")
            #print(f"[DEBUG] Continuations: {repr(cont1)} | {repr(cont2)}")
            print(f"[DEBUG] Context tokens length: {len(context_tokens)}")
            print(f"[DEBUG] First 5 context tokens: {context_tokens[:5]}")
            print(f"[DEBUG] Last 5 context tokens: {context_tokens[-5:]}")
            print(f"[DEBUG] Continuation token IDs: {no_token_id} | {yes_token_id}")

        
        # Prefill
        model(
            batch_input,
            None,
            position_ids,
            batch_causal_mask,
            torch.tensor(batch_pos, dtype=torch.long, device=device),
            IN_PREFILL=True
        )
        
        batch_pos = batch_end
    
    # Single generation step to get logits
    current_pos = prompt_length
    last_token = torch.tensor([[context_tokens[-1]]], dtype=torch.long, device=device)
    
    outputs = model(
        last_token,
        None,
        torch.tensor([current_pos], dtype=torch.long, device=device),
        causal_mask[:, :, current_pos:current_pos+1, :],
        torch.tensor(current_pos, dtype=torch.long, device=device),
        IN_PREFILL=False
    )
    


    # Extract logits and compute log probabilities
    logits = outputs[0, -1, :]  # Last token logits  
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Get scores for both tokens
    no_log_prob = log_probs[no_token_id].item()
    yes_log_prob = log_probs[yes_token_id].item()

    #---------------------------------
        
    '''
    prompt_length = len(context_tokens)
    
    # Create full context mask for ANEMLL
    causal_mask = torch.full((1, 1, context_length, context_length), float('-inf'))
    for i in range(context_length):
        causal_mask[0, 0, i, :i+1] = 0
    
    print(f"\n=== BATCH PREFILL PHASE (ANEMLL Style) ===")
    
    with torch.no_grad():
        # Batch prefill
        input_ids = torch.tensor([context_tokens], dtype=torch.long)
        
        # Run batched prefill - this doesn't work with embedded model (no batch prefill logic)
        # So we'll simulate it with single pass
        print(f"Running batch prefill for {prompt_length} tokens...")
        position_ids = torch.arange(prompt_length, dtype=torch.long)
        current_pos = torch.tensor(prompt_length - 1, dtype=torch.long)
        
        # Simulate batch prefill (embedded model doesn't have real batch logic)
        batch_mask = causal_mask[:, :, :prompt_length, :prompt_length]
        outputs = model(input_ids, batch_mask, position_ids, current_pos)
        print(f"Batch prefill complete")
        
        # SINGLE TOKEN GENERATION PHASE  
        print(f"\n=== SINGLE TOKEN GENERATION PHASE ===")
        
        # Single generation step to get logits for " no"
        print(f"Testing ' no' token (ID: {no_token_id})...")
        current_pos = prompt_length
        last_token = torch.tensor([[context_tokens[-1]]], dtype=torch.long)
        
        # For embedded model, we need to extend the context with the test token
        extended_tokens = context_tokens + [no_token_id]
        extended_input = torch.tensor([extended_tokens], dtype=torch.long)
        extended_position_ids = torch.arange(len(extended_tokens), dtype=torch.long)
        extended_mask = causal_mask[:, :, :len(extended_tokens), :len(extended_tokens)]
        
        no_outputs = model(extended_input, extended_mask, extended_position_ids, torch.tensor(len(extended_tokens) - 1, dtype=torch.long))
        no_logits = no_outputs[0, -1, :]  # Last token logits
        no_log_prob = F.log_softmax(no_logits, dim=-1)[no_token_id].item()
        
        # Single generation step to get logits for " yes"
        print(f"Testing ' yes' token (ID: {yes_token_id})...")
        extended_tokens = context_tokens + [yes_token_id]
        extended_input = torch.tensor([extended_tokens], dtype=torch.long)
        extended_position_ids = torch.arange(len(extended_tokens), dtype=torch.long)
        extended_mask = causal_mask[:, :, :len(extended_tokens), :len(extended_tokens)]
        
        yes_outputs = model(extended_input, extended_mask, extended_position_ids, torch.tensor(len(extended_tokens) - 1, dtype=torch.long))
        yes_logits = yes_outputs[0, -1, :]  # Last token logits  
        yes_log_prob = F.log_softmax(yes_logits, dim=-1)[yes_token_id].item()
    '''
    
    print(f"\n=== BATCH PREFILL + SINGLE TOKEN RESULTS ===")
    print(f"  ' no' log probability: {no_log_prob:.4f}")
    print(f"  ' yes' log probability: {yes_log_prob:.4f}")
    
    # Determine which is higher
    if yes_log_prob > no_log_prob:
        print(f"  Model predicts: YES (score diff: {yes_log_prob - no_log_prob:.4f})")
    else:
        print(f"  Model predicts: NO (score diff: {no_log_prob - yes_log_prob:.4f})")
    
    print("=" * 80)
    #print("Comparison with other methods:")
    #print("  HF Transformers: no=-3.3420, yes=-2.8594 → Predicts YES ✅")
    #print("  MLX:             no=-3.2088, yes=-2.7088 → Predicts YES ✅")
    # Show actual results from this test
    prediction_emoji = "" # "✅" if yes_log_prob > no_log_prob else "❌"
    prediction_text = "YES" if yes_log_prob > no_log_prob else "NO"
    print(f"  Embedded (batch+single): no={no_log_prob:.4f}, yes={yes_log_prob:.4f} → Predicts {prediction_text} {prediction_emoji}")
    print("=" * 80)


if __name__ == "__main__":
    # Run both tests to compare single-pass vs batch prefill + single token
    print("Running single-pass inference test:")

    print("ANEMLL PyTorch Qwen2.5 BoolQ Complete Baseline Test (Embedded Code)")
    print("=" * 80)
    
    # Load model and tokenizer
    model_path = "Qwen/Qwen2.5-0.5B"
    print(f"Loading PyTorch Qwen2.5 model from: {model_path}")
    
    # Check if it's a local path or HuggingFace ID
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        try:
            print(f"Checking HuggingFace cache for {model_path}...")
            local_path = snapshot_download(repo_id=model_path, local_files_only=True)
            print(f"Found in cache: {local_path}")
            model_path = local_path
        except Exception:
            print(f"Not in cache, downloading {model_path}...")
            local_path = snapshot_download(repo_id=model_path)
            print(f"Downloaded to: {local_path}")
            model_path = local_path
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create config
    config = Qwen25Config.from_json(f'{model_path}/config.json')
    
    # Create and load model
    print(f"Creating embedded PyTorch model...")
    model = Qwen25ForCausalLM(config)
    model.debug = True
    print(f"Loading pretrained weights...")
    success = model.load_pretrained_weights(model_path)
    if not success:
        raise RuntimeError(f"Failed to load pretrained weights from {model_path}")
    
    model.eval()
    
    # Ensure cached tensors in rotary embeddings are on correct device
    device = torch.device(TEST_DEVICE)
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'rotary_emb'):
            rotary_emb = layer.self_attn.rotary_emb
            if hasattr(rotary_emb, 'cos_cached'):
                rotary_emb.cos_cached = rotary_emb.cos_cached.to(device)
            if hasattr(rotary_emb, 'sin_cached'):
                rotary_emb.sin_cached = rotary_emb.sin_cached.to(device)
    
    print(f"Model loaded successfully!")

    test_boolq_pytorch_baseline_embedded(model)
    
    print("\n" + "="*80)
    print("Running batch prefill + single token test:")
    #test_batch_prefill_single_token(model)

#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

from ..ane_converter.base_converter import BaseConverter
import coremltools as ct
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import math
import os
import safetensors
import gc
from tqdm import tqdm
import json
from .base_model import BaseModel
import torch.nn.init as init

# Model configuration constants
MODEL_DTYPE = torch.float16  # Hardcoded to float16 for ANE support
MLP_UP_SPLIT = 1   # Number of splits for MLP up-projection
MLP_DOWN_SPLIT = 1 # Number of splits for MLP down-projection
ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
    "swish": F.silu,
}
#TEST_DEVICE = "mps" #if torch.backends.mps.is_available() else "cpu"  # Device for testing
TEST_DEVICE = "cpu" 
# Context and state length configuration
CONTEXT_LENGTH = 512  # Default context window size for testing
STATE_LENGTH = 512   # KV cache state length

# Cache configuration
FORCE_UNIFIED_CACHE = True  # Force using a single unified KV cache
ENABLE_UNIFIED_CACHE = True  # Enable unified KV cache by default

# LM head configuration
ENABLE_CONV2D = bool(1)      # Use Conv2d for LM head
ENABLE_VACAB_SPLIT = bool(1)  # Split vocab into 2 parts
ENABLE_VACAB_SPLIT8 = bool(1)  # Split vocab into 8 parts
ENABLE_LOGITS2 = bool(0)    # Return 2 logits arrays
ENABLE_COREML = bool(0)     # CoreML-specific returns

# Debug flags
ENABLE_DEBUG =  bool(0)  # General debug info
ENABLE_DEBUG2 = bool(0)  # Detailed debug for single token generation
ENABLE_DEBUG3 = bool(0)  # Detailed debug for prefill mode
ENABLE_VALUES = bool(0)  # Print tensor values for debugging

class LlamaConfig:
    def __init__(self, **kwargs):
        self.architectures = kwargs.get("architectures", ["LlamaForCausalLM"])
        self.attention_bias = kwargs.get("attention_bias", False)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.bos_token_id = kwargs.get("bos_token_id", 128000)
        self.eos_token_id = kwargs.get("eos_token_id", 128001)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.hidden_size = kwargs.get("hidden_size", 4096)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.intermediate_size = kwargs.get("intermediate_size", 14336)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 8192)
        self.model_type = kwargs.get("model_type", "llama")
        self.num_attention_heads = kwargs.get("num_attention_heads", 32)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)
        self.pretraining_tp = kwargs.get("pretraining_tp", 1)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-05)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        if self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling.get("rope_type", "llama3")
        self.rope_theta = kwargs.get("rope_theta", 500000.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)
        self.torch_required = kwargs.get("torch_dtype", "bfloat16")
        self.transformers_version = kwargs.get("transformers_version", "4.40.0.dev0")
        self.use_cache = kwargs.get("use_cache", True)
        self.vocab_size = kwargs.get("vocab_size", 128257)  # Set to 128257 to match HF
        self.context_length = kwargs.get("context_length", CONTEXT_LENGTH)
        self.state_length = kwargs.get("state_length", STATE_LENGTH)

    @classmethod
    def from_json(cls, json_file):
        print(f"Loading config from {json_file}")
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in self.__dict__.items())

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.hidden_size = hidden_size
        #self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):

        if ENABLE_DEBUG:
            print(f"LlamaRMSNorm forward - self.weight.dtype: {self.weight.dtype}")
        mean = hidden_states.mean(-1, keepdim=True)
        hidden_states = hidden_states - mean

        return F.layer_norm(hidden_states, self.weight.shape, self.weight, bias=None, eps=float(self.eps)).to(TEST_DEVICE).to(MODEL_DTYPE)
        #Convert to 4D tensor
        #hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], 1, hidden_states.shape[2])
        input_dtype = hidden_states.dtype
        #hidden_states = hidden_states.to(MODEL_DTYPE)
        mean = hidden_states.mean(-1, keepdim=True)
        hidden_states = hidden_states - mean
        variance = hidden_states.pow(2).mean(-1, keepdim=True)         
        #variance = torch.clamp(variance, max=5e2)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states=(self.weight * hidden_states).to(input_dtype)
        #hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[3])
        return hidden_states.to(MODEL_DTYPE)
        
class NA_LayerNormANE(nn.Module):
    """ LayerNorm optimized for Apple Neural Engine (ANE) execution
    """
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
        super().__init__()
        # Principle 1: Picking the Right Data Format (machinelearning.apple.com/research/apple-neural-engine)
        self.num_channels = hidden_size
        self.eps = eps
        self.max_clip = 3e2  # ANE tuned value!
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.weight.data = self.weight.data.to(MODEL_DTYPE)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(MODEL_DTYPE)
        hidden_states = torch.clamp(hidden_states, min=-1000, max=1000).to(MODEL_DTYPE)  # ANE tuned value!
        mean = hidden_states.mean(-1, keepdim=True)
        hidden_states = hidden_states - mean
        variance = (hidden_states * hidden_states).mean(-1, keepdim=True).to(MODEL_DTYPE)       
        variance = torch.clamp(variance, min=self.eps, max=self.max_clip).to(MODEL_DTYPE)  # ANE tuned value!
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(MODEL_DTYPE) 
        hidden_states = (self.weight * hidden_states).to(MODEL_DTYPE)
        return hidden_states

class LlamaRotaryEmbedding(nn.Module):
    """Rotary positional embeddings for LLaMA model."""
    
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        # Apply rope_scaling factor if present
        self.base = config.rope_theta
        if hasattr(config, 'rope_scaling') and config.rope_scaling and 'factor' in config.rope_scaling:
            self.base = config.rope_theta * config.rope_scaling['factor']
        
        # Generate and cache the inverse frequency buffer
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(TEST_DEVICE) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache cos and sin values for positions
        t = torch.arange(self.max_position_embeddings, device=TEST_DEVICE).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Shape: [1, max_pos, head_dim] - consistent for both single token and batched
        self.cos_cached = emb.cos().view(1, self.max_position_embeddings, self.dim)
        self.sin_cached = emb.sin().view(1, self.max_position_embeddings, self.dim)
        
        if ENABLE_DEBUG2:
            print(f"\n[TRACE] LlamaRotaryEmbedding initialized:")
            print(f"  dim: {self.dim}")
            print(f"  max_position_embeddings: {self.max_position_embeddings}")
            print(f"  base: {self.base}")
            print(f"  inv_freq shape: {self.inv_freq.shape}")
            print(f"  cos_cached shape: {self.cos_cached.shape}")
            print(f"  sin_cached shape: {self.sin_cached.shape}")
            print(f"  cos_cached[0,0,:5]: {self.cos_cached[0,0,:5].tolist()}")
            print(f"  sin_cached[0,0,:5]: {self.sin_cached[0,0,:5].tolist()}")

    def forward(self, x, seq_len=None):
        # Simply return the pre-computed values, converting to the correct dtype
        return self.cos_cached.to(dtype=x.dtype), self.sin_cached.to(dtype=x.dtype)

    def rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings to input tensor x."""
        # Ensure tensor is contiguous and get dimensions
        x = x.contiguous()
        half_dim = x.shape[-1] // 2
        
        # Split x into two halves for rotation
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        
        # Ensure cos and sin have the right dimensions
        if cos.dim() == 4:  # Single token case [1, 1, 1, head_dim]
            cos = cos[..., :half_dim]
            sin = sin[..., :half_dim]
            if ENABLE_DEBUG2:
                print(f"      cos_sliced shape: {cos.shape}")
                print(f"      sin_sliced shape: {sin.shape}")
                print(f"      cos_sliced first 5 values: {cos.flatten()[:5].tolist()}")
                print(f"      sin_sliced first 5 values: {sin.flatten()[:5].tolist()}")
        else:  # Batched case [1, seq_len, head_dim]
            cos = cos.unsqueeze(1)[..., :half_dim]  # Add head dimension
            sin = sin.unsqueeze(1)[..., :half_dim]  # Add head dimension
            if ENABLE_DEBUG2:
                print(f"      cos_sliced shape: {cos.shape}")
                print(f"      sin_sliced shape: {sin.shape}")
                print(f"      cos_sliced first 5 values: {cos.flatten()[:5].tolist()}")
                print(f"      sin_sliced first 5 values: {sin.flatten()[:5].tolist()}")

        # Apply rotation using complex number multiplication
        rotated = torch.cat([
            x1 * cos - x2 * sin,  # Real part
            x2 * cos + x1 * sin   # Imaginary part
        ], dim=-1)
        
        if ENABLE_DEBUG2:
            print(f"      rotated shape: {rotated.shape}")
            print(f"      rotated first 5 values: {rotated.flatten()[:5].tolist()}")

        return rotated.to(MODEL_DTYPE)

class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        if MLP_UP_SPLIT > 1:
            self.gate_projs = nn.ModuleList([
                nn.Conv2d(
                    self.hidden_size,
                    self.intermediate_size // MLP_UP_SPLIT,
                    kernel_size=1,
                    bias=getattr(config, 'mlp_bias', False),
                    dtype=MODEL_DTYPE
                )
                for _ in range(MLP_UP_SPLIT)
            ])
            self.up_projs = nn.ModuleList([
                nn.Conv2d(
                    self.hidden_size,
                    self.intermediate_size // MLP_UP_SPLIT,
                    kernel_size=1,
                    bias=getattr(config, 'mlp_bias', False),
                    dtype=MODEL_DTYPE
                )
                for _ in range(MLP_UP_SPLIT)
            ])
        else:
            self.gate_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, kernel_size=1, bias=getattr(config, 'mlp_bias', False), dtype=MODEL_DTYPE)
            self.up_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, kernel_size=1, bias=getattr(config, 'mlp_bias', False), dtype=MODEL_DTYPE)

        if MLP_DOWN_SPLIT > 1:
            self.down_projs = nn.ModuleList([
                nn.Conv2d(
                    self.intermediate_size,
                    self.hidden_size // MLP_DOWN_SPLIT,
                    kernel_size=1,
                    bias=getattr(config, 'mlp_bias', False),
                    dtype=MODEL_DTYPE
                )
                for _ in range(MLP_DOWN_SPLIT)
            ])
        else:
            self.down_proj = nn.Conv2d(self.intermediate_size, self.hidden_size, kernel_size=1, bias=getattr(config, 'mlp_bias', False), dtype=MODEL_DTYPE)

        self.act_fn = ACT2FN.get(config.hidden_act, F.silu)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(2)  # Ensure x has shape [bsz, hidden_size, 1, seq_len]

        if MLP_UP_SPLIT > 1:
            gate_outputs = [gate_proj(x) for gate_proj in self.gate_projs]
            up_outputs = [up_proj(x) for up_proj in self.up_projs]
            
            a = torch.cat(gate_outputs, dim=1)
            b = torch.cat(up_outputs, dim=1)
        else:
            a = self.gate_proj(x)
            b = self.up_proj(x)

        c = self.act_fn(a)
        d = c * b

        if MLP_DOWN_SPLIT > 1:
            e_splits = [down_proj(d) for down_proj in self.down_projs]
            e = torch.cat(e_splits, dim=1)
        else:
            e = self.down_proj(d)

        return e.squeeze(2).permute(0, 2, 1)  # Final output shape: [bsz, seq_len, hidden_size]


    
    def forward(self, x):
        # Reshape input for Conv2D operations and ensure float16 dtype
        x = x.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        
        # Up projection with optional splitting
        if MLP_UP_SPLIT > 1:
            gate_outputs = [proj(x) for proj in self.gate_projs]
            up_outputs = [proj(x) for proj in self.up_projs]
            
            gate_states = torch.cat(gate_outputs, dim=1)
            up_states = torch.cat(up_outputs, dim=1)
        else:
            gate_states = self.gate_proj(x)
            up_states = self.up_proj(x)
        
        # Apply activation function
        gate_states = self.act_fn(gate_states)
        hidden_states = gate_states * up_states
        
        # Down projection with optional splitting
        if MLP_DOWN_SPLIT > 1:
            outputs = [proj(hidden_states) for proj in self.down_projs]
            hidden_states = torch.cat(outputs, dim=1)
        else:
            hidden_states = self.down_proj(hidden_states)
        
        # Reshape back to original format
        return hidden_states.squeeze(2).permute(0, 2, 1)

class LlamaAttention(nn.Module):
    """Attention mechanism optimized for Apple Neural Engine."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                           f" and `num_heads`: {self.num_heads}).")

        self.q_proj = nn.Conv2d(self.hidden_size, self.num_heads * self.head_dim, kernel_size=1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.k_proj = nn.Conv2d(self.hidden_size, self.num_key_value_heads * self.head_dim, kernel_size=1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.v_proj = nn.Conv2d(self.hidden_size, self.num_key_value_heads * self.head_dim, kernel_size=1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
        
        self.rotary_emb = LlamaRotaryEmbedding(config)
        self.scaling_factor = torch.tensor(1.0 / math.sqrt(self.head_dim), dtype=MODEL_DTYPE, device=TEST_DEVICE)


    def get_new_kv_cache(self, hidden_states, current_pos, rotary_emb):
        bsz, q_len, _ = hidden_states.shape
        device = hidden_states.device
        
        if ENABLE_DEBUG2:
            print(f"\nSINGLE TOKEN - Input shapes:")
            print(f"  hidden_states: {hidden_states.shape}, current_pos: {current_pos}, q_len: {q_len}")
        
        # Project QKV and ensure MODEL_DTYPE
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        if ENABLE_DEBUG2:
            print(f"  After permute+unsqueeze: {hidden_states.shape}")
        # Perform projections with fixed dimensions like in the original working code
        query_states = self.q_proj(hidden_states).view(1, self.num_heads, 1, self.head_dim).to(MODEL_DTYPE)
        key_states = self.k_proj(hidden_states).view(1, self.num_key_value_heads, 1, self.head_dim).to(MODEL_DTYPE)
        value_states = self.v_proj(hidden_states).view(1, self.num_key_value_heads, 1, self.head_dim).to(MODEL_DTYPE)
        
        if ENABLE_DEBUG2:
            print(f"  After projection:")
            print(f"    query_states: {query_states.shape}")
            print(f"    key_states: {key_states.shape}")
            print(f"    value_states: {value_states.shape}")
        
        # Use provided rotary embeddings
        cos, sin = rotary_emb
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if ENABLE_DEBUG2:
            print(f"  After applying rotary:")
            print(f"    query_states: {query_states.shape}")
            print(f"    key_states: {key_states.shape}")
            print(f"    value_states: {value_states.shape}")

        return query_states, key_states, value_states

    def get_new_kv_cache_prefill(self, hidden_states, current_pos, rotary_emb, batch_size):
        """Get new key-value cache entries optimized for prefilling with batched tokens."""
        _, batch, _ = hidden_states.shape # [1, BATCH, hidden_size=2048]
        device = hidden_states.device
        
        if ENABLE_DEBUG3:
            print(f"\nPREFILL - Input shapes:")
            print(f"  hidden_states: {hidden_states.shape}, current_pos: {current_pos}, batch: {batch}")
        
        # Project QKV and ensure MODEL_DTYPE - optimized for batch processing
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)  # [1, hidden_size, 1, batch]

        # Project all tokens at once using Conv2d
        query_states = self.q_proj(hidden_states)  # [1, num_heads * head_dim, 1, batch]
        key_states = self.k_proj(hidden_states)    # [1, num_kv_heads * head_dim, 1, batch]
        value_states = self.v_proj(hidden_states)  # [1, num_kv_heads * head_dim, 1, batch]

        # Reshape to final dimensions
        query_states = query_states.view(1, self.num_heads, self.head_dim, batch).permute(0, 1, 3, 2)  # [1, num_heads, batch, head_dim]
        key_states = key_states.view(1, self.num_key_value_heads, self.head_dim, batch).permute(0, 1, 3, 2)  # [1, num_kv_heads, batch, head_dim]
        value_states = value_states.view(1, self.num_key_value_heads, self.head_dim, batch).permute(0, 1, 3, 2)  # [1, num_kv_heads, batch, head_dim]

        # Get rotary embeddings for all positions at once
        cos, sin = rotary_emb
        cos = cos.permute(0, 2, 1, 3)  # [1, 1, batch, head_dim]
        sin = sin.permute(0, 2, 1, 3)  # [1, 1, batch, head_dim]

        # Apply rotary embeddings to all positions at once
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        return query_states.to(MODEL_DTYPE), key_states.to(MODEL_DTYPE), value_states.to(MODEL_DTYPE)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        current_pos: int,
        IN_PREFILL: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with support for both single token and prefill modes."""
        batch_size, seq_length, _ = hidden_states.shape
        
        print(f"\n[DEBUG] LlamaAttention Forward:")
        print(f"  Input shapes:")
        print(f"    hidden_states: {hidden_states.shape}")
        print(f"    attention_mask: {attention_mask.shape if attention_mask is not None else None}")
        print(f"    position_ids: {position_ids.shape if position_ids is not None else None}")
        print(f"    current_pos: {current_pos}")
        print(f"    IN_PREFILL: {IN_PREFILL}")
        
        assert current_pos is not None, "current_pos must be provided"
        assert position_ids is not None, "position_ids must be provided"
        assert attention_mask is not None, "attention_mask must be provided"

        # Generate position IDs if not provided
        # Get rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_length, current_pos=current_pos)
        
        # Get KV states based on mode
        if IN_PREFILL:
            query_states, key_states, value_states = self.get_new_kv_cache_prefill(
                hidden_states, current_pos, (cos, sin), batch_size=seq_length
            )
        else:
            query_states, key_states, value_states = self.get_new_kv_cache(
                hidden_states, current_pos, (cos, sin)
            )
        
        # Update KV cache if provided
        if kwargs.get('kv_cache_layer', None) is not None:
            key_cache, value_cache = kwargs['kv_cache_layer']
            print(f"  KV Cache shapes:")
            print(f"    key_cache: {key_cache.shape}")
            print(f"    value_cache: {value_cache.shape}")
            print(f"    key_states: {key_states.shape}")
            print(f"    value_states: {value_states.shape}")
            print(f"    current_pos: {current_pos}")
            print(f"    seq_length: {seq_length}")
            
            if current_pos is not None:
                # Update at current position
                print(f"  Updating cache at position {current_pos}")
                print(f"    Update slice - key_cache[:, {current_pos}:{current_pos + seq_length}]")
                print(f"    key_states shape: {key_states.shape}")
                print(f"    value_states shape: {value_states.shape}")
                
                # Reshape key_states and value_states to match cache dimensions
                key_states_reshaped = key_states.squeeze(0)  # Remove batch dimension
                value_states_reshaped = value_states.squeeze(0)  # Remove batch dimension
                
                print(f"    After reshape:")
                print(f"      key_states_reshaped: {key_states_reshaped.shape}")
                print(f"      value_states_reshaped: {value_states_reshaped.shape}")
                
                # Update cache
                key_cache[:, current_pos:current_pos + seq_length] = key_states_reshaped
                value_cache[:, current_pos:current_pos + seq_length] = value_states_reshaped
                
                # Use full cache for attention
                key_states = key_cache[:, :current_pos + seq_length].unsqueeze(0)  # Add batch dimension back
                value_states = value_cache[:, :current_pos + seq_length].unsqueeze(0)  # Add batch dimension back
                
                print(f"  After cache update:")
                print(f"    key_states: {key_states.shape}")
                print(f"    value_states: {value_states.shape}")
        
        # Compute scaled dot-product attention
        attention_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling_factor
        
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask
        
        attention_weights = self.ANE_softmax(attention_weights, dim=-1)
        attention_output = torch.matmul(attention_weights, value_states)
        
        print(f"  Attention computation shapes:")
        print(f"    attention_weights: {attention_weights.shape}")
        print(f"    attention_output: {attention_output.shape}")
        
        # Reshape for output projection
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_length, self.hidden_size)
        attention_output = attention_output.permute(0, 2, 1).unsqueeze(2)
        
        # Project output using Conv2D
        output = self.o_proj(attention_output.squeeze(2).permute(0, 2, 1))
        print(f"  Final output shape: {output.shape}\n")
        
        return output

    def ANE_softmax(self, x, dim=-1):
        #return F.softmax(x, dim=dim)
        
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        x = x - x_max
        exp_x = torch.exp(x)
        softmax_output = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
        return softmax_output

        
    def apply_rotary_pos_emb(self, q_states, k_states, cos, sin):
        """
        Applies rotary position embeddings to both q_states and k_states.
        """
        if ENABLE_VALUES:
            print(f"\n[DEBUG] apply_rotary_pos_emb input:")
            print(f"  q_states shape: {q_states.shape}")
            print(f"  k_states shape: {k_states.shape}")
            print(f"  cos shape: {cos.shape}")
            print(f"  sin shape: {sin.shape}")
            print(f"  q_states first 5: {q_states[0,0,0,:5].tolist()}")
            print(f"  k_states first 5: {k_states[0,0,0,:5].tolist()}")
        
        def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            x = x.contiguous()
            half_dim = x.shape[-1] // 2
            
            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]
            
            if ENABLE_VALUES:
                print(f"\n[DEBUG] rotate:")
                print(f"  x1 first 5: {x1[0,0,0,:5].tolist()}")
                print(f"  x2 first 5: {x2[0,0,0,:5].tolist()}")
            
            if cos.dim() == 4:
                cos = cos[..., :half_dim]
                sin = sin[..., :half_dim]
            else:
                cos = cos.unsqueeze(1)[..., :half_dim]
                sin = sin.unsqueeze(1)[..., :half_dim]
            
            
            rotated = torch.cat([
                x1 * cos - x2 * sin,
                x2 * cos + x1 * sin
            ], dim=-1)
            
            
            return rotated.to(MODEL_DTYPE)
        
        q_rotated = rotate(q_states, cos, sin)
        k_rotated = rotate(k_states, cos, sin)
                
        return q_rotated, k_rotated

    def forward_regular(self, hidden_states, query_states, kv_cache_layer=None, causal_mask=None):
        """Forward pass for single token generation."""
        bsz, q_len, _ = hidden_states.shape
                
        # Get KV cache
        K_layer_cache, V_layer_cache = kv_cache_layer
        
        # Slice only up to CONTEXT_LENGTH from the cache
        K_layer_cache = K_layer_cache[..., :self.config.context_length, :]
        V_layer_cache = V_layer_cache[..., :self.config.context_length, :]
        
        # Repeat KV for multi-head attention
        key_states = self.repeat_kv(K_layer_cache, self.num_key_value_groups)
        value_states = self.repeat_kv(V_layer_cache, self.num_key_value_groups)

        # Compute attention using optimized path for batch_size=1
        
        # Compute attention scores directly without einsum
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling_factor
        
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask[:, :, :self.config.context_length]
        
        if ENABLE_VALUES:
            print(f"[TRACE]  SINGLE attn_weights first 10={attn_weights[0, -1, 0:10, 0:10].tolist()}")
            print(f"[TRACE]  q_len={q_len}")

        # Optimized softmax for batch_size=1
        attn_weights = self.ANE_softmax(attn_weights, dim=-1)
        
        # Compute attention output directly without einsum
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape before projecting: [1, heads, q_len, head_dim] -> [1, q_len, heads*head_dim]
        if ENABLE_DEBUG2:
            print(f"[TRACE]  B4 reshape attn_output shape={attn_output.shape}")
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        # Project output
        attn_output = self.o_proj(attn_output)
        
        return attn_output

    def forward_prefill(self, hidden_states, query_states, kv_cache_layer=None, causal_mask=None):
        """Forward pass for prefill mode"""
        bsz, q_len, _ = hidden_states.shape
                
        # Get KV cache
        K_layer_cache, V_layer_cache = kv_cache_layer
        
        # Slice only up to CONTEXT_LENGTH from the cache
        K_layer_cache = K_layer_cache[..., :self.config.context_length, :]
        V_layer_cache = V_layer_cache[..., :self.config.context_length, :]
        
        if ENABLE_DEBUG3:  
            print("[forward_prefill.0] K_layer_cache.shape=", K_layer_cache.shape)
        
        # Repeat KV for multi-head attention
        key_states = self.repeat_kv(K_layer_cache, self.num_key_value_groups)
        value_states = self.repeat_kv(V_layer_cache, self.num_key_value_groups)
        
        if ENABLE_DEBUG3:  
            print("[forward_prefill.1] hidden_states.shape=", hidden_states.shape)
            print("[forward_prefill.1] query_states.shape=", query_states.shape)
            print("[forward_prefill.1] key_states.shape=", key_states.shape)
            print("[forward_prefill.1] value_states.shape=", value_states.shape)
        
        # Compute scaled dot-product attention
        attn_weights = torch.einsum('bhqd,bhkd->bhqk', query_states, key_states) * self.scaling_factor
        
        if causal_mask is not None:
            if ENABLE_DEBUG3:  
                print("[forward_prefill.2] causal_mask.shape=", causal_mask.shape)
                print("[forward_prefill.2] attn_weights.shape=", attn_weights.shape)
            attn_weights = attn_weights + causal_mask[:, :, :self.config.context_length]
        
        if ENABLE_VALUES:
            print(f"[forward_prefill]  BATCH attn_weights first 10={attn_weights[0, -1, 0:10, 0:10].tolist()}")
        
        attn_weights = self.ANE_softmax(attn_weights, dim=-1)
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, value_states)  # [batch=1, heads=32, q_len=1, head_dim=64]
        
        # Reshape before projecting: [batch, heads, q_len, head_dim] -> [batch, q_len, heads*head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        # Project output
        attn_output = self.o_proj(attn_output)
        
        return attn_output

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

class LlamaDecoderLayer(nn.Module):
    """Transformer decoder layer for LLaMA."""

    def __init__(self, config, layer_idx: int, use_ane_norm=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        
        # Use ANE_NORM if enabled, otherwise use RMSNorm
        if use_ane_norm:
            self.input_layernorm = LayerNormANE(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = LayerNormANE(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache_layer=None, current_pos=None, IN_PREFILL=False):
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Pass the layer's KV cache to attention, Class LlamaAttention
        if IN_PREFILL:
            hidden_states = self.self_attn.forward_prefill(
                hidden_states=hidden_states,
                query_states=query_states,
                causal_mask=causal_mask,
                kv_cache_layer=kv_cache_layer
            )
        else:
            hidden_states = self.self_attn.forward_regular(
                hidden_states=hidden_states,
                query_states=query_states,
                causal_mask=causal_mask,
                kv_cache_layer=kv_cache_layer
            )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

def get_kv_cache_idx(layer_idx, num_layers, num_groups=1):
    """Helper function to get KV cache indices."""
    layers_per_group = num_layers // num_groups
    group_idx = layer_idx // layers_per_group
    layer_in_group_idx = layer_idx % layers_per_group
    return group_idx, layer_in_group_idx, layers_per_group

class LlamaModel(BaseModel):
    """LLaMA model implementation."""

    def __init__(self, config, use_ane_norm=False, model_path=None):
        super().__init__(config)
        self.use_ane_norm = use_ane_norm
        self.model_path = model_path
        

        # Initialize layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx=i, use_ane_norm=use_ane_norm) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Initialize normalization
        if use_ane_norm:
            self.norm = LayerNormANE(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            
        # Initialize KV cache with MODEL_DTYPE
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        if FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
            cache_size = (
                2 * config.num_hidden_layers,
                config.num_key_value_heads,
                self.config.state_length,
                self.head_dim
            )
            self.register_buffer("kv_cache_0", torch.zeros(cache_size, dtype=MODEL_DTYPE, device=TEST_DEVICE))
            if ENABLE_DEBUG:
                print(f"Initialized unified KV kv_cache_0 with shape: {self.kv_cache_0.shape}")
        else:
            layers_per_group = config.num_hidden_layers
            for i in range(config.num_hidden_layers):
                cache_size = (
                    2 * layers_per_group,
                    config.num_key_value_heads,
                    self.config.state_length,
                    self.head_dim
                )
                self.register_buffer(
                    f"kv_cache_{i}", 
                    torch.zeros(cache_size, dtype=MODEL_DTYPE, device=TEST_DEVICE)
                )


    def get_rotary_embeddings_s(self, current_pos):
        """Get rotary embeddings for the current position"""
        if ENABLE_VALUES:
            print(f"\n[DEBUG] get_rotary_embeddings_s:")
            print(f"  current_pos: {current_pos}")
        
        sin = self.layers[0].self_attn.rotary_emb.sin_cached[:, current_pos].view(1, 1, 1, -1)
        cos = self.layers[0].self_attn.rotary_emb.cos_cached[:, current_pos].view(1, 1, 1, -1)
        if ENABLE_VALUES:
            print(f"  cos shape: {cos.shape}")
            print(f"  sin shape: {sin.shape}")
            print(f"  cos first 5: {cos[0,0,0,:5].tolist()}")
            print(f"  sin first 5: {sin[0,0,0,:5].tolist()}")
        
        return cos, sin

    def get_rotary_embedding_prefill(self, positions):
        """Get rotary embeddings for a sequence of positions.
        Args:
            positions: Tensor of position indices
        Returns:
            Tuple of (cos, sin) tensors with shape [1, seq_len, 1, head_dim]
        """
        # Get rotary embeddings from the first attention layer
        rotary_emb = self.layers[0].self_attn.rotary_emb
        
        # Get embeddings for the range of positions directly
        seq_len = positions.size(0)
        cos = rotary_emb.cos_cached[:, positions].view(1, seq_len, 1, rotary_emb.dim)
        sin = rotary_emb.sin_cached[:, positions].view(1, seq_len, 1, rotary_emb.dim)
        
        if ENABLE_DEBUG3:
            print(f"cos shape: {cos.shape}")
            print(f"sin shape: {sin.shape}")

            print(f"[TRACE] get_rotary_embedding_prefill Batched rotary from pos {positions[0]}:")
            print(f"  cos shape: {cos.shape}, values[0,:5]: {cos[0,0,0,:5].tolist()}")
            print(f"  sin shape: {sin.shape}, values[0,:5]: {sin[0,0,0,:5].tolist()}")

        return cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE)

    def process_layer_prefill(self, layer_idx, hidden_states,  position_ids, causal_mask, current_pos, rotary_emb, layer_offset):
        """Process a single transformer layer in prefill mode"""
        layer = self.layers[layer_idx]
        batch_size = position_ids.shape[0]

        if ENABLE_VALUES:
            print("BATCH------------------------------------------------------------------------------------------------- ")
            print(f"Layer {layer_idx} Input hidden states (first 16 values):")
            print(f"  {hidden_states[0, 0, :16].tolist()}")
        normalized_states = layer.input_layernorm(hidden_states)

        if ENABLE_VALUES:
            print("BATCH------------------------------------------------------------------------------------------------- ")
            print(f"Layer {layer_idx} after input_layernorm hidden states (first 16 values):")
            print(f"  {hidden_states[0, 0, :16].tolist()}")
        
        if ENABLE_DEBUG3:
            print(f"position_ids={position_ids}")
            print(f"position_ids.shape={position_ids.shape}")
            print(f"rotary_emb.shape={rotary_emb[0].shape}")
            print("[process_layer_prefill] causal_mask.shape=", causal_mask.shape)

        # Get query, key and value states for prefill
        query_states, key_states, value_states = layer.self_attn.get_new_kv_cache_prefill(
            normalized_states,
            current_pos,
            rotary_emb,
            batch_size
        )

        # Get group indices
        group_idx, layer_in_group_idx, layers_per_group = get_kv_cache_idx(layer_idx, self.config.num_hidden_layers)

        # Get the combined KV cache for this group
        if FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
            kv_cache = getattr(self, "kv_cache_0")
        else:
            kv_cache = getattr(self, f"kv_cache_{group_idx}")

        key_idx = layer_in_group_idx
        value_idx = layer_in_group_idx + layers_per_group

        # Store the full sequence length in prefill mode
        seq_length = key_states.shape[2]  # Get actual sequence length
        kv_cache[key_idx:key_idx + 1, :, current_pos:current_pos + seq_length, :] = key_states
        kv_cache[value_idx:value_idx + 1, :, current_pos:current_pos + seq_length, :] = value_states
        
        # Get the key and value states for this layer from the merged cache
        key_cache = kv_cache[key_idx:key_idx + 1].squeeze(0)
        value_cache = kv_cache[value_idx:value_idx + 1].squeeze(0)

        # Run attention with the updated KV cache
        attn_output = layer.self_attn.forward_prefill(
            hidden_states=normalized_states,
            query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask,
        )

        hidden_states = hidden_states + attn_output

        # Skip MLP for last layer in prefill mode since output isn't used
        is_last_layer = (layer_idx == len(self.layers) - 1)
        if not is_last_layer:
            # Add post-attention normalization and MLP
            post_attn = layer.post_attention_layernorm(hidden_states)
            hidden_states = hidden_states + layer.mlp(post_attn)
        else:
            print("Skipping MLP for last layer in prefill mode")

        if ENABLE_VALUES:
            print("BATCH------------------------------------------------------------------------------------------------- ")
            print(f"Layer {layer_idx} Output hidden states (first 16 values):")
            print(f"  {hidden_states[0, 0, :16].tolist()}")

        return hidden_states

    def process_layer_regular(self, layer_idx, hidden_states,  position_ids, causal_mask, current_pos, rotary_emb, layer_offset):
        """Process a single transformer layer in regular (non-prefill) mode"""
        layer = self.layers[layer_idx]
        batch_size = position_ids.shape[0]

        if ENABLE_VALUES:
            print("SINGLE------------------------------------------------------------------------------------------------- ")
            print(f"Layer {layer_idx} Input hidden states (first 16 values):")
            print(f"  {hidden_states[0, 0, :16].tolist()}")

        if ENABLE_DEBUG2:
            print(f"normalized_states.shape={hidden_states.shape}")

        normalized_states = layer.input_layernorm(hidden_states)
        if ENABLE_DEBUG2:
            print(f"normalized_states.shape={normalized_states.shape}")

        if ENABLE_VALUES:
            print("SINGLE------------------------------------------------------------------------------------------------- ")
            print(f"Layer {layer_idx} after input_layernorm hidden states (first 16 values):")
            print(f"  {hidden_states[0, 0, :16].tolist()}")
        
        if ENABLE_DEBUG2:
            print(f"position_ids={position_ids}")
            print(f"position_ids.shape={position_ids.shape}")
            print(f"rotary_emb.shape={rotary_emb[0].shape}")

        # Get query, key and value states for regular processing
        query_states, key_states, value_states = layer.self_attn.get_new_kv_cache(
            normalized_states,
            current_pos,
            rotary_emb
        )

        # Get group indices
        group_idx, layer_in_group_idx, layers_per_group = get_kv_cache_idx(layer_idx, self.config.num_hidden_layers)

        # Get the combined KV cache for this group
        if FORCE_UNIFIED_CACHE or ENABLE_UNIFIED_CACHE:
            kv_cache = getattr(self, "kv_cache_0")
        else:
            kv_cache = getattr(self, f"kv_cache_{group_idx}")

        key_idx = layer_in_group_idx
        value_idx = layer_in_group_idx + layers_per_group

        # For non-prefill, store single position
        kv_cache[key_idx:key_idx + 1, :, current_pos:current_pos + 1, :] = key_states
        kv_cache[value_idx:value_idx + 1, :, current_pos:current_pos + 1, :] = value_states
        
        # Get the key and value states for this layer from the merged cache
        key_cache = kv_cache[key_idx:key_idx + 1].squeeze(0)
        value_cache = kv_cache[value_idx:value_idx + 1].squeeze(0)

        # Run attention with the updated KV cache
        attn_output = layer.self_attn.forward_regular(
            hidden_states=normalized_states,
            query_states=query_states,
            kv_cache_layer=(key_cache, value_cache),
            causal_mask=causal_mask,
        )

        hidden_states = hidden_states + attn_output

        # Add post-attention normalization and MLP
        post_attn = layer.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + layer.mlp(post_attn)

        if ENABLE_VALUES:
            print("SINGLE------------------------------------------------------------------------------------------------- ")
            print(f"Layer {layer_idx} Output hidden states (first 16 values):")
            print(f"  {hidden_states[0, 0, :16].tolist()}")
        return hidden_states

    def process_layer(self, layer_idx, hidden_states,  position_ids, causal_mask, current_pos, rotary_emb, layer_offset, IN_PREFILL=False):
        """Process a single transformer layer, delegating to the appropriate mode-specific implementation"""
        if IN_PREFILL:
           return self.process_layer_prefill(layer_idx, hidden_states,  position_ids, causal_mask, current_pos, rotary_emb, layer_offset)
        else:
            return self.process_layer_regular(layer_idx, hidden_states,  position_ids, causal_mask, current_pos, rotary_emb, layer_offset)

    def process_layers(self, hidden_states,  position_ids, causal_mask, current_pos, rotary_emb, start_layer=0, end_layer=None, IN_PREFILL=False):
        """Process a range of transformer layers"""
        if end_layer is None:
            end_layer = len(self.layers)

        layer_offset = 0
        if not ENABLE_UNIFIED_CACHE:
            layer_offset = start_layer

        for i in range(start_layer, end_layer):
            hidden_states = self.process_layer(
                i, hidden_states,  position_ids,
                causal_mask, current_pos, rotary_emb, layer_offset, IN_PREFILL
            )
            #if TEST_ANE > 0 and i >= TEST_ANE:
            #    break
        return hidden_states



    def forward(self, hidden_states, position_ids=None, causal_mask=None, current_pos=None,
                start_layer=0, end_layer=None, IN_PREFILL=False):
        """
        Forward pass with support for partial layer execution.
        
        Args:
            hidden_states: Input tensor
            update_mask: Mask for KV cache updates
            position_ids: Position IDs
            causal_mask: Causal attention mask
            current_pos: Current position in the sequence
            start_layer: Start processing from this layer (inclusive)
            end_layer: End processing at this layer (exclusive)
        """
        if ENABLE_DEBUG2:
            print(f"LlamaModel.forward - hidden_states shape: {hidden_states.shape}")

        # Get rotary embeddings
        if IN_PREFILL:
            rotary_emb = self.get_rotary_embedding_prefill(position_ids)
        else:
            rotary_emb = self.get_rotary_embeddings_s(current_pos)

        # Process layers
        hidden_states = self.process_layers(
            hidden_states, position_ids, causal_mask,
            current_pos, rotary_emb, start_layer, end_layer, IN_PREFILL=IN_PREFILL,
        )
        if ENABLE_DEBUG2:
            print(f"LlamaModel.forward - hidden_states last 10: {hidden_states[-1, -1, -10:].tolist()}")

        # Apply final normalization if this is the last block
        if end_layer is None or end_layer == len(self.layers):
            if IN_PREFILL:
                print("Skipping final normalization for prefill, data not used!")
                # return first batch only to mimizie memory usage and avoid optimization!
                return hidden_states[:,0:1,:]
            else:
                if ENABLE_VALUES:
                    print(f"LlamaModel.forward b4 self.norm hidden_states last 10: {hidden_states[-1, -1, -10:].tolist()}")
                hidden_states = self.norm(hidden_states)
                if ENABLE_VALUES:
                    print(f"LlamaModel.forward AFTER self.norm hiddhistoryen_states last 10: {hidden_states[-1, -1, -10:].tolist()}")

        return hidden_states

    #LlamaModel.forward_prefill
    def forward_prefill(self, hidden_states, position_ids=None, causal_mask=None, current_pos=None, start_layer=None, end_layer=None):
        """
        Forward pass for prefilling KV cache
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Get rotary embeddings
        rotary_emb = self.get_rotary_embedding_prefill(position_ids)
        
        # Process layers within the specified range if provided
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
            # Process all layers for non-split mode
            hidden_states = self.process_layers(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                rotary_emb=rotary_emb,
                IN_PREFILL=True
            )
        
        # Apply final normalization if this is the last block

        if end_layer is None or end_layer == len(self.layers):
            if ENABLE_VALUES:
                print(f"LlamaModel.forward b4 self.norm hidden_states last 10: {hidden_states[-1, -1, -10:].tolist()}")
            
            hidden_states = self.norm(hidden_states)
            
            if ENABLE_VALUES:
                print(f"LlamaModel.forward AFTER self.norm hidden_states last 10: {hidden_states[-1, -1, -10:].tolist()}")

        return hidden_states



def stable_l2_norm(x, eps):
    """Compute stable L2 norm optimized for ANE.
    
    Args:
        x: Input tensor
        eps: Small value to prevent division by zero
    
    Returns:
        Normalized tensor and scale factor
    """
    # Find maximum absolute value for scaling
    max_val = x.abs().max(axis=-1, keepdim=True).values
    max_val = torch.clamp(max_val, min=eps)
    
    # Scale input to prevent overflow
    xscaled = x / max_val
    
    # Compute L2 norm on scaled values
    scaled_norm = torch.linalg.norm(xscaled, dim=-1, keepdim=True)
    scaled_norm = torch.clamp(scaled_norm, min=eps)
    
    return x / scaled_norm, max_val

class LlamaForCausalLM(nn.Module):
    """LLaMA model with causal language modeling head."""
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, use_ane_norm=False, enable_coreml=False):
        super().__init__()
        self.config = config
        self.enable_coreml = enable_coreml
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size).to(TEST_DEVICE)
        self.model = LlamaModel(config, use_ane_norm=use_ane_norm).to(TEST_DEVICE)
        
        # Initialize lm_head as Conv2d for ANE optimization
        if ENABLE_CONV2D:
            if ENABLE_VACAB_SPLIT8:
                self.lm_head8_1 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head8_2 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head8_3 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head8_4 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head8_5 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head8_6 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head8_7 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
                self.lm_head8_8 = nn.Conv2d(config.hidden_size, config.vocab_size//8, 1, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
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
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=MODEL_DTYPE).to(TEST_DEVICE)
            print("Created linear lm_head")

    def load_pretrained_weights(self, model_path, **kwargs):
        """Load pretrained weights for both the base model and embeddings."""
        print("Loading pretrained weights...")
        
        file_dict = {}
        for file in tqdm(os.listdir(model_path)):
            if file.endswith(".safetensors"):
                file_dict.update(safetensors.torch.load_file(os.path.join(model_path, file)))

        # Handle lm_head weight
        lm_head_present = False
        embed_tokens_key = None
        for k, v in file_dict.items():
            if k == "lm_head.weight":
                lm_head_present = True
            if "embed_tokens.weight" in k:
                embed_tokens_key = k

        if not lm_head_present:
            print("lm_head.weight not found in the model file dictionary")
            if embed_tokens_key:
                print(f"Using {embed_tokens_key} for lm_head.weight")
                file_dict['lm_head.weight'] = file_dict[embed_tokens_key].clone()
            else:
                print("embed_tokens.weight not found. Unable to set lm_head.weight")
                return False

        # Filter and reshape weights
        filtered_state_dict = {}
        for k, v in file_dict.items():
            if k == "model.embed_tokens.weight":
                print(f"Loading {k} with shape {v.shape}")
                filtered_state_dict["embed_tokens.weight"] = v  # Keep original dtype
                print(f"Moving model.embed_tokens.weight to embed_tokens.weight")
            elif k == "lm_head.weight":
                if ENABLE_CONV2D:
                    reshaped_weight = v.view(v.shape[0], v.shape[1], 1, 1)
                    if ENABLE_VACAB_SPLIT8:
                        vocab_split = self.config.vocab_size // 8
                        splits = torch.split(reshaped_weight, vocab_split)
                        for i, split in enumerate(splits):
                            filtered_state_dict[f"lm_head8_{i+1}.weight"] = split
                            print(f"Split lm_head weight into lm_head8_{i+1}.weight with shape {split.shape}")
                    elif ENABLE_VACAB_SPLIT:
                        vocab_split = self.config.vocab_size // 2
                        split1, split2 = torch.split(reshaped_weight, [vocab_split, self.config.vocab_size - vocab_split])
                        filtered_state_dict["lm_head2_1.weight"] = split1
                        filtered_state_dict["lm_head2_2.weight"] = split2
                    else:
                        filtered_state_dict["lm_head1.weight"] = reshaped_weight
                else:
                    filtered_state_dict["lm_head.weight"] = v

        # Load filtered weights
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        # Load weights for the base model
        base_filtered_dict = {}
        for k, v in file_dict.items():
            if k.startswith("model."):
                new_key = k.replace("model.", "")
                if "layers." in new_key:
                    # Handle attention weights
                    if 'self_attn' in new_key and 'weight' in new_key:
                        if 'o_proj' in new_key:
                            base_filtered_dict[new_key] = v
                            print(f"Keeping o_proj weights as 2D: {new_key} shape {v.shape}")
                        else:
                            reshaped_weight = v.view(v.shape[0], v.shape[1], 1, 1)
                            base_filtered_dict[new_key] = reshaped_weight
                            print(f"Reshaped {new_key} from {v.shape} to {reshaped_weight.shape}")
                    # Handle MLP weights
                    elif 'mlp' in new_key and 'weight' in new_key:
                        reshaped_weight = v.view(v.shape[0], v.shape[1], 1, 1)
                        base_filtered_dict[new_key] = reshaped_weight
                        print(f"Reshaped {new_key} from {v.shape} to {reshaped_weight.shape}")
                    else:
                        base_filtered_dict[new_key] = v
                elif new_key == "norm.weight":
                    base_filtered_dict[new_key] = v

        # Load base model weights
        missing_keys, unexpected_keys = self.model.load_state_dict(base_filtered_dict, strict=False)
        
        # Filter out expected missing keys
        expected_missing = ['kv_cache_0']  # KV cache buffer is initialized separately
        expected_missing.extend([f'layers.{i}.self_attn.rotary_emb.inv_freq' for i in range(self.config.num_hidden_layers)])
        
        actual_missing = [k for k in missing_keys if k not in expected_missing]
        if not actual_missing and not unexpected_keys:
            print("Pretrained weights loaded successfully")
            if missing_keys:
                print("Note: The following expected buffers were initialized:")
                for k in missing_keys:
                    print(f"  - {k}")
            return True
        else:
            print("Pretrained weights loaded with some issues:")
            if actual_missing:
                print(f"Missing keys: {actual_missing}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            return False

    def prefill_kv_cache(self, input_ids, position_ids, start_pos, causal_mask):
        """
        Pre-fills KV cache for a batch of tokens starting from start_pos.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_length]
            start_pos: Starting position in the KV cache
            causal_mask: Causal attention mask
            
        Returns:
            None (updates KV cache in-place)
        """
        #print(f"[DEBUG] Prefill phase started. start_pos={start_pos}")

        batch_size, seq_length = input_ids.shape
        
        # Create position IDs for this batch - each token should get its correct position
        
        if ENABLE_DEBUG3:       
            print(f"[DEBUG] Position IDs: {position_ids.tolist()}")
        
        
        # Get embeddings and run through model
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.to(MODEL_DTYPE)

        # Get correct causal mask for the sequence
        # For prefill, each token should attend to all previous tokens in the sequence
        if causal_mask is not None:
            # Take the full sequence slice of causal mask
            causal_mask_prefill = causal_mask[:, :, :seq_length, :]
            #print(f"[DEBUG] Using causal mask shape: {causal_mask_prefill.shape}")
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

    def forward(
        self,
        input_ids: torch.LongTensor,
        update_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        current_pos: int,
        causal_mask: torch.Tensor,
        IN_PREFILL: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for causal language modeling."""
        if ENABLE_DEBUG:
            print(f"LlamaForCausalLM::forward called with input_ids: {input_ids.shape}, update_mask: {update_mask.shape}, position_ids: {position_ids.shape}, causal_mask: {causal_mask.shape}, current_pos: {current_pos}")
        # Get embeddings

        # Assert input_ids is a 2D tensor
        assert len(input_ids.shape) == 2, f"input_ids must be a 2D tensor, got shape {input_ids.shape}"
        if not IN_PREFILL:
            assert len(position_ids.shape) == 1, f"position_ids must be a 1D tensor for Inference, got shape {position_ids.shape}"
        else:
            assert position_ids.shape[-1] == input_ids.shape[-1], f"position_ids last dimension should match input_ids for Prefill, got shape {position_ids.shape} and input_ids shape {input_ids.shape}"

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.to(MODEL_DTYPE)

        if ENABLE_VALUES:
            print(f"LlamaForCausalLM::embed_tokens weight dtype: {self.embed_tokens.weight.dtype}")
            print(f"LlamaForCausalLM::embed_tokens input_idss (values):{input_ids.tolist()}")
            print(f"LlamaForCausalLM::embed_tokens model input hidden states (first 16 values):{hidden_states[0,0,:16].tolist()}")

        # Process through transformer layers
        hidden_states = self.model(
            hidden_states=hidden_states,
            position_ids=position_ids,
            current_pos=current_pos,
            causal_mask=causal_mask,  # Pass causal_mask through
            start_layer=0,
            end_layer=None,
            IN_PREFILL=IN_PREFILL,  # Added trailing comma
        )
        
        # Project to vocabulary using appropriate head
        if ENABLE_CONV2D:
            # Reshape for Conv2d and ensure float16
            hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
            
            if ENABLE_VACAB_SPLIT8:
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
            # Use linear head
            logits = self.lm_head(hidden_states)
        
        return logits

class LlamaConverter(BaseConverter):
    """Handles LLAMA model conversion to CoreML."""

    def __init__(self, config, model_path=None, use_ane_norm=False):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.use_ane_norm = use_ane_norm
        # Initialize model with enable_coreml=True for CoreML conversion
        self.model = LlamaForCausalLM(config, use_ane_norm=use_ane_norm, enable_coreml=True)
        
        if False and model_path:
            self.model.model.load_pretrained_weights(
                model_path,
                enable_conv2d=True,
                mlp_up_split=1,
                mlp_down_split=1
            )

    def convert(self):
        self.preprocess()
        # LLAMA model needs special handling before CoreML conversion
        coreml_model = self.convert_to_coreml(self.model)
        self.postprocess()
        return coreml_model

    def convert_to_coreml(self, model):
        """Convert LLAMA model using CoreMLTools."""
        return ct.convert(model)  # Placeholder for actual logic
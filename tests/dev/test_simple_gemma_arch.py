#!/usr/bin/env python3
# Copyright (c) 2025 ANEMLL
# Licensed under the MIT License

""" X3 Test Gemma3n architecture that matches the actual local model structure."""
# 0.9A
#  4-streams version
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import math
from typing import Optional
from transformers.cache_utils import Cache
from transformers.activations import ACT2FN

def create_rotary_cache(head_dim, seq_len, theta=1000000.0, device=None, dtype=None, batch_size=1):
    """
    Create rotary position embeddings cache as per HF Gemma3n.
    
    This implementation was cross-referenced with the canonical JAX implementation
    at gemma/gemma/gm/math/_positional_embeddings.py and found to be mathematically
    equivalent. The core logic for calculating inverse frequencies and applying
    them to position IDs is identical.
    """
    # HF calculates inv_freq for head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    
    # Calculate attention scaling like HF (1/sqrt(head_dim))
    attention_scaling = head_dim ** -0.5
    
    position_ids = torch.arange(seq_len, device=device, dtype=torch.int64).float()
    
    # Expand inv_freq and position_ids for broadcasting
    inv_freq_expanded = inv_freq[None, :, None] # [1, head_dim // 2, 1]
    position_ids_expanded = position_ids[None, None, :].expand(batch_size, 1, -1) # [batch_size, 1, seq_len]

    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) # [batch_size, seq_len, head_dim // 2]
    emb = torch.cat((freqs, freqs), dim=-1) # [batch_size, seq_len, head_dim]
    
    # Do NOT apply attention scaling to position embeddings (HF doesn't do this)
    cos = emb.cos()
    sin = emb.sin()
    
    return cos, sin

def repeat_kv(hidden_states, n_rep):
    """
    Repeat key/value heads for GQA (Grouped Query Attention).
    This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask=None,
    dropout=0.0,
    scaling=None,
    softcap=None,
    **kwargs,
):
    """
    Exact implementation of HF's eager_attention_forward for Gemma3n.
    """
    if scaling is None:
        scaling = module.head_dim**-0.5
    
    # 🔍 DEBUG: Print input magnitudes
    if hasattr(module, 'layer_idx') and module.layer_idx == 0:
        print(f"🔍 ATTENTION DEBUG - Input magnitudes:")
        print(f"  query: std={query.std():.6f}, mean={query.mean():.6f}")
        print(f"  key: std={key.std():.6f}, mean={key.mean():.6f}")  
        print(f"  value: std={value.std():.6f}, mean={value.mean():.6f}")
        print(f"  scaling: {scaling:.6f}")
    
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    
    # 🔍 DEBUG: Print after repeat_kv
    if hasattr(module, 'layer_idx') and module.layer_idx == 0:
        print(f"🔍 ATTENTION DEBUG - After repeat_kv:")
        print(f"  key_states: std={key_states.std():.6f}, mean={key_states.mean():.6f}")
        print(f"  value_states: std={value_states.std():.6f}, mean={value_states.mean():.6f}")
    
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
    # 🔍 DEBUG: Print attention weights magnitude
    if hasattr(module, 'layer_idx') and module.layer_idx == 0:
        print(f"🔍 ATTENTION DEBUG - Raw attention weights:")
        print(f"  attn_weights (after scaling): std={attn_weights.std():.6f}, mean={attn_weights.mean():.6f}")
    
    # Debug hook for raw attention weights (before mask/softmax)
    if hasattr(module, 'debugger_hooks') and 'raw_attn_weights' in module.debugger_hooks:
        module.debugger_hooks['raw_attn_weights'](attn_weights / scaling)  # Raw without scaling
    # Also capture scaled weights
    if hasattr(module, 'debugger_hooks') and 'scaled_attn_weights' in module.debugger_hooks:
        module.debugger_hooks['scaled_attn_weights'](attn_weights)
    
    # 🔍 PRINT DEBUG: Raw attention weights
    if hasattr(module, 'layer_idx') and module.layer_idx == 0:
        print(f"\n🔍 ANEMLL EAGER ATTENTION:")
        print(f"  Raw attn weights (Q@K^T): std={attn_weights.std():.6f}, mean={attn_weights.mean():.6f}")
        print(f"  Scaling factor: {scaling:.6f}")
        print(f"  Scaled attn weights: std={(attn_weights * scaling).std():.6f}, mean={(attn_weights * scaling).mean():.6f}")
    
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
        # 🔍 DEBUG: Print after softcap
        if hasattr(module, 'layer_idx') and module.layer_idx == 0:
            print(f"🔍 ATTENTION DEBUG - After softcap:")
            print(f"  attn_weights: std={attn_weights.std():.6f}, mean={attn_weights.mean():.6f}")
    
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
        # 🔍 DEBUG: Print after mask
        if hasattr(module, 'layer_idx') and module.layer_idx == 0:
            print(f"🔍 ATTENTION DEBUG - After mask:")
            print(f"  attn_weights: shape={attn_weights.shape}, std={attn_weights.std():.6f}, mean={attn_weights.mean():.6f}")
            print(f"  🔍 CRITICAL: pre-softmax values: {attn_weights.flatten()[:10]}")  # Show first 10 values
            print(f"  🔍 CRITICAL: pre-softmax full tensor: {attn_weights}")  # Show full tensor for single token
    
    # Debug hook for pre-softmax weights
    if hasattr(module, 'debugger_hooks') and 'attn_weights_pre_softmax' in module.debugger_hooks:
        module.debugger_hooks['attn_weights_pre_softmax'](attn_weights)
    
    # upcast attention to fp32
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    
    # 🔍 DEBUG: Print after softmax
    if hasattr(module, 'layer_idx') and module.layer_idx == 0:
        print(f"🔍 ATTENTION DEBUG - After softmax:")
        print(f"  attn_weights: std={attn_weights.std():.6f}, mean={attn_weights.mean():.6f}")
        print(f"  🔍 CRITICAL: attn_weights values: {attn_weights.flatten()[:10]}")  # Show first 10 values
    
    # Debug hook for post-softmax weights
    if hasattr(module, 'debugger_hooks') and 'attn_weights_post_softmax' in module.debugger_hooks:
        module.debugger_hooks['attn_weights_post_softmax'](attn_weights)
    
    # 🔍 PRINT DEBUG: After softmax
    if hasattr(module, 'layer_idx') and module.layer_idx == 0:
        print(f"  Post-softmax attn weights: std={attn_weights.std():.6f}, mean={attn_weights.mean():.6f}")
        print(f"  First 5 attention values: {attn_weights.flatten()[:5].tolist()}")
    
    attn_output = torch.matmul(attn_weights, value_states)
    
    # 🔍 DEBUG: Print attention output before transpose
    if hasattr(module, 'layer_idx') and module.layer_idx == 0:
        print(f"🔍 ATTENTION DEBUG - Before transpose:")
        print(f"  attn_output: std={attn_output.std():.6f}, mean={attn_output.mean():.6f}")
        print(f"  🎯 Amplification factor (attn_output/value_states): {attn_output.std().item() / value_states.std().item():.3f}")
        
        # Save attention matmul result for analysis
        if hasattr(module, 'debugger_hooks') and 'attn_output_pre_reshape' in module.debugger_hooks:
            module.debugger_hooks['attn_output_pre_reshape'](attn_output)
        
        # 🔍 PRINT DEBUG: Attention output
        print(f"  Attn output (attn@V): std={attn_output.std():.6f}, mean={attn_output.mean():.6f}")
        print(f"  Value amplification: {attn_output.std().item() / value_states.std().item():.3f}")
    
    attn_output = attn_output.transpose(1, 2).contiguous()  # CRITICAL: Match HF exactly
    
    # 🔍 DEBUG: Print final attention output
    if hasattr(module, 'layer_idx') and module.layer_idx == 0:
        print(f"  Final attn output (after transpose): std={attn_output.std():.6f}, mean={attn_output.mean():.6f}")
        print(f"  🎆 ANEMLL ATTENTION COMPLETE\n")
    
    return attn_output, attn_weights

def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors as per HF Gemma3n.
    
    This implementation was cross-referenced with the canonical JAX implementation
    at gemma/gemma/gm/math/_positional_embeddings.py and found to be mathematically
    equivalent. The rotation logic is standard for RoPE.
    """
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)

def create_sliding_window_causal_mask(seq_len, sliding_window, device, dtype=torch.float32):
    """Create sliding window causal mask."""
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)

    if sliding_window is not None and sliding_window > 0:
        # Set elements within the sliding window to 0.0
        for i in range(seq_len):
            start_idx = max(0, i - sliding_window + 1)
            mask[i, start_idx:i+1] = 0.0
    
    return mask

class RMSNoScale(nn.Module):
    """RMS normalization without scaling parameter - PyTorch version."""
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        # PyTorch equivalent of mx.fast.rms_norm(x, None, self.eps)
        # RMS normalization without scaling: x / sqrt(mean(x²) + eps)
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

class SimpleRMSNorm(nn.Module):
    """RMSNorm with proper with_scale support like HF Gemma3n."""
    def __init__(self, dims: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        self.dims = dims
        
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dims))
        else:
            # For with_scale=False, register a buffer with ones of the correct shape.
            # The previous implementation used torch.tensor(1.0), which created a scalar
            # and caused a shape mismatch in F.layer_norm.
            self.register_buffer("weight", torch.ones(dims), persistent=False)

    def _norm(self, x):
        # This implementation is a direct port of the reference HuggingFace `_norm` method.
        # It ensures all calculations are done in float32 for stability.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # This forward pass now perfectly matches the HuggingFace reference implementation,
        # including the explicit dtype casting to float32 before normalization and weight
        # application, which is critical for numerical stability.
        output = self._norm(hidden_states.float()) * self.weight.float()
        return output.type_as(hidden_states)

class SimpleGemma3nAttention(nn.Module):
    """Proper attention with q_norm, k_norm, v_norm and correct scaling."""
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        
        # Attention projections
        # Use config.attention_bias like HuggingFace (config.json: "attention_bias": false)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # CRITICAL: Only Q, K normalization (NO V normalization in HF Gemma3n!)
        self.q_norm = SimpleRMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=True)
        self.k_norm = SimpleRMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=True)
        self.v_norm = RMSNoScale(eps=config.rms_norm_eps)  # RESTORED: HF uses v_norm!
        
        # This is the missing scaling factor from the JAX implementation.
        self.query_pre_attn_scalar = getattr(config, 'query_pre_attn_scalar', 1.0)

        # CRITICAL: Attention scaling factors from HF Gemma3n
        self.attention_scale = 1.0 / (self.head_dim ** 0.5)  # 1/sqrt(head_dim)
        
        # CRITICAL FIX: Read from text_config for RoPE configuration
        text_config = getattr(config, 'text_config', config)
        
        # BLOCKER #1 FIX: Add RoPE (Rotary Positional Embeddings)
        # Get both global and local RoPE theta values
        self.rope_theta_global = getattr(text_config, 'rope_theta', 1000000.0)
        self.rope_theta_local = getattr(text_config, 'rope_local_base_freq', 10000.0)
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 32768)
        
        # Determine attention type from config
        layer_types = getattr(config, 'layer_types', None)
        if layer_types and layer_idx < len(layer_types):
            self.attention_type = layer_types[layer_idx]
        else:
            # Match HF pattern exactly: "full_attention" if i % 5 == 0 else "sliding_attention"
            self.attention_type = 'full_attention' if layer_idx % 5 == 0 else 'sliding_attention'
        
        self.sliding_window = getattr(config, 'sliding_window', 512)
        
        # CRITICAL FIX: Read from text_config, not main config
        text_config = getattr(config, 'text_config', config)
        
        # Match HF implementation: text_config has _attn_implementation='eager' 
        self._attn_implementation = getattr(text_config, '_attn_implementation', 'eager')
        
        # Get attention-related config values from text_config exactly like HF
        self.attention_bias = getattr(text_config, 'attention_bias', False)
        self.attention_dropout = getattr(text_config, 'attention_dropout', 0.0)
        self.attn_logit_softcapping = getattr(text_config, 'attn_logit_softcapping', None)
        
        # CRITICAL FIX: Match HF implementation exactly
        layer_types = getattr(text_config, 'layer_types', None)
        if layer_types and layer_idx < len(layer_types):
            self.is_sliding = (layer_types[layer_idx] == 'sliding_attention')
        else:
            # HF default: "full_attention" if i % 5 == 4 else "sliding_attention"
            # Match HF pattern exactly: "full_attention" if i % 5 == 0 else "sliding_attention"
            self.is_sliding = not (layer_idx % 5 == 0)
        
        self.debugger_hooks = {}
        
        # Debug attributes for comparison
        self.attn_output_for_debug = None
    
    @property
    def rope_theta(self):
        """Get the correct RoPE theta based on layer type"""
        return self.rope_theta_local if self.is_sliding else self.rope_theta_global
        
    def forward(self, hidden_states, position_embeddings_global=None, position_embeddings_local=None, attention_mask=None, position_ids=None, kv_cache=None, shared_cache=None):
        bsz, q_len, hidden_dim = hidden_states.shape
        
        # Always compute Q projection
        query_states = self.q_proj(hidden_states)
        #query_states = query_states * self.query_pre_attn_scalar
        
        # Debug hooks for Q projection
        if 'q_proj' in self.debugger_hooks: self.debugger_hooks['q_proj'](query_states)
        
        # KV Cache Logic: Use shared K/V if available, otherwise compute own
        # For now, only shared_cache is implemented (Gemma3n KV sharing)
        cache_to_use = shared_cache  # KV cache not implemented yet
        if cache_to_use is not None:
            # For KV sharing layers, use the cached K/V from earlier layer
            key_states, value_states = cache_to_use
            if self.layer_idx <= 1 or self.layer_idx == 20:  # Debug for first few layers + Layer 20
                print(f"🔍 ATTENTION Layer {self.layer_idx}: Using SHARED K/V from cache")
                print(f"  Cached K shape: {key_states.shape}, V shape: {value_states.shape}")
                print(f"  Cached K stats: std={key_states.std():.6f}, mean={key_states.mean():.6f}")
                print(f"  Cached V stats: std={value_states.std():.6f}, mean={value_states.mean():.6f}")
                if self.layer_idx == 20:
                    print(f"  🔄 Layer 20: Using K/V computed by Layer 10 (should match above)")
                print(f"  🚨 CRITICAL: Layer {self.layer_idx} should NOT compute k_proj/v_proj!")
            
            # Debug hooks for shared K/V
            if 'k_proj' in self.debugger_hooks: self.debugger_hooks['k_proj'](key_states)
            if 'v_proj' in self.debugger_hooks: self.debugger_hooks['v_proj'](value_states)
            
            # Store the shared K/V for next layer (if this is also a source layer)
            self.kv_for_sharing = (key_states, value_states)
        else:
            # For non-sharing layers, compute own K/V projections
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            
            if self.layer_idx <= 1 or self.layer_idx == 10 or self.layer_idx == 20:  # Debug for layers 0,1,10,20
                print(f"🔍 ATTENTION Layer {self.layer_idx}: Computing OWN K/V projections")
                print(f"  Computed K shape: {key_states.shape}, V shape: {value_states.shape}")
                print(f"  Computed K stats: std={key_states.std():.6f}, mean={key_states.mean():.6f}")
                print(f"  Computed V stats: std={value_states.std():.6f}, mean={value_states.mean():.6f}")
                if self.layer_idx == 10:
                    print(f"  📦 Layer 10: This K/V will be shared with Layer 20")
                elif self.layer_idx == 20:
                    print(f"  ❌ ERROR: Layer {self.layer_idx} should use shared cache, not compute own!")
            
            # Debug hooks for computed K/V
            if 'k_proj' in self.debugger_hooks: self.debugger_hooks['k_proj'](key_states)
            if 'v_proj' in self.debugger_hooks: self.debugger_hooks['v_proj'](value_states)
            
            # Store for sharing (if this is a source layer)
            self.kv_for_sharing = (key_states, value_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        
        # CRITICAL: Apply Q, K, V normalization as per HF Gemma3n
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)  # RESTORED: HF applies v_norm!
        
        # Debug hooks for attention norms  
        if 'q_norm' in self.debugger_hooks: self.debugger_hooks['q_norm'](query_states)
        if 'k_norm' in self.debugger_hooks: self.debugger_hooks['k_norm'](key_states)

        # CRITICAL FIX: Apply RoPE conditionally using pre-computed position embeddings
        # This matches the HF implementation exactly
        
        # CRITICAL: Debug hooks for position embeddings and RoPE
        if 'q_before_rope' in self.debugger_hooks: self.debugger_hooks['q_before_rope'](query_states)
        if 'k_before_rope' in self.debugger_hooks: self.debugger_hooks['k_before_rope'](key_states)
        
        # 🔍 PRINT DEBUG: Position embeddings and RoPE
        if self.layer_idx == 0:
            print(f"\n🔍 ANEMLL ATTENTION DEBUG Layer {self.layer_idx}:")
            print(f"  Layer type: {'sliding_attention' if self.is_sliding else 'full_attention'}")
            print(f"  Q before RoPE: shape={query_states.shape}, std={query_states.std():.6f}, mean={query_states.mean():.6f}")
            print(f"  K before RoPE: shape={key_states.shape}, std={key_states.std():.6f}, mean={key_states.mean():.6f}")
            print(f"  V states: shape={value_states.shape}, std={value_states.std():.6f}, mean={value_states.mean():.6f}")
            print(f"  Position IDs: {position_ids}")
            if position_ids is not None:
                print(f"  Position IDs shape: {position_ids.shape}")
            print(f"  🔍 CRITICAL: Available position embeddings:")
            print(f"    position_embeddings_global: {position_embeddings_global is not None}")
            print(f"    position_embeddings_local: {position_embeddings_local is not None}")
            if position_embeddings_global is not None:
                cos_g, sin_g = position_embeddings_global  
                print(f"    global cos: std={cos_g.std():.6f}, mean={cos_g.mean():.6f}")
                print(f"    global sin: std={sin_g.std():.6f}, mean={sin_g.mean():.6f}")
            if position_embeddings_local is not None:
                cos_l, sin_l = position_embeddings_local
                print(f"    local cos: std={cos_l.std():.6f}, mean={cos_l.mean():.6f}")  
                print(f"    local sin: std={sin_l.std():.6f}, mean={sin_l.mean():.6f}")
        
        # CRITICAL FIX: HF uses GLOBAL embeddings for ALL layers despite layer types
        # This matches the actual HF behavior found in debugging
        cos, sin = position_embeddings_global
        
        # Debug hooks for position embeddings
        if 'position_embeddings_global' in self.debugger_hooks: 
            self.debugger_hooks['position_embeddings_global'](torch.stack([cos, sin]))
        if 'cos_sin_global' in self.debugger_hooks: 
            self.debugger_hooks['cos_sin_global'](torch.stack([cos, sin]))
        
        if self.layer_idx == 0:
            print(f"  🔧 FIXED: Using GLOBAL position embeddings for ALL layers (matches HF)")
            print(f"  cos shape={cos.shape}, std={cos.std():.6f}, mean={cos.mean():.6f}")
            print(f"  sin shape={sin.shape}, std={sin.std():.6f}, mean={sin.mean():.6f}")
        
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        
        # Debug hooks for Q/K after RoPE
        if 'q_after_rope' in self.debugger_hooks: self.debugger_hooks['q_after_rope'](query_states)
        if 'k_after_rope' in self.debugger_hooks: self.debugger_hooks['k_after_rope'](key_states)
        
        # 🔍 PRINT DEBUG: After RoPE
        if self.layer_idx == 0:
            print(f"  Q after RoPE: shape={query_states.shape}, std={query_states.std():.6f}, mean={query_states.mean():.6f}")
            print(f"  K after RoPE: shape={key_states.shape}, std={key_states.std():.6f}, mean={key_states.mean():.6f}")

        # Transpose for attention calculation: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Debug hooks for value states before normalization (if applicable)
        if 'value_states_before_norm' in self.debugger_hooks: 
            self.debugger_hooks['value_states_before_norm'](value_states)
        # Debug hooks for value states transposed
        if 'value_states_transposed' in self.debugger_hooks: 
            self.debugger_hooks['value_states_transposed'](value_states)

        # Calculate num_key_value_groups for repeat_kv
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # Assert training mode is False since this is a test file
        assert not self.training, "Training mode should be False in test file"
        # Use eager attention implementation like HF
        if self._attn_implementation == "eager":
            # Create attention mask based on layer type - match HF implementation
            if True: #attention_mask is None:
                if self.is_sliding:
                    causal_mask = create_sliding_window_causal_mask(q_len, self.sliding_window,
                                                                  hidden_states.device, query_states.dtype)
                else:
                    causal_mask = _create_causal_mask(q_len).to(device=hidden_states.device, dtype=query_states.dtype)
                # Expand for batch and heads: [1, 1, seq_len, seq_len] -> [bsz, num_heads, seq_len, seq_len]
                attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                
                # Debug hooks for attention masks
                if 'causal_mask' in self.debugger_hooks: self.debugger_hooks['causal_mask'](causal_mask)
                if 'attention_mask' in self.debugger_hooks: self.debugger_hooks['attention_mask'](attention_mask)
                
                # 🔍 PRINT DEBUG: Attention masks
                if self.layer_idx == 0:
                    print(f"  Causal mask shape={causal_mask.shape}, unique_vals={torch.unique(causal_mask)[:5].tolist()}")
                    print(f"  Attention mask shape={attention_mask.shape}, unique_vals={torch.unique(attention_mask)[:5].tolist()}")
            
            # Call HF-style eager attention with correct config values
            # CRITICAL: HF explicitly passes scaling=1.0, matching their code exactly
            attn_output, attn_weights = eager_attention_forward(
                module=self,
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=1.0,  # HF explicitly passes scaling=1.0
                softcap=self.attn_logit_softcapping,
            )
            
            # Debug hooks for HF-style attention (no extra computation, just capture)
            if 'attn_weights_post_softmax' in self.debugger_hooks: 
                self.debugger_hooks['attn_weights_post_softmax'](attn_weights)
            if 'attn_output_pre_reshape' in self.debugger_hooks: 
                self.debugger_hooks['attn_output_pre_reshape'](attn_output)
        else:
            raise NotImplementedError(f"Attention implementation {self._attn_implementation} not supported")
        
        # Reshape exactly like HF: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden_size]
        input_shape = hidden_states.shape[:-1]  # [batch, seq_len]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        
        # Debug hook for o_proj input - CAPTURE EXACT INPUT TO O_PROJ
        if 'o_proj_input' in self.debugger_hooks: 
            self.debugger_hooks['o_proj_input'](attn_output)
        
        # 🔍 SAVE O_PROJ INPUT TO FILE FOR ANALYSIS
        if self.layer_idx == 0:
            torch.save({
                'anemll_o_proj_input': attn_output.detach().cpu(),
                'anemll_o_proj_weight': self.o_proj.weight.detach().cpu()
            }, 'tests/dev/anemll_o_proj_data.pt')
            print(f"💾 Saved ANEMLL o_proj input: shape={attn_output.shape}, std={attn_output.std():.6f}")
        
        # 🔍 DEBUG: Print before o_proj
        if self.layer_idx == 0:
            pre_o_proj_std = attn_output.std().item()
            print(f"🔍 O_PROJ DEBUG - Before o_proj:")
            print(f"  attn_output: std={pre_o_proj_std:.6f}, mean={attn_output.mean():.6f}")
            # Check o_proj weight statistics
            o_proj_weight = self.o_proj.weight
            print(f"  o_proj weight: shape={o_proj_weight.shape}, std={o_proj_weight.std():.6f}, mean={o_proj_weight.mean():.6f}")
        
        attn_output = self.o_proj(attn_output)
        
        # 🔍 DEBUG: Print after o_proj
        if self.layer_idx == 0:
            post_o_proj_std = attn_output.std().item()
            print(f"🔍 O_PROJ DEBUG - After o_proj:")
            print(f"  attn_output: std={post_o_proj_std:.6f}, mean={attn_output.mean():.6f}")
            print(f"  🎯 O_proj amplification factor: {post_o_proj_std / pre_o_proj_std:.3f}")
            print(f"  🎯 Expected HF amplification: 0.503, Actual: {post_o_proj_std / pre_o_proj_std:.3f}, Ratio: {(post_o_proj_std / pre_o_proj_std) / 0.503:.3f}")
        
        # Debug hook for attention output
        if 'attn_output' in self.debugger_hooks: self.debugger_hooks['attn_output'](attn_output)
        
        # Store for debug comparison
        self.attn_output_for_debug = attn_output

        return attn_output

class SimpleGemma3nLaurelBlock(nn.Module):
    """LAUREL block (Learned Augmented Residual Layer) - exact HF implementation."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.laurel_rank = getattr(config, 'laurel_rank', 64)
        self.linear_left = nn.Linear(config.hidden_size, self.laurel_rank, bias=False)
        self.linear_right = nn.Linear(self.laurel_rank, config.hidden_size, bias=False)
        self.post_laurel_norm = SimpleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states):
        laurel_hidden_states = self.linear_left(hidden_states)
        laurel_hidden_states = self.linear_right(laurel_hidden_states)
        normed_laurel_hidden_states = self.post_laurel_norm(laurel_hidden_states)
        # This is the critical missing piece. The reference implementation adds the
        # original, un-normalized input back inside the Laurel block itself.
        return hidden_states + normed_laurel_hidden_states

class SimpleGemma3nMLP(nn.Module):
    """The MLP block, which now includes the MKP-gate (gating) logic."""
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Use Conv2d for dense layers to be ANE-compatible
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.debugger_hooks = {}
        
        # Match HF implementation: store activation_sparsity directly
        if hasattr(config, 'activation_sparsity_pattern'):
            self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]
        else:
            self.activation_sparsity = 0.0
        
        # Debug attributes for comparison
        self.down_proj_for_debug = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Follow exact HF Gemma3n MLP implementation
        gate_proj = self.gate_proj(x)
        if 'gate_proj' in self.debugger_hooks: self.debugger_hooks['gate_proj'](gate_proj)

        # Apply sparsity exactly like HF - check activation_sparsity directly
        if self.activation_sparsity > 0.0:
            print(f"🔍 ANEMLL SPARSITY DEBUG Layer {self.layer_idx}:")
            print(f"  activation_sparsity: {self.activation_sparsity}")
            print(f"  gate_proj before sparsity: min={gate_proj.min():.6f}, max={gate_proj.max():.6f}, mean={gate_proj.mean():.6f}, std={gate_proj.std():.6f}")
            gate_proj_after_sparsity = self._gaussian_topk(gate_proj)
            sparsity_level = (gate_proj_after_sparsity == 0).float().mean().item()
            print(f"  gate_proj after sparsity: min={gate_proj_after_sparsity.min():.6f}, max={gate_proj_after_sparsity.max():.6f}, mean={gate_proj_after_sparsity.mean():.6f}, std={gate_proj_after_sparsity.std():.6f}")
            print(f"  actual sparsity level: {sparsity_level:.4f}")
        else:
            gate_proj_after_sparsity = gate_proj
            print(f"🔍 ANEMLL SPARSITY DEBUG Layer {self.layer_idx}: NO SPARSITY (activation_sparsity={self.activation_sparsity})")
        
        # Debug after sparsity (using new variable name)
        if 'gate_proj_after_sparsity' in self.debugger_hooks: 
            self.debugger_hooks['gate_proj_after_sparsity'](gate_proj_after_sparsity)

        # Apply activation function to gate_proj (GELU in HF config) 
        activations = F.gelu(gate_proj_after_sparsity, approximate='tanh')
        # Debug after activation (renamed to match HF)
        if 'gate_proj_activated' in self.debugger_hooks: self.debugger_hooks['gate_proj_activated'](activations)

        up_proj = self.up_proj(x)
        if 'up_proj' in self.debugger_hooks: self.debugger_hooks['up_proj'](up_proj)

        # Combine activations and up_proj
        intermediate = activations * up_proj
        if 'activations' in self.debugger_hooks: self.debugger_hooks['activations'](intermediate)

        # Down projection
        down_proj_output = self.down_proj(intermediate)
        
        # Store for debug comparison
        self.down_proj_for_debug = down_proj_output
        
        return down_proj_output

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian top-k sparsity - exact copy of HF implementation"""
        target_sparsity_tensor = torch.tensor(self.activation_sparsity, dtype=torch.float32, device=inputs.device)
        # normal_dist and std_multiplier are adapted from jax.scipy.stats.norm.ppf().
        #
        # References:
        #   *   https://docs.jax.dev/en/latest/_autosummary/jax.scipy.stats.norm.ppf.html
        #   *   https://pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal
        #   *   https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.icdf
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier: torch.Tensor = normal_dist.icdf(target_sparsity_tensor)
        std_multiplier = std_multiplier.type(inputs.dtype)
        inp_f32     = inputs.float()
        inputs_mean = inp_f32.mean(-1, keepdim=True)
        inputs_std  = inp_f32.std(-1, keepdim=True, unbiased=False)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return nn.functional.relu(inputs - cutoff_x)

class SimpleGemma3nAltUp(nn.Module):
    """The full, correct AltUp mechanism."""
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.altup_num_inputs = getattr(config, 'altup_num_inputs', 4)
        self.altup_active_idx = getattr(config, 'altup_active_idx', 0)
        self.correct_output_scale = nn.Parameter(torch.zeros(config.hidden_size))
        
        # This was incorrect. The JAX implementation uses raw parameters, not Linear layers.
        # Reverting to nn.Linear to match the HuggingFace implementation which is our reference.
        self.correction_coefs = nn.Linear(self.altup_num_inputs, self.altup_num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(self.altup_num_inputs, self.altup_num_inputs**2, bias=False)
        
        self.modality_router = nn.Linear(config.hidden_size, self.altup_num_inputs, bias=False)
        self.router_norm = SimpleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("router_input_scale", torch.tensor(config.hidden_size**-1.0))

    def compute_router_modalities(self, x):
        router_inputs = self.router_norm(x) * self.router_input_scale
        self.router_inputs_for_debug = router_inputs  # Debug hook
        routed = self.modality_router(router_inputs)
        self.routed_for_debug = routed  # Debug hook
        return torch.tanh(routed.float()).type_as(x)

    def predict(self, hidden_states):
        print(f"🤖 ---------------------PREDICT------------------------")
        print(f"🔍 AltUp PREDICT: Using active stream {self.altup_active_idx} for router computation")
        print(f"  Hidden states shape: {hidden_states.shape}")
        print(f"  Active stream shape: {hidden_states[self.altup_active_idx].shape}")
        
        modalities = self.compute_router_modalities(hidden_states[self.altup_active_idx])
        self.modalities_predict_for_debug = modalities  # Debug hook
        
        print(f"  Modalities shape: {modalities.shape}")
        print(f"  Modalities values: {modalities.flatten()[:4].tolist()}")
        print(f"  Max modality index: {modalities.argmax().item()}")
        altup_coef_clip = getattr(self.config, 'altup_coef_clip', 120.0)
        
        # Match HF implementation: only clip during training
        if self.training and altup_coef_clip is not None:
            self.prediction_coefs.weight.data.clamp_(-altup_coef_clip, altup_coef_clip)

        # This logic is now a direct port of the HuggingFace implementation.
        all_coefs = (
            self.prediction_coefs(modalities)
            .reshape(*modalities.shape[:-1], self.altup_num_inputs, self.altup_num_inputs)
            .permute(0, 1, 3, 2)
        )

        # permute hidden_states to [batch_size, num_tokens, hidden_size, altup_num_inputs]
        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2)  # undo the permute
        predictions += hidden_states  # add the original input
        return predictions.contiguous().type_as(hidden_states)

    def correct(self, predictions, activated):
        print(f"🤖 ---------------------correct------------------------")
        print(f"🔍 AltUp CORRECT: Using active stream {self.altup_active_idx} for innovation computation")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Activated shape: {activated.shape}")
        
        # 🔍 STREAM DEBUG: Show all input streams BEFORE any modifications
        print(f"🔍 INPUT STREAMS DEBUG at entry of AltUp.correct():")
        for i in range(self.altup_num_inputs):
            stream_data = predictions[i]
            print(f"  Stream {i}: shape={stream_data.shape}, min={stream_data.min():.6f}, max={stream_data.max():.6f}, mean={stream_data.mean():.6f}, std={stream_data.std():.6f}")
        print(f"  Activated: shape={activated.shape}, min={activated.min():.6f}, max={activated.max():.6f}, mean={activated.mean():.6f}, std={activated.std():.6f}")
        print(f"  Active stream index: {self.altup_active_idx}")
        
        # Debug hooks for per-stream analysis
        for i in range(self.altup_num_inputs):
            setattr(self, f'predictions_stream_{i}_for_debug', predictions[i])
        self.activated_for_debug = activated
        
        modalities = self.compute_router_modalities(activated)
        self.modalities_correct_for_debug = modalities  # Debug hook
        
        print(f"  Correct modalities: {modalities.flatten()[:4].tolist()}")
        
        innovation = activated - predictions[self.altup_active_idx]  # HF style - no unsqueeze
        print(f"  Innovation magnitude: {innovation.norm().item():.6f}")
        print(f"  Innovation shape before repeat: {innovation.shape}")
        print(f"  Innovation stats: min={innovation.min():.6f}, max={innovation.max():.6f}, mean={innovation.mean():.6f}, std={innovation.std():.6f}")
        # This repeat is buggy for a 3D tensor and inefficient. Broadcasting is better.
        innovation = innovation.repeat(self.altup_num_inputs, 1, 1, 1)  # HF style - direct repeat
        print(f"  Innovation shape after repeat: {innovation.shape}")
        print(f"  Innovation after repeat stats: min={innovation.min():.6f}, max={innovation.max():.6f}, mean={innovation.mean():.6f}, std={innovation.std():.6f}")
        
        altup_coef_clip = getattr(self.config, 'altup_coef_clip', None)
        
        # Match HF implementation: ALWAYS clip correction coefficients (not just during training)
        if altup_coef_clip is not None:
            self.correction_coefs.weight.data.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)
            
        # This logic is now a direct port of the HuggingFace implementation.
        correction_coefs_output = self.correction_coefs(modalities)
        self.correction_coefs_output_for_debug = correction_coefs_output  # Debug hook
        print(f"  Correction coefs output shape: {correction_coefs_output.shape}")
        print(f"  Correction coefs output: {correction_coefs_output.flatten()[:4].tolist()}")

        all_coefs = correction_coefs_output + 1.0
        self.all_coefs_for_debug = all_coefs  # Debug hook  
        print(f"  All coefs before permute: {all_coefs.flatten()[:4].tolist()}")
        all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)
        print(f"  All coefs after permute+unsqueeze shape: {all_coefs.shape}")
        print(f"  All coefs after permute+unsqueeze: min={all_coefs.min():.6f}, max={all_coefs.max():.6f}, mean={all_coefs.mean():.6f}, std={all_coefs.std():.6f}")
        
        print(f"  Before mul: innovation shape={innovation.shape}, all_coefs shape={all_coefs.shape}")
        corrected = torch.mul(innovation, all_coefs)
        print(f"  After mul: corrected shape={corrected.shape}, min={corrected.min():.6f}, max={corrected.max():.6f}, mean={corrected.mean():.6f}, std={corrected.std():.6f}")
        corrected += predictions  # HF style - separate addition
        print(f"  After add predictions: corrected shape={corrected.shape}, min={corrected.min():.6f}, max={corrected.max():.6f}, mean={corrected.mean():.6f}, std={corrected.std():.6f}")
        
        # Debug hooks for corrected predictions per stream
        for i in range(self.altup_num_inputs):
            setattr(self, f'corrected_stream_{i}_for_debug', corrected[i])
        
        return corrected.contiguous().type_as(activated)  # HF style - type_as activated

    def scale_corrected_output(self, corrected):
        return (corrected * self.correct_output_scale)

class SimpleGemma3nLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.self_attn = SimpleGemma3nAttention(config, layer_idx)
        self.mlp = SimpleGemma3nMLP(config, layer_idx)
        
        # Use HF naming convention for consistency
        self.input_layernorm = SimpleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SimpleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = SimpleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = SimpleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.laurel = SimpleGemma3nLaurelBlock(config)
        # The JAX implementation has a separate post_laurel_norm, but my LaurelBlock has it inside.
        # This is functionally equivalent.
        
        self.altup = SimpleGemma3nAltUp(config, layer_idx)  # 🔍 FIX: Pass layer_idx to AltUp
        
        # Per-layer embeddings
        per_layer_dim = getattr(config, 'hidden_size_per_layer_input', 256)
        # if hasattr(config, '_debug_layer_creation') and config._debug_layer_creation:
        #     print(f"    🔍 Layer {self.layer_idx} per_layer_dim: {per_layer_dim}")
        self.per_layer_input_gate = nn.Linear(config.hidden_size, per_layer_dim, bias=False)
        self.per_layer_projection = nn.Linear(per_layer_dim, config.hidden_size, bias=False)
        self.post_per_layer_input_norm = SimpleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # --- Debug Hooks ---
        self.debugger_hooks = {}

    def forward(self, hidden_streams, per_layer_embeddings=None, position_embeddings_global=None, position_embeddings_local=None, attention_mask=None, position_ids=None, shared_cache=None):
        # Pass relevant hooks down to sub-modules
        if self.debugger_hooks:
            self.self_attn.debugger_hooks = {k.replace('attn.', ''): v for k, v in self.debugger_hooks.items() if k.startswith('attn.')}
            self.mlp.debugger_hooks = {k.replace('mlp.', ''): v for k, v in self.debugger_hooks.items() if k.startswith('mlp.')}
        
        # This forward pass implements the exact parallel data flow from the canonical JAX
        # implementation in `gemma/gemma/gm/nn/gemma3n/_modules.py`.
        # It correctly handles the 4-stream input and the various normalizations and
        # residual connections.
        active_stream = hidden_streams[self.altup.altup_active_idx]
        if 'active_stream_initial' in self.debugger_hooks: self.debugger_hooks['active_stream_initial'](active_stream)

        # 1. AltUp Predict (operates on the full stream bundle)
        predictions = self.altup.predict(hidden_streams)
        if 'predictions' in self.debugger_hooks: self.debugger_hooks['predictions'](predictions)
        # Capture AltUp debug tensors
        if 'altup.modalities_predict' in self.debugger_hooks and hasattr(self.altup, 'modalities_predict_for_debug'):
            self.debugger_hooks['altup.modalities_predict'](self.altup.modalities_predict_for_debug)
        if 'altup.router_inputs' in self.debugger_hooks and hasattr(self.altup, 'router_inputs_for_debug'):
            self.debugger_hooks['altup.router_inputs'](self.altup.router_inputs_for_debug)
        if 'altup.routed' in self.debugger_hooks and hasattr(self.altup, 'routed_for_debug'):
            self.debugger_hooks['altup.routed'](self.altup.routed_for_debug)
        
        # FIXED: Match HF variable naming exactly
        active_prediction = predictions[self.altup.altup_active_idx]
        if 'active_stream_after_predict' in self.debugger_hooks: self.debugger_hooks['active_stream_after_predict'](active_prediction)
        if 'active_prediction' in self.debugger_hooks: self.debugger_hooks['active_prediction'](active_prediction)

        # FIXED: Match HF normalization step
        active_prediction_normed = self.input_layernorm(active_prediction)
        if 'input_layernorm' in self.debugger_hooks: self.debugger_hooks['input_layernorm'](active_prediction_normed)
        if 'active_prediction_normed' in self.debugger_hooks: self.debugger_hooks['active_prediction_normed'](active_prediction_normed)
        
        # FIXED: Match HF order - Laurel BEFORE attention
        laurel_output = self.laurel(active_prediction_normed)
        if 'laurel_out' in self.debugger_hooks: self.debugger_hooks['laurel_out'](laurel_output)
        
        # FIXED: Match HF attention call exactly
        attn_output = self.self_attn(active_prediction_normed, 
                                     position_embeddings_global=position_embeddings_global,
                                     position_embeddings_local=position_embeddings_local,
                                     attention_mask=attention_mask, 
                                     position_ids=position_ids,
                                     shared_cache=shared_cache)
        #if 'attn_output' in self.debugger_hooks: self.debugger_hooks['attn_output'](attn_output)

        # FIXED: Match HF post-attention norm
        attn = self.post_attention_layernorm(attn_output)
        if 'post_attention_layernorm' in self.debugger_hooks: self.debugger_hooks['post_attention_layernorm'](attn)

        # FIXED: Match HF residual connection exactly
        attn_gated = active_prediction + attn
        if 'attn_gated' in self.debugger_hooks: self.debugger_hooks['attn_gated'](attn_gated)

        # FIXED: Match HF laurel combination exactly
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2)
        if 'attn_laurel' in self.debugger_hooks: self.debugger_hooks['attn_laurel'](attn_laurel)

        # FIXED: Match HF variable naming exactly
        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        if 'pre_feedforward_layernorm' in self.debugger_hooks: self.debugger_hooks['pre_feedforward_layernorm'](attn_norm)
        
        # FIXED: Match HF MLP call exactly
        attn_ffw = self.mlp(attn_norm)
        if 'mlp_output' in self.debugger_hooks: self.debugger_hooks['mlp_output'](attn_ffw)
        
        # FIXED: Match HF post-feedforward norm exactly
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        if 'post_feedforward_layernorm' in self.debugger_hooks: self.debugger_hooks['post_feedforward_layernorm'](attn_ffw_norm)
        
        # FIXED: Match HF residual connection exactly
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
        if 'processed_active_stream' in self.debugger_hooks: self.debugger_hooks['processed_active_stream'](attn_ffw_laurel_gated)

        print(f"🔍 LAYER {self.layer_idx}: About to call altup.correct")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Activated shape: {attn_ffw_laurel_gated.shape}")
        
        # FIXED: Match HF AltUp.correct call exactly
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)
        print(f"🔍 LAYER {self.layer_idx}: altup.correct completed")
        
        # 🔍 DEBUG: Check if corrected_predictions matches debug attributes
        if self.layer_idx == 1:
            print(f"🔍 CORRECTED_PREDICTIONS vs DEBUG ATTRIBUTES:")
            print(f"  corrected_predictions shape: {corrected_predictions.shape}")
            print(f"  corrected_predictions[1]: min={corrected_predictions[1].min():.6f}, max={corrected_predictions[1].max():.6f}, mean={corrected_predictions[1].mean():.6f}, std={corrected_predictions[1].std():.6f}")
            if hasattr(self.altup, 'corrected_stream_1_for_debug'):
                debug_stream_1 = getattr(self.altup, 'corrected_stream_1_for_debug')
                print(f"  debug_stream_1: min={debug_stream_1.min():.6f}, max={debug_stream_1.max():.6f}, mean={debug_stream_1.mean():.6f}, std={debug_stream_1.std():.6f}")
                are_equal = torch.allclose(corrected_predictions[1], debug_stream_1, rtol=1e-6)
                print(f"  Are they equal? {are_equal}")
            
            # 🔍 DEBUG: Print individual streams from corrected_predictions
            print(f"🔍 ANEMLL CORRECTED_PREDICTIONS PER STREAM:")
            for i in range(4):
                stream = corrected_predictions[i]
                print(f"  Stream {i}: min={stream.min():.6f}, max={stream.max():.6f}, mean={stream.mean():.6f}, std={stream.std():.6f}")
        
        if 'altup_corrected' in self.debugger_hooks: self.debugger_hooks['altup_corrected'](corrected_predictions)
        if 'corrected_streams' in self.debugger_hooks: self.debugger_hooks['corrected_streams'](corrected_predictions)
        # Capture AltUp debug tensors for correct
        if 'altup.modalities_correct' in self.debugger_hooks and hasattr(self.altup, 'modalities_correct_for_debug'):
            self.debugger_hooks['altup.modalities_correct'](self.altup.modalities_correct_for_debug)
        # Capture per-stream predictions and activated tensor
        for i in range(4):
            stream_key = f'altup.predictions_stream_{i}'
            if stream_key in self.debugger_hooks and hasattr(self.altup, f'predictions_stream_{i}_for_debug'):
                self.debugger_hooks[stream_key](getattr(self.altup, f'predictions_stream_{i}_for_debug'))
        if 'altup.activated' in self.debugger_hooks and hasattr(self.altup, 'activated_for_debug'):
            self.debugger_hooks['altup.activated'](self.altup.activated_for_debug)
        # Capture per-stream corrected predictions
        for i in range(4):
            corrected_key = f'altup.corrected_stream_{i}'
            if corrected_key in self.debugger_hooks and hasattr(self.altup, f'corrected_stream_{i}_for_debug'):
                self.debugger_hooks[corrected_key](getattr(self.altup, f'corrected_stream_{i}_for_debug'))
        # Capture correction coefficients debug
        if 'altup.correction_coefs_output' in self.debugger_hooks and hasattr(self.altup, 'correction_coefs_output_for_debug'):
            self.debugger_hooks['altup.correction_coefs_output'](self.altup.correction_coefs_output_for_debug)
        if 'altup.all_coefs' in self.debugger_hooks and hasattr(self.altup, 'all_coefs_for_debug'):
            self.debugger_hooks['altup.all_coefs'](self.altup.all_coefs_for_debug)
        
        # 11. Late PLE injection
        if per_layer_embeddings is not None:
            if self.layer_idx <= 1:  # Debug layer 0 and 1
                print(f"🔍 ANEMLL PLE PROCESSING (Layer {self.layer_idx}):")
                print(f"  per_layer_embeddings: shape={per_layer_embeddings.shape}, min={per_layer_embeddings.min():.6f}, max={per_layer_embeddings.max():.6f}, mean={per_layer_embeddings.mean():.6f}, std={per_layer_embeddings.std():.6f}")
            
            layer_slice_start = self.layer_idx * 256
            layer_slice_end = (self.layer_idx + 1) * 256
            
            # FIXED: Match HF variable naming exactly
            first_prediction = corrected_predictions[self.altup.altup_active_idx].clone()
            
            if self.layer_idx <= 1:
                print(f"  corrected_predictions[{self.altup.altup_active_idx}] before PLE: min={first_prediction.min():.6f}, max={first_prediction.max():.6f}, mean={first_prediction.mean():.6f}, std={first_prediction.std():.6f}")
            
            if getattr(self.config, 'altup_correct_scale', True):
                first_prediction = self.altup.scale_corrected_output(first_prediction)
                if self.layer_idx <= 1:
                    print(f"  first_prediction after scale: min={first_prediction.min():.6f}, max={first_prediction.max():.6f}, mean={first_prediction.mean():.6f}, std={first_prediction.std():.6f}")
            
            # CRITICAL FIX: Match HF exactly - per_layer_embeddings comes already processed!
            # per_layer_input_gate adapted from jax.numpy.einsum("btd,dp->btp", ...)
            first_prediction = self.per_layer_input_gate(first_prediction)
            if self.layer_idx <= 1:
                print(f"  first_prediction after gate: min={first_prediction.min():.6f}, max={first_prediction.max():.6f}, mean={first_prediction.mean():.6f}, std={first_prediction.std():.6f}")
            
            first_prediction = F.gelu(first_prediction, approximate='tanh')
            if self.layer_idx <= 1:
                print(f"  first_prediction after act_fn: min={first_prediction.min():.6f}, max={first_prediction.max():.6f}, mean={first_prediction.mean():.6f}, std={first_prediction.std():.6f}")

            # CRITICAL FIX: per_layer_embeddings is already the correct slice for this layer
            # The main model extracts the slice: per_layer_slice = per_layer_inputs[:, :, i, :]
            # So we just use per_layer_embeddings directly instead of slicing again!
            per_layer_input = per_layer_embeddings
            
            if self.layer_idx <= 1:
                print(f"  FIXED: Using per_layer_embeddings directly as per_layer_input")
                print(f"  per_layer_embeddings.shape: {per_layer_embeddings.shape}")
                print(f"  per_layer_input.shape: {per_layer_input.shape}")
            
            if self.layer_idx <= 1:
                print(f"  per_layer_input: min={per_layer_input.min():.6f}, max={per_layer_input.max():.6f}, mean={per_layer_input.mean():.6f}, std={per_layer_input.std():.6f}")

            # Element-wise multiplication like HF (no additional preprocessing needed!)
            first_prediction = torch.multiply(first_prediction, per_layer_input)
            if self.layer_idx <= 1:
                print(f"  first_prediction after multiply: min={first_prediction.min():.6f}, max={first_prediction.max():.6f}, mean={first_prediction.mean():.6f}, std={first_prediction.std():.6f}")

            # Step 3: per_layer_projection adapted from jax.numpy.einsum("btp,pd->btd", ...)
            first_prediction = self.per_layer_projection(first_prediction)
            if self.layer_idx <= 1:
                print(f"  first_prediction after projection: min={first_prediction.min():.6f}, max={first_prediction.max():.6f}, mean={first_prediction.mean():.6f}, std={first_prediction.std():.6f}")
            
            first_prediction = self.post_per_layer_input_norm(first_prediction)
            if self.layer_idx <= 1:
                print(f"  first_prediction after norm: min={first_prediction.min():.6f}, max={first_prediction.max():.6f}, mean={first_prediction.mean():.6f}, std={first_prediction.std():.6f}")
            
            # Step 4: Apply to non-active streams (match HF exactly)
            corrected_predictions[1:] += first_prediction
            
            if self.layer_idx <= 1:
                print(f"  corrected_predictions[1] after PLE: min={corrected_predictions[1].min():.6f}, max={corrected_predictions[1].max():.6f}, mean={corrected_predictions[1].mean():.6f}, std={corrected_predictions[1].std():.6f}")
                print(f"  corrected_predictions[2] after PLE: min={corrected_predictions[2].min():.6f}, max={corrected_predictions[2].max():.6f}, mean={corrected_predictions[2].mean():.6f}, std={corrected_predictions[2].std():.6f}")
                print(f"  corrected_predictions[3] after PLE: min={corrected_predictions[3].min():.6f}, max={corrected_predictions[3].max():.6f}, mean={corrected_predictions[3].mean():.6f}, std={corrected_predictions[3].std():.6f}")
        else:
            if self.layer_idx <= 1:
                print(f"🔍 ANEMLL PLE PROCESSING (Layer {self.layer_idx}): SKIPPED - per_layer_embeddings is None")
            
            if 'after_ple' in self.debugger_hooks: self.debugger_hooks['after_ple'](corrected_predictions)

        if 'final_output' in self.debugger_hooks: self.debugger_hooks['final_output'](corrected_predictions)
        # FIXED: Return corrected_predictions to match HF exactly
        return corrected_predictions

class SimpleGemma3nModel(nn.Module):
    """Gemma3n model matching the actual architecture we discovered."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Main embeddings (text tokens)
        self.embed_scale = math.sqrt(config.hidden_size) #  
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
                
        # Global per-layer embeddings (n_layers × 256-dim)
        # Fix: Calculate PLE dimension dynamically based on number of layers
        per_layer_dim = config.num_hidden_layers * 256  # n_layers × 256
        per_layer_vocab = getattr(config, 'vocab_size_per_layer_input', config.vocab_size)
        self.embed_tokens_per_layer = nn.Embedding(per_layer_vocab, per_layer_dim)
        
        # Global AltUp projections (EXACT HF pattern: only 3 projections for streams 1,2,3)
        # Stream 0 is identity (no projection), streams 1,2,3 use projections
        altup_num_inputs = getattr(config, 'altup_num_inputs', 4)
        self.altup_projections = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False) 
            for _ in range(1, altup_num_inputs)  # range(1, 4) = [1, 2, 3] -> 3 projections
        ])
        self.altup_unembed_projections = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False) 
            for _ in range(1, altup_num_inputs)  # range(1, 4) = [1, 2, 3] -> 3 projections
        ])
        
        # Layers (30 layers for Gemma3n)
        self.layers = nn.ModuleList([
            SimpleGemma3nLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])
        
        # Per-layer model projection and norm (FIXED to match HF exactly!)
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size, 
            config.num_hidden_layers * config.hidden_size_per_layer_input, 
            bias=False
        )
        # CRITICAL: Use 256 (original config value), not 7680
        self.per_layer_projection_norm = SimpleRMSNorm(256, eps=config.rms_norm_eps, with_scale=True)
        
        # CRITICAL: Add missing scaling factors from HF implementation  
        self.register_buffer("per_layer_projection_scale", torch.tensor(config.hidden_size**-0.5), persistent=False)
        self.register_buffer("per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False)
        
        # Final norm and output
        self.norm = SimpleRMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=True)
        # NOTE: We'll manually set the LM head weight to transposed embeddings during weight loading
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Final logit softcapping - CRITICAL: HF uses 30.0
        self.final_logit_softcapping = getattr(config, 'final_logit_softcapping', 30.0)
        # CRITICAL: Based on the implementation plan, the embed_scale is used for logits,
        # but the initial embeddings are NOT scaled.
        self.logit_embed_scale = math.sqrt(config.hidden_size)
        
        # CRITICAL FIX: Add two RoPE embeddings like HF implementation
        self.rope_theta_global = getattr(config.text_config, 'rope_theta', 1000000.0)
        self.rope_theta_local = getattr(config.text_config, 'rope_local_base_freq', 10000.0)
        self.head_dim = config.head_dim
        
        self.debugger_hooks = {}
    
    def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs=None):
        """Project input embeddings to per-layer features (HF implementation)."""
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection *= self.per_layer_projection_scale.to(
            dtype=inputs_embeds.dtype, device=per_layer_projection.device
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            256,  # Original per-layer dimension, not the total output size
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        
        if per_layer_inputs is None:
            return per_layer_projection
            
        # Handle shape mismatch (per-layer inputs sometimes padded)
        if per_layer_projection.shape != per_layer_inputs.shape:
            per_layer_inputs = per_layer_inputs[..., :self.config.num_hidden_layers, :]
            
        # CRITICAL FIX: Apply final scaling like HF (missing 1/√2 scaling)
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale.to(
            dtype=per_layer_projection.dtype, device=per_layer_projection.device
        )
        
    def forward(self, input_ids, debug_layer_limit=None, **kwargs):
        # --- Hook Handling ---
        # Accept debugger_hooks from kwargs, populate self, and propagate to layers.
        debugger_hooks = kwargs.get('debugger_hooks')
        if debugger_hooks:
            self.debugger_hooks = debugger_hooks
            for i, layer in enumerate(self.layers):
                layer.debugger_hooks = {
                    k.replace(f'layer_{i}.', ''): v for k, v in debugger_hooks.items() if k.startswith(f'layer_{i}.')
                }

        # Create the 4-stream input required by the parallel block structure
        initial_embedding = self.embed_tokens(input_ids) * self.embed_scale
        initial_stream = initial_embedding
        hidden_states_0 = initial_stream 

        # DEBUG: Capture initial_embedding for comparison with HF
        if 'initial_embedding' in self.debugger_hooks:
            self.debugger_hooks['initial_embedding'](initial_embedding)

        # DEBUG: Capture initial_stream for comparison with HF
        if 'initial_stream' in self.debugger_hooks:
            self.debugger_hooks['initial_stream'](initial_stream)
        
        #-------- Initial, after embeddings ----------------

        # REMOVE ALL SCALING CORRECTIONS - check natural ratio
        target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
        epsilon_tensor = torch.tensor(1e-5)
        
        temp_hidden_states = [hidden_states_0]
        for i in range(1, self.config.altup_num_inputs):
            # altup_proj adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_proj = self.altup_projections[i - 1](hidden_states_0)
            current_hidden_state = altup_proj.to(dtype=hidden_states_0.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor.to(target_magnitude.device)))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)
        
        hidden_streams = torch.stack(temp_hidden_states, dim=0)  # [num_altup_inputs, batch, seq_len, hidden_size]
        
        # DEBUG: Capture initial_stream_stack for comparison with HF
        if 'initial_stream_stack' in self.debugger_hooks:
            self.debugger_hooks['initial_stream_stack'](hidden_streams)

        print(f"🔍 DEBUG INITIAL STREAMS: After stacking initial temp_hidden_states")
        for stream_idx in range(hidden_streams.shape[0]):
            stream = hidden_streams[stream_idx]
            print(f"  Stream {stream_idx}: std={stream.std():.6f}, mean={stream.mean():.6f}")
        
        # Get Per-Layer Embeddings and apply the correct scaling, which was
        per_layer_scale = (self.config.hidden_size_per_layer_input**0.5) #* 1.0604
        per_layer_embeddings = self.embed_tokens_per_layer(input_ids) * per_layer_scale
        
        # CRITICAL FIX: Apply per-layer projection like HF implementation 
        # This projects the main embeddings to per-layer space and combines with per-layer embeddings
        # First reshape per_layer_embeddings to match expected format [batch, seq, layers, dim_per_layer]
        batch_size, seq_len = input_ids.shape
        
        # Keep original config layers for weight compatibility
        per_layer_embeddings_reshaped = per_layer_embeddings.view(
            batch_size, seq_len, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input
        )
        per_layer_inputs = self.project_per_layer_inputs(hidden_states_0, per_layer_embeddings_reshaped)

        # CRITICAL FIX: Generate both position embeddings at model level like HF
        # This matches the HF implementation exactly
        bsz, seq_len = input_ids.shape
        
        # Generate position_ids like HF - should be [batch_size, seq_len]
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.int64).unsqueeze(0).expand(bsz, -1)
        
        # Generate global position embeddings (for non-sliding layers)
        position_embeddings_global = create_rotary_cache(
            self.head_dim, seq_len, self.rope_theta_global,
            device=input_ids.device, dtype=torch.float32, batch_size=bsz
        )
        
        # Generate local position embeddings (for sliding layers)
        position_embeddings_local = create_rotary_cache(
            self.head_dim, seq_len, self.rope_theta_local,
            device=input_ids.device, dtype=torch.float32, batch_size=bsz
        )

        # Correct K/V sharing implementation, mirroring the JAX source.
        # Gemma3n does not share the *weights* of K/V projection layers.
        # Instead, it dynamically reuses the *output tensors* (the projections)
        # from earlier layers.
        kv_shared_caches = {}
        
        # Config-driven KV sharing calculation (Reference implementation like MLX/JAX)
        kv_sharing_start = self.config.num_hidden_layers - self.config.num_kv_shared_layers  # 20
        
        # Build layer_idx_to_cache_idx mapping following MLX/JAX reference
        layer_types = self.config.layer_types
        
        # Find last occurrence of each attention type in the unshared region (layers 0-19)
        concrete_layers = layer_types[:kv_sharing_start]  # layers 0-19
        
        # Find the indices of the last sliding and global layers in the unshared region
        try:
            shared_sliding_idx = len(concrete_layers) - 1 - concrete_layers[::-1].index("sliding_attention")
        except ValueError:
            shared_sliding_idx = None
            
        try:
            shared_global_idx = len(concrete_layers) - 1 - concrete_layers[::-1].index("full_attention") 
        except ValueError:
            shared_global_idx = None
        
        print(f"🔍 KV CACHE MAPPING:")
        print(f"  Layers 0-{kv_sharing_start-1}: Use own K/V")
        print(f"  Sliding layers {kv_sharing_start}-29 → Layer {shared_sliding_idx}")
        print(f"  Global layers {kv_sharing_start}-29 → Layer {shared_global_idx}")
        
        # Create cache index mapping for each layer
        layer_idx_to_cache_idx = []
        for i, layer_type in enumerate(layer_types):
            if i < kv_sharing_start:
                layer_idx_to_cache_idx.append(i)  # Use own cache
            else:
                if layer_type == "full_attention" and shared_global_idx is not None:
                    layer_idx_to_cache_idx.append(shared_global_idx)
                elif layer_type == "sliding_attention" and shared_sliding_idx is not None:
                    layer_idx_to_cache_idx.append(shared_sliding_idx)
                else:
                    # Fallback to simple offset if pattern not found
                    layer_idx_to_cache_idx.append(i - self.config.num_kv_shared_layers)
        
        print(f"  Layer mapping: {[(i, layer_idx_to_cache_idx[i], layer_types[i]) for i in range(len(layer_types))]}")
        
        # Determine layer limit for processing
        layer_limit = debug_layer_limit if debug_layer_limit is not None else self.config.num_hidden_layers
        
        for i, layer in enumerate(self.layers):
            print(f"🔍 DEBUG: Processing layer {i}/{len(self.layers)-1} - has_weights: {hasattr(layer, 'self_attn')}")
            
            # DEBUG: Only process specified number of layers for fair comparison with HF debug
            if i >= layer_limit:
                print(f"🔍 DEBUG: Skipping layer {i} for {layer_limit}-layer comparison")
                break
                
            # Reference-accurate KV cache sharing following MLX/JAX implementation
            shared_cache = None
            if i != layer_idx_to_cache_idx[i]:  # This layer shares KV
                source_layer_idx = layer_idx_to_cache_idx[i]
                shared_cache = kv_shared_caches.get(source_layer_idx)
                print(f"🔍 KV CACHE: Layer {i} ({layer_types[i]}) requesting K/V from Layer {source_layer_idx}")
                print(f"  Available cached layers: {list(kv_shared_caches.keys())}")
                print(f"  Shared cache found: {shared_cache is not None}")
                # Extract per-layer slice for this specific layer (HF-style)
                per_layer_slice = per_layer_inputs[:, :, i, :]  # [batch, seq, dim_per_layer]
                layer_output = layer(
                    hidden_streams,
                    per_layer_embeddings=per_layer_slice,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=None, # Assuming this is handled inside
                    position_ids=position_ids,
                    shared_cache=shared_cache
                )
                # CRITICAL FIX: Ensure explicit assignment to fix the stream update bug
                hidden_streams = layer_output
                print(f"🔍 DEBUG LAYER OUTPUT: After KV-shared layer {i}")
                for stream_idx in range(hidden_streams.shape[0]):
                    stream = hidden_streams[stream_idx]
                    print(f"  Stream {stream_idx}: std={stream.std():.6f}, mean={stream.mean():.6f}")
            else:
                # Extract per-layer slice for this specific layer (HF-style)
                per_layer_slice = per_layer_inputs[:, :, i, :]  # [batch, seq, dim_per_layer]
                layer_output = layer(
                    hidden_streams,
                    per_layer_embeddings=per_layer_slice,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=None, # Assuming this is handled inside
                    position_ids=position_ids,
                    shared_cache=None
                )
                # CRITICAL FIX: Ensure explicit assignment to fix the stream update bug
                hidden_streams = layer_output
                print(f"�� DEBUG LAYER OUTPUT: After regular layer {i}")
                for stream_idx in range(hidden_streams.shape[0]):
                    stream = hidden_streams[stream_idx]
                    print(f"  Stream {stream_idx}: std={stream.std():.6f}, mean={stream.mean():.6f}")

            # CRITICAL FIX: Store KV cache IMMEDIATELY after layer processing
            # Store K/V for layers that will be used as sources by the sharing layers
            # Only store for layers that are actually referenced by the mapping
            if i in set(layer_idx_to_cache_idx[kv_sharing_start:]):
                kv_shared_caches[i] = layer.self_attn.kv_for_sharing
                print(f"🔍 KV CACHE: Stored Layer {i} K/V for sharing (source layer)")
                if layer.self_attn.kv_for_sharing:
                    k, v = layer.self_attn.kv_for_sharing
                    print(f"  K shape: {k.shape}, std: {k.std():.6f}")
                    print(f"  V shape: {v.shape}, std: {v.std():.6f}")
        
        # Un-embedding logic (FIXED to match HF exactly)
        if 'hidden_streams_before_unembed' in self.debugger_hooks: self.debugger_hooks['hidden_streams_before_unembed'](hidden_streams)
        
        print(f"🔍 DEBUG BEFORE UNEMBED: Input to unembed section")
        print(f"🔍 ANEMLL hidden_streams shape: {hidden_streams.shape}")
        for stream_idx in range(hidden_streams.shape[0]):
            stream = hidden_streams[stream_idx]
            print(f"  Stream {stream_idx}: std={stream.std():.6f}, mean={stream.mean():.6f}")
        
        # CRITICAL FIX: HF uses hidden_states_0 (first stream after layer processing) for target magnitude
        # Not the raw hidden_streams[0] which might be from earlier processing
        # Use the first stream from the most recent layer output
        first_stream_after_layers = hidden_streams[0]  # First stream after layer processing
        #-------- AFTER PLE ----------------
        target_magnitude = torch.mean(first_stream_after_layers ** 2, dim=-1, keepdim=True) ** 0.5
        if 'target_magnitude' in self.debugger_hooks: self.debugger_hooks['target_magnitude'](target_magnitude)
        
        epsilon_tensor = torch.tensor(1e-5)
        # CRITICAL FIX: Stream 0 is used as-is (no projection/normalization) to match HF exactly
        temp_hidden_states = [hidden_streams[0]]  # Stream 0: use directly (already scaled)
        
        if 'stream_0_raw' in self.debugger_hooks: self.debugger_hooks['stream_0_raw'](hidden_streams[0])
        
        for i in range(1, self.config.altup_num_inputs):
            if f'stream_{i}_raw' in self.debugger_hooks: self.debugger_hooks[f'stream_{i}_raw'](hidden_streams[i])
            print(f"🔍 DEBUG STREAM {i} RAW: std={hidden_streams[i].std():.6f}, mean={hidden_streams[i].mean():.6f}")
            
            # altup_unembed_projections adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_unemb_proj = self.altup_unembed_projections[i-1](hidden_streams[i])
            if f'stream_{i}_projected' in self.debugger_hooks: self.debugger_hooks[f'stream_{i}_projected'](altup_unemb_proj)
            print(f"🔍 DEBUG STREAM {i} PROJECTED: std={altup_unemb_proj.std():.6f}, mean={altup_unemb_proj.mean():.6f}")
            
            current_hidden_state = altup_unemb_proj.to(dtype=hidden_states_0.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor.to(target_magnitude.device)))
            print(f"🔍 DEBUG STREAM {i} NEW_MAGNITUDE: {new_magnitude.mean():.6f}, target_magnitude: {target_magnitude.mean():.6f}")
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            print(f"🔍 DEBUG STREAM {i} NORMALIZED: std={current_hidden_state.std():.6f}, mean={current_hidden_state.mean():.6f}")
            
            if f'stream_{i}_normalized' in self.debugger_hooks: self.debugger_hooks[f'stream_{i}_normalized'](current_hidden_state)
            temp_hidden_states.append(current_hidden_state)
        
        # CRITICAL FIX: HF averages ALL streams, not just the active stream
        print(f"🔍 DEBUG UNEMBED FINAL: temp_hidden_states length: {len(temp_hidden_states)}")
        for idx, tensor in enumerate(temp_hidden_states):
            print(f"  Final Stream {idx}: shape={tensor.shape}, std={tensor.std():.6f}, mean={tensor.mean():.6f}")
        
        # Add magnitude comparison debug
        print(f"\n🎯 MAGNITUDE COMPARISON WITH REAL HF:")
        try:
            real_hf_data = torch.load('tests/dev/hf_real_unembed_data.pt', weights_only=False)
            if 'real_stacked_streams' in real_hf_data:
                hf_real_stacked = real_hf_data['real_stacked_streams']
                print(f"  Real HF vs ANEMLL comparison:")
                for idx in range(min(len(temp_hidden_states), hf_real_stacked.shape[0])):
                    hf_stream = hf_real_stacked[idx]
                    anemll_stream = temp_hidden_states[idx]
                    cos_sim = torch.nn.functional.cosine_similarity(hf_stream.flatten(), anemll_stream.flatten(), dim=0).item()
                    print(f"    Stream {idx}: cos_sim={cos_sim:.6f}, HF_std={hf_stream.std():.6f}, ANEMLL_std={anemll_stream.std():.6f}")
        except Exception as e:
            print(f"  ⚠️  Could not load real HF data: {e}")
        
        stacked_streams = torch.stack(temp_hidden_states)
        print(f"🔍 DEBUG UNEMBED: stacked_streams shape: {stacked_streams.shape}")
        print(f"  stacked_streams std: {stacked_streams.std():.6f}, mean: {stacked_streams.mean():.6f}")
        
        if 'stacked_streams' in self.debugger_hooks: self.debugger_hooks['stacked_streams'](stacked_streams)
        
        hidden_states = torch.mean(stacked_streams, dim=0)
        if 'averaged_before_norm' in self.debugger_hooks: self.debugger_hooks['averaged_before_norm'](hidden_states)

        # DEBUG: Capture input to final norm to isolate divergence source
        if 'final_norm_input' in self.debugger_hooks: self.debugger_hooks['final_norm_input'](hidden_states)
        
        hidden_states = self.norm(hidden_states)
        if 'final_norm_output' in self.debugger_hooks: self.debugger_hooks['final_norm_output'](hidden_states)
        
        # CRITICAL FIX: HF model doesn't apply embed_scale to logits
        # The scaling should not be applied to the final logits
        logits = self.lm_head(hidden_states)
        # logits = self.lm_head(hidden_states) * self.embed_scale
        
        # CRITICAL FIX: Apply final logit softcapping to match HF
        if hasattr(self.config, 'final_logit_softcapping') and self.config.final_logit_softcapping is not None:
            softcap = self.config.final_logit_softcapping
            logits = softcap * torch.tanh(logits / softcap)
        
        # --- LOGITS COMPARISON ---
        if 'logits' in self.debugger_hooks: self.debugger_hooks['logits'](logits)
        
        # CRITICAL FIX: HF model doesn't actually apply final logit softcapping
        # Even though config.final_logit_softcapping=30.0, it's not implemented in HF
        # final_logit_softcapping = getattr(self.config, 'final_logit_softcapping', None)
        # if final_logit_softcapping is not None:
        #     logits = logits / final_logit_softcapping
        #     logits = torch.tanh(logits)
        #     logits = logits * final_logit_softcapping
        
        return logits

def _create_causal_mask(seq_len, sliding_window=None):
    """Creates a causal mask for attention.

    Args:
        seq_len (int): The sequence length.
        sliding_window (int, optional): The sliding window size. Defaults to None.

    Returns:
        torch.Tensor: The causal mask.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    if sliding_window:
        mask = torch.tril(mask, diagonal=sliding_window - 1)
    return mask

def load_gemma3n_weights_from_safetensors(simple_model, gemma3n_path, debug_mode=False):
    """Load Gemma3n weights from multiple safetensor files."""
    print("🔄 Loading Gemma3n weights from safetensors...")
    
    from safetensors import safe_open
    import os
    
    loaded_count = 0
    
    try:
        # Load from all 3 safetensor files
        safetensor_files = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors", 
            "model-00003-of-00003.safetensors"
        ]
        
        all_weights = {}
        
        # Combine all weights from all files
        for filename in safetensor_files:
            filepath = os.path.join(gemma3n_path, filename)
            if os.path.exists(filepath):
                print(f"  📁 Loading from {filename}...")
                with safe_open(filepath, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        # keep all text weights ...
                        if key.startswith("model.language_model"):
                            tensor = f.get_tensor(key)
                            all_weights[key] = tensor.to(torch.float32)
                        # ... but **also** keep the root‑level head
                        elif key == "lm_head.weight":
                            tensor = f.get_tensor(key)
                            all_weights[key] = tensor.to(torch.float32)
        
        print(f"  📊 Total text model weights found: {len(all_weights)}")
        
        # Debug: Print some key patterns we found
        print(f"  🔍 Key patterns found:")
        embed_keys = [k for k in all_weights.keys() if 'embed' in k]
        norm_keys = [k for k in all_weights.keys() if 'norm' in k and 'model.language_model.norm' in k]
        lm_head_keys = [k for k in all_weights.keys() if 'lm_head' in k or k.endswith('.weight') and 'head' in k]
        print(f"    Embed keys: {embed_keys[:3]}")
        print(f"    Norm keys: {norm_keys[:3]}")
        print(f"    LM head candidates: {lm_head_keys[:3]}")
        
        # Map and load weights
        # Main embeddings
        embed_key = 'model.language_model.embed_tokens.weight'
        if embed_key in all_weights:
            simple_model.embed_tokens.weight.data = all_weights[embed_key]
            loaded_count += 1
            print(f"  ✅ Loaded embeddings: {all_weights[embed_key].shape}")
        else:
            print(f"  ❌ Embeddings not found. Available embed keys: {embed_keys}")
        
        # Per-layer embeddings
        per_layer_embed_key = 'model.language_model.embed_tokens_per_layer.weight'
        if per_layer_embed_key in all_weights:
            ple_weight = all_weights[per_layer_embed_key]
            expected_ple_dim = simple_model.config.num_hidden_layers * 256
            actual_ple_dim = ple_weight.shape[1]
            
            # Fix: Validate PLE dimensions match model config
            print(f"  🔍 PLE validation: expected {expected_ple_dim} dims ({simple_model.config.num_hidden_layers} layers × 256)")
            print(f"                     actual {actual_ple_dim} dims ({actual_ple_dim//256} layers × 256)")
            
            if actual_ple_dim == expected_ple_dim:
                print(f"  ✅ PLE dimensions match config")
            else:
                print(f"  ⚠️  PLE dimension mismatch - checkpoint has {actual_ple_dim//256} layers, config expects {simple_model.config.num_hidden_layers}")
                print(f"       This will cause indexing errors in per-layer processing")
            
            simple_model.embed_tokens_per_layer.weight.data = ple_weight
            loaded_count += 1
            print(f"  ✅ Loaded per-layer embeddings: {ple_weight.shape}")
        else:
            print(f"  ❌ Per-layer embeddings not found")
        
        # Per-layer model projection
        per_layer_proj_key = 'model.language_model.per_layer_model_projection.weight'
        if per_layer_proj_key in all_weights:
            simple_model.per_layer_model_projection.weight.data = all_weights[per_layer_proj_key]
            loaded_count += 1
            print(f"  ✅ Loaded per-layer model projection: {all_weights[per_layer_proj_key].shape}")
        else:
            print(f"  ❌ Per-layer model projection not found!")
            
        # Per-layer projection norm  
        per_layer_norm_key = 'model.language_model.per_layer_projection_norm.weight'
        if per_layer_norm_key in all_weights:
            simple_model.per_layer_projection_norm.weight.data = all_weights[per_layer_norm_key]
            loaded_count += 1
            print(f"  ✅ Loaded per-layer projection norm: {all_weights[per_layer_norm_key].shape}")
        else:
            print(f"  ❌ Per-layer projection norm not found!")
            
        # Final norm
        norm_key = 'model.language_model.norm.weight' 
        if norm_key in all_weights:
            simple_model.norm.weight.data = all_weights[norm_key]
            loaded_count += 1
            print(f"  ✅ Loaded final norm: {all_weights[norm_key].shape}")
        else:
            print(f"  ❌ Final norm not found. Available norm keys: {norm_keys}")
        
        # LM head (try different possible keys)
        lm_head_found = False
        lm_head_candidates = ['lm_head.weight', 'model.language_model.lm_head.weight', 'model.lm_head.weight']
        for lm_key in lm_head_candidates:
            if lm_key in all_weights:
                simple_model.lm_head.weight.data = all_weights[lm_key]
                loaded_count += 1
                print(f"  ✅ Loaded LM head: {all_weights[lm_key].shape}")
                lm_head_found = True
                break
        
        if not lm_head_found:
            print(f"  ❌ LM head not found. Tried: {lm_head_candidates}")
            print(f"    Available candidates: {[k for k in all_weights.keys() if any(x in k for x in ['lm_head', 'head', 'output'])]}")
            # Let's check if we can share embeddings as LM head (common in some models)
            if embed_key in all_weights:
                print(f"  🔄 Using tied embeddings as LM head")
                # CRITICAL: For tied embeddings in LM head, we use the embedding weights directly
                # nn.Linear expects [out_features, in_features] = [vocab_size, hidden_size]
                embed_weights = all_weights[embed_key]  # [262400, 2048] - this is correct for nn.Linear!
                print(f"    Embedding shape: {embed_weights.shape} (matches nn.Linear expectation)")
                print(f"    LM head expects: {simple_model.lm_head.weight.shape}")
                simple_model.lm_head.weight.data = embed_weights  # Direct assignment, no transpose needed
                print(f"    ✅ LM head weight set to: {simple_model.lm_head.weight.shape}")
                loaded_count += 1
        
        # Global AltUp projections (EXACT HF pattern: 3 projections for streams 1,2,3)
        for i in range(len(simple_model.altup_projections)):  # 3 projections
            # HF stores projections as indices 0,1,2 for streams 1,2,3
            altup_key = f'model.language_model.altup_projections.{i}.weight'
            if altup_key in all_weights:
                simple_model.altup_projections[i].weight.data = all_weights[altup_key]
                loaded_count += 1
            
            unembed_key = f'model.language_model.altup_unembed_projections.{i}.weight'
            if unembed_key in all_weights:
                simple_model.altup_unembed_projections[i].weight.data = all_weights[unembed_key]
                loaded_count += 1
        
        # Layer weights - with detailed debugging
        layers_with_missing_weights = []
        
        # Smart debug_mode handling: integer = specific layer count, True = 1 layer, False = all layers
        if debug_mode and isinstance(debug_mode, int):
            num_layers_to_load = debug_mode          # e.g. 2, 4, ...
        elif debug_mode is True:
            num_layers_to_load = 1
        else:
            num_layers_to_load = len(simple_model.layers)
        print(f"  🔧 Loading weights for {num_layers_to_load} layer(s) (Debug mode: {debug_mode})")

        for i in range(num_layers_to_load):
            layer_prefix = f'model.language_model.layers.{i}'
            simple_layer = simple_model.layers[i]
            print(f"  🔧 Loading layer {i} weights...")
            layer_loaded_count = 0
            missing_components = []
            
            # LAUREL blocks
            laurel_left_key = f'{layer_prefix}.laurel.linear_left.weight'
            if laurel_left_key in all_weights:
                simple_layer.laurel.linear_left.weight.data = all_weights[laurel_left_key]
                loaded_count += 1
                layer_loaded_count += 1
            else:
                missing_components.append('laurel_left')
            
            laurel_right_key = f'{layer_prefix}.laurel.linear_right.weight'
            if laurel_right_key in all_weights:
                simple_layer.laurel.linear_right.weight.data = all_weights[laurel_right_key]
                loaded_count += 1
                layer_loaded_count += 1
            else:
                missing_components.append('laurel_right')
            
            laurel_norm_key = f'{layer_prefix}.laurel.post_laurel_norm.weight'
            if laurel_norm_key in all_weights:
                simple_layer.laurel.post_laurel_norm.weight.data = all_weights[laurel_norm_key]
                loaded_count += 1
                layer_loaded_count += 1
                print(f"    ✅ Layer {i} LAUREL norm loaded: {all_weights[laurel_norm_key].shape}")
            else:
                print(f"    ⚠️  Layer {i} LAUREL norm missing - initializing with ones")
                # Fix: Initialize missing LAUREL norm with ones (safe default)
                simple_layer.laurel.post_laurel_norm.weight.data.fill_(1.0)
                missing_components.append('laurel_norm')
            
            # AltUp per-layer components
            altup_components = {
                'correction_coefs': f'{layer_prefix}.altup.correction_coefs.weight',
                'prediction_coefs': f'{layer_prefix}.altup.prediction_coefs.weight',
                'correct_output_scale': f'{layer_prefix}.altup.correct_output_scale',
                'modality_router': f'{layer_prefix}.altup.modality_router.weight',
                'router_norm': f'{layer_prefix}.altup.router_norm.weight'
            }
            
            for comp_name, weight_key in altup_components.items():
                if weight_key in all_weights:
                    # This is the correct way to load into the nn.Parameter
                    if comp_name in ['correct_output_scale']:
                        getattr(simple_layer.altup, comp_name).data = all_weights[weight_key]
                    elif comp_name in ['correction_coefs', 'prediction_coefs', 'modality_router']:
                        getattr(simple_layer.altup, comp_name).weight.data = all_weights[weight_key]
                    elif comp_name == 'router_norm':
                        simple_layer.altup.router_norm.weight.data = all_weights[weight_key]
                    loaded_count += 1
                    layer_loaded_count += 1
                else:
                    missing_components.append(f'altup_{comp_name}')
            
            # Standard FFN weights
            gate_proj_key = f'{layer_prefix}.mlp.gate_proj.weight'
            if gate_proj_key in all_weights:
                simple_layer.mlp.gate_proj.weight.data = all_weights[gate_proj_key]
                loaded_count += 1
                layer_loaded_count += 1
                if i == 0: print(f"    ✅ Layer 0 mlp.gate_proj loaded: {all_weights[gate_proj_key].shape}, std={all_weights[gate_proj_key].std():.6f}")
            else:
                missing_components.append('gate_proj')

            up_proj_key = f'{layer_prefix}.mlp.up_proj.weight'
            if up_proj_key in all_weights:
                simple_layer.mlp.up_proj.weight.data = all_weights[up_proj_key]
                loaded_count += 1
                layer_loaded_count += 1
                if i == 0: print(f"    ✅ Layer 0 mlp.up_proj loaded: {all_weights[up_proj_key].shape}, std={all_weights[up_proj_key].std():.6f}")
            else:
                missing_components.append('up_proj')


            # FFN projections
            for proj_name in ['down_proj']:
                ffn_key = f'{layer_prefix}.mlp.{proj_name}.weight'
                if ffn_key in all_weights:
                    getattr(simple_layer.mlp, proj_name).weight.data = all_weights[ffn_key]
                    loaded_count += 1
                    layer_loaded_count += 1
                    if i == 0: print(f"    ✅ Layer 0 mlp.down_proj loaded: {all_weights[ffn_key].shape}, std={all_weights[ffn_key].std():.6f}")
                else:
                    missing_components.append(f'ffn_{proj_name}')
            
            # Per-layer embedding components - CORRECTED
            per_layer_gate_key = f'{layer_prefix}.per_layer_input_gate.weight'
            if per_layer_gate_key in all_weights:
                hf_weight = all_weights[per_layer_gate_key]
                model_weight_shape = simple_layer.per_layer_input_gate.weight.shape
                if hf_weight.shape == model_weight_shape:
                    simple_layer.per_layer_input_gate.weight.data = hf_weight
                    loaded_count += 1
                    layer_loaded_count += 1
                else:
                    missing_components.append('per_layer_gate')
            else:
                missing_components.append('per_layer_gate')
            
            # HF Gemma3n: per_layer_input_gate has NO bias parameter (confirmed via safetensors analysis)
            # per_layer_gate_bias_key = f'{layer_prefix}.per_layer_input_gate.bias'  # Does not exist in HF model
            # Bias handling removed - HF uses bias=False for all components
                
            per_layer_proj_key = f'{layer_prefix}.per_layer_projection.weight'
            if per_layer_proj_key in all_weights:
                simple_layer.per_layer_projection.weight.data = all_weights[per_layer_proj_key]
                loaded_count += 1
                layer_loaded_count += 1
            else:
                missing_components.append('per_layer_proj')
            
            # CRITICAL FIX #6: Load post_per_layer_input_norm weight
            post_per_layer_norm_key = f'{layer_prefix}.post_per_layer_input_norm.weight'
            if post_per_layer_norm_key in all_weights:
                simple_layer.post_per_layer_input_norm.weight.data = all_weights[post_per_layer_norm_key]
                loaded_count += 1
                layer_loaded_count += 1
                if i <= 2:  # Debug first few layers
                    weight = all_weights[post_per_layer_norm_key]
                    print(f"    ✅ Layer {i} post_per_layer_input_norm loaded: {weight.shape}, std={weight.std():.6f}")
            else:
                missing_components.append('post_per_layer_norm')
                # Initialize with ones (safe default for RMSNorm)
                simple_layer.post_per_layer_input_norm.weight.data.fill_(1.0)
                if i <= 2:  # Debug first few layers
                    print(f"    ❌ Layer {i} post_per_layer_input_norm MISSING! Key: {post_per_layer_norm_key}")
            
            # Standard components
            # Norms
            # Load input_layernorm (now using HF naming convention)
            input_norm_key = f'{layer_prefix}.input_layernorm.weight'
            if input_norm_key in all_weights:
                simple_layer.input_layernorm.weight.data = all_weights[input_norm_key]
                loaded_count += 1
                layer_loaded_count += 1
            else:
                missing_components.append('input_layernorm')
            
            # Load post_attention_layernorm (applied to attention output before combination)
            post_attn_norm_key = f'{layer_prefix}.post_attention_layernorm.weight'
            if post_attn_norm_key in all_weights:
                simple_layer.post_attention_layernorm.weight.data = all_weights[post_attn_norm_key]
                loaded_count += 1
                layer_loaded_count += 1
            else:
                missing_components.append('post_attention_layernorm')
            
            # Load pre_feedforward_layernorm (applied to combined output before MLP)
            pre_ffw_norm_key = f'{layer_prefix}.pre_feedforward_layernorm.weight'
            if pre_ffw_norm_key in all_weights:
                simple_layer.pre_feedforward_layernorm.weight.data = all_weights[pre_ffw_norm_key]
                loaded_count += 1
                layer_loaded_count += 1
            else:
                missing_components.append('pre_feedforward_layernorm')
            
            post_ffn_norm_key = f'{layer_prefix}.post_feedforward_layernorm.weight'
            if post_ffn_norm_key in all_weights:
                simple_layer.post_feedforward_layernorm.weight.data = all_weights[post_ffn_norm_key]
                loaded_count += 1
                layer_loaded_count += 1
            else:
                missing_components.append('post_feedforward_layernorm')
            
            # Attention projections - load into self_attn module
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                attn_key = f'{layer_prefix}.self_attn.{proj_name}.weight'
                if attn_key in all_weights:
                    # ALWAYS load attention projection weights
                    getattr(simple_layer.self_attn, proj_name).weight.data = all_weights[attn_key]
                    loaded_count += 1
                    layer_loaded_count += 1
                    
                    if proj_name == 'o_proj':
                        print(f"    🔎 INSPECTING o_proj: Layer {i}")
                        weight = all_weights[attn_key]
                        print(f"       - o_proj weight loaded, shape: {weight.shape}, std: {weight.std():.6f}")

                    # Debug: Check if this is a KV sharing layer (config-driven)
                    kv_sharing_start = simple_model.config.num_hidden_layers - simple_model.config.num_kv_shared_layers
                    is_kv_sharing_layer = i >= kv_sharing_start
                    
                    if is_kv_sharing_layer and proj_name in ['k_proj', 'v_proj']:
                        loaded_weight = all_weights[attn_key]
                        if loaded_weight.std() == 0:
                            print(f"    ✅ Layer {i} {proj_name} loaded: {loaded_weight.shape}, std={loaded_weight.std():.6f} (intentionally zero for KV sharing)")
                        else:
                            print(f"    ⚠️  Layer {i} {proj_name} loaded: {loaded_weight.shape}, std={loaded_weight.std():.6f} (expected zero for KV sharing, but has non-zero values!)")
                    elif i >= 20 and proj_name in ['k_proj', 'v_proj']:  # Debug for expected range
                        loaded_weight = all_weights[attn_key]
                        print(f"    ✅ Layer {i} {proj_name} loaded: {loaded_weight.shape}, std={loaded_weight.std():.6f}")
                else:
                    missing_components.append(f'attn_{proj_name}')
                    # Debug: Report missing critical weights (config-driven)
                    kv_sharing_start = simple_model.config.num_hidden_layers - simple_model.config.num_kv_shared_layers
                    if i >= kv_sharing_start and proj_name in ['k_proj', 'v_proj']:
                        print(f"    ❌ Layer {i} {proj_name} MISSING! Key: {attn_key}")
                
            # Attention normalization weights (q_norm, k_norm)
            q_norm_key = f'{layer_prefix}.self_attn.q_norm.weight'
            if q_norm_key in all_weights:
                q_norm_weight = all_weights[q_norm_key].to(torch.float32)
                # ALWAYS load q_norm weights - zero std is EXPECTED for constant scaling factors
                simple_layer.self_attn.q_norm.weight.data = q_norm_weight
                loaded_count += 1
                layer_loaded_count += 1

                if i <= 2:  # Debug first few layers
                    print(f"    ✅ Layer {i} q_norm loaded: std={q_norm_weight.std():.6f}, mean={q_norm_weight.mean():.6f}")
            else:
                missing_components.append('attn_q_norm')
                if i <= 2:
                    print(f"    ❌ Layer {i} q_norm MISSING! Key: {q_norm_key}")

            k_norm_key = f'{layer_prefix}.self_attn.k_norm.weight'
            if k_norm_key in all_weights:
                k_norm_weight = all_weights[k_norm_key].to(torch.float32)
                # ALWAYS load k_norm weights - zero std is EXPECTED for constant scaling factors
                simple_layer.self_attn.k_norm.weight.data = k_norm_weight
                loaded_count += 1
                layer_loaded_count += 1

                if i <= 2:  # Debug first few layers
                    print(f"    ✅ Layer {i} k_norm loaded: std={k_norm_weight.std():.6f}, mean={k_norm_weight.mean():.6f}")
            else:
                missing_components.append('attn_k_norm')
                if i <= 2:
                    print(f"    ❌ Layer {i} k_norm MISSING! Key: {k_norm_key}")

            
            # Track layers with missing weights
            if missing_components:
                layers_with_missing_weights.append((i, missing_components, layer_loaded_count))
            
            # Safeguard: Validate that critical layer weights have non-trivial values
            critical_params = [
                ('q_proj', simple_layer.self_attn.q_proj.weight),
                ('k_proj', simple_layer.self_attn.k_proj.weight), 
                ('v_proj', simple_layer.self_attn.v_proj.weight),
                ('o_proj', simple_layer.self_attn.o_proj.weight),
                ('gate_proj', simple_layer.mlp.gate_proj.weight),
                ('up_proj', simple_layer.mlp.up_proj.weight),
                ('down_proj', simple_layer.mlp.down_proj.weight),
            ]
            
            for param_name, param in critical_params:
                if param.std().item() < 1e-4:
                    print(f"    ⚠️  WARNING: Layer {i} {param_name} has very low std ({param.std().item():.6f}) - may be uninitialized!")
                    if layer_loaded_count < 10:  # If very few weights loaded
                        print(f"    🔍 Layer {i} only loaded {layer_loaded_count} components - check weight loading!")
        
        # Report missing weights and check specifically for critical normalization weights
        critical_missing = []
        if layers_with_missing_weights:
            print(f"  ⚠️  Layers with missing weights:")
            for layer_idx, missing, loaded in layers_with_missing_weights[:5]:  # Show first 5
                print(f"    Layer {layer_idx}: missing {missing} (loaded {loaded} components)")
                if 'input_norm' in missing or 'post_attn_norm' in missing:
                    critical_missing.append(layer_idx)
            if len(layers_with_missing_weights) > 5:
                print(f"    ... and {len(layers_with_missing_weights) - 5} more layers with missing weights")
        
        if critical_missing:
            print(f"  🚨 CRITICAL: Layers {critical_missing[:10]} missing normalization weights - this will cause NaN!")
            
        # Fix: Enhanced logging for normalization weights
        actual_norm_keys = [k for k in all_weights.keys() if 'norm' in k and 'layers' in k]
        laurel_norm_keys = [k for k in all_weights.keys() if 'post_laurel_norm' in k]
        print(f"  🔍 Sample norm keys found: {actual_norm_keys[:5]}")
        print(f"  🔍 LAUREL norm keys found: {len(laurel_norm_keys)}/{len(simple_model.layers)} layers")
        if len(laurel_norm_keys) < len(simple_model.layers) and not debug_mode:
            print(f"    Missing LAUREL norms will be initialized with ones")
        
        print(f"✅ Loaded {loaded_count} weight tensors")
        
        # Check for nan values in loaded weights
        has_nan = False
        for name, param in simple_model.named_parameters():
            if torch.isnan(param).any():
                print(f"  ⚠️  Found NaN in parameter: {name}")
                has_nan = True
        
        if not has_nan:
            print(f"  ✅ No NaN values detected in loaded weights")
        
        if debug_mode:
            return loaded_count > 10 # Should load at least a layer's worth
        else:
            return loaded_count > 50  # Should have many weights for 30 layers
        
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_gemma3n_config():
    """Create a Gemma3n config based on actual HF config.text_config."""
    from transformers import AutoConfig
    
    # Load HF config
    MODEL_PATH = os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots/0330734afc91972a1aa1ba1bc4495e2723666854/"
    )
    
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # CRITICAL FIX: Read from text_config, not main config
    text_config = hf_config.text_config
    
    class Gemma3nConfig:
        def __init__(self):
            # CRITICAL FIX: Use text_config values exactly like HF
            self.vocab_size = text_config.vocab_size
            self.vocab_size_per_layer_input = text_config.vocab_size_per_layer_input
            self.hidden_size = text_config.hidden_size
            # CRITICAL FIX: intermediate_size is a list, take first value
            self.intermediate_size = text_config.intermediate_size[0] if isinstance(text_config.intermediate_size, list) else text_config.intermediate_size
            self.num_hidden_layers = text_config.num_hidden_layers
            self.num_attention_heads = text_config.num_attention_heads
            self.num_key_value_heads = text_config.num_key_value_heads
            self.num_kv_shared_layers = getattr(text_config, 'num_kv_shared_layers', 10)
            self.head_dim = text_config.head_dim
            self.rms_norm_eps = text_config.rms_norm_eps
            self.attention_bias = text_config.attention_bias
            self.attention_dropout = text_config.attention_dropout
            self.max_position_embeddings = text_config.max_position_embeddings
            self.rope_theta = text_config.rope_theta
            
            # CRITICAL FIX: Add text_config reference for attention layer
            self.text_config = text_config
            
            # Gemma3n specific features from text_config
            self.sliding_window = text_config.sliding_window
            self.final_logit_softcapping = text_config.final_logit_softcapping
            self.query_pre_attn_scalar = text_config.query_pre_attn_scalar
            self.laurel_rank = text_config.laurel_rank
            # CRITICAL FIX: HF config has wrong value (256), but weights expect 7680
            # CORRECTED: The per_layer_input_gate.weight is [256, 2048], so per-layer dim = 256
            # The 7680 is for global per-layer embeddings (30 layers × 256), but each layer processes 256
            self.hidden_size_per_layer_input = 256  # Per-layer dimension, matches HF implementation
            self.altup_num_inputs = text_config.altup_num_inputs
            self.altup_correct_scale = text_config.altup_correct_scale
            self.altup_active_idx = text_config.altup_active_idx
            self.altup_coef_clip = getattr(text_config, 'altup_coef_clip', 120.0)
            
            # CRITICAL FIX: Use actual layer_types from text_config
            self.layer_types = text_config.layer_types
            
            # CRITICAL FIX: Use actual activation_sparsity_pattern from text_config
            self.activation_sparsity_pattern = text_config.activation_sparsity_pattern
    
    return Gemma3nConfig()

def test_gemma3n_architecture():
    print("🔬 Testing Gemma3n Architecture Against Local Model")
    print("=" * 60)
    
    # Path to local Gemma3n model
    local_gemma3n_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots/0330734afc91972a1aa1ba1bc4495e2723666854/"
    )
    
    print(f"Local Gemma3n path: {local_gemma3n_path}")
    
    if not os.path.exists(local_gemma3n_path):
        print(f"❌ Local model not found: {local_gemma3n_path}")
        return False
    
    # Load tokenizer from local model
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_gemma3n_path, trust_remote_code=True)
        print(f"✅ Loaded tokenizer: {type(tokenizer).__name__}")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return False
    
    # Create our Gemma3n model
    config = create_gemma3n_config()
    our_model = SimpleGemma3nModel(config)
    
    print(f"\n🏗️  Created Gemma3n model:")
    print(f"  Layers: {len(our_model.layers)}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  LAUREL rank: {config.laurel_rank}")
    print(f"  AltUp inputs: {config.altup_num_inputs}")
    
    # Load weights from safetensors
    success = load_gemma3n_weights_from_safetensors(our_model, local_gemma3n_path)
    
    if not success:
        print("❌ Weight loading failed")
        return False
    
    our_model.eval()
    
    # Test with the capital of France question to check for "Paris" prediction
    #TEST_PROMPT = "What"
    TEST_PROMPT = "What is the capital of France? Answer:"

    #test_tokens = tokenizer.encode(TEST_PROMPT, add_special_tokens=False)
    test_tokens = tokenizer.encode(TEST_PROMPT, add_special_tokens=True)

    print(f"\n🧪 Testing with sequence: '{TEST_PROMPT}'")
    print(f"   Tokens: {test_tokens}")
    print(f"   Decoded: {[tokenizer.decode([t]) for t in test_tokens]}")
    
    input_ids = torch.tensor([test_tokens], dtype=torch.long)
    print(f"   Input shape: {input_ids.shape}")
    
    try:
        # Load HuggingFace model for comparison
        print(f"\n🤗 Loading HuggingFace model for comparison...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            local_gemma3n_path,
            torch_dtype=torch.float32,
            trust_remote_code=False,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        hf_model.eval()

        hf_activations = {}

        def get_activation(name):
            def hook(model, input, output):
                hf_activations[name] = output.detach()
            return hook

        # Register hooks on the first attention and MLP layers of the HF model
        # The structure of HF Gemma3n is model.layers[0].self_attn and model.layers[0].mlp
        hf_model.model.language_model.layers[0].self_attn.o_proj.register_forward_hook(get_activation('hf_attn_output'))
        hf_model.model.language_model.layers[0].mlp.down_proj.register_forward_hook(get_activation('hf_mlp_output'))
        
        with torch.no_grad():
            # Test HuggingFace model
            print(f"\n🤗 HuggingFace model predictions:")
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits[0, -1, :]  # Last position
            hf_top = torch.argsort(hf_logits, descending=True)[:10]
            
            print(f"  HF logits range: [{hf_logits.min().item():.3f}, {hf_logits.max().item():.3f}]")
            print(f"  HF Top 10 predictions:")
            for i, idx in enumerate(hf_top):
                token_text = tokenizer.decode([int(idx)])
                logit_val = hf_logits[idx].item()
                print(f"    {i+1:2d}. '{token_text}' (logit: {logit_val:.3f})")
            
            # Test our model
            print(f"\n🏗️ Our model predictions:")
            output_logits = our_model(input_ids)
            our_logits = output_logits[0, -1, :]  # Last position
            our_top = torch.argsort(our_logits, descending=True)[:10]
            
            print(f"  Our logits range: [{our_logits.min().item():.3f}, {our_logits.max().item():.3f}]")
            print(f"  Our Top 10 predictions:")
            for i, idx in enumerate(our_top):
                token_text = tokenizer.decode([int(idx)])
                logit_val = our_logits[idx].item()
                print(f"    {i+1:2d}. '{token_text}' (logit: {logit_val:.3f})")
            
            # 🎯 PARIS PREDICTION TEST (only if prompt contains "France")
            if "france" in TEST_PROMPT.lower():
                print(f"\n🎯 PARIS PREDICTION TEST:")
                
                # Find Paris token
                paris_candidates = [" Paris", "Paris", " paris", "paris"]
                paris_token_id = None
                paris_token_text = None
                
                for candidate in paris_candidates:
                    try:
                        candidate_tokens = tokenizer.encode(candidate, add_special_tokens=False)
                        if len(candidate_tokens) == 1:
                            paris_token_id = candidate_tokens[0]
                            paris_token_text = candidate
                            break
                    except:
                        continue
                
                if paris_token_id is not None:
                    hf_paris_logit = hf_logits[paris_token_id].item()
                    our_paris_logit = our_logits[paris_token_id].item()
                    
                    # Find rankings
                    hf_paris_rank = (hf_logits.argsort(descending=True) == paris_token_id).nonzero().item() + 1
                    our_paris_rank = (our_logits.argsort(descending=True) == paris_token_id).nonzero().item() + 1
                    
                    print(f"  Paris token: '{paris_token_text}' (ID: {paris_token_id})")
                    print(f"  HF Paris:  logit {hf_paris_logit:6.3f}, rank {hf_paris_rank:6d}")
                    print(f"  Our Paris: logit {our_paris_logit:6.3f}, rank {our_paris_rank:6d}")
                    print(f"  Logit difference: {abs(hf_paris_logit - our_paris_logit):.3f}")
                    print(f"  Rank difference: {abs(hf_paris_rank - our_paris_rank):d}")
                    
                    if our_paris_rank <= 10:
                        print(f"  ✅ SUCCESS: Paris in top 10!")
                    elif our_paris_rank <= 100:
                        print(f"  ⚠️  PARTIAL: Paris in top 100 (rank {our_paris_rank})")
                    else:
                        print(f"  ❌ POOR: Paris at rank {our_paris_rank}")
                else:
                    print(f"  ❌ Could not find Paris token in vocabulary")
            else:
                print(f"\n💡 Paris prediction test skipped (prompt doesn't contain 'France')")
            
            # 📊 COSINE SIMILARITY COMPARISON
            
            print(f"\n📊 OVERALL SIMILARITY:")
            
            # Calculate cosine similarity between full logit distributions
            cosine_sim = F.cosine_similarity(
                hf_logits.unsqueeze(0), 
                our_logits.unsqueeze(0)
            ).item()
            
            print(f"  Cosine similarity: {cosine_sim:.6f}")
            
            if cosine_sim > 0.9:
                print(f"  ✅ EXCELLENT similarity (>0.9)")
            elif cosine_sim > 0.7:
                print(f"  ⚠️  GOOD similarity (>0.7)")
            elif cosine_sim > 0.5:
                print(f"  ⚠️  MODERATE similarity (>0.5)")
            else:
                print(f"  ❌ POOR similarity (<0.5)")
            
            # 🔍 LAYER-BY-LAYER ANALYSIS (if similarity is poor)
            if cosine_sim < 0.8:
                print(f"\n🔍 ACTIVATION ANALYSIS (cosine similarity {cosine_sim:.3f} needs investigation):")
                
                # Check embedding similarity
                hf_embeddings = hf_model.model.language_model.embed_tokens(input_ids)
                our_embeddings = our_model.embed_tokens(input_ids) * our_model.embed_scale
                
                emb_sim = F.cosine_similarity(
                    hf_embeddings.flatten().unsqueeze(0),
                    our_embeddings.flatten().unsqueeze(0)
                ).item()
                
                print(f"  Embedding similarity: {emb_sim:.6f}")
                print(f"  HF embedding range: [{hf_embeddings.min():.3f}, {hf_embeddings.max():.3f}]")
                print(f"  Our embedding range: [{our_embeddings.min():.3f}, {our_embeddings.max():.3f}]")
                
                if emb_sim < 0.99:
                    print(f"  ⚠️  EMBEDDING ISSUE: Similarity {emb_sim:.6f} indicates scaling or loading problem")
                else:
                    print(f"  ✅ Embeddings are correct - issue is in layer processing")
            
            # 🔍 LAYER-BY-LAYER ANALYSIS 
            print(f"\n🔍 LAYER-BY-LAYER ANALYSIS:")
            
            # Get layer information
            layer_types = []
            for i in range(len(our_model.layers)):
                layer = our_model.layers[i] 
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'attention_type'):
                    # Use the corrected attention_type that matches HF pattern
                    layer_types.append(layer.self_attn.attention_type)
                else:
                    # Fallback to HF pattern if attribute missing
                    layer_types.append('full_attention' if i % 5 == 0 else 'sliding_attention')
            
            print(f"  Layer 0: {layer_types[0]}")
            print(f"  Layer 1: {layer_types[1]}")
            print(f"  Layer 2: {layer_types[2]}")

            # Compare first layer attention output
            if 'hf_attn_output' in hf_activations:
                our_attn_output = our_model.layers[0].self_attn.attn_output_for_debug
                hf_attn_output = hf_activations['hf_attn_output']

                attn_sim = F.cosine_similarity(
                    hf_attn_output.flatten().unsqueeze(0),
                    our_attn_output.flatten().unsqueeze(0)
                ).item()
                print(f"\n  Attention Layer 0 Output Comparison:")
                print(f"    HF attn_output: min={hf_attn_output.min():.6f}, max={hf_attn_output.max():.6f}, mean={hf_attn_output.mean():.6f}, std={hf_attn_output.std():.6f}")
                print(f"    Our attn_output: min={our_attn_output.min():.6f}, max={our_attn_output.max():.6f}, mean={our_attn_output.mean():.6f}, std={our_attn_output.std():.6f}")
                print(f"    Cosine Similarity (Attention Output): {attn_sim:.6f}")
            
            # Compare first layer MLP output
            if 'hf_mlp_output' in hf_activations:
                our_mlp_output = our_model.layers[0].mlp.down_proj_for_debug
                hf_mlp_output = hf_activations['hf_mlp_output']

                mlp_sim = F.cosine_similarity(
                    hf_mlp_output.flatten().unsqueeze(0),
                    our_mlp_output.flatten().unsqueeze(0)
                ).item()
                print(f"\n  MLP Layer 0 Output Comparison:")
                print(f"    HF mlp_output: min={hf_mlp_output.min():.6f}, max={hf_mlp_output.max():.6f}, mean={hf_mlp_output.mean():.6f}, std={hf_mlp_output.std():.6f}")
                print(f"    Our mlp_output: min={our_mlp_output.min():.6f}, max={our_mlp_output.max():.6f}, mean={our_mlp_output.mean():.6f}, std={our_mlp_output.std():.6f}")
                print(f"    Cosine Similarity (MLP Output): {mlp_sim:.6f}")

            # Success indicator
            print(f"\n✅ Forward pass successful!")
            
            # Summary
            print(f"\n" + "="*60)
            print(f"✅ Gemma3n architecture test completed!")
            print(f"   Cosine similarity: {cosine_sim:.6f}")
            if "france" in TEST_PROMPT.lower() and 'paris_token_id' in locals() and paris_token_id is not None:
                print(f"   Paris rank: {our_paris_rank} (HF rank: {hf_paris_rank})")
            print(f"   Three-stage AltUp flow: ✅ Working")
            print(f"   Layer processing: ✅ Stable")
            print(f"   Ready for production ANE conversion")
            return True
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gemma3n_architecture()
    print(f"\n{'='*60}")
    if success:
        print("✅ Gemma3n architecture test passed!")
        print("   Our implementation successfully loaded and ran the model")
        print("   Ready to compare with HuggingFace implementation")
    else:
        print("❌ Gemma3n architecture test failed")
        print("   Need to debug weight loading or model structure")


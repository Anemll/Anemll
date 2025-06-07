#!/usr/bin/env python3
"""Detailed debugging of attention computation with KV cache."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))
from anemll.models.qwen_model import *
import glob
from transformers import AutoTokenizer

def debug_layer_attention():
    """Debug attention computation at layer level."""
    
    print("🔬 Detailed Attention Layer Debugging")
    print("=" * 60)
    
    # Setup
    model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Simple test case
    tokens = [785, 6722]  # ['The', ' capital'] 
    print(f"Testing tokens: {tokens} → {[tokenizer.decode([t]) for t in tokens]}")
    
    # Create models
    model_no_cache = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=True)
    model_no_cache.load_pretrained_weights(model_path)
    model_no_cache.eval()
    
    model_kv_cache = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model_kv_cache.load_pretrained_weights(model_path)
    model_kv_cache.eval()
    model_kv_cache.model.kv_cache_0.zero_()
    
    # === Step 1: Prefill KV cache ===
    print(f"\nStep 1: Prefilling KV cache with tokens {tokens}")
    input_ids = torch.tensor([tokens], dtype=torch.long, device=TEST_DEVICE)
    position_ids = torch.arange(len(tokens), dtype=torch.long, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, len(tokens), 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    model_kv_cache.prefill_kv_cache(
        input_ids=input_ids,
        position_ids=position_ids,
        start_pos=0,
        causal_mask=causal_mask
    )
    
    # === Step 2: Compare attention computation at first layer ===
    print(f"\nStep 2: Comparing attention computation at layer 0")
    
    layer_idx = 0
    layer_no_cache = model_no_cache.model.layers[layer_idx]
    layer_kv_cache = model_kv_cache.model.layers[layer_idx]
    
    # === NO CACHE: Full sequence forward ===
    print(f"\n🟢 NO CACHE: Processing full sequence")
    
    # Add dummy token for next position
    full_tokens = tokens + [tokenizer.pad_token_id]  # [785, 6722, pad]
    full_input_ids = torch.tensor([full_tokens], dtype=torch.long, device=TEST_DEVICE)
    
    # Get embeddings and normalize
    hidden_states_full = model_no_cache.model.embed_tokens(full_input_ids)
    print(f"  Full hidden states shape: {hidden_states_full.shape}")
    
    normalized_states_full = layer_no_cache.input_layernorm(hidden_states_full)
    print(f"  Normalized states shape: {normalized_states_full.shape}")
    print(f"  Normalized states norm: {torch.norm(normalized_states_full).item():.6f}")
    
    # Manual attention computation for no-cache
    hs = normalized_states_full.permute(0, 2, 1).unsqueeze(2)
    query_states_full = layer_no_cache.self_attn.q_proj(hs).view(1, 16, 128, 3).permute(0, 1, 3, 2)
    key_states_full = layer_no_cache.self_attn.k_proj(hs).view(1, 8, 128, 3).permute(0, 1, 3, 2)
    value_states_full = layer_no_cache.self_attn.v_proj(hs).view(1, 8, 128, 3).permute(0, 1, 3, 2)
    
    print(f"  Q/K/V shapes: {query_states_full.shape}, {key_states_full.shape}, {value_states_full.shape}")
    
    # === KV CACHE: Current token + cached K/V ===
    print(f"\n🔴 KV CACHE: Processing current token with cached K/V")
    
    # Current token (last token in sequence)
    current_token = tokens[-1]  # 6722 (' capital')
    current_input_ids = torch.tensor([[current_token]], dtype=torch.long, device=TEST_DEVICE)
    current_pos = len(tokens) - 1  # Position 1
    
    # Get embeddings and normalize for current token
    hidden_states_current = model_kv_cache.model.embed_tokens(current_input_ids)
    print(f"  Current hidden states shape: {hidden_states_current.shape}")
    
    normalized_states_current = layer_kv_cache.input_layernorm(hidden_states_current)
    print(f"  Current normalized states shape: {normalized_states_current.shape}")
    print(f"  Current normalized states norm: {torch.norm(normalized_states_current).item():.6f}")
    
    # Get rotary embeddings
    cos, sin = model_kv_cache.model.get_rotary_embeddings_s(current_pos)
    print(f"  Rotary embeddings shapes: cos {cos.shape}, sin {sin.shape}")
    
    # Get Q/K/V for current token
    query_states_current, key_states_current, value_states_current = layer_kv_cache.self_attn.get_new_kv_cache(
        normalized_states_current, current_pos, (cos, sin)
    )
    print(f"  Current Q/K/V shapes: {query_states_current.shape}, {key_states_current.shape}, {value_states_current.shape}")
    
    # Get cached K/V
    kv_cache = model_kv_cache.model.kv_cache_0
    key_cache = kv_cache[layer_idx, :, :len(tokens), :]  # [8, 2, 128] - all cached keys
    value_cache = kv_cache[layer_idx + config.num_hidden_layers, :, :len(tokens), :]  # [8, 2, 128] - all cached values
    
    print(f"  Cached K/V shapes: {key_cache.shape}, {value_cache.shape}")
    print(f"  Cached K/V norms: {torch.norm(key_cache).item():.6f}, {torch.norm(value_cache).item():.6f}")
    
    # === Step 3: Compare the key and value tensors ===
    print(f"\nStep 3: Comparing K/V tensors")
    
    # Extract the K/V for position 1 (current position) from both methods
    # No cache: extract position 1 from full computation
    key_pos1_no_cache = key_states_full[:, :, 1:2, :]  # Position 1 from full computation
    value_pos1_no_cache = value_states_full[:, :, 1:2, :]
    
    # KV cache: current token K/V
    key_pos1_kv_cache = key_states_current
    value_pos1_kv_cache = value_states_current
    
    print(f"  Key pos 1 shapes - No cache: {key_pos1_no_cache.shape}, KV cache: {key_pos1_kv_cache.shape}")
    print(f"  Value pos 1 shapes - No cache: {value_pos1_no_cache.shape}, KV cache: {value_pos1_kv_cache.shape}")
    
    # Compare norms
    key_diff = torch.norm(key_pos1_no_cache - key_pos1_kv_cache).item()
    value_diff = torch.norm(value_pos1_no_cache - value_pos1_kv_cache).item()
    
    print(f"  Key difference norm: {key_diff:.6f}")
    print(f"  Value difference norm: {value_diff:.6f}")
    
    if key_diff > 1e-3 or value_diff > 1e-3:
        print(f"  ❌ K/V values differ significantly!")
        
        # Check if the issue is in rotary embeddings
        print(f"\n  🔍 Checking rotary embeddings...")
        
        # Compare rotary embeddings for position 1
        full_position_ids = torch.arange(3, dtype=torch.long, device=TEST_DEVICE)
        cos_full, sin_full = layer_no_cache.self_attn.rotary_emb(normalized_states_full, full_position_ids)
        
        # Extract position 1 from full rotary embeddings
        cos_pos1_full = cos_full[:, 1:2, :, :]
        sin_pos1_full = sin_full[:, 1:2, :, :]
        
        cos_diff = torch.norm(cos_pos1_full - cos).item()
        sin_diff = torch.norm(sin_pos1_full - sin).item()
        
        print(f"    Cos difference: {cos_diff:.8f}")
        print(f"    Sin difference: {sin_diff:.8f}")
        
    else:
        print(f"  ✅ K/V values match!")

if __name__ == "__main__":
    debug_layer_attention() 
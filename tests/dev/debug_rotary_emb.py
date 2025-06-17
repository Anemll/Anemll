#!/usr/bin/env python3
"""Debug rotary embeddings between KV cache and no-cache methods."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))
from anemll.models.qwen_model import *
import glob
from transformers import AutoTokenizer

def debug_rotary_embeddings():
    """Compare rotary embeddings for the same position."""
    
    print("🔍 Debugging Rotary Embeddings")
    print("=" * 50)
    
    # Setup
    model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Create a single model to test both methods
    model = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    model.eval()
    
    tokens = [785, 6722]  # ['The', ' capital']
    position_to_test = 1  # Test position 1 (' capital')
    
    print(f"Testing rotary embeddings for position {position_to_test}")
    print(f"Tokens: {tokens} → {[tokenizer.decode([t]) for t in tokens]}")
    
    # === Method 1: Full sequence rotary embeddings ===
    print(f"\n🟢 Method 1: Full sequence rotary embeddings")
    
    # Process full sequence [785, 6722, pad] with positions [0, 1, 2]
    full_tokens = tokens + [tokenizer.pad_token_id]
    input_ids_full = torch.tensor([full_tokens], dtype=torch.long, device=TEST_DEVICE)
    position_ids_full = torch.arange(len(full_tokens), dtype=torch.long, device=TEST_DEVICE)
    
    # Get embeddings and normalize
    hidden_states_full = model.model.embed_tokens(input_ids_full)
    layer0 = model.model.layers[0]
    normalized_states_full = layer0.input_layernorm(hidden_states_full)
    
    # Get rotary embeddings for full sequence
    cos_full, sin_full = layer0.self_attn.rotary_emb(normalized_states_full, position_ids_full)
    
    print(f"  Full sequence cos shape: {cos_full.shape}")
    print(f"  Full sequence sin shape: {sin_full.shape}")
    
    # Extract rotary embeddings for position 1
    cos_pos1_full = cos_full[:, position_to_test:position_to_test+1, :]  # Position 1
    sin_pos1_full = sin_full[:, position_to_test:position_to_test+1, :]  # Position 1
    
    print(f"  Position {position_to_test} cos shape: {cos_pos1_full.shape}")
    print(f"  Position {position_to_test} sin shape: {sin_pos1_full.shape}")
    print(f"  Position {position_to_test} cos norm: {torch.norm(cos_pos1_full).item():.6f}")
    print(f"  Position {position_to_test} sin norm: {torch.norm(sin_pos1_full).item():.6f}")
    
    # === Method 2: Single position rotary embeddings (KV cache method) ===
    print(f"\n🔴 Method 2: Single position rotary embeddings (KV cache)")
    
    # Get rotary embeddings for single position
    cos_single, sin_single = model.model.get_rotary_embeddings_s(position_to_test)
    
    print(f"  Single position cos shape: {cos_single.shape}")
    print(f"  Single position sin shape: {sin_single.shape}")
    print(f"  Single position cos norm: {torch.norm(cos_single).item():.6f}")
    print(f"  Single position sin norm: {torch.norm(sin_single).item():.6f}")
    
    # === Compare the embeddings ===
    print(f"\n📊 COMPARISON")
    print("=" * 30)
    
    # Reshape for comparison (both should be [1, 1, 1, 128])
    cos_pos1_full_reshaped = cos_pos1_full.view(1, 1, 1, -1)
    sin_pos1_full_reshaped = sin_pos1_full.view(1, 1, 1, -1)
    
    cos_diff = torch.norm(cos_pos1_full_reshaped - cos_single).item()
    sin_diff = torch.norm(sin_pos1_full_reshaped - sin_single).item()
    
    print(f"Cos difference: {cos_diff:.8f}")
    print(f"Sin difference: {sin_diff:.8f}")
    
    if cos_diff < 1e-6 and sin_diff < 1e-6:
        print("✅ Rotary embeddings match!")
        return True
    else:
        print("❌ Rotary embeddings differ!")
        
        # Show actual values for debugging
        print(f"\nFirst few cos values:")
        print(f"  Full sequence: {cos_pos1_full_reshaped[0, 0, 0, :5].tolist()}")
        print(f"  Single pos:    {cos_single[0, 0, 0, :5].tolist()}")
        
        print(f"\nFirst few sin values:")
        print(f"  Full sequence: {sin_pos1_full_reshaped[0, 0, 0, :5].tolist()}")
        print(f"  Single pos:    {sin_single[0, 0, 0, :5].tolist()}")
        
        return False

if __name__ == "__main__":
    debug_rotary_embeddings() 
#!/usr/bin/env python3
"""Debug the exact storage process in the KV cache fix."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))
from anemll.models.qwen_model import *
import glob
from transformers import AutoTokenizer

def debug_storage_fix():
    """Debug the exact storage process in the KV cache."""
    
    print("🔍 Storage Fix Debug: Exact Shape and Storage Tracing")
    print("=" * 70)
    
    # Setup
    model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Test case: just focus on position 2 storage
    tokens = [785, 6722]  # ['The', ' capital']
    print(f"Test tokens: {tokens} → {[tokenizer.decode([t]) for t in tokens]}")
    
    # Create KV cache model
    model = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    model.eval()
    model.model.kv_cache_0.zero_()
    
    # Process first two tokens to populate cache
    for i, token in enumerate(tokens):
        input_ids = torch.tensor([[token]], dtype=torch.long, device=TEST_DEVICE)
        position_ids = torch.tensor([i], dtype=torch.long, device=TEST_DEVICE)
        current_pos = torch.tensor([i], dtype=torch.long, device=TEST_DEVICE)
        causal_mask = torch.zeros((1, 1, 1, 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        
        with torch.no_grad():
            model(
                input_ids=input_ids,
                update_mask=torch.ones((1, 1, 512, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False
            )
    
    print(f"After processing 'The capital', cache state:")
    kv_cache = model.model.kv_cache_0
    print(f"Position 0 key norm: {torch.norm(kv_cache[0, :, 0, :]).item():.6f}")
    print(f"Position 1 key norm: {torch.norm(kv_cache[0, :, 1, :]).item():.6f}")
    print(f"Position 2 key norm: {torch.norm(kv_cache[0, :, 2, :]).item():.6f}")
    
    print(f"\n" + "="*50)
    print(f"TESTING STORAGE AT POSITION 2")
    print("="*50)
    
    # Now test the storage process manually at position 2
    placeholder_token = tokenizer.pad_token_id
    input_ids = torch.tensor([[placeholder_token]], dtype=torch.long, device=TEST_DEVICE)
    position_ids = torch.tensor([2], dtype=torch.long, device=TEST_DEVICE)
    current_pos = torch.tensor([2], dtype=torch.long, device=TEST_DEVICE)
    
    print(f"Testing with placeholder token: {placeholder_token} ('{tokenizer.decode([placeholder_token])}')")
    
    # Get embeddings and normalized states
    hidden_states = model.model.embed_tokens(input_ids)
    layer0 = model.model.layers[0]
    normalized_states = layer0.input_layernorm(hidden_states)
    
    # Get rotary embeddings
    cos, sin = model.model.get_rotary_embeddings_s(current_pos)
    
    # Get new K/V states
    query_states, key_states, value_states = layer0.self_attn.get_new_kv_cache(
        normalized_states, current_pos, (cos, sin)
    )
    
    print(f"Computed K/V states:")
    print(f"  Key states shape: {key_states.shape}")
    print(f"  Value states shape: {value_states.shape}")
    print(f"  Key states norm: {torch.norm(key_states).item():.6f}")
    print(f"  Value states norm: {torch.norm(value_states).item():.6f}")
    
    # Get cache indices
    group_idx, layer_in_group_idx, layers_per_group = get_kv_cache_idx(0, config.num_hidden_layers)
    key_idx = layer_in_group_idx  # Should be 0
    value_idx = layer_in_group_idx + layers_per_group  # Should be 28
    
    print(f"Cache indices: key_idx={key_idx}, value_idx={value_idx}")
    
    # Test the storage process
    pos = current_pos.item()  # Should be 2
    print(f"Storing at position: {pos}")
    
    # Check shapes before storage
    print(f"Cache slice shapes:")
    print(f"  kv_cache[{key_idx}, :, {pos}, :] shape: {kv_cache[key_idx, :, pos, :].shape}")
    print(f"  After squeeze - key_states shape: {key_states.squeeze(0).squeeze(1).shape}")
    print(f"  After squeeze - value_states shape: {value_states.squeeze(0).squeeze(1).shape}")
    
    # Store manually and check
    print(f"\nBefore storage:")
    print(f"  Position {pos} key norm: {torch.norm(kv_cache[key_idx, :, pos, :]).item():.6f}")
    print(f"  Position {pos} value norm: {torch.norm(kv_cache[value_idx, :, pos, :]).item():.6f}")
    
    # Execute the storage
    kv_cache[key_idx, :, pos, :] = key_states.squeeze(0).squeeze(1)
    kv_cache[value_idx, :, pos, :] = value_states.squeeze(0).squeeze(1)
    
    print(f"\nAfter storage:")
    print(f"  Position {pos} key norm: {torch.norm(kv_cache[key_idx, :, pos, :]).item():.6f}")
    print(f"  Position {pos} value norm: {torch.norm(kv_cache[value_idx, :, pos, :]).item():.6f}")
    
    # Check if they match what we stored
    stored_key = kv_cache[key_idx, :, pos, :]
    stored_value = kv_cache[value_idx, :, pos, :]
    original_key = key_states.squeeze(0).squeeze(1)
    original_value = value_states.squeeze(0).squeeze(1)
    
    key_diff = torch.norm(stored_key - original_key).item()
    value_diff = torch.norm(stored_value - original_value).item()
    
    print(f"\nStorage verification:")
    print(f"  Key storage diff: {key_diff:.10f}")
    print(f"  Value storage diff: {value_diff:.10f}")
    
    # Now test retrieval as in forward_regular
    print(f"\n" + "="*50)
    print(f"TESTING RETRIEVAL PROCESS")
    print("="*50)
    
    # Get the cache as it would be retrieved in forward_regular
    key_cache = kv_cache[key_idx:key_idx + 1].squeeze(0)  # Shape [8, 512, 128]
    value_cache = kv_cache[value_idx:value_idx + 1].squeeze(0)  # Shape [8, 512, 128]
    
    print(f"Retrieved cache shapes: {key_cache.shape}, {value_cache.shape}")
    
    # Test slicing as in forward_regular
    cache_len = pos + 1  # Should be 3
    K_layer_cache = key_cache[..., :cache_len, :]  # Should be [8, 3, 128]
    V_layer_cache = value_cache[..., :cache_len, :]  # Should be [8, 3, 128]
    
    print(f"After slicing to cache_len={cache_len}:")
    print(f"  K_layer_cache shape: {K_layer_cache.shape}")
    print(f"  V_layer_cache shape: {V_layer_cache.shape}")
    print(f"  Position {pos} key norm in sliced cache: {torch.norm(K_layer_cache[:, pos, :]).item():.6f}")
    print(f"  Position {pos} value norm in sliced cache: {torch.norm(V_layer_cache[:, pos, :]).item():.6f}")
    
    # Compare with what we originally computed
    retrieved_key = K_layer_cache[:, pos, :]
    retrieved_value = V_layer_cache[:, pos, :]
    
    retrieval_key_diff = torch.norm(retrieved_key - original_key).item()
    retrieval_value_diff = torch.norm(retrieved_value - original_value).item()
    
    print(f"\nRetrieval verification:")
    print(f"  Retrieved key vs original diff: {retrieval_key_diff:.10f}")
    print(f"  Retrieved value vs original diff: {retrieval_value_diff:.10f}")
    
    print(f"\nFinal analysis:")
    if retrieval_key_diff < 1e-6 and retrieval_value_diff < 1e-6:
        print("✅ Storage and retrieval working correctly!")
    else:
        print("❌ Storage or retrieval has issues")

if __name__ == "__main__":
    from anemll.models.qwen_model import TEST_DEVICE, MODEL_DTYPE
    debug_storage_fix() 
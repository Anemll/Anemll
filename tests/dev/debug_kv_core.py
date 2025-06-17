#!/usr/bin/env python3
"""Debug the core KV cache storage and retrieval mechanism."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))
from anemll.models.qwen_model import *
import glob
from transformers import AutoTokenizer

def debug_kv_core():
    """Debug KV cache storage and retrieval at the core level."""
    
    print("🔍 Core KV Cache Debug: Storage & Retrieval Tracing")
    print("=" * 70)
    
    # Setup
    model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Simple 2-token test case
    tokens = [785, 6722]  # ['The', ' capital']
    print(f"Test tokens: {tokens} → {[tokenizer.decode([t]) for t in tokens]}")
    
    # Create KV cache model
    model = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    model.eval()
    model.model.kv_cache_0.zero_()
    
    print(f"\nInitial KV cache shape: {model.model.kv_cache_0.shape}")
    print(f"Initial KV cache norm: {torch.norm(model.model.kv_cache_0).item():.6f}")
    
    # === Step 1: Trace First Token Processing ===
    print(f"\n" + "="*50)
    print(f"STEP 1: Processing token {tokens[0]} ('{tokenizer.decode([tokens[0]])}') at position 0")
    print("="*50)
    
    # Process first token
    input_ids = torch.tensor([[tokens[0]]], dtype=torch.long, device=TEST_DEVICE)
    position_ids = torch.tensor([0], dtype=torch.long, device=TEST_DEVICE)
    current_pos = torch.tensor([0], dtype=torch.long, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    # Hook into the first layer to see what gets stored
    layer0 = model.model.layers[0]
    
    print(f"Before processing:")
    kv_cache = model.model.kv_cache_0
    layer0_key_cache = kv_cache[0, :, :1, :]  # Layer 0 keys, position 0
    layer0_value_cache = kv_cache[28, :, :1, :]  # Layer 0 values, position 0
    print(f"  Layer 0 key cache [0, :, 0, :] norm: {torch.norm(layer0_key_cache).item():.6f}")
    print(f"  Layer 0 value cache [28, :, 0, :] norm: {torch.norm(layer0_value_cache).item():.6f}")
    
    with torch.no_grad():
        logits1 = model(
            input_ids=input_ids,
            update_mask=torch.ones((1, 1, 512, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
    
    print(f"After processing:")
    layer0_key_cache = kv_cache[0, :, :1, :]  # Layer 0 keys, position 0
    layer0_value_cache = kv_cache[28, :, :1, :]  # Layer 0 values, position 0
    print(f"  Layer 0 key cache [0, :, 0, :] norm: {torch.norm(layer0_key_cache).item():.6f}")
    print(f"  Layer 0 value cache [28, :, 0, :] norm: {torch.norm(layer0_value_cache).item():.6f}")
    print(f"  First few key values: {layer0_key_cache[0, 0, :5].tolist()}")
    print(f"  First few value values: {layer0_value_cache[0, 0, :5].tolist()}")
    
    # === Step 2: Trace Second Token Processing ===
    print(f"\n" + "="*50)
    print(f"STEP 2: Processing token {tokens[1]} ('{tokenizer.decode([tokens[1]])}') at position 1")
    print("="*50)
    
    # Process second token
    input_ids = torch.tensor([[tokens[1]]], dtype=torch.long, device=TEST_DEVICE)
    position_ids = torch.tensor([1], dtype=torch.long, device=TEST_DEVICE)
    current_pos = torch.tensor([1], dtype=torch.long, device=TEST_DEVICE)
    
    print(f"Before processing token 1:")
    layer0_key_cache_pos1 = kv_cache[0, :, 1:2, :]  # Layer 0 keys, position 1
    layer0_value_cache_pos1 = kv_cache[28, :, 1:2, :]  # Layer 0 values, position 1
    print(f"  Layer 0 key cache [0, :, 1, :] norm: {torch.norm(layer0_key_cache_pos1).item():.6f}")
    print(f"  Layer 0 value cache [28, :, 1, :] norm: {torch.norm(layer0_value_cache_pos1).item():.6f}")
    
    with torch.no_grad():
        logits2 = model(
            input_ids=input_ids,
            update_mask=torch.ones((1, 1, 512, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
    
    print(f"After processing token 1:")
    layer0_key_cache_pos1 = kv_cache[0, :, 1:2, :]  # Layer 0 keys, position 1
    layer0_value_cache_pos1 = kv_cache[28, :, 1:2, :]  # Layer 0 values, position 1
    print(f"  Layer 0 key cache [0, :, 1, :] norm: {torch.norm(layer0_key_cache_pos1).item():.6f}")
    print(f"  Layer 0 value cache [28, :, 1, :] norm: {torch.norm(layer0_value_cache_pos1).item():.6f}")
    print(f"  First few key values: {layer0_key_cache_pos1[0, 0, :5].tolist()}")
    print(f"  First few value values: {layer0_value_cache_pos1[0, 0, :5].tolist()}")
    
    # === Step 3: Trace Prediction Step ===
    print(f"\n" + "="*50)
    print(f"STEP 3: Predicting next token at position 2")
    print("="*50)
    
    # Now predict next token using placeholder
    placeholder_token = tokenizer.pad_token_id
    input_ids = torch.tensor([[placeholder_token]], dtype=torch.long, device=TEST_DEVICE)
    position_ids = torch.tensor([2], dtype=torch.long, device=TEST_DEVICE)
    current_pos = torch.tensor([2], dtype=torch.long, device=TEST_DEVICE)
    
    print(f"Using placeholder token: {placeholder_token} ('{tokenizer.decode([placeholder_token])}')")
    
    # Before prediction, let's examine what should be retrieved from cache
    print(f"Cache state before prediction:")
    layer0_keys_all = kv_cache[0, :, :2, :]  # Layer 0 keys, positions 0-1
    layer0_values_all = kv_cache[28, :, :2, :]  # Layer 0 values, positions 0-1
    print(f"  Cached keys shape: {layer0_keys_all.shape}")
    print(f"  Cached values shape: {layer0_values_all.shape}")
    print(f"  Cached keys norm: {torch.norm(layer0_keys_all).item():.6f}")
    print(f"  Cached values norm: {torch.norm(layer0_values_all).item():.6f}")
    
    # Let's manually trace what happens in the attention computation
    print(f"\nManual attention computation trace:")
    
    # Get the embeddings for the placeholder token
    hidden_states = model.model.embed_tokens(input_ids)
    print(f"  Placeholder embedding norm: {torch.norm(hidden_states).item():.6f}")
    
    # Get normalized states
    normalized_states = layer0.input_layernorm(hidden_states)
    print(f"  Normalized states norm: {torch.norm(normalized_states).item():.6f}")
    
    # Get rotary embeddings
    cos, sin = model.model.get_rotary_embeddings_s(current_pos)
    print(f"  Rotary cos/sin shapes: {cos.shape}, {sin.shape}")
    print(f"  Rotary cos/sin norms: {torch.norm(cos).item():.6f}, {torch.norm(sin).item():.6f}")
    
    # Get Q/K/V for current token
    query_states, key_states, value_states = layer0.self_attn.get_new_kv_cache(
        normalized_states, current_pos, (cos, sin)
    )
    print(f"  Current Q/K/V shapes: {query_states.shape}, {key_states.shape}, {value_states.shape}")
    print(f"  Current Q/K/V norms: {torch.norm(query_states).item():.6f}, {torch.norm(key_states).item():.6f}, {torch.norm(value_states).item():.6f}")
    
    # Now let's see what the forward_regular method retrieves and uses
    print(f"\nAttention computation with cached K/V:")
    
    # Get cached K/V for this layer
    key_cache = kv_cache[0, :, :, :]  # All keys for layer 0
    value_cache = kv_cache[28, :, :, :]  # All values for layer 0
    
    # Simulate forward_regular logic
    cache_len = current_pos.item() + 1  # Should be 3 (positions 0, 1, 2)
    print(f"  Cache length for attention: {cache_len}")
    
    # Slice cache to active positions
    K_layer_cache = key_cache[..., :cache_len, :]  # Positions 0, 1, 2
    V_layer_cache = value_cache[..., :cache_len, :]
    print(f"  Active cache shapes: {K_layer_cache.shape}, {V_layer_cache.shape}")
    print(f"  Active cache norms: {torch.norm(K_layer_cache).item():.6f}, {torch.norm(V_layer_cache).item():.6f}")
    
    # Check what's actually in position 2 (should be the new key/value we just computed)
    pos2_key = K_layer_cache[:, 2:3, :]
    pos2_value = V_layer_cache[:, 2:3, :]
    print(f"  Position 2 K/V norms: {torch.norm(pos2_key).item():.6f}, {torch.norm(pos2_value).item():.6f}")
    
    # Compare with what we computed manually
    key_diff = torch.norm(pos2_key.squeeze(1) - key_states.squeeze(1)).item()
    value_diff = torch.norm(pos2_value.squeeze(1) - value_states.squeeze(1)).item()
    print(f"  Position 2 K/V vs manual computation diff: {key_diff:.6f}, {value_diff:.6f}")
    
    # Now run the actual prediction and see the result
    print(f"\nRunning actual prediction:")
    with torch.no_grad():
        logits3 = model(
            input_ids=input_ids,
            update_mask=torch.ones((1, 1, 512, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
    
    next_token = torch.argmax(logits3[0, 0, :]).item()
    print(f"  Predicted token: {next_token} ('{tokenizer.decode([next_token])}')")
    print(f"  Logits norm: {torch.norm(logits3).item():.6f}")
    
    # === Step 4: Compare with Expected ===
    print(f"\n" + "="*50)
    print(f"STEP 4: Comparison with Expected Behavior")
    print("="*50)
    
    # What should happen: predict next token after "The capital"
    # Let's see what a reasonable baseline would predict
    print(f"Context processed: {[tokenizer.decode([t]) for t in tokens]}")
    print(f"KV cache prediction: {next_token} ('{tokenizer.decode([next_token])}')")
    
    # Check if the attention is actually using the cached context
    # The key insight: if attention weights are mostly on position 2 (placeholder), 
    # then it's ignoring the cached context
    print(f"\nAnalysis:")
    print(f"- Cached 2 positions with meaningful tokens: {[tokenizer.decode([t]) for t in tokens]}")
    print(f"- Position 2 has placeholder token: '{tokenizer.decode([placeholder_token])}'")
    print(f"- If attention mostly focuses on position 2, it's ignoring context")
    print(f"- If attention focuses on positions 0-1, it's using context correctly")

if __name__ == "__main__":
    from anemll.models.qwen_model import TEST_DEVICE, MODEL_DTYPE
    debug_kv_core() 
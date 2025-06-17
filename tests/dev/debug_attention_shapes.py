#!/usr/bin/env python3
"""Debug exact tensor shapes through attention computation."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))
from anemll.models.qwen_model import *
import glob
from transformers import AutoTokenizer

def debug_attention_shapes():
    """Debug the exact tensor shapes through attention computation."""
    
    print("🔍 Attention Shapes Debug: Exact Tensor Flow Tracing")
    print("=" * 70)
    
    # Setup
    model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Create KV cache model
    model = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    model.eval()
    model.model.kv_cache_0.zero_()
    
    # Process two tokens first
    tokens = [785, 6722]  # ['The', ' capital']
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
    
    print(f"Processed context: {[tokenizer.decode([t]) for t in tokens]}")
    
    # Now trace step-by-step through attention for position 2
    print(f"\n" + "="*50)
    print(f"STEP-BY-STEP ATTENTION COMPUTATION AT POSITION 2")
    print("="*50)
    
    placeholder_token = tokenizer.pad_token_id
    input_ids = torch.tensor([[placeholder_token]], dtype=torch.long, device=TEST_DEVICE)
    position_ids = torch.tensor([2], dtype=torch.long, device=TEST_DEVICE)
    current_pos = torch.tensor([2], dtype=torch.long, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    print(f"Input token: {placeholder_token} ('{tokenizer.decode([placeholder_token])}')")
    print(f"Position: {current_pos.item()}")
    
    # Manual step-by-step computation
    hidden_states = model.model.embed_tokens(input_ids)
    print(f"1. Embeddings shape: {hidden_states.shape}")
    
    # Focus on layer 0
    layer0 = model.model.layers[0]
    normalized_states = layer0.input_layernorm(hidden_states)
    print(f"2. Normalized states shape: {normalized_states.shape}")
    
    # Get rotary embeddings
    cos, sin = model.model.get_rotary_embeddings_s(current_pos)
    print(f"3. Rotary cos/sin shapes: {cos.shape}, {sin.shape}")
    
    # Get current Q/K/V
    query_states, key_states, value_states = layer0.self_attn.get_new_kv_cache(
        normalized_states, current_pos, (cos, sin)
    )
    print(f"4. Current Q/K/V shapes: {query_states.shape}, {key_states.shape}, {value_states.shape}")
    
    # Get KV cache and simulate storage process
    kv_cache = model.model.kv_cache_0
    key_idx = 0
    value_idx = 28
    pos = current_pos.item()
    
    # Store current K/V (as in our fix)
    kv_cache[key_idx, :, pos, :] = key_states.squeeze(0).squeeze(1)
    kv_cache[value_idx, :, pos, :] = value_states.squeeze(0).squeeze(1)
    print(f"5. Stored current K/V at position {pos}")
    
    # Get cache for attention (as in forward_regular)
    key_cache = kv_cache[key_idx:key_idx + 1].squeeze(0)  # Shape [8, 512, 128]
    value_cache = kv_cache[value_idx:value_idx + 1].squeeze(0)  # Shape [8, 512, 128]
    print(f"6. Retrieved cache shapes: {key_cache.shape}, {value_cache.shape}")
    
    # Slice to active length
    cache_len = pos + 1  # Should be 3
    K_layer_cache = key_cache[..., :cache_len, :]  # Should be [8, 3, 128]
    V_layer_cache = value_cache[..., :cache_len, :]  # Should be [8, 3, 128]
    print(f"7. Sliced cache shapes (cache_len={cache_len}): {K_layer_cache.shape}, {V_layer_cache.shape}")
    
    # Check what's in each position
    for i in range(cache_len):
        key_norm = torch.norm(K_layer_cache[:, i, :]).item()
        value_norm = torch.norm(V_layer_cache[:, i, :]).item()
        print(f"   Position {i}: Key norm = {key_norm:.6f}, Value norm = {value_norm:.6f}")
    
    # Repeat KV for multi-head attention
    n_rep = layer0.self_attn.num_heads // layer0.self_attn.num_kv_heads
    print(f"8. n_rep = {n_rep} (num_heads={layer0.self_attn.num_heads}, num_kv_heads={layer0.self_attn.num_kv_heads})")
    
    key_states_repeated = layer0.self_attn.repeat_kv(K_layer_cache, n_rep)
    value_states_repeated = layer0.self_attn.repeat_kv(V_layer_cache, n_rep)
    print(f"9. After repeat_kv: {key_states_repeated.shape}, {value_states_repeated.shape}")
    
    # Check attention computation
    print(f"\n--- ATTENTION COMPUTATION ---")
    print(f"Query shape: {query_states.shape}")
    print(f"Key shape (repeated): {key_states_repeated.shape}")
    print(f"Value shape (repeated): {value_states_repeated.shape}")
    
    # Compute attention weights
    attn_weights = torch.matmul(
        query_states.to(MODEL_DTYPE), 
        key_states_repeated.transpose(-1, -2).to(MODEL_DTYPE)
    ) * layer0.self_attn.scale
    print(f"10. Attention weights shape: {attn_weights.shape}")
    print(f"    Attention weights norm: {torch.norm(attn_weights).item():.6f}")
    
    # Apply causal mask
    if causal_mask is not None:
        q_seq_len = query_states.shape[-2]  # Should be 1
        k_seq_len = key_states_repeated.shape[-2]  # Should be 3
        print(f"11. Applying causal mask: q_seq_len={q_seq_len}, k_seq_len={k_seq_len}")
        mask_slice = causal_mask.to(MODEL_DTYPE)[:, :, :q_seq_len, :k_seq_len]
        print(f"    Mask slice shape: {mask_slice.shape}")
        attn_weights = attn_weights + mask_slice
    
    # Softmax
    attn_weights = torch.softmax(attn_weights, dim=-1)
    print(f"12. After softmax:")
    print(f"    Attention weights shape: {attn_weights.shape}")
    print(f"    Attention weights: {attn_weights[0, 0, 0, :].tolist()}")
    
    # Check if attention is distributed properly
    attn_sum = torch.sum(attn_weights[0, 0, 0, :]).item()
    print(f"    Attention sum (should be ~1.0): {attn_sum:.6f}")
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states_repeated.to(MODEL_DTYPE))
    print(f"13. Attention output shape: {attn_output.shape}")
    print(f"    Attention output norm: {torch.norm(attn_output).item():.6f}")
    
    # Compare with forward_regular result
    print(f"\n--- COMPARE WITH FORWARD_REGULAR ---")
    attn_output_official = layer0.self_attn.forward_regular(
        hidden_states=normalized_states,
        query_states=query_states,
        kv_cache_layer=(key_cache, value_cache),
        causal_mask=causal_mask,
        current_pos=current_pos,
    )
    print(f"Official forward_regular output shape: {attn_output_official.shape}")
    print(f"Official forward_regular output norm: {torch.norm(attn_output_official).item():.6f}")
    
    # Check difference
    # Need to reshape our manual computation to match
    attn_output_reshaped = attn_output.transpose(1, 2).contiguous()
    attn_output_reshaped = attn_output_reshaped.reshape(1, 1, layer0.self_attn.num_heads * layer0.self_attn.head_dim)
    attn_output_reshaped = layer0.self_attn.o_proj(attn_output_reshaped.permute(0, 2, 1).unsqueeze(2))
    attn_output_reshaped = attn_output_reshaped.squeeze(2).permute(0, 2, 1)
    
    diff = torch.norm(attn_output_reshaped - attn_output_official).item()
    print(f"Difference between manual and official: {diff:.10f}")
    
    if diff < 1e-6:
        print("✅ Manual computation matches official implementation")
    else:
        print("❌ Manual computation differs from official implementation")
        print(f"Manual result norm: {torch.norm(attn_output_reshaped).item():.6f}")
        print(f"Official result norm: {torch.norm(attn_output_official).item():.6f}")

if __name__ == "__main__":
    from anemll.models.qwen_model import TEST_DEVICE, MODEL_DTYPE
    debug_attention_shapes() 
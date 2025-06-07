#!/usr/bin/env python3
"""Debug attention computation with vs without KV cache."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))
from anemll.models.qwen_model import *
import glob
from transformers import AutoTokenizer

def test_attention_comparison():
    """Compare attention computation with and without KV cache step by step."""
    
    print("🔍 Debugging Attention Computation: KV Cache vs No Cache")
    print("=" * 70)
    
    # Load model and tokenizer
    model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Test with simple prompt
    prompt = "The capital"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids.to(TEST_DEVICE)
    prompt_tokens = input_ids[0].tolist()
    
    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {prompt_tokens} → {[tokenizer.decode([t]) for t in prompt_tokens]}")
    
    # === Method 1: No KV Cache (Working) ===
    print(f"\n🟢 Method 1: No KV Cache (Should work correctly)")
    print("-" * 50)
    
    model1 = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=True)
    model1.load_pretrained_weights(model_path)
    model1.eval()
    
    # Add position for next token prediction  
    next_pos = len(prompt_tokens)
    extended_input_ids = torch.cat([input_ids, torch.tensor([[tokenizer.pad_token_id]], device=TEST_DEVICE)], dim=1)
    position_ids = torch.arange(next_pos + 1, dtype=torch.long, device=TEST_DEVICE)
    current_pos = torch.tensor([next_pos], dtype=torch.long, device=TEST_DEVICE)
    
    # Create causal mask
    seq_len = next_pos + 1
    causal_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            causal_mask[0, 0, i, j] = float('-inf')
    
    with torch.no_grad():
        logits1 = model1(
            input_ids=extended_input_ids,
            update_mask=torch.ones((1, 1, 512, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
    
    next_token1 = torch.argmax(logits1[0, 0, :]).item()
    print(f"No cache result: {next_token1} ('{tokenizer.decode([next_token1])}')")
    
    # === Method 2: With KV Cache (Broken) ===
    print(f"\n🔴 Method 2: With KV Cache (Currently broken)")
    print("-" * 50)
    
    model2 = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model2.load_pretrained_weights(model_path)
    model2.eval()
    model2.model.kv_cache_0.zero_()
    
    # Prefill with prompt
    prefill_position_ids = torch.arange(len(prompt_tokens), dtype=torch.long, device=TEST_DEVICE)
    prefill_causal_mask = torch.zeros((1, 1, len(prompt_tokens), 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    print(f"Prefilling {len(prompt_tokens)} tokens...")
    model2.prefill_kv_cache(
        input_ids=input_ids,
        position_ids=prefill_position_ids,
        start_pos=0,
        causal_mask=prefill_causal_mask
    )
    
    # Generate next token
    next_input_ids = torch.tensor([[prompt_tokens[-1]]], dtype=torch.long, device=TEST_DEVICE)
    next_position_ids = torch.tensor([next_pos], dtype=torch.long, device=TEST_DEVICE)
    next_current_pos = torch.tensor([next_pos], dtype=torch.long, device=TEST_DEVICE)
    next_causal_mask = torch.zeros((1, 1, 1, 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    with torch.no_grad():
        logits2 = model2(
            input_ids=next_input_ids,
            update_mask=torch.ones((1, 1, 512, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            position_ids=next_position_ids,
            causal_mask=next_causal_mask,
            current_pos=next_current_pos,
            IN_PREFILL=False
        )
    
    next_token2 = torch.argmax(logits2[0, 0, :]).item()
    print(f"KV cache result: {next_token2} ('{tokenizer.decode([next_token2])}')")
    
    # === Compare Results ===
    print(f"\n📊 COMPARISON")
    print("=" * 50)
    print(f"No KV cache:  {next_token1} ('{tokenizer.decode([next_token1])}')")
    print(f"With KV cache: {next_token2} ('{tokenizer.decode([next_token2])}')")
    print(f"Match: {'✅' if next_token1 == next_token2 else '❌'}")
    
    if next_token1 != next_token2:
        print(f"\n🔍 DEBUGGING: Let's check intermediate values...")
        debug_attention_internals(model1, model2, input_ids, prompt_tokens, config, tokenizer)
    
    return next_token1 == next_token2

def debug_attention_internals(model1, model2, input_ids, prompt_tokens, config, tokenizer):
    """Debug the internal attention computation differences."""
    
    print(f"\n🔬 Deep Dive: Attention Internals")
    print("-" * 50)
    
    # Let's manually trace through the first attention layer to see what's different
    # We'll compare the attention weights and outputs
    
    next_pos = len(prompt_tokens)
    
    # === Setup for no-cache model ===
    extended_input_ids = torch.cat([input_ids, torch.tensor([[tokenizer.pad_token_id]], device=TEST_DEVICE)], dim=1)
    position_ids_full = torch.arange(next_pos + 1, dtype=torch.long, device=TEST_DEVICE)
    
    # === Setup for KV cache model ===
    # First prefill the cache
    model2.model.kv_cache_0.zero_()
    prefill_position_ids = torch.arange(len(prompt_tokens), dtype=torch.long, device=TEST_DEVICE)
    prefill_causal_mask = torch.zeros((1, 1, len(prompt_tokens), 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    model2.prefill_kv_cache(
        input_ids=input_ids,
        position_ids=prefill_position_ids,
        start_pos=0,
        causal_mask=prefill_causal_mask
    )
    
    # Now let's manually compute attention for both methods at layer 0
    layer_idx = 0
    layer1 = model1.model.layers[layer_idx]  # No cache model
    layer2 = model2.model.layers[layer_idx]  # KV cache model
    
    print(f"\nAnalyzing attention layer {layer_idx}...")
    
    # === No-cache forward ===
    print(f"No-cache computation:")
    hidden_states1 = model1.model.embed_tokens(extended_input_ids)
    normalized_states1 = layer1.input_layernorm(hidden_states1)
    
    print(f"  Hidden states shape: {hidden_states1.shape}")
    print(f"  Normalized states shape: {normalized_states1.shape}")
    print(f"  Normalized states norm: {torch.norm(normalized_states1).item():.6f}")
    
    # === KV-cache forward ===  
    print(f"KV-cache computation:")
    # For next token generation
    next_input_ids = torch.tensor([[prompt_tokens[-1]]], dtype=torch.long, device=TEST_DEVICE)
    hidden_states2 = model2.model.embed_tokens(next_input_ids)
    normalized_states2 = layer2.input_layernorm(hidden_states2)
    
    print(f"  Hidden states shape: {hidden_states2.shape}")
    print(f"  Normalized states shape: {normalized_states2.shape}")
    print(f"  Normalized states norm: {torch.norm(normalized_states2).item():.6f}")
    
    # Check if the embeddings for the same token are similar
    last_token_id = prompt_tokens[-1]
    embed1 = model1.model.embed_tokens(torch.tensor([[last_token_id]], device=TEST_DEVICE))
    embed2 = model2.model.embed_tokens(torch.tensor([[last_token_id]], device=TEST_DEVICE))
    embed_diff = torch.norm(embed1 - embed2).item()
    print(f"  Embedding difference for token {last_token_id}: {embed_diff:.8f}")
    
    # Check the KV cache content
    kv_cache = model2.model.kv_cache_0
    key_cache = kv_cache[layer_idx, :, :len(prompt_tokens), :]  # Keys for this layer
    value_cache = kv_cache[layer_idx + config.num_hidden_layers, :, :len(prompt_tokens), :]  # Values
    
    print(f"  KV cache shapes - Key: {key_cache.shape}, Value: {value_cache.shape}")
    print(f"  KV cache norms - Key: {torch.norm(key_cache).item():.6f}, Value: {torch.norm(value_cache).item():.6f}")
    
    # Check if we're getting non-zero cache values
    key_nonzero = torch.count_nonzero(key_cache).item()
    value_nonzero = torch.count_nonzero(value_cache).item()
    print(f"  Non-zero cache elements - Key: {key_nonzero}, Value: {value_nonzero}")

if __name__ == "__main__":
    success = test_attention_comparison()
    if success:
        print(f"\n🎉 Attention computation matches!")
    else:
        print(f"\n❌ Attention computation differs - need to debug further") 
#!/usr/bin/env python3
"""Simple debug: Compare attention computation between no-cache vs KV cache."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))
from anemll.models.qwen_model import *
import glob
from transformers import AutoTokenizer

def debug_simple_attention():
    """Compare attention step by step."""
    
    print("🔍 Simple Attention Debug: No Cache vs KV Cache")
    print("=" * 60)
    
    # Setup
    model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Simple test case: just predict next token after "What"
    prompt = "What"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids.to(TEST_DEVICE)
    tokens = input_ids[0].tolist()
    
    print(f"Prompt: '{prompt}' → tokens: {tokens}")
    print(f"Token meaning: {[tokenizer.decode([t]) for t in tokens]}")
    
    # === Method 1: No KV Cache ===
    print(f"\n🟢 Method 1: No Cache")
    print("-" * 30)
    
    model1 = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=True)
    model1.load_pretrained_weights(model_path)
    model1.eval()
    
    # Add dummy token for prediction
    next_pos = len(tokens)
    extended_tokens = tokens + [tokenizer.pad_token_id]
    extended_input_ids = torch.tensor([extended_tokens], dtype=torch.long, device=TEST_DEVICE)
    position_ids = torch.arange(len(extended_tokens), dtype=torch.long, device=TEST_DEVICE)
    current_pos = torch.tensor([next_pos], dtype=torch.long, device=TEST_DEVICE)
    
    # Create causal mask
    seq_len = len(extended_tokens)
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
    print(f"Result: {next_token1} ('{tokenizer.decode([next_token1])}')")
    print(f"Logits norm: {torch.norm(logits1).item():.3f}")
    print(f"Logits range: [{logits1.min().item():.3f}, {logits1.max().item():.3f}]")
    
    # === Method 2: KV Cache (Single Token Generation) ===
    print(f"\n🔴 Method 2: KV Cache")
    print("-" * 30)
    
    model2 = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model2.load_pretrained_weights(model_path)
    model2.eval()
    model2.model.kv_cache_0.zero_()
    
    # Process the token to fill KV cache
    token_input_ids = torch.tensor([tokens], dtype=torch.long, device=TEST_DEVICE)
    token_position_ids = torch.arange(len(tokens), dtype=torch.long, device=TEST_DEVICE)
    token_current_pos = torch.tensor([0], dtype=torch.long, device=TEST_DEVICE)  # Process position 0
    token_causal_mask = torch.zeros((1, 1, 1, 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    # Process each token individually
    for pos in range(len(tokens)):
        single_token = torch.tensor([[tokens[pos]]], dtype=torch.long, device=TEST_DEVICE)
        single_position = torch.tensor([pos], dtype=torch.long, device=TEST_DEVICE) 
        single_current_pos = torch.tensor([pos], dtype=torch.long, device=TEST_DEVICE)
        single_causal_mask = torch.zeros((1, 1, 1, 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        
        with torch.no_grad():
            _ = model2(
                input_ids=single_token,
                update_mask=torch.ones((1, 1, 512, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
                position_ids=single_position,
                causal_mask=single_causal_mask,
                current_pos=single_current_pos,
                IN_PREFILL=False
            )
    
    # Now predict next token
    next_token_id = torch.tensor([[tokens[-1]]], dtype=torch.long, device=TEST_DEVICE)  # Last token
    next_position_ids = torch.tensor([next_pos], dtype=torch.long, device=TEST_DEVICE)
    next_current_pos = torch.tensor([next_pos], dtype=torch.long, device=TEST_DEVICE)
    next_causal_mask = torch.zeros((1, 1, 1, 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    with torch.no_grad():
        logits2 = model2(
            input_ids=next_token_id,
            update_mask=torch.ones((1, 1, 512, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            position_ids=next_position_ids,
            causal_mask=next_causal_mask,
            current_pos=next_current_pos,
            IN_PREFILL=False
        )
    
    next_token2 = torch.argmax(logits2[0, 0, :]).item()
    print(f"Result: {next_token2} ('{tokenizer.decode([next_token2])}')")
    print(f"Logits norm: {torch.norm(logits2).item():.3f}")
    print(f"Logits range: [{logits2.min().item():.3f}, {logits2.max().item():.3f}]")
    
    # === Compare ===
    print(f"\n📊 COMPARISON")
    print("-" * 30)
    print(f"No cache:  {next_token1} ('{tokenizer.decode([next_token1])}')")
    print(f"KV cache:  {next_token2} ('{tokenizer.decode([next_token2])}')")
    
    if next_token1 == next_token2:
        print(f"✅ MATCH!")
    else:
        print(f"❌ MISMATCH")
        
        # Compare logits
        logits_diff = torch.norm(logits1 - logits2).item()
        print(f"Logits difference: {logits_diff:.3f}")
        
        # Check if the KV cache has reasonable values
        kv_cache = model2.model.kv_cache_0
        cache_norm = torch.norm(kv_cache).item()
        cache_nonzero = torch.count_nonzero(kv_cache).item()
        print(f"KV cache norm: {cache_norm:.3f}")
        print(f"KV cache non-zero elements: {cache_nonzero}")
        
        # Show some cache values
        layer0_key = kv_cache[0, :, :len(tokens), :]  # Layer 0 keys
        layer0_value = kv_cache[28, :, :len(tokens), :]  # Layer 0 values (offset by num_layers)
        print(f"Layer 0 key cache shape: {layer0_key.shape}, norm: {torch.norm(layer0_key).item():.3f}")
        print(f"Layer 0 value cache shape: {layer0_value.shape}, norm: {torch.norm(layer0_value).item():.3f}")

if __name__ == "__main__":
    from anemll.models.qwen_model import TEST_DEVICE, MODEL_DTYPE
    debug_simple_attention() 
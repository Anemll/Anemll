#!/usr/bin/env python3
"""Test single-token-at-a-time prefill vs batch prefill to isolate KV cache issues."""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import glob

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

from anemll.models.qwen_model import *

def test_single_token_prefill_vs_batch():
    """Compare single-token-at-a-time prefill with batch prefill."""
    
    print("🔍 Testing Single Token Prefill vs Batch Prefill")
    print("=" * 60)
    
    # Find model path
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    if not model_dirs:
        print("❌ Error: Qwen model not found in cache")
        return False
    
    model_dir = model_dirs[0]
    
    # Load config and create model
    config_path = os.path.join(model_dir, "config.json")
    config = QwenConfig.from_json(config_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    
    # Test prompt
    prompt = "The capital of France is"
    print(f"Prompt: '{prompt}'")
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids.to(TEST_DEVICE)
    prompt_tokens = input_ids[0].tolist()
    
    print(f"Tokenized to {len(prompt_tokens)} tokens: {prompt_tokens}")
    print(f"Token meanings: {[tokenizer.decode([t]) for t in prompt_tokens]}")
    
    # Test 1: Single-token-at-a-time prefill
    print(f"\n🟡 Method 1: Single-Token-at-a-Time Prefill")
    print("-" * 50)
    
    model1 = QwenForCausalLM(config, enable_coreml=False)
    model1.load_pretrained_weights(model_dir)
    model1.eval()
    
    # Reset cache
    model1.model.kv_cache_0.zero_()
    
    # Prefill one token at a time
    for i, token in enumerate(prompt_tokens):
        print(f"  Prefilling token {i}: {token} ('{tokenizer.decode([token])}')")
        
        # Single token input
        single_input_ids = torch.tensor([[token]], dtype=torch.long, device=TEST_DEVICE)
        position_ids = torch.tensor([i], dtype=torch.long, device=TEST_DEVICE)
        current_pos = torch.tensor([i], dtype=torch.long, device=TEST_DEVICE)
        update_mask = torch.ones((1, 1, config.context_length, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        causal_mask = torch.zeros((1, 1, 1, config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        
        # Forward pass to update KV cache
        with torch.no_grad():
            _ = model1(
                input_ids=single_input_ids,
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False
            )
    
    # Generate next token using single-token prefilled cache
    print(f"  Generating next token...")
    next_position = len(prompt_tokens)
    last_token = prompt_tokens[-1]
    
    next_input_ids = torch.tensor([[last_token]], dtype=torch.long, device=TEST_DEVICE)
    next_position_ids = torch.tensor([next_position], dtype=torch.long, device=TEST_DEVICE)
    next_current_pos = torch.tensor([next_position], dtype=torch.long, device=TEST_DEVICE)
    
    with torch.no_grad():
        logits1 = model1(
            input_ids=next_input_ids,
            update_mask=update_mask,
            position_ids=next_position_ids,
            causal_mask=causal_mask,
            current_pos=next_current_pos,
            IN_PREFILL=False
        )
    
    next_token1 = torch.argmax(logits1[0, 0, :]).item()
    print(f"  Single-token prefill → Next token: {next_token1} ('{tokenizer.decode([next_token1])}')")
    
    # Test 2: Batch prefill (our current implementation)
    print(f"\n🟠 Method 2: Batch Prefill")
    print("-" * 50)
    
    model2 = QwenForCausalLM(config, enable_coreml=False)
    model2.load_pretrained_weights(model_dir)
    model2.eval()
    
    # Reset cache
    model2.model.kv_cache_0.zero_()
    
    # Batch prefill
    position_ids = torch.arange(len(prompt_tokens), dtype=torch.long, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, len(prompt_tokens), config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    print(f"  Batch prefilling {len(prompt_tokens)} tokens...")
    model2.prefill_kv_cache(
        input_ids=input_ids,
        position_ids=position_ids,
        start_pos=0,
        causal_mask=causal_mask
    )
    
    # Generate next token using batch-prefilled cache
    print(f"  Generating next token...")
    with torch.no_grad():
        logits2 = model2(
            input_ids=next_input_ids,
            update_mask=update_mask,
            position_ids=next_position_ids,
            causal_mask=causal_mask,
            current_pos=next_current_pos,
            IN_PREFILL=False
        )
    
    next_token2 = torch.argmax(logits2[0, 0, :]).item()
    print(f"  Batch prefill → Next token: {next_token2} ('{tokenizer.decode([next_token2])}')")
    
    # Test 3: Original Hugging Face model
    print(f"\n🟢 Method 3: Original Hugging Face Model")
    print("-" * 50)
    
    original_model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.float16
    )
    original_model.eval()
    
    with torch.no_grad():
        original_generated = original_model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    original_next_token = original_generated[0][len(prompt_tokens)].item()
    print(f"  Original HF model → Next token: {original_next_token} ('{tokenizer.decode([original_next_token])}')")
    
    # Compare results
    print(f"\n📊 COMPARISON")
    print("=" * 50)
    print(f"Single-token prefill: {next_token1} ('{tokenizer.decode([next_token1])}')")
    print(f"Batch prefill:        {next_token2} ('{tokenizer.decode([next_token2])}')")
    print(f"Original HF model:    {original_next_token} ('{tokenizer.decode([original_next_token])}')")
    
    single_matches_hf = (next_token1 == original_next_token)
    batch_matches_hf = (next_token2 == original_next_token)
    single_matches_batch = (next_token1 == next_token2)
    
    print(f"\nMatches:")
    print(f"  Single-token vs HF: {'✅' if single_matches_hf else '❌'}")
    print(f"  Batch vs HF:        {'✅' if batch_matches_hf else '❌'}")
    print(f"  Single vs Batch:    {'✅' if single_matches_batch else '❌'}")
    
    if single_matches_hf and batch_matches_hf:
        print(f"\n🎉 SUCCESS: Both prefill methods match original HF model!")
        return True
    elif single_matches_hf and not batch_matches_hf:
        print(f"\n⚠️  Issue in BATCH PREFILL: Single-token prefill works but batch doesn't")
        return False
    elif not single_matches_hf and batch_matches_hf:
        print(f"\n⚠️  Issue in SINGLE-TOKEN PREFILL: Batch prefill works but single-token doesn't")
        return False
    else:
        print(f"\n❌ Both prefill methods differ from original HF model")
        return False

def test_kv_cache_content_comparison():
    """Compare KV cache contents between single-token and batch prefill."""
    
    print(f"\n🔍 Testing KV Cache Content Comparison")
    print("=" * 60)
    
    # Find model path
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    model_dir = model_dirs[0]
    
    # Load config
    config_path = os.path.join(model_dir, "config.json")
    config = QwenConfig.from_json(config_path)
    
    # Test tokens
    test_tokens = [100, 200, 300]  # Simple tokens
    
    # Method 1: Single-token prefill
    model1 = QwenForCausalLM(config, enable_coreml=False)
    model1.load_pretrained_weights(model_dir)
    model1.eval()
    model1.model.kv_cache_0.zero_()
    
    for i, token in enumerate(test_tokens):
        single_input_ids = torch.tensor([[token]], dtype=torch.long, device=TEST_DEVICE)
        position_ids = torch.tensor([i], dtype=torch.long, device=TEST_DEVICE)
        current_pos = torch.tensor([i], dtype=torch.long, device=TEST_DEVICE)
        update_mask = torch.ones((1, 1, config.context_length, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        causal_mask = torch.zeros((1, 1, 1, config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        
        with torch.no_grad():
            _ = model1(
                input_ids=single_input_ids,
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False
            )
    
    cache1 = model1.model.kv_cache_0.clone()
    
    # Method 2: Batch prefill
    model2 = QwenForCausalLM(config, enable_coreml=False)
    model2.load_pretrained_weights(model_dir)
    model2.eval()
    model2.model.kv_cache_0.zero_()
    
    batch_input_ids = torch.tensor([test_tokens], dtype=torch.long, device=TEST_DEVICE)
    position_ids = torch.arange(len(test_tokens), dtype=torch.long, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, len(test_tokens), config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    model2.prefill_kv_cache(
        input_ids=batch_input_ids,
        position_ids=position_ids,
        start_pos=0,
        causal_mask=causal_mask
    )
    
    cache2 = model2.model.kv_cache_0.clone()
    
    # Compare caches
    cache_diff = torch.sum(torch.abs(cache1 - cache2)).item()
    cache_max_diff = torch.max(torch.abs(cache1 - cache2)).item()
    
    print(f"KV Cache comparison:")
    print(f"  Total absolute difference: {cache_diff}")
    print(f"  Maximum absolute difference: {cache_max_diff}")
    
    if cache_diff < 1e-4:
        print(f"✅ KV caches are nearly identical (diff < 1e-4)")
        return True
    else:
        print(f"❌ KV caches differ significantly")
        
        # Show where the differences are
        for layer_idx in range(config.num_hidden_layers):
            key_idx = layer_idx
            value_idx = layer_idx + config.num_hidden_layers
            
            key_diff = torch.sum(torch.abs(cache1[key_idx] - cache2[key_idx])).item()
            value_diff = torch.sum(torch.abs(cache1[value_idx] - cache2[value_idx])).item()
            
            if key_diff > 1e-4 or value_diff > 1e-4:
                print(f"  Layer {layer_idx}: Key diff = {key_diff:.6f}, Value diff = {value_diff:.6f}")
        
        return False

if __name__ == "__main__":
    print("🔍 Debugging KV Cache Prefill Methods")
    print("=" * 70)
    
    # Test 1: Compare prefill methods
    test1_result = test_single_token_prefill_vs_batch()
    
    # Test 2: Compare KV cache contents
    test2_result = test_kv_cache_content_comparison()
    
    if test1_result and test2_result:
        print(f"\n🎉 ALL DEBUGGING TESTS PASSED!")
        print(f"✅ Both prefill methods work correctly")
        print(f"✅ KV cache contents are consistent")
    else:
        print(f"\n🔍 Issues found - debugging needed")
        if not test1_result:
            print(f"❌ Prefill method comparison failed")
        if not test2_result:
            print(f"❌ KV cache content comparison failed") 
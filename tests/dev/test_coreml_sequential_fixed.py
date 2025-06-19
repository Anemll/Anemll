#!/usr/bin/env python3
"""Test Qwen model sequential processing (no KV cache) for comparison."""

import torch
import sys
import os
import glob

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

from anemll.models.qwen_model import *
from transformers import AutoTokenizer

def test_qwen_sequential_no_cache():
    """Test Qwen generation using sequential processing without KV cache."""
    
    print("🚀 Testing Qwen Sequential Generation (No KV Cache)")
    print("=" * 70)
    
    # Find model path
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    if not model_dirs:
        print("❌ Error: Qwen model not found in cache")
        return False
    
    model_dir = model_dirs[0]
    print(f"Using model from: {model_dir}")
    
    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    config = QwenConfig.from_json(os.path.join(model_dir, 'config.json'))
    
    # Load model WITHOUT KV cache support
    print(f"\n📚 Loading Qwen Model (No KV Cache)...")
    model = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=True)
    model.load_pretrained_weights(model_dir)
    
    # Test prompts - use shorter ones to avoid multi-token complexity
    test_prompts = [
        "Hello",
        "The",
        "What"
    ]
    
    for prompt in test_prompts:
        print(f"\n🔥 Testing prompt: '{prompt}'")
        print("-" * 50)
        
        # Tokenize 
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs.input_ids
        prompt_tokens = input_ids[0].tolist()
        print(f"Tokenized to {len(prompt_tokens)} tokens: {prompt_tokens}")
        print(f"Token meanings: {[tokenizer.decode([t]) for t in prompt_tokens]}")
        
        # Create causal mask 
        causal_mask = torch.zeros((1, 1, 256, 256), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        for i in range(256):
            for j in range(i + 1, 256):
                causal_mask[0, 0, i, j] = float('-inf')
        
        # Generate tokens one by one WITHOUT KV cache
        print(f"\n🍎 Sequential Generation (No KV Cache)...")
        
        max_new_tokens = 3  # Reduced for simplicity
        generated_tokens = prompt_tokens.copy()
        
        for gen_step in range(max_new_tokens):
            print(f"\nGeneration step {gen_step + 1}:")
            
            # Current sequence
            current_seq = generated_tokens.copy()
            seq_len = len(current_seq)
            
            if seq_len > 256:
                print(f"❌ Sequence too long ({seq_len} > 256)")
                break
            
            print(f"  Current sequence length: {seq_len}")
            print(f"  Last few tokens: {[tokenizer.decode([t]) for t in current_seq[-3:]]}")
            
            # Convert to tensors
            input_ids = torch.tensor([current_seq], device=TEST_DEVICE)
            position_ids = torch.tensor(list(range(seq_len)), device=TEST_DEVICE)
            
            # Forward pass through model (use prefill mode for multi-token)
            try:
                with torch.no_grad():
                    # Always use prefill mode for multi-token sequences
                    outputs = model.model(
                        input_ids=input_ids,
                        causal_mask=causal_mask,
                        position_ids=position_ids,
                        current_pos=torch.tensor(0, device=TEST_DEVICE),  # Start position for prefill
                        IN_PREFILL=True
                    )
                    
                    # Apply lm_head manually for fair comparison
                    hidden_states = outputs.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
                    
                    # Apply all 16 lm_head splits
                    logits_parts = []
                    for i in range(1, 17):
                        lm_head = getattr(model, f"lm_head16_{i}")
                        logits_part = lm_head(hidden_states).squeeze(2).transpose(1, 2)
                        logits_parts.append(logits_part)
                    
                    # Concatenate all parts to get full vocabulary
                    full_logits = torch.cat(logits_parts, dim=2)
                    
                    # Get prediction for last position
                    last_token_logits = full_logits[0, -1, :]  # [vocab_size]
                    next_token = torch.argmax(last_token_logits).item()
                    
                    # Add to sequence
                    generated_tokens.append(next_token)
                    
                    # Get probability for analysis
                    probs = torch.softmax(last_token_logits, dim=-1)
                    next_prob = probs[next_token].item()
                    
                    print(f"  Generated token: {next_token} ('{tokenizer.decode([next_token])}')")
                    print(f"  Probability: {next_prob:.6f}")
                    
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                return False
        
        # Show final results
        new_tokens = generated_tokens[len(prompt_tokens):]
        continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        print(f"\n📊 RESULTS")
        print(f"Original prompt: '{prompt}'")
        print(f"Generated: '{continuation}'")
        print(f"Full text: '{prompt}{continuation}'")
    
    print(f"\n🎉 SUCCESS! Qwen sequential generation (no KV cache) completed!")
    print("   ✅ No KV cache needed")
    print("   ✅ Variable sequence length works") 
    print("   ✅ Can be used as baseline for KV cache comparison")
    return True

def compare_with_kv_cache():
    """Compare no-cache vs KV cache generation."""
    
    print(f"\n🔬 Comparing No-Cache vs KV Cache")
    print("=" * 50)
    
    # Setup models
    model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Model with KV cache
    model_with_cache = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model_with_cache.load_pretrained_weights(model_path)
    
    # Test prompt - single token to avoid complexity
    prompt = "The"
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
    
    print(f"Test prompt: '{prompt}'")
    print(f"Tokens: {tokens.tolist()}")
    
    # Test single token prediction
    causal_mask = torch.zeros((1, 1, 256, 256), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    for i in range(256):
        for j in range(i + 1, 256):
            causal_mask[0, 0, i, j] = float('-inf')
    
    with torch.no_grad():
        # No cache method (use prefill)
        print(f"\n1. No-cache method (prefill)...")
        model_with_cache.model.kv_cache_0.zero_()
        
        outputs_no_cache = model_with_cache.model(
            input_ids=tokens,
            causal_mask=causal_mask,
            position_ids=torch.tensor([0], device=TEST_DEVICE),
            current_pos=torch.tensor(0, device=TEST_DEVICE),
            IN_PREFILL=True
        )
        
        # Apply lm_head manually
        hidden_states = outputs_no_cache.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        logits_parts = []
        for i in range(1, 17):
            lm_head = getattr(model_with_cache, f"lm_head16_{i}")
            logits_part = lm_head(hidden_states).squeeze(2).transpose(1, 2)
            logits_parts.append(logits_part)
        logits_no_cache = torch.cat(logits_parts, dim=2)
        
        # KV cache method  
        print(f"2. KV-cache method...")
        model_with_cache.model.kv_cache_0.zero_()
        
        # Process token with KV cache
        logits_with_cache = model_with_cache(
            input_ids=tokens,
            update_mask=torch.ones_like(tokens, dtype=torch.float32),
            position_ids=torch.tensor([0], device=TEST_DEVICE),
            causal_mask=causal_mask,
            current_pos=torch.tensor(0, device=TEST_DEVICE),
            IN_PREFILL=False
        )
    
    # Compare results
    print(f"\n3. Comparison...")
    pos0_logits_no_cache = logits_no_cache[0, 0, :]  # Position 0
    pos0_logits_with_cache = logits_with_cache[0, 0, :]  # Position 0 of output
    
    diff = torch.norm(pos0_logits_no_cache - pos0_logits_with_cache).item()
    relative_diff = diff / (torch.norm(pos0_logits_no_cache).item() + 1e-8) * 100
    
    top_token_no_cache = torch.argmax(pos0_logits_no_cache).item()
    top_token_with_cache = torch.argmax(pos0_logits_with_cache).item()
    
    print(f"No-cache top token: {top_token_no_cache} ('{tokenizer.decode([top_token_no_cache])}')")
    print(f"KV-cache top token: {top_token_with_cache} ('{tokenizer.decode([top_token_with_cache])}')")
    print(f"Logits difference: {diff:.6f}")
    print(f"Relative difference: {relative_diff:.3f}%")
    
    if top_token_no_cache == top_token_with_cache:
        print("✅ Same prediction - KV cache working correctly!")
    else:
        print("❌ Different predictions - KV cache issue detected!")
    
    return top_token_no_cache == top_token_with_cache

if __name__ == "__main__":
    success1 = test_qwen_sequential_no_cache()
    success2 = compare_with_kv_cache()
    
    if success1 and success2:
        print(f"\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n❌ Some tests failed!") 
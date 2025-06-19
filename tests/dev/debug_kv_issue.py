#!/usr/bin/env python3
"""Debug the exact KV cache issue by comparing step-by-step vs prefill."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))

from anemll.models.qwen_model import *
import glob
from transformers import AutoTokenizer

def debug_kv_issue():
    """Debug the exact KV cache issue."""
    
    print("🔍 KV Cache Issue Debug")
    print("=" * 50)
    
    # Setup
    print("\n1. Setting up model...")
    model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    model = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    
    # Simple 2-token test: "The capital"
    print("\n2. Creating test case...")
    tokens = torch.tensor([[785, 6722]], device=TEST_DEVICE)  # "The capital"
    print(f"Test tokens: {tokens.tolist()}")
    print(f"Decoded: '{tokenizer.decode(tokens[0])}'")
    
    # Create causal mask
    causal_mask = torch.zeros((1, 1, 256, 256), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    for i in range(256):
        for j in range(i + 1, 256):
            causal_mask[0, 0, i, j] = float('-inf')
    
    print("\n3. Method 1: Prefill both tokens at once...")
    model.model.kv_cache_0.zero_()
    
    with torch.no_grad():
        # Use prefill mode for 2 tokens
        model.prefill_kv_cache(
            input_ids=tokens,  # [1, 2]
            position_ids=torch.tensor([0, 1], device=TEST_DEVICE),
            start_pos=0,
            causal_mask=causal_mask
        )
        
        cache_norm_prefill = torch.norm(model.model.kv_cache_0).item()
        print(f"  After prefill cache norm: {cache_norm_prefill:.3f}")
        
        # Check what's in cache at both positions
        pos0_norm = torch.norm(model.model.kv_cache_0[:, :, 0, :]).item()
        pos1_norm = torch.norm(model.model.kv_cache_0[:, :, 1, :]).item()
        print(f"  Position 0 cache norm: {pos0_norm:.3f}")
        print(f"  Position 1 cache norm: {pos1_norm:.3f}")
        
        # Now predict next token after "The capital"
        next_token_input = torch.tensor([[151643]], device=TEST_DEVICE)  # endoftext as placeholder
        current_pos = torch.tensor(2, device=TEST_DEVICE)
        position_ids = torch.tensor([2], device=TEST_DEVICE)
        update_mask = torch.ones_like(next_token_input, dtype=torch.float32)
        
        logits_prefill = model(
            input_ids=next_token_input,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
        
        probs_prefill = torch.softmax(logits_prefill[0, 0], dim=-1)
        top_token_prefill = torch.argmax(probs_prefill).item()
        top_prob_prefill = probs_prefill[top_token_prefill].item()
        
        print(f"  Prefill method prediction: {top_token_prefill} ('{tokenizer.decode([top_token_prefill])}')")
        print(f"  Prefill probability: {top_prob_prefill:.6f}")
    
    print("\n4. Method 2: Sequential token processing...")
    model.model.kv_cache_0.zero_()
    
    with torch.no_grad():
        # Process "The" (token 785) at position 0
        token0_input = torch.tensor([[785]], device=TEST_DEVICE)
        current_pos_0 = torch.tensor(0, device=TEST_DEVICE)
        position_ids_0 = torch.tensor([0], device=TEST_DEVICE)
        update_mask_0 = torch.ones_like(token0_input, dtype=torch.float32)
        
        logits_0 = model(
            input_ids=token0_input,
            update_mask=update_mask_0,
            position_ids=position_ids_0,
            causal_mask=causal_mask,
            current_pos=current_pos_0,
            IN_PREFILL=False
        )
        
        cache_norm_after_0 = torch.norm(model.model.kv_cache_0).item()
        print(f"  After token 0 cache norm: {cache_norm_after_0:.3f}")
        
        # Process " capital" (token 6722) at position 1
        token1_input = torch.tensor([[6722]], device=TEST_DEVICE) 
        current_pos_1 = torch.tensor(1, device=TEST_DEVICE)
        position_ids_1 = torch.tensor([1], device=TEST_DEVICE)
        update_mask_1 = torch.ones_like(token1_input, dtype=torch.float32)
        
        logits_1 = model(
            input_ids=token1_input,
            update_mask=update_mask_1,
            position_ids=position_ids_1,
            causal_mask=causal_mask,
            current_pos=current_pos_1,
            IN_PREFILL=False
        )
        
        cache_norm_after_1 = torch.norm(model.model.kv_cache_0).item()
        print(f"  After token 1 cache norm: {cache_norm_after_1:.3f}")
        
        # Check what's in cache at both positions
        pos0_norm = torch.norm(model.model.kv_cache_0[:, :, 0, :]).item()
        pos1_norm = torch.norm(model.model.kv_cache_0[:, :, 1, :]).item()
        print(f"  Position 0 cache norm: {pos0_norm:.3f}")
        print(f"  Position 1 cache norm: {pos1_norm:.3f}")
        
        # Now predict next token after "The capital" 
        next_token_input = torch.tensor([[151643]], device=TEST_DEVICE)  # endoftext as placeholder
        current_pos_2 = torch.tensor(2, device=TEST_DEVICE)
        position_ids_2 = torch.tensor([2], device=TEST_DEVICE)
        update_mask_2 = torch.ones_like(next_token_input, dtype=torch.float32)
        
        logits_sequential = model(
            input_ids=next_token_input,
            update_mask=update_mask_2,
            position_ids=position_ids_2,
            causal_mask=causal_mask,
            current_pos=current_pos_2,
            IN_PREFILL=False
        )
        
        probs_sequential = torch.softmax(logits_sequential[0, 0], dim=-1)
        top_token_sequential = torch.argmax(probs_sequential).item()
        top_prob_sequential = probs_sequential[top_token_sequential].item()
        
        print(f"  Sequential method prediction: {top_token_sequential} ('{tokenizer.decode([top_token_sequential])}')")
        print(f"  Sequential probability: {top_prob_sequential:.6f}")
    
    print("\n5. Comparing results...")
    print(f"Prefill method: {top_token_prefill} ('{tokenizer.decode([top_token_prefill])}') - {top_prob_prefill:.6f}")
    print(f"Sequential method: {top_token_sequential} ('{tokenizer.decode([top_token_sequential])}') - {top_prob_sequential:.6f}")
    
    if top_token_prefill == top_token_sequential:
        print("✅ Both methods produce same prediction - KV cache is working correctly!")
    else:
        print("❌ Different predictions - this shows the KV cache issue!")
        
        # Calculate logits difference
        logits_diff = torch.norm(logits_prefill - logits_sequential).item()
        relative_diff = logits_diff / (torch.norm(logits_prefill).item() + 1e-8) * 100
        print(f"Logits difference norm: {logits_diff:.6f}")
        print(f"Relative difference: {relative_diff:.3f}%")
    
    print("\n" + "=" * 50)
    print("✅ KV cache issue debug complete!")

if __name__ == "__main__":
    debug_kv_issue() 
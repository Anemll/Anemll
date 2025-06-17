#!/usr/bin/env python3
"""Fair comparison: Both methods process the exact same sequence."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))
from anemll.models.qwen_model import *
import glob
from transformers import AutoTokenizer

def test_fair_comparison():
    """Fair comparison with identical input sequences."""
    
    print("⚖️ Fair Comparison: Identical Input Sequences")
    print("=" * 60)
    
    # Setup
    model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Use a 2-token sequence to make it clearer
    prompt = "The capital"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids.to(TEST_DEVICE)
    tokens = input_ids[0].tolist()
    
    print(f"Prompt: '{prompt}' → tokens: {tokens}")
    print(f"Token meanings: {[tokenizer.decode([t]) for t in tokens]}")
    
    # The key insight: Both methods should process the SAME token at the next position
    # and should produce the SAME result
    next_pos = len(tokens)
    next_token_for_prediction = tokenizer.pad_token_id  # Use same token for both
    
    print(f"Both methods will process token {next_token_for_prediction} ('{tokenizer.decode([next_token_for_prediction])}') at position {next_pos}")
    
    # === Method 1: No KV Cache ===
    print(f"\n🟢 Method 1: No KV Cache")
    print("-" * 40)
    
    model_no_cache = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=True)
    model_no_cache.load_pretrained_weights(model_path)
    model_no_cache.eval()
    
    # Process full sequence: [token1, token2, next_token_for_prediction]
    full_sequence = tokens + [next_token_for_prediction]
    full_input_ids = torch.tensor([full_sequence], dtype=torch.long, device=TEST_DEVICE)
    position_ids = torch.arange(len(full_sequence), dtype=torch.long, device=TEST_DEVICE)
    current_pos = torch.tensor([next_pos], dtype=torch.long, device=TEST_DEVICE)
    
    # Create causal mask
    seq_len = len(full_sequence)
    causal_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            causal_mask[0, 0, i, j] = float('-inf')
    
    print(f"Processing full sequence: {full_sequence}")
    print(f"Token meanings: {[tokenizer.decode([t]) for t in full_sequence]}")
    
    with torch.no_grad():
        logits1 = model_no_cache(
            input_ids=full_input_ids,
            update_mask=torch.ones((1, 1, 512, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
    
    next_token1 = torch.argmax(logits1[0, 0, :]).item()
    print(f"Result: {next_token1} ('{tokenizer.decode([next_token1])}')")
    print(f"Logits norm: {torch.norm(logits1).item():.3f}")
    
    # === Method 2: KV Cache ===
    print(f"\n🔴 Method 2: KV Cache")
    print("-" * 40)
    
    model_kv_cache = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model_kv_cache.load_pretrained_weights(model_path)
    model_kv_cache.eval()
    model_kv_cache.model.kv_cache_0.zero_()
    
    # Step 1: Fill KV cache with context tokens (same as before)
    print(f"Step 1: Processing context tokens {tokens}")
    for pos, token_id in enumerate(tokens):
        single_token = torch.tensor([[token_id]], dtype=torch.long, device=TEST_DEVICE)
        single_position = torch.tensor([pos], dtype=torch.long, device=TEST_DEVICE)
        single_current_pos = torch.tensor([pos], dtype=torch.long, device=TEST_DEVICE)
        single_causal_mask = torch.zeros((1, 1, 1, 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        
        with torch.no_grad():
            _ = model_kv_cache(
                input_ids=single_token,
                update_mask=torch.ones((1, 1, 512, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
                position_ids=single_position,
                causal_mask=single_causal_mask,
                current_pos=single_current_pos,
                IN_PREFILL=False
            )
        print(f"  Processed token {token_id} ('{tokenizer.decode([token_id])}') at position {pos}")
    
    # Step 2: Process the SAME token that the no-cache version processed
    print(f"Step 2: Processing token {next_token_for_prediction} at position {next_pos}")
    
    next_input_ids = torch.tensor([[next_token_for_prediction]], dtype=torch.long, device=TEST_DEVICE)
    next_position_ids = torch.tensor([next_pos], dtype=torch.long, device=TEST_DEVICE)
    next_current_pos = torch.tensor([next_pos], dtype=torch.long, device=TEST_DEVICE)
    next_causal_mask = torch.zeros((1, 1, 1, 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    with torch.no_grad():
        logits2 = model_kv_cache(
            input_ids=next_input_ids,
            update_mask=torch.ones((1, 1, 512, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            position_ids=next_position_ids,
            causal_mask=next_causal_mask,
            current_pos=next_current_pos,
            IN_PREFILL=False
        )
    
    next_token2 = torch.argmax(logits2[0, 0, :]).item()
    print(f"Result: {next_token2} ('{tokenizer.decode([next_token2])}')")
    print(f"Logits norm: {torch.norm(logits2).item():.3f}")
    
    # === Compare ===
    print(f"\n📊 FAIR COMPARISON")
    print("-" * 40)
    print(f"Both methods processed: {[tokenizer.decode([t]) for t in full_sequence]}")
    print(f"No cache result:  {next_token1} ('{tokenizer.decode([next_token1])}')")
    print(f"KV cache result:  {next_token2} ('{tokenizer.decode([next_token2])}')")
    
    if next_token1 == next_token2:
        print(f"✅ PERFECT MATCH! KV cache working correctly!")
        return True
    else:
        print(f"❌ Still different")
        
        # Compare logits more carefully
        logits_diff = torch.norm(logits1 - logits2).item()
        logits_relative_diff = logits_diff / torch.norm(logits1).item()
        print(f"Logits absolute difference: {logits_diff:.3f}")
        print(f"Logits relative difference: {logits_relative_diff:.6f} ({logits_relative_diff*100:.3f}%)")
        
        # Check if they're close but not identical
        top5_no_cache = torch.topk(logits1[0, 0, :], 5)
        top5_kv_cache = torch.topk(logits2[0, 0, :], 5)
        
        print(f"\nTop 5 predictions comparison:")
        print(f"No cache:")
        for i, (token_id, logit) in enumerate(zip(top5_no_cache.indices, top5_no_cache.values)):
            print(f"  {i+1}. {token_id.item()}: '{tokenizer.decode([token_id.item()])}' ({logit.item():.3f})")
            
        print(f"KV cache:")
        for i, (token_id, logit) in enumerate(zip(top5_kv_cache.indices, top5_kv_cache.values)):
            print(f"  {i+1}. {token_id.item()}: '{tokenizer.decode([token_id.item()])}' ({logit.item():.3f})")
        
        # Check if prediction ranks are close
        no_cache_rank = None
        kv_cache_rank = None
        for i, token_id in enumerate(top5_no_cache.indices):
            if token_id.item() == next_token2:
                no_cache_rank = i + 1
                break
        for i, token_id in enumerate(top5_kv_cache.indices):
            if token_id.item() == next_token1:
                kv_cache_rank = i + 1
                break
                
        if no_cache_rank:
            print(f"\nKV cache prediction ('{tokenizer.decode([next_token2])}') ranks #{no_cache_rank} in no-cache predictions")
        if kv_cache_rank:
            print(f"No-cache prediction ('{tokenizer.decode([next_token1])}') ranks #{kv_cache_rank} in KV cache predictions")
        
        return False

if __name__ == "__main__":
    from anemll.models.qwen_model import TEST_DEVICE, MODEL_DTYPE
    test_fair_comparison() 
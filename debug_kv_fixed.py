#!/usr/bin/env python3
"""Test corrected KV cache approach - predict from full context."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))
from anemll.models.qwen_model import *
import glob
from transformers import AutoTokenizer

def test_corrected_kv_cache():
    """Test corrected KV cache approach."""
    
    print("🔧 Corrected KV Cache Test")
    print("=" * 50)
    
    # Setup
    model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    prompt = "What"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids.to(TEST_DEVICE)
    tokens = input_ids[0].tolist()
    
    print(f"Prompt: '{prompt}' → tokens: {tokens}")
    
    # === Baseline: No KV Cache ===
    print(f"\n🟢 Baseline: No KV Cache")
    model_no_cache = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=True)
    model_no_cache.load_pretrained_weights(model_path)
    model_no_cache.eval()
    
    # For no-cache, we process the full context + dummy token
    next_pos = len(tokens)
    extended_tokens = tokens + [tokenizer.pad_token_id]
    extended_input_ids = torch.tensor([extended_tokens], dtype=torch.long, device=TEST_DEVICE)
    position_ids = torch.arange(len(extended_tokens), dtype=torch.long, device=TEST_DEVICE)
    current_pos = torch.tensor([next_pos], dtype=torch.long, device=TEST_DEVICE)
    
    seq_len = len(extended_tokens)
    causal_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            causal_mask[0, 0, i, j] = float('-inf')
    
    with torch.no_grad():
        logits1 = model_no_cache(
            input_ids=extended_input_ids,
            update_mask=torch.ones((1, 1, 512, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
    
    next_token1 = torch.argmax(logits1[0, 0, :]).item()
    print(f"Result: {next_token1} ('{tokenizer.decode([next_token1])}')")
    
    # === Corrected KV Cache Approach ===
    print(f"\n🔧 Corrected: KV Cache with Proper Context")
    model_kv_cache = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model_kv_cache.load_pretrained_weights(model_path)
    model_kv_cache.eval()
    model_kv_cache.model.kv_cache_0.zero_()
    
    # Step 1: Fill KV cache with context tokens
    print(f"  Step 1: Processing context tokens {tokens}")
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
        print(f"    Processed token {token_id} ('{tokenizer.decode([token_id])}') at position {pos}")
    
    # Step 2: Predict next token using a placeholder token
    # The key insight: we need to provide SOME token for the model to process at the next position
    # But the actual content doesn't matter - the model should attend to the cached context
    print(f"  Step 2: Predicting next token at position {len(tokens)}")
    
    # Use a neutral token (e.g., pad token) as the "current" token at the next position
    placeholder_token = tokenizer.pad_token_id
    next_input_ids = torch.tensor([[placeholder_token]], dtype=torch.long, device=TEST_DEVICE)
    next_position_ids = torch.tensor([len(tokens)], dtype=torch.long, device=TEST_DEVICE)
    next_current_pos = torch.tensor([len(tokens)], dtype=torch.long, device=TEST_DEVICE)
    next_causal_mask = torch.zeros((1, 1, 1, 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    print(f"    Using placeholder token: {placeholder_token} ('{tokenizer.decode([placeholder_token])}')")
    
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
    
    # === Compare ===
    print(f"\n📊 COMPARISON")
    print("-" * 30)
    print(f"No cache:      {next_token1} ('{tokenizer.decode([next_token1])}')")
    print(f"Corrected KV:  {next_token2} ('{tokenizer.decode([next_token2])}')")
    
    if next_token1 == next_token2:
        print(f"✅ MATCH! KV cache working correctly!")
        return True
    else:
        print(f"❌ Still different")
        
        # Compare logits
        logits_diff = torch.norm(logits1 - logits2).item()
        print(f"Logits difference: {logits_diff:.3f}")
        
        # Show top predictions
        print(f"\nTop 3 predictions:")
        top3_no_cache = torch.topk(logits1[0, 0, :], 3)
        top3_kv_cache = torch.topk(logits2[0, 0, :], 3)
        
        print(f"No cache:")
        for i, (token_id, logit) in enumerate(zip(top3_no_cache.indices, top3_no_cache.values)):
            print(f"  {i+1}. {token_id.item()}: '{tokenizer.decode([token_id.item()])}' ({logit.item():.3f})")
            
        print(f"KV cache:")
        for i, (token_id, logit) in enumerate(zip(top3_kv_cache.indices, top3_kv_cache.values)):
            print(f"  {i+1}. {token_id.item()}: '{tokenizer.decode([token_id.item()])}' ({logit.item():.3f})")
        
        return False

if __name__ == "__main__":
    from anemll.models.qwen_model import TEST_DEVICE, MODEL_DTYPE
    test_corrected_kv_cache() 
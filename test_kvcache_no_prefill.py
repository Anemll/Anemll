#!/usr/bin/env python3
"""Test KV cache without prefill - single token generation only."""

import numpy as np
import torch
import os
from transformers import AutoTokenizer
from anemll.models.qwen_model import QwenForCausalLM, QwenConfig
import warnings
warnings.filterwarnings('ignore')

def test_kvcache_without_prefill():
    print("🔬 KV Cache Test - No Prefill (Single Token Generation Only)")
    print("=" * 80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    
    # Load PyTorch Qwen model
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    
    # Test prompts
    test_prompts = [
        "What",
        "The capital", 
        "Hello",
        "Python is"
    ]
    
    for prompt_text in test_prompts:
        print(f"\n📝 Testing prompt: '{prompt_text}'")
        print("-" * 60)
        
        # === Method 1: Disable KV Cache (Known Working) ===
        print(f"🟢 Method 1: Disable KV Cache (baseline)")
        
        model_no_cache = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=True)
        model_no_cache.load_pretrained_weights(model_path)
        model_no_cache.eval()
        
        # Tokenize prompt
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs.input_ids.to(TEST_DEVICE)
        prompt_tokens = input_ids[0].tolist()
        
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
            logits1 = model_no_cache(
                input_ids=extended_input_ids,
                update_mask=torch.ones((1, 1, 512, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False
            )
        
        next_token1 = torch.argmax(logits1[0, 0, :]).item()
        print(f"  Result: {next_token1} ('{tokenizer.decode([next_token1])}')")
        
        # === Method 2: KV Cache WITHOUT Prefill ===
        print(f"🔴 Method 2: KV Cache WITHOUT Prefill (single token generation)")
        
        model_kv_cache = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
        model_kv_cache.load_pretrained_weights(model_path)
        model_kv_cache.eval()
        model_kv_cache.model.kv_cache_0.zero_()  # Start with clean cache
        
        # Generate tokens one by one, building up KV cache incrementally
        current_tokens = prompt_tokens.copy()
        
        for pos in range(len(prompt_tokens)):
            # Process one token at a time
            token_id = current_tokens[pos]
            single_input_ids = torch.tensor([[token_id]], dtype=torch.long, device=TEST_DEVICE)
            single_position_ids = torch.tensor([pos], dtype=torch.long, device=TEST_DEVICE)
            single_current_pos = torch.tensor([pos], dtype=torch.long, device=TEST_DEVICE)
            single_causal_mask = torch.zeros((1, 1, 1, 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
            
            with torch.no_grad():
                _ = model_kv_cache(
                    input_ids=single_input_ids,
                    update_mask=torch.ones((1, 1, 512, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
                    position_ids=single_position_ids,
                    causal_mask=single_causal_mask,
                    current_pos=single_current_pos,
                    IN_PREFILL=False
                )
        
        # Now generate next token with filled KV cache
        next_pos = len(prompt_tokens)
        next_input_ids = torch.tensor([[prompt_tokens[-1]]], dtype=torch.long, device=TEST_DEVICE)  # Use last token
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
        print(f"  Result: {next_token2} ('{tokenizer.decode([next_token2])}')")
        
        # === Compare Results ===
        print(f"📊 Comparison:")
        print(f"  No KV cache:    {next_token1} ('{tokenizer.decode([next_token1])}')")
        print(f"  KV cache (no prefill): {next_token2} ('{tokenizer.decode([next_token2])}')")
        
        if next_token1 == next_token2:
            print(f"  ✅ MATCH - KV cache working correctly!")
        else:
            print(f"  ❌ MISMATCH - Issue in KV cache single token generation")
            
            # Show top 5 predictions for both
            top5_no_cache = torch.topk(logits1[0, 0, :], 5)
            top5_kv_cache = torch.topk(logits2[0, 0, :], 5)
            
            print(f"  Top 5 - No cache:")
            for i, (token_id, logit) in enumerate(zip(top5_no_cache.indices, top5_no_cache.values)):
                print(f"    {i+1}. {token_id.item()}: '{tokenizer.decode([token_id.item()])}' (logit: {logit.item():.3f})")
                
            print(f"  Top 5 - KV cache:")
            for i, (token_id, logit) in enumerate(zip(top5_kv_cache.indices, top5_kv_cache.values)):
                print(f"    {i+1}. {token_id.item()}: '{tokenizer.decode([token_id.item()])}' (logit: {logit.item():.3f})")
        
        print()

if __name__ == "__main__":
    from anemll.models.qwen_model import TEST_DEVICE, MODEL_DTYPE
    test_kvcache_without_prefill() 
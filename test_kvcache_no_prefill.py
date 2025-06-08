#!/usr/bin/env python3
"""Test KV cache without prefill - single token generation only."""

import numpy as np
import torch
import os
import glob
from transformers import AutoTokenizer
from anemll.models.qwen_model import QwenForCausalLM, QwenConfig
import warnings
warnings.filterwarnings('ignore')

def test_kvcache_without_prefill():
    print("🔬 KV Cache Test - No Prefill (Single Token Generation Only)")
    print("=" * 80)
    
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
    
    # CoreML-style fixed window parameters
    CONTEXT_LENGTH = 256  # Fixed context window size (matches model's state_length)
    PAD_TOKEN_ID = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    
    # Test prompts
    test_prompts = [
        "What",
    #    "The capital", 
    #    "Hello",
    #    "Python is"
    ]
    
    for prompt_text in test_prompts:
        print(f"\n📝 Testing prompt: '{prompt_text}' (CoreML fixed window simulation)")
        print("-" * 60)
        
        # Tokenize prompt
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(TEST_DEVICE)
        original_tokens = input_ids[0].tolist()
        seq_len = len(original_tokens)
        
        print(f"  Original tokens: {original_tokens}")
        print(f"  Decoded: {[tokenizer.decode([t]) for t in original_tokens]}")
        print(f"  Sequence length: {seq_len}")
        
        # === CoreML-style Fixed Window Setup ===
        # Pad input to full context length
        padded_tokens = original_tokens + [PAD_TOKEN_ID] * (CONTEXT_LENGTH - seq_len)
        padded_input_ids = torch.tensor([padded_tokens], dtype=torch.long, device=TEST_DEVICE)
        
        # For single token processing (CoreML style), we only care about the last real token position
        position_ids = torch.tensor([seq_len - 1], dtype=torch.long, device=TEST_DEVICE)  # Single position
        
        # Current position is the last position of actual content
        current_pos = torch.tensor([seq_len - 1], dtype=torch.long, device=TEST_DEVICE)
        
        # Create single-row causal mask for single token (CoreML style)
        # For single token processing, we need [1, 1, 1, CONTEXT_LENGTH] shape
        causal_mask = torch.zeros((1, 1, 1, CONTEXT_LENGTH), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        # Allow attention to positions up to current position (seq_len - 1)
        # Block attention to future positions 
        for j in range(seq_len, CONTEXT_LENGTH):
            causal_mask[0, 0, 0, j] = float('-inf')
        
        # Create update mask for single token processing
        update_mask = torch.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        update_mask[0, 0, seq_len - 1, 0] = 1.0  # Only update the last real token position
        
        print(f"    input_ids shape: {padded_input_ids.shape}")
        print(f"    update_mask shape: {update_mask.shape}")
        print(f"    position_ids shape: {position_ids.shape}")
        print(f"    causal_mask shape: {causal_mask.shape}")
        print(f"    current_pos shape: {current_pos.shape}")
        
        # === Method 1: Disable KV Cache (baseline) ===
        print(f"🟢 Method 1: Disable KV Cache (CoreML fixed window simulation)")
        
        model_no_cache = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=True)
        model_no_cache.load_pretrained_weights(model_dir)
        model_no_cache.eval()
        
        with torch.no_grad():
            logits1 = model_no_cache(
                input_ids=padded_input_ids,
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False
            )
        
        # Get logits for the last real token position (not the padded positions)
        next_token1 = torch.argmax(logits1[0, seq_len - 1, :]).item()
        print(f"  Result: {next_token1} ('{tokenizer.decode([next_token1])}')")

        # Print top 5 tokens and their probabilities for Method 1
        logits1_last = logits1[0, seq_len - 1, :]  # [vocab_size] - use last real token position
        probs1 = torch.softmax(logits1_last, dim=-1)
        topk1 = torch.topk(probs1, 5)
        print("  Top 5 tokens (Method 1):")
        for rank, (idx, prob) in enumerate(zip(topk1.indices.tolist(), topk1.values.tolist()), 1):
            decoded = tokenizer.decode([idx])
            print(f"    {rank}. {idx} ('{decoded}') prob: {prob:.4f}")
        
        # === Method 2: Enable KV Cache (Test) ===
        print(f"🔴 Method 2: Enable KV Cache (CoreML fixed window simulation)")
        
        model_kv_cache = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
        model_kv_cache.load_pretrained_weights(model_dir)
        model_kv_cache.eval()
        model_kv_cache.model.kv_cache_0.zero_()  # Start with clean cache
        
        # Process the same sequence, but with KV cache enabled
        # This should produce the same result as Method 1

        with torch.no_grad():
            logits2 = model_kv_cache(
                input_ids=padded_input_ids,
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False
            )
        
        next_token2 = torch.argmax(logits2[0, seq_len - 1, :]).item()  # Use last real token position
        print(f"  Result: {next_token2} ('{tokenizer.decode([next_token2])}')")
        
        # === Method 3: KV Cache with Incremental Processing ===
        print(f"🟡 Method 3: KV Cache with Incremental Processing (CoreML style)")
        
        model_kv_cache3 = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
        model_kv_cache3.load_pretrained_weights(model_dir)
        model_kv_cache3.eval()
        model_kv_cache3.model.kv_cache_0.zero_()  # Start with clean cache
        
        # Process tokens one by one to build cache incrementally (CoreML style)
        final_logits = None
        for pos in range(seq_len):
            token_id = original_tokens[pos]
            
            # Create single token input with padding (CoreML style)
            single_token_padded = [token_id] + [PAD_TOKEN_ID] * (CONTEXT_LENGTH - 1)
            single_input_ids = torch.tensor([single_token_padded], dtype=torch.long, device=TEST_DEVICE)
            single_position_ids = torch.tensor([pos], dtype=torch.long, device=TEST_DEVICE)
            single_current_pos = torch.tensor([pos], dtype=torch.long, device=TEST_DEVICE)
            
            # Single token causal mask (full context size)
            single_causal_mask = torch.zeros((1, 1, 1, CONTEXT_LENGTH), dtype=MODEL_DTYPE, device=TEST_DEVICE)
            # Allow attention to all previous positions and current position
            for j in range(pos + 1):
                single_causal_mask[0, 0, 0, j] = 0.0
            # Block attention to future positions
            for j in range(pos + 1, CONTEXT_LENGTH):
                single_causal_mask[0, 0, 0, j] = float('-inf')
            
            # Single token update mask
            single_update_mask = torch.ones((1, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
            
            with torch.no_grad():
                logits = model_kv_cache3(
                    input_ids=single_input_ids,
                    update_mask=single_update_mask,
                    position_ids=single_position_ids,
                    causal_mask=single_causal_mask,
                    current_pos=single_current_pos,
                    IN_PREFILL=False
                )
            final_logits = logits  # Keep the last prediction
        
        next_token3 = torch.argmax(final_logits[0, 0, :]).item()
        print(f"  Result: {next_token3} ('{tokenizer.decode([next_token3])}')")
        
        # === Compare Results ===
        print(f"📊 Comparison:")
        print(f"  Method 1 (No KV cache):           {next_token1} ('{tokenizer.decode([next_token1])}')")
        print(f"  Method 2 (KV cache, batch):       {next_token2} ('{tokenizer.decode([next_token2])}')")  
        print(f"  Method 3 (KV cache, incremental): {next_token3} ('{tokenizer.decode([next_token3])}')")
        
        # Check which methods match
        matches = []
        if next_token1 == next_token2:
            matches.append("1-2")
        if next_token1 == next_token3:
            matches.append("1-3")
        if next_token2 == next_token3:
            matches.append("2-3")
            
        if matches:
            print(f"  ✅ MATCHES: {', '.join(matches)}")
        else:
            print(f"  ❌ NO MATCHES - All methods produce different results")
            
        # Show detailed comparison if there are mismatches
        if not (next_token1 == next_token2 == next_token3):
            print(f"  📊 Top 3 predictions for each method:")
            
            for method_name, logits, next_token, pos_idx in [
                ("Method 1", logits1[0, seq_len - 1, :], next_token1, seq_len - 1),
                ("Method 2", logits2[0, seq_len - 1, :], next_token2, seq_len - 1), 
                ("Method 3", final_logits[0, 0, :], next_token3, 0)
            ]:
                top3 = torch.topk(logits, 3)
                print(f"    {method_name} (pos {pos_idx}):")
                for i, (token_id, logit) in enumerate(zip(top3.indices, top3.values)):
                    marker = "→" if token_id.item() == next_token else " "
                    print(f"      {marker} {i+1}. {token_id.item()}: '{tokenizer.decode([token_id.item()])}' (logit: {logit.item():.3f})")
        
        print()
    
    return True

if __name__ == "__main__":
    from anemll.models.qwen_model import TEST_DEVICE, MODEL_DTYPE
    test_kvcache_without_prefill() 
#!/usr/bin/env python3
"""Debug Qwen KV cache with 2-token sequence to find divergence point."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))

from anemll.models.qwen_model import *
import glob
from transformers import AutoTokenizer

def debug_two_tokens():
    """Debug 2-token sequence to find where KV cache diverges."""
    
    print("🔍 Two-Token KV Cache Debug")
    print("=" * 50)
    
    # Setup
    print("\n1. Setting up model...")
    model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    model = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    
    print(f"Cache shape: {model.model.kv_cache_0.shape}")
    
    # Two-token test
    print("\n2. Creating test case...")
    test_text = "The capital"
    tokens = tokenizer(test_text, return_tensors="pt")["input_ids"]
    print(f"Test tokens: {tokens.tolist()} = '{tokenizer.decode(tokens[0])}'")
    print(f"Token 0: {tokens[0, 0].item()} = '{tokenizer.decode([tokens[0, 0].item()])}'")
    print(f"Token 1: {tokens[0, 1].item()} = '{tokenizer.decode([tokens[0, 1].item()])}'")
    
    def apply_full_lm_head(hidden_states):
        """Apply the full 16-way split lm_head to get complete vocab logits."""
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        logits_parts = []
        for i in range(1, 17):
            lm_head = getattr(model, f"lm_head16_{i}")
            logits_part = lm_head(hidden_states).squeeze(2).transpose(1, 2)
            logits_parts.append(logits_part)
        full_logits = torch.cat(logits_parts, dim=2)
        return full_logits
    
    # Create causal mask
    causal_mask = torch.zeros((1, 1, 256, 256), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    for i in range(256):
        for j in range(i + 1, 256):
            causal_mask[0, 0, i, j] = float('-inf')
    
    print("\n3. Test Method 1: Process both tokens together (prefill-style)...")
    model.model.kv_cache_0.zero_()
    
    with torch.no_grad():
        # Process both tokens at once  
        input_ids_both = tokens.to(TEST_DEVICE)  # [1, 2]
        position_ids_both = torch.tensor([0, 1], device=TEST_DEVICE)
        
        print(f"  Input shapes: {input_ids_both.shape}, position_ids: {position_ids_both.shape}")
        
        output_both = model.model(
            input_ids=input_ids_both,
            causal_mask=causal_mask,
            position_ids=position_ids_both,
            current_pos=torch.tensor(0, device=TEST_DEVICE),
            IN_PREFILL=False
        )
        
        logits_both = apply_full_lm_head(output_both)
        
        # Get prediction for position 1 (after "The capital")
        probs_both_pos1 = torch.softmax(logits_both[0, 1], dim=-1)
        top_token_both = torch.argmax(probs_both_pos1).item()
        top_prob_both = probs_both_pos1[top_token_both].item()
        
        print(f"  Both tokens logits shape: {logits_both.shape}")
        print(f"  Position 1 top token: {top_token_both} ('{tokenizer.decode([top_token_both])}')")
        print(f"  Position 1 top probability: {top_prob_both:.6f}")
    
    print("\n4. Test Method 2: Process tokens sequentially with KV cache...")
    model.model.kv_cache_0.zero_()
    
    with torch.no_grad():
        # Process token 0: "The"
        print("\n4a. Processing token 0: 'The'")
        input_ids_0 = tokens[:, :1].to(TEST_DEVICE)  # [1, 1]
        current_pos_0 = torch.tensor(0, device=TEST_DEVICE)
        position_ids_0 = torch.tensor([0], device=TEST_DEVICE) 
        update_mask_0 = torch.ones_like(input_ids_0, dtype=torch.float32)
        
        logits_0 = model(
            input_ids=input_ids_0,
            update_mask=update_mask_0,
            position_ids=position_ids_0,
            causal_mask=causal_mask,
            current_pos=current_pos_0,
            IN_PREFILL=False
        )
        
        print(f"    Token 0 processed, cache norm: {torch.norm(model.model.kv_cache_0).item():.3f}")
        
        # Process token 1: " capital"
        print("\n4b. Processing token 1: ' capital'")
        input_ids_1 = tokens[:, 1:2].to(TEST_DEVICE)  # [1, 1] 
        current_pos_1 = torch.tensor(1, device=TEST_DEVICE)
        position_ids_1 = torch.tensor([1], device=TEST_DEVICE)
        update_mask_1 = torch.ones_like(input_ids_1, dtype=torch.float32)
        
        logits_1 = model(
            input_ids=input_ids_1,
            update_mask=update_mask_1,
            position_ids=position_ids_1,
            causal_mask=causal_mask,
            current_pos=current_pos_1,
            IN_PREFILL=False
        )
        
        print(f"    Token 1 processed, cache norm: {torch.norm(model.model.kv_cache_0).item():.3f}")
        
        # Get prediction after processing both sequentially
        probs_seq_pos1 = torch.softmax(logits_1[0, 0], dim=-1)
        top_token_seq = torch.argmax(probs_seq_pos1).item()
        top_prob_seq = probs_seq_pos1[top_token_seq].item()
        
        print(f"  Sequential logits shape: {logits_1.shape}")
        print(f"  After token 1 top token: {top_token_seq} ('{tokenizer.decode([top_token_seq])}')")
        print(f"  After token 1 top probability: {top_prob_seq:.6f}")
    
    # Compare results
    print("\n5. Comparing results...")
    print(f"Method 1 (both together): {top_token_both} ('{tokenizer.decode([top_token_both])}') - {top_prob_both:.6f}")
    print(f"Method 2 (sequential): {top_token_seq} ('{tokenizer.decode([top_token_seq])}') - {top_prob_seq:.6f}")
    
    if top_token_both == top_token_seq:
        print("✅ Both methods produce same prediction!")
    else:
        print(f"❌ Different predictions!")
        print(f"This suggests the KV cache mechanism has an issue with multi-token sequences.")
    
    # Check cache state
    print(f"\n6. Final cache analysis...")
    print(f"Cache positions filled:")
    for pos in range(3):
        pos_norm = torch.norm(model.model.kv_cache_0[:, :, pos, :]).item()
        print(f"  Position {pos}: {pos_norm:.6f}")
    
    print("\n" + "=" * 50)
    print("✅ Two-token debug complete!")

if __name__ == "__main__":
    debug_two_tokens() 
#!/usr/bin/env python3
"""Debug Qwen KV cache step by step with zero cache and single token."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))

from anemll.models.qwen_model import *
import glob
from transformers import AutoTokenizer

def debug_simple_kv():
    """Debug Qwen KV cache step by step with simplified approach."""
    
    print("🔍 Simple Qwen KV Cache Debug")
    print("=" * 50)
    
    # Setup
    print("\n1. Setting up model...")
    model_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config = QwenConfig.from_json(os.path.join(model_path, 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    model = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    
    print(f"Model config: context={config.context_length}, state={config.state_length}")
    print(f"Cache shape: {model.model.kv_cache_0.shape}")
    
    # Simple test: single token 
    print("\n2. Creating test case...")
    test_text = "The"
    tokens = tokenizer(test_text, return_tensors="pt")["input_ids"]
    print(f"Test token: {tokens.tolist()} = '{tokenizer.decode(tokens[0])}'")
    
    # Clear cache completely
    print("\n3. Clearing cache...")
    model.model.kv_cache_0.zero_()
    cache_norm_before = torch.norm(model.model.kv_cache_0).item()
    print(f"Cache norm before: {cache_norm_before:.8f}")
    
    # Setup inputs for position 0
    print("\n4. Setting up inputs for position 0...")
    input_ids = tokens.to(TEST_DEVICE)
    current_pos = torch.tensor(0, device=TEST_DEVICE)
    position_ids = torch.tensor([0], device=TEST_DEVICE)
    update_mask = torch.ones_like(input_ids, dtype=torch.float32)
    causal_mask = torch.zeros((1, 1, 256, 256), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    print(f"Input shapes:")
    print(f"  input_ids: {input_ids.shape} = {input_ids.tolist()}")
    print(f"  position_ids: {position_ids.shape} = {position_ids.tolist()}")
    print(f"  current_pos: {current_pos.item()}")
    
    def apply_full_lm_head(hidden_states):
        """Apply the full 16-way split lm_head to get complete vocab logits."""
        # Reshape for Conv2d 
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        
        # Apply all 16 lm_head splits
        logits_parts = []
        for i in range(1, 17):
            lm_head = getattr(model, f"lm_head16_{i}")
            logits_part = lm_head(hidden_states).squeeze(2).transpose(1, 2)
            logits_parts.append(logits_part)
        
        # Concatenate all parts to get full vocabulary
        full_logits = torch.cat(logits_parts, dim=2)
        return full_logits
    
    # Forward pass
    print("\n5. Forward pass...")
    with torch.no_grad():
        # Test WITHOUT KV cache first (bypass KV cache mechanism)
        print("\n5a. Without KV cache (bypass KV cache)...")
        try:
            model.model.kv_cache_0.zero_()  # Clear cache
            
            # Get hidden states by calling the model's forward but with a disabled cache approach
            # We'll run through the standard model forward but extract hidden states before lm_head
            output_no_cache = model.model(
                input_ids=input_ids,
                causal_mask=causal_mask,
                position_ids=position_ids,
                current_pos=current_pos,
                IN_PREFILL=False
            )
            
            # Apply full lm_head manually
            logits_no_cache = apply_full_lm_head(output_no_cache)
            
            probs_no_cache = torch.softmax(logits_no_cache[0, 0], dim=-1)
            top_token_no_cache = torch.argmax(probs_no_cache).item()
            top_prob_no_cache = probs_no_cache[top_token_no_cache].item()
            
            print(f"  Logits shape: {logits_no_cache.shape}")
            print(f"  Top token: {top_token_no_cache} ('{tokenizer.decode([top_token_no_cache])}')")
            print(f"  Top probability: {top_prob_no_cache:.6f}")
            print(f"  Logits norm: {torch.norm(logits_no_cache).item():.3f}")
            
        except Exception as e:
            print(f"  Error in no-cache forward: {e}")
            logits_no_cache = None
        
        # Test WITH KV cache using the full model forward
        print("\n5b. With KV cache (full model forward)...")
        model.model.kv_cache_0.zero_()  # Clear cache again
        
        logits_with_cache = model(
            input_ids=input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
        
        probs_with_cache = torch.softmax(logits_with_cache[0, 0], dim=-1)
        top_token_with_cache = torch.argmax(probs_with_cache).item()
        top_prob_with_cache = probs_with_cache[top_token_with_cache].item()
        
        print(f"  Logits shape: {logits_with_cache.shape}")
        print(f"  Top token: {top_token_with_cache} ('{tokenizer.decode([top_token_with_cache])}')")
        print(f"  Top probability: {top_prob_with_cache:.6f}")
        print(f"  Logits norm: {torch.norm(logits_with_cache).item():.3f}")
    
    # Check cache after forward
    print("\n6. Checking cache after forward...")
    cache_norm_after = torch.norm(model.model.kv_cache_0).item()
    print(f"Cache norm after: {cache_norm_after:.8f}")
    
    if cache_norm_after > 0:
        print("✅ Cache was populated!")
        
        # Check specific cache values at position 0
        k_layer0_pos0 = model.model.kv_cache_0[0, :, 0, :]  # Layer 0 keys at position 0
        v_layer0_pos0 = model.model.kv_cache_0[28, :, 0, :]  # Layer 0 values at position 0
        
        print(f"Layer 0 cache at position 0:")
        print(f"  K norm: {torch.norm(k_layer0_pos0).item():.6f}")
        print(f"  V norm: {torch.norm(v_layer0_pos0).item():.6f}")
        
        # Check if any other positions were filled
        pos1_norm = torch.norm(model.model.kv_cache_0[:, :, 1, :]).item()
        print(f"Position 1 cache norm: {pos1_norm:.6f}")
        
    else:
        print("❌ Cache was not populated!")
    
    # Compare results if both worked
    if logits_no_cache is not None:
        print("\n7. Comparing results...")
        logits_diff = torch.norm(logits_no_cache - logits_with_cache).item()
        print(f"Logits difference norm: {logits_diff:.6f}")
        
        if top_token_no_cache == top_token_with_cache:
            print("✅ Same top token prediction")
        else:
            print(f"❌ Different top tokens: {top_token_no_cache} vs {top_token_with_cache}")
            
        # Calculate relative difference
        relative_diff = logits_diff / (torch.norm(logits_no_cache).item() + 1e-8) * 100
        print(f"Relative difference: {relative_diff:.3f}%")
    
    print("\n" + "=" * 50)
    print("✅ Debug complete!")

if __name__ == "__main__":
    debug_simple_kv() 
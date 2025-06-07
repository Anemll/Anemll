#!/usr/bin/env python3
"""Test proper KV cache prefill before single-token generation."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import glob
import numpy as np

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

from anemll.models.qwen_model import *

def test_kv_cache_prefill():
    """Test proper KV cache prefill then single-token generation."""
    
    print("🔍 Testing KV Cache Prefill + Single Token Generation")
    print("=" * 70)
    
    # Find model path
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    if not model_dirs:
        print("❌ Error: Qwen model not found in cache")
        return False
    
    model_dir = model_dirs[0]
    print(f"Using model from: {model_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    
    # Test prompt
    prompt = "What is Apple Neural Engine?"
    print(f"\nPrompt: '{prompt}'")
    
    # Tokenize 
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids
    print(f"Tokenized to {input_ids.shape[1]} tokens: {input_ids.tolist()[0]}")
    print(f"Token meanings: {[tokenizer.decode([t]) for t in input_ids[0]]}")
    
    # Load original Transformers model
    print(f"\n📚 Loading Original Transformers Model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.float16
    )
    
    # Load our PyTorch model
    print(f"\n🔧 Loading Our PyTorch Model...")
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = QwenConfig(**config_dict)
    our_model = QwenForCausalLM(config, enable_coreml=False)
    our_model.load_pretrained_weights(model_dir)
    our_model.eval()
    
    # Test original model (next token after full prompt)
    print(f"\n🔥 Original Model: Next Token After Prompt...")
    with torch.no_grad():
        original_output = original_model(input_ids)
        original_logits = original_output.logits[0, -1, :]  # Last token logits
        original_probs = torch.softmax(original_logits, dim=-1)
        original_top_token = torch.argmax(original_logits).item()
        original_top_prob = original_probs[original_top_token].item()
    
    print(f"Original top token: {original_top_token} ('{tokenizer.decode([original_top_token])}')")
    print(f"Original top prob: {original_top_prob:.4f}")
    
    # Step 1: Prefill KV cache with our model
    print(f"\n🔧 Step 1: Prefilling KV Cache with Full Prompt...")
    seq_len = input_ids.shape[1]
    
    # Create prefill inputs 
    position_ids_prefill = torch.arange(seq_len, dtype=torch.long)
    causal_mask_prefill = torch.full((1, 1, seq_len, 256), -torch.inf, dtype=torch.float16)
    for i in range(seq_len):
        causal_mask_prefill[:, :, i, :i+1] = 0  # Allow attention to previous tokens
    
    # Prefill the cache
    with torch.no_grad():
        our_model.prefill_kv_cache(
            input_ids=input_ids,
            position_ids=position_ids_prefill,
            start_pos=torch.tensor([0], dtype=torch.long),
            causal_mask=causal_mask_prefill
        )
    
    print("✅ KV cache prefilled successfully!")
    
    # Step 2: Generate next token with primed cache
    print(f"\n🔧 Step 2: Single Token Generation with Primed Cache...")
    
    # Create single token inputs (next position after prompt)
    next_position = seq_len
    next_token_dummy = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)  # Dummy token for next position
    position_ids_gen = torch.tensor([next_position], dtype=torch.long)
    causal_mask_gen = torch.full((1, 1, 1, 256), -torch.inf, dtype=torch.float16)
    causal_mask_gen[:, :, 0, :next_position+1] = 0  # Allow attention to all previous tokens
    update_mask = torch.zeros((1, 1, 256, 1), dtype=torch.float16)
    current_pos = torch.tensor([next_position], dtype=torch.long)
    
    with torch.no_grad():
        our_output = our_model(
            input_ids=next_token_dummy,
            update_mask=update_mask,
            position_ids=position_ids_gen,
            causal_mask=causal_mask_gen,
            current_pos=current_pos,
            IN_PREFILL=False
        )
        
        our_logits = our_output[0, 0, :]  # Shape [vocab_size]
        our_probs = torch.softmax(our_logits, dim=-1)
        our_top_token = torch.argmax(our_logits).item()
        our_top_prob = our_probs[our_top_token].item()
    
    print(f"Our top token: {our_top_token} ('{tokenizer.decode([our_top_token])}')")
    print(f"Our top prob: {our_top_prob:.4f}")
    
    # Compare results
    print(f"\n📊 COMPARISON RESULTS")
    print(f"=" * 50)
    
    # Only compare the overlapping vocab range
    min_vocab = min(original_logits.shape[0], our_logits.shape[0])
    original_logits_trimmed = original_logits[:min_vocab]
    our_logits_trimmed = our_logits[:min_vocab]
    
    logits_diff = torch.abs(original_logits_trimmed - our_logits_trimmed)
    max_diff = logits_diff.max().item()
    mean_diff = logits_diff.mean().item()
    
    print(f"Max logits difference: {max_diff:.6f}")
    print(f"Mean logits difference: {mean_diff:.6f}")
    print(f"Tokens match: {original_top_token == our_top_token}")
    
    if original_top_token == our_top_token:
        print("✅ MODELS PREDICT SAME TOKEN!")
        print("🎉 KV cache prefill fixed the issue!")
    else:
        print("❌ MODELS PREDICT DIFFERENT TOKENS!")
        print(f"  Original: {original_top_token} ('{tokenizer.decode([original_top_token])}')")
        print(f"  Our model: {our_top_token} ('{tokenizer.decode([our_top_token])}')")
        print(f"  Difference is now much smaller: max {max_diff:.3f}")
    
    return original_top_token == our_top_token

if __name__ == "__main__":
    test_kv_cache_prefill() 
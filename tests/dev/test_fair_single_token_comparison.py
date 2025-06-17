#!/usr/bin/env python3
"""Fair comparison between original Qwen and our implementation using single token generation."""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import glob
import argparse

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

from anemll.models.qwen_model import QwenForCausalLM, QwenConfig

def test_fair_single_token_comparison(model_path=None):
    """Compare models with single token generation after processing context."""
    
    print("🔍 Fair Single Token Comparison: Original vs Our Implementation")
    print("=" * 70)
    
    # Use provided model path or default
    if model_path:
        model_dir = os.path.expanduser(model_path)
        if not os.path.exists(model_dir):
            print(f"❌ Error: Model not found at {model_dir}")
            return False
    else:
        # Find model path in cache
        cache_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
        model_dirs = glob.glob(os.path.expanduser(cache_path + "*"))
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
    seq_len = input_ids.shape[1]
    
    print(f"Tokenized to {seq_len} tokens: {input_ids.tolist()[0]}")
    print(f"Token meanings: {[tokenizer.decode([t]) for t in input_ids[0]]}")
    
    # Load config first
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        import json
        config_dict = json.load(f)
    
    # Load original Transformers model
    print(f"\n📚 Loading Original Transformers Model...")
    
    try:
        # Try loading normally first
        original_model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            torch_dtype=torch.float16
        )
    except RuntimeError as e:
        if "FP8" in str(e) or "GPU" in str(e):
            print("  FP8 model detected, loading and dequantizing...")
            # Use our FP8 loader
            from load_fp8_model import load_fp8_model_as_fp16
            original_model = load_fp8_model_as_fp16(model_dir)
        else:
            raise e
    
    original_model.eval()
    
    # Load our PyTorch model
    print(f"\n🔧 Loading Our PyTorch Model...")
    
    config = QwenConfig(**config_dict)
    our_model = QwenForCausalLM(config, enable_coreml=False)
    our_model.load_pretrained_weights(model_dir)
    our_model.eval()
    
    print(f"\n🎯 STEP 1: Process context (all tokens except last)")
    print("-" * 50)
    
    # Split input into context and last token
    context_ids = input_ids[:, :-1]  # All tokens except last
    last_token_id = input_ids[:, -1:]  # Last token only
    
    print(f"Context tokens: {context_ids.tolist()[0]}")
    print(f"Last token: {last_token_id.tolist()[0]}")
    
    # Process context with original model to get past_key_values
    print(f"\n🔥 Original Model - Processing Context...")
    with torch.no_grad():
        context_output = original_model(context_ids, use_cache=True)
        past_key_values = context_output.past_key_values
        print(f"Generated past_key_values with {len(past_key_values)} layers")
    
    print(f"\n🎯 STEP 2: Generate next token using last input token")
    print("-" * 50)
    
    # Original model with single token + KV cache
    print(f"\n🔥 Original Model - Single Token Generation:")
    with torch.no_grad():
        # Use past_key_values to maintain context
        original_single_output = original_model(
            last_token_id,
            past_key_values=past_key_values,
            use_cache=True
        )
        original_logits = original_single_output.logits[0, -1, :]
        original_probs = torch.softmax(original_logits, dim=-1)
        original_top_token = torch.argmax(original_logits).item()
        original_top_prob = original_probs[original_top_token].item()
    
    print(f"  Input shape: {last_token_id.shape}")
    print(f"  Output logits shape: {original_single_output.logits.shape}")
    print(f"  Logits range: [{original_logits.min():.3f}, {original_logits.max():.3f}]")
    print(f"  Top token: {original_top_token} ('{tokenizer.decode([original_top_token])}')")
    print(f"  Top prob: {original_top_prob:.4f}")
    
    # Our model with single token
    print(f"\n🔧 Our Model - Single Token Generation:")
    
    # Create inputs for our model
    current_pos = torch.tensor([seq_len - 1], dtype=torch.long)  # Position 5 for 6th token
    position_ids = torch.tensor([[seq_len - 1]], dtype=torch.long)
    
    # Create masks
    update_mask = torch.zeros((1, 1, 256, 1), dtype=torch.float16)
    causal_mask = torch.full((1, 1, 1, 256), -torch.inf, dtype=torch.float16)
    causal_mask[:, :, 0, :seq_len] = 0  # Allow attention to all previous tokens + current
    
    with torch.no_grad():
        # First, prefill the KV cache with the context
        context_len = seq_len - 1  # All tokens except the last one
        
        # Create prefill inputs for context
        prefill_position_ids = torch.arange(context_len, dtype=torch.long)
        prefill_causal_mask = torch.full((1, 1, context_len, 256), -torch.inf, dtype=torch.float16)
        for i in range(context_len):
            prefill_causal_mask[:, :, i, :i+1] = 0  # Allow attention to previous tokens
        
        # Prefill the cache with context (all tokens except last)
        if hasattr(our_model, 'prefill_kv_cache'):
            print(f"  Using prefill_kv_cache method...")
            our_model.prefill_kv_cache(
                input_ids=context_ids,
                position_ids=prefill_position_ids,
                start_pos=torch.tensor([0], dtype=torch.long),
                causal_mask=prefill_causal_mask
            )
        else:
            print(f"  Using regular forward with IN_PREFILL=True...")
            _ = our_model(
                input_ids=context_ids,
                update_mask=torch.zeros((1, 1, 256, 1), dtype=torch.float16),
                position_ids=prefill_position_ids.unsqueeze(0),
                causal_mask=prefill_causal_mask,
                current_pos=torch.tensor([context_len - 1], dtype=torch.long),
                IN_PREFILL=True
            )
        
        # Now generate with just the last token (simulating incremental generation)
        our_output = our_model(
            input_ids=last_token_id,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
        
        our_logits = our_output[0, -1, :]
        our_probs = torch.softmax(our_logits, dim=-1)
        our_top_token = torch.argmax(our_logits).item()
        our_top_prob = our_probs[our_top_token].item()
    
    print(f"  Input shape: {last_token_id.shape}")
    print(f"  Output logits shape: {our_output.shape}")
    print(f"  Logits range: [{our_logits.min():.3f}, {our_logits.max():.3f}]")
    print(f"  Top token: {our_top_token} ('{tokenizer.decode([our_top_token])}')")
    print(f"  Top prob: {our_top_prob:.4f}")
    
    # Compare results
    print(f"\n📊 COMPARISON RESULTS")
    print("=" * 50)
    
    logits_diff = torch.abs(original_logits - our_logits)
    max_diff = logits_diff.max().item()
    mean_diff = logits_diff.mean().item()
    
    print(f"Max logits difference: {max_diff:.6f}")
    print(f"Mean logits difference: {mean_diff:.6f}")
    print(f"Tokens match: {original_top_token == our_top_token}")
    
    if original_top_token == our_top_token:
        print("✅ MODELS PREDICT SAME TOKEN!")
    else:
        print("❌ MODELS PREDICT DIFFERENT TOKENS!")
        
        # Show top 5 predictions from each model
        original_top5 = torch.topk(original_probs, 5)
        our_top5 = torch.topk(our_probs, 5)
        
        print(f"\nOriginal top 5: {original_top5.indices.tolist()}")
        print(f"Original tokens: {[tokenizer.decode([t]) for t in original_top5.indices]}")
        print(f"Our top 5: {our_top5.indices.tolist()}")
        print(f"Our tokens: {[tokenizer.decode([t]) for t in our_top5.indices]}")
    
    return original_top_token == our_top_token

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fair single token comparison between models")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the model directory (e.g., ~/Models/HF/qwen3_1.7B)")
    args = parser.parse_args()
    
    test_fair_single_token_comparison(args.model)
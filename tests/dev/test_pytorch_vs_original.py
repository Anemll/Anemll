#!/usr/bin/env python3
"""Compare our PyTorch Qwen model with original Transformers model."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import glob
import numpy as np
import argparse

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

from anemll.models.qwen_model import *

def test_pytorch_vs_original(model_path=None):
    """Compare our PyTorch model with original Transformers model."""
    
    print("🔍 Comparing Our PyTorch Model vs Original Transformers")
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
    
    # Test with full sequence instead of single token
    print(f"\n🧪 Testing Full Sequence Processing")
    print(f"Full input: {input_ids.tolist()[0]}")
    print(f"Sequence length: {input_ids.shape[1]}")
    
    # Test original model
    print(f"\n🔥 Original Model Forward Pass...")
    with torch.no_grad():
        original_output = original_model(input_ids)
        original_all_logits = original_output.logits  # [1, seq_len, vocab_size]
        original_logits = original_all_logits[0, -1, :]  # Last token logits
        original_probs = torch.softmax(original_logits, dim=-1)
        original_top_token = torch.argmax(original_logits).item()
        original_top_prob = original_probs[original_top_token].item()
    
    print(f"Original all logits shape: {original_all_logits.shape}")
    print(f"Original logits range: [{original_logits.min():.3f}, {original_logits.max():.3f}]")
    print(f"Original top token: {original_top_token} ('{tokenizer.decode([original_top_token])}')")
    print(f"Original top prob: {original_top_prob:.4f}")
    
    # Test our model with full sequence
    print(f"\n🔧 Our Model Forward Pass (Full Sequence)...")
    
    # Create inputs for our model format with full sequence
    seq_len = input_ids.shape[1]
    update_mask = torch.zeros((1, 1, 256, 1), dtype=torch.float16)
    causal_mask = torch.full((1, 1, seq_len, 256), -torch.inf, dtype=torch.float16)
    for i in range(seq_len):
        causal_mask[:, :, i, :i+1] = 0  # Allow attention to previous tokens
    current_pos = torch.tensor([seq_len - 1], dtype=torch.long)
    position_ids = torch.arange(seq_len, dtype=torch.long)
    
    with torch.no_grad():
        our_output = our_model(
            input_ids=input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
        
        our_logits = our_output[0, -1, :]  # Last token logits - shape [vocab_size]
        our_probs = torch.softmax(our_logits, dim=-1)
        our_top_token = torch.argmax(our_logits).item()
        our_top_prob = our_probs[our_top_token].item()
    
    print(f"Our logits range: [{our_logits.min():.3f}, {our_logits.max():.3f}]")
    print(f"Our top token: {our_top_token} ('{tokenizer.decode([our_top_token])}')")
    print(f"Our top prob: {our_top_prob:.4f}")
    
    # Compare logits
    print(f"\n📊 COMPARISON RESULTS")
    print(f"=" * 50)
    
    # Only compare the overlapping vocab range
    min_vocab = min(original_logits.shape[0], our_logits.shape[0])
    original_logits_trimmed = original_logits[:min_vocab]
    our_logits_trimmed = our_logits[:min_vocab]
    
    logits_diff = torch.abs(original_logits_trimmed - our_logits_trimmed)
    max_diff = logits_diff.max().item()
    mean_diff = logits_diff.mean().item()
    
    print(f"Vocab size comparison: Original {original_logits.shape[0]} vs Our {our_logits.shape[0]}")
    print(f"Max logits difference: {max_diff:.6f}")
    print(f"Mean logits difference: {mean_diff:.6f}")
    print(f"Tokens match: {original_top_token == our_top_token}")
    
    if original_top_token == our_top_token:
        print("✅ MODELS PREDICT SAME TOKEN!")
    else:
        print("❌ MODELS PREDICT DIFFERENT TOKENS!")
        print(f"  Original: {original_top_token} ('{tokenizer.decode([original_top_token])}')")
        print(f"  Our model: {our_top_token} ('{tokenizer.decode([our_top_token])}')")
    
    # Now test with single token to make fair comparison
    print(f"\n📋 SINGLE TOKEN COMPARISON")
    print(f"=" * 50)
    print(f"Testing with only the last token as input...")
    
    # Get just the last token
    last_token_id = input_ids[:, -1:]  # Shape [1, 1]
    print(f"Last token: {last_token_id[0].item()} ('{tokenizer.decode(last_token_id[0])}')")
    
    # Original model with single token
    print(f"\n🔥 Original Model (Single Token):")
    with torch.no_grad():
        single_original_output = original_model(last_token_id)
        single_original_logits = single_original_output.logits[0, -1, :]
        single_original_top = torch.argmax(single_original_logits).item()
        single_original_probs = torch.softmax(single_original_logits, dim=-1)
        single_original_top_prob = single_original_probs[single_original_top].item()
    
    print(f"  Logits shape: {single_original_output.logits.shape}")
    print(f"  Top token: {single_original_top} ('{tokenizer.decode([single_original_top])}')")
    print(f"  Top prob: {single_original_top_prob:.4f}")
    
    # Our model with single token
    print(f"\n🔧 Our Model (Single Token):")
    # Use the actual position of the last token (position 5 for 6th token)
    actual_position = seq_len - 1  # This should be 5 for a 6-token sequence
    single_update_mask = torch.zeros((1, 1, 256, 1), dtype=torch.float16)
    single_causal_mask = torch.zeros((1, 1, 1, 256), dtype=torch.float16)
    single_causal_mask[:, :, :, actual_position+1:] = -torch.inf  # Attend to all previous positions
    single_current_pos = torch.tensor([actual_position], dtype=torch.long)
    single_position_ids = torch.tensor([[actual_position]], dtype=torch.long)
    
    print(f"  Using position: {actual_position}")
    
    with torch.no_grad():
        single_our_output = our_model(
            input_ids=last_token_id,
            update_mask=single_update_mask,
            position_ids=single_position_ids,
            causal_mask=single_causal_mask,
            current_pos=single_current_pos,
            IN_PREFILL=False
        )
        single_our_logits = single_our_output[0, -1, :]
        single_our_top = torch.argmax(single_our_logits).item()
        single_our_probs = torch.softmax(single_our_logits, dim=-1)
        single_our_top_prob = single_our_probs[single_our_top].item()
    
    print(f"  Logits shape: {single_our_output.shape}")
    print(f"  Top token: {single_our_top} ('{tokenizer.decode([single_our_top])}')")
    print(f"  Top prob: {single_our_top_prob:.4f}")
    
    print(f"\n  Single token match: {single_original_top == single_our_top}")
    
    return original_top_token == our_top_token

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PyTorch implementation with HuggingFace Qwen model")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the model directory (e.g., ~/Models/HF/qwen3_1.7B)")
    args = parser.parse_args()
    
    test_pytorch_vs_original(args.model) 
#!/usr/bin/env python3
"""Compare single token vs full sequence predictions in our PyTorch model."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import glob

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

from anemll.models.qwen_model import *

def test_single_vs_full_sequence():
    """Compare single token vs full sequence predictions."""
    
    print("🔍 Testing Single Token vs Full Sequence Predictions")
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
    
    # Test original model (for reference)
    print(f"\n🔥 Original Model: Next Token After Full Prompt...")
    with torch.no_grad():
        original_output = original_model(input_ids)
        original_logits = original_output.logits[0, -1, :]  # Last token logits
        original_top_token = torch.argmax(original_logits).item()
    
    print(f"Original top token: {original_top_token} ('{tokenizer.decode([original_top_token])}')")
    
    # Test 1: Our model with full sequence 
    print(f"\n🔧 Test 1: Our Model with Full Sequence...")
    seq_len = input_ids.shape[1]
    update_mask = torch.zeros((1, 1, 256, 1), dtype=torch.float16)
    causal_mask = torch.full((1, 1, seq_len, 256), -torch.inf, dtype=torch.float16)
    for i in range(seq_len):
        causal_mask[:, :, i, :i+1] = 0  # Allow attention to previous tokens
    current_pos = torch.tensor([seq_len - 1], dtype=torch.long)
    position_ids = torch.arange(seq_len, dtype=torch.long)
    
    with torch.no_grad():
        our_output_full = our_model(
            input_ids=input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
        
        our_logits_full = our_output_full[0, -1, :]  # Last token logits
        our_top_token_full = torch.argmax(our_logits_full).item()
    
    print(f"Our model (full sequence) top token: {our_top_token_full} ('{tokenizer.decode([our_top_token_full])}')")
    print(f"Matches original: {our_top_token_full == original_top_token}")
    
    # Test 2: Our model with single token (last token only)
    print(f"\n🔧 Test 2: Our Model with Single Token (like CoreML test)...")
    
    # Use just the last token
    last_token = input_ids[0, -1:].unsqueeze(0)  # Shape [1, 1]
    single_position_ids = torch.tensor([seq_len - 1], dtype=torch.long)  # Position of last token
    single_causal_mask = torch.full((1, 1, 1, 256), -torch.inf, dtype=torch.float16)
    single_causal_mask[:, :, 0, :seq_len] = 0  # Allow attention to all previous tokens
    single_current_pos = torch.tensor([seq_len - 1], dtype=torch.long)
    
    with torch.no_grad():
        our_output_single = our_model(
            input_ids=last_token,
            update_mask=update_mask,
            position_ids=single_position_ids,
            causal_mask=single_causal_mask,
            current_pos=single_current_pos,
            IN_PREFILL=False
        )
        
        our_logits_single = our_output_single[0, 0, :]  # Shape [vocab_size]
        our_top_token_single = torch.argmax(our_logits_single).item()
    
    print(f"Our model (single token) top token: {our_top_token_single} ('{tokenizer.decode([our_top_token_single])}')")
    print(f"Matches original: {our_top_token_single == original_top_token}")
    print(f"Matches full sequence: {our_top_token_single == our_top_token_full}")
    
    # Compare logits
    print(f"\n📊 LOGITS COMPARISON")
    print(f"=" * 50)
    
    # Compare full vs single  
    logits_diff = torch.abs(our_logits_full - our_logits_single)
    max_diff = logits_diff.max().item()
    mean_diff = logits_diff.mean().item()
    
    print(f"Full vs Single sequence max difference: {max_diff:.6f}")
    print(f"Full vs Single sequence mean difference: {mean_diff:.6f}")
    
    if max_diff < 0.1:
        print("✅ Single token mode works correctly!")
    else:
        print("❌ Single token mode has significant differences!")
        print("   This explains why CoreML produces wrong results.")
        print("   The issue is NOT in CoreML conversion but in single-token processing.")
    
    return our_top_token_single == original_top_token

if __name__ == "__main__":
    test_single_vs_full_sequence() 
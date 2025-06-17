#!/usr/bin/env python3
"""Test fixed window approach with PyTorch model to validate workflow."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import glob
import json

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

from anemll.models.qwen_model import *

def test_fixed_window_approach():
    """Test fixed window approach with PyTorch model."""
    
    print("🪟 Testing Fixed Window Approach (PyTorch)")
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
    prompt_tokens = input_ids[0].tolist()
    print(f"Tokenized to {len(prompt_tokens)} tokens: {prompt_tokens}")
    print(f"Token meanings: {[tokenizer.decode([t]) for t in prompt_tokens]}")
    
    # Load our PyTorch model
    print(f"\n🔧 Loading Our PyTorch Model...")
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = QwenConfig(**config_dict)
    our_model = QwenForCausalLM(config, enable_coreml=False)
    our_model.load_pretrained_weights(model_dir)
    our_model.eval()
    
    # Fixed window parameters
    CONTEXT_LENGTH = 256
    PAD_TOKEN = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    print(f"\n🪟 Fixed Window Setup:")
    print(f"  Context length: {CONTEXT_LENGTH}")
    print(f"  Pad token: {PAD_TOKEN}")
    print(f"  Prompt length: {len(prompt_tokens)}")
    
    # Create fixed window with prompt + padding
    window = prompt_tokens + [PAD_TOKEN] * (CONTEXT_LENGTH - len(prompt_tokens))
    current_pos = len(prompt_tokens)  # Position to predict (first position after prompt)
    
    print(f"  Initial current_pos: {current_pos}")
    print(f"  Window[0:10]: {window[:10]}")
    print(f"  Window[{current_pos-2}:{current_pos+3}]: {window[current_pos-2:current_pos+3]}")
    
    # Generate tokens using fixed window approach
    max_new_tokens = 5
    generated_tokens = []
    
    print(f"\n🚀 Generating {max_new_tokens} tokens with Fixed Window...")
    
    for gen_step in range(max_new_tokens):
        print(f"\n--- Generation Step {gen_step + 1} ---")
        print(f"Current pos: {current_pos}")
        print(f"Token at current_pos-1: {window[current_pos-1]} ('{tokenizer.decode([window[current_pos-1]])}')")
        
        # Create inputs for full window
        input_ids = torch.tensor([window], dtype=torch.long)  # [1, CONTEXT_LENGTH]
        position_ids = torch.arange(CONTEXT_LENGTH, dtype=torch.long)  # [0, 1, 2, ..., 255]
        
        # Create causal mask - allow attention up to current_pos
        causal_mask = torch.full((1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH), -torch.inf, dtype=torch.float16)
        for i in range(CONTEXT_LENGTH):
            for j in range(min(i + 1, current_pos)):  # Allow attention up to current_pos
                causal_mask[0, 0, i, j] = 0
        
        # Update mask and current_pos tensor
        update_mask = torch.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=torch.float16)
        current_pos_tensor = torch.tensor([current_pos - 1], dtype=torch.long)  # Position of last token
        
        print(f"📊 Input shapes:")
        print(f"  input_ids: {input_ids.shape}")
        print(f"  position_ids: {position_ids.shape}")
        print(f"  causal_mask: {causal_mask.shape}")
        print(f"  current_pos_tensor: {current_pos_tensor.shape}")
        print(f"  Window content around pos {current_pos}: {window[current_pos-3:current_pos+2]}")
        
        # Run inference
        with torch.no_grad():
            our_output = our_model(
                input_ids=input_ids,
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos_tensor,
                IN_PREFILL=False
            )
            
            # Get logits at current_pos-1 position (the position we want to predict next token for)
            logits_at_pos = our_output[0, current_pos-1, :]  # [vocab_size]
            next_token = torch.argmax(logits_at_pos).item()
            
            print(f"🎯 Prediction:")
            print(f"  Extracting logits from position {current_pos-1}")
            print(f"  Predicted token: {next_token} ('{tokenizer.decode([next_token])}')")
            
            # Show top 5 predictions
            top_logits, top_indices = torch.topk(logits_at_pos, 5)
            print(f"  Top 5 predictions:")
            for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
                token_text = tokenizer.decode([idx.item()])
                print(f"    {i+1}. Token {idx.item()}: '{token_text}' (logit: {logit:.4f})")
            
            # Place predicted token in window at current_pos
            window[current_pos] = next_token
            generated_tokens.append(next_token)
            current_pos += 1
            
            print(f"  Updated window[{current_pos-3}:{current_pos+2}]: {window[current_pos-3:current_pos+2]}")
            
            # Break if we reach context limit
            if current_pos >= CONTEXT_LENGTH:
                print("⚠️  Reached context length limit")
                break
    
    # Show results
    print(f"\n📝 GENERATION RESULTS")
    print(f"=" * 50)
    print(f"Original prompt: '{prompt}'")
    print(f"Generated tokens: {generated_tokens}")
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Generated text: '{generated_text}'")
    
    # Decode full sequence up to current_pos
    full_sequence = window[:current_pos]
    full_text = tokenizer.decode(full_sequence, skip_special_tokens=True)
    print(f"Full text: '{full_text}'")
    
    # Compare with sequential approach from working test
    print(f"\n🔍 COMPARISON WITH SEQUENTIAL TEST:")
    print(f"Expected first token: 2585 (' How')")
    print(f"Got first token: {generated_tokens[0] if generated_tokens else 'None'}")
    print(f"Match: {generated_tokens[0] == 2585 if generated_tokens else False}")
    
    return generated_tokens[0] == 2585 if generated_tokens else False

if __name__ == "__main__":
    success = test_fixed_window_approach()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Fixed window approach {'works' if success else 'needs debugging'}") 
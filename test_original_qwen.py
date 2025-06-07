#!/usr/bin/env python3
"""Test original Transformers Qwen3 model for comparison."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import glob
import os

def test_original_qwen():
    """Test the original Transformers Qwen3 model."""
    
    print("🔄 Testing original Transformers Qwen3 model...")
    
    # Load model
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    if not model_dirs:
        print("❌ Error: Qwen model not found in cache")
        return False
    
    model_dir = model_dirs[0]
    print(f"Loading model from: {model_dir}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16
    )
    
    # Move to appropriate device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Test prompt
    prompt = "What is Apple Neural Engine?"
    print(f"\nPrompt: '{prompt}'")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids.to(device)
    print(f"Tokenized to {input_ids.shape[1]} tokens: {input_ids.tolist()[0]}")
    print(f"Token meanings: {[tokenizer.decode([t]) for t in input_ids[0]]}")
    
    # Generate
    print(f"\n🎯 Generating with original model...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=False,  # Deterministic for comparison
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    end_time = time.time()
    
    # Decode results
    generated_ids = outputs[0][input_ids.shape[1]:]  # Only new tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n{'='*60}")
    print(f"✅ ORIGINAL MODEL RESULTS")
    print(f"{'='*60}")
    print(f"Generated {len(generated_ids)} tokens in {end_time - start_time:.2f}s")
    print(f"Speed: {len(generated_ids) / (end_time - start_time):.1f} tokens/second")
    print(f"\nFull conversation:")
    print(f"Q: {prompt}")
    print(f"A: {generated_text}")
    
    print(f"\nGenerated token IDs: {generated_ids.tolist()}")
    print(f"First few tokens: {[tokenizer.decode([t]) for t in generated_ids[:10]]}")
    
    return True

if __name__ == "__main__":
    test_original_qwen() 
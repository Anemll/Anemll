#!/usr/bin/env python3
"""Compare with original Hugging Face Qwen3 model to understand expected behavior."""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import glob
import argparse

def test_original_hf_model(model_path=None):
    """Test the original Hugging Face model to see expected behavior."""
    
    print("🟢 Testing Original Hugging Face Qwen3 Model")
    print("=" * 60)
    
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
    
    print(f"Using model: {model_dir}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    original_model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.float16
    )
    original_model.eval()
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "Hello, my name is",
        "2 + 2 =",
        "The weather today is"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs.input_ids
        prompt_tokens = input_ids[0].tolist()
        
        print(f"  Tokens: {prompt_tokens}")
        print(f"  Meanings: {[tokenizer.decode([t]) for t in prompt_tokens]}")
        
        # Generate multiple tokens
        with torch.no_grad():
            generated = original_model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Extract new tokens
        new_tokens = generated[0][len(prompt_tokens):].tolist()
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        
        print(f"  First new token: {new_tokens[0]} ('{tokenizer.decode([new_tokens[0]])}')")
        print(f"  New text: '{new_text}'")
        print(f"  Full text: '{full_text}'")
    
    return True

def inspect_hf_model_config(model_path=None):
    """Inspect the HF model configuration to understand the architecture."""
    
    print(f"\n🔍 Inspecting HF Model Configuration")
    print("=" * 60)
    
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
    
    print(f"Using model: {model_dir}")
    
    # Load model
    original_model = AutoModelForCausalLM.from_pretrained(model_dir)
    config = original_model.config
    
    print(f"HF Model Config:")
    print(f"  Model type: {type(original_model).__name__}")
    print(f"  Config type: {type(config).__name__}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  head_dim: {getattr(config, 'head_dim', 'not present')}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  max_position_embeddings: {config.max_position_embeddings}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  rms_norm_eps: {config.rms_norm_eps}")
    print(f"  rope_theta: {config.rope_theta}")
    
    # Check attention module
    attention = original_model.model.layers[0].self_attn
    print(f"\nAttention Module:")
    print(f"  Type: {type(attention).__name__}")
    print(f"  head_dim: {attention.head_dim}")
    print(f"  q_proj weight shape: {attention.q_proj.weight.shape}")
    print(f"  k_proj weight shape: {attention.k_proj.weight.shape}")
    print(f"  v_proj weight shape: {attention.v_proj.weight.shape}")
    print(f"  o_proj weight shape: {attention.o_proj.weight.shape}")
    
    # Calculate expected shapes
    expected_q_out = config.num_attention_heads * attention.head_dim
    expected_kv_out = config.num_key_value_heads * attention.head_dim
    
    print(f"\nExpected shapes:")
    print(f"  Q projection: {config.hidden_size} -> {expected_q_out} ({config.num_attention_heads} * {attention.head_dim})")
    print(f"  K projection: {config.hidden_size} -> {expected_kv_out} ({config.num_key_value_heads} * {attention.head_dim})")
    print(f"  V projection: {config.hidden_size} -> {expected_kv_out} ({config.num_key_value_heads} * {attention.head_dim})")
    print(f"  O projection: {expected_q_out} -> {config.hidden_size}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Hugging Face Qwen3 models")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the model directory (e.g., ~/Models/HF/qwen3_1.7B)")
    args = parser.parse_args()
    
    print("🔍 Hugging Face Qwen3 Model Analysis")
    print("=" * 70)
    
    # Test 1: See what the original model generates
    test_original_hf_model(args.model)
    
    # Test 2: Inspect the model configuration
    inspect_hf_model_config(args.model) 
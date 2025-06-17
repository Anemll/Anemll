#!/usr/bin/env python3
"""Test first forward pass comparison between our model and HF model."""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import glob

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

from anemll.models.qwen_model import *

def test_first_forward_pass():
    """Compare first forward pass without KV cache complexity."""
    
    print("🔍 Testing First Forward Pass - No KV Cache")
    print("=" * 60)
    
    # Find model path
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    model_dir = model_dirs[0]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    
    # Test with simple prompt
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids
    prompt_tokens = input_ids[0].tolist()
    
    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {prompt_tokens}")
    print(f"Token meanings: {[tokenizer.decode([t]) for t in prompt_tokens]}")
    
    # Test 1: Original HF model
    print(f"\n🟢 Original HF Model")
    print("-" * 30)
    
    original_model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)
    original_model.eval()
    
    with torch.no_grad():
        # Get logits for the full sequence
        hf_outputs = original_model(input_ids)
        hf_logits = hf_outputs.logits  # [1, seq_len, vocab_size]
        
        # Get logits for the last position
        hf_last_logits = hf_logits[0, -1, :]  # [vocab_size]
        hf_next_token = torch.argmax(hf_last_logits).item()
        
        print(f"HF logits shape: {hf_logits.shape}")
        print(f"HF last position logits shape: {hf_last_logits.shape}")
        print(f"HF next token: {hf_next_token} ('{tokenizer.decode([hf_next_token])}')")
        print(f"HF top 5 tokens: {torch.topk(hf_last_logits, 5).indices.tolist()}")
        print(f"HF top 5 probs: {torch.softmax(hf_last_logits, dim=-1)[torch.topk(hf_last_logits, 5).indices].tolist()}")
    
    # Test 2: Our model - without KV cache, just regular forward
    print(f"\n🟡 Our Model - Regular Forward (No KV Cache)")
    print("-" * 30)
    
    # Load config and create model
    config_path = os.path.join(model_dir, "config.json")
    config = QwenConfig.from_json(config_path)
    
    our_model = QwenForCausalLM(config, enable_coreml=False)
    our_model.load_pretrained_weights(model_dir)
    our_model.eval()
    
    # Try to use the regular transformer forward pass (like HF)
    # Let's access the model's transformer forward method directly
    with torch.no_grad():
        # Get embeddings
        input_embeds = our_model.model.embed_tokens(input_ids.to(TEST_DEVICE))
        print(f"Input embeddings shape: {input_embeds.shape}")
        
        # Try using the standard forward method of the base transformer
        # Let's manually process without KV cache for comparison
        hidden_states = input_embeds
        seq_len = hidden_states.shape[1]
        
        # Apply each layer manually to see where divergence occurs
        for layer_idx, layer in enumerate(our_model.model.layers):
            print(f"  Processing layer {layer_idx}...")
            
            # Residual connection
            residual = hidden_states
            
            # Pre-attention norm
            hidden_states = layer.input_layernorm(hidden_states)
            
            # Attention without KV cache - use the standard forward method
            # This bypasses our KV cache logic
            try:
                attn_output = layer.self_attn(
                    hidden_states.to(TEST_DEVICE),
                    torch.zeros((1, 1, seq_len, seq_len), dtype=MODEL_DTYPE, device=TEST_DEVICE),  # causal mask
                    torch.arange(seq_len, dtype=torch.long, device=TEST_DEVICE),  # position_ids
                    torch.tensor([0], dtype=torch.long, device=TEST_DEVICE),  # current_pos (unused)
                )
                print(f"    Attention output shape: {attn_output.shape}")
            except Exception as e:
                print(f"    ❌ Attention failed: {e}")
                return False
            
            hidden_states = residual + attn_output
            
            # Post-attention norm and MLP
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            mlp_output = layer.mlp(hidden_states)
            hidden_states = residual + mlp_output
            
            print(f"    Layer {layer_idx} output shape: {hidden_states.shape}")
            
            # Check for NaN or inf
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                print(f"    ❌ NaN/inf detected in layer {layer_idx}")
                return False
        
        # Final norm
        hidden_states = our_model.model.norm(hidden_states)
        print(f"Final hidden states shape: {hidden_states.shape}")
        
        # Get logits from lm_head
        our_logits = our_model.forward(
            input_ids=input_ids.to(TEST_DEVICE),
            update_mask=torch.ones((1, 1, config.context_length, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            position_ids=torch.arange(seq_len, dtype=torch.long, device=TEST_DEVICE),
            causal_mask=torch.zeros((1, 1, seq_len, config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE),
            current_pos=torch.tensor([seq_len-1], dtype=torch.long, device=TEST_DEVICE),
            IN_PREFILL=True  # Use prefill mode
        )
        
        print(f"Our logits shape: {our_logits.shape}")
        
        # Extract logits for last position
        if our_logits.dim() == 3:
            our_last_logits = our_logits[0, -1, :]
        else:
            our_last_logits = our_logits[0, :]
            
        our_next_token = torch.argmax(our_last_logits).item()
        
        print(f"Our next token: {our_next_token} ('{tokenizer.decode([our_next_token])}')")
        print(f"Our top 5 tokens: {torch.topk(our_last_logits, 5).indices.tolist()}")
        print(f"Our top 5 probs: {torch.softmax(our_last_logits, dim=-1)[torch.topk(our_last_logits, 5).indices].tolist()}")
    
    # Compare results
    print(f"\n📊 COMPARISON")
    print("=" * 30)
    print(f"HF model:  {hf_next_token} ('{tokenizer.decode([hf_next_token])}')")
    print(f"Our model: {our_next_token} ('{tokenizer.decode([our_next_token])}')")
    print(f"Match: {'✅' if hf_next_token == our_next_token else '❌'}")
    
    if hf_next_token == our_next_token:
        print(f"\n🎉 SUCCESS: Models produce same output!")
        return True
    else:
        print(f"\n❌ Models produce different outputs")
        return False

if __name__ == "__main__":
    test_first_forward_pass() 
#!/usr/bin/env python3
"""Compare Qwen KV cache against working Llama implementation with simplified test."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))

# Import both models
from anemll.models.qwen_model import *
from anemll.models.llama_model import LlamaConfig, LlamaModel, LlamaForCausalLM
import glob
from transformers import AutoTokenizer

def debug_compare_llama():
    """Compare single token KV cache behavior between Qwen and Llama."""
    
    print("🔍 Comparing Qwen vs Llama KV Cache Implementation")
    print("=" * 70)
    
    # Setup Qwen
    print("\n1. Setting up Qwen model...")
    model_path_qwen = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
    config_qwen = QwenConfig.from_json(os.path.join(model_path_qwen, 'config.json'))
    # Simplify config
    config_qwen.context_length = 256
    config_qwen.state_length = 256
    
    qwen_model = QwenForCausalLM(config_qwen, enable_coreml=False)
    qwen_model.load_pretrained_weights(model_path_qwen)
    
    print(f"Qwen config: num_layers={config_qwen.num_hidden_layers}, "
          f"num_heads={config_qwen.num_attention_heads}, num_kv_heads={config_qwen.num_key_value_heads}, "
          f"head_dim={config_qwen.head_dim}, context/state={config_qwen.context_length}")
    
    # Setup Llama (use smaller config for comparison)
    print("\n2. Setting up Llama model...")
    config_llama = LlamaConfig()
    config_llama.num_hidden_layers = config_qwen.num_hidden_layers
    config_llama.num_attention_heads = config_qwen.num_attention_heads
    config_llama.num_key_value_heads = config_qwen.num_key_value_heads
    config_llama.hidden_size = config_qwen.hidden_size
    config_llama.vocab_size = config_qwen.vocab_size
    config_llama.context_length = 256
    config_llama.state_length = 256
    
    llama_model = LlamaForCausalLM(config_llama, enable_coreml=False)
    
    print(f"Llama config: num_layers={config_llama.num_hidden_layers}, "
          f"num_heads={config_llama.num_attention_heads}, num_kv_heads={config_llama.num_key_value_heads}, "
          f"context/state={config_llama.context_length}")
    
    # Create simple test: single token "Hello"
    print("\n3. Creating simple test case...")
    tokenizer = AutoTokenizer.from_pretrained(model_path_qwen, use_fast=False)
    test_text = "Hello"
    tokens = tokenizer(test_text, return_tensors="pt")["input_ids"]
    print(f"Test tokens: {tokens.tolist()}")
    print(f"Decoded: '{tokenizer.decode(tokens[0])}'")
    
    # Test single token forward with zero cache
    print("\n4. Testing single token forward (position 0) with fresh cache...")
    
    # Setup inputs
    input_ids = tokens.to(TEST_DEVICE)  # [1, 1]
    current_pos = torch.tensor(0, device=TEST_DEVICE)
    position_ids = torch.tensor([0], device=TEST_DEVICE)
    update_mask = torch.ones_like(input_ids, dtype=torch.float32)
    
    # Create causal masks (both should be zeros for single token)
    causal_mask_qwen = torch.zeros((1, 1, 256, 256), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    causal_mask_llama = torch.zeros((1, 1, 256, 256), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    print(f"Input shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  position_ids: {position_ids.shape}")
    print(f"  current_pos: {current_pos}")
    print(f"  causal_mask: {causal_mask_qwen.shape}")
    
    # Clear caches
    qwen_model.model.kv_cache_0.zero_()
    llama_model.model.kv_cache_0.zero_()
    
    print("\n5. Forward pass through both models...")
    
    # Forward through Qwen
    with torch.no_grad():
        qwen_logits = qwen_model(
            input_ids=input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask_qwen,
            current_pos=current_pos,
            IN_PREFILL=False
        )
        qwen_probs = torch.softmax(qwen_logits[0, 0], dim=-1)
        qwen_top_token = torch.argmax(qwen_probs).item()
        qwen_top_prob = qwen_probs[qwen_top_token].item()
    
    # Forward through Llama  
    with torch.no_grad():
        llama_logits = llama_model(
            input_ids=input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask_llama,
            current_pos=current_pos,
            IN_PREFILL=False
        )
        llama_probs = torch.softmax(llama_logits[0, 0], dim=-1)
        llama_top_token = torch.argmax(llama_probs).item()
        llama_top_prob = llama_probs[llama_top_token].item()
    
    print("\n6. Comparing results...")
    print(f"Qwen output:")
    print(f"  Logits shape: {qwen_logits.shape}")
    print(f"  Top token: {qwen_top_token} ('{tokenizer.decode([qwen_top_token])}')")
    print(f"  Top probability: {qwen_top_prob:.6f}")
    print(f"  Logits norm: {torch.norm(qwen_logits).item():.3f}")
    
    print(f"\nLlama output:")
    print(f"  Logits shape: {llama_logits.shape}")
    print(f"  Top token: {llama_top_token} ('{tokenizer.decode([llama_top_token])}')")
    print(f"  Top probability: {llama_top_prob:.6f}")
    print(f"  Logits norm: {torch.norm(llama_logits).item():.3f}")
    
    # Compare KV cache states
    print(f"\n7. Comparing KV cache after single token...")
    qwen_cache = qwen_model.model.kv_cache_0
    llama_cache = llama_model.model.kv_cache_0
    
    print(f"Cache shapes:")
    print(f"  Qwen cache: {qwen_cache.shape}")
    print(f"  Llama cache: {llama_cache.shape}")
    
    # Check if any cache values were set
    qwen_cache_used = torch.norm(qwen_cache).item()
    llama_cache_used = torch.norm(llama_cache).item()
    
    print(f"Cache utilization:")
    print(f"  Qwen cache norm: {qwen_cache_used:.6f}")
    print(f"  Llama cache norm: {llama_cache_used:.6f}")
    
    # Check cache at position 0
    if qwen_cache_used > 0:
        qwen_k_pos0 = qwen_cache[0, :, 0, :]  # Layer 0, all heads, position 0
        qwen_v_pos0 = qwen_cache[config_qwen.num_hidden_layers, :, 0, :]  # Value cache
        print(f"  Qwen K[0] norm: {torch.norm(qwen_k_pos0).item():.6f}")
        print(f"  Qwen V[0] norm: {torch.norm(qwen_v_pos0).item():.6f}")
    
    if llama_cache_used > 0:
        llama_k_pos0 = llama_cache[0, :, 0, :]  # Layer 0, all heads, position 0
        llama_v_pos0 = llama_cache[config_llama.num_hidden_layers, :, 0, :]  # Value cache
        print(f"  Llama K[0] norm: {torch.norm(llama_k_pos0).item():.6f}")
        print(f"  Llama V[0] norm: {torch.norm(llama_v_pos0).item():.6f}")
    
    print("\n" + "=" * 70)
    print("✅ Comparison complete!")

if __name__ == "__main__":
    debug_compare_llama() 
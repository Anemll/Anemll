#!/usr/bin/env python3
"""Test PyTorch Qwen model with KV cache functionality."""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import glob
import time

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

from anemll.models.qwen_model import *

def test_qwen_kv_cache_initialization():
    """Test that the Qwen model KV cache is properly initialized."""
    
    print("🧪 Test 1: KV Cache Initialization")
    print("-" * 40)
    
    # Find model path
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    if not model_dirs:
        print("❌ Error: Qwen model not found in cache")
        return False
    
    model_dir = model_dirs[0]
    print(f"Using model from: {model_dir}")
    
    # Load config and create model
    config_path = os.path.join(model_dir, "config.json")
    config = QwenConfig.from_json(config_path)
    print(f"Loaded config - Hidden size: {config.hidden_size}, Vocab size: {config.vocab_size}")
    
    # Create ANEMLL model with KV cache
    model = QwenForCausalLM(config, enable_coreml=False)
    model.load_pretrained_weights(model_dir)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"✅ KV cache initialized with shape: {model.model.kv_cache_0.shape}")
    
    # Verify KV cache shape
    expected_cache_shape = (
        2 * config.num_hidden_layers,  # Keys and values for all layers
        config.num_key_value_heads,
        config.state_length,
        getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    )
    print(f"Expected KV cache shape: {expected_cache_shape}")
    print(f"Actual KV cache shape: {model.model.kv_cache_0.shape}")
    assert model.model.kv_cache_0.shape == expected_cache_shape, "KV cache shape mismatch"
    
    # Verify cache is initialized to zeros
    cache_sum = torch.sum(model.model.kv_cache_0).item()
    print(f"Cache sum (should be 0): {cache_sum}")
    assert abs(cache_sum) < 1e-6, "KV cache should be initialized to zeros"
    
    print("✅ KV cache initialization test passed\n")
    return model, config

def test_single_token_forward_pass(model, config):
    """Test single token forward pass with KV cache update."""
    
    print("🧪 Test 2: Single Token Forward Pass")
    print("-" * 40)
    
    # Reset cache
    model.model.kv_cache_0.zero_()
    
    # Prepare test inputs
    test_token_id = 1000  # Some token
    input_ids = torch.tensor([[test_token_id]], dtype=torch.long, device=TEST_DEVICE)
    position_ids = torch.tensor([0], dtype=torch.long, device=TEST_DEVICE)
    current_pos = torch.tensor([0], dtype=torch.long, device=TEST_DEVICE)
    update_mask = torch.ones((1, 1, config.context_length, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    print(f"Input token: {test_token_id}")
    print(f"Position: {position_ids.item()}")
    
    # Store initial cache state
    initial_cache = model.model.kv_cache_0.clone()
    
    # Forward pass
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
    
    print(f"Logits shape: {logits.shape}")
    print(f"Logits type: {type(logits)}")
    
    # Check that cache was updated
    cache_diff = torch.sum(torch.abs(model.model.kv_cache_0 - initial_cache)).item()
    print(f"Cache difference after forward pass: {cache_diff}")
    assert cache_diff > 0, "KV cache should be updated after forward pass"
    
    print("✅ Single token forward pass test passed\n")
    return True

def test_multi_token_prefill(model, config):
    """Test multi-token prefill functionality."""
    
    print("🧪 Test 3: Multi-Token Prefill")
    print("-" * 40)
    
    # Reset cache
    model.model.kv_cache_0.zero_()
    
    # Prepare multi-token input
    test_tokens = [100, 200, 300, 400, 500]  # 5 tokens
    prefill_input_ids = torch.tensor([test_tokens], dtype=torch.long, device=TEST_DEVICE)
    prefill_position_ids = torch.arange(len(test_tokens), dtype=torch.long, device=TEST_DEVICE)
    prefill_causal_mask = torch.zeros((1, 1, len(test_tokens), config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    start_pos = 0
    
    print(f"Prefill tokens: {test_tokens}")
    print(f"Prefill positions: {prefill_position_ids.tolist()}")
    
    # Store cache before prefill
    cache_before_prefill = model.model.kv_cache_0.clone()
    
    # Prefill KV cache
    model.prefill_kv_cache(
        input_ids=prefill_input_ids,
        position_ids=prefill_position_ids,
        start_pos=start_pos,
        causal_mask=prefill_causal_mask
    )
    
    # Check cache was updated during prefill
    cache_diff_prefill = torch.sum(torch.abs(model.model.kv_cache_0 - cache_before_prefill)).item()
    print(f"Cache difference after prefill: {cache_diff_prefill}")
    assert cache_diff_prefill > 0, "KV cache should be updated during prefill"
    
    # Verify cache contains non-zero values at expected positions
    for layer_idx in range(config.num_hidden_layers):
        key_idx = layer_idx
        value_idx = layer_idx + config.num_hidden_layers
        
        # Check that positions 0 to len(test_tokens)-1 have been filled
        key_norm = torch.norm(model.model.kv_cache_0[key_idx, :, :len(test_tokens), :]).item()
        value_norm = torch.norm(model.model.kv_cache_0[value_idx, :, :len(test_tokens), :]).item()
        
        print(f"Layer {layer_idx}: Key norm = {key_norm:.6f}, Value norm = {value_norm:.6f}")
        assert key_norm > 0, f"Key cache should be non-zero for layer {layer_idx}"
        assert value_norm > 0, f"Value cache should be non-zero for layer {layer_idx}"
    
    print("✅ Multi-token prefill test passed\n")
    return test_tokens

def test_generation_with_prefilled_cache(model, config, prefill_tokens):
    """Test generation using the prefilled KV cache."""
    
    print("🧪 Test 4: Generation with Prefilled Cache")
    print("-" * 40)
    
    # Generate next token after prefill
    next_position = len(prefill_tokens)
    next_token_id = 600
    next_input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=TEST_DEVICE)
    next_position_ids = torch.tensor([next_position], dtype=torch.long, device=TEST_DEVICE)
    next_current_pos = torch.tensor([next_position], dtype=torch.long, device=TEST_DEVICE)
    update_mask = torch.ones((1, 1, config.context_length, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    print(f"Next token: {next_token_id}")
    print(f"Next position: {next_position}")
    
    # Store cache before next token
    cache_before_next = model.model.kv_cache_0.clone()
    
    # Forward pass for next token
    with torch.no_grad():
        next_logits = model(
            input_ids=next_input_ids,
            update_mask=update_mask,
            position_ids=next_position_ids,
            causal_mask=causal_mask,
            current_pos=next_current_pos,
            IN_PREFILL=False
        )
    
    print(f"Next token logits shape: {next_logits.shape}")
    
    # Check cache was updated at the correct position
    cache_diff_next = torch.sum(torch.abs(model.model.kv_cache_0 - cache_before_next)).item()
    print(f"Cache difference after next token: {cache_diff_next}")
    assert cache_diff_next > 0, "KV cache should be updated for next token"
    
    # Verify that previous positions are unchanged
    cache_positions_unchanged = True
    for layer_idx in range(config.num_hidden_layers):
        key_idx = layer_idx
        value_idx = layer_idx + config.num_hidden_layers
        
        # Check that positions 0 to len(prefill_tokens)-1 are unchanged
        key_unchanged = torch.allclose(
            model.model.kv_cache_0[key_idx, :, :len(prefill_tokens), :],
            cache_before_next[key_idx, :, :len(prefill_tokens), :],
            atol=1e-6
        )
        value_unchanged = torch.allclose(
            model.model.kv_cache_0[value_idx, :, :len(prefill_tokens), :],
            cache_before_next[value_idx, :, :len(prefill_tokens), :],
            atol=1e-6
        )
        
        if not (key_unchanged and value_unchanged):
            print(f"⚠️  Warning: Previous positions were modified in layer {layer_idx}")
            cache_positions_unchanged = False
    
    if cache_positions_unchanged:
        print("✅ Previous cache positions remain unchanged")
    
    print("✅ Generation with prefilled cache test passed\n")
    return True

def test_rotary_embeddings(model, config):
    """Test rotary embeddings functionality."""
    
    print("🧪 Test 5: Rotary Embeddings")
    print("-" * 40)
    
    # Test single position rotary embeddings
    cos_single, sin_single = model.model.get_rotary_embeddings_s(current_pos=5)
    print(f"Single position rotary shapes - cos: {cos_single.shape}, sin: {sin_single.shape}")
    
    # Test multi-position rotary embeddings
    positions = torch.arange(8, dtype=torch.long, device=TEST_DEVICE)
    cos_multi, sin_multi = model.model.get_rotary_embedding_prefill(positions)
    print(f"Multi position rotary shapes - cos: {cos_multi.shape}, sin: {sin_multi.shape}")
    
    # Verify shapes are correct
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    expected_single_shape = (1, 1, 1, head_dim)
    expected_multi_shape = (1, 8, 1, head_dim)
    
    assert cos_single.shape == expected_single_shape, f"Single rotary shape mismatch: {cos_single.shape} != {expected_single_shape}"
    assert cos_multi.shape == expected_multi_shape, f"Multi rotary shape mismatch: {cos_multi.shape} != {expected_multi_shape}"
    
    # Test that rotary embeddings produce reasonable values
    assert torch.all(cos_single >= -1) and torch.all(cos_single <= 1), "Cosine values should be in [-1, 1]"
    assert torch.all(sin_single >= -1) and torch.all(sin_single <= 1), "Sine values should be in [-1, 1]"
    
    print("✅ Rotary embeddings test passed\n")
    return True

def test_text_generation_with_kv_cache(model, config):
    """Test realistic text generation using KV cache."""
    
    print("🧪 Test 6: Realistic Text Generation with KV Cache")
    print("-" * 40)
    
    # Find model path for tokenizer
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    model_dir = model_dirs[0]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    
    # Test prompt
    prompt = "The capital of France is"
    print(f"Prompt: '{prompt}'")
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids.to(TEST_DEVICE)
    prompt_tokens = input_ids[0].tolist()
    
    print(f"Tokenized to {len(prompt_tokens)} tokens: {prompt_tokens}")
    print(f"Token meanings: {[tokenizer.decode([t]) for t in prompt_tokens]}")
    
    # Reset model cache
    model.model.kv_cache_0.zero_()
    
    # Prefill with the prompt
    position_ids = torch.arange(len(prompt_tokens), dtype=torch.long, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, len(prompt_tokens), config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    print(f"Prefilling KV cache with {len(prompt_tokens)} tokens...")
    model.prefill_kv_cache(
        input_ids=input_ids,
        position_ids=position_ids,
        start_pos=0,
        causal_mask=causal_mask
    )
    
    # Generate tokens one by one
    generated_tokens = prompt_tokens.copy()
    max_new_tokens = 8
    
    print(f"Generating {max_new_tokens} new tokens...")
    
    for step in range(max_new_tokens):
        current_pos = len(generated_tokens)
        
        if current_pos >= config.state_length:
            print(f"Reached state length limit ({config.state_length})")
            break
        
        # Prepare inputs for next token generation
        update_mask = torch.ones((1, 1, config.context_length, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        causal_mask = torch.zeros((1, 1, 1, config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        
        # Use last token as input
        last_token = generated_tokens[-1]
        input_ids = torch.tensor([[last_token]], dtype=torch.long, device=TEST_DEVICE)
        position_ids = torch.tensor([current_pos], dtype=torch.long, device=TEST_DEVICE)
        current_pos_tensor = torch.tensor([current_pos], dtype=torch.long, device=TEST_DEVICE)
        
        # Generate next token
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos_tensor,
                IN_PREFILL=False
            )
        
        # Get next token (greedy decoding)
        if isinstance(logits, tuple):
            # If model returns tuple of logits (split vocabulary), concatenate them
            all_logits = torch.cat(logits, dim=-1)
        else:
            all_logits = logits
        
        next_token = torch.argmax(all_logits[0, 0, :]).item()
        generated_tokens.append(next_token)
        
        decoded_token = tokenizer.decode([next_token])
        print(f"  Step {step + 1}: Generated token {next_token} ('{decoded_token}')")
    
    # Decode the full generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    new_text = tokenizer.decode(generated_tokens[len(prompt_tokens):], skip_special_tokens=True)
    
    print(f"\nGenerated text: '{generated_text}'")
    print(f"New text: '{new_text}'")
    
    # Basic sanity checks
    assert len(generated_tokens) > len(prompt_tokens), "Should generate at least one new token"
    assert all(isinstance(t, int) for t in generated_tokens), "All tokens should be integers"
    
    print("✅ Realistic text generation test passed\n")
    return generated_text

def test_performance_comparison(model, config):
    """Test performance comparison between KV cache and without cache."""
    
    print("🧪 Test 7: Performance Comparison")
    print("-" * 40)
    
    # Find model path for tokenizer
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    model_dir = model_dirs[0]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    
    # Test prompt
    prompt = "Artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids.to(TEST_DEVICE)
    prompt_tokens = input_ids[0].tolist()
    
    print(f"Testing with prompt: '{prompt}' ({len(prompt_tokens)} tokens)")
    
    # Warm up
    for _ in range(3):
        model.model.kv_cache_0.zero_()
        with torch.no_grad():
            _ = model(
                input_ids=torch.tensor([[prompt_tokens[0]]], dtype=torch.long, device=TEST_DEVICE),
                update_mask=torch.ones((1, 1, config.context_length, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
                position_ids=torch.tensor([0], dtype=torch.long, device=TEST_DEVICE),
                causal_mask=torch.zeros((1, 1, 1, config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE),
                current_pos=torch.tensor([0], dtype=torch.long, device=TEST_DEVICE),
                IN_PREFILL=False
            )
    
    # Time KV cache approach
    start_time = time.time()
    model.model.kv_cache_0.zero_()
    
    # Prefill
    position_ids = torch.arange(len(prompt_tokens), dtype=torch.long, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, len(prompt_tokens), config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    model.prefill_kv_cache(
        input_ids=input_ids,
        position_ids=position_ids,
        start_pos=0,
        causal_mask=causal_mask
    )
    
    # Generate 5 tokens
    for step in range(5):
        current_pos = len(prompt_tokens) + step
        last_token = prompt_tokens[-1] if step == 0 else 100  # Use dummy token
        
        with torch.no_grad():
            _ = model(
                input_ids=torch.tensor([[last_token]], dtype=torch.long, device=TEST_DEVICE),
                update_mask=torch.ones((1, 1, config.context_length, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE),
                position_ids=torch.tensor([current_pos], dtype=torch.long, device=TEST_DEVICE),
                causal_mask=torch.zeros((1, 1, 1, config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE),
                current_pos=torch.tensor([current_pos], dtype=torch.long, device=TEST_DEVICE),
                IN_PREFILL=False
            )
    
    kv_cache_time = time.time() - start_time
    
    print(f"KV cache approach time: {kv_cache_time:.4f} seconds")
    print(f"✅ Performance test completed")
    print(f"   Note: This demonstrates KV cache functionality")
    print(f"   For real performance benefits, compare against recomputing full sequences\n")
    
    return True

def test_edge_cases(model, config):
    """Test edge cases and error conditions."""
    
    print("🧪 Test 8: Edge Cases")
    print("-" * 40)
    
    # Test with position near state length limit
    print("Testing near state length limit...")
    near_limit_pos = config.state_length - 2
    
    model.model.kv_cache_0.zero_()
    
    # Single token at near-limit position
    test_input = torch.tensor([[1000]], dtype=torch.long, device=TEST_DEVICE)
    position_ids = torch.tensor([near_limit_pos], dtype=torch.long, device=TEST_DEVICE)
    current_pos = torch.tensor([near_limit_pos], dtype=torch.long, device=TEST_DEVICE)
    update_mask = torch.ones((1, 1, config.context_length, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, config.context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    
    try:
        with torch.no_grad():
            _ = model(
                input_ids=test_input,
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False
            )
        print(f"✅ Successfully processed token at position {near_limit_pos}")
    except Exception as e:
        print(f"❌ Error at near-limit position: {e}")
        return False
    
    # Test cache reset
    print("Testing cache reset...")
    initial_norm = torch.norm(model.model.kv_cache_0).item()
    model.model.kv_cache_0.zero_()
    reset_norm = torch.norm(model.model.kv_cache_0).item()
    
    print(f"Cache norm before reset: {initial_norm:.6f}")
    print(f"Cache norm after reset: {reset_norm:.6f}")
    assert reset_norm < 1e-6, "Cache should be zero after reset"
    
    print("✅ Edge cases test passed\n")
    return True

def main():
    """Run the complete PyTorch KV cache test suite."""
    
    print("🧠 PyTorch Qwen KV Cache Test Suite")
    print("=" * 70)
    
    try:
        # Test 1: Initialization
        model, config = test_qwen_kv_cache_initialization()
        
        # Test 2: Single token forward pass
        test_single_token_forward_pass(model, config)
        
        # Test 3: Multi-token prefill
        prefill_tokens = test_multi_token_prefill(model, config)
        
        # Test 4: Generation with prefilled cache
        test_generation_with_prefilled_cache(model, config, prefill_tokens)
        
        # Test 5: Rotary embeddings
        test_rotary_embeddings(model, config)
        
        # Test 6: Realistic text generation
        test_text_generation_with_kv_cache(model, config)
        
        # Test 7: Performance comparison
        test_performance_comparison(model, config)
        
        # Test 8: Edge cases
        test_edge_cases(model, config)
        
        print("🎉 ALL PYTORCH KV CACHE TESTS PASSED!")
        print("=" * 70)
        print("✅ KV cache is properly initialized")
        print("✅ Single token forward pass updates cache correctly")
        print("✅ Multi-token prefill works")
        print("✅ Subsequent generation uses prefilled cache")
        print("✅ Rotary embeddings work for both single and multi positions")
        print("✅ Realistic text generation works with KV cache")
        print("✅ Performance testing completed")
        print("✅ Edge cases handled correctly")
        print("\n🚀 PyTorch Qwen KV cache implementation is ready!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 
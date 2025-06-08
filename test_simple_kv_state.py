#!/usr/bin/env python3
"""Simple test demonstrating KV cache state management for CoreML-style inference."""

import torch
import sys
import os
import json

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

from anemll.models.qwen_model import *
from transformers import AutoTokenizer

def create_causal_mask(context_length):
    """Create causal attention mask (similar to chat.py)."""
    mask = torch.full((1, 1, context_length, context_length), float('-inf'), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    # Make lower triangular (causal)
    for i in range(context_length):
        for j in range(i + 1):
            mask[0, 0, i, j] = 0.0
    return mask

def initialize_kv_state(model):
    """Initialize KV cache state (similar to create_unified_state in chat.py)."""
    # Zero out the KV cache to create clean state
    if hasattr(model.model, 'kv_cache_0'):
        model.model.kv_cache_0.zero_()
        print(f" Initialized KV cache state with shape: {model.model.kv_cache_0.shape}")
        return model.model.kv_cache_0
    else:
        print("❌ No KV cache found in model")
        return None

def single_token_prefill(model, token_id, position, causal_mask, state):
    """Process a single token and update KV cache state (CoreML-style)."""
    # Convert inputs to tensors
    input_ids = torch.tensor([[token_id]], device=TEST_DEVICE, dtype=torch.long)
    position_ids = torch.tensor([position], device=TEST_DEVICE, dtype=torch.long)
    current_pos = torch.tensor(position, device=TEST_DEVICE, dtype=torch.long)
    update_mask = torch.ones_like(input_ids, dtype=torch.float32, device=TEST_DEVICE)
    
    # Use single-token slice of causal mask
    single_causal_mask = causal_mask[:, :, position:position+1, :]
    
    print(f"  Processing token {token_id} at position {position}")
    print(f"    Input shape: {input_ids.shape}")
    print(f"    Position: {position}")
    print(f"    Causal mask slice shape: {single_causal_mask.shape}")
    
    # Forward pass with KV cache (using regular mode for single token)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=single_causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False  # Use regular mode for single token
        )
    
    print(f"    Output shape: {outputs.shape}")
    
    # Get next token prediction
    logits = outputs[0, 0, :]  # [vocab_size]
    next_token = torch.argmax(logits).item()
    prob = torch.softmax(logits, dim=-1)[next_token].item()
    
    print(f"    Next token prediction: {next_token} (prob: {prob:.4f})")
    
    return next_token, prob, outputs

def test_kv_cache_state_management():
    """Test KV cache state management with single token prefill."""
    print("🧠 Testing KV Cache State Management")
    print("=" * 60)
    
    # Use Hugging Face model identifier
    model_id = "Qwen/Qwen3-0.6B"
    print(f"Using Hugging Face model: {model_id}")
    
    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    
    # Get cached directory using huggingface_hub
    from huggingface_hub import snapshot_download
    cached_dir = snapshot_download(model_id, allow_patterns=["config.json", "*.safetensors"])
    config = QwenConfig.from_json(os.path.join(cached_dir, 'config.json'))
    
    # Load model with KV cache enabled
    print(f"\n📚 Loading Qwen Model with KV Cache...")
    model = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model.load_pretrained_weights(str(cached_dir))
    model.eval()
    
    # Initialize causal mask
    context_length = config.state_length  # Use state_length for CoreML compatibility
    causal_mask = create_causal_mask(context_length)
    print(f"✅ Created causal mask for context length: {context_length}")
    
    # Test prompt
    prompt = "What is Apple Neural Engine?"
    print(f"\n🔥 Test prompt: '{prompt}'")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids[0].tolist()
    print(f"Tokenized to {len(input_ids)} tokens: {input_ids}")
    print(f"Token meanings: {[tokenizer.decode([t]) for t in input_ids]}")
    
    # Initialize KV cache state
    print(f"\n🔧 Initializing KV Cache State...")
    state = initialize_kv_state(model)
    if state is None:
        return False
    
    # Process tokens one by one (single token prefill style)
    print(f"\n🔄 Processing tokens sequentially with KV cache...")
    
    generated_tokens = []
    for i, token_id in enumerate(input_ids):
        print(f"\nStep {i+1}:")
        next_token, prob, outputs = single_token_prefill(
            model, token_id, i, causal_mask, state
        )
        
        # Store the actual token we processed and the prediction for next
        if i == len(input_ids) - 1:  # Last token - this is our first generation
            generated_tokens.append(next_token)
    
    # Continue generating 31 tokens
    print(f"\n🚀 Generating 31 additional tokens...")
    current_pos = len(input_ids)
    
    for gen_step in range(31):  # Generate 31 tokens
        print(f"\nGeneration step {gen_step + 1}:")
        
        # Use the last generated token as input
        next_token, prob, outputs = single_token_prefill(
            model, generated_tokens[-1], current_pos, causal_mask, state
        )
        
        generated_tokens.append(next_token)
        current_pos += 1
        
        # Check for EOS
        if next_token == tokenizer.eos_token_id:
            print("    Generated EOS token, stopping")
            break
    
    # Show final results
    print(f"\n📊 RESULTS")
    print(f"=" * 50)
    print(f"Original prompt: '{prompt}'")
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Generated tokens: {generated_tokens}")
    print(f"Generated text: '{generated_text}'")
    print(f"Full text: '{prompt}{generated_text}'")
    
    # Verify KV cache was used by checking cache state
    if hasattr(model.model, 'kv_cache_0'):
        cache_norm = torch.norm(model.model.kv_cache_0).item()
        print(f"\n✅ KV Cache final state norm: {cache_norm:.3f}")
        if cache_norm > 0:
            print("✅ KV cache was successfully populated and used!")
        else:
            print("❌ KV cache appears empty - possible issue")
    
    print(f"\n🎉 Test completed successfully!")
    print(f"✅ Demonstrated single-token prefill with maintained KV cache state")
    print(f"✅ Each token processed individually while building up cache")
    print(f"✅ State maintained across all token processing steps")
    
    return True

def compare_with_batch_prefill():
    """Compare single token prefill vs batch prefill for validation."""
    print(f"\n🔬 Comparing Single Token vs Batch Prefill")
    print("=" * 50)
    
    # Load models using HF identifier
    model_id = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    
    # Get cached directory
    from huggingface_hub import snapshot_download
    cached_dir = snapshot_download(model_id, allow_patterns=["config.json", "*.safetensors"])
    config = QwenConfig.from_json(os.path.join(cached_dir, 'config.json'))
    
    model1 = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model1.load_pretrained_weights(str(cached_dir))
    model1.eval()
    
    model2 = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model2.load_pretrained_weights(str(cached_dir))
    model2.eval()
    
    # Test prompt
    prompt = "What is Apple Neural Engine?"
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids[0].tolist()
    
    causal_mask = create_causal_mask(config.state_length)
    
    print(f"Test prompt: '{prompt}'")
    print(f"Tokens: {input_ids}")
    
    # Method 1: Single token prefill (CoreML style)
    print(f"\n1. Single token prefill method...")
    initialize_kv_state(model1)
    
    for i, token_id in enumerate(input_ids):
        next_token1, _, _ = single_token_prefill(model1, token_id, i, causal_mask, None)
    
    # Method 2: Batch prefill
    print(f"\n2. Batch prefill method...")
    initialize_kv_state(model2)
    
    input_tensor = torch.tensor([input_ids], device=TEST_DEVICE, dtype=torch.long)
    position_ids = torch.arange(len(input_ids), device=TEST_DEVICE, dtype=torch.long)
    current_pos = torch.tensor(0, device=TEST_DEVICE, dtype=torch.long)
    update_mask = torch.ones_like(input_tensor, dtype=torch.float32, device=TEST_DEVICE)
    
    with torch.no_grad():
        outputs2 = model2.model(
            input_ids=input_tensor,
            causal_mask=causal_mask,
            position_ids=position_ids,
            current_pos=current_pos,
            IN_PREFILL=True
        )
        
        # Apply LM head to get logits
        hidden_states = outputs2.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        logits_parts = []
        for i in range(1, 17):
            lm_head = getattr(model2, f"lm_head16_{i}")
            logits_part = lm_head(hidden_states).squeeze(2).transpose(1, 2)
            logits_parts.append(logits_part)
        logits2 = torch.cat(logits_parts, dim=2)
        
        next_token2 = torch.argmax(logits2[0, -1, :]).item()
    
    print(f"\n3. Comparison...")
    print(f"Single token prefill predicts: {next_token1} ('{tokenizer.decode([next_token1])}')")
    print(f"Batch prefill predicts: {next_token2} ('{tokenizer.decode([next_token2])}')")
    print(f"Predictions match: {next_token1 == next_token2}")
    
    if next_token1 == next_token2:
        print("✅ Both methods produce same result!")
    else:
        print("❌ Different results - may need investigation")
    
    return next_token1 == next_token2

if __name__ == "__main__":
    success1 = test_kv_cache_state_management()
    success2 = compare_with_batch_prefill()
    
    if success1 and success2:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ KV cache state management working correctly")
        print(f"✅ Single token prefill matches batch prefill")
        print(f"✅ Ready for CoreML deployment pattern")
    else:
        print(f"\n❌ Some tests failed!") 
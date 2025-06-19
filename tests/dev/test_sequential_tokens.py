#!/usr/bin/env python3
"""Test sequential token processing without KV cache."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import json

# Add the anemll directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anemll"))

from anemll.models.qwen_model import *

def test_sequential_token_processing():
    """Test processing tokens sequentially without KV cache."""
    
    print("🔄 Testing Sequential Token Processing (No KV Cache)")
    print("=" * 70)
    
    # Use Hugging Face model identifier
    model_id = "Qwen/Qwen3-0.6B"
    print(f"Using Hugging Face model: {model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    
    # Download model files using huggingface_hub
    from huggingface_hub import snapshot_download
    cached_dir = snapshot_download(model_id, allow_patterns=["config.json", "*.safetensors"])
    print(f"Model cached at: {cached_dir}")
    
    # Test prompt
    prompt = "What is Apple Neural Engine?"
    print(f"\nPrompt: '{prompt}'")
    
    # Tokenize 
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids
    prompt_tokens = input_ids[0].tolist()
    print(f"Tokenized to {len(prompt_tokens)} tokens: {prompt_tokens}")
    print(f"Token meanings: {[tokenizer.decode([t]) for t in prompt_tokens]}")
    
    # Load original Transformers model for reference
    print(f"\n📚 Loading Original Transformers Model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Load our PyTorch model
    print(f"\n🔧 Loading Our PyTorch Model...")
    config_path = os.path.join(cached_dir, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = QwenConfig(**config_dict)
    our_model = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=True)
    our_model.load_pretrained_weights(str(cached_dir))
    our_model.eval()
    
    # Test original model (for reference)
    print(f"\n🔥 Original Model: Next Token After Full Prompt...")
    with torch.no_grad():
        original_output = original_model(input_ids)
        original_logits = original_output.logits[0, -1, :]  # Last token logits
        original_top_token = torch.argmax(original_logits).item()
    
    print(f"Original top token: {original_top_token} ('{tokenizer.decode([original_top_token])}')")
    
    # Test our model with sequential processing
    print(f"\n🔄 Our Model: Sequential Token Processing...")
    
    # Start with empty sequence, add tokens one by one
    current_tokens = []
    
    for i, token in enumerate(prompt_tokens):
        current_tokens.append(token)
        current_sequence = torch.tensor([current_tokens], dtype=torch.long)
        seq_len = len(current_tokens)
        
        print(f"\nStep {i+1}: Processing token {token} ('{tokenizer.decode([token])}')")
        print(f"  Current sequence: {current_tokens}")
        print(f"  Sequence length: {seq_len}")
        
        # Create inputs for current sequence
        position_ids = torch.arange(seq_len, dtype=torch.long)
        update_mask = torch.zeros((1, 1, 256, 1), dtype=torch.float16)
        causal_mask = torch.full((1, 1, seq_len, 256), -torch.inf, dtype=torch.float16)
        for j in range(seq_len):
            causal_mask[:, :, j, :j+1] = 0  # Allow attention to previous tokens
        current_pos = torch.tensor([seq_len - 1], dtype=torch.long)
        
        with torch.no_grad():
            our_output = our_model(
                input_ids=current_sequence,
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False
            )
            
            # Get prediction for next token (logits for the last processed token)
            last_token_logits = our_output[0, -1, :]  # Last token logits
            top_token = torch.argmax(last_token_logits).item()
            
            print(f"  Next token prediction: {top_token} ('{tokenizer.decode([top_token])}')")
    
    # Final prediction after processing full prompt
    print(f"\n🎯 FINAL COMPARISON")
    print(f"=" * 50)
    print(f"Original model predicts: {original_top_token} ('{tokenizer.decode([original_top_token])}')")
    print(f"Our model (sequential) predicts: {top_token} ('{tokenizer.decode([top_token])}')")
    print(f"Tokens match: {top_token == original_top_token}")
    
    if top_token == original_top_token:
        print("✅ SUCCESS! Sequential processing works correctly!")
        print("   This means we can generate tokens without KV cache.")
    else:
        print("❌ Still different. The issue may be deeper than KV cache.")
    
    # Test generation: add a few tokens using sequential approach
    print(f"\n🚀 Testing Token Generation (Sequential)...")
    
    max_new_tokens =32
    generated_tokens = current_tokens.copy()
    
    for gen_step in range(max_new_tokens):
        # Get current sequence
        current_sequence = torch.tensor([generated_tokens], dtype=torch.long)
        seq_len = len(generated_tokens)
        
        # Create inputs
        position_ids = torch.arange(seq_len, dtype=torch.long)
        update_mask = torch.zeros((1, 1, 256, 1), dtype=torch.float16)
        causal_mask = torch.full((1, 1, seq_len, 256), -torch.inf, dtype=torch.float16)
        for j in range(seq_len):
            causal_mask[:, :, j, :j+1] = 0
        current_pos = torch.tensor([seq_len - 1], dtype=torch.long)
        
        # Generate next token
        with torch.no_grad():
            our_output = our_model(
                input_ids=current_sequence,
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False
            )
            
            last_token_logits = our_output[0, -1, :]
            next_token = torch.argmax(last_token_logits).item()
            generated_tokens.append(next_token)
            
            print(f"Generated token {gen_step+1}: {next_token} ('{tokenizer.decode([next_token])}')")
    
    # Show final result
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    new_tokens_text = tokenizer.decode(generated_tokens[len(prompt_tokens):], skip_special_tokens=True)
    
    print(f"\n📝 GENERATION RESULTS")
    print(f"=" * 50)
    print(f"Original prompt: '{prompt}'")
    print(f"Generated continuation: '{new_tokens_text}'")
    print(f"Full text: '{generated_text}'")
    
    # Compare with original model generation
    print(f"\n🔥 Original Model Generation for Comparison...")
    with torch.no_grad():
        original_generated = original_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    original_new_tokens = original_generated[0][len(prompt_tokens):]
    original_continuation = tokenizer.decode(original_new_tokens, skip_special_tokens=True)
    
    print(f"Original continuation: '{original_continuation}'")
    print(f"Continuations match: {new_tokens_text.strip() == original_continuation.strip()}")
    
    return top_token == original_top_token

if __name__ == "__main__":
    test_sequential_token_processing() 
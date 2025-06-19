#!/usr/bin/env python3
"""Step-by-step comparison between original Qwen and our implementation."""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import glob
import os

# Add our models to path
sys.path.append('.')
from anemll.models.qwen_model import QwenForCausalLM, QwenConfig

def test_step_by_step_comparison():
    """Compare our implementation step by step with the original."""
    
    print("🔍 Step-by-step comparison: Original vs Our Implementation")
    print("="*70)
    
    # Load original model
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    if not model_dirs:
        print("❌ Error: Qwen model not found in cache")
        return False
    
    model_dir = model_dirs[0]
    print(f"Loading original model from: {model_dir}")
    
    # Load tokenizer and original model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    original_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16
    )
    
    device = "cpu"  # Use CPU for debugging
    original_model = original_model.to(device)
    original_model.eval()
    
    # Load our model
    print(f"Loading our implementation...")
    config = QwenConfig.from_json(f"{model_dir}/config.json")
    our_model = QwenForCausalLM(config)
    our_model.load_pretrained_weights(model_dir)
    our_model.eval()
    
    # Test prompt
    prompt = "What is Apple Neural Engine?"
    print(f"\nPrompt: '{prompt}'")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids.to(device)
    seq_len = input_ids.shape[1]
    
    print(f"Tokenized to {seq_len} tokens: {input_ids.tolist()[0]}")
    print(f"Token meanings: {[tokenizer.decode([t]) for t in input_ids[0]]}")
    
    # Create inputs for our model
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    current_pos = torch.tensor([seq_len - 1], dtype=torch.long)
    
    # Create causal mask
    causal_mask = torch.full((1, 1, seq_len, seq_len), -torch.inf, dtype=torch.float16)
    for i in range(seq_len):
        causal_mask[:, :, i, :i+1] = 0
    
    # Update mask (dummy for our implementation)
    update_mask = torch.ones_like(input_ids, dtype=torch.bool)
    
    print(f"\n🎯 FORWARD PASS COMPARISON:")
    print("-" * 50)
    
    # Original model forward pass
    with torch.no_grad():
        original_outputs = original_model(input_ids)
        original_logits = original_outputs.logits  # [1, seq_len, vocab_size]
        original_last_logits = original_logits[0, -1, :]  # Last token logits
        original_next_token = torch.argmax(original_last_logits).item()
        original_probs = torch.softmax(original_last_logits, dim=-1)
        original_top5 = torch.topk(original_probs, 5)
        
        print(f"✅ Original model:")
        print(f"  Logits shape: {original_logits.shape}")
        print(f"  Next token: {original_next_token} ('{tokenizer.decode([original_next_token])}')")
        print(f"  Top 5 tokens: {original_top5.indices.tolist()}")
        print(f"  Top 5 probs: {original_top5.values.tolist()}")
        print(f"  Logits range: [{original_last_logits.min().item():.3f}, {original_last_logits.max().item():.3f}]")
    
    # Our model forward pass
    with torch.no_grad():
        try:
            our_logits = our_model(
                input_ids=input_ids,
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False
            )
            our_last_logits = our_logits[0, -1, :]  # Last token logits
            our_next_token = torch.argmax(our_last_logits).item()
            our_probs = torch.softmax(our_last_logits, dim=-1)
            our_top5 = torch.topk(our_probs, 5)
            
            print(f"✅ Our model:")
            print(f"  Logits shape: {our_logits.shape}")
            print(f"  Next token: {our_next_token} ('{tokenizer.decode([our_next_token])}')")
            print(f"  Top 5 tokens: {our_top5.indices.tolist()}")
            print(f"  Top 5 probs: {our_top5.values.tolist()}")
            print(f"  Logits range: [{our_last_logits.min().item():.3f}, {our_last_logits.max().item():.3f}]")
            
        except Exception as e:
            print(f"❌ Our model failed: {e}")
            return False
    
    # Compare logits
    print(f"\n🔍 LOGITS COMPARISON:")
    print("-" * 50)
    
    # Check if shapes match
    if original_logits.shape != our_logits.shape:
        print(f"❌ Shape mismatch: original {original_logits.shape} vs ours {our_logits.shape}")
        return False
    
    # Compare last token logits
    logits_diff = torch.abs(original_last_logits - our_last_logits)
    max_diff = logits_diff.max().item()
    mean_diff = logits_diff.mean().item()
    
    print(f"Max logits difference: {max_diff:.6f}")
    print(f"Mean logits difference: {mean_diff:.6f}")
    
    # Check top tokens alignment
    tokens_match = (original_next_token == our_next_token)
    print(f"Next token matches: {tokens_match}")
    
    if not tokens_match:
        print(f"❌ Token mismatch!")
        print(f"  Original: {original_next_token} ('{tokenizer.decode([original_next_token])}')")
        print(f"  Ours: {our_next_token} ('{tokenizer.decode([our_next_token])}')")
        
        # Show logits for these specific tokens
        print(f"  Original logit for token {original_next_token}: {original_last_logits[original_next_token].item():.6f}")
        print(f"  Our logit for token {original_next_token}: {our_last_logits[original_next_token].item():.6f}")
        print(f"  Original logit for token {our_next_token}: {original_last_logits[our_next_token].item():.6f}")
        print(f"  Our logit for token {our_next_token}: {our_last_logits[our_next_token].item():.6f}")
    
    # Generate a few tokens to see behavior
    print(f"\n🎯 MULTI-TOKEN GENERATION:")
    print("-" * 50)
    
    def generate_tokens(model, is_original=True, max_tokens=5):
        """Generate a few tokens for comparison."""
        current_ids = input_ids.clone()
        generated = []
        
        for step in range(max_tokens):
            if is_original:
                with torch.no_grad():
                    outputs = model(current_ids)
                    next_token_logits = outputs.logits[0, -1, :]
            else:
                # Update position_ids and other inputs
                seq_len = current_ids.shape[1]
                pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
                curr_pos = torch.tensor([seq_len - 1], dtype=torch.long)
                
                # Create causal mask for current sequence length
                mask = torch.full((1, 1, seq_len, seq_len), -torch.inf, dtype=torch.float16)
                for i in range(seq_len):
                    mask[:, :, i, :i+1] = 0
                
                update_mask = torch.ones_like(current_ids, dtype=torch.bool)
                
                with torch.no_grad():
                    logits = model(
                        input_ids=current_ids,
                        update_mask=update_mask,
                        position_ids=pos_ids,
                        causal_mask=mask,
                        current_pos=curr_pos,
                        IN_PREFILL=False
                    )
                    next_token_logits = logits[0, -1, :]
            
            next_token = torch.argmax(next_token_logits).item()
            generated.append(next_token)
            
            # Add token to sequence
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            current_ids = torch.cat([current_ids, next_token_tensor], dim=1)
            
            # Stop if we hit EOS
            if next_token == tokenizer.eos_token_id:
                break
        
        return generated
    
    # Generate with both models
    original_generated = generate_tokens(original_model, is_original=True)
    our_generated = generate_tokens(our_model, is_original=False)
    
    print(f"Original generated: {original_generated}")
    print(f"Original tokens: {[tokenizer.decode([t]) for t in original_generated]}")
    print(f"Our generated: {our_generated}")
    print(f"Our tokens: {[tokenizer.decode([t]) for t in our_generated]}")
    
    # Check if sequences match
    sequences_match = (original_generated == our_generated)
    print(f"Generated sequences match: {sequences_match}")
    
    if sequences_match:
        print("🎉 SUCCESS: Models produce identical outputs!")
        return True
    else:
        print("❌ FAILURE: Models produce different outputs")
        
        # Show first difference
        for i, (orig, ours) in enumerate(zip(original_generated, our_generated)):
            if orig != ours:
                print(f"First difference at position {i}:")
                print(f"  Original: {orig} ('{tokenizer.decode([orig])}')")
                print(f"  Ours: {ours} ('{tokenizer.decode([ours])}')")
                break
        
        return False

if __name__ == "__main__":
    success = test_step_by_step_comparison()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed - investigate differences")
        sys.exit(1) 
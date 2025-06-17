#!/usr/bin/env python3
"""Minimal test with official Hugging Face Qwen3 model to establish ground truth."""

import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

def test_minimal_official_qwen3():
    print("🔬 Minimal Official Qwen3 Test - Single Token (Ground Truth)")
    print("=" * 65)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    
    # Load official Qwen3 model from the same local path
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    print(f"Loading official Qwen3 model from: {model_path}")
    
    # Load the official transformers model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.float16
    )
    model = model.to("cpu")  # Move to CPU manually
    model.eval()
    print("✅ Official Qwen3 model loaded successfully")
    
    # Test with single meaningful token (same as other tests)
    test_token = 3838  # "What"
    print(f"Testing with single token: {test_token} ('{tokenizer.decode([test_token])}')")
    
    # Create minimal inputs (context length 256, but only first position used)
    CONTEXT_LENGTH = 256
    PAD_TOKEN = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # Input: single token at position 0, rest padding
    input_ids = torch.tensor([[test_token] + [PAD_TOKEN] * (CONTEXT_LENGTH - 1)], dtype=torch.long)
    
    # Create attention mask (1s for real tokens, 0s for padding)
    attention_mask = torch.zeros_like(input_ids)
    attention_mask[0, 0] = 1  # Only first position is real
    
    print(f"\n📊 Inputs:")
    print(f"  input_ids[0][:10]: {input_ids[0][:10]}")
    print(f"  attention_mask[0][:10]: {attention_mask[0][:10]}")
    print(f"  Context length: {CONTEXT_LENGTH}")
    print(f"  Real token at position 0: {test_token} ('What')")
    print(f"  ℹ️  In sequence generation with official model:")
    print(f"     Position 0 processes 'What' → predicts next token")
    print(f"     Official model outputs logits for ALL positions")
    print(f"     We extract logits from position 0 to predict position 1")
    
    # Run inference with official model
    print(f"\n🚀 Running official Qwen3 inference...")
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get logits from the model
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        print(f"  Official model logits shape: {logits.shape}")
        
        # Extract logits from position 0 (to predict what comes at position 1)
        # This matches our CoreML test approach
        position_to_extract = 0
        logits_at_pos = logits[0, position_to_extract, :]  # [vocab_size]
        print(f"  Extracted logits from position {position_to_extract}")
        print(f"  Logits for prediction shape: {logits_at_pos.shape}")
    
    # Get top 10 predictions
    top_indices = torch.argsort(logits_at_pos, descending=True)[:10]
    print(f"\n🔍 Top 10 predictions after 'What' (Official Qwen3):")
    for i, idx in enumerate(top_indices):
        token_text = tokenizer.decode([int(idx)])
        print(f"  {i+1:2d}. Token {int(idx):5d}: '{token_text}' (logit: {logits_at_pos[idx]:.3f})")
    
    print(f"\n🤔 Expected reasonable predictions after 'What':")
    print(f"  Common completions: ' is', ' are', ' can', ' will', ' do', etc.")
    
    # Check if any top predictions make sense
    reasonable_tokens = [' is', ' are', ' can', ' will', ' do', ' does', ' would', ' should']
    found_reasonable = False
    reasonable_found = []
    
    for i, idx in enumerate(top_indices[:10]):  # Check top 10
        token_text = tokenizer.decode([int(idx)])
        if token_text.strip().lower() in [t.strip().lower() for t in reasonable_tokens]:
            print(f"  ✅ Found reasonable prediction at rank {i+1}: '{token_text}'")
            reasonable_found.append((i+1, token_text))
            found_reasonable = True
    
    if not found_reasonable:
        print(f"  ❌ No reasonable predictions in top 10 - unexpected for official model")
    else:
        print(f"  ✅ Found {len(reasonable_found)} reasonable predictions in top 10")
    
    # Additional analysis - check logit distribution
    print(f"\n📈 Logit Analysis:")
    print(f"  Logit range: [{logits_at_pos.min():.3f}, {logits_at_pos.max():.3f}]")
    print(f"  Logit std: {logits_at_pos.std():.3f}")
    
    # Check if top predictions are clustered or spread out
    top_5_logits = [logits_at_pos[idx].item() for idx in top_indices[:5]]
    print(f"  Top 5 logit values: {[f'{l:.3f}' for l in top_5_logits]}")
    
    return found_reasonable, top_indices[:5].tolist(), [logits_at_pos[idx].item() for idx in top_indices[:5]]

def compare_with_other_tests():
    """Compare results with our other test implementations."""
    print(f"\n🔄 COMPARISON SUMMARY:")
    print(f"=" * 50)
    
    # Run our official test
    success, top_tokens, top_logits = test_minimal_official_qwen3()
    
    print(f"\n📊 Official Qwen3 Results:")
    print(f"  Success: {success}")
    print(f"  Top 5 tokens: {top_tokens}")
    print(f"  Top 5 logits: {[f'{l:.3f}' for l in top_logits]}")
    
    print(f"\n🔗 This should be compared with:")
    print(f"  1. test_minimal_pytorch_qwen.py (our custom implementation)")
    print(f"  2. test_minimal_coreml.py (CoreML conversion)")
    print(f"")
    print(f"Expected hierarchy of correctness:")
    print(f"  Official Qwen3 ≈ Our PyTorch qwen_model.py >> CoreML conversion")
    print(f"")
    print(f"If official and our PyTorch models match → conversion issue confirmed")
    print(f"If they don't match → our implementation has bugs")
    
    return success

if __name__ == "__main__":
    success = compare_with_other_tests()
    print(f"\n{'🏆 GROUND TRUTH ESTABLISHED' if success else '⚠️  UNEXPECTED GROUND TRUTH'}")
    print(f"Official model {'works as expected' if success else 'shows unexpected behavior'}") 
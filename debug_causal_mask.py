#!/usr/bin/env python3
"""Debug the causal mask to see if it's causing the attention issue."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'anemll'))
from anemll.models.qwen_model import *
import glob
from transformers import AutoTokenizer

def debug_causal_mask():
    """Debug the causal mask being applied to attention."""
    
    print("🔍 Causal Mask Debug: Understanding Attention Patterns")
    print("=" * 70)
    
    # Setup basic mask
    causal_mask = torch.zeros((1, 1, 1, 512), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"Causal mask type: {causal_mask.dtype}")
    print(f"Causal mask values (first few): {causal_mask[0, 0, 0, :10].tolist()}")
    
    # Test what the mask looks like for our scenario
    q_seq_len = 1  # Single token query
    k_seq_len = 3  # 3 positions in cache
    
    print(f"\nFor q_seq_len={q_seq_len}, k_seq_len={k_seq_len}:")
    mask_slice = causal_mask[:, :, :q_seq_len, :k_seq_len]
    print(f"Mask slice shape: {mask_slice.shape}")
    print(f"Mask slice values: {mask_slice[0, 0, 0, :].tolist()}")
    
    # Compare with what a proper causal mask should look like
    print(f"\n--- PROPER CAUSAL MASK COMPARISON ---")
    
    # For autoregressive generation, a proper causal mask should:
    # - Allow current token to attend to ALL previous tokens + itself
    # - Since we're at position 2, we should attend to positions 0, 1, 2
    
    # Create a proper causal mask for demonstration
    proper_mask = torch.zeros((1, 1, 1, 3), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    # In causal attention, zeros usually mean "allowed" and large negative values mean "masked"
    # Since we want to attend to all positions 0, 1, 2, we keep all zeros
    
    print(f"Proper causal mask for position 2: {proper_mask[0, 0, 0, :].tolist()}")
    print("(All zeros = all positions allowed)")
    
    # Test what happens with different mask values
    print(f"\n--- TESTING DIFFERENT MASK VALUES ---")
    
    # Simulate attention weights without mask
    raw_scores = torch.tensor([1.0, 0.8, 1.2])  # Arbitrary raw attention scores
    print(f"Raw attention scores: {raw_scores.tolist()}")
    
    # Apply no mask (all zeros)
    masked_scores_zero = raw_scores + torch.zeros(3)
    attention_zero = torch.softmax(masked_scores_zero, dim=-1)
    print(f"With zero mask: {attention_zero.tolist()}")
    
    # Apply slight negative mask to position 2 (current token)
    test_mask = torch.tensor([0.0, 0.0, -1.0])  # Penalize current token
    masked_scores_penalized = raw_scores + test_mask
    attention_penalized = torch.softmax(masked_scores_penalized, dim=-1)
    print(f"With current token penalized: {attention_penalized.tolist()}")
    
    # Apply strong negative mask to position 2
    test_mask_strong = torch.tensor([0.0, 0.0, -10.0])  # Strongly penalize current token
    masked_scores_strong = raw_scores + test_mask_strong
    attention_strong = torch.softmax(masked_scores_strong, dim=-1)
    print(f"With current token strongly penalized: {attention_strong.tolist()}")
    
    print(f"\n--- COMPARING WITH NO-CACHE MODEL ---")
    
    # What should the attention pattern look like in a no-cache model?
    # In no-cache, when processing sequence [The, capital, <eos>], the attention for <eos> should be:
    # - <eos> can attend to The, capital, and itself
    # - But typically, content tokens get higher attention than special tokens
    
    print("In a no-cache model processing ['The', ' capital', '<|endoftext|>']:")
    print("- Token '<|endoftext|>' can attend to all previous tokens")
    print("- Typically, attention focuses more on content words ('The', ' capital')")
    print("- The model should use context to predict the next token")
    
    print(f"\n--- KEY INSIGHT ---")
    print("Our KV cache attention weights: [0.325, 0.305, 0.371]")
    print("- Position 0 ('The'): 32.5%")
    print("- Position 1 (' capital'): 30.5%") 
    print("- Position 2 ('<|endoftext|>'): 37.1% ← HIGHEST!")
    print("")
    print("The problem: Current placeholder token gets HIGHEST attention!")
    print("This means the model is focusing on the placeholder instead of context.")
    print("")
    print("Possible solutions:")
    print("1. Check if causal mask is incorrect")
    print("2. Investigate why current token gets high attention scores")
    print("3. Consider if the placeholder token choice affects attention")

if __name__ == "__main__":
    from anemll.models.qwen_model import TEST_DEVICE, MODEL_DTYPE
    debug_causal_mask() 
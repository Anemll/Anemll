#!/usr/bin/env python3
"""Debug script to compare ANE softmax vs standard softmax behavior."""

import torch
import torch.nn.functional as F
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from anemll.models.qwen2_5_model import ANESoftmax

def test_softmax_equivalence():
    """Test if ANE softmax produces equivalent results to standard softmax."""
    print("=" * 80)
    print("TESTING ANE SOFTMAX EQUIVALENCE")
    print("=" * 80)
    
    # Create test tensors with different characteristics
    test_cases = [
        ("Small values", torch.randn(2, 4, 8) * 0.1),
        ("Large values", torch.randn(2, 4, 8) * 10),
        ("Mixed values", torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[10.0, -5.0, 0.1, 2.0]]])),
        ("Attention-like", torch.randn(1, 14, 4, 4) * 0.125),  # Similar to actual attention weights
    ]
    
    ane_softmax = ANESoftmax(dim=-1)
    
    for name, x in test_cases:
        print(f"\n--- {name} ---")
        print(f"Input shape: {x.shape}")
        print(f"Input range: [{x.min().item():.3f}, {x.max().item():.3f}]")
        
        # Standard softmax
        std_result = F.softmax(x, dim=-1, dtype=torch.float32)
        
        # ANE softmax
        ane_result = ane_softmax(x, dtype=torch.float32)
        
        # Manual ANE-style softmax (subtract mean then softmax)
        x_float = x.to(torch.float32)
        mean = x_float.mean(-1, keepdim=True)
        manual_result = F.softmax(x_float - mean, dim=-1, dtype=torch.float32)
        
        print(f"Standard softmax sum: {std_result.sum(-1).flatten()[0].item():.6f}")
        print(f"ANE softmax sum: {ane_result.sum(-1).flatten()[0].item():.6f}")
        print(f"Manual ANE sum: {manual_result.sum(-1).flatten()[0].item():.6f}")
        
        # Check if ANE matches manual
        ane_vs_manual = torch.allclose(ane_result, manual_result, rtol=1e-5)
        print(f"ANE matches manual: {ane_vs_manual}")
        
        # Check if ANE matches standard
        ane_vs_std = torch.allclose(ane_result, std_result, rtol=1e-5)
        print(f"ANE matches standard: {ane_vs_std}")
        
        if not ane_vs_std:
            diff = (ane_result - std_result).abs()
            print(f"Max difference: {diff.max().item():.6f}")
            print(f"Mean difference: {diff.mean().item():.6f}")
            
            # Show a few values for comparison
            print("First few values comparison:")
            print(f"Standard: {std_result.flatten()[:5]}")
            print(f"ANE:      {ane_result.flatten()[:5]}")
            print(f"Manual:   {manual_result.flatten()[:5]}")

def test_attention_scenario():
    """Test specifically with attention-like scenarios."""
    print("\n" + "=" * 80)
    print("TESTING ATTENTION-LIKE SCENARIO")
    print("=" * 80)
    
    # Simulate attention weights before softmax
    batch_size, num_heads, seq_len = 1, 14, 4
    scale = 1.0 / (64 ** 0.5)  # head_dim = 64
    
    # Create query and key like in actual attention
    query = torch.randn(batch_size, num_heads, seq_len, 64)
    key = torch.randn(batch_size, num_heads, seq_len, 64)
    
    # Compute attention weights
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights range: [{attn_weights.min().item():.3f}, {attn_weights.max().item():.3f}]")
    
    # Add causal mask (like in the model)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * -1000, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
    attn_weights = attn_weights + causal_mask
    
    print(f"After causal mask range: [{attn_weights.min().item():.3f}, {attn_weights.max().item():.3f}]")
    
    # Test both softmax methods
    ane_softmax = ANESoftmax(dim=-1)
    
    std_result = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    ane_result = ane_softmax(attn_weights, dtype=torch.float32)
    
    print(f"Standard softmax sum per row: {std_result.sum(-1)[0, 0]}")
    print(f"ANE softmax sum per row: {ane_result.sum(-1)[0, 0]}")
    
    # Check for extreme values
    print(f"Standard max value: {std_result.max().item():.6f}")
    print(f"ANE max value: {ane_result.max().item():.6f}")
    print(f"Standard min value: {std_result.min().item():.6f}")
    print(f"ANE min value: {ane_result.min().item():.6f}")
    
    # Look at the actual probability distributions
    print("\nFirst attention head, first row (probabilities):")
    print(f"Standard: {std_result[0, 0, 0].tolist()}")
    print(f"ANE:      {ane_result[0, 0, 0].tolist()}")

if __name__ == "__main__":
    test_softmax_equivalence()
    test_attention_scenario()
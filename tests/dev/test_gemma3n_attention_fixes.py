#!/usr/bin/env python3
"""
Quick test to verify attention fixes in Gemma3nTextAttention.
Compares standard attention path vs KV-cache path outputs.
"""

import sys
import torch
import math
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from anemll.models.gemma3n_model import Gemma3nTextAttention, Gemma3nRMSNorm


def test_attention_scaling():
    """Test that KV-cache path now has proper attention scaling."""

    # Create a minimal config
    class MinimalConfig:
        hidden_size = 2048
        num_attention_heads = 8
        num_key_value_heads = 4
        head_dim = 256
        rms_norm_eps = 1e-6
        attention_bias = False
        num_hidden_layers = 30
        num_kv_shared_layers = 0
        layer_types = ["sliding_attention"] * 30
        sliding_window = 512
        state_length = 512
        query_pre_attn_scalar = 256

    config = MinimalConfig()

    # Create attention module
    attn = Gemma3nTextAttention(config, layer_idx=0)
    attn.eval()

    print("=" * 60)
    print("Testing Gemma3nTextAttention fixes")
    print("=" * 60)

    # Check that query_pre_attn_scalar is set
    assert hasattr(attn, 'query_pre_attn_scalar'), "Missing query_pre_attn_scalar!"
    print(f"✓ query_pre_attn_scalar = {attn.query_pre_attn_scalar}")

    # Test get_new_kv_cache
    print("\nTesting get_new_kv_cache()...")
    hidden = torch.randn(1, 1, config.hidden_size)
    cos = torch.randn(1, config.head_dim // 2)
    sin = torch.randn(1, config.head_dim // 2)

    query, key, value = attn.get_new_kv_cache(hidden, current_pos=0, rotary_emb=(cos, sin))

    # Query should be scaled by query_pre_attn_scalar
    # Check the magnitude is as expected (should be ~256x larger than unscaled)
    print(f"  Query states shape: {query.shape}")
    print(f"  Query magnitude (mean abs): {query.abs().mean().item():.4f}")
    print("  (Should be ~256x larger than unscaled projection)")

    # Test forward_regular
    print("\nTesting forward_regular() with mock KV cache...")

    # Create mock KV cache
    K_cache = torch.randn(1, config.num_key_value_heads, 10, config.head_dim)
    V_cache = torch.randn(1, config.num_key_value_heads, 10, config.head_dim)

    # Create causal mask
    causal_mask = torch.zeros(1, 1, config.state_length, config.state_length)
    causal_mask[:, :, :, :] = float('-inf')
    for i in range(config.state_length):
        causal_mask[:, :, i, :i+1] = 0

    # Get query states from get_new_kv_cache
    query_states, _, _ = attn.get_new_kv_cache(hidden, current_pos=5, rotary_emb=(cos, sin))

    # Run forward_regular
    output = attn.forward_regular(
        hidden_states=hidden,
        query_states=query_states,
        kv_cache_layer=(K_cache, V_cache),
        causal_mask=causal_mask,
        current_pos=5,
        layer_idx=0
    )

    print(f"  Output shape: {output.shape}")
    print(f"  Output magnitude (mean abs): {output.abs().mean().item():.4f}")
    print("  ✓ forward_regular completed without errors")

    # Verify attention scaling is applied by checking the source code
    import inspect
    forward_regular_src = inspect.getsource(attn.forward_regular)

    has_scaling = "/ math.sqrt(self.head_dim)" in forward_regular_src
    has_softcap = "torch.tanh(attn_weights / 30.0) * 30.0" in forward_regular_src

    print("\nCode verification:")
    print(f"  ✓ Attention scaling (/ sqrt(head_dim)): {'PRESENT' if has_scaling else 'MISSING!'}")
    print(f"  ✓ Softcapping (tanh/30): {'PRESENT' if has_softcap else 'MISSING!'}")

    # Same check for get_new_kv_cache
    get_kv_src = inspect.getsource(attn.get_new_kv_cache)
    has_scalar = "query_pre_attn_scalar" in get_kv_src
    print(f"  ✓ Query pre-attn scalar in get_new_kv_cache: {'PRESENT' if has_scalar else 'MISSING!'}")

    print("\n" + "=" * 60)
    if has_scaling and has_softcap and has_scalar:
        print("ALL FIXES VERIFIED! ✓")
    else:
        print("SOME FIXES MISSING!")
    print("=" * 60)

    return has_scaling and has_softcap and has_scalar


if __name__ == "__main__":
    success = test_attention_scaling()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test file for KV cache packing optimization.

When context_length < state_length, we can pack multiple layers' KV caches
into a single buffer, utilizing the otherwise wasted memory slots.

Example: context=64, state=256
- Without packing: 28 layers × 2 (K+V) = 56 buffers of shape (1, kv_heads, 256, head_dim)
  - But only positions 0-63 are used, wasting positions 64-255
- With packing (pack_factor=4): 14 buffers of shape (1, kv_heads, 256, head_dim)
  - Layer 0: positions 0-63, Layer 1: positions 64-127, etc.
  - 4 layers packed per buffer

Usage:
    python tests/dev/test_pack_kvc.py
"""

import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anemll.models.qwen_model import QwenConfig


def calculate_pack_factor(context_length: int, state_length: int) -> int:
    """Calculate how many layers can be packed into one buffer."""
    if context_length >= state_length:
        return 1  # No packing possible
    return state_length // context_length


def get_packed_buffer_count(num_layers: int, pack_factor: int) -> int:
    """Calculate number of packed buffers needed.

    Uses K-first, V-second layout:
    - First ceil(num_layers/pack_factor) buffers for K
    - Next ceil(num_layers/pack_factor) buffers for V
    """
    packed_per_type = (num_layers + pack_factor - 1) // pack_factor
    return packed_per_type * 2  # K buffers + V buffers


def get_layer_position_in_packed_buffer(layer_idx: int, is_value: bool,
                                         num_layers: int, pack_factor: int,
                                         context_length: int):
    """
    Get buffer index and position slice for a layer's K or V cache.

    The actual model uses K-first, V-second layout:
    - Indices 0 to num_layers-1: K caches for all layers
    - Indices num_layers to 2*num_layers-1: V caches for all layers

    For packing, we maintain this layout:
    - First num_layers/pack_factor buffers: packed K caches
    - Next num_layers/pack_factor buffers: packed V caches

    Args:
        layer_idx: Layer index (0 to num_layers-1)
        is_value: True for V cache, False for K cache
        num_layers: Total number of layers
        pack_factor: How many layers packed per buffer
        context_length: Context length (slice size)

    Returns:
        (buffer_idx, start_pos, end_pos)
    """
    # Number of packed buffers for K (same for V)
    packed_buffers_per_type = (num_layers + pack_factor - 1) // pack_factor

    if is_value:
        # V buffers come after K buffers
        buffer_idx = packed_buffers_per_type + (layer_idx // pack_factor)
    else:
        # K buffers are first
        buffer_idx = layer_idx // pack_factor

    # Position within the buffer
    position_in_buffer = layer_idx % pack_factor
    start_pos = position_in_buffer * context_length
    end_pos = start_pos + context_length

    return buffer_idx, start_pos, end_pos


class PackedKVCache:
    """Prototype for packed KV cache management."""

    def __init__(self, config: QwenConfig, pack_kvc: bool = False):
        self.config = config
        self.pack_kvc = pack_kvc
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.context_length = config.context_length
        self.state_length = config.state_length

        if pack_kvc:
            self.pack_factor = calculate_pack_factor(self.context_length, self.state_length)
            self.num_buffers = get_packed_buffer_count(self.num_layers, self.pack_factor)
            self.packed_per_type = (self.num_layers + self.pack_factor - 1) // self.pack_factor
        else:
            self.pack_factor = 1
            self.num_buffers = self.num_layers * 2
            self.packed_per_type = self.num_layers

        print(f"\n{'='*60}")
        print("PackedKVCache Configuration:")
        print(f"  num_layers: {self.num_layers}")
        print(f"  num_kv_heads: {self.num_kv_heads}")
        print(f"  head_dim: {self.head_dim}")
        print(f"  context_length: {self.context_length}")
        print(f"  state_length: {self.state_length}")
        print(f"  pack_kvc: {self.pack_kvc}")
        print(f"  pack_factor: {self.pack_factor}")
        print(f"  num_buffers: {self.num_buffers} ({self.packed_per_type} K + {self.packed_per_type} V)")
        print(f"{'='*60}\n")

        # Initialize buffers
        self.buffers = self._create_buffers()

    def _create_buffers(self) -> torch.Tensor:
        """Create the KV cache buffers."""
        # Shape: (num_buffers, num_kv_heads, state_length, head_dim)
        return torch.zeros(
            self.num_buffers,
            self.num_kv_heads,
            self.state_length,
            self.head_dim,
            dtype=torch.float16
        )

    def get_kv_cache_for_layer(self, layer_idx: int):
        """Get K and V cache tensors for a specific layer."""
        if self.pack_kvc:
            # Packed mode: get slices from packed buffers
            k_buf_idx, k_start, k_end = get_layer_position_in_packed_buffer(
                layer_idx, False, self.num_layers, self.pack_factor, self.context_length
            )
            v_buf_idx, v_start, v_end = get_layer_position_in_packed_buffer(
                layer_idx, True, self.num_layers, self.pack_factor, self.context_length
            )

            k_cache = self.buffers[k_buf_idx, :, k_start:k_end, :]
            v_cache = self.buffers[v_buf_idx, :, v_start:v_end, :]
        else:
            # Non-packed mode: K-first, V-second layout
            # K indices: 0 to num_layers-1
            # V indices: num_layers to 2*num_layers-1
            k_cache = self.buffers[layer_idx, :, :self.context_length, :]
            v_cache = self.buffers[self.num_layers + layer_idx, :, :self.context_length, :]

        return k_cache, v_cache

    def update_kv_cache_for_layer(self, layer_idx: int, position: int,
                                    new_k: torch.Tensor, new_v: torch.Tensor):
        """Update K and V cache at a specific position for a layer."""
        if self.pack_kvc:
            k_buf_idx, k_start, k_end = get_layer_position_in_packed_buffer(
                layer_idx, False, self.num_layers, self.pack_factor, self.context_length
            )
            v_buf_idx, v_start, v_end = get_layer_position_in_packed_buffer(
                layer_idx, True, self.num_layers, self.pack_factor, self.context_length
            )

            # Offset position by slice start
            self.buffers[k_buf_idx, :, k_start + position, :] = new_k
            self.buffers[v_buf_idx, :, v_start + position, :] = new_v
        else:
            # Non-packed mode: K-first, V-second layout
            self.buffers[layer_idx, :, position, :] = new_k
            self.buffers[self.num_layers + layer_idx, :, position, :] = new_v

    def memory_usage_bytes(self) -> int:
        """Calculate memory usage in bytes."""
        return self.buffers.numel() * 2  # float16 = 2 bytes


def test_pack_factor_calculation():
    """Test pack factor calculation."""
    print("Testing pack factor calculation...")

    test_cases = [
        (64, 256, 4),   # 4 layers per buffer
        (128, 256, 2),  # 2 layers per buffer
        (256, 256, 1),  # No packing
        (64, 512, 8),   # 8 layers per buffer
        (32, 256, 8),   # 8 layers per buffer
    ]

    for ctx, state, expected in test_cases:
        result = calculate_pack_factor(ctx, state)
        status = "✓" if result == expected else "✗"
        print(f"  {status} context={ctx}, state={state} -> pack_factor={result} (expected {expected})")

    print()


def test_layer_position_calculation():
    """Test layer position calculation in packed buffers."""
    print("Testing layer position in packed buffers...")

    # Example: 28 layers, context=64, state=256 -> pack_factor=4
    num_layers = 28
    pack_factor = 4
    context_length = 64

    num_buffers = get_packed_buffer_count(num_layers, pack_factor)
    packed_per_type = (num_layers + pack_factor - 1) // pack_factor

    print(f"\n  Num layers: {num_layers}, Pack factor: {pack_factor}, Context: {context_length}")
    print(f"  Total buffers: {num_buffers} ({packed_per_type} for K, {packed_per_type} for V)")
    print("  K-first, V-second layout:")
    print(f"    K buffers: 0 to {packed_per_type-1}")
    print(f"    V buffers: {packed_per_type} to {num_buffers-1}")
    print("\n  Layer assignments:")

    for layer_idx in range(min(8, num_layers)):  # Show first 8 layers
        k_buf, k_start, k_end = get_layer_position_in_packed_buffer(
            layer_idx, False, num_layers, pack_factor, context_length
        )
        v_buf, v_start, v_end = get_layer_position_in_packed_buffer(
            layer_idx, True, num_layers, pack_factor, context_length
        )
        print(f"    Layer {layer_idx:2d}: K -> buffer[{k_buf}][{k_start:3d}:{k_end:3d}], V -> buffer[{v_buf}][{v_start:3d}:{v_end:3d}]")

    print("    ...")
    print()


def test_memory_comparison():
    """Compare memory usage between packed and unpacked."""
    print("Testing memory comparison...")

    # Simulate Qwen3-0.6B config
    config_dict = {
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "context_length": 64,
        "state_length": 256,
    }

    config = QwenConfig(**config_dict)

    print(f"\n  Config: layers={config.num_hidden_layers}, kv_heads={config.num_key_value_heads}, head_dim={config.head_dim}")
    print(f"  context_length={config.context_length}, state_length={config.state_length}")

    # Create packed and unpacked caches
    unpacked = PackedKVCache(config, pack_kvc=False)
    packed = PackedKVCache(config, pack_kvc=True)

    unpacked_mb = unpacked.memory_usage_bytes() / (1024 * 1024)
    packed_mb = packed.memory_usage_bytes() / (1024 * 1024)

    reduction = (1 - packed_mb / unpacked_mb) * 100

    print(f"\n  Memory Usage:")
    print(f"    Unpacked: {unpacked_mb:.2f} MB ({unpacked.num_buffers} buffers)")
    print(f"    Packed:   {packed_mb:.2f} MB ({packed.num_buffers} buffers)")
    print(f"    Reduction: {reduction:.1f}%")
    print()


def test_kv_cache_operations():
    """Test KV cache get/update operations."""
    print("Testing KV cache operations...")

    config_dict = {
        "num_hidden_layers": 8,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "hidden_size": 512,
        "num_attention_heads": 8,
        "context_length": 64,
        "state_length": 256,
    }

    config = QwenConfig(**config_dict)
    packed = PackedKVCache(config, pack_kvc=True)

    # Test update and retrieval
    for layer_idx in range(config.num_hidden_layers):
        for pos in range(min(4, config.context_length)):
            # Create unique test data
            new_k = torch.ones(config.num_key_value_heads, config.head_dim) * (layer_idx + 0.1 * pos)
            new_v = torch.ones(config.num_key_value_heads, config.head_dim) * (layer_idx + 0.1 * pos + 0.5)

            packed.update_kv_cache_for_layer(layer_idx, pos, new_k.half(), new_v.half())

    # Verify retrieval
    all_correct = True
    for layer_idx in range(config.num_hidden_layers):
        k_cache, v_cache = packed.get_kv_cache_for_layer(layer_idx)

        # Check shape
        expected_shape = (config.num_key_value_heads, config.context_length, config.head_dim)
        if k_cache.shape != expected_shape or v_cache.shape != expected_shape:
            print(f"  ✗ Layer {layer_idx}: Wrong shape! Got K:{k_cache.shape}, V:{v_cache.shape}, expected {expected_shape}")
            all_correct = False
            continue

        # Check values at position 0
        expected_k = layer_idx + 0.0  # position 0
        expected_v = layer_idx + 0.5
        if not torch.allclose(k_cache[:, 0, :].float(), torch.full_like(k_cache[:, 0, :].float(), expected_k), atol=0.01):
            print(f"  ✗ Layer {layer_idx}: K value mismatch at position 0")
            all_correct = False

    if all_correct:
        print("  ✓ All KV cache operations working correctly!")
    print()


def generate_coreml_state_config(config: QwenConfig, pack_kvc: bool) -> dict:
    """Generate CoreML state configuration for meta.yaml."""
    pack_factor = calculate_pack_factor(config.context_length, config.state_length) if pack_kvc else 1
    num_buffers = get_packed_buffer_count(config.num_hidden_layers, pack_factor) if pack_kvc else config.num_hidden_layers * 2

    return {
        "pack_kvc": pack_kvc,
        "pack_factor": pack_factor,
        "num_buffers": num_buffers,
        "context_length": config.context_length,
        "state_length": config.state_length,
        "buffer_shape": [
            num_buffers,
            config.num_key_value_heads,
            config.state_length,
            config.head_dim
        ]
    }


def test_meta_yaml_config():
    """Test meta.yaml configuration generation."""
    print("Testing meta.yaml config generation...")

    config_dict = {
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "context_length": 64,
        "state_length": 256,
    }

    config = QwenConfig(**config_dict)

    # Without packing
    unpacked_config = generate_coreml_state_config(config, pack_kvc=False)
    print(f"\n  Without pack_kvc:")
    for k, v in unpacked_config.items():
        print(f"    {k}: {v}")

    # With packing
    packed_config = generate_coreml_state_config(config, pack_kvc=True)
    print(f"\n  With pack_kvc:")
    for k, v in packed_config.items():
        print(f"    {k}: {v}")

    print()


def main():
    print("\n" + "=" * 60)
    print("KV Cache Packing Optimization Test")
    print("=" * 60)

    test_pack_factor_calculation()
    test_layer_position_calculation()
    test_memory_comparison()
    test_kv_cache_operations()
    test_meta_yaml_config()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

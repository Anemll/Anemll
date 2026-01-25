#!/usr/bin/env python3
"""Test to identify KV cache tracing issue.

This test examines whether the KV cache update mechanism behaves differently
during JIT tracing vs eager execution.
"""

import sys
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

import torch
import torch.nn as nn
import numpy as np

from anemll.models.gemma3_model import (
    Gemma3ForCausalLM,
    Gemma3Config,
    MODEL_DTYPE,
    TEST_DEVICE,
)
from anemll.ane_converter.gemma3_converter import Gemma3Converter


def test_kv_cache_update_eager_vs_traced():
    """Compare KV cache updates in eager mode vs JIT traced mode."""

    model_path = '/Users/anemll/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3'
    context_length = 64

    print("Loading model...")
    config = Gemma3Config.from_json(f'{model_path}/config.json')
    config.context_length = context_length
    config.state_length = context_length

    model = Gemma3ForCausalLM(config)
    converter = Gemma3Converter(model, context_length=context_length, batch_size=context_length)
    converter.load_weights_from_hf(model_path)
    model.eval()

    # Create FFNWrapper (same as converter)
    class FFNWrapper(torch.nn.Module):
        def __init__(self, model, start_layer=0, end_layer=None):
            super().__init__()
            self.model = model
            self.start_layer = start_layer
            self.end_layer = end_layer

        def forward(self, hidden_states, position_ids, causal_mask, current_pos):
            out = self.model.model.process_layers(
                hidden_states,
                position_ids,
                causal_mask,
                current_pos,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                IN_PREFILL=False,
            )
            if self.end_layer is None or self.end_layer == len(self.model.model.layers):
                out = self.model.model.norm(out)
            return out

    wrapper = FFNWrapper(model)
    wrapper.eval()

    # Create sample inputs for a single token
    hidden_states = torch.randn(1, 1, config.hidden_size, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    position_ids = torch.tensor([0], dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    causal_mask[0, 0, 0, 1:] = float('-inf')
    current_pos = torch.tensor([0], dtype=torch.int32, device=TEST_DEVICE)

    print("\n" + "="*70)
    print("TEST 1: Eager execution - single pass")
    print("="*70)

    # Reset KV cache
    model.model.kv_cache_0.zero_()
    kv_cache_before = model.model.kv_cache_0.clone()

    with torch.no_grad():
        out_eager = wrapper(hidden_states, position_ids, causal_mask, current_pos)

    kv_cache_after_eager = model.model.kv_cache_0.clone()

    print(f"KV cache change at position 0:")
    print(f"  Non-zero before: {(kv_cache_before != 0).sum().item()}")
    print(f"  Non-zero after: {(kv_cache_after_eager != 0).sum().item()}")

    # Check specific position
    kv_diff = (kv_cache_after_eager - kv_cache_before).abs().max().item()
    print(f"  Max KV cache change: {kv_diff:.4f}")

    print("\n" + "="*70)
    print("TEST 2: JIT trace the wrapper")
    print("="*70)

    # Reset KV cache
    model.model.kv_cache_0.zero_()

    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (hidden_states, position_ids, causal_mask, current_pos)
        )

    print("Traced model created")

    # Now run traced model
    model.model.kv_cache_0.zero_()
    kv_cache_before = model.model.kv_cache_0.clone()

    with torch.no_grad():
        out_traced = traced(hidden_states, position_ids, causal_mask, current_pos)

    kv_cache_after_traced = model.model.kv_cache_0.clone()

    print(f"KV cache change at position 0 (traced):")
    print(f"  Non-zero before: {(kv_cache_before != 0).sum().item()}")
    print(f"  Non-zero after: {(kv_cache_after_traced != 0).sum().item()}")

    # Compare outputs
    output_diff = (out_eager - out_traced).abs().max().item()
    print(f"\nOutput comparison (eager vs traced at pos 0):")
    print(f"  Max diff: {output_diff:.6f}")

    # Compare KV caches
    kv_diff = (kv_cache_after_eager - kv_cache_after_traced).abs().max().item()
    print(f"  KV cache diff: {kv_diff:.6f}")

    print("\n" + "="*70)
    print("TEST 3: Multiple tokens - eager vs traced")
    print("="*70)

    # Test with position 1 (second token)
    position_ids_1 = torch.tensor([1], dtype=torch.int32, device=TEST_DEVICE)
    causal_mask_1 = torch.zeros((1, 1, 1, context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    causal_mask_1[0, 0, 0, 2:] = float('-inf')
    current_pos_1 = torch.tensor([1], dtype=torch.int32, device=TEST_DEVICE)

    # Run eager mode for second token
    model.model.kv_cache_0.zero_()
    model.model.kv_cache_0.copy_(kv_cache_after_eager)  # Start from position 0 state

    with torch.no_grad():
        out_eager_1 = wrapper(hidden_states, position_ids_1, causal_mask_1, current_pos_1)

    kv_cache_eager_1 = model.model.kv_cache_0.clone()

    # Run traced mode for second token
    model.model.kv_cache_0.zero_()
    model.model.kv_cache_0.copy_(kv_cache_after_traced)  # Start from position 0 state (traced)

    with torch.no_grad():
        out_traced_1 = traced(hidden_states, position_ids_1, causal_mask_1, current_pos_1)

    kv_cache_traced_1 = model.model.kv_cache_0.clone()

    print(f"Token 1 - Output comparison:")
    output_diff_1 = (out_eager_1 - out_traced_1).abs().max().item()
    print(f"  Max output diff: {output_diff_1:.6f}")

    kv_diff_1 = (kv_cache_eager_1 - kv_cache_traced_1).abs().max().item()
    print(f"  Max KV cache diff: {kv_diff_1:.6f}")

    # Check where KV cache was updated
    print(f"\nKV cache update positions (eager):")
    for layer in range(min(3, config.num_hidden_layers)):
        key_idx = layer
        value_idx = layer + config.num_hidden_layers

        k_pos0 = kv_cache_eager_1[key_idx, :, 0, :].abs().sum().item()
        k_pos1 = kv_cache_eager_1[key_idx, :, 1, :].abs().sum().item()
        v_pos0 = kv_cache_eager_1[value_idx, :, 0, :].abs().sum().item()
        v_pos1 = kv_cache_eager_1[value_idx, :, 1, :].abs().sum().item()

        print(f"  Layer {layer}: K[0]={k_pos0:.2f}, K[1]={k_pos1:.2f}, V[0]={v_pos0:.2f}, V[1]={v_pos1:.2f}")

    print(f"\nKV cache update positions (traced):")
    for layer in range(min(3, config.num_hidden_layers)):
        key_idx = layer
        value_idx = layer + config.num_hidden_layers

        k_pos0 = kv_cache_traced_1[key_idx, :, 0, :].abs().sum().item()
        k_pos1 = kv_cache_traced_1[key_idx, :, 1, :].abs().sum().item()
        v_pos0 = kv_cache_traced_1[value_idx, :, 0, :].abs().sum().item()
        v_pos1 = kv_cache_traced_1[value_idx, :, 1, :].abs().sum().item()

        print(f"  Layer {layer}: K[0]={k_pos0:.2f}, K[1]={k_pos1:.2f}, V[0]={v_pos0:.2f}, V[1]={v_pos1:.2f}")

    print("\n" + "="*70)
    print("TEST 4: Check if current_pos slicing works correctly")
    print("="*70)

    # The issue might be that JIT tracing "bakes in" the concrete position value
    # Let's check if the traced model updates KV cache at the correct position

    # Reset to clean state
    model.model.kv_cache_0.zero_()

    # Run traced model at position 5
    position_ids_5 = torch.tensor([5], dtype=torch.int32, device=TEST_DEVICE)
    causal_mask_5 = torch.zeros((1, 1, 1, context_length), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    causal_mask_5[0, 0, 0, 6:] = float('-inf')
    current_pos_5 = torch.tensor([5], dtype=torch.int32, device=TEST_DEVICE)

    with torch.no_grad():
        out_traced_5 = traced(hidden_states, position_ids_5, causal_mask_5, current_pos_5)

    kv_cache_traced_5 = model.model.kv_cache_0.clone()

    print("KV cache after running traced model at position 5:")
    # Check which positions have non-zero values
    for pos in range(7):
        layer0_k = kv_cache_traced_5[0, :, pos, :].abs().sum().item()
        print(f"  Position {pos}: Layer0 K sum = {layer0_k:.4f}")

    # The KEY issue: if position 5 shows values at position 0 instead,
    # it means the JIT tracing baked in the position value

    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    # Check if position 5 was updated or position 0
    pos0_sum = kv_cache_traced_5[0, :, 0, :].abs().sum().item()
    pos5_sum = kv_cache_traced_5[0, :, 5, :].abs().sum().item()

    if pos0_sum > pos5_sum and pos5_sum < 0.01:
        print("ISSUE DETECTED: JIT tracing baked in position 0!")
        print("  Position 0 has values but position 5 is empty")
        print("  This means current_pos tensor indexing is not dynamic in traced graph")
    elif pos5_sum > pos0_sum:
        print("KV cache updated at correct position 5")
        print("  The tracing appears to handle dynamic positions correctly")
    else:
        print("Unclear - both positions have values")
        print(f"  Position 0 sum: {pos0_sum:.4f}")
        print(f"  Position 5 sum: {pos5_sum:.4f}")


if __name__ == "__main__":
    test_kv_cache_update_eager_vs_traced()

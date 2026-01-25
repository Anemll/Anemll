#!/usr/bin/env python3
"""Test CoreML state (KV cache) update mechanism.

This test specifically checks if CoreML's state mechanism correctly updates
the KV cache at the specified position.
"""

import sys
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

import os
import torch
import numpy as np
import coremltools as ct

from anemll.models.gemma3_model import (
    Gemma3ForCausalLM,
    Gemma3Config,
    MODEL_DTYPE,
    TEST_DEVICE,
)
from anemll.ane_converter.gemma3_converter import Gemma3Converter


def test_coreml_state_positions():
    """Test if CoreML state updates at the correct position."""

    coreml_dir = '/tmp/gemma3_test3'
    context_length = 64

    # Load FFN model
    ffn_path = f'{coreml_dir}/gemma3_270m_FFN_chunk_01of01.mlpackage'
    if not os.path.exists(ffn_path):
        print(f"ERROR: CoreML model not found at {ffn_path}")
        return

    print(f"Loading CoreML FFN model from {ffn_path}...")
    ffn_model = ct.models.MLModel(ffn_path)

    # Create state
    state = ffn_model.make_state()

    # Get initial state using new coremltools 9.0 API
    print("\nReading initial state...")
    try:
        initial_kv = state.read_state("model_model_kv_cache_0")
        print(f"Initial KV cache shape: {initial_kv.shape}")
        print(f"Initial KV cache all zeros: {np.allclose(initial_kv, 0)}")
    except Exception as e:
        print(f"Could not read state: {e}")
        # Fall back to old method
        initial_kv = None

    # Create sample hidden states
    hidden_size = 640  # Gemma3 270M hidden size
    hidden_states = np.random.randn(1, 1, hidden_size).astype(np.float16)

    print("\n" + "="*70)
    print("TEST 1: Update at position 0")
    print("="*70)

    position_ids = np.array([0], dtype=np.int32)
    causal_mask = np.zeros((1, 1, 1, context_length), dtype=np.float16)
    causal_mask[0, 0, 0, 1:] = -np.inf
    current_pos = np.array([0], dtype=np.int32)

    result = ffn_model.predict({
        'hidden_states': hidden_states,
        'position_ids': position_ids,
        'causal_mask': causal_mask,
        'current_pos': current_pos,
    }, state)

    print(f"Output shape: {result['output_hidden_states'].shape}")

    # Read KV cache after update
    try:
        kv_after_0 = state.read_state("model_model_kv_cache_0")
        print(f"\nKV cache after position 0:")
        for pos in range(5):
            layer0_k_sum = np.abs(kv_after_0[0, :, pos, :]).sum()
            print(f"  Position {pos}: Layer0 K sum = {layer0_k_sum:.4f}")
    except Exception as e:
        print(f"Could not read state: {e}")

    print("\n" + "="*70)
    print("TEST 2: Update at position 5 (without clearing state)")
    print("="*70)

    # Keep the same state, update at position 5
    position_ids = np.array([5], dtype=np.int32)
    causal_mask = np.zeros((1, 1, 1, context_length), dtype=np.float16)
    causal_mask[0, 0, 0, 6:] = -np.inf
    current_pos = np.array([5], dtype=np.int32)

    result = ffn_model.predict({
        'hidden_states': hidden_states,
        'position_ids': position_ids,
        'causal_mask': causal_mask,
        'current_pos': current_pos,
    }, state)

    # Read KV cache after update
    try:
        kv_after_5 = state.read_state("model_model_kv_cache_0")
        print(f"\nKV cache after position 5:")
        for pos in range(7):
            layer0_k_sum = np.abs(kv_after_5[0, :, pos, :]).sum()
            print(f"  Position {pos}: Layer0 K sum = {layer0_k_sum:.4f}")

        # Check if position 5 was actually updated
        pos5_changed = not np.allclose(kv_after_5[0, :, 5, :], kv_after_0[0, :, 5, :])
        print(f"\nPosition 5 was updated: {pos5_changed}")
    except Exception as e:
        print(f"Could not read state: {e}")

    print("\n" + "="*70)
    print("TEST 3: Fresh state, update only at position 5")
    print("="*70)

    # Create new state
    state2 = ffn_model.make_state()

    position_ids = np.array([5], dtype=np.int32)
    causal_mask = np.zeros((1, 1, 1, context_length), dtype=np.float16)
    causal_mask[0, 0, 0, 6:] = -np.inf
    current_pos = np.array([5], dtype=np.int32)

    result = ffn_model.predict({
        'hidden_states': hidden_states,
        'position_ids': position_ids,
        'causal_mask': causal_mask,
        'current_pos': current_pos,
    }, state2)

    # Read KV cache
    try:
        kv_fresh_5 = state2.read_state("model_model_kv_cache_0")
        print(f"\nKV cache after ONLY updating position 5 (fresh state):")
        for pos in range(7):
            layer0_k_sum = np.abs(kv_fresh_5[0, :, pos, :]).sum()
            print(f"  Position {pos}: Layer0 K sum = {layer0_k_sum:.4f}")

        # Check which position was actually updated
        pos0_sum = np.abs(kv_fresh_5[0, :, 0, :]).sum()
        pos5_sum = np.abs(kv_fresh_5[0, :, 5, :]).sum()

        print(f"\nDIAGNOSIS:")
        if pos0_sum > 0.01 and pos5_sum < 0.01:
            print("  ISSUE: Position 0 was updated instead of position 5!")
            print("  CoreML appears to have baked in position 0 during conversion")
        elif pos5_sum > 0.01 and pos0_sum < 0.01:
            print("  OK: Position 5 was correctly updated")
        elif pos0_sum > 0.01 and pos5_sum > 0.01:
            print("  ISSUE: Both positions have values - unexpected behavior")
            print(f"  Position 0 sum: {pos0_sum:.4f}")
            print(f"  Position 5 sum: {pos5_sum:.4f}")
        else:
            print("  ISSUE: No positions were updated!")

    except Exception as e:
        print(f"Could not read state: {e}")

    print("\n" + "="*70)
    print("TEST 4: Compare with PyTorch at same position")
    print("="*70)

    # Load PyTorch model for comparison
    hf_model_path = '/Users/anemll/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3'

    print("Loading PyTorch model...")
    config = Gemma3Config.from_json(f'{hf_model_path}/config.json')
    config.context_length = context_length
    config.state_length = context_length

    model = Gemma3ForCausalLM(config)
    converter = Gemma3Converter(model, context_length=context_length, batch_size=context_length)
    converter.load_weights_from_hf(hf_model_path)
    model.eval()

    # Reset KV cache
    model.model.kv_cache_0.zero_()

    # Run at position 5 with same hidden states
    pt_hidden = torch.from_numpy(hidden_states).to(MODEL_DTYPE)
    pt_position_ids = torch.tensor([5], dtype=torch.int32)
    pt_causal_mask = torch.zeros((1, 1, 1, context_length), dtype=MODEL_DTYPE)
    pt_causal_mask[0, 0, 0, 6:] = float('-inf')
    pt_current_pos = torch.tensor([5], dtype=torch.int32)

    with torch.no_grad():
        pt_out = model.model.process_layers(
            pt_hidden,
            pt_position_ids,
            pt_causal_mask,
            pt_current_pos,
            start_layer=0,
            end_layer=None,
            IN_PREFILL=False,
        )
        pt_out = model.model.norm(pt_out)

    pt_kv_cache = model.model.kv_cache_0.numpy()

    print(f"\nPyTorch KV cache (run at position 5 only):")
    for pos in range(7):
        layer0_k_sum = np.abs(pt_kv_cache[0, :, pos, :]).sum()
        print(f"  Position {pos}: Layer0 K sum = {layer0_k_sum:.4f}")

    # Compare CoreML and PyTorch KV cache at position 5
    try:
        kv_diff_pos5 = np.abs(kv_fresh_5[0, :, 5, :] - pt_kv_cache[0, :, 5, :]).max()
        print(f"\nKV cache comparison at position 5:")
        print(f"  Max diff: {kv_diff_pos5:.4f}")
    except:
        pass


if __name__ == "__main__":
    test_coreml_state_positions()

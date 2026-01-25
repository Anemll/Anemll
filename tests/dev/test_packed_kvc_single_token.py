#!/usr/bin/env python3
"""Test packed KVC with single token prefill to isolate batched prefill issues."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import coremltools as ct
from pathlib import Path

def test_single_token_prefill(model_path, num_tokens=5):
    """Test prefill one token at a time vs batched."""

    print(f"\n{'='*60}")
    print("Testing Packed KVC Single Token Prefill")
    print(f"{'='*60}")

    # Load models
    model_dir = Path(model_path)

    # If path is a file, use its parent directory
    if model_dir.is_file():
        model_dir = model_dir.parent

    # Find the model file - look for monolithic_full pattern
    model_files = list(model_dir.glob("*_monolithic_full*.mlmodelc")) + list(model_dir.glob("*_monolithic_full*.mlpackage"))
    if not model_files:
        # Try alternative patterns
        model_files = list(model_dir.glob("*monolithic*.mlmodelc")) + list(model_dir.glob("*monolithic*.mlpackage"))
    if not model_files:
        print(f"No monolithic model found in {model_dir}")
        print(f"Files in directory: {list(model_dir.iterdir())}")
        return

    # Prefer "full" variant if available
    full_models = [f for f in model_files if 'full' in f.name]
    model_file = full_models[0] if full_models else model_files[0]
    print(f"Loading model: {model_file}")

    # Load infer and prefill functions
    infer_model = ct.models.MLModel(str(model_file), function_name='infer')
    prefill_model = ct.models.MLModel(str(model_file), function_name='prefill')

    # Get model info
    print(f"\nInfer model inputs:")
    for inp in infer_model.input_description:
        print(f"  {inp.name}: {inp.type}")

    print(f"\nPrefill model inputs:")
    for inp in prefill_model.input_description:
        print(f"  {inp.name}: {inp.type}")

    # Create state
    state = infer_model.make_state()
    print(f"\nCreated state")

    # Test parameters
    context_length = 64  # Packed KVC context
    batch_size = 64

    # Create test tokens (simple sequence)
    test_tokens = [1, 2, 3, 4, 5]  # 5 tokens

    print(f"\n{'='*60}")
    print("Test 1: Single Token Inference (no prefill)")
    print(f"{'='*60}")

    # Reset state
    state = infer_model.make_state()

    # Run single token inference for each token
    for i, token in enumerate(test_tokens):
        input_ids = np.array([[token]], dtype=np.int32)
        position_ids = np.array([i], dtype=np.int32)
        causal_mask = np.zeros((1, 1, 1, context_length), dtype=np.float16)
        current_pos = np.array([i], dtype=np.int32)

        inputs = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'causal_mask': causal_mask,
            'current_pos': current_pos,
        }

        try:
            output = infer_model.predict(inputs, state)
            # Get first logits output
            logits_key = [k for k in output.keys() if 'logits' in k][0]
            logits = output[logits_key]
            next_token = np.argmax(logits.flatten())
            print(f"  Token {i}: input={token}, output_token={next_token}, logits_shape={logits.shape}")
        except Exception as e:
            print(f"  Token {i}: ERROR - {e}")
            return

    print(f"\n{'='*60}")
    print("Test 2: Batched Prefill")
    print(f"{'='*60}")

    # Reset state
    state = infer_model.make_state()

    # Prepare batched input (pad to batch_size)
    padded_tokens = test_tokens + [0] * (batch_size - len(test_tokens))
    input_ids = np.array([padded_tokens], dtype=np.int32)
    position_ids = np.arange(batch_size, dtype=np.int32)
    causal_mask = np.triu(np.ones((1, 1, batch_size, context_length), dtype=np.float16) * -np.inf, k=1)
    causal_mask = np.zeros((1, 1, batch_size, context_length), dtype=np.float16)  # Simple zero mask for now
    current_pos = np.array([0], dtype=np.int32)

    inputs = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'causal_mask': causal_mask,
        'current_pos': current_pos,
    }

    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  position_ids shape: {position_ids.shape}")
    print(f"  causal_mask shape: {causal_mask.shape}")
    print(f"  current_pos: {current_pos}")

    try:
        output = prefill_model.predict(inputs, state)
        logits_key = [k for k in output.keys() if 'logits' in k][0]
        logits = output[logits_key]
        print(f"  Prefill output shape: {logits.shape}")
        print(f"  Prefill succeeded!")
    except Exception as e:
        print(f"  Prefill ERROR: {e}")
        return

    # Now try inference after prefill
    print(f"\n{'='*60}")
    print("Test 3: Inference after Prefill")
    print(f"{'='*60}")

    next_pos = len(test_tokens)
    input_ids = np.array([[6]], dtype=np.int32)  # Next token
    position_ids = np.array([next_pos], dtype=np.int32)
    causal_mask = np.zeros((1, 1, 1, context_length), dtype=np.float16)
    current_pos = np.array([next_pos], dtype=np.int32)

    inputs = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'causal_mask': causal_mask,
        'current_pos': current_pos,
    }

    try:
        output = infer_model.predict(inputs, state)
        logits_key = [k for k in output.keys() if 'logits' in k][0]
        logits = output[logits_key]
        next_token = np.argmax(logits.flatten())
        print(f"  Input token: 6, position: {next_pos}")
        print(f"  Output token: {next_token}")
        print(f"  Inference after prefill succeeded!")
    except Exception as e:
        print(f"  Inference ERROR: {e}")

    print(f"\n{'='*60}")
    print("Test 4: Compare Single Token Prefill vs Batched")
    print(f"{'='*60}")

    # Reset state and do single-token "prefill" using inference model
    state_single = infer_model.make_state()
    single_outputs = []

    for i, token in enumerate(test_tokens):
        input_ids = np.array([[token]], dtype=np.int32)
        position_ids = np.array([i], dtype=np.int32)
        causal_mask = np.zeros((1, 1, 1, context_length), dtype=np.float16)
        current_pos = np.array([i], dtype=np.int32)

        inputs = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'causal_mask': causal_mask,
            'current_pos': current_pos,
        }

        output = infer_model.predict(inputs, state_single)
        logits_key = [k for k in output.keys() if 'logits' in k][0]
        single_outputs.append(np.argmax(output[logits_key].flatten()))

    # Now get next token after single-token prefill
    input_ids = np.array([[6]], dtype=np.int32)
    position_ids = np.array([len(test_tokens)], dtype=np.int32)
    causal_mask = np.zeros((1, 1, 1, context_length), dtype=np.float16)
    current_pos = np.array([len(test_tokens)], dtype=np.int32)

    inputs = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'causal_mask': causal_mask,
        'current_pos': current_pos,
    }

    output_single = infer_model.predict(inputs, state_single)
    logits_key = [k for k in output_single.keys() if 'logits' in k][0]
    next_token_single = np.argmax(output_single[logits_key].flatten())

    print(f"  Single-token prefill outputs: {single_outputs}")
    print(f"  Next token after single-token: {next_token_single}")
    print(f"  Next token after batched prefill: {next_token}")

    if next_token_single == next_token:
        print(f"\n  ✓ MATCH: Single and batched prefill produce same result!")
    else:
        print(f"\n  ✗ MISMATCH: Single={next_token_single}, Batched={next_token}")
        print(f"    This suggests an issue with batched prefill in packed KVC mode")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model directory')
    args = parser.parse_args()

    test_single_token_prefill(args.model)

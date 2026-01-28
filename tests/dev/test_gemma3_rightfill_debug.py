#!/usr/bin/env python3
"""
Debug test for Gemma3 with RIGHT-FILL causal mask.

The static slicing implementation stores tokens at the END of the cache:
- Token 0 is stored at position sliding_window-1 (e.g., 1023)
- Token 1 is stored at position sliding_window-1 (after shifting)
- etc.

This requires a RIGHT-FILL causal mask where:
- For N tokens generated, we attend only to the last N positions of the cache
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoTokenizer
import coremltools as ct

# Paths
MODEL_PATH = "/Users/anemll/.cache/huggingface/hub/models--google--gemma-3-4b-it-qat-int4-unquantized/snapshots/554bd242505753eef6dfae71f76ddd50c335fc46"
OUTPUT_DIR = "/Volumes/Models/ANE/gemma3_4b_qat4_2chunk"

# Test parameters
CONTEXT_LENGTH = 4096
SLIDING_WINDOW = 1024  # local cache size


def make_rightfill_causal_mask_single(sliding_window, num_tokens_filled):
    """Create causal mask for single token with right-fill pattern.

    For right-fill, tokens are stored at the END of the cache:
    - After 1 token: data at position sw-1
    - After 2 tokens: data at positions sw-2, sw-1
    - etc.

    For single-token inference, we're adding token (num_tokens_filled).
    We need to attend to all previously filled positions (at the end) plus ourselves.

    Args:
        sliding_window: Size of the KV cache (e.g., 1024)
        num_tokens_filled: How many tokens already in cache (0 for first token)

    Returns:
        Mask of shape [1, 1, 1, sliding_window]
    """
    mask = np.full((1, 1, 1, sliding_window), -np.inf, dtype=np.float16)

    # After this token, we'll have (num_tokens_filled + 1) tokens
    # They occupy positions: [sw - num_tokens_filled - 1, sw)
    # But wait - the current token is about to be stored, and we should attend
    # to all previous tokens AND the current one (which will be at sw-1)

    # Actually, during attention computation, the new token isn't stored yet.
    # So we attend to the num_tokens_filled tokens that ARE in the cache.
    # After the current iteration, the new token will be stored.

    # For the FIRST token (num_tokens_filled=0):
    # - No previous tokens to attend to
    # - Only attend to self (will be at sw-1 after store)
    # - But self-attention happens BEFORE store, so cache is empty
    # - We should attend to nothing? Or to sw-1 (where we'll store)?

    # Looking at Gemma3's attention: Q @ K^T where K comes from cache
    # If cache is all zeros, Q @ 0 = 0, then softmax gives uniform attention
    # This shouldn't cause overflow...

    # Wait, the issue might be elsewhere. Let me just unmask the last N positions.
    if num_tokens_filled > 0:
        start_pos = sliding_window - num_tokens_filled
        mask[:, :, :, start_pos:] = 0  # Allow attention to filled positions

    # Always allow attention to the last position (where current token will be)
    mask[:, :, :, sliding_window - 1] = 0

    return mask


def make_rightfill_causal_mask_full(context_length, sliding_window):
    """Create a full context causal mask with right-fill pattern.

    For right-fill:
    - Query at position 0 attends to: sw-1 only (first token at end)
    - Query at position 1 attends to: sw-2, sw-1 (after shift, 2 tokens at end)
    - Query at position i attends to: sw-min(i+1,sw), ..., sw-1

    Returns:
        Mask of shape [1, 1, context_length, sliding_window]
    """
    mask = np.full((1, 1, context_length, sliding_window), -np.inf, dtype=np.float16)

    for i in range(context_length):
        # At position i, we have i+1 tokens (including current)
        # They occupy the last min(i+1, sliding_window) positions
        num_filled = min(i + 1, sliding_window)
        start_pos = sliding_window - num_filled
        mask[:, :, i, start_pos:] = 0

    return mask


def load_coreml_model(path, function_name=None, compute_unit=ct.ComputeUnit.ALL):
    """Load CoreML model."""
    print(f"Loading {path}...")
    if function_name:
        print(f"  function_name={function_name}")
        return ct.models.CompiledMLModel(path, function_name=function_name, compute_units=compute_unit)
    else:
        return ct.models.CompiledMLModel(path, compute_units=compute_unit)


def test_with_rightfill_mask():
    """Test single-token inference with right-fill causal mask."""
    print("\n" + "="*60)
    print("Testing with RIGHT-FILL Causal Mask")
    print("="*60)

    # Load models
    embed_path = os.path.join(OUTPUT_DIR, "gemma3_embeddings_lut4.mlmodelc")
    ffn1_path = os.path.join(OUTPUT_DIR, "gemma3_FFN_PF_lut4_chunk_01of02.mlmodelc")
    ffn2_path = os.path.join(OUTPUT_DIR, "gemma3_FFN_PF_lut4_chunk_02of02.mlmodelc")
    lmhead_path = os.path.join(OUTPUT_DIR, "gemma3_lm_head_lut4.mlmodelc")

    for p in [embed_path, ffn1_path, ffn2_path, lmhead_path]:
        if not os.path.exists(p):
            print(f"Model not found: {p}")
            return

    embed_model = load_coreml_model(embed_path)
    ffn1 = load_coreml_model(ffn1_path, function_name='infer')
    ffn2 = load_coreml_model(ffn2_path, function_name='infer')
    lmhead = load_coreml_model(lmhead_path)

    # Create states
    state1 = ffn1.make_state()
    state2 = ffn2.make_state()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Test sequence
    prompt = "What is"
    test_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"Test tokens: {test_tokens}")
    print(f"Decoded: {tokenizer.decode(test_tokens)}")

    print("\nProcessing with RIGHT-FILL mask...")

    for pos, token_id in enumerate(test_tokens):
        print(f"\n--- Position {pos}, Token ID: {token_id} ({tokenizer.decode([token_id])!r}) ---")

        # Get embedding
        input_ids = np.array([[token_id]], dtype=np.int32)
        embed_out = embed_model.predict({"input_ids": input_ids})

        hidden_states = None
        for key in embed_out:
            if isinstance(embed_out[key], np.ndarray) and len(embed_out[key].shape) == 3:
                hidden_states = embed_out[key]
                break

        if hidden_states is None:
            print("Could not find hidden states!")
            return

        print(f"  Embedding: shape={hidden_states.shape}, range=[{hidden_states.min():.4f}, {hidden_states.max():.4f}]")

        # Create RIGHT-FILL causal mask for this position
        # We use SLIDING_WINDOW for local cache, but model may expect CONTEXT_LENGTH
        # Let's check what the model expects
        rightfill_mask = make_rightfill_causal_mask_single(CONTEXT_LENGTH, pos)
        print(f"  Right-fill mask: attending to last {pos+1} positions")

        position_ids = np.array([pos], dtype=np.int32)
        current_pos = np.array([pos], dtype=np.int32)

        # FFN chunk 1
        inputs = {
            "hidden_states": hidden_states.astype(np.float16),
            "position_ids": position_ids,
            "causal_mask": rightfill_mask.astype(np.float16),
            "current_pos": current_pos,
        }

        out1 = ffn1.predict(inputs, state1)
        hidden1 = None
        for key in out1:
            if isinstance(out1[key], np.ndarray) and len(out1[key].shape) == 3:
                hidden1 = out1[key]
                break

        if hidden1 is None:
            print("  FFN1 output not found!")
            continue

        print(f"  FFN1: shape={hidden1.shape}, range=[{hidden1.min():.4f}, {hidden1.max():.4f}]")

        # Check for overflow
        if np.abs(hidden1).max() > 60000:
            print("  WARNING: FFN1 output near FP16 overflow!")

        # FFN chunk 2
        inputs2 = {
            "hidden_states": hidden1.astype(np.float16),
            "position_ids": position_ids,
            "causal_mask": rightfill_mask.astype(np.float16),
            "current_pos": current_pos,
        }

        out2 = ffn2.predict(inputs2, state2)
        hidden2 = None
        for key in out2:
            if isinstance(out2[key], np.ndarray) and len(out2[key].shape) == 3:
                hidden2 = out2[key]
                break

        if hidden2 is None:
            print("  FFN2 output not found!")
            continue

        print(f"  FFN2: shape={hidden2.shape}, range=[{hidden2.min():.4f}, {hidden2.max():.4f}]")

        # LM Head
        lmhead_out = lmhead.predict({"hidden_states": hidden2.astype(np.float16)})

        logits = None
        for key in lmhead_out:
            if isinstance(lmhead_out[key], np.ndarray) and lmhead_out[key].size > 10000:
                logits = lmhead_out[key]
                break

        if logits is not None:
            logits_flat = logits.flatten()
            top_idx = np.argmax(logits_flat)
            top_token = tokenizer.decode([top_idx])
            print(f"  LM Head: top token = {top_idx} ({top_token!r})")

            top5_idx = np.argsort(logits_flat)[-5:][::-1]
            print(f"  Top 5: {[(int(i), tokenizer.decode([i])) for i in top5_idx]}")


def compare_masks():
    """Compare left-fill vs right-fill masks."""
    print("\n" + "="*60)
    print("Comparing Left-Fill vs Right-Fill Masks")
    print("="*60)

    sw = 8  # Small example for visualization

    print("\nLeft-fill mask (standard) for 3 tokens:")
    left_mask = np.full((3, sw), -np.inf)
    for i in range(3):
        left_mask[i, :i+1] = 0  # Attend to positions 0..i
    print("Position | Attends to")
    for i in range(3):
        attend = np.where(left_mask[i] == 0)[0]
        print(f"    {i}    | {attend.tolist()}")

    print("\nRight-fill mask for 3 tokens:")
    right_mask = np.full((3, sw), -np.inf)
    for i in range(3):
        num_filled = i + 1
        start = sw - num_filled
        right_mask[i, start:] = 0  # Attend to last (i+1) positions
    print("Position | Attends to")
    for i in range(3):
        attend = np.where(right_mask[i] == 0)[0]
        print(f"    {i}    | {attend.tolist()}")

    print("\nWith right-fill, tokens are stored at the END of the cache:")
    print("  Token 0 -> stored at position 7 (sw-1)")
    print("  Token 1 -> stored at position 7 (after shift, token 0 moves to 6)")
    print("  Token 2 -> stored at position 7 (after shift, tokens 0,1 at 5,6)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Show mask comparison")
    args = parser.parse_args()

    if args.compare:
        compare_masks()
    else:
        compare_masks()  # Show this first for understanding
        test_with_rightfill_mask()

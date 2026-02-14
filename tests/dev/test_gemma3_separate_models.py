#!/usr/bin/env python3
"""
Test Gemma3 with separate individual models (no multi-function).

Tests loading and inference with:
- gemma3_FFN_lut4_chunk_XX (infer)
- gemma3_FFN_rotate_lut4_chunk_XX (infer_rotate)
- gemma3_prefill_lut4_chunk_XX (prefill)
- gemma3_prefill_rotate_lut4_chunk_XX (prefill_rotate)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoTokenizer
import coremltools as ct

# Paths
MODEL_PATH = "/Users/anemll/.cache/huggingface/hub/models--google--gemma-3-4b-it-qat-int4-unquantized/snapshots/554bd242505753eef6dfae71f76ddd50c335fc46"
OUTPUT_DIR = "/Volumes/Models/ANE/gemma3_4b_qat4_2chunk"

# Test parameters
CONTEXT_LENGTH = 4096
BATCH_SIZE = 64


def load_model(path, compute_unit=ct.ComputeUnit.CPU_AND_NE):
    """Load CoreML model without function_name (single-function models)."""
    print(f"Loading {os.path.basename(path)}...", end=" ")
    try:
        model = ct.models.CompiledMLModel(path, compute_units=compute_unit)
        print("OK")
        return model
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def test_load_all_models():
    """Test loading all individual models."""
    print("\n" + "="*60)
    print("Testing Loading All Individual Models")
    print("="*60)

    models = {}

    # Embeddings and LM head
    embed_path = os.path.join(OUTPUT_DIR, "gemma3_embeddings_lut4.mlmodelc")
    lmhead_path = os.path.join(OUTPUT_DIR, "gemma3_lm_head_lut4.mlmodelc")

    models['embed'] = load_model(embed_path)
    models['lmhead'] = load_model(lmhead_path)

    # Individual FFN/prefill models for each chunk
    for chunk in ["01of02", "02of02"]:
        models[f'ffn_{chunk}'] = load_model(
            os.path.join(OUTPUT_DIR, f"gemma3_FFN_lut4_chunk_{chunk}.mlmodelc"))
        models[f'ffn_rot_{chunk}'] = load_model(
            os.path.join(OUTPUT_DIR, f"gemma3_FFN_rotate_lut4_chunk_{chunk}.mlmodelc"))
        models[f'pf_{chunk}'] = load_model(
            os.path.join(OUTPUT_DIR, f"gemma3_prefill_lut4_chunk_{chunk}.mlmodelc"))
        models[f'pf_rot_{chunk}'] = load_model(
            os.path.join(OUTPUT_DIR, f"gemma3_prefill_rotate_lut4_chunk_{chunk}.mlmodelc"))

    # Summary
    loaded = sum(1 for m in models.values() if m is not None)
    print(f"\nLoaded {loaded}/{len(models)} models")

    return models


def test_single_token_inference(models):
    """Test single-token inference with FFN models (infer mode)."""
    print("\n" + "="*60)
    print("Testing Single-Token Inference (FFN models)")
    print("="*60)

    if not models.get('embed') or not models.get('lmhead'):
        print("Missing embed or lmhead model")
        return

    if not models.get('ffn_01of02') or not models.get('ffn_02of02'):
        print("Missing FFN models")
        return

    # Create states for FFN models
    state1 = models['ffn_01of02'].make_state()
    state2 = models['ffn_02of02'].make_state()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Test sequence
    prompt = "What is"
    test_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"Test tokens: {test_tokens}")
    print(f"Decoded: {tokenizer.decode(test_tokens)}")

    print("\nProcessing tokens one by one with FFN models...")

    for pos, token_id in enumerate(test_tokens):
        print(f"\n--- Position {pos}, Token ID: {token_id} ({tokenizer.decode([token_id])!r}) ---")

        # Get embedding
        input_ids = np.array([[token_id]], dtype=np.int32)
        embed_out = models['embed'].predict({"input_ids": input_ids})

        hidden_states = None
        for key in embed_out:
            if isinstance(embed_out[key], np.ndarray) and len(embed_out[key].shape) == 3:
                hidden_states = embed_out[key]
                break

        if hidden_states is None:
            print("Could not find hidden states!")
            return

        print(f"  Embedding: shape={hidden_states.shape}, range=[{hidden_states.min():.4f}, {hidden_states.max():.4f}]")

        # Standard causal mask - all zeros for single token
        causal_mask = np.zeros((1, 1, 1, CONTEXT_LENGTH), dtype=np.float16)
        position_ids = np.array([pos], dtype=np.int32)
        current_pos = np.array([pos], dtype=np.int32)

        # FFN chunk 1
        inputs = {
            "hidden_states": hidden_states.astype(np.float16),
            "position_ids": position_ids,
            "causal_mask": causal_mask,
            "current_pos": current_pos,
        }

        out1 = models['ffn_01of02'].predict(inputs, state1)
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
            "causal_mask": causal_mask,
            "current_pos": current_pos,
        }

        out2 = models['ffn_02of02'].predict(inputs2, state2)
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
        lmhead_out = models['lmhead'].predict({"hidden_states": hidden2.astype(np.float16)})

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


def test_prefill_batch(models):
    """Test batch prefill with prefill models."""
    print("\n" + "="*60)
    print("Testing Batch Prefill (prefill models)")
    print("="*60)

    if not models.get('embed') or not models.get('lmhead'):
        print("Missing embed or lmhead model")
        return

    if not models.get('pf_01of02') or not models.get('pf_02of02'):
        print("Missing prefill models")
        return

    # Create states for prefill models
    state1 = models['pf_01of02'].make_state()
    state2 = models['pf_02of02'].make_state()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Test with a batch of tokens
    prompt = "What is the capital of France?"
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"Prompt tokens: {tokens} ({len(tokens)} tokens)")
    print(f"Decoded: {tokenizer.decode(tokens)}")

    # Pad to batch size
    seq_len = len(tokens)
    padded_len = BATCH_SIZE
    padded_tokens = tokens + [0] * (padded_len - seq_len)

    # Get embeddings for batch
    input_ids = np.array([padded_tokens], dtype=np.int32)
    embed_out = models['embed'].predict({"input_ids": input_ids})

    hidden_states = None
    for key in embed_out:
        if isinstance(embed_out[key], np.ndarray) and len(embed_out[key].shape) == 3:
            hidden_states = embed_out[key]
            break

    if hidden_states is None:
        print("Could not find hidden states!")
        return

    print(f"Embedding output: shape={hidden_states.shape}, range=[{hidden_states.min():.4f}, {hidden_states.max():.4f}]")

    # Create prefill causal mask [1, 1, batch_size, context_length]
    causal_mask = np.full((1, 1, BATCH_SIZE, CONTEXT_LENGTH), -np.inf, dtype=np.float16)
    for i in range(BATCH_SIZE):
        causal_mask[0, 0, i, :i+1] = 0  # Allow attention to positions 0..i

    # Position IDs
    position_ids = np.arange(BATCH_SIZE, dtype=np.int32)

    # Query position (last valid token)
    query_pos = np.array([seq_len - 1], dtype=np.int32)

    print(f"\nPrefill inputs:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  position_ids: {position_ids.shape}")
    print(f"  causal_mask: {causal_mask.shape}")
    print(f"  query_pos: {query_pos}")

    # Prefill chunk 1
    inputs = {
        "hidden_states": hidden_states.astype(np.float16),
        "position_ids": position_ids,
        "causal_mask": causal_mask.astype(np.float16),
        "query_pos": query_pos,
    }

    print("\nRunning prefill chunk 1...")
    try:
        out1 = models['pf_01of02'].predict(inputs, state1)
        hidden1 = None
        for key in out1:
            if isinstance(out1[key], np.ndarray) and len(out1[key].shape) == 3:
                hidden1 = out1[key]
                break

        if hidden1 is not None:
            print(f"  PF1: shape={hidden1.shape}, range=[{hidden1.min():.4f}, {hidden1.max():.4f}]")

            # Prefill chunk 2
            inputs2 = {
                "hidden_states": hidden1.astype(np.float16),
                "position_ids": position_ids,
                "causal_mask": causal_mask.astype(np.float16),
                "query_pos": query_pos,
            }

            print("\nRunning prefill chunk 2...")
            out2 = models['pf_02of02'].predict(inputs2, state2)
            hidden2 = None
            for key in out2:
                if isinstance(out2[key], np.ndarray) and len(out2[key].shape) == 3:
                    hidden2 = out2[key]
                    break

            if hidden2 is not None:
                print(f"  PF2: shape={hidden2.shape}, range=[{hidden2.min():.4f}, {hidden2.max():.4f}]")

                # LM Head
                lmhead_out = models['lmhead'].predict({"hidden_states": hidden2.astype(np.float16)})

                logits = None
                for key in lmhead_out:
                    if isinstance(lmhead_out[key], np.ndarray) and lmhead_out[key].size > 10000:
                        logits = lmhead_out[key]
                        break

                if logits is not None:
                    logits_flat = logits.flatten()
                    top_idx = np.argmax(logits_flat)
                    top_token = tokenizer.decode([top_idx])
                    print(f"\nLM Head: top token = {top_idx} ({top_token!r})")

                    top5_idx = np.argsort(logits_flat)[-5:][::-1]
                    print(f"Top 5: {[(int(i), tokenizer.decode([i])) for i in top5_idx]}")
    except Exception as e:
        print(f"Prefill failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["load", "infer", "prefill", "all"], default="all",
                       help="Test to run")
    args = parser.parse_args()

    models = test_load_all_models()

    if args.test in ["infer", "all"]:
        test_single_token_inference(models)

    if args.test in ["prefill", "all"]:
        test_prefill_batch(models)

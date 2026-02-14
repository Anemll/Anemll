#!/usr/bin/env python3
"""
Debug test for Gemma3 single-token inference.
Compares PyTorch model output with CoreML model output to verify correctness.
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
BATCH_SIZE = 64


def load_coreml_model(path, function_name=None, compute_unit=ct.ComputeUnit.ALL):
    """Load CoreML model."""
    print(f"Loading {path}...")
    if function_name:
        print(f"  function_name={function_name}")
        return ct.models.CompiledMLModel(path, function_name=function_name, compute_units=compute_unit)
    else:
        return ct.models.CompiledMLModel(path, compute_units=compute_unit)


def test_embeddings():
    """Test embeddings model."""
    print("\n" + "="*60)
    print("Testing Embeddings Model")
    print("="*60)

    embed_path = os.path.join(OUTPUT_DIR, "gemma3_embeddings_lut4.mlmodelc")
    if not os.path.exists(embed_path):
        print(f"Embeddings model not found: {embed_path}")
        return None

    embed_model = load_coreml_model(embed_path)

    # Test with a single token
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    test_text = "Hello"
    input_ids = tokenizer.encode(test_text, add_special_tokens=True, return_tensors="np")
    print(f"Input text: {test_text}")
    print(f"Input IDs: {input_ids}")
    print(f"Input shape: {input_ids.shape}")

    # Run embeddings
    output = embed_model.predict({"input_ids": input_ids.astype(np.int32)})

    # Get hidden states
    hidden_states = None
    for key in output:
        if 'hidden' in key.lower() or 'embed' in key.lower() or 'output' in key.lower():
            hidden_states = output[key]
            print(f"Output key '{key}': shape={hidden_states.shape}")
            print(f"  range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
            print(f"  mean: {hidden_states.mean():.4f}, std: {hidden_states.std():.4f}")

    return embed_model, hidden_states, input_ids


def test_single_token_ffn():
    """Test FFN with single token (no prefill)."""
    print("\n" + "="*60)
    print("Testing Single-Token FFN (infer function)")
    print("="*60)

    # First get embeddings
    result = test_embeddings()
    if result is None:
        return
    embed_model, hidden_states, input_ids = result

    # Load FFN model
    ffn_path = os.path.join(OUTPUT_DIR, "gemma3_FFN_PF_lut4_chunk_01of02.mlmodelc")
    if not os.path.exists(ffn_path):
        print(f"FFN model not found: {ffn_path}")
        return

    print("\nLoading FFN model with 'infer' function...")
    try:
        ffn_model = load_coreml_model(ffn_path, function_name='infer')
    except Exception as e:
        print(f"Failed to load 'infer' function: {e}")
        print("Trying without function_name...")
        ffn_model = load_coreml_model(ffn_path)

    # Create state
    state = ffn_model.make_state()

    # Test single token at position 0
    print("\nTesting single token at position 0...")

    # Use only the first token's hidden state
    seq_len = hidden_states.shape[1]
    print(f"Hidden states shape: {hidden_states.shape}")

    # For single token, we need shape [1, 1, hidden_size]
    single_hidden = hidden_states[:, 0:1, :]  # Take first token
    print(f"Single hidden shape: {single_hidden.shape}")

    # Prepare inputs for infer function
    position_ids = np.array([0], dtype=np.int32)
    causal_mask = np.zeros((1, 1, 1, CONTEXT_LENGTH), dtype=np.float16)
    current_pos = np.array([0], dtype=np.int32)

    inputs = {
        "hidden_states": single_hidden.astype(np.float16),
        "position_ids": position_ids,
        "causal_mask": causal_mask,
        "current_pos": current_pos,
    }

    print("Running FFN infer...")
    try:
        output = ffn_model.predict(inputs, state)
        print("FFN output keys:", list(output.keys()))
        for key, val in output.items():
            if isinstance(val, np.ndarray):
                print(f"  {key}: shape={val.shape}, range=[{val.min():.4f}, {val.max():.4f}]")
    except Exception as e:
        print(f"FFN infer failed: {e}")
        import traceback
        traceback.print_exc()


def test_sequential_tokens():
    """Test sequential single-token inference (simulating generation without prefill)."""
    print("\n" + "="*60)
    print("Testing Sequential Single-Token Inference")
    print("="*60)

    # Load embeddings
    embed_path = os.path.join(OUTPUT_DIR, "gemma3_embeddings_lut4.mlmodelc")
    if not os.path.exists(embed_path):
        print(f"Embeddings not found: {embed_path}")
        return
    embed_model = load_coreml_model(embed_path)

    # Load FFN chunk 1
    ffn1_path = os.path.join(OUTPUT_DIR, "gemma3_FFN_PF_lut4_chunk_01of02.mlmodelc")
    ffn2_path = os.path.join(OUTPUT_DIR, "gemma3_FFN_PF_lut4_chunk_02of02.mlmodelc")

    if not os.path.exists(ffn1_path):
        print(f"FFN chunk 1 not found: {ffn1_path}")
        return
    if not os.path.exists(ffn2_path):
        print(f"FFN chunk 2 not found: {ffn2_path}")
        return

    print("Loading FFN models with 'infer' function...")
    ffn1 = load_coreml_model(ffn1_path, function_name='infer')
    ffn2 = load_coreml_model(ffn2_path, function_name='infer')

    # Load lm_head
    lmhead_path = os.path.join(OUTPUT_DIR, "gemma3_lm_head_lut4.mlmodelc")
    if not os.path.exists(lmhead_path):
        print(f"LM head not found: {lmhead_path}")
        return
    lmhead = load_coreml_model(lmhead_path)

    # Create states
    state1 = ffn1.make_state()
    state2 = ffn2.make_state()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Test sequence - process tokens one by one
    test_tokens = [2, 107, 1]  # BOS, some token, EOS (example)
    # Or use actual prompt
    prompt = "What is"
    test_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"Test tokens: {test_tokens}")
    print(f"Decoded: {tokenizer.decode(test_tokens)}")

    print("\nProcessing tokens one by one (no batch prefill)...")

    for pos, token_id in enumerate(test_tokens):
        print(f"\n--- Position {pos}, Token ID: {token_id} ({tokenizer.decode([token_id])!r}) ---")

        # Get embedding for single token
        input_ids = np.array([[token_id]], dtype=np.int32)
        embed_out = embed_model.predict({"input_ids": input_ids})

        # Find hidden states output
        hidden_states = None
        for key in embed_out:
            if isinstance(embed_out[key], np.ndarray) and len(embed_out[key].shape) == 3:
                hidden_states = embed_out[key]
                break

        if hidden_states is None:
            print("Could not find hidden states from embeddings!")
            return

        print(f"  Embedding output: {hidden_states.shape}, range=[{hidden_states.min():.4f}, {hidden_states.max():.4f}]")

        # Run through FFN chunks
        position_ids = np.array([pos], dtype=np.int32)
        causal_mask = np.zeros((1, 1, 1, CONTEXT_LENGTH), dtype=np.float16)
        current_pos = np.array([pos], dtype=np.int32)

        # FFN chunk 1
        inputs = {
            "hidden_states": hidden_states.astype(np.float16),
            "position_ids": position_ids,
            "causal_mask": causal_mask,
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

        print(f"  FFN1 output: {hidden1.shape}, range=[{hidden1.min():.4f}, {hidden1.max():.4f}]")

        # FFN chunk 2
        inputs2 = {
            "hidden_states": hidden1.astype(np.float16),
            "position_ids": position_ids,
            "causal_mask": causal_mask,
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

        print(f"  FFN2 output: {hidden2.shape}, range=[{hidden2.min():.4f}, {hidden2.max():.4f}]")

        # LM Head
        lmhead_out = lmhead.predict({"hidden_states": hidden2.astype(np.float16)})

        # Find logits
        logits = None
        for key in lmhead_out:
            if isinstance(lmhead_out[key], np.ndarray):
                val = lmhead_out[key]
                # Logits should be large (vocab size)
                if val.size > 10000:
                    logits = val
                    break

        if logits is not None:
            logits_flat = logits.flatten()
            top_idx = np.argmax(logits_flat)
            top_token = tokenizer.decode([top_idx])
            print(f"  LM Head: top token = {top_idx} ({top_token!r})")

            # Show top 5 tokens
            top5_idx = np.argsort(logits_flat)[-5:][::-1]
            print(f"  Top 5: {[(int(i), tokenizer.decode([i])) for i in top5_idx]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["embed", "ffn", "seq"], default="seq",
                       help="Test to run: embed, ffn, or seq (sequential)")
    args = parser.parse_args()

    if args.test == "embed":
        test_embeddings()
    elif args.test == "ffn":
        test_single_token_ffn()
    else:
        test_sequential_tokens()

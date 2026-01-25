#!/usr/bin/env python3
"""
Modular debug test for Gemma3n CoreML parts.
Tests each component independently and uses CoreML debugging utilities.

Usage:
    python test_gemma3n_parts_debug.py --bundle /tmp/gemma3n-fixed/bundle --part all
    python test_gemma3n_parts_debug.py --bundle /tmp/gemma3n-fixed/bundle --part init
    python test_gemma3n_parts_debug.py --bundle /tmp/gemma3n-fixed/bundle --part chunks
    python test_gemma3n_parts_debug.py --bundle /tmp/gemma3n-fixed/bundle --part combine
    python test_gemma3n_parts_debug.py --bundle /tmp/gemma3n-fixed/bundle --part lm_head
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer


def load_model(path: Path) -> ct.models.MLModel:
    """Load a CoreML model with error handling."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    print(f"Loading: {path.name}")
    return ct.models.MLModel(str(path))


def print_model_spec(model: ct.models.MLModel, name: str):
    """Print model specification details."""
    spec = model.get_spec()
    print(f"\n{'='*60}")
    print(f"{name} Model Specification")
    print(f"{'='*60}")

    # Inputs
    print("\nInputs:")
    for inp in spec.description.input:
        if inp.type.WhichOneof('Type') == 'multiArrayType':
            shape = list(inp.type.multiArrayType.shape)
            dtype = inp.type.multiArrayType.dataType
            print(f"  {inp.name}: shape={shape}, dtype={dtype}")
        elif inp.type.WhichOneof('Type') == 'stateType':
            print(f"  {inp.name}: (state)")
        else:
            print(f"  {inp.name}: {inp.type.WhichOneof('Type')}")

    # Outputs
    print("\nOutputs:")
    for out in spec.description.output:
        if out.type.WhichOneof('Type') == 'multiArrayType':
            shape = list(out.type.multiArrayType.shape)
            dtype = out.type.multiArrayType.dataType
            print(f"  {out.name}: shape={shape}, dtype={dtype}")
        else:
            print(f"  {out.name}: {out.type.WhichOneof('Type')}")

    # States (if any)
    if hasattr(spec.description, 'state') and spec.description.state:
        print("\nStates:")
        for state in spec.description.state:
            print(f"  {state.name}")


def summarize_array(arr: np.ndarray, name: str, max_vals: int = 5):
    """Print array statistics."""
    flat = arr.reshape(-1)
    print(f"{name}:")
    print(f"  shape={arr.shape}, dtype={arr.dtype}")
    print(f"  min={arr.min():.6f}, max={arr.max():.6f}")
    print(f"  mean={arr.mean():.6f}, std={arr.std():.6f}")
    if len(flat) <= max_vals:
        print(f"  values={flat}")
    else:
        print(f"  first {max_vals}: {flat[:max_vals]}")


def build_causal_mask(context_length: int, dtype=np.float16) -> np.ndarray:
    """Build standard causal attention mask."""
    causal = np.zeros((1, 1, context_length, context_length), dtype=dtype)
    i_idx, j_idx = np.triu_indices(context_length, k=1)
    causal[:, :, i_idx, j_idx] = float("-inf")
    return causal


def test_infer_init(bundle: Path, token_id: int = 2, verbose: bool = True):
    """Test infer_init model independently."""
    print("\n" + "="*60)
    print("Testing: infer_init")
    print("="*60)

    model = load_model(bundle / "gemma3n_infer_init.mlpackage")

    if verbose:
        print_model_spec(model, "infer_init")

    # Test with a single token
    input_ids = np.array([[token_id]], dtype=np.int32)
    print(f"\nInput: token_id={token_id}")

    out = model.predict({"input_ids": input_ids})

    print("\nOutputs:")
    for key, val in out.items():
        summarize_array(val, f"  {key}")

    # Check for NaN/Inf
    for key, val in out.items():
        if np.any(np.isnan(val)):
            print(f"  WARNING: {key} contains NaN!")
        if np.any(np.isinf(val)):
            print(f"  WARNING: {key} contains Inf!")

    return out


def test_infer_chunks(bundle: Path, hidden_states: np.ndarray, per_layer_inputs: np.ndarray,
                      context_length: int = 512, current_pos: int = 0, chunk_idx: int = None,
                      verbose: bool = True):
    """Test infer chunks independently.

    Args:
        chunk_idx: If specified, only test this single chunk (0-indexed). Otherwise test all.
    """
    print("\n" + "="*60)
    print("Testing: infer chunks")
    print("="*60)

    # Find all chunks
    chunks = sorted(bundle.glob("gemma3n_infer_chunk_*of*.mlpackage"))
    print(f"Found {len(chunks)} chunks")

    if chunk_idx is not None:
        if chunk_idx >= len(chunks):
            print(f"ERROR: chunk_idx {chunk_idx} out of range (0-{len(chunks)-1})")
            return hidden_states
        chunks = [chunks[chunk_idx]]
        print(f"Testing only chunk {chunk_idx}")

    causal = build_causal_mask(context_length)

    # Create shared state
    model0 = load_model(chunks[0])
    state = model0.make_state()

    if verbose:
        print_model_spec(model0, f"infer_chunk")

    current_hidden = hidden_states.copy()

    for i, chunk_path in enumerate(chunks):
        actual_idx = chunk_idx if chunk_idx is not None else i
        print(f"\n--- Chunk {actual_idx} ---")
        model = load_model(chunk_path) if i > 0 else model0

        print(f"Input hidden_states:")
        summarize_array(current_hidden, "  hidden_states")

        out = model.predict(
            {
                "hidden_states": current_hidden,
                "per_layer_inputs": per_layer_inputs,
                "causal_mask": causal,
                "current_pos": np.array([current_pos], dtype=np.int32),
            },
            state,
        )

        current_hidden = out["output_hidden_states"]
        print(f"Output hidden_states:")
        summarize_array(current_hidden, "  output_hidden_states")

        # Check for NaN/Inf
        if np.any(np.isnan(current_hidden)):
            print(f"  WARNING: output contains NaN!")
        if np.any(np.isinf(current_hidden)):
            print(f"  WARNING: output contains Inf!")

    return current_hidden


def test_combine_streams(bundle: Path, hidden_states: np.ndarray, verbose: bool = True):
    """Test combine_streams model independently."""
    print("\n" + "="*60)
    print("Testing: combine_streams")
    print("="*60)

    model = load_model(bundle / "gemma3n_combine_streams.mlpackage")

    if verbose:
        print_model_spec(model, "combine_streams")

    print(f"\nInput hidden_states (4-stream):")
    summarize_array(hidden_states, "  hidden_states")

    out = model.predict({"hidden_states": hidden_states})
    combined = out["output_hidden_states"]

    print(f"\nOutput hidden_states (combined):")
    summarize_array(combined, "  output_hidden_states")

    # Check for NaN/Inf
    if np.any(np.isnan(combined)):
        print(f"  WARNING: output contains NaN!")
    if np.any(np.isinf(combined)):
        print(f"  WARNING: output contains Inf!")

    return combined


def test_lm_head(bundle: Path, hidden_states: np.ndarray, verbose: bool = True):
    """Test lm_head model independently."""
    print("\n" + "="*60)
    print("Testing: lm_head")
    print("="*60)

    model = load_model(bundle / "gemma3n_lm_head.mlpackage")

    if verbose:
        print_model_spec(model, "lm_head")

    print(f"\nInput hidden_states:")
    summarize_array(hidden_states, "  hidden_states")

    out = model.predict({"hidden_states": hidden_states.astype(np.float16)})

    # Concatenate split logits
    logits_parts = []
    for i in range(1, 17):  # 16-way split
        key = f"logits_split_{i}"
        if key in out:
            logits_parts.append(out[key])

    if logits_parts:
        logits = np.concatenate(logits_parts, axis=-1)[0, 0]  # [vocab_size]
        print(f"\nConcatenated logits:")
        summarize_array(logits, "  logits")

        # Top-5 predictions
        top5_ids = np.argsort(logits)[-5:][::-1]
        print(f"\nTop-5 logit values:")
        for rank, idx in enumerate(top5_ids, 1):
            print(f"  {rank}. token_id={idx}, logit={logits[idx]:.4f}")

        # Check for NaN/Inf
        if np.any(np.isnan(logits)):
            print(f"  WARNING: logits contain NaN!")
        if np.any(np.isinf(logits)):
            print(f"  WARNING: logits contain Inf!")

        return logits, top5_ids
    else:
        print("  No logits splits found in output!")
        print(f"  Available keys: {list(out.keys())}")
        return None, None


def test_full_pipeline(bundle: Path, token_id: int = 2, context_length: int = 512, verbose: bool = True):
    """Test full pipeline: init -> chunks -> combine -> lm_head."""
    print("\n" + "="*60)
    print("Testing: FULL PIPELINE")
    print("="*60)

    # 1. Init
    init_out = test_infer_init(bundle, token_id, verbose=False)
    hidden_states = init_out["hidden_states"]
    per_layer_inputs = init_out["per_layer_inputs"]

    # 2. Chunks
    hidden_states = test_infer_chunks(bundle, hidden_states, per_layer_inputs,
                                       context_length, current_pos=0, verbose=False)

    # 3. Combine
    combined = test_combine_streams(bundle, hidden_states, verbose=False)

    # 4. LM Head
    logits, top5 = test_lm_head(bundle, combined, verbose=False)

    # Load tokenizer for decoding
    try:
        tokenizer_path = bundle / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(bundle))
            print("\n" + "="*60)
            print("Top-5 Decoded Predictions:")
            print("="*60)
            for rank, idx in enumerate(top5, 1):
                token_text = tokenizer.decode([idx])
                print(f"  {rank}. '{token_text}' (id={idx}, logit={logits[idx]:.4f})")
    except Exception as e:
        print(f"Could not decode tokens: {e}")


def main():
    parser = argparse.ArgumentParser(description="Modular debug test for Gemma3n CoreML parts")
    parser.add_argument("--bundle", required=True, help="Path to CoreML bundle directory")
    parser.add_argument("--part", default="all", choices=["all", "init", "chunks", "combine", "lm_head", "pipeline"],
                        help="Which part to test")
    parser.add_argument("--token-id", type=int, default=2, help="Token ID for testing")
    parser.add_argument("--context-length", type=int, default=512, help="Context length")
    parser.add_argument("--chunk-idx", type=int, default=None, help="Test only this chunk index (0-indexed)")
    parser.add_argument("--verbose", action="store_true", help="Print model specs")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    if not bundle.exists():
        print(f"Bundle not found: {bundle}")
        sys.exit(1)

    print(f"Bundle: {bundle}")
    print(f"Testing: {args.part}")
    if args.chunk_idx is not None:
        print(f"Chunk: {args.chunk_idx}")

    if args.part == "init" or args.part == "all":
        test_infer_init(bundle, args.token_id, args.verbose)

    if args.part == "chunks" or args.part == "all":
        # Need init output first
        init_out = test_infer_init(bundle, args.token_id, verbose=False)
        test_infer_chunks(bundle, init_out["hidden_states"], init_out["per_layer_inputs"],
                          args.context_length, chunk_idx=args.chunk_idx, verbose=args.verbose)

    if args.part == "combine" or args.part == "all":
        # Need chunks output first
        init_out = test_infer_init(bundle, args.token_id, verbose=False)
        hidden = test_infer_chunks(bundle, init_out["hidden_states"], init_out["per_layer_inputs"],
                                   args.context_length, chunk_idx=args.chunk_idx, verbose=False)
        test_combine_streams(bundle, hidden, args.verbose)

    if args.part == "lm_head" or args.part == "all":
        # Need combine output first
        init_out = test_infer_init(bundle, args.token_id, verbose=False)
        hidden = test_infer_chunks(bundle, init_out["hidden_states"], init_out["per_layer_inputs"],
                                   args.context_length, chunk_idx=args.chunk_idx, verbose=False)
        combined = test_combine_streams(bundle, hidden, verbose=False)
        test_lm_head(bundle, combined, args.verbose)

    if args.part == "pipeline":
        test_full_pipeline(bundle, args.token_id, args.context_length, args.verbose)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test Gemma3n prefill with multiple tokens to verify KV cache accumulation.
This simulates proper autoregressive generation.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

from gemma3n_coreml_inputs import (
    create_position_mask,
    create_position_one_hot,
    create_rotary_embeddings,
)


def load_model(path: Path) -> ct.models.MLModel:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return ct.models.MLModel(str(path))


def summarize(arr: np.ndarray, name: str):
    print(f"{name}: shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--context-length", type=int, default=512)
    args = parser.parse_args()

    bundle = Path(args.bundle)
    tokenizer = AutoTokenizer.from_pretrained(str(bundle))

    # Tokenize prompt
    tokens = tokenizer(args.prompt, return_tensors="np")["input_ids"][0].tolist()
    print(f"Prompt: '{args.prompt}'")
    print(f"Tokens ({len(tokens)}): {tokens}")
    for t in tokens:
        print(f"  {t} -> {repr(tokenizer.decode([t]))}")

    # Load models
    print("\nLoading models...")
    infer_init = load_model(bundle / "gemma3n_infer_init.mlpackage")
    combine = load_model(bundle / "gemma3n_combine_streams.mlpackage")
    lm_head = load_model(bundle / "gemma3n_lm_head.mlpackage")

    chunks = sorted(bundle.glob("gemma3n_infer_chunk_*of*.mlpackage"))
    infer_chunks = [load_model(p) for p in chunks]
    print(f"Loaded {len(infer_chunks)} chunks")

    # Create KV state and derive dimensions
    state = infer_chunks[0].make_state()
    kv_cache = state.read_state("model_kv_cache_0")
    ctx_len = kv_cache.shape[2]
    head_dim = kv_cache.shape[3]

    print("\n" + "="*60)
    print("PREFILL: Processing prompt tokens one-by-one")
    print("="*60)

    last_hidden = None
    for pos, tok in enumerate(tokens):
        print(f"\n--- Position {pos}: token {tok} ({repr(tokenizer.decode([tok]))}) ---")

        # Init: token -> hidden_states + per_layer_inputs
        init_out = infer_init.predict({"input_ids": np.array([[int(tok)]], dtype=np.int32)})
        hidden = init_out["hidden_states"]
        pli = init_out["per_layer_inputs"]

        pos_mask = create_position_mask(pos, ctx_len)
        pos_one_hot = create_position_one_hot(pos, ctx_len)
        cos_local, sin_local, cos_global, sin_global = create_rotary_embeddings(pos, head_dim)

        # Process through all chunks (with KV cache)
        for i, chunk in enumerate(infer_chunks):
            out = chunk.predict(
                {
                    "hidden_states": hidden,
                    "per_layer_inputs": pli,
                    "causal_mask": pos_mask,
                    "current_pos": np.array([pos], dtype=np.int32),
                    "position_one_hot": pos_one_hot,
                    "rotary_cos_local": cos_local,
                    "rotary_sin_local": sin_local,
                    "rotary_cos_global": cos_global,
                    "rotary_sin_global": sin_global,
                },
                state,
            )
            hidden = out["output_hidden_states"]

        summarize(hidden, f"  After chunks")

        # Combine streams
        combined = combine.predict({"hidden_states": hidden})["output_hidden_states"]
        summarize(combined, f"  After combine")
        last_hidden = combined

    # Generate prediction from last token
    print("\n" + "="*60)
    print("GENERATION: Predicting next token after prompt")
    print("="*60)

    lm_out = lm_head.predict({"hidden_states": last_hidden.astype(np.float16)})

    # Concat logits
    logits_parts = [lm_out[f"logits_split_{i}"] for i in range(1, 17)]
    logits = np.concatenate(logits_parts, axis=-1)[0, 0]

    top5_ids = np.argsort(logits)[-5:][::-1]
    print("\nTop-5 predictions (after full prompt):")
    for rank, idx in enumerate(top5_ids, 1):
        token_text = tokenizer.decode([idx])
        print(f"  {rank}. '{token_text}' (id={idx}, logit={logits[idx]:.4f})")

    # Now generate one more token
    print("\n" + "="*60)
    print("GENERATION STEP 1: Process predicted token")
    print("="*60)

    next_token = top5_ids[0]  # greedy
    print(f"Selected: {next_token} ({repr(tokenizer.decode([next_token]))})")

    pos = len(tokens)
    init_out = infer_init.predict({"input_ids": np.array([[int(next_token)]], dtype=np.int32)})
    hidden = init_out["hidden_states"]
    pli = init_out["per_layer_inputs"]

    pos_mask = create_position_mask(pos, ctx_len)
    pos_one_hot = create_position_one_hot(pos, ctx_len)
    cos_local, sin_local, cos_global, sin_global = create_rotary_embeddings(pos, head_dim)

    for i, chunk in enumerate(infer_chunks):
        out = chunk.predict(
            {
                "hidden_states": hidden,
                "per_layer_inputs": pli,
                "causal_mask": pos_mask,
                "current_pos": np.array([pos], dtype=np.int32),
                "position_one_hot": pos_one_hot,
                "rotary_cos_local": cos_local,
                "rotary_sin_local": sin_local,
                "rotary_cos_global": cos_global,
                "rotary_sin_global": sin_global,
            },
            state,
        )
        hidden = out["output_hidden_states"]

    combined = combine.predict({"hidden_states": hidden})["output_hidden_states"]
    lm_out = lm_head.predict({"hidden_states": combined.astype(np.float16)})

    logits_parts = [lm_out[f"logits_split_{i}"] for i in range(1, 17)]
    logits = np.concatenate(logits_parts, axis=-1)[0, 0]

    top5_ids = np.argsort(logits)[-5:][::-1]
    print("\nTop-5 predictions (after 1 generated token):")
    for rank, idx in enumerate(top5_ids, 1):
        token_text = tokenizer.decode([idx])
        print(f"  {rank}. '{token_text}' (id={idx}, logit={logits[idx]:.4f})")


if __name__ == "__main__":
    main()

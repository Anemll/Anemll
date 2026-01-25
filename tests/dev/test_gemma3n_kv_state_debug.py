#!/usr/bin/env python3
"""
Debug KV cache state using coremltools 9.0 state read/write APIs.
Tests if KV cache is being properly updated during inference.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import coremltools as ct

from gemma3n_coreml_inputs import (
    create_position_mask,
    create_position_one_hot,
    create_rotary_embeddings,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--context-length", type=int, default=512)
    args = parser.parse_args()

    bundle = Path(args.bundle)

    # Load chunk 0 (has KV cache state)
    chunk0_path = bundle / "gemma3n_infer_chunk_00of04.mlpackage"
    print(f"Loading: {chunk0_path}")
    model = ct.models.MLModel(str(chunk0_path))

    # Load infer_init for embeddings
    init_path = bundle / "gemma3n_infer_init.mlpackage"
    print(f"Loading: {init_path}")
    infer_init = ct.models.MLModel(str(init_path))

    # Create state
    state = model.make_state()

    # Check what states are available
    print("\n" + "="*60)
    print("Inspecting KV cache state")
    print("="*60)

    ctx_len = args.context_length
    head_dim = 256

    # Try to read the KV cache state
    try:
        kv_cache = state.read_state(name="model_kv_cache_0")
        ctx_len = kv_cache.shape[2]
        head_dim = kv_cache.shape[3]
        print(f"\nKV cache shape: {kv_cache.shape}")
        print(f"KV cache dtype: {kv_cache.dtype}")
        print(f"KV cache stats: min={kv_cache.min():.6f}, max={kv_cache.max():.6f}, mean={kv_cache.mean():.6f}")
        print(f"KV cache is all zeros: {np.allclose(kv_cache, 0)}")
    except Exception as e:
        print(f"Could not read state 'model_kv_cache_0': {e}")
        # Try without prefix
        try:
            kv_cache = state.read_state(name="kv_cache_0")
            print(f"\nKV cache shape (no prefix): {kv_cache.shape}")
        except Exception as e2:
            print(f"Could not read state 'kv_cache_0': {e2}")

    # Get embeddings for token 2 (BOS)
    print("\n" + "="*60)
    print("Running inference at position 0")
    print("="*60)

    init_out = infer_init.predict({"input_ids": np.array([[2]], dtype=np.int32)})
    hidden = init_out["hidden_states"]
    pli = init_out["per_layer_inputs"]
    pos = 0
    mask = create_position_mask(pos, ctx_len)
    one_hot = create_position_one_hot(pos, ctx_len)
    cos_local, sin_local, cos_global, sin_global = create_rotary_embeddings(pos, head_dim)

    # Run chunk 0 at position 0
    out = model.predict(
        {
            "hidden_states": hidden,
            "per_layer_inputs": pli,
            "causal_mask": mask,
            "current_pos": np.array([pos], dtype=np.int32),
            "position_one_hot": one_hot,
            "rotary_cos_local": cos_local,
            "rotary_sin_local": sin_local,
            "rotary_cos_global": cos_global,
            "rotary_sin_global": sin_global,
        },
        state,
    )
    print(f"Output shape: {out['output_hidden_states'].shape}")

    # Read KV cache after position 0
    try:
        kv_cache = state.read_state(name="model_kv_cache_0")
        print(f"\nKV cache after pos 0:")
        print(f"  Is all zeros: {np.allclose(kv_cache, 0)}")
        print(f"  min={kv_cache.min():.6f}, max={kv_cache.max():.6f}")

        # Check specific positions in the cache
        print(f"\n  Position 0 stats: min={kv_cache[:, :, 0, :].min():.6f}, max={kv_cache[:, :, 0, :].max():.6f}")
        print(f"  Position 1 stats: min={kv_cache[:, :, 1, :].min():.6f}, max={kv_cache[:, :, 1, :].max():.6f}")
        print(f"  Position 0 is zeros: {np.allclose(kv_cache[:, :, 0, :], 0)}")
        print(f"  Position 1 is zeros: {np.allclose(kv_cache[:, :, 1, :], 0)}")
    except Exception as e:
        print(f"Could not read state: {e}")

    # Run at position 5
    print("\n" + "="*60)
    print("Running inference at position 5")
    print("="*60)

    # Get embeddings for token 563 ("is")
    init_out = infer_init.predict({"input_ids": np.array([[563]], dtype=np.int32)})
    hidden = init_out["hidden_states"]
    pli = init_out["per_layer_inputs"]
    pos = 5
    mask = create_position_mask(pos, ctx_len)
    one_hot = create_position_one_hot(pos, ctx_len)
    cos_local, sin_local, cos_global, sin_global = create_rotary_embeddings(pos, head_dim)

    out = model.predict(
        {
            "hidden_states": hidden,
            "per_layer_inputs": pli,
            "causal_mask": mask,
            "current_pos": np.array([pos], dtype=np.int32),
            "position_one_hot": one_hot,
            "rotary_cos_local": cos_local,
            "rotary_sin_local": sin_local,
            "rotary_cos_global": cos_global,
            "rotary_sin_global": sin_global,
        },
        state,
    )
    print(f"Output shape: {out['output_hidden_states'].shape}")

    # Read KV cache after position 5
    try:
        kv_cache = state.read_state(name="model_kv_cache_0")
        print(f"\nKV cache after pos 5:")
        print(f"  Position 0 is zeros: {np.allclose(kv_cache[:, :, 0, :], 0)}")
        print(f"  Position 1 is zeros: {np.allclose(kv_cache[:, :, 1, :], 0)}")
        print(f"  Position 5 is zeros: {np.allclose(kv_cache[:, :, 5, :], 0)}")
        print(f"  Position 6 is zeros: {np.allclose(kv_cache[:, :, 6, :], 0)}")

        # Show non-zero positions
        for pos in range(min(10, kv_cache.shape[2])):
            if not np.allclose(kv_cache[:, :, pos, :], 0):
                print(f"  Position {pos} has data: min={kv_cache[:, :, pos, :].min():.4f}, max={kv_cache[:, :, pos, :].max():.4f}")
    except Exception as e:
        print(f"Could not read state: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()

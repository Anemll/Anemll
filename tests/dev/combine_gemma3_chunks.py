#!/usr/bin/env python3
"""
Combine Gemma3 chunked FFN and prefill models into multi-function models.

This creates combined models for each chunk with:
- 'infer': Single token inference
- 'prefill': Batch token processing

Run with:
    python tests/dev/combine_gemma3_chunks.py /Volumes/Models/ANE/gemma3_1b_lut6_ctx4096_chunked
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import coremltools as ct


def combine_chunk(ffn_path: str, prefill_path: str, output_path: str) -> bool:
    """Combine FFN and prefill models into multi-function model."""

    print(f"\nCombining:")
    print(f"  FFN:     {ffn_path}")
    print(f"  Prefill: {prefill_path}")
    print(f"  Output:  {output_path}")

    if not os.path.exists(ffn_path):
        print(f"Error: FFN model not found: {ffn_path}")
        return False
    if not os.path.exists(prefill_path):
        print(f"Error: Prefill model not found: {prefill_path}")
        return False

    try:
        # Create multifunction descriptor
        desc = ct.utils.MultiFunctionDescriptor()

        # Add FFN model as 'infer' function
        print("  Adding FFN as 'infer' function...")
        desc.add_function(
            ffn_path,
            src_function_name="main",
            target_function_name="infer"
        )

        # Add prefill model as 'prefill' function
        print("  Adding prefill as 'prefill' function...")
        desc.add_function(
            prefill_path,
            src_function_name="main",
            target_function_name="prefill"
        )

        # Set default function
        desc.default_function_name = "infer"

        # Save combined model
        print("  Saving combined model...")
        combined = ct.utils.save_multifunction(desc, output_path)

        print(f"  Successfully created: {output_path}")
        return True

    except Exception as e:
        print(f"Error combining models: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Combine Gemma3 chunked FFN and prefill models")
    parser.add_argument("model_dir", help="Directory containing chunked models")
    parser.add_argument("--prefix", default="gemma3_1b", help="Model prefix (default: gemma3_1b)")
    parser.add_argument("--lut", default="lut6", help="LUT suffix (default: lut6)")
    parser.add_argument("--chunks", type=int, default=4, help="Number of chunks (default: 4)")
    args = parser.parse_args()

    model_dir = args.model_dir
    prefix = args.prefix
    lut = args.lut
    num_chunks = args.chunks

    print(f"\nCombining chunked models in: {model_dir}")
    print(f"  Prefix: {prefix}")
    print(f"  LUT: {lut}")
    print(f"  Chunks: {num_chunks}")

    success_count = 0

    for i in range(1, num_chunks + 1):
        chunk_str = f"{i:02d}of{num_chunks:02d}"

        ffn_path = os.path.join(model_dir, f"{prefix}_FFN_{lut}_chunk_{chunk_str}.mlpackage")
        prefill_path = os.path.join(model_dir, f"{prefix}_prefill_{lut}_chunk_{chunk_str}.mlpackage")
        output_path = os.path.join(model_dir, f"{prefix}_FFN_PF_{lut}_chunk_{chunk_str}.mlpackage")

        if combine_chunk(ffn_path, prefill_path, output_path):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Combined {success_count}/{num_chunks} chunks successfully")

    if success_count == num_chunks:
        print(f"\nUpdate meta.yaml to use:")
        print(f"  ffn: {prefix}_FFN_PF_{lut}_chunk_01of{num_chunks:02d}.mlpackage")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

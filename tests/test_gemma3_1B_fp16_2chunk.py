#!/usr/bin/env python3
#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

"""
Test script for Gemma3 1B model conversion with 2 chunks and NO LUT (FP16).
This script converts the model without quantization for maximum quality.

Usage:
    # Convert with default settings (FP16, 4096 context, 2 chunks)
    python tests/test_gemma3_1B_fp16_2chunk.py

    # Custom output directory
    python tests/test_gemma3_1B_fp16_2chunk.py --output /Volumes/Models/ANE/gemma3_1b_fp16_2chunk

    # Different context length
    python tests/test_gemma3_1B_fp16_2chunk.py --context 2048

Note: FP16 models are larger but have better quality than LUT-quantized versions.
      2 chunks splits the FFN layers for better ANE memory management.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_gemma3_1b_fp16_2chunk(model_name: str = "google/gemma-3-1b-it",
                               output_dir: str = "/tmp/test-gemma3-1b-fp16-2chunk",
                               num_chunks: int = 2,
                               context_length: int = 4096,
                               batch_size: int = 64,
                               skip_check: bool = True):
    """Run Gemma3 1B model conversion with FP16 (no LUT) and 2 chunks.

    Args:
        model_name: HuggingFace model name or local path
        output_dir: Directory for converted models
        num_chunks: Number of chunks (default: 2)
        context_length: Context length for conversion (default: 4096)
        batch_size: Batch size for prefill (default: 64)
        skip_check: If True, skip dependency check (default: True)
    """

    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print("=== Gemma3 1B Model Test Suite (FP16, 2 Chunks) ===")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Chunks: {num_chunks}")
    print(f"LUT: None (FP16)")
    print(f"Context: {context_length}")
    print(f"Batch: {batch_size}")
    print()

    if context_length > 512:
        print("Note: Context > 512 enables 4-function model with rotation support")
        print("      (infer, infer_rotate, prefill, prefill_rotate)")
        print()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build convert_model.sh command
    convert_script = project_root / "anemll" / "utils" / "convert_model.sh"
    if not convert_script.exists():
        print(f"Error: Convert script not found at {convert_script}")
        return 1

    cmd = [
        str(convert_script),
        "--model", model_name,
        "--output", output_dir,
        "--context", str(context_length),
        "--batch", str(batch_size),
        "--chunk", str(num_chunks),
        "--prefix", "gemma3",
    ]

    # No LUT - pass empty values for all three parts
    # This keeps the model in FP16 format
    cmd.extend(["--lut1", ""])
    cmd.extend(["--lut2", ""])
    cmd.extend(["--lut3", ""])

    # Skip dependency check if requested
    if skip_check:
        cmd.append("--skip-check")

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True, cwd=project_root)

        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("Gemma3 1B FP16 2-chunk conversion completed successfully!")
            print("=" * 60)
            print()
            print("To test the converted model, run:")
            print(f"  python tests/chat.py --meta {output_dir}/meta.yaml --prompt \"Hello!\"")
            print()
            print("Features enabled:")
            print("  - FP16 precision (no quantization)")
            print("  - 2-chunk FFN splitting")
            print("  - Split KV cache (15 local + 3 global attention layers)")
            print("  - 4-function model with rotation support")
            print("  - 16-way LM head splitting for 262K vocabulary")
            return 0
        else:
            print(f"\nConversion failed with return code: {result.returncode}")
            return 1

    except subprocess.CalledProcessError as e:
        print(f"\nConversion failed with error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Test Gemma3 1B model conversion (FP16, 2 chunks, 4096 context)"
    )
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it",
                       help="HuggingFace model name (default: google/gemma-3-1b-it)")
    parser.add_argument("--output", type=str, default="/tmp/test-gemma3-1b-fp16-2chunk",
                       help="Output directory (default: /tmp/test-gemma3-1b-fp16-2chunk)")
    parser.add_argument("--chunks", type=int, default=2,
                       help="Number of chunks (default: 2)")
    parser.add_argument("--context", type=int, default=4096,
                       help="Context length (default: 4096)")
    parser.add_argument("--batch", type=int, default=64,
                       help="Batch size (default: 64)")
    parser.add_argument("--skip-check", action="store_true", default=True,
                       help="Skip dependency check (default: True)")
    parser.add_argument("--run-check", action="store_true",
                       help="Run dependency check (overrides default skip)")

    args = parser.parse_args()

    # Skip check is default True, but --run-check overrides it
    skip_check = args.skip_check and not args.run_check

    return run_gemma3_1b_fp16_2chunk(
        model_name=args.model,
        output_dir=args.output,
        num_chunks=args.chunks,
        context_length=args.context,
        batch_size=args.batch,
        skip_check=skip_check
    )


if __name__ == "__main__":
    sys.exit(main())

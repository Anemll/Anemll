#!/usr/bin/env python3
#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

"""
Test script for Gemma3 1B model conversion (standard chunked format).
This script tests the Gemma3 1B model conversion pipeline with recommended settings.

Unlike the 270M model which uses monolithic format, the 1B model uses the standard
chunked format with separate embeddings, FFN, and LM head components.

Usage:
    # Test with default settings (LUT6, 4096 context, single chunk)
    python tests/test_gemma3_1B_model.py

    # Test with custom output directory
    python tests/test_gemma3_1B_model.py --output /path/to/output

    # Test with different LUT quantization
    python tests/test_gemma3_1B_model.py --lut 4

    # Test with different context length
    python tests/test_gemma3_1B_model.py --context 2048

    # Skip dependency check (useful with uv or non-standard pip setup)
    python tests/test_gemma3_1B_model.py --skip-check

Note: This script is optimized for Gemma 3 1B with:
    - 4096 context length (uses 4-function model with rotation support)
    - LUT6 quantization for all model parts
    - Single chunk (no FFN splitting)
    - Non-monolithic format (separate embeddings, FFN, LM head)

For Gemma3 270M (monolithic format), use test_gemma3_model.py instead.

Note: If you encounter dependency check failures with uv or non-standard pip setups,
      use --skip-check to bypass the check.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_gemma3_1b_tests(model_name: str = "google/gemma-3-1b-it",
                        output_dir: str = "/tmp/test-gemma3-1b",
                        num_chunks: int = 1,
                        lut_bits: int = 6,
                        context_length: int = 4096,
                        batch_size: int = 64,
                        skip_check: bool = False):
    """Run Gemma3 1B model conversion and testing.

    Args:
        model_name: HuggingFace model name or local path
        output_dir: Directory for converted models
        num_chunks: Number of chunks to split FFN/prefill (1 = no chunking)
        lut_bits: LUT quantization bits (default: 6)
        context_length: Context length for conversion (default: 4096)
        batch_size: Batch size for prefill (default: 64)
        skip_check: If True, skip dependency check (useful with uv/non-standard pip)
    """

    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print("=== Gemma3 1B Model Test Suite (Chunked Format) ===")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Chunks: {num_chunks}")
    print(f"LUT: {lut_bits}")
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

    # Add LUT options for all three parts
    if lut_bits:
        cmd.extend(["--lut1", str(lut_bits)])  # Embeddings
        cmd.extend(["--lut2", str(lut_bits)])  # FFN
        cmd.extend(["--lut3", str(lut_bits)])  # LM Head
    else:
        # No LUT - use empty values
        cmd.extend(["--lut1", ""])
        cmd.extend(["--lut2", ""])
        cmd.extend(["--lut3", ""])

    # Skip dependency check if requested (useful with uv or non-standard pip)
    if skip_check:
        cmd.append("--skip-check")

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True, cwd=project_root)

        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("Gemma3 1B conversion completed successfully!")
            print("=" * 60)
            print()
            print("To test the converted model, run:")
            print(f"  python tests/chat.py --meta {output_dir}/meta.yaml --prompt \"Hello!\"")
            print()
            print("Features enabled:")
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
        description="Test Gemma3 1B model conversion (LUT6, 4096 context)"
    )
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it",
                       help="HuggingFace model name (default: google/gemma-3-1b-it)")
    parser.add_argument("--output", type=str, default="/tmp/test-gemma3-1b",
                       help="Output directory (default: /tmp/test-gemma3-1b)")
    parser.add_argument("--chunks", type=int, default=1,
                       help="Number of chunks (default: 1)")
    parser.add_argument("--lut", type=int, default=6,
                       help="LUT bits for all parts (default: 6)")
    parser.add_argument("--context", type=int, default=4096,
                       help="Context length (default: 4096)")
    parser.add_argument("--batch", type=int, default=64,
                       help="Batch size (default: 64)")
    parser.add_argument("--skip-check", action="store_true", default=True,
                       help="Skip dependency check (default: True, useful with uv or non-standard pip)")
    parser.add_argument("--run-check", action="store_true",
                       help="Run dependency check (overrides default skip)")

    args = parser.parse_args()

    # Skip check is default True, but --run-check overrides it
    skip_check = args.skip_check and not args.run_check

    return run_gemma3_1b_tests(
        model_name=args.model,
        output_dir=args.output,
        num_chunks=args.chunks,
        lut_bits=args.lut,
        context_length=args.context,
        batch_size=args.batch,
        skip_check=skip_check
    )


if __name__ == "__main__":
    sys.exit(main())

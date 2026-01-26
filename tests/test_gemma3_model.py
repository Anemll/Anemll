#!/usr/bin/env python3
#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

"""
Test script for Gemma3 270M model conversion (monolithic format with argmax).
This script tests the Gemma3 270M model conversion pipeline with monolithic output.

The 270M model is small enough to fit in a single monolithic CoreML model,
making it ideal for quick testing and development. Uses argmax in model for
efficient token generation (outputs token IDs instead of full logits).

Usage:
    # Test with default Gemma3 270M model (smallest, fastest test)
    python tests/test_gemma3_model.py

    # Test with custom output directory
    python tests/test_gemma3_model.py --output /tmp/gemma3-test

    # Test with LUT quantization (default: LUT4)
    python tests/test_gemma3_model.py --lut 6

    # Test without argmax
    python tests/test_gemma3_model.py --no-argmax

    # Skip dependency check (useful with uv or non-standard pip setup)
    python tests/test_gemma3_model.py --skip-check

Note: For Gemma3 1B model with 4096 context (non-monolithic/chunked format),
      use test_gemma3_1B_model.py instead.

Note: If you encounter dependency check failures with uv or non-standard pip setups,
      use --skip-check to bypass the check.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_gemma3_tests(model_name: str = "google/gemma-3-270m-it",
                     output_dir: str = "/tmp/test-gemma3-270m",
                     lut_bits: int = 4,
                     context_length: int = 512,
                     batch_size: int = 64,
                     use_argmax: bool = True,
                     skip_check: bool = False):
    """Run Gemma3 270M model conversion and testing (monolithic format).

    Args:
        model_name: HuggingFace model name or local path
        output_dir: Directory for converted models
        lut_bits: LUT quantization bits (default: 4)
        context_length: Context length for conversion
        batch_size: Batch size for prefill
        use_argmax: If True, compute argmax inside model (default: True)
        skip_check: If True, skip dependency check (useful with uv/non-standard pip)

    Note: The 270M model produces a monolithic CoreML model suitable for
          quick testing. For larger models like 1B, use test_gemma3_1B_model.py.
    """

    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print("=== Gemma3 270M Model Test Suite (Monolithic + Argmax) ===")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"LUT: {lut_bits}")
    print(f"Context: {context_length}")
    print(f"Batch: {batch_size}")
    print(f"Argmax: {use_argmax}")
    print()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Use convert_monolith.sh for monolithic model conversion
    convert_script = project_root / "anemll" / "utils" / "convert_monolith.sh"
    if not convert_script.exists():
        print(f"Error: Convert script not found at {convert_script}")
        return 1

    cmd = [
        str(convert_script),
        "--model", model_name,
        "--output", output_dir,
        "--context", str(context_length),
        "--batch", str(batch_size),
        "--lut", str(lut_bits),
        "--prefix", "gemma3",
    ]

    # Add argmax flag for efficient token generation
    if use_argmax:
        cmd.append("--argmax")

    # Skip dependency check if requested (useful with uv or non-standard pip)
    if skip_check:
        cmd.append("--skip-check")

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True, cwd=project_root)

        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("Gemma3 270M conversion completed successfully!")
            print("=" * 60)
            print()
            print("To test the converted model, run:")
            print(f"  python tests/chat.py --meta {output_dir}/meta.yaml --prompt \"Hello!\"")
            print()
            print("Features:")
            print("  - Monolithic model (single CoreML file)")
            if use_argmax:
                print("  - Argmax in model (outputs token IDs, not logits)")
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
        description="Test Gemma3 270M model conversion (monolithic format with argmax)"
    )
    parser.add_argument("--model", type=str, default="google/gemma-3-270m-it",
                       help="HuggingFace model name (default: google/gemma-3-270m-it)")
    parser.add_argument("--output", type=str, default="/tmp/test-gemma3-270m",
                       help="Output directory (default: /tmp/test-gemma3-270m)")
    parser.add_argument("--lut", type=int, default=4,
                       help="LUT bits for all parts (default: 4)")
    parser.add_argument("--context", type=int, default=512,
                       help="Context length (default: 512)")
    parser.add_argument("--batch", type=int, default=64,
                       help="Batch size (default: 64)")
    parser.add_argument("--no-argmax", action="store_true",
                       help="Disable argmax in model (output full logits)")
    parser.add_argument("--skip-check", action="store_true", default=True,
                       help="Skip dependency check (default: True, useful with uv or non-standard pip)")
    parser.add_argument("--run-check", action="store_true",
                       help="Run dependency check (overrides default skip)")

    args = parser.parse_args()

    # Skip check is default True, but --run-check overrides it
    skip_check = args.skip_check and not args.run_check

    return run_gemma3_tests(
        model_name=args.model,
        output_dir=args.output,
        lut_bits=args.lut,
        context_length=args.context,
        batch_size=args.batch,
        use_argmax=not args.no_argmax,
        skip_check=skip_check
    )


if __name__ == "__main__":
    sys.exit(main())

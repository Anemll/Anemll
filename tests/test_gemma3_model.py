#!/usr/bin/env python3
#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

"""
Test script for Gemma3 model conversion and inference.
This script tests the Gemma3 model conversion pipeline.

Usage:
    # Test with default Gemma3 270M model (smallest, fastest test)
    python tests/test_gemma3_model.py

    # Test with specific model
    python tests/test_gemma3_model.py --model google/gemma-3-270m-it

    # Test with custom output directory
    python tests/test_gemma3_model.py --output /tmp/gemma3-test

    # Test without LUT quantization (default)
    python tests/test_gemma3_model.py --no-lut

    # Test with LUT quantization
    python tests/test_gemma3_model.py --lut 6
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_gemma3_tests(model_name: str = "google/gemma-3-1b-it",
                     output_dir: str = "/tmp/test-gemma3",
                     num_chunks: int = 1,
                     lut_bits: int = None,
                     context_length: int = 512):
    """Run Gemma3 model conversion and testing.

    Args:
        model_name: HuggingFace model name or local path
        output_dir: Directory for converted models
        num_chunks: Number of chunks to split FFN/prefill (1 = no chunking)
        lut_bits: LUT quantization bits (None = no quantization)
        context_length: Context length for conversion
    """

    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print("=== Gemma3 Model Test Suite ===")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Chunks: {num_chunks}")
    print(f"LUT: {lut_bits if lut_bits else 'none'}")
    print(f"Context: {context_length}")
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
        "--batch", "64",
        "--chunk", str(num_chunks),
        "--prefix", "gemma3",
    ]

    # Add LUT options
    if lut_bits:
        cmd.extend(["--lut2", str(lut_bits)])
        cmd.extend(["--lut3", str(lut_bits)])
    else:
        # No LUT - use empty values
        cmd.extend(["--lut1", ""])
        cmd.extend(["--lut2", ""])
        cmd.extend(["--lut3", ""])

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True, cwd=project_root)

        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("Gemma3 conversion and test completed successfully!")
            print("=" * 60)
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
    parser = argparse.ArgumentParser(description="Test Gemma3 model conversion")
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it",
                       help="HuggingFace model name (default: google/gemma-3-1b-it)")
    parser.add_argument("--output", type=str, default="/tmp/test-gemma3",
                       help="Output directory (default: /tmp/test-gemma3)")
    parser.add_argument("--chunks", type=int, default=1,
                       help="Number of chunks (default: 1)")
    parser.add_argument("--lut", type=int, default=None,
                       help="LUT bits (default: none)")
    parser.add_argument("--no-lut", action="store_true",
                       help="Disable LUT quantization")
    parser.add_argument("--context", type=int, default=512,
                       help="Context length (default: 512)")

    args = parser.parse_args()

    # Handle --no-lut flag
    lut_bits = None if args.no_lut else args.lut

    return run_gemma3_tests(
        model_name=args.model,
        output_dir=args.output,
        num_chunks=args.chunks,
        lut_bits=lut_bits,
        context_length=args.context
    )


if __name__ == "__main__":
    sys.exit(main())

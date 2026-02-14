#!/usr/bin/env python3
"""
Combine Gemma3 split cache monolithic models into multi-function model.

This creates a combined model with:
- 'infer': Single token inference (fill mode, for positions < sliding_window)
- 'infer_rotate': Single token inference (rotation mode, for positions >= sliding_window)
- 'prefill': Batch token processing (fill mode, for positions 0 to sliding_window-1)
- 'prefill_rotate': Batch token processing (rotation mode, for positions >= sliding_window)

Run with:
    python tests/dev/combine_gemma3_split_cache.py
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import coremltools as ct


def combine_models(infer_path: str, infer_rotate_path: str, prefill_path: str,
                   prefill_rotate_path: str, output_path: str) -> bool:
    """Combine infer, infer_rotate, prefill, and prefill_rotate models into multi-function model.

    Args:
        infer_path: Path to infer model (fill mode, for pos < 512)
        infer_rotate_path: Path to infer_rotate model (rotation mode, for pos >= 512)
                          If None, only infer is used for all positions.
        prefill_path: Path to prefill model (fill mode, for pos 0 to 511)
        prefill_rotate_path: Path to prefill_rotate model (rotation mode, for pos >= 512)
                            If None, only prefill is used for all positions.
        output_path: Path for combined output model
    """

    print(f"\nCombining models:")
    print(f"  Infer:          {infer_path}")
    if infer_rotate_path:
        print(f"  Infer_rotate:   {infer_rotate_path}")
    print(f"  Prefill:        {prefill_path}")
    if prefill_rotate_path:
        print(f"  Prefill_rotate: {prefill_rotate_path}")
    print(f"  Output:         {output_path}")

    # Check files exist
    if not os.path.exists(infer_path):
        print(f"Error: Infer model not found: {infer_path}")
        return False
    if infer_rotate_path and not os.path.exists(infer_rotate_path):
        print(f"Error: Infer_rotate model not found: {infer_rotate_path}")
        return False
    if not os.path.exists(prefill_path):
        print(f"Error: Prefill model not found: {prefill_path}")
        return False
    if prefill_rotate_path and not os.path.exists(prefill_rotate_path):
        print(f"Error: Prefill_rotate model not found: {prefill_rotate_path}")
        return False

    try:
        # Create multifunction descriptor
        desc = ct.utils.MultiFunctionDescriptor()

        # Add infer model as 'infer' function (fill mode)
        print("\nAdding infer model (fill mode for pos < 512)...")
        desc.add_function(
            infer_path,
            src_function_name="main",
            target_function_name="infer"
        )

        # Add infer_rotate model as 'infer_rotate' function (rotation mode)
        if infer_rotate_path:
            print("Adding infer_rotate model (rotation mode for pos >= 512)...")
            desc.add_function(
                infer_rotate_path,
                src_function_name="main",
                target_function_name="infer_rotate"
            )

        # Add prefill model as 'prefill' function (fill mode)
        print("Adding prefill model (fill mode for pos 0 to 511)...")
        desc.add_function(
            prefill_path,
            src_function_name="main",
            target_function_name="prefill"
        )

        # Add prefill_rotate model as 'prefill_rotate' function (rotation mode)
        if prefill_rotate_path:
            print("Adding prefill_rotate model (rotation mode for pos >= 512)...")
            desc.add_function(
                prefill_rotate_path,
                src_function_name="main",
                target_function_name="prefill_rotate"
            )

        # Set default function
        desc.default_function_name = "infer"

        # Save combined model
        print(f"\nSaving combined model to: {output_path}")

        # Remove existing output if present
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        ct.utils.save_multifunction(desc, output_path)

        # Load the saved model to verify
        print("\nLoading combined model to verify...")
        combined_model = ct.models.MLModel(output_path)

        print("\nCombined model info:")
        # Get functions from spec
        spec = combined_model.get_spec()
        if hasattr(spec, 'description') and hasattr(spec.description, 'functions'):
            available_funcs = [f.name for f in spec.description.functions]
            print(f"  Functions: {available_funcs}")
        else:
            print("  Could not enumerate functions")

        # Verify expected functions are present
        print("\nVerifying functions...")
        expected_funcs = ["infer", "prefill"]
        if infer_rotate_path:
            expected_funcs.insert(1, "infer_rotate")
        if prefill_rotate_path:
            expected_funcs.append("prefill_rotate")
        for func_name in expected_funcs:
            if func_name in available_funcs:
                print(f"  {func_name}: OK")
            else:
                print(f"  {func_name}: NOT FOUND")

        print("\nModel combination complete!")
        print("\nUsage:")
        print("  - prefill: Fill positions 0 to sliding_window-1 (512)")
        print("  - prefill_rotate: Fill positions >= sliding_window with cache rotation")
        print("  - infer: Generate tokens at positions < sliding_window (512)")
        print("  - infer_rotate: Generate tokens at positions >= sliding_window (512)")
        print("\nFor long context (> 512 tokens):")
        print("  1. Use prefill for positions 0-511")
        print("  2. Use prefill_rotate for positions 512+")
        print("  3. Use infer_rotate for token generation")
        return True

    except Exception as e:
        print(f"Error combining models: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine Gemma3 split cache models")
    parser.add_argument("--input-dir", default="/Volumes/Models/ANE/gemma3_split_cache_1024",
                       help="Input directory containing the models")
    parser.add_argument("--prefix", default="gemma3_split1024",
                       help="Model filename prefix")
    parser.add_argument("--suffix", default="_LUT6_CTX512_ATT512",
                       help="Model filename suffix (before .mlpackage)")
    parser.add_argument("--output", default=None,
                       help="Output path (default: {input_dir}/gemma3_split_cache.mlpackage)")
    parser.add_argument("--no-rotate", action="store_true",
                       help="Skip infer_rotate model (legacy mode)")
    parser.add_argument("--no-prefill-rotate", action="store_true",
                       help="Skip prefill_rotate model (for models without long context support)")
    args = parser.parse_args()

    input_dir = args.input_dir
    prefix = args.prefix
    suffix = args.suffix

    infer_path = os.path.join(input_dir, f"{prefix}_monolithic{suffix}.mlpackage")
    infer_rotate_path = os.path.join(input_dir, f"{prefix}_monolithic_rotate{suffix}.mlpackage") if not args.no_rotate else None
    prefill_path = os.path.join(input_dir, f"{prefix}_monolithic_prefill{suffix}.mlpackage")
    prefill_rotate_path = os.path.join(input_dir, f"{prefix}_monolithic_prefill_rotate{suffix}.mlpackage") if not args.no_prefill_rotate else None
    output_path = args.output or os.path.join(input_dir, "gemma3_split_cache.mlpackage")

    success = combine_models(infer_path, infer_rotate_path, prefill_path, prefill_rotate_path, output_path)

    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: Combined model created!")
        print(f"Output: {output_path}")
        print("=" * 60)
    else:
        print("\nFailed to combine models")
        sys.exit(1)

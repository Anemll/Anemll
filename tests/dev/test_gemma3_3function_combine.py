#!/usr/bin/env python3
"""
Test script for combining Gemma3 models with 3 functions:
- Non-rotate file: infer + prefill + infer_rotate (3 functions)
- Rotate file: prefill_rotate only (1 function)

This addresses the ANE limitation where CPU_AND_NE fails to load 'prefill'
from a 2-function model but may work with different combinations.
"""

import os
import sys
import argparse
import glob
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from pathlib import Path

def find_source_files(input_dir, prefix, lut_bits, chunk_no, total_chunks):
    """Find source mlpackage files for a specific chunk."""
    lut_suffix = f'_lut{lut_bits}' if lut_bits else ''
    chunk_suffix = f'_chunk_{chunk_no:02d}of{total_chunks:02d}'

    files = {
        'infer': f'{prefix}_FFN{lut_suffix}{chunk_suffix}.mlpackage',
        'prefill': f'{prefix}_prefill{lut_suffix}{chunk_suffix}.mlpackage',
        'infer_rotate': f'{prefix}_FFN_rotate{lut_suffix}{chunk_suffix}.mlpackage',
        'prefill_rotate': f'{prefix}_prefill_rotate{lut_suffix}{chunk_suffix}.mlpackage',
    }

    result = {}
    for func_name, filename in files.items():
        path = os.path.join(input_dir, filename)
        if os.path.exists(path):
            result[func_name] = path
            print(f"  Found {func_name}: {filename}")
        else:
            print(f"  Missing {func_name}: {filename}")

    return result


def combine_3function(source_files, output_path, prefix, lut_bits, chunk_no, total_chunks):
    """
    Combine infer, prefill, and infer_rotate into a single 3-function model.
    """
    lut_suffix = f'_lut{lut_bits}' if lut_bits else ''
    chunk_suffix = f'_chunk_{chunk_no:02d}of{total_chunks:02d}'

    print(f"\nCombining 3 functions into: {output_path}")

    # Load source models
    infer_model = ct.models.MLModel(source_files['infer'])
    prefill_model = ct.models.MLModel(source_files['prefill'])
    infer_rotate_model = ct.models.MLModel(source_files['infer_rotate'])

    # Create multi-function descriptor
    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(
        source_files['infer'],
        src_function_name='main',
        target_function_name='infer'
    )
    desc.add_function(
        source_files['prefill'],
        src_function_name='main',
        target_function_name='prefill'
    )
    desc.add_function(
        source_files['infer_rotate'],
        src_function_name='main',
        target_function_name='infer_rotate'
    )
    desc.default_function_name = 'infer'

    # Save combined model
    print(f"  Adding function: infer")
    print(f"  Adding function: prefill")
    print(f"  Adding function: infer_rotate")
    print(f"  Default function: infer")

    combined = ct.utils.MultiFunctionDescriptor.generate(desc)
    combined.save(output_path)
    print(f"  Saved to: {output_path}")

    return output_path


def save_prefill_rotate_separate(source_files, output_path):
    """
    Save prefill_rotate as a separate single-function model.
    """
    print(f"\nSaving prefill_rotate separately: {output_path}")

    # Load and re-save with proper function name
    model = ct.models.MLModel(source_files['prefill_rotate'])

    # Create single-function model with explicit name
    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(
        source_files['prefill_rotate'],
        src_function_name='main',
        target_function_name='prefill_rotate'
    )
    desc.default_function_name = 'prefill_rotate'

    combined = ct.utils.MultiFunctionDescriptor.generate(desc)
    combined.save(output_path)
    print(f"  Saved to: {output_path}")

    return output_path


def test_loading(model_path, compute_unit, function_names):
    """Test loading a model with specific functions."""
    print(f"\nTesting loading: {Path(model_path).name}")
    print(f"  Compute unit: {compute_unit}")

    results = {}
    for func_name in function_names:
        try:
            model = ct.models.CompiledMLModel(
                str(model_path),
                compute_unit,
                function_name=func_name
            )
            results[func_name] = True
            print(f"  {func_name}: SUCCESS")
        except Exception as e:
            results[func_name] = False
            print(f"  {func_name}: FAILED - {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test combining Gemma3 models with 3 functions (infer+prefill+infer_rotate)'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing source mlpackage files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: same as input)')
    parser.add_argument('--prefix', type=str, default='gemma3',
                       help='Model prefix (default: gemma3)')
    parser.add_argument('--lut', type=int, default=4,
                       help='LUT bits (default: 4)')
    parser.add_argument('--chunk', type=int, required=True,
                       help='Total number of chunks')
    parser.add_argument('--chunk-no', type=int, default=1,
                       help='Chunk number to process (default: 1)')
    parser.add_argument('--compile', action='store_true',
                       help='Compile to mlmodelc after combining')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing compiled models, skip combining')
    args = parser.parse_args()

    output_dir = args.output or args.input
    os.makedirs(output_dir, exist_ok=True)

    lut_suffix = f'_lut{args.lut}' if args.lut else ''
    chunk_suffix = f'_chunk_{args.chunk_no:02d}of{args.chunk:02d}'

    # Output file names
    main_output = os.path.join(
        output_dir,
        f'{args.prefix}_FFN_PF_3func{lut_suffix}{chunk_suffix}.mlpackage'
    )
    rotate_output = os.path.join(
        output_dir,
        f'{args.prefix}_prefill_rotate{lut_suffix}{chunk_suffix}.mlpackage'
    )

    main_compiled = main_output.replace('.mlpackage', '.mlmodelc')
    rotate_compiled = rotate_output.replace('.mlpackage', '.mlmodelc')

    if not args.test_only:
        print(f"\n{'='*60}")
        print(f"Finding source files in: {args.input}")
        print(f"{'='*60}")

        source_files = find_source_files(
            args.input, args.prefix, args.lut, args.chunk_no, args.chunk
        )

        required = ['infer', 'prefill', 'infer_rotate', 'prefill_rotate']
        missing = [f for f in required if f not in source_files]
        if missing:
            print(f"\nError: Missing required files: {missing}")
            return 1

        print(f"\n{'='*60}")
        print(f"Combining models")
        print(f"{'='*60}")

        # Combine 3 functions
        combine_3function(
            source_files, main_output,
            args.prefix, args.lut, args.chunk_no, args.chunk
        )

        # Save prefill_rotate separately
        save_prefill_rotate_separate(source_files, rotate_output)

        if args.compile:
            print(f"\n{'='*60}")
            print(f"Compiling models")
            print(f"{'='*60}")

            import subprocess

            for pkg_path in [main_output, rotate_output]:
                compiled_path = pkg_path.replace('.mlpackage', '.mlmodelc')
                print(f"\nCompiling: {Path(pkg_path).name}")
                cmd = [
                    'xcrun', 'coremlcompiler', 'compile',
                    pkg_path, output_dir,
                    '--add-mlprogram-if-eligible', 'force'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  SUCCESS: {Path(compiled_path).name}")
                else:
                    print(f"  FAILED: {result.stderr}")
                    return 1

    # Test loading
    print(f"\n{'='*60}")
    print(f"Testing model loading")
    print(f"{'='*60}")

    if not os.path.exists(main_compiled):
        print(f"\nCompiled model not found: {main_compiled}")
        print("Run with --compile flag or compile manually first")
        return 1

    # Test with different compute units
    for cu_name, cu in [
        ('ALL', ct.ComputeUnit.ALL),
        ('CPU_AND_NE', ct.ComputeUnit.CPU_AND_NE),
        ('CPU_ONLY', ct.ComputeUnit.CPU_ONLY),
    ]:
        print(f"\n--- Testing with ComputeUnit.{cu_name} ---")

        # Test 3-function model
        test_loading(main_compiled, cu, ['infer', 'prefill', 'infer_rotate'])

        # Test prefill_rotate model
        if os.path.exists(rotate_compiled):
            test_loading(rotate_compiled, cu, ['prefill_rotate'])

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"{'='*60}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

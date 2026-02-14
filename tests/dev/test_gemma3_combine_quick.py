#!/usr/bin/env python3
"""
Quick test: Combine infer + prefill + infer_rotate into one model,
keep prefill_rotate separate. Test loading with CPU_AND_NE.
"""

import os
import sys
import subprocess
import coremltools as ct

INPUT_DIR = '/Volumes/Models/ANE/gemma3_4b_qat4_2chunk'
OUTPUT_DIR = INPUT_DIR  # Same directory
PREFIX = 'gemma3'
LUT = 4
CHUNK_NO = 1
TOTAL_CHUNKS = 2

# Source files
infer_src = f'{INPUT_DIR}/{PREFIX}_FFN_lut{LUT}_chunk_{CHUNK_NO:02d}of{TOTAL_CHUNKS:02d}.mlpackage'
prefill_src = f'{INPUT_DIR}/{PREFIX}_prefill_lut{LUT}_chunk_{CHUNK_NO:02d}of{TOTAL_CHUNKS:02d}.mlpackage'
infer_rot_src = f'{INPUT_DIR}/{PREFIX}_FFN_rotate_lut{LUT}_chunk_{CHUNK_NO:02d}of{TOTAL_CHUNKS:02d}.mlpackage'
prefill_rot_src = f'{INPUT_DIR}/{PREFIX}_prefill_rotate_lut{LUT}_chunk_{CHUNK_NO:02d}of{TOTAL_CHUNKS:02d}.mlpackage'

# Output files
output_3func = f'{OUTPUT_DIR}/{PREFIX}_FFN_PF_3func_lut{LUT}_chunk_{CHUNK_NO:02d}of{TOTAL_CHUNKS:02d}.mlpackage'
output_prefill_rot = f'{OUTPUT_DIR}/{PREFIX}_prefill_rot_only_lut{LUT}_chunk_{CHUNK_NO:02d}of{TOTAL_CHUNKS:02d}.mlpackage'


def check_files():
    print("Checking source files...")
    files = [
        ('infer', infer_src),
        ('prefill', prefill_src),
        ('infer_rotate', infer_rot_src),
        ('prefill_rotate', prefill_rot_src),
    ]
    all_ok = True
    for name, path in files:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  {name}: {status} - {os.path.basename(path)}")
        if not exists:
            all_ok = False
    return all_ok


def combine_3func():
    """Combine infer + prefill + infer_rotate into one model."""
    print(f"\nCombining 3 functions into: {os.path.basename(output_3func)}")

    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(infer_src, src_function_name='main', target_function_name='infer')
    desc.add_function(prefill_src, src_function_name='main', target_function_name='prefill')
    desc.add_function(infer_rot_src, src_function_name='main', target_function_name='infer_rotate')
    desc.default_function_name = 'infer'

    ct.utils.save_multifunction(desc, output_3func)
    print(f"  Saved: {output_3func}")


def save_prefill_rot_only():
    """Save prefill_rotate as single-function model."""
    print(f"\nSaving prefill_rotate only: {os.path.basename(output_prefill_rot)}")

    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(prefill_rot_src, src_function_name='main', target_function_name='prefill_rotate')
    desc.default_function_name = 'prefill_rotate'

    ct.utils.save_multifunction(desc, output_prefill_rot)
    print(f"  Saved: {output_prefill_rot}")


def compile_model(mlpackage_path):
    """Compile mlpackage to mlmodelc with force-mlprogram."""
    print(f"\nCompiling: {os.path.basename(mlpackage_path)}")
    cmd = [
        'xcrun', 'coremlcompiler', 'compile',
        mlpackage_path, OUTPUT_DIR,
        '--add-mlprogram-if-eligible', 'force'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        compiled = mlpackage_path.replace('.mlpackage', '.mlmodelc')
        print(f"  SUCCESS: {os.path.basename(compiled)}")
        return compiled
    else:
        print(f"  FAILED: {result.stderr}")
        return None


def test_loading(compiled_path, functions):
    """Test loading with CPU_AND_NE."""
    print(f"\nTesting: {os.path.basename(compiled_path)}")

    for cu_name, cu in [('CPU_AND_NE', ct.ComputeUnit.CPU_AND_NE)]:
        print(f"  ComputeUnit: {cu_name}")
        for func in functions:
            try:
                model = ct.models.CompiledMLModel(compiled_path, cu, function_name=func)
                print(f"    {func}: SUCCESS")
            except Exception as e:
                print(f"    {func}: FAILED - {str(e)[:80]}")


def main():
    print("="*60)
    print("Gemma3 3-Function Combine Test")
    print("="*60)

    if not check_files():
        print("\nMissing source files!")
        return 1

    # Step 1: Combine 3 functions
    combine_3func()

    # Step 2: Save prefill_rotate separately
    save_prefill_rot_only()

    # Step 3: Compile both
    compiled_3func = compile_model(output_3func)
    compiled_prefill_rot = compile_model(output_prefill_rot)

    if not compiled_3func or not compiled_prefill_rot:
        print("\nCompilation failed!")
        return 1

    # Step 4: Test loading
    print("\n" + "="*60)
    print("Testing loading with CPU_AND_NE")
    print("="*60)

    test_loading(compiled_3func, ['infer', 'prefill', 'infer_rotate'])
    test_loading(compiled_prefill_rot, ['prefill_rotate'])

    # Also test the original 2-function split-rotate files for comparison
    print("\n" + "="*60)
    print("Comparison: Original 2-function split-rotate files")
    print("="*60)

    orig_nonrot = f'{INPUT_DIR}/{PREFIX}_FFN_PF_lut{LUT}_chunk_{CHUNK_NO:02d}of{TOTAL_CHUNKS:02d}.mlmodelc'
    orig_rot = f'{INPUT_DIR}/{PREFIX}_FFN_PF_lut{LUT}_chunk_{CHUNK_NO:02d}of{TOTAL_CHUNKS:02d}_rot.mlmodelc'

    if os.path.exists(orig_nonrot):
        test_loading(orig_nonrot, ['infer', 'prefill'])
    if os.path.exists(orig_rot):
        test_loading(orig_rot, ['infer_rotate', 'prefill_rotate'])

    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    return 0


if __name__ == '__main__':
    sys.exit(main())

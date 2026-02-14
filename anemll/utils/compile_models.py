#!/usr/bin/env python3
#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

import os
import sys
import glob
import subprocess
from typing import List, Optional
import argparse
import coremltools as ct

def compile_model(model_path: str, target_dir: str = "./", force_mlprogram: bool = False) -> bool:
    """Compile a single CoreML model using coremlcompiler.
    
    Args:
        model_path: Path to .mlpackage file
        target_dir: Target directory for compiled model
    
    Returns:
        bool: True if compilation succeeded
    """
    try:
        cmd = ["xcrun", "coremlcompiler", "compile", model_path, target_dir]
        # Optionally force ML Program when compiling .mlpackage files.
        if force_mlprogram and model_path.endswith(".mlpackage"):
            cmd += ["--add-mlprogram-if-eligible", "force"]
        print(f"\nCompiling {os.path.basename(model_path)}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Compilation successful")
            return True
        else:
            print(f"Compilation failed with error:\n{result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error compiling {model_path}: {str(e)}")
        return False

def get_part_name(part: str) -> str:
    """Map part number to model name component.

    Args:
        part: Part number ("1", "2", "3", "monolithic")

    Returns:
        str: Model name component
    """
    part_map = {
        "1": "embeddings",
        "2": "FFN_PF",  # For combined FFN/prefill
        "3": "lm_head",
        "monolithic": "monolithic_full"  # For combined monolithic model
    }
    return part_map.get(part, part)

def find_chunk_models(lut_bits: int, num_chunks: int, part: str = "2", prefix: str = "llama") -> List[str]:
    """Find chunked model files matching pattern.
    
    Args:
        lut_bits: LUT quantization bits
        num_chunks: Number of chunks
        part: Model part (1=embeddings, 2=FFN/prefill, 3=lm_head)
        prefix: Model name prefix
    
    Returns:
        List of matching model paths
    """
    # For part 2, look for combined FFN_PF files
    if part == "2":
        pattern = f"{prefix}_FFN_PF_lut{lut_bits}_chunk_*of{num_chunks:02d}.mlpackage"
        models = sorted(glob.glob(pattern))
        if not models:
            print("No combined FFN_PF models found, looking for individual FFN and prefill models...")
            ffn_pattern = f"{prefix}_FFN_lut{lut_bits}_chunk_*of{num_chunks:02d}.mlpackage"
            prefill_pattern = f"{prefix}_prefill_lut{lut_bits}_chunk_*of{num_chunks:02d}.mlpackage"
            models = sorted(glob.glob(ffn_pattern) + glob.glob(prefill_pattern))
        return models
    else:
        part_name = get_part_name(part)
        pattern = f"{prefix}_{part_name}_lut{lut_bits}_chunk_*of{num_chunks:02d}.mlpackage"
        return sorted(glob.glob(pattern))

def compile_chunks(lut_bits: int, num_chunks: int, target_dir: str = "./", force_mlprogram: bool = False) -> bool:
    """Compile all chunks of a model.
    
    Args:
        lut_bits: LUT quantization bits
        num_chunks: Number of chunks
        target_dir: Target directory for compiled models
    
    Returns:
        bool: True if all compilations succeeded
    """
    models = find_chunk_models(lut_bits, num_chunks)
    if not models:
        print(f"No chunk models found matching pattern")
        return False
        
    success = True
    for model in models:
        if not compile_model(model, target_dir, force_mlprogram=force_mlprogram):
            success = False
            
    return success

def compile_part(part: str, lut_bits: Optional[int] = None, target_dir: str = "./", prefix: str = "llama", force_mlprogram: bool = False) -> bool:
    """Compile a specific model part.
    
    Args:
        part: Model part (1=embeddings, 2=FFN/prefill, 3=lm_head)
        lut_bits: Optional LUT quantization bits
        target_dir: Target directory for compiled model
        prefix: Model name prefix
    
    Returns:
        bool: True if compilation succeeded
    """
    # Get part name and construct model path
    part_name = get_part_name(part)
    lut_suffix = f"_lut{lut_bits}" if lut_bits else ""
    model_path = f"{prefix}_{part_name}{lut_suffix}.mlpackage"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False
        
    return compile_model(model_path, target_dir, force_mlprogram=force_mlprogram)

def parse_lut_arg(lut_value):
    """Parse LUT argument and extract just the bits value.

    Args:
        lut_value: String value from command line (e.g., '6' or '6,4')

    Returns:
        int or None: Just the lut_bits value (per_channel is ignored for file naming)
    """
    if lut_value is None:
        return None

    lut_str = str(lut_value).strip().lower()

    # Handle "none" as no LUT quantization
    if lut_str in ('none', 'no', 'false', ''):
        return None

    if ',' in lut_value:
        # Extract just the bits value, ignore per_channel for file naming
        parts = lut_value.split(',')
        try:
            return int(parts[0])
        except ValueError:
            raise ValueError(f"Invalid LUT bits value: {parts[0]}")
    else:
        try:
            return int(lut_value)
        except ValueError:
            raise ValueError(f"Invalid LUT bits value: {lut_value}")

def parse_args():
    parser = argparse.ArgumentParser(description='Compile models to MLModelC format')
    parser.add_argument('part', type=str, help='Model part to compile (1, 2, 3, monolithic, or all)')
    parser.add_argument('--lut', type=str, help='LUT bits used in quantization (optional). Format: "bits" or "bits,per_channel" (e.g., "6" or "6,4")')
    parser.add_argument('--chunk', type=int, help='Number of chunks (for part 2)')
    parser.add_argument('--prefix', type=str, default='llama', help='Model name prefix')
    parser.add_argument('--input', type=str, default='.', help='Input directory containing models')
    parser.add_argument('--output', type=str, default=None, help='Output directory (default: same as input)')
    parser.add_argument('--recursive', action='store_true', help='When part=all, search for .mlpackage recursively')
    parser.add_argument('--force-mlprogram', action='store_true', help='Force ML Program when compiling .mlpackage models')
    parser.add_argument('--split-rotate', action='store_true', help='For Gemma3 split-rotate (compile FFN+PF files separately)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Parse LUT argument to extract just the bits value
    args.lut = parse_lut_arg(args.lut)

    # Set output dir to input dir if not specified
    output_dir = args.output if args.output else args.input
    os.makedirs(output_dir, exist_ok=True)

    if args.part == "all":
        # Compile every .mlpackage under input (optionally recursive).
        pattern = "**/*.mlpackage" if args.recursive else "*.mlpackage"
        search_root = os.path.abspath(args.input)
        models = sorted(glob.glob(os.path.join(search_root, pattern), recursive=args.recursive))
        if not models:
            print(f"Error: No .mlpackage files found under {search_root} (recursive={args.recursive})")
            return 1
        print(f"Found {len(models)} .mlpackage files under {search_root}")
        success = True
        for model_path in models:
            if not compile_model(model_path, output_dir, force_mlprogram=args.force_mlprogram):
                success = False
        return 0 if success else 1

    # Construct input filename based on part and parameters
    if args.part == '1':
        # Try both naming patterns for embeddings (with and without LUT suffix)
        if args.lut:
            input_name = f'{args.prefix}_embeddings_lut{args.lut}.mlpackage'
            input_path = os.path.join(args.input, input_name)
            if not os.path.exists(input_path):
                # Fallback to old naming pattern (without LUT suffix)
                input_name = f'{args.prefix}_embeddings.mlpackage'
                print(f"Warning: {input_name} not found, trying fallback naming pattern")
        else:
            input_name = f'{args.prefix}_embeddings.mlpackage'
    elif args.part == '3':
        # Make LUT optional for part 3
        lut_suffix = f'_lut{args.lut}' if args.lut else ''
        input_name = f'{args.prefix}_lm_head{lut_suffix}.mlpackage'
    elif args.part == '2':
        if args.chunk is None:
            print("Error: --chunk required for part 2")
            return 1
        # Make LUT optional for part 2
        lut_suffix = f'_lut{args.lut}' if args.lut else ''
        # For split-rotate, compile non-rotate and rotate files separately
        # Non-rotate file: FFN_PF_chunk_XXofYY (infer + prefill)
        # Rotate file: FFN_PF_chunk_XXofYY_rot (infer_rotate + prefill_rotate)
        # NOTE: Multi-function models REQUIRE ML Program format for function_name loading
        if args.split_rotate:
            for i in range(args.chunk):
                non_rotate_name = f'{args.prefix}_FFN_PF{lut_suffix}_chunk_{i+1:02d}of{args.chunk:02d}.mlpackage'
                rotate_name = f'{args.prefix}_FFN_PF{lut_suffix}_chunk_{i+1:02d}of{args.chunk:02d}_rot.mlpackage'
                # Always force ML Program for multi-function models (required for function_name)
                compile_model(os.path.join(args.input, non_rotate_name), output_dir, force_mlprogram=True)
                compile_model(os.path.join(args.input, rotate_name), output_dir, force_mlprogram=True)
        else:
            # Standard combined FFN_PF files
            for i in range(args.chunk):
                chunk_name = f'{args.prefix}_FFN_PF{lut_suffix}_chunk_{i+1:02d}of{args.chunk:02d}.mlpackage'
                input_path = os.path.join(args.input, chunk_name)
                compile_model(input_path, output_dir, force_mlprogram=args.force_mlprogram)
        return 0
    elif args.part == 'monolithic':
        # Compile monolithic combined model
        lut_suffix = f'_lut{args.lut}' if args.lut else ''
        input_name = f'{args.prefix}_monolithic_full{lut_suffix}.mlpackage'
        input_path = os.path.join(args.input, input_name)
        if not os.path.exists(input_path):
            print(f"Error: Monolithic model not found: {input_path}")
            return 1
        compile_model(input_path, output_dir, force_mlprogram=args.force_mlprogram)
        return 0
    else:
        print(f"Error: Invalid part {args.part}")
        return 1

    input_path = os.path.join(args.input, input_name)
    if not os.path.exists(input_path):
        print(f"Error: Model file not found: {input_path}")
        return 1
    compile_model(input_path, output_dir, force_mlprogram=args.force_mlprogram)
    return 0

if __name__ == "__main__":
    main() 

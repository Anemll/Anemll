#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

import coremltools as ct
import os
import sys
import pkg_resources
import argparse
from pathlib import Path

# Add package root to path when running as script
if __name__ == '__main__':
    import pathlib
    package_root = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.insert(0, package_root)
    from anemll.ane_converter.metadata import AddCombinedMetadata
else:
    from ..ane_converter.metadata import AddCombinedMetadata

def parse_model_args(args):
    """Parse command line arguments in the format name=model.mlpackage func=funcname."""
    models_dict = {}
    current_model = None
    
    for arg in args:
        if '=' not in arg:
            continue
            
        key, value = arg.split('=')
        if key.startswith('name'):
            current_model = value
            if current_model not in models_dict:
                models_dict[current_model] = {'path': value}
        elif key.startswith('func') and current_model is not None:
            models_dict[current_model]['function'] = value
            
    return models_dict

def combine_models_custom(models_dict):
    """Combine models based on provided dictionary of model paths and function names."""
    if not models_dict:
        print("Error: No valid model specifications provided!")
        return False
        
    desc = ct.utils.MultiFunctionDescriptor()
    models_found = False
    function_names = []
    
    # Add each model to the descriptor
    for model_info in models_dict.values():
        model_path = model_info.get('path')
        target_function_name = model_info.get('function')
        
        if not all([model_path, target_function_name]):
            print(f"Warning: Incomplete specification for model {model_path}, skipping...")
            continue
            
        if os.path.exists(model_path):
            models_found = True
            print(f"Adding model: {model_path} as function {target_function_name}")
            desc.add_function(
                model_path,
                src_function_name="main",
                target_function_name=target_function_name
            )
            function_names.append(target_function_name)
        else:
            print(f"Warning: Model {model_path} not found, skipping...")
    
    if not models_found:
        print("Error: No valid models found to combine!")
        return False
    
    # Set default function to the first specified function
    first_model = next(iter(models_dict.values()))
    desc.default_function_name = first_model.get('function')
    
    # Save the combined model
    output_path = "combined_model.mlpackage"
    print(f"\nSaving multifunction model to: {output_path}")
    combined_model = ct.utils.save_multifunction(desc, output_path)
    
    # Add metadata
    AddCombinedMetadata(combined_model, [model for model_info in models_dict.values()])
    combined_model.save(output_path)
    print("Done!")
    return True


def validate_chunk_files(num_chunks, lut_bits=None, mode=None, prefix='llama'):
    """Validate that all required chunk files exist."""
    print("\nDebug: Validating chunk files:")
    print(f"  Current dir: {os.getcwd()}")
    print(f"  Num chunks: {num_chunks}")
    print(f"  LUT bits: {lut_bits}")
    print(f"  Prefix: {prefix}")
    
    missing_files = []
    
    # Get file patterns - use "prefill" instead of "PF" to match actual files
    if lut_bits:
        ffn_template = f"{prefix}_FFN_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        pf_template = f"{prefix}_prefill_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
    else:
        ffn_template = f"{prefix}_FFN_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        pf_template = f"{prefix}_prefill_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
    
    print("\nLooking for files with patterns:")
    print(f"  FFN: {ffn_template.format(1).replace('01', '{:02d}')}")
    print(f"  PF:  {pf_template.format(1).replace('01', '{:02d}')}")
    
    print("\nFiles in directory:")
    for f in os.listdir('.'):
        if f.endswith('.mlpackage'):
            print(f"  {f}")
    
    for i in range(1, num_chunks + 1):
        # Check FFN files
        ffn_file = ffn_template.format(i)
        if not os.path.exists(ffn_file):
            missing_files.append(ffn_file)
            
        # Check prefill files
        pf_file = pf_template.format(i)
        if not os.path.exists(pf_file):
            missing_files.append(pf_file)
    
    if missing_files:
        print("\nError: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        return False
        
    return True

def combine_chunks(num_chunks, lut_bits=None, mode=None, prefix='llama'):
    """Combine FFN and prefill models into chunks."""
    try:
        # Use same naming pattern as validate_chunk_files
        if lut_bits:
            ffn_template = f"{prefix}_FFN_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            pf_template = f"{prefix}_prefill_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            combined_template = f"{prefix}_FFN_PF_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        else:
            ffn_template = f"{prefix}_FFN_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            pf_template = f"{prefix}_prefill_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            combined_template = f"{prefix}_FFN_PF_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        
        for chunk_idx in range(num_chunks):
            try:
                # Get input model paths
                ffn_path = ffn_template.format(chunk_idx + 1)
                prefill_path = pf_template.format(chunk_idx + 1)
                output_path = combined_template.format(chunk_idx + 1)
                temp_path = f"temp_{output_path}"
                
                print(f"\nProcessing chunk {chunk_idx+1}:")
                print(f"  FFN: {ffn_path}")
                print(f"  Prefill: {prefill_path}")
                print(f"  Output: {output_path}")
                
                # Load models
                ffn_model = ct.models.MLModel(ffn_path)
                prefill_model = ct.models.MLModel(prefill_path)
                
                # Create combined model
                desc = ct.utils.MultiFunctionDescriptor()
                desc.add_function(ffn_path, "main", "infer")
                desc.add_function(prefill_path, "main", "prefill")
                desc.default_function_name = "infer"
                
                print("Creating combined model...")
                ct.utils.save_multifunction(desc, temp_path)
                
                # Load the temp model to add metadata
                print("Loading combined model...")
                combined_model = ct.models.MLModel(temp_path)
                if combined_model is None:
                    raise ValueError(f"Failed to load combined model")
                
                # Add metadata and save final
                print("Adding metadata...")
                AddCombinedMetadata(combined_model, [ffn_model, prefill_model])
                print(f"Saving final model to: {output_path}")
                combined_model.save(output_path)
                
                # Clean up temp file
                import shutil
                shutil.rmtree(temp_path, ignore_errors=True)
                
                print(f"Successfully combined chunk {chunk_idx+1}")
                
            except Exception as e:
                print(f"\nError processing chunk {chunk_idx+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
                
        return True
        
    except Exception as e:
        print(f"\nError during combination process: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='Combine FFN and prefill models')
    parser.add_argument('--lut', type=int, help='LUT bits used in quantization (optional)')
    parser.add_argument('--chunk', type=int, required=True,
                      help='Number of chunks')
    parser.add_argument('--input', type=str, default='.',
                      help='Input directory containing model files (default: current directory)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output directory for combined models (default: same as input)')
    parser.add_argument('--prefix', type=str, default='llama',
                      help='Prefix for model names (default: llama)')
    return parser.parse_args()

def get_model_names(args):
    """Get input and output model names based on LUT setting"""
    if args.lut:
        ffn_template = f"{args.prefix}_FFN_lut{args.lut}_chunk_{{:02d}}of{args.chunk:02d}.mlpackage"
        pf_template = f"{args.prefix}_PF_lut{args.lut}_chunk_{{:02d}}of{args.chunk:02d}.mlpackage"
        combined_template = f"{args.prefix}_FFN_PF_lut{args.lut}_chunk_{{:02d}}of{args.chunk:02d}.mlpackage"
    else:
        ffn_template = f"{args.prefix}_FFN_chunk_{{:02d}}of{args.chunk:02d}.mlpackage"
        pf_template = f"{args.prefix}_prefill_chunk_{{:02d}}of{args.chunk:02d}.mlpackage"
        combined_template = f"{args.prefix}_FFN_PF_chunk_{{:02d}}of{args.chunk:02d}.mlpackage"
    return ffn_template, pf_template, combined_template

def combine_models(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First validate all files exist
    if not validate_chunk_files(args.chunk, args.lut, None, args.prefix):
        raise FileNotFoundError("Missing required model files")
    
    # Combine the chunks
    if combine_chunks(args.chunk, args.lut, None, args.prefix):
        print("\nAll chunks combined successfully!")
        return True
    else:
        print("\nCombination process failed.")
        return False

def main():
    args = parse_args()
    try:
        # Change to input directory for processing
        orig_dir = os.getcwd()
        os.chdir(args.input)
        
        # Run combination
        success = combine_models(args)
        
        # Move files if needed
        if success and args.output and args.input != args.output:
            output_dir = Path(args.output)
            for chunk in range(1, args.chunk + 1):
                combined_file = f"{args.prefix}_FFN_PF_lut{args.lut}_chunk_{chunk:02d}of{args.chunk:02d}.mlpackage"
                if os.path.exists(combined_file):
                    os.rename(combined_file, output_dir / combined_file)
        
        # Return to original directory
        os.chdir(orig_dir)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 
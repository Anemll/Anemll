#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

import coremltools as ct
import os
import sys
import argparse
import shutil
from pathlib import Path

# Add package root to path when running as script
if __name__ == '__main__':
    import pathlib
    package_root = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.insert(0, package_root)
    from anemll.ane_converter.metadata import AddCombinedMetadata
else:
    from ..ane_converter.metadata import AddCombinedMetadata


def _save_multifunction_dedup(sources, output_path, dedup_weights=False, verbose=False):
    """Save a multifunction model, optionally applying anemll-dedup weight dedup.

    Args:
        sources: List of (mlpackage_path, src_function_name, target_function_name).
                 First entry is the anchor for dedup.
        output_path: Where to save the combined mlpackage.
        dedup_weights: If True, run surgical weight dedup before combining.
        verbose: Print dedup details.
    """
    if dedup_weights and len(sources) > 1:
        try:
            from anemll.utils.dedup_weights import prepare_dedup_sources
        except ImportError:
            print("Warning: dedup_weights module not available, falling back to standard combine")
            dedup_weights = False

    if dedup_weights and len(sources) > 1:
        with prepare_dedup_sources(sources, verbose=verbose) as deduped:
            desc = ct.utils.MultiFunctionDescriptor()
            for path, src_fn, tgt_fn in deduped:
                desc.add_function(path, src_fn, tgt_fn)
            desc.default_function_name = sources[0][2]  # first entry's target fn
            ct.utils.save_multifunction(desc, output_path)
    else:
        desc = ct.utils.MultiFunctionDescriptor()
        for path, src_fn, tgt_fn in sources:
            desc.add_function(path, src_fn, tgt_fn)
        desc.default_function_name = sources[0][2]
        ct.utils.save_multifunction(desc, output_path)

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

def combine_chunks(num_chunks, lut_bits=None, mode=None, prefix='llama', dedup_weights=False):
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

                # Load models for metadata
                ffn_model = ct.models.MLModel(ffn_path)
                prefill_model = ct.models.MLModel(prefill_path)

                # Create combined model (with optional anemll-dedup dedup)
                print("Creating combined model...")
                sources = [(ffn_path, "main", "infer"), (prefill_path, "main", "prefill")]
                _save_multifunction_dedup(sources, temp_path, dedup_weights=dedup_weights)

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

def combine_chunks_split_rotate(num_chunks, lut_bits=None, prefix='gemma3', dedup_weights=False):
    """Combine models into 2-function files, splitting by rotation mode.

    This is for large models where 4 functions in one file is too complex for ANE.
    Creates TWO separate files per chunk:
    - Non-rotate file: 'infer' + 'prefill' functions (FFN_PF_chunk_XXofYY)
    - Rotate file: 'infer_rotate' + 'prefill_rotate' functions (FFN_PF_chunk_XXofYY_rot)

    Args:
        num_chunks: Number of chunks
        lut_bits: LUT quantization bits (or None)
        prefix: Model name prefix
        dedup_weights: If True, apply anemll-dedup weight dedup before combining

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Build file name templates
        if lut_bits:
            ffn_template = f"{prefix}_FFN_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            ffn_rotate_template = f"{prefix}_FFN_rotate_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            pf_template = f"{prefix}_prefill_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            pf_rotate_template = f"{prefix}_prefill_rotate_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            # Output templates - non-rotate and rotate files
            combined_template = f"{prefix}_FFN_PF_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            combined_rot_template = f"{prefix}_FFN_PF_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}_rot.mlpackage"
        else:
            ffn_template = f"{prefix}_FFN_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            ffn_rotate_template = f"{prefix}_FFN_rotate_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            pf_template = f"{prefix}_prefill_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            pf_rotate_template = f"{prefix}_prefill_rotate_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            combined_template = f"{prefix}_FFN_PF_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            combined_rot_template = f"{prefix}_FFN_PF_chunk_{{:02d}}of{num_chunks:02d}_rot.mlpackage"

        for chunk_idx in range(num_chunks):
            try:
                # Get input model paths
                ffn_path = ffn_template.format(chunk_idx + 1)
                ffn_rotate_path = ffn_rotate_template.format(chunk_idx + 1)
                prefill_path = pf_template.format(chunk_idx + 1)
                prefill_rotate_path = pf_rotate_template.format(chunk_idx + 1)
                output_path = combined_template.format(chunk_idx + 1)
                output_rot_path = combined_rot_template.format(chunk_idx + 1)

                print(f"\nProcessing split-rotate chunk {chunk_idx+1}:")
                print(f"  FFN: {ffn_path}")
                print(f"  FFN_rotate: {ffn_rotate_path}")
                print(f"  Prefill: {prefill_path}")
                print(f"  Prefill_rotate: {prefill_rotate_path}")
                print(f"  Output (non-rotate): {output_path}")
                print(f"  Output (rotate): {output_rot_path}")

                # Check all input files exist
                missing = []
                for path in [ffn_path, ffn_rotate_path, prefill_path, prefill_rotate_path]:
                    if not os.path.exists(path):
                        missing.append(path)
                if missing:
                    print(f"Error: Missing input files:")
                    for f in missing:
                        print(f"  - {f}")
                    return False

                # Load models for metadata
                ffn_model = ct.models.MLModel(ffn_path)
                ffn_rotate_model = ct.models.MLModel(ffn_rotate_path)
                prefill_model = ct.models.MLModel(prefill_path)
                prefill_rotate_model = ct.models.MLModel(prefill_rotate_path)

                # Create non-rotate combined model with 2 functions (infer + prefill)
                print("Creating non-rotate 2-function model (infer + prefill)...")
                sources = [(ffn_path, "main", "infer"), (prefill_path, "main", "prefill")]
                temp_path = f"temp_{output_path}"
                _save_multifunction_dedup(sources, temp_path, dedup_weights=dedup_weights)
                combined = ct.models.MLModel(temp_path)
                AddCombinedMetadata(combined, [ffn_model, prefill_model])
                combined.save(output_path)
                shutil.rmtree(temp_path, ignore_errors=True)
                print(f"  Non-rotate model saved: {output_path}")

                # Create rotate combined model with 2 functions (infer_rotate + prefill_rotate)
                print("Creating rotate 2-function model (infer_rotate + prefill_rotate)...")
                sources_rot = [(ffn_rotate_path, "main", "infer_rotate"),
                               (prefill_rotate_path, "main", "prefill_rotate")]
                temp_rot_path = f"temp_{output_rot_path}"
                _save_multifunction_dedup(sources_rot, temp_rot_path, dedup_weights=dedup_weights)
                combined_rot = ct.models.MLModel(temp_rot_path)
                AddCombinedMetadata(combined_rot, [ffn_rotate_model, prefill_rotate_model])
                combined_rot.save(output_rot_path)
                shutil.rmtree(temp_rot_path, ignore_errors=True)
                print(f"  Rotate model saved: {output_rot_path}")

                print(f"Successfully created split-rotate chunk {chunk_idx+1}")

            except Exception as e:
                print(f"\nError processing chunk {chunk_idx+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False

        return True

    except Exception as e:
        print(f"\nError during split-rotate combination process: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def combine_chunks_gemma3(num_chunks, lut_bits=None, prefix='gemma3', dedup_weights=False):
    """Combine FFN, FFN_rotate, prefill, and prefill_rotate models into chunks with 4 functions.

    This is for Gemma3 models that require rotation support for positions >= sliding_window.
    Creates a combined model with 4 functions:
    - 'infer': Single token inference (fill mode, positions < sliding_window)
    - 'infer_rotate': Single token inference (rotation mode, positions >= sliding_window)
    - 'prefill': Batch prefill (fill mode, positions < sliding_window)
    - 'prefill_rotate': Batch prefill (rotation mode, positions >= sliding_window)

    Args:
        num_chunks: Number of chunks
        lut_bits: LUT quantization bits (or None)
        prefix: Model name prefix
        dedup_weights: If True, apply anemll-dedup weight dedup before combining

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Build file name templates
        if lut_bits:
            ffn_template = f"{prefix}_FFN_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            ffn_rotate_template = f"{prefix}_FFN_rotate_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            pf_template = f"{prefix}_prefill_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            pf_rotate_template = f"{prefix}_prefill_rotate_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            combined_template = f"{prefix}_FFN_PF_lut{lut_bits}_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
        else:
            ffn_template = f"{prefix}_FFN_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            ffn_rotate_template = f"{prefix}_FFN_rotate_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            pf_template = f"{prefix}_prefill_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            pf_rotate_template = f"{prefix}_prefill_rotate_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"
            combined_template = f"{prefix}_FFN_PF_chunk_{{:02d}}of{num_chunks:02d}.mlpackage"

        for chunk_idx in range(num_chunks):
            try:
                # Get input model paths
                ffn_path = ffn_template.format(chunk_idx + 1)
                ffn_rotate_path = ffn_rotate_template.format(chunk_idx + 1)
                prefill_path = pf_template.format(chunk_idx + 1)
                prefill_rotate_path = pf_rotate_template.format(chunk_idx + 1)
                output_path = combined_template.format(chunk_idx + 1)
                temp_path = f"temp_{output_path}"

                print(f"\nProcessing Gemma3 chunk {chunk_idx+1}:")
                print(f"  FFN: {ffn_path}")
                print(f"  FFN_rotate: {ffn_rotate_path}")
                print(f"  Prefill: {prefill_path}")
                print(f"  Prefill_rotate: {prefill_rotate_path}")
                print(f"  Output: {output_path}")

                # Check all input files exist
                missing = []
                for path in [ffn_path, ffn_rotate_path, prefill_path, prefill_rotate_path]:
                    if not os.path.exists(path):
                        missing.append(path)
                if missing:
                    print(f"Error: Missing input files:")
                    for f in missing:
                        print(f"  - {f}")
                    return False

                # Load models for metadata
                ffn_model = ct.models.MLModel(ffn_path)
                ffn_rotate_model = ct.models.MLModel(ffn_rotate_path)
                prefill_model = ct.models.MLModel(prefill_path)
                prefill_rotate_model = ct.models.MLModel(prefill_rotate_path)

                # Create combined model with 4 functions (with optional anemll-dedup dedup)
                print("Creating 4-function combined model...")
                sources = [
                    (ffn_path, "main", "infer"),
                    (ffn_rotate_path, "main", "infer_rotate"),
                    (prefill_path, "main", "prefill"),
                    (prefill_rotate_path, "main", "prefill_rotate"),
                ]
                _save_multifunction_dedup(sources, temp_path, dedup_weights=dedup_weights)

                # Load the temp model to add metadata
                print("Loading combined model...")
                combined_model = ct.models.MLModel(temp_path)
                if combined_model is None:
                    raise ValueError(f"Failed to load combined model")

                # Add metadata and save final
                print("Adding metadata...")
                AddCombinedMetadata(combined_model, [ffn_model, ffn_rotate_model, prefill_model, prefill_rotate_model])
                print(f"Saving final model to: {output_path}")
                combined_model.save(output_path)

                # Clean up temp file
                shutil.rmtree(temp_path, ignore_errors=True)

                print(f"Successfully combined Gemma3 chunk {chunk_idx+1} with 4 functions")

            except Exception as e:
                print(f"\nError processing chunk {chunk_idx+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False

        return True

    except Exception as e:
        print(f"\nError during Gemma3 combination process: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def parse_lut_arg(lut_value):
    """Parse LUT argument and extract just the bits value.

    Args:
        lut_value: String value from command line (e.g., '6' or '6,4')

    Returns:
        int or None: Just the lut_bits value (per_channel is ignored for file naming)
    """
    if lut_value is None:
        return None
    if isinstance(lut_value, str) and lut_value.lower() == "none":
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

def combine_monolithic(lut_bits=None, prefix='qwen', input_dir='.', output_dir='.', dedup_weights=False):
    """Combine monolithic infer and prefill models into single multi-function model.

    This creates a combined model with two functions:
    - 'infer': Single token inference
    - 'prefill': Batch token processing for initial sequence

    Args:
        lut_bits: LUT quantization bits (or None)
        prefix: Model name prefix
        input_dir: Directory containing the input models
        output_dir: Directory to save the combined model
        dedup_weights: If True, apply anemll-dedup weight dedup before combining

    Returns:
        bool: True if successful, False otherwise
    """
    # Build file names
    if lut_bits:
        infer_name = f"{prefix}_monolithic_lut{lut_bits}.mlpackage"
        prefill_name = f"{prefix}_monolithic_prefill_lut{lut_bits}.mlpackage"
        output_name = f"{prefix}_monolithic_full_lut{lut_bits}.mlpackage"
    else:
        infer_name = f"{prefix}_monolithic.mlpackage"
        prefill_name = f"{prefix}_monolithic_prefill.mlpackage"
        output_name = f"{prefix}_monolithic_full.mlpackage"

    infer_path = os.path.join(input_dir, infer_name)
    prefill_path = os.path.join(input_dir, prefill_name)
    output_path = os.path.join(output_dir, output_name)
    temp_path = os.path.join(output_dir, f"temp_{output_name}")

    print(f"\nCombining monolithic models:")
    print(f"  Infer:   {infer_path}")
    print(f"  Prefill: {prefill_path}")
    print(f"  Output:  {output_path}")
    if dedup_weights:
        print(f"  anemll-dedup:   enabled")

    # Check input files exist
    if not os.path.exists(infer_path):
        print(f"Error: Infer model not found: {infer_path}")
        return False
    if not os.path.exists(prefill_path):
        print(f"Error: Prefill model not found: {prefill_path}")
        return False

    try:
        # Load models for metadata
        print("Loading models...")
        infer_model = ct.models.MLModel(infer_path)
        prefill_model = ct.models.MLModel(prefill_path)

        # Create combined model (with optional anemll-dedup dedup)
        print("Creating multi-function model...")
        sources = [(infer_path, "main", "infer"), (prefill_path, "main", "prefill")]
        _save_multifunction_dedup(sources, temp_path, dedup_weights=dedup_weights)

        # Load and add metadata
        print("Adding metadata...")
        combined_model = ct.models.MLModel(temp_path)
        AddCombinedMetadata(combined_model, [infer_model, prefill_model])

        # Save final model
        print(f"Saving to: {output_path}")
        combined_model.save(output_path)

        # Clean up temp file
        shutil.rmtree(temp_path, ignore_errors=True)

        print("Monolithic model combination complete!")
        return True

    except Exception as e:
        print(f"Error combining monolithic models: {str(e)}")
        import traceback
        traceback.print_exc()
        # Clean up temp file on error
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path, ignore_errors=True)
        return False


def combine_monolithic_rotate(lut_bits=None, prefix='gemma3', input_dir='.', output_dir='.', dedup_weights=False):
    """Combine monolithic models with rotation support into 4-function model.

    This creates a combined model with four functions for context >= 512:
    - 'infer': Single token inference (positions < sliding_window)
    - 'infer_rotate': Single token inference (positions >= sliding_window)
    - 'prefill': Batch prefill (positions < sliding_window)
    - 'prefill_rotate': Batch prefill (positions >= sliding_window)

    Args:
        lut_bits: LUT quantization bits (or None)
        prefix: Model name prefix
        input_dir: Directory containing the input models
        output_dir: Directory to save the combined model
        dedup_weights: If True, apply anemll-dedup weight dedup before combining

    Returns:
        bool: True if successful, False otherwise
    """
    # Build file names
    if lut_bits:
        infer_name = f"{prefix}_monolithic_lut{lut_bits}.mlpackage"
        infer_rotate_name = f"{prefix}_monolithic_rotate_lut{lut_bits}.mlpackage"
        prefill_name = f"{prefix}_monolithic_prefill_lut{lut_bits}.mlpackage"
        prefill_rotate_name = f"{prefix}_monolithic_prefill_rotate_lut{lut_bits}.mlpackage"
        output_name = f"{prefix}_monolithic_full_lut{lut_bits}.mlpackage"
    else:
        infer_name = f"{prefix}_monolithic.mlpackage"
        infer_rotate_name = f"{prefix}_monolithic_rotate.mlpackage"
        prefill_name = f"{prefix}_monolithic_prefill.mlpackage"
        prefill_rotate_name = f"{prefix}_monolithic_prefill_rotate.mlpackage"
        output_name = f"{prefix}_monolithic_full.mlpackage"

    infer_path = os.path.join(input_dir, infer_name)
    infer_rotate_path = os.path.join(input_dir, infer_rotate_name)
    prefill_path = os.path.join(input_dir, prefill_name)
    prefill_rotate_path = os.path.join(input_dir, prefill_rotate_name)
    output_path = os.path.join(output_dir, output_name)
    temp_path = os.path.join(output_dir, f"temp_{output_name}")

    print(f"\nCombining monolithic models with rotation (4 functions):")
    print(f"  Infer:          {infer_path}")
    print(f"  Infer_rotate:   {infer_rotate_path}")
    print(f"  Prefill:        {prefill_path}")
    print(f"  Prefill_rotate: {prefill_rotate_path}")
    print(f"  Output:         {output_path}")
    if dedup_weights:
        print(f"  anemll-dedup:          enabled")

    # Check input files exist
    missing = []
    for path in [infer_path, infer_rotate_path, prefill_path, prefill_rotate_path]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print(f"Error: Missing input files:")
        for f in missing:
            print(f"  - {f}")
        return False

    try:
        # Load models for metadata
        print("Loading models...")
        infer_model = ct.models.MLModel(infer_path)
        infer_rotate_model = ct.models.MLModel(infer_rotate_path)
        prefill_model = ct.models.MLModel(prefill_path)
        prefill_rotate_model = ct.models.MLModel(prefill_rotate_path)

        # Create combined model with 4 functions (with optional anemll-dedup dedup)
        print("Creating 4-function multi-function model...")
        sources = [
            (infer_path, "main", "infer"),
            (infer_rotate_path, "main", "infer_rotate"),
            (prefill_path, "main", "prefill"),
            (prefill_rotate_path, "main", "prefill_rotate"),
        ]
        _save_multifunction_dedup(sources, temp_path, dedup_weights=dedup_weights)

        # Load and add metadata
        print("Adding metadata...")
        combined_model = ct.models.MLModel(temp_path)
        AddCombinedMetadata(combined_model, [infer_model, infer_rotate_model, prefill_model, prefill_rotate_model])

        # Save final model
        print(f"Saving to: {output_path}")
        combined_model.save(output_path)

        # Clean up temp file
        shutil.rmtree(temp_path, ignore_errors=True)

        print("Monolithic 4-function model combination complete!")
        return True

    except Exception as e:
        print(f"Error combining monolithic models with rotation: {str(e)}")
        import traceback
        traceback.print_exc()
        # Clean up temp file on error
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path, ignore_errors=True)
        return False


def parse_args():
    parser = argparse.ArgumentParser(description='Combine FFN and prefill models')
    parser.add_argument('--lut', type=str, help='LUT bits used in quantization (optional). Format: "bits" or "bits,per_channel" (e.g., "6" or "6,4")')
    parser.add_argument('--chunk', type=int, default=None,
                      help='Number of chunks (required for chunked mode)')
    parser.add_argument('--monolithic', action='store_true',
                      help='Combine monolithic infer and prefill models')
    parser.add_argument('--gemma3', action='store_true',
                      help='Combine Gemma3 models with 4 functions (infer, infer_rotate, prefill, prefill_rotate). '
                           'Required for Gemma3 models with context > 512 (sliding window)')
    parser.add_argument('--rotate', action='store_true',
                      help='Include rotation functions (infer_rotate, prefill_rotate) for context >= 512. '
                           'Creates 4-function model instead of 2-function model.')
    parser.add_argument('--split-rotate', action='store_true',
                      help='Split rotate functions into separate files (for large models). '
                           'Creates 2 files per chunk: non-rotate (infer+prefill) and rotate (infer_rotate+prefill_rotate).')
    parser.add_argument('--input', type=str, default='.',
                      help='Input directory containing model files (default: current directory)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output directory for combined models (default: same as input)')
    parser.add_argument('--prefix', type=str, default='llama',
                      help='Prefix for model names (default: llama)')
    parser.add_argument('--anemll-dedup', action='store_true', default=True,
                      dest='dedup_weights',
                      help='Enable anemll-dedup surgical weight dedup before combining (default: enabled)')
    parser.add_argument('--skip-anemll-dedup', dest='dedup_weights', action='store_false',
                      help='Skip anemll-dedup weight dedup (use standard combine)')
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
    dedup = getattr(args, 'dedup_weights', False)
    if combine_chunks(args.chunk, args.lut, None, args.prefix, dedup_weights=dedup):
        print("\nAll chunks combined successfully!")
        return True
    else:
        print("\nCombination process failed.")
        return False

def main():
    args = parse_args()

    # Parse LUT argument to extract just the bits value
    args.lut = parse_lut_arg(args.lut)

    try:
        dedup = args.dedup_weights
        if dedup:
            print("anemll-dedup weight dedup: enabled")
        else:
            print("anemll-dedup weight dedup: disabled")

        # Handle monolithic mode
        if args.monolithic:
            input_dir = args.input
            output_dir = args.output if args.output else args.input
            if args.rotate:
                # 4-function model with rotation support (context >= 512)
                success = combine_monolithic_rotate(
                    lut_bits=args.lut,
                    prefix=args.prefix,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    dedup_weights=dedup,
                )
            else:
                # 2-function model (context < 512)
                success = combine_monolithic(
                    lut_bits=args.lut,
                    prefix=args.prefix,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    dedup_weights=dedup,
                )
            sys.exit(0 if success else 1)

        # Chunked mode requires --chunk
        if args.chunk is None:
            print("Error: --chunk is required for chunked mode (or use --monolithic)")
            sys.exit(1)

        # Change to input directory for processing
        orig_dir = os.getcwd()
        os.chdir(args.input)

        # Handle split-rotate mode (2 files per chunk, 2 functions each)
        if args.split_rotate:
            print(f"\nCombining models with split-rotate (2 files per chunk)...")
            print(f"  Non-rotate file (_chunk_XXofYY): infer + prefill")
            print(f"  Rotate file (_chunk_XXofYY_rot): infer_rotate + prefill_rotate")
            success = combine_chunks_split_rotate(args.chunk, args.lut, args.prefix, dedup_weights=dedup)
        # Handle Gemma3 mode (4 functions in 1 file)
        elif args.gemma3:
            print(f"\nCombining Gemma3 models with 4 functions (infer, infer_rotate, prefill, prefill_rotate)...")
            success = combine_chunks_gemma3(args.chunk, args.lut, args.prefix, dedup_weights=dedup)
        else:
            # Run standard 2-function combination
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

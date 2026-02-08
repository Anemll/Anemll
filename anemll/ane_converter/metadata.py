#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

from enum import Enum
import os

# Use importlib.metadata (Python 3.8+) instead of deprecated pkg_resources
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Fallback for Python < 3.8 (though project requires 3.9+)
    from importlib_metadata import version, PackageNotFoundError

class ModelPart(Enum):
    EMBEDDINGS = "1"
    FFN = "2"
    PREFILL = "2_prefill"
    LM_HEAD = "3"
    FULL = "123"

def get_anemll_version():
    """Get Anemll version from PKG-INFO."""
    try:
        # First try to get version from installed package
        package_version = version('anemll')
        if package_version:
            return package_version
    except PackageNotFoundError:
        try:
            # If running as script, try to find PKG-INFO relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pkg_info_paths = [
                # Try development layout
                os.path.join(current_dir, '..', '..', 'PKG-INFO'),
                os.path.join(current_dir, '..', '..', 'anemll.egg-info', 'PKG-INFO'),
                # Try installed layout
                os.path.join(current_dir, '..', 'PKG-INFO'),
                os.path.join(current_dir, '..', 'anemll.egg-info', 'PKG-INFO'),
            ]
            
            for pkg_info_path in pkg_info_paths:
                if os.path.exists(pkg_info_path):
                    print(f"DEBUG: Found PKG-INFO at {pkg_info_path}")
                    with open(pkg_info_path, 'r') as f:
                        for line in f:
                            if line.startswith('Version:'):
                                return line.split(':')[1].strip()
                                
            print("DEBUG: No PKG-INFO found in paths:", pkg_info_paths)
            
        except Exception as e:
            print(f"DEBUG: Version lookup error: {str(e)}")
            
    return "0.1.1"  # Return default version if all else fails

def AddMetadata(model, params=None):
    """Add unified metadata to CoreML models.
    
    Args:
        model: CoreML model to add metadata to
        params: Dictionary containing metadata parameters:
            - version: Anemll version
            - context_length: Context length used
            - num_chunks: Total number of chunks
            - chunk_no: Current chunk number
            - batch_size: Batch size (for prefill)
            - function_names: List of function names (for combined models)
            - lut_bits: LUT quantization bits
            - split_part: Model part identifier (ModelPart enum value)
    """
    version = get_anemll_version()
    
    # Initialize user_defined_metadata if it doesn't exist
    if not hasattr(model, 'user_defined_metadata'):
        model.user_defined_metadata = {}
    
    # Initialize basic metadata
    if not hasattr(model, 'author'):
        model.author = ""
    if not hasattr(model, 'version'):
        model.version = ""
    if not hasattr(model, 'short_description'):
        model.short_description = ""
        
    # Set basic metadata
    model.author = f"Converted with Anemll v{version}"
    model.version = version
    
    # Add user-defined metadata
    model.user_defined_metadata["com.anemll.info"] = f"Converted with Anemll v{version}"
    
    if params:
        # Set short_description if provided
        if 'short_description' in params:
            model.short_description = params['short_description']
            print(f"DEBUG: Setting description to: {params['short_description']}")
        
        # Add CoreML-specific fields if present
        for key in ['com.github.apple.coremltools.source',
                   'com.github.apple.coremltools.source_dialect',
                   'com.github.apple.coremltools.version']:
            if key in params:
                model.user_defined_metadata[key] = params[key]
        
        if 'context_length' in params and params['context_length'] is not None:
            model.user_defined_metadata["com.anemll.context_length"] = str(params['context_length'])
            
        if 'num_chunks' in params and params['num_chunks'] is not None:
            model.user_defined_metadata["com.anemll.num_chunks"] = str(params['num_chunks'])
            
        if 'chunk_no' in params and params['chunk_no'] is not None:
            model.user_defined_metadata["com.anemll.chunk_no"] = str(params['chunk_no'])
            
        if 'batch_size' in params and params['batch_size'] is not None:
            model.user_defined_metadata["com.anemll.batch_size"] = str(params['batch_size'])
            
        if 'lut_bits' in params and params['lut_bits'] is not None:
            model.user_defined_metadata["com.anemll.lut_bits"] = str(params['lut_bits'])

        if 'argmax_in_model' in params and params['argmax_in_model']:
            model.user_defined_metadata["com.anemll.argmax_in_model"] = "true"

        if 'vocab_size' in params and params['vocab_size'] is not None:
            model.user_defined_metadata["com.anemll.vocab_size"] = str(params['vocab_size'])

        if 'lm_head_chunk_sizes' in params and params['lm_head_chunk_sizes'] is not None:
            chunk_sizes = params['lm_head_chunk_sizes']
            if isinstance(chunk_sizes, (list, tuple)):
                chunk_sizes = ",".join(str(int(x)) for x in chunk_sizes)
            model.user_defined_metadata["com.anemll.lm_head_chunk_sizes"] = str(chunk_sizes)
            
        if 'function_names' in params and params['function_names'] is not None:
            model.short_description = f"Combined model with functions: {', '.join(params['function_names'])}"
            model.user_defined_metadata["com.anemll.functions"] = ",".join(params['function_names'])
            
        if 'split_part' in params and params['split_part'] is not None:
            split_part = params['split_part']
            descriptions = {
                ModelPart.EMBEDDINGS.value: "Anemll Model (Embeddings) converted to CoreML",
                ModelPart.PREFILL.value: "Anemll Model (Prefill) converted to CoreML",
                ModelPart.LM_HEAD.value: "Anemll Model (LM Head) converted to CoreML",
                ModelPart.FFN.value: "Anemll Model (FFN) converted to CoreML",
                ModelPart.FULL.value: "Anemll Model (Full) converted to CoreML"
            }
            model.short_description = descriptions.get(split_part, f"Anemll Model Part {split_part} converted to CoreML")

def ReadMetadata(model):
    """Read metadata from a CoreML model.
    
    Args:
        model: CoreML model to read metadata from
        
    Returns:
        dict: Dictionary containing metadata parameters
    """
    metadata = {}
    
    # Initialize model metadata if needed
    if not hasattr(model, 'user_defined_metadata'):
        model.user_defined_metadata = {}
    if not hasattr(model, 'author'):
        model.author = ""
    if not hasattr(model, 'version'):
        model.version = ""
    if not hasattr(model, 'short_description'):
        model.short_description = ""
    
    # Read standard metadata
    metadata['author'] = model.author
    metadata['version'] = model.version
    metadata['short_description'] = model.short_description
        
    # Read user-defined metadata
    for key, value in model.user_defined_metadata.items():
        if key.startswith('com.anemll.'):
            # Strip prefix and store
            clean_key = key.replace('com.anemll.', '')
            metadata[clean_key] = value
        # Add CoreML-specific fields
        elif key in ['com.github.apple.coremltools.source',
                    'com.github.apple.coremltools.source_dialect',
                    'com.github.apple.coremltools.version']:
            metadata[key] = value
                
    return metadata

def CombineMetadata(models):
    """Combine metadata from multiple models.
    
    Args:
        models: List of CoreML models
        
    Returns:
        dict: Combined metadata parameters
    """
    combined = {}
    
    for model in models:
        metadata = ReadMetadata(model)
        
        # Combine each field, preferring non-None values
        for key, value in metadata.items():
            if value is not None and (key not in combined or combined[key] is None):
                combined[key] = value
                
        # Special handling for version - use latest
        if 'version' in metadata and metadata['version'] is not None:
            if 'version' not in combined or metadata['version'] > combined['version']:
                combined['version'] = metadata['version']
                
        # Special handling for functions - merge lists
        if 'functions' in metadata and metadata['functions'] is not None:
            if 'functions' not in combined:
                combined['functions'] = []
            combined['functions'].extend(metadata['functions'].split(','))
            combined['functions'] = list(set(combined['functions']))  # Remove duplicates
    
    return combined

def AddCombinedMetadata(target_model, source_models):
    """Read metadata from source models, combine it, and add to target model."""
    print("\nDEBUG: Starting AddCombinedMetadata...")
    try:
        combined = {}
        
        print("DEBUG: Processing source models...")
        descriptions = []
        for i, model in enumerate(source_models):
            print(f"DEBUG: Reading metadata from model {i+1}")
            metadata = ReadMetadata(model)
            print(f"DEBUG: Got metadata: {metadata}")
            
            # Save descriptions for combining
            if 'short_description' in metadata:
                descriptions.append(metadata['short_description'])
            
            # Combine each field, preferring non-None values
            for key, value in metadata.items():
                if key != 'short_description' and value is not None and (key not in combined or combined[key] is None):
                    combined[key] = value
        
        # Create combined description
        if descriptions:
            # Extract model types from descriptions
            model_types = []
            for desc in descriptions:
                if "(" in desc and ")" in desc:
                    model_type = desc[desc.find("(")+1:desc.find(")")]
                    if model_type:
                        model_types.append(model_type)
            
            if model_types:
                combined['short_description'] = f"Anemll Model: Multifunction {'+'.join(model_types)}"
            else:
                combined['short_description'] = "Anemll Model: Multifunction Combined"
        
        print(f"\nDEBUG: Combined metadata: {combined}")
        print("DEBUG: Adding metadata to target model...")
        AddMetadata(target_model, combined)
        print("DEBUG: Metadata added successfully")
        
    except Exception as e:
        print(f"\nError in AddCombinedMetadata: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        raise

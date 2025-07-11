#!/usr/bin/env python3
"""Test script to verify CoreML model compilation and ANE loading."""

import coremltools as ct
from pathlib import Path
import subprocess
import shutil

def compile_and_load_model(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_NE):
    """Compile and load a CoreML model, ensuring it runs on ANE."""
    mlpackage_path = Path(mlpackage_path)
    
    if not mlpackage_path.exists():
        print(f"Error: Model not found at {mlpackage_path}")
        return None
    
    # Compile the model to .mlmodelc
    mlmodelc_path = mlpackage_path.with_suffix('.mlmodelc')
    
    # Delete existing compiled model if it exists
    if mlmodelc_path.exists():
        shutil.rmtree(mlmodelc_path)
        print(f"Deleted existing compiled model: {mlmodelc_path}")
    
    # Compile the model
    print(f"Compiling {mlpackage_path.name} to .mlmodelc...")
    try:
        result = subprocess.run(
            ['xcrun', 'coremlcompiler', 'compile', str(mlpackage_path), str(mlpackage_path.parent)],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Model compiled successfully to: {mlmodelc_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling model: {e.stderr}")
        return None
    
    # Load the compiled model
    try:
        # Use CompiledMLModel for .mlmodelc files with specified compute units
        model = ct.models.CompiledMLModel(str(mlmodelc_path), compute_units)
        print(f"Model loaded with compute units: {compute_units}")
        
        # Check which compute unit is actually being used
        print(f"Model is configured for: {compute_units}")
        
        return model
    except Exception as e:
        print(f"Error loading compiled model: {e}")
        return None

# Test with an existing model
test_path = Path("/Volumes/Models/ANE/conv2d_qwen_rmsnorm_10_layers.mlpackage")
if test_path.exists():
    print(f"\nTesting with: {test_path}")
    model = compile_and_load_model(test_path)
    if model:
        print("\nModel loaded successfully!")
        print("You can now use model.predict() with this model")
else:
    print(f"Test model not found at: {test_path}")
    print("Please run the main test first to generate models")
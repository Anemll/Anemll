#!/usr/bin/env python3
"""Inspect the converted CoreML model to understand its input/output specifications."""

import coremltools as ct
from pathlib import Path

def inspect_coreml_model():
    """Inspect the CoreML model specification."""
    
    coreml_path = "../qwen-test/qwen.mlpackage"
    if not Path(coreml_path).exists():
        print(f"❌ Error: CoreML model not found at {coreml_path}")
        return False
    
    print(f"📊 Inspecting CoreML model: {coreml_path}")
    
    try:
        model = ct.models.MLModel(coreml_path)
        spec = model.get_spec()
        
        print("\n" + "="*60)
        print("📥 MODEL INPUTS:")
        print("="*60)
        
        for input_desc in spec.description.input:
            name = input_desc.name
            input_type = input_desc.type
            
            print(f"\nInput: {name}")
            
            if input_type.HasField('multiArrayType'):
                array_type = input_type.multiArrayType
                shape = []
                for dim in array_type.shape:
                    if hasattr(dim, 'sizeRange'):
                        if dim.sizeRange.lowerBound == dim.sizeRange.upperBound:
                            shape.append(str(dim.sizeRange.lowerBound))
                        else:
                            shape.append(f"{dim.sizeRange.lowerBound}-{dim.sizeRange.upperBound}")
                    else:
                        shape.append(str(dim))
                print(f"  Type: MultiArray")
                print(f"  Shape: {shape}")
                print(f"  DataType: {array_type.dataType}")
            else:
                print(f"  Type: {input_type}")
        
        print("\n" + "="*60)
        print("📤 MODEL OUTPUTS:")
        print("="*60)
        
        for output_desc in spec.description.output:
            name = output_desc.name
            output_type = output_desc.type
            
            print(f"\nOutput: {name}")
            
            if output_type.HasField('multiArrayType'):
                array_type = output_type.multiArrayType
                shape = []
                for dim in array_type.shape:
                    if hasattr(dim, 'sizeRange'):
                        if dim.sizeRange.lowerBound == dim.sizeRange.upperBound:
                            shape.append(str(dim.sizeRange.lowerBound))
                        else:
                            shape.append(f"{dim.sizeRange.lowerBound}-{dim.sizeRange.upperBound}")
                    else:
                        shape.append(str(dim))
                print(f"  Type: MultiArray")
                print(f"  Shape: {shape}")
                print(f"  DataType: {array_type.dataType}")
            else:
                print(f"  Type: {output_type}")
        
        print("\n" + "="*60)
        print("🔍 MODEL METADATA:")
        print("="*60)
        
        metadata = model.user_defined_metadata
        if metadata:
            for key, value in metadata.items():
                print(f"{key}: {value}")
        else:
            print("No user-defined metadata found")
        
        print(f"\nModel description: {spec.description.metadata.shortDescription}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")
        return False

if __name__ == "__main__":
    inspect_coreml_model() 
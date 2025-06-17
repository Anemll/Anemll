#!/usr/bin/env python3
"""
Dequantize a quantized model to FP16/BF16 format.
This script attempts to load quantized models and save them in unquantized format.
"""

import argparse
import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def dequantize_model(input_path, output_path, dtype='float16'):
    """Dequantize a model and save it in FP16/BF16 format."""
    
    print(f"Loading model from: {input_path}")
    
    # Check if it's a quantized model
    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        quant_config = config.get('quantization_config', {})
        if quant_config:
            print(f"Detected quantization: {quant_config.get('quant_method', 'unknown')}")
            
            if quant_config.get('quant_method') == 'fp8':
                print("\nWARNING: FP8 models require GPU for proper dequantization.")
                print("This script will attempt CPU-based conversion which may not be accurate.")
                print("For best results, use a GPU or download the original unquantized model.\n")
    
    # Set torch dtype
    if dtype == 'float16':
        torch_dtype = torch.float16
    elif dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    try:
        # Try to load the model
        print("Loading model (this may take a while)...")
        
        # For quantized models, we need special handling
        config = AutoConfig.from_pretrained(input_path)
        
        # Remove quantization config
        if hasattr(config, 'quantization_config'):
            print("Removing quantization config...")
            delattr(config, 'quantization_config')
        
        # Try to load model with ignore_mismatched_sizes for robustness
        model = AutoModelForCausalLM.from_pretrained(
            input_path,
            config=config,
            torch_dtype=torch_dtype,
            device_map='cpu',
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        
        print(f"Model loaded successfully. Converting to {dtype}...")
        
        # Ensure all parameters are in the target dtype
        for param in model.parameters():
            param.data = param.data.to(torch_dtype)
        
        # Save the model
        print(f"Saving dequantized model to: {output_path}")
        model.save_pretrained(output_path, torch_dtype=torch_dtype)
        
        # Also copy tokenizer files
        print("Copying tokenizer files...")
        tokenizer = AutoTokenizer.from_pretrained(input_path)
        tokenizer.save_pretrained(output_path)
        
        print("\n✅ Model successfully dequantized and saved!")
        print(f"Output directory: {output_path}")
        print(f"Format: {dtype}")
        
    except Exception as e:
        print(f"\n❌ Error during dequantization: {e}")
        print("\nTroubleshooting tips:")
        print("1. For FP8 models, a GPU may be required for proper conversion")
        print("2. Some quantization formats may not be supported")
        print("3. Try downloading the original unquantized model instead")
        print("\nAlternative approach in Python:")
        print("```python")
        print("from transformers import AutoModelForCausalLM")
        print(f"model = AutoModelForCausalLM.from_pretrained('{input_path}')")
        print(f"model.save_pretrained('{output_path}', torch_dtype=torch.{dtype})")
        print("```")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Dequantize a model to FP16/BF16 format')
    parser.add_argument('--input', '-i', required=True, help='Input model directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory for dequantized model')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], 
                        default='float16', help='Output dtype (default: float16)')
    
    args = parser.parse_args()
    
    # Expand paths
    input_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)
    
    # Check input exists
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Check if output already exists
    if os.path.exists(output_path):
        response = input(f"Output path already exists: {output_path}\nOverwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Dequantize
    success = dequantize_model(input_path, output_path, args.dtype)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
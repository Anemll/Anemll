#!/usr/bin/env python3
"""Utilities for loading FP8 models and converting to FP16."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors import safe_open
import glob
import os

def load_fp8_model_as_fp16(model_dir):
    """Load an FP8 model and convert it to FP16.
    
    FP8 models store weights in uint8 format with scale factors.
    We'll dequantize them to FP16 for CPU inference.
    """
    print(f"Loading FP8 model from {model_dir}")
    
    # Load config
    config = AutoConfig.from_pretrained(model_dir)
    
    # Remove quantization config to create unquantized model
    if hasattr(config, 'quantization_config'):
        delattr(config, 'quantization_config')
    
    # Create model architecture
    model = AutoModelForCausalLM.from_config(config)
    
    # Find all safetensor files
    safetensor_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    
    # Load all weights and scales
    weights = {}
    scales = {}
    
    for safetensor_file in sorted(safetensor_files):
        with safe_open(safetensor_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key.endswith('_scale_inv'):
                    # This is a scale factor
                    base_key = key[:-10]  # Remove '_scale_inv'
                    scales[base_key] = tensor
                else:
                    weights[key] = tensor
    
    print(f"Loaded {len(weights)} weights and {len(scales)} scale factors")
    
    # Dequantize FP8 weights
    dequantized_weights = {}
    
    for key, weight in weights.items():
        if key in scales:
            # This is an FP8 quantized weight
            scale_inv = scales[key]
            
            # FP8 weights are stored as uint8, we need to:
            # 1. Convert to float
            # 2. Apply scale
            # 3. Convert to FP16
            
            # FP8 dequantization
            if weight.dtype == torch.uint8:
                # FP8 weights are stored as uint8
                # The scale_inv is the reciprocal of the scale factor
                
                # Convert to float32 for dequantization
                float_weight = weight.to(torch.float32)
                scale_inv_f32 = scale_inv.to(torch.float32)
                
                # FP8 E4M3 format: sign bit + 4 exponent bits + 3 mantissa bits
                # The weights are already in a format that can be multiplied by scale
                # scale_inv is 1/scale, so we divide by scale_inv (multiply by scale)
                dequantized = float_weight / scale_inv_f32
                
                # Ensure the shape is correct by expanding scale if needed
                if scale_inv_f32.numel() == 1:
                    # Scalar scale
                    pass
                elif scale_inv_f32.shape != weight.shape:
                    # Per-channel or per-tensor scale
                    # Reshape scale to match weight dimensions
                    if len(weight.shape) == 2:
                        # Linear layer weight [out_features, in_features]
                        if scale_inv_f32.shape[0] == weight.shape[0]:
                            scale_inv_f32 = scale_inv_f32.view(-1, 1)
                    dequantized = float_weight / scale_inv_f32
                
                # Convert to FP16
                dequantized_weights[key] = dequantized.to(torch.float16)
            else:
                # Not quantized, just ensure it's FP16
                dequantized_weights[key] = weight.to(torch.float16)
        else:
            # No scale, not quantized
            dequantized_weights[key] = weight.to(torch.float16)
    
    # Load the dequantized weights
    model.load_state_dict(dequantized_weights)
    model = model.to(torch.float16)
    
    print("Successfully loaded and dequantized FP8 model to FP16")
    return model

def test_fp8_loading():
    """Test loading an FP8 model."""
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "~/Models/HF/qwen3_4b_fp8"
    
    model_path = os.path.expanduser(model_path)
    
    try:
        model = load_fp8_model_as_fp16(model_path)
        print(f"Model loaded successfully: {type(model)}")
        
        # Test a simple forward pass
        test_input = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            output = model(test_input)
            print(f"Test forward pass successful, output shape: {output.logits.shape}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fp8_loading()
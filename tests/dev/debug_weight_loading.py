#!/usr/bin/env python3

import os
import torch
import sys

# Set environment variable BEFORE importing the module
os.environ['ENABLE_SP_QUANT'] = '1'

# Add project root to path
sys.path.append('/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from transformers import AutoModel
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def debug_weight_loading():
    print("Debugging weight loading...")
    
    # Load transformers model to see what weights exist
    model_path = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    transformers_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)
    
    print("=== TRANSFORMERS MODEL WEIGHTS ===")
    for name, param in transformers_model.named_parameters():
        if 'layernorm' in name or 'norm' in name:
            print(f"{name}: {param.shape}, first 3 values: {param.data.flatten()[:3]}")
    
    # Create ANEMLL model
    config = Qwen25Config()
    anemll_model = Qwen25ForCausalLM(config)
    
    print("\n=== ANEMLL MODEL WEIGHTS BEFORE LOADING ===")
    for name, param in anemll_model.named_parameters():
        if 'layernorm' in name or 'norm' in name:
            print(f"{name}: {param.shape}, first 3 values: {param.data.flatten()[:3]}")
    
    print("\n=== LOADING WEIGHTS ===")
    anemll_model.load_pretrained_weights(model_path)
    
    print("\n=== ANEMLL MODEL WEIGHTS AFTER LOADING ===")
    for name, param in anemll_model.named_parameters():
        if 'layernorm' in name or 'norm' in name:
            print(f"{name}: {param.shape}, first 3 values: {param.data.flatten()[:3]}")
    
    print("\n=== COMPARISON ===")
    # Check if specific weights match
    t_weight = transformers_model.layers[0].input_layernorm.weight
    a_weight = anemll_model.model.layers[0].input_layernorm.weight
    
    print(f"Layer 0 input_layernorm comparison:")
    print(f"Transformers: {t_weight[:5]}")
    print(f"ANEMLL: {a_weight[:5]}")
    print(f"Match: {torch.allclose(t_weight, a_weight, atol=1e-5)}")
    print(f"Dtype - Transformers: {t_weight.dtype}, ANEMLL: {a_weight.dtype}")

if __name__ == "__main__":
    debug_weight_loading()
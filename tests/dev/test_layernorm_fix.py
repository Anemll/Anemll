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

def test_layernorm_fix():
    print("Testing LayerNorm weight loading fix...")
    
    # Load transformers model to get weights
    model_path = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor" 
    transformers_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)
    
    # Get transformers state dict
    transformers_state_dict = transformers_model.state_dict()
    
    # Create ANEMLL model
    config = Qwen25Config()
    anemll_model = Qwen25ForCausalLM(config)
    
    print("=== BEFORE MANUAL LOADING ===")
    print(f"ANEMLL input_layernorm weight[:3]: {anemll_model.model.layers[0].input_layernorm.weight[:3]}")
    
    # Manually load just the LayerNorm weights to test the key mapping
    print("\n=== SIMULATING WEIGHT LOADING LOGIC ===")
    conv_state = {}
    
    # Simulate the weight loading logic from load_pretrained_weights
    for k, v in transformers_state_dict.items():
        # Fix: Add "model." prefix for LayerNorm weights to match ANEMLL parameter names
        if 'layernorm' in k or k.endswith('.norm.weight'):
            if not k.startswith('model.'):
                new_k = f"model.{k}"
            else:
                new_k = k
            print(f"Processing: {k} -> {new_k}")
            conv_state[new_k] = v.to(torch.float16)  # MODEL_DTYPE
    
    print(f"\nFound {len(conv_state)} LayerNorm weights")
    
    # Load the weights into the model
    missing_keys, unexpected_keys = anemll_model.load_state_dict(conv_state, strict=False)
    
    print(f"Missing keys: {len(missing_keys)} (first 5: {missing_keys[:5]})")
    print(f"Unexpected keys: {len(unexpected_keys)} (first 5: {unexpected_keys[:5]})")
    
    print("\n=== AFTER MANUAL LOADING ===")
    print(f"ANEMLL input_layernorm weight[:3]: {anemll_model.model.layers[0].input_layernorm.weight[:3]}")
    print(f"Transformers input_layernorm weight[:3]: {transformers_model.layers[0].input_layernorm.weight[:3]}")
    
    # Test if they match now
    match = torch.allclose(
        anemll_model.model.layers[0].input_layernorm.weight, 
        transformers_model.layers[0].input_layernorm.weight, 
        atol=1e-5
    )
    print(f"LayerNorm weights match: {match}")
    
    if match:
        print("🎉 SUCCESS: LayerNorm weight loading is fixed!")
    else:
        print("❌ FAIL: LayerNorm weights still don't match")

if __name__ == "__main__":
    test_layernorm_fix()
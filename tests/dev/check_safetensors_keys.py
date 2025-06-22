#!/usr/bin/env python3

import os
import torch
import sys

# Set environment variable BEFORE importing the module
os.environ['ENABLE_SP_QUANT'] = '1'

# Add project root to path
sys.path.append('/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from transformers import AutoModel
import safetensors.torch
from transformers.utils.hub import cached_file

def check_safetensors_keys():
    print("Checking safetensors keys...")
    
    model_path = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    
    try:
        safetensors_path = cached_file(model_path, "model.safetensors")
        state_dict = safetensors.torch.load_file(safetensors_path)
        
        print("LayerNorm keys in safetensors:")
        layernorm_keys = [k for k in state_dict.keys() if 'layernorm' in k or k.endswith('.norm.weight')]
        for key in layernorm_keys[:5]:  # First 5
            print(f"  {key}")
            
        print(f"\nTotal LayerNorm keys: {len(layernorm_keys)}")
        
    except Exception as e:
        print(f"Could not load safetensors: {e}")
        
        # Fallback to transformers model
        transformers_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)
        layernorm_keys = [k for k, v in transformers_model.named_parameters() if 'layernorm' in k or 'norm' in k]
        print("LayerNorm keys in transformers model:")
        for key in layernorm_keys[:5]:
            print(f"  {key}")

if __name__ == "__main__":
    check_safetensors_keys()
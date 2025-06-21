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

def debug_parameter_names():
    print("Debugging parameter names...")
    
    # Load transformers model
    model_path = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    transformers_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)
    
    # Create ANEMLL model
    config = Qwen25Config()
    anemll_model = Qwen25ForCausalLM(config)
    
    print("=== TRANSFORMERS SAFETENSORS KEY NAMES ===")
    import safetensors.torch
    from transformers.utils.hub import cached_file
    
    # Get the safetensors file
    try:
        safetensors_path = cached_file(model_path, "model.safetensors")
        state_dict = safetensors.torch.load_file(safetensors_path)
        
        layernorm_keys = [k for k in state_dict.keys() if 'layernorm' in k or 'norm' in k]
        print("LayerNorm keys in safetensors:")
        for key in layernorm_keys[:10]:  # First 10
            print(f"  {key}")
        
    except Exception as e:
        print(f"Could not load safetensors: {e}")
        print("Using transformers state_dict instead:")
        layernorm_keys = [k for k, v in transformers_model.named_parameters() if 'layernorm' in k or 'norm' in k]
        for key in layernorm_keys[:10]:
            print(f"  {key}")
    
    print("\n=== ANEMLL PARAMETER NAMES ===")
    anemll_layernorm_keys = [k for k, v in anemll_model.named_parameters() if 'layernorm' in k or 'norm' in k]
    print("LayerNorm keys in ANEMLL:")
    for key in anemll_layernorm_keys[:10]:  # First 10
        print(f"  {key}")
    
    print("\n=== KEY MAPPING ANALYSIS ===")
    if 'layernorm_keys' in locals():
        print("Expected transformers -> ANEMLL mapping:")
        for t_key in layernorm_keys[:5]:
            # Simulate the key transformation in load_pretrained_weights
            new_k = t_key.replace("model.", "") if t_key.startswith("model.") else t_key
            print(f"  {t_key} -> {new_k}")
            
            # Check if this matches any ANEMLL key
            matches = [ak for ak in anemll_layernorm_keys if ak == new_k]
            if matches:
                print(f"    ✅ MATCHES: {matches[0]}")
            else:
                print(f"    ❌ NO MATCH in ANEMLL")

if __name__ == "__main__":
    debug_parameter_names()
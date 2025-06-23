#!/usr/bin/env python3
"""
Test that bias scaling is working correctly during model loading.
"""

import os
os.environ['ENABLE_SP_QUANT'] = '1'

import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from safetensors import safe_open
from huggingface_hub import snapshot_download

def check_bias_scaling():
    """Check bias values before and after scaling."""
    print("="*80)
    print("CHECKING BIAS SCALING")
    print("="*80)
    
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    model_path = snapshot_download(model_name)
    
    # Load the raw safetensors file to see original bias values
    import glob
    safetensor_files = glob.glob(f"{model_path}/*.safetensors")
    
    print(f"\nChecking raw bias values from: {safetensor_files[0]}")
    
    with safe_open(safetensor_files[0], framework="pt", device="cpu") as f:
        # Check a specific problematic bias
        key = "model.layers.0.self_attn.k_proj.bias"
        if key in f.keys():
            raw_bias = f.get_tensor(key)
            print(f"\n{key}:")
            print(f"  Raw bias max: {raw_bias.max().item():.6f}")
            print(f"  Raw bias min: {raw_bias.min().item():.6f}")
            print(f"  Raw bias first 5: {raw_bias.flatten()[:5].tolist()}")
            
            # Get the corresponding scale
            scale_key = "model.layers.0.self_attn.k_proj.output_scales"
            if scale_key in f.keys():
                scale = f.get_tensor(scale_key)
                print(f"\n  Output scale shape: {scale.shape}")
                print(f"  Output scale first 5: {scale.flatten()[:5].tolist()}")
                
                # Show what the scaled bias should be
                if scale.shape[-1] == 1 and scale.shape[0] == raw_bias.shape[0]:
                    scale = scale.squeeze(-1)
                scaled_bias = raw_bias / scale
                print(f"\n  Scaled bias max: {scaled_bias.max().item():.6f}")
                print(f"  Scaled bias min: {scaled_bias.min().item():.6f}")
                print(f"  Scaled bias first 5: {scaled_bias.flatten()[:5].tolist()}")

    # Now load the model and check what's actually in it
    print("\n" + "="*60)
    print("Now loading model to check actual bias values...")
    
    from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
    
    config = Qwen25Config.from_json(f'{model_path}/config.json')
    model = Qwen25ForCausalLM(config, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    
    # Check the same bias in the loaded model
    loaded_bias = model.model.layers[0].self_attn.k_proj.bias
    print(f"\nLoaded model bias:")
    print(f"  Max: {loaded_bias.max().item():.6f}")
    print(f"  Min: {loaded_bias.min().item():.6f}")
    print(f"  First 5: {loaded_bias.flatten()[:5].tolist()}")
    
    # Check if it matches the scaled version
    print("\n" + "="*60)
    print("SUMMARY:")
    if abs(loaded_bias.max().item()) > 50:
        print("❌ BIAS NOT PROPERLY SCALED! Still has large values.")
    else:
        print("✅ Bias appears to be properly scaled.")

if __name__ == "__main__":
    check_bias_scaling()
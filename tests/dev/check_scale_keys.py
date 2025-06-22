#!/usr/bin/env python3
"""
Check the actual keys in the safetensors file to see scale naming.
"""

from safetensors import safe_open
from huggingface_hub import snapshot_download

def check_scale_keys():
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    model_path = snapshot_download(model_name)
    
    print("Checking actual keys in safetensors file...")
    
    with safe_open(f"{model_path}/model.safetensors", framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
    
    # Filter for scale keys
    scale_keys = [k for k in all_keys if 'scale' in k.lower()]
    print(f"\nFound {len(scale_keys)} scale-related keys:")
    
    for key in scale_keys[:10]:  # Show first 10
        print(f"  {key}")
    
    # Show pattern
    q_proj_keys = [k for k in all_keys if 'q_proj' in k]
    print(f"\nQ projection related keys:")
    for key in q_proj_keys[:5]:
        print(f"  {key}")

if __name__ == "__main__":
    check_scale_keys()
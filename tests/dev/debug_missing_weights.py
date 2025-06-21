#!/usr/bin/env python3

import os
import torch
import sys

# Set environment variable BEFORE importing the module
os.environ['ENABLE_SP_QUANT'] = '1'

# Add project root to path
sys.path.append('/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def debug_missing_weights():
    print("Debugging missing weights in ANEMLL model...")
    
    model_id = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    
    # Download model if needed
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading model {model_id}...")
        model_path = snapshot_download(repo_id=model_id)
        print(f"Model downloaded to: {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        return
    
    # Create ANEMLL model
    config = Qwen25Config()
    anemll_model = Qwen25ForCausalLM(config)
    
    print("\n=== BEFORE WEIGHT LOADING ===")
    print("ANEMLL parameter names (first 20):")
    param_names = list(anemll_model.named_parameters())
    for name, param in param_names[:20]:
        print(f"  {name}: {param.shape}")
    
    print(f"\nTotal ANEMLL parameters: {len(param_names)}")
    
    # Load weights and capture output
    print("\n=== LOADING WEIGHTS ===")
    try:
        result = anemll_model.load_pretrained_weights(model_path)
        print(f"\nWeight loading result: {result}")
        print("Done.")
    except RuntimeError as e:
        print(f"\nRuntimeError caught: {e}")
        print("This confirms the hard stop is working!")
        print("Done.")

if __name__ == "__main__":
    debug_missing_weights()
#!/usr/bin/env python3

import os
import torch
import sys

# Set environment variable BEFORE importing the module
os.environ['ENABLE_SP_QUANT'] = '1'

# Add project root to path
sys.path.append('/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from transformers import AutoModel, AutoTokenizer
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def test_eps_values():
    print("Testing RMSNorm eps values...")
    
    # Load transformers model
    model_path = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    transformers_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)
    
    # Load ANEMLL model
    config = Qwen25Config()
    anemll_model = Qwen25ForCausalLM(config)
    
    print(f"Transformers config rms_norm_eps: {transformers_model.config.rms_norm_eps}")
    print(f"ANEMLL config rms_norm_eps: {config.rms_norm_eps}")
    
    # Check actual RMSNorm layer eps values
    transformers_eps = transformers_model.layers[0].input_layernorm.variance_epsilon
    anemll_eps = anemll_model.model.layers[0].input_layernorm.eps
    
    print(f"Transformers RMSNorm eps: {transformers_eps}")
    print(f"ANEMLL RMSNorm eps: {anemll_eps}")
    
    print(f"Values match: {transformers_eps == anemll_eps}")

if __name__ == "__main__":
    test_eps_values()
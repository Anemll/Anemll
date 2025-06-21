#!/usr/bin/env python3

import os
import torch
import torch.nn.functional as F
import sys

# Set environment variable BEFORE importing the module
os.environ['ENABLE_SP_QUANT'] = '1'

# Add project root to path
sys.path.append('/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from transformers import AutoModel, AutoTokenizer
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def test_rmsnorm_implementations():
    print("Testing RMSNorm implementations in detail...")
    
    # Load transformers model
    model_path = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    transformers_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)
    
    # Load ANEMLL model
    config = Qwen25Config()
    anemll_model = Qwen25ForCausalLM(config)
    
    # Get the first layer norm from each model
    transformers_norm = transformers_model.layers[0].input_layernorm
    anemll_norm = anemll_model.model.layers[0].input_layernorm
    
    # Create test input
    test_input = torch.randn(1, 5, 896, dtype=torch.float32)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input mean: {test_input.mean():.6f}")
    print(f"Test input std: {test_input.std():.6f}")
    
    # Get weights and eps
    print(f"\nWeight shapes:")
    print(f"Transformers weight shape: {transformers_norm.weight.shape}")
    print(f"ANEMLL weight shape: {anemll_norm.weight.shape}")
    print(f"Weights equal: {torch.allclose(transformers_norm.weight, anemll_norm.weight, atol=1e-6)}")
    
    print(f"\nEps values:")
    print(f"Transformers eps: {transformers_norm.variance_epsilon}")
    print(f"ANEMLL eps: {anemll_norm.eps}")
    
    # Manual transformers RMSNorm implementation
    variance = test_input.pow(2).mean(-1, keepdim=True)
    manual_transformers_output = test_input * torch.rsqrt(variance + transformers_norm.variance_epsilon)
    manual_transformers_output = transformers_norm.weight * manual_transformers_output
    
    # ANEMLL implementation (current - with commented mean subtraction)
    anemll_output_current = F.layer_norm(test_input, anemll_norm.weight.shape, anemll_norm.weight, bias=None, eps=float(anemll_norm.eps))
    
    # ANEMLL implementation (with mean subtraction)
    mean = test_input.mean(-1, keepdim=True)
    test_input_minus_mean = test_input - mean
    anemll_output_with_mean = F.layer_norm(test_input_minus_mean, anemll_norm.weight.shape, anemll_norm.weight, bias=None, eps=float(anemll_norm.eps))
    
    # Actual transformers output
    transformers_output = transformers_norm(test_input)
    
    print(f"\nOutput comparisons:")
    print(f"Manual transformers vs actual transformers: {torch.allclose(manual_transformers_output, transformers_output, atol=1e-5)}")
    print(f"ANEMLL current vs transformers: {torch.allclose(anemll_output_current, transformers_output, atol=1e-5)}")
    print(f"ANEMLL with mean vs transformers: {torch.allclose(anemll_output_with_mean, transformers_output, atol=1e-5)}")
    
    print(f"\nOutput means:")
    print(f"Transformers: {transformers_output.mean():.6f}")
    print(f"ANEMLL current: {anemll_output_current.mean():.6f}")
    print(f"ANEMLL with mean: {anemll_output_with_mean.mean():.6f}")
    print(f"Manual transformers: {manual_transformers_output.mean():.6f}")
    
    print(f"\nOutput differences (first few values):")
    print(f"Transformers - ANEMLL current: {(transformers_output - anemll_output_current).flatten()[:5]}")
    print(f"Transformers - ANEMLL with mean: {(transformers_output - anemll_output_with_mean).flatten()[:5]}")
    
    # Test what F.layer_norm actually does
    print(f"\nF.layer_norm behavior test:")
    simple_input = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
    simple_weight = torch.ones(3)
    simple_eps = 1e-6
    
    # Standard layer norm (subtracts mean, divides by std)
    layernorm_output = F.layer_norm(simple_input, simple_weight.shape, simple_weight, bias=None, eps=simple_eps)
    
    # Manual layer norm
    mean = simple_input.mean(-1, keepdim=True)
    variance = ((simple_input - mean) ** 2).mean(-1, keepdim=True)
    manual_layernorm = (simple_input - mean) / torch.sqrt(variance + simple_eps)
    
    print(f"Simple input: {simple_input}")
    print(f"F.layer_norm output: {layernorm_output}")
    print(f"Manual layer norm: {manual_layernorm}")
    print(f"Match: {torch.allclose(layernorm_output, manual_layernorm, atol=1e-6)}")

if __name__ == "__main__":
    test_rmsnorm_implementations()
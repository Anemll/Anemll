#!/usr/bin/env python3
"""
Compare LayerNorm weights and biases between transformers and ANEMLL.
Focus on the first layer input_layernorm where divergence starts.
"""

import os

# CRITICAL: Set environment variables BEFORE importing
os.environ['ENABLE_SP_QUANT'] = '1'
if 'SKIP_SP_FORWARD' in os.environ:
    del os.environ['SKIP_SP_FORWARD']

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config


def compare_tensors(name, tensor1, tensor2, rtol=1e-6, atol=1e-6):
    """Compare two tensors with detailed output."""
    if tensor1 is None and tensor2 is None:
        print(f"{name}: Both are None ✅")
        return True
    elif tensor1 is None or tensor2 is None:
        print(f"{name}: One is None, other is not ❌")
        print(f"  HF: {tensor1}")
        print(f"  ANEMLL: {tensor2}")
        return False
    
    # Convert to same dtype for comparison
    t1 = tensor1.float().detach()
    t2 = tensor2.float().detach()
    
    print(f"\n{name}:")
    print(f"  Shape: {t1.shape} vs {t2.shape}")
    print(f"  Dtype: {tensor1.dtype} vs {tensor2.dtype}")
    print(f"  Mean: {t1.mean().item():.8f} vs {t2.mean().item():.8f}")
    print(f"  Std: {t1.std().item():.8f} vs {t2.std().item():.8f}")
    print(f"  Min: {t1.min().item():.8f} vs {t2.min().item():.8f}")
    print(f"  Max: {t1.max().item():.8f} vs {t2.max().item():.8f}")
    
    if t1.shape != t2.shape:
        print(f"  ❌ SHAPE MISMATCH!")
        return False
    
    # Check if close
    close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    if close:
        print(f"  ✅ Match (rtol={rtol}, atol={atol})")
    else:
        diff = torch.abs(t1 - t2)
        max_diff = diff.max().item()
        rel_diff = (diff / (torch.abs(t1) + 1e-8)).max().item()
        
        print(f"  ❌ MISMATCH!")
        print(f"  Max absolute diff: {max_diff:.8f}")
        print(f"  Max relative diff: {rel_diff:.8f}")
        
        # Show first few values
        print(f"  First 10 values:")
        print(f"    HF: {t1.flatten()[:10].tolist()}")
        print(f"    ANEMLL: {t2.flatten()[:10].tolist()}")
        
        # Show where max difference occurs
        max_idx = torch.argmax(diff)
        max_pos = np.unravel_index(max_idx.item(), diff.shape)
        print(f"  Max diff at position {max_pos}:")
        print(f"    HF: {t1[max_pos].item():.8f}")
        print(f"    ANEMLL: {t2[max_pos].item():.8f}")
    
    return close


def compare_layernorm_parameters():
    """Compare LayerNorm weights and biases in detail."""
    print("="*80)
    print("LAYERNORM PARAMETER COMPARISON")
    print("="*80)
    
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    model_path = snapshot_download(model_name)
    
    # Load transformers model
    print("\n1. Loading transformers model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    hf_model.eval()
    
    # Load ANEMLL model
    print("2. Loading ANEMLL model...")
    config = Qwen25Config.from_json(f'{model_path}/config.json')
    anemll_model = Qwen25ForCausalLM(config, disable_kv_cache=False)
    anemll_model.load_pretrained_weights(model_path)
    anemll_model.eval()
    
    # Compare first layer input layernorm
    print(f"\n{'='*60}")
    print("FIRST LAYER INPUT LAYERNORM PARAMETERS")
    print(f"{'='*60}")
    
    hf_ln = hf_model.model.layers[0].input_layernorm
    anemll_ln = anemll_model.model.layers[0].input_layernorm
    
    print(f"\nLayerNorm types:")
    print(f"  HF: {type(hf_ln)}")
    print(f"  ANEMLL: {type(anemll_ln)}")
    
    # Compare weights
    hf_weight = hf_ln.weight
    anemll_weight = anemll_ln.weight
    weight_match = compare_tensors("LayerNorm weight", hf_weight, anemll_weight)
    
    # Compare biases
    hf_bias = getattr(hf_ln, 'bias', None)
    anemll_bias = getattr(anemll_ln, 'bias', None)
    bias_match = compare_tensors("LayerNorm bias", hf_bias, anemll_bias)
    
    # Compare epsilon values
    print(f"\nEpsilon values:")
    hf_eps = getattr(hf_ln, 'eps', None)
    anemll_eps = getattr(anemll_ln, 'eps', None)
    print(f"  HF eps: {hf_eps}")
    print(f"  ANEMLL eps: {anemll_eps}")
    eps_match = hf_eps == anemll_eps
    print(f"  Epsilon match: {'✅' if eps_match else '❌'}")
    
    # Check normalized_shape
    print(f"\nNormalized shape:")
    hf_norm_shape = getattr(hf_ln, 'normalized_shape', None)
    anemll_norm_shape = getattr(anemll_ln, 'normalized_shape', None)
    print(f"  HF normalized_shape: {hf_norm_shape}")
    print(f"  ANEMLL normalized_shape: {anemll_norm_shape}")
    
    # Also compare post_attention_layernorm
    print(f"\n{'='*60}")
    print("FIRST LAYER POST ATTENTION LAYERNORM PARAMETERS")
    print(f"{'='*60}")
    
    hf_post_ln = hf_model.model.layers[0].post_attention_layernorm
    anemll_post_ln = anemll_model.model.layers[0].post_attention_layernorm
    
    hf_post_weight = hf_post_ln.weight
    anemll_post_weight = anemll_post_ln.weight
    post_weight_match = compare_tensors("Post LayerNorm weight", hf_post_weight, anemll_post_weight)
    
    hf_post_bias = getattr(hf_post_ln, 'bias', None)
    anemll_post_bias = getattr(anemll_post_ln, 'bias', None)
    post_bias_match = compare_tensors("Post LayerNorm bias", hf_post_bias, anemll_post_bias)
    
    # Compare final layernorm
    print(f"\n{'='*60}")
    print("FINAL LAYERNORM PARAMETERS")
    print(f"{'='*60}")
    
    hf_final_ln = hf_model.model.norm
    anemll_final_ln = anemll_model.model.norm
    
    hf_final_weight = hf_final_ln.weight
    anemll_final_weight = anemll_final_ln.weight
    final_weight_match = compare_tensors("Final LayerNorm weight", hf_final_weight, anemll_final_weight)
    
    hf_final_bias = getattr(hf_final_ln, 'bias', None)
    anemll_final_bias = getattr(anemll_final_ln, 'bias', None)
    final_bias_match = compare_tensors("Final LayerNorm bias", hf_final_bias, anemll_final_bias)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Input LayerNorm weight match: {'✅' if weight_match else '❌'}")
    print(f"Input LayerNorm bias match: {'✅' if bias_match else '❌'}")
    print(f"Input LayerNorm eps match: {'✅' if eps_match else '❌'}")
    print(f"Post LayerNorm weight match: {'✅' if post_weight_match else '❌'}")
    print(f"Post LayerNorm bias match: {'✅' if post_bias_match else '❌'}")
    print(f"Final LayerNorm weight match: {'✅' if final_weight_match else '❌'}")
    print(f"Final LayerNorm bias match: {'✅' if final_bias_match else '❌'}")
    
    if not weight_match:
        print("\n🔍 Input LayerNorm WEIGHT differs - this could cause the divergence!")
    elif not bias_match:
        print("\n🔍 Input LayerNorm BIAS differs - this could cause the divergence!")
    elif not eps_match:
        print("\n🔍 Input LayerNorm EPSILON differs - this could cause the divergence!")
    else:
        print("\n✅ All LayerNorm parameters match - issue might be in computation")


if __name__ == "__main__":
    compare_layernorm_parameters()
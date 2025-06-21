#!/usr/bin/env python3
"""
Layer-by-layer comparison with properly enabled SP quantization.
Now that environment timing is fixed, let's find where outputs diverge.
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


def compare_tensors(name, tensor1, tensor2, rtol=1e-4, atol=1e-4):
    """Compare two tensors and report differences."""
    # Convert to same dtype for comparison
    t1 = tensor1.float().detach()
    t2 = tensor2.float().detach()
    
    # Basic stats
    print(f"\n{name}:")
    print(f"  Shape: {t1.shape} vs {t2.shape}")
    print(f"  Mean: {t1.mean().item():.6f} vs {t2.mean().item():.6f}")
    print(f"  Std: {t1.std().item():.6f} vs {t2.std().item():.6f}")
    print(f"  Min: {t1.min().item():.6f} vs {t2.min().item():.6f}")
    print(f"  Max: {t1.max().item():.6f} vs {t2.max().item():.6f}")
    
    if t1.shape != t2.shape:
        print(f"  ❌ SHAPE MISMATCH!")
        return False
    
    # Check if close
    close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    if close:
        print(f"  ✅ Match (rtol={rtol}, atol={atol})")
    else:
        # Find maximum difference
        diff = torch.abs(t1 - t2)
        max_diff = diff.max().item()
        rel_diff = (diff / (torch.abs(t1) + 1e-8)).max().item()
        
        print(f"  ❌ MISMATCH!")
        print(f"  Max absolute diff: {max_diff:.6f}")
        print(f"  Max relative diff: {rel_diff:.6f}")
        
        # Show first few values
        print(f"  First 5 values:")
        print(f"    HF: {t1.flatten()[:5].tolist()}")
        print(f"    ANEMLL: {t2.flatten()[:5].tolist()}")
    
    return close


def detailed_layer_comparison():
    """Compare ANEMLL vs transformers layer by layer with proper quantization."""
    print("="*80)
    print("DETAILED LAYER-BY-LAYER COMPARISON")
    print("With Properly Enabled SP Quantization")
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
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test input
    text = "Who are you?"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"\nInput text: '{text}'")
    print(f"Token IDs: {input_ids.tolist()}")
    
    # Compare embeddings
    print(f"\n{'='*60}")
    print("STEP 1: EMBEDDING COMPARISON")
    print(f"{'='*60}")
    
    with torch.no_grad():
        hf_embeds = hf_model.model.embed_tokens(input_ids)
        anemll_embeds = anemll_model.model.embed_tokens(input_ids)
    
    embed_match = compare_tensors("Embeddings", hf_embeds, anemll_embeds)
    
    # Compare first layer step by step
    print(f"\n{'='*60}")
    print("STEP 2: FIRST LAYER DETAILED COMPARISON")
    print(f"{'='*60}")
    
    hf_layer = hf_model.model.layers[0]
    anemll_layer = anemll_model.model.layers[0]
    
    # 2.1 Input LayerNorm Weights
    print("\n--- 2.1 Input LayerNorm Weights ---")
    hf_ln_weight = hf_layer.input_layernorm.weight
    anemll_ln_weight = anemll_layer.input_layernorm.weight
    
    print(f"HF LayerNorm weight shape: {hf_ln_weight.shape}")
    print(f"ANEMLL LayerNorm weight shape: {anemll_ln_weight.shape}")
    print(f"HF LayerNorm weight[:5]: {hf_ln_weight[:5]}")
    print(f"ANEMLL LayerNorm weight[:5]: {anemll_ln_weight[:5]}")
    print(f"LayerNorm weights match: {torch.allclose(hf_ln_weight, anemll_ln_weight, atol=1e-5)}")
    
    if not torch.allclose(hf_ln_weight, anemll_ln_weight, atol=1e-5):
        print(f"Weight difference stats:")
        diff = torch.abs(hf_ln_weight - anemll_ln_weight)
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")
        print(f"  Std diff: {diff.std():.6f}")
    
    # 2.2 Input LayerNorm Outputs
    print("\n--- 2.2 Input LayerNorm Outputs ---")
    # First check the input dtypes and values
    print(f"HF embeds dtype: {hf_embeds.dtype}")
    print(f"ANEMLL embeds dtype: {anemll_embeds.dtype}")
    print(f"HF embeds first 5 values: {hf_embeds.flatten()[:5].tolist()}")
    print(f"ANEMLL embeds first 5 values: {anemll_embeds.flatten()[:5].tolist()}")
    
    with torch.no_grad():
        hf_ln_input = hf_layer.input_layernorm(hf_embeds)
        anemll_ln_input = anemll_layer.input_layernorm(anemll_embeds)
    
    print(f"HF LayerNorm output dtype: {hf_ln_input.dtype}")
    print(f"ANEMLL LayerNorm output dtype: {anemll_ln_input.dtype}")
    
    ln_match = compare_tensors("Input LayerNorm", hf_ln_input, anemll_ln_input)
    
    # 2.2 Attention projections
    print("\n--- 2.2 Attention Projections ---")
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        # HF attention projections
        hf_q = hf_layer.self_attn.q_proj(hf_ln_input)
        hf_k = hf_layer.self_attn.k_proj(hf_ln_input)
        hf_v = hf_layer.self_attn.v_proj(hf_ln_input)
        
        # ANEMLL attention projections (need Conv2d format)
        anemll_ln_input_conv = anemll_ln_input.permute(0, 2, 1).unsqueeze(2)  # [B, hidden, 1, seq]
        
        # Apply input scaling and projection
        anemll_attn = anemll_layer.self_attn
        
        # Q projection with scaling
        q_input_scaled = anemll_ln_input_conv * anemll_attn.q_proj_input_scale.view(1, -1, 1, 1)
        anemll_q_raw = anemll_attn.q_proj(q_input_scaled)
        anemll_q = anemll_q_raw * anemll_attn.q_proj_output_scale.view(1, -1, 1, 1)
        anemll_q = anemll_q.squeeze(2).permute(0, 2, 1)  # Back to [B, seq, hidden]
        
        # K projection with scaling
        k_input_scaled = anemll_ln_input_conv * anemll_attn.k_proj_input_scale.view(1, -1, 1, 1)
        anemll_k_raw = anemll_attn.k_proj(k_input_scaled)
        anemll_k = anemll_k_raw * anemll_attn.k_proj_output_scale.view(1, -1, 1, 1)
        anemll_k = anemll_k.squeeze(2).permute(0, 2, 1)
        
        # V projection with scaling
        v_input_scaled = anemll_ln_input_conv * anemll_attn.v_proj_input_scale.view(1, -1, 1, 1)
        anemll_v_raw = anemll_attn.v_proj(v_input_scaled)
        anemll_v = anemll_v_raw * anemll_attn.v_proj_output_scale.view(1, -1, 1, 1)
        anemll_v = anemll_v.squeeze(2).permute(0, 2, 1)
    
    # Compare projections
    q_match = compare_tensors("Q projection", hf_q, anemll_q)
    k_match = compare_tensors("K projection", hf_k, anemll_k)
    v_match = compare_tensors("V projection", hf_v, anemll_v)
    
    # 2.3 Compare projection weights directly
    print("\n--- 2.3 Weight Comparison ---")
    
    # Q weights
    hf_q_weight = hf_layer.self_attn.q_proj.weight  # [896, 896]
    anemll_q_weight = anemll_layer.self_attn.q_proj.weight.squeeze(-1).squeeze(-1).t()  # Convert Conv2d to Linear
    weight_q_match = compare_tensors("Q projection weight", hf_q_weight, anemll_q_weight)
    
    # Compare scales
    print("\n--- 2.4 Quantization Scales ---")
    print(f"Q input scale mean: {anemll_attn.q_proj_input_scale.mean().item():.6f}")
    print(f"Q output scale mean: {anemll_attn.q_proj_output_scale.mean().item():.6f}")
    print(f"Q input scale first 5: {anemll_attn.q_proj_input_scale.flatten()[:5].tolist()}")
    print(f"Q output scale first 5: {anemll_attn.q_proj_output_scale.flatten()[:5].tolist()}")
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Embeddings match: {'✅' if embed_match else '❌'}")
    print(f"Input LayerNorm match: {'✅' if ln_match else '❌'}")
    print(f"Q projection match: {'✅' if q_match else '❌'}")
    print(f"K projection match: {'✅' if k_match else '❌'}")
    print(f"V projection match: {'✅' if v_match else '❌'}")
    print(f"Q weight match: {'✅' if weight_q_match else '❌'}")
    
    if not q_match:
        print("\n🔍 ISSUE FOUND: Q projection outputs differ!")
        print("This is likely where the divergence starts.")
    elif not k_match:
        print("\n🔍 ISSUE FOUND: K projection outputs differ!")
    elif not v_match:
        print("\n🔍 ISSUE FOUND: V projection outputs differ!")
    else:
        print("\n✅ All projections match - need to check attention computation")


if __name__ == "__main__":
    detailed_layer_comparison()
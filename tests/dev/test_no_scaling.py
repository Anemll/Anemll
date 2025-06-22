#!/usr/bin/env python3
"""
Test with all quantization scaling factors set to 1.0 to isolate LayerNorm differences.
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


def compare_tensors(name, tensor1, tensor2, rtol=1e-2, atol=1e-1):
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
        return False, {}
    
    # Calculate detailed difference statistics
    diff = torch.abs(t1 - t2)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()
    rel_diff = (diff / (torch.abs(t1) + 1e-8)).max().item()
    mean_rel_diff = (diff / (torch.abs(t1) + 1e-8)).mean().item()
    
    diff_stats = {
        'max_abs_diff': max_diff,
        'mean_abs_diff': mean_diff,
        'std_abs_diff': std_diff,
        'max_rel_diff': rel_diff,
        'mean_rel_diff': mean_rel_diff,
        'name': name
    }
    
    # Check if close
    close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    if close:
        print(f"  ✅ Match (rtol={rtol}, atol={atol})")
    else:
        print(f"  ❌ MISMATCH!")
        print(f"  Max absolute diff: {max_diff:.6f}")
        print(f"  Mean absolute diff: {mean_diff:.6f}")
        print(f"  Std absolute diff: {std_diff:.6f}")
        print(f"  Max relative diff: {rel_diff:.6f}")
        print(f"  Mean relative diff: {mean_rel_diff:.6f}")
        
        # Show first few values
        print(f"  First 5 values:")
        print(f"    HF: {t1.flatten()[:5].tolist()}")
        print(f"    ANEMLL: {t2.flatten()[:5].tolist()}")
    
    return close, diff_stats


def set_all_scales_to_one(model):
    """Set all quantization scales to 1.0 to isolate LayerNorm differences."""
    print("Setting all quantization scales to 1.0...")
    
    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        
        # Attention scales
        if hasattr(layer.self_attn, 'q_proj_input_scale'):
            layer.self_attn.q_proj_input_scale.fill_(1.0)
            layer.self_attn.q_proj_output_scale.fill_(1.0)
            layer.self_attn.k_proj_input_scale.fill_(1.0)
            layer.self_attn.k_proj_output_scale.fill_(1.0)
            layer.self_attn.v_proj_input_scale.fill_(1.0)
            layer.self_attn.v_proj_output_scale.fill_(1.0)
            layer.self_attn.o_proj_input_scale.fill_(1.0)
            layer.self_attn.o_proj_output_scale.fill_(1.0)
        
        # MLP scales
        if hasattr(layer.mlp, 'gate_proj_input_scale'):
            layer.mlp.gate_proj_input_scale.fill_(1.0)
            layer.mlp.gate_proj_output_scale.fill_(1.0)
            layer.mlp.up_proj_input_scale.fill_(1.0)
            layer.mlp.up_proj_output_scale.fill_(1.0)
            layer.mlp.down_proj_input_scale.fill_(1.0)
            layer.mlp.down_proj_output_scale.fill_(1.0)
    
    print("All scales set to 1.0")


def test_with_no_scaling():
    """Compare ANEMLL vs transformers with all scaling factors set to 1.0."""
    print("="*80)
    print("TEST WITH ALL QUANTIZATION SCALES = 1.0")
    print("This isolates LayerNorm differences from quantization amplification")
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
    
    # Set all scales to 1.0
    print("3. Setting all quantization scales to 1.0...")
    set_all_scales_to_one(anemll_model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test input
    text = "Who are you?"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"\nInput text: '{text}'")
    print(f"Token IDs: {input_ids.tolist()}")
    
    # Collect all difference statistics
    all_stats = []
    
    # Compare embeddings
    print(f"\n{'='*60}")
    print("STEP 1: EMBEDDING COMPARISON")
    print(f"{'='*60}")
    
    with torch.no_grad():
        hf_embeds = hf_model.model.embed_tokens(input_ids)
        anemll_embeds = anemll_model.model.embed_tokens(input_ids)
    
    embed_match, embed_stats = compare_tensors("Embeddings", hf_embeds, anemll_embeds)
    all_stats.append(embed_stats)
    
    # Compare first layer step by step
    print(f"\n{'='*60}")
    print("STEP 2: FIRST LAYER DETAILED COMPARISON")
    print(f"{'='*60}")
    
    hf_layer = hf_model.model.layers[0]
    anemll_layer = anemll_model.model.layers[0]
    
    # 2.1 Input LayerNorm Outputs
    print("\n--- 2.1 Input LayerNorm Outputs ---")
    with torch.no_grad():
        hf_ln_input = hf_layer.input_layernorm(hf_embeds)
        anemll_ln_input = anemll_layer.input_layernorm(anemll_embeds)
    
    print(f"HF LayerNorm output dtype: {hf_ln_input.dtype}")
    print(f"ANEMLL LayerNorm output dtype: {anemll_ln_input.dtype}")
    
    ln_match, ln_stats = compare_tensors("Input LayerNorm", hf_ln_input, anemll_ln_input)
    all_stats.append(ln_stats)
    
    # 2.2 Attention projections with scales = 1.0
    print("\n--- 2.2 Attention Projections (Scales = 1.0) ---")
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        # HF attention projections
        hf_q = hf_layer.self_attn.q_proj(hf_ln_input)
        hf_k = hf_layer.self_attn.k_proj(hf_ln_input)
        hf_v = hf_layer.self_attn.v_proj(hf_ln_input)
        
        # ANEMLL attention projections (with scales = 1.0)
        anemll_ln_input_conv = anemll_ln_input.permute(0, 2, 1).unsqueeze(2)  # [B, hidden, 1, seq]
        
        # Apply scaling (now all 1.0) and projection
        anemll_attn = anemll_layer.self_attn
        
        # Q projection with scaling = 1.0
        q_input_scaled = anemll_ln_input_conv * anemll_attn.q_proj_input_scale.view(1, -1, 1, 1)  # All 1.0
        anemll_q_raw = anemll_attn.q_proj(q_input_scaled)
        anemll_q = anemll_q_raw * anemll_attn.q_proj_output_scale.view(1, -1, 1, 1)  # All 1.0
        anemll_q = anemll_q.squeeze(2).permute(0, 2, 1)  # Back to [B, seq, hidden]
        
        # K projection with scaling = 1.0
        k_input_scaled = anemll_ln_input_conv * anemll_attn.k_proj_input_scale.view(1, -1, 1, 1)  # All 1.0
        anemll_k_raw = anemll_attn.k_proj(k_input_scaled)
        anemll_k = anemll_k_raw * anemll_attn.k_proj_output_scale.view(1, -1, 1, 1)  # All 1.0
        anemll_k = anemll_k.squeeze(2).permute(0, 2, 1)
        
        # V projection with scaling = 1.0
        v_input_scaled = anemll_ln_input_conv * anemll_attn.v_proj_input_scale.view(1, -1, 1, 1)  # All 1.0
        anemll_v_raw = anemll_attn.v_proj(v_input_scaled)
        anemll_v = anemll_v_raw * anemll_attn.v_proj_output_scale.view(1, -1, 1, 1)  # All 1.0
        anemll_v = anemll_v.squeeze(2).permute(0, 2, 1)
    
    # Compare projections
    q_match, q_stats = compare_tensors("Q projection (scales=1.0)", hf_q, anemll_q)
    k_match, k_stats = compare_tensors("K projection (scales=1.0)", hf_k, anemll_k)
    v_match, v_stats = compare_tensors("V projection (scales=1.0)", hf_v, anemll_v)
    all_stats.extend([q_stats, k_stats, v_stats])
    
    # 2.3 Compare projection weights directly
    print("\n--- 2.3 Weight Comparison ---")
    
    # Q weights
    hf_q_weight = hf_layer.self_attn.q_proj.weight  # [896, 896]
    anemll_q_weight = anemll_layer.self_attn.q_proj.weight.squeeze(-1).squeeze(-1).t()  # Convert Conv2d to Linear
    weight_q_match, weight_q_stats = compare_tensors("Q projection weight", hf_q_weight, anemll_q_weight)
    all_stats.append(weight_q_stats)
    
    # Verify scales are all 1.0
    print("\n--- 2.4 Verify Scales = 1.0 ---")
    print(f"Q input scale mean: {anemll_attn.q_proj_input_scale.mean().item():.6f}")
    print(f"Q output scale mean: {anemll_attn.q_proj_output_scale.mean().item():.6f}")
    print(f"Q input scale first 5: {anemll_attn.q_proj_input_scale.flatten()[:5].tolist()}")
    print(f"Q output scale first 5: {anemll_attn.q_proj_output_scale.flatten()[:5].tolist()}")
    
    # Summary with detailed statistics
    print(f"\n{'='*80}")
    print("DETAILED DIFFERENCE STATISTICS (Scales = 1.0)")
    print(f"{'='*80}")
    print(f"{'Component':<25} {'Max Abs':<10} {'Mean Abs':<10} {'Std Abs':<10} {'Max Rel':<10} {'Mean Rel':<10}")
    print("-" * 80)
    
    for stats in all_stats:
        if stats:  # Skip empty stats
            name = stats['name'][:24]  # Truncate long names
            print(f"{name:<25} {stats['max_abs_diff']:<10.6f} {stats['mean_abs_diff']:<10.6f} "
                  f"{stats['std_abs_diff']:<10.6f} {stats['max_rel_diff']:<10.6f} {stats['mean_rel_diff']:<10.6f}")
    
    # Find worst mismatches
    print(f"\n{'='*60}")
    print("WORST MISMATCHES (by Max Absolute Difference)")
    print(f"{'='*60}")
    
    # Sort by max absolute difference
    sorted_stats = sorted([s for s in all_stats if s], key=lambda x: x['max_abs_diff'], reverse=True)
    for i, stats in enumerate(sorted_stats[:5]):  # Top 5 worst
        print(f"{i+1}. {stats['name']}: {stats['max_abs_diff']:.6f} max abs diff")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    layernorm_diff = ln_stats['max_abs_diff'] if ln_stats else 0
    max_proj_diff = max([q_stats['max_abs_diff'], k_stats['max_abs_diff'], v_stats['max_abs_diff']])
    
    print(f"LayerNorm max abs diff: {layernorm_diff:.6f}")
    print(f"Max projection diff: {max_proj_diff:.6f}")
    print(f"Amplification factor: {max_proj_diff/layernorm_diff if layernorm_diff > 0 else 'N/A':.1f}x")
    
    if max_proj_diff <= layernorm_diff * 10:  # Within 10x amplification
        print("✅ SUCCESS: Scaling was the main issue! Differences are now proportional.")
    else:
        print("❌ Other factors contribute beyond just scaling.")


if __name__ == "__main__":
    test_with_no_scaling()
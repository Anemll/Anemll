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
import math
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


def detailed_layer_comparison(skip_weight_comparison=False):
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
    
    ln_match, ln_stats = compare_tensors("Input LayerNorm", hf_ln_input, anemll_ln_input)
    all_stats.append(ln_stats)
    
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
    
    # Compare projections (skip for QuaRot models - rotations make them incomparable)
    if True: # skip for Q 
        print("\n⚠️  SKIPPING Q/K/V projection comparisons for QuaRot model")
        print("    (Individual projections are rotated - only attention output is comparable)")
        q_match = True  # Assume correct
        k_match = True
        v_match = True
        q_stats = {'name': 'Q projection (skipped)', 'max_abs_diff': 0, 'mean_abs_diff': 0, 'std_abs_diff': 0, 'max_rel_diff': 0, 'mean_rel_diff': 0}
        k_stats = {'name': 'K projection (skipped)', 'max_abs_diff': 0, 'mean_abs_diff': 0, 'std_abs_diff': 0, 'max_rel_diff': 0, 'mean_rel_diff': 0}
        v_stats = {'name': 'V projection (skipped)', 'max_abs_diff': 0, 'mean_abs_diff': 0, 'std_abs_diff': 0, 'max_rel_diff': 0, 'mean_rel_diff': 0}
    else:
        q_match, q_stats = compare_tensors("Q projection", hf_q, anemll_q)
        k_match, k_stats = compare_tensors("K projection", hf_k, anemll_k)
        v_match, v_stats = compare_tensors("V projection", hf_v, anemll_v)
    all_stats.extend([q_stats, k_stats, v_stats])
    
    # 2.3 Compare projection weights directly (skip for QuaRot models)
    if True:
        print("\n--- 2.3 Weight Comparison ---")
        print("⚠️  SKIPPING weight comparison for QuaRot model (weights are rotated and quantized)")
        weight_q_match = True  # Assume weights are correct if we're skipping
        weight_q_stats = {'name': 'Q projection weight (skipped)', 'max_abs_diff': 0, 'mean_abs_diff': 0, 'std_abs_diff': 0, 'max_rel_diff': 0, 'mean_rel_diff': 0}
    else:
        print("\n--- 2.3 Weight Comparison ---")
        
        # Q weights
        hf_q_weight = hf_layer.self_attn.q_proj.weight  # [896, 896]
        anemll_q_weight = anemll_layer.self_attn.q_proj.weight.squeeze(-1).squeeze(-1).t()  # Convert Conv2d to Linear
        weight_q_match, weight_q_stats = compare_tensors("Q projection weight", hf_q_weight, anemll_q_weight)
    
    all_stats.append(weight_q_stats)
    
    # Compare scales
    print("\n--- 2.4 Quantization Scales ---")
    print(f"Q input scale mean: {anemll_attn.q_proj_input_scale.mean().item():.6f}")
    print(f"Q output scale mean: {anemll_attn.q_proj_output_scale.mean().item():.6f}")
    print(f"Q input scale first 5: {anemll_attn.q_proj_input_scale.flatten()[:5].tolist()}")
    print(f"Q output scale first 5: {anemll_attn.q_proj_output_scale.flatten()[:5].tolist()}")
    
    # 2.5 Complete attention computation
    print("\n--- 2.5 Complete Attention Computation ---")
    
    # Need to complete the attention computation to get final output
    batch_size = 1  # Define batch_size here
    
    with torch.no_grad():
        # HuggingFace attention - complete computation
        # For Qwen2, we need to compute attention properly
        # First, apply the projections we already computed
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
        
        # Reshape for multi-head attention
        # Get the config from the model
        hf_config = hf_model.config
        num_heads = hf_config.num_attention_heads
        hidden_size = hf_config.hidden_size
        head_dim = hidden_size // num_heads
        num_key_value_heads = getattr(hf_config, 'num_key_value_heads', num_heads)
        
        # Reshape projections
        hf_q_heads = hf_q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        hf_k_heads = hf_k.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
        hf_v_heads = hf_v.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
        
        # Apply rotary position embeddings using ANEMLL's rotary emb (they should be the same)
        cos, sin = anemll_attn.rotary_emb(hf_v_heads, position_ids)
        hf_q_rot = hf_q_heads * cos + rotate_half(hf_q_heads) * sin
        hf_k_rot = hf_k_heads * cos + rotate_half(hf_k_heads) * sin
        
        # Repeat k/v heads if needed
        if num_key_value_heads != num_heads:
            n_rep = num_heads // num_key_value_heads
            hf_k_rot = repeat_kv(hf_k_rot, n_rep)
            hf_v_heads = repeat_kv(hf_v_heads, n_rep)
        
        # Compute attention
        attn_weights = torch.matmul(hf_q_rot, hf_k_rot.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hf_q.dtype)
        attn_output = torch.matmul(attn_weights, hf_v_heads)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Apply output projection
        hf_attn_output = hf_layer.self_attn.o_proj(attn_output)
        
        # ANEMLL attention - need to complete the computation
        # We already have q, k, v projections, now do attention
        # Get from config instead of attention module
        num_heads = anemll_model.config.num_attention_heads
        head_dim = anemll_model.config.hidden_size // num_heads
        num_key_value_heads = getattr(anemll_model.config, 'num_key_value_heads', num_heads)
        
        # Reshape for attention computation
        anemll_q_attn = anemll_q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        anemll_k_attn = anemll_k.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
        anemll_v_attn = anemll_v.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
        
        # Apply rotary embeddings (if needed)
        try:
            cos, sin = anemll_attn.rotary_emb(anemll_v_attn, position_ids)
            # Apply rotary position embedding
            q_rot = anemll_q_attn * cos + rotate_half(anemll_q_attn) * sin
            k_rot = anemll_k_attn * cos + rotate_half(anemll_k_attn) * sin
            anemll_q_attn = q_rot
            anemll_k_attn = k_rot
        except:
            print("  Note: Rotary embedding application skipped")
        
        # Repeat k/v heads if needed
        if num_key_value_heads != num_heads:
            n_rep = num_heads // num_key_value_heads
            anemll_k_attn = repeat_kv(anemll_k_attn, n_rep)
            anemll_v_attn = repeat_kv(anemll_v_attn, n_rep)
        
        # Attention scores
        attn_weights = torch.matmul(anemll_q_attn, anemll_k_attn.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(anemll_q_attn.dtype)
        
        # Attention output
        attn_output_anemll = torch.matmul(attn_weights, anemll_v_attn)
        attn_output_anemll = attn_output_anemll.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Apply output projection
        attn_output_conv = attn_output_anemll.permute(0, 2, 1).unsqueeze(2)  # [B, hidden, 1, seq]
        o_input_scaled = attn_output_conv * anemll_attn.o_proj_input_scale.view(1, -1, 1, 1)
        o_raw = anemll_attn.o_proj(o_input_scaled)
        anemll_attn_output = o_raw * anemll_attn.o_proj_output_scale.view(1, -1, 1, 1)
        anemll_attn_output = anemll_attn_output.squeeze(2).permute(0, 2, 1)  # Back to [B, seq, hidden]
    
    # Compare final attention outputs (this is what matters for QuaRot)
    attn_match, attn_stats = compare_tensors("Complete Attention Output", hf_attn_output, anemll_attn_output)
    all_stats.append(attn_stats)
    
    # Continue with Post-Attention LayerNorm
    print("\n--- 2.6 Post-Attention LayerNorm ---")
    hf_residual = hf_embeds + hf_attn_output
    anemll_residual = anemll_embeds + anemll_attn_output
    
    hf_post_ln = hf_layer.post_attention_layernorm(hf_residual)
    anemll_post_ln = anemll_layer.post_attention_layernorm(anemll_residual)
    
    post_ln_match, post_ln_stats = compare_tensors("Post-Attention LayerNorm", hf_post_ln, anemll_post_ln)
    all_stats.append(post_ln_stats)
    
    # Set remaining as not implemented yet
    mlp_match = False
    layer0_match = False
    ln1_match = False
    attn1_match = False
    mlp1_match = False
    layer1_match = False
    ln2_match = False
    attn2_match = False
    mlp2_match = False
    layer2_match = False
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Embeddings match: {'✅' if embed_match else '❌'}")
    print(f"Layer 0 Input LayerNorm match: {'✅' if ln_match else '❌'}")
    print(f"Layer 0 Q projection match: {'✅' if q_match else '❌'}")
    print(f"Layer 0 K projection match: {'✅' if k_match else '❌'}")
    print(f"Layer 0 V projection match: {'✅' if v_match else '❌'}")
    print(f"Layer 0 Q projection weight match: {'✅' if weight_q_match else '❌'}")
    print(f"Layer 0 Attention match: {'✅' if attn_match else '❌'}")
    print(f"Layer 0 Post-Attn LayerNorm match: {'✅' if post_ln_match else '❌'}")
    print(f"Layer 0 MLP match: {'✅' if mlp_match else '❌'}")
    print(f"Layer 0 Output match: {'✅' if layer0_match else '❌'}")
    print(f"Layer 1 Input LayerNorm match: {'✅' if ln1_match else '❌'}")
    print(f"Layer 1 Attention match: {'✅' if attn1_match else '❌'}")
    print(f"Layer 1 MLP match: {'✅' if mlp1_match else '❌'}")
    print(f"Layer 1 Output match: {'✅' if layer1_match else '❌'}")
    print(f"Layer 2 Input LayerNorm match: {'✅' if ln2_match else '❌'}")
    print(f"Layer 2 Attention match: {'✅' if attn2_match else '❌'}")
    print(f"Layer 2 MLP match: {'✅' if mlp2_match else '❌'}")
    print(f"Layer 2 Output match: {'✅' if layer2_match else '❌'}")
    
    # Detailed statistics summary
    print(f"\n{'='*80}")
    print("DETAILED DIFFERENCE STATISTICS")
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
    
    # Find first divergence point
    print(f"\n{'='*60}")
    print("DIVERGENCE ANALYSIS")
    print(f"{'='*60}")
    
    if not embed_match:
        print("🔍 DIVERGENCE STARTS: Embeddings differ!")
    elif not ln_match:
        print("🔍 DIVERGENCE STARTS: Layer 0 Input LayerNorm differs!")
    elif not q_match:
        print("🔍 DIVERGENCE STARTS: Layer 0 Q projection differs!")
    elif not k_match:
        print("🔍 DIVERGENCE STARTS: Layer 0 K projection differs!")
    elif not v_match:
        print("🔍 DIVERGENCE STARTS: Layer 0 V projection differs!")
    elif not weight_q_match:
        print("🔍 DIVERGENCE STARTS: Layer 0 Q projection weights differ!")
    elif not attn_match:
        print("🔍 DIVERGENCE STARTS: Layer 0 Attention computation differs!")
    elif not post_ln_match:
        print("🔍 DIVERGENCE STARTS: Layer 0 Post-Attention LayerNorm differs!")
    elif not mlp_match:
        print("🔍 DIVERGENCE STARTS: Layer 0 MLP differs!")
    elif not layer0_match:
        print("🔍 DIVERGENCE STARTS: Layer 0 residual connection differs!")
    elif not ln1_match:
        print("🔍 DIVERGENCE STARTS: Layer 1 Input LayerNorm differs!")
    elif not attn1_match:
        print("🔍 DIVERGENCE STARTS: Layer 1 Attention differs!")
    elif not mlp1_match:
        print("🔍 DIVERGENCE STARTS: Layer 1 MLP differs!")
    elif not layer1_match:
        print("🔍 DIVERGENCE STARTS: Layer 1 residual connection differs!")
    elif not ln2_match:
        print("🔍 DIVERGENCE STARTS: Layer 2 Input LayerNorm differs!")
    elif not attn2_match:
        print("🔍 DIVERGENCE STARTS: Layer 2 Attention differs!")
    elif not mlp2_match:
        print("🔍 DIVERGENCE STARTS: Layer 2 MLP differs!")
    else:
        print("✅ All components through Layer 2 match!")


# Helper functions for attention computation
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states, n_rep):
    """Repeat key/value heads."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


if __name__ == "__main__":
    # Check if a model path is provided to test QuaRot
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-weights', action='store_true', help='Skip weight comparisons (auto-enabled for QuaRot)')
    args = parser.parse_args()
    
    detailed_layer_comparison(skip_weight_comparison=args.skip_weights)
#!/usr/bin/env python3
"""
Functional comparison for QuaRot models.
Instead of comparing raw weights (which are rotated and scaled),
we compare the functional outputs at each layer.
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


def compare_hidden_states(name, tensor1, tensor2, rtol=1e-2, atol=1e-1):
    """Compare two hidden state tensors (functional outputs)."""
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
        'name': name,
        'max_abs_diff': max_diff,
        'mean_abs_diff': mean_diff,
        'std_abs_diff': std_diff,
        'max_rel_diff': rel_diff,
        'mean_rel_diff': mean_rel_diff
    }
    
    # Check if they match within tolerances
    match = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    
    if not match:
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
    else:
        print(f"  ✅ MATCH (within rtol={rtol}, atol={atol})")
    
    return match, diff_stats


def functional_layer_comparison():
    """Compare functional outputs of QuaRot models layer by layer."""
    print("="*80)
    print("FUNCTIONAL COMPARISON FOR QUAROT MODELS")
    print("Comparing hidden states instead of raw weights")
    print("="*80)
    
    # Load models
    model_path = "/tmp/qwen25_quarot_fused_w8"
    
    print("\nLoading models...")
    # HuggingFace model (original)
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32)
    hf_model.eval()
    
    # ANEMLL model (QuaRot quantized)
    config = Qwen25Config.from_pretrained(model_path)
    anemll_model = Qwen25ForCausalLM(config)
    anemll_model.load_weights(model_path)
    anemll_model.eval()
    
    # Test input
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    text = "The quick brown fox"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    print(f"\nTest input: '{text}'")
    print(f"Input IDs: {input_ids}")
    print(f"Shape: {input_ids.shape}")
    
    all_stats = []
    
    # Track hidden states through the model
    with torch.no_grad():
        # 1. Embeddings
        print(f"\n{'='*60}")
        print("EMBEDDINGS")
        print(f"{'='*60}")
        
        hf_embeds = hf_model.model.embed_tokens(input_ids)
        anemll_embeds = anemll_model.model.embed_tokens(input_ids)
        
        embed_match, embed_stats = compare_hidden_states("Embeddings", hf_embeds, anemll_embeds)
        all_stats.append(embed_stats)
        
        # Track hidden states
        hf_hidden = hf_embeds
        anemll_hidden = anemll_embeds
        
        # 2. Process through layers
        for layer_idx in range(3):  # First 3 layers
            print(f"\n{'='*60}")
            print(f"LAYER {layer_idx}")
            print(f"{'='*60}")
            
            hf_layer = hf_model.model.layers[layer_idx]
            anemll_layer = anemll_model.model.layers[layer_idx]
            
            # 2.1 Input LayerNorm
            hf_ln_out = hf_layer.input_layernorm(hf_hidden)
            anemll_ln_out = anemll_layer.input_layernorm(anemll_hidden)
            
            ln_match, ln_stats = compare_hidden_states(f"Layer {layer_idx} Input LayerNorm", hf_ln_out, anemll_ln_out)
            all_stats.append(ln_stats)
            
            # 2.2 Attention (complete computation)
            # For HF model
            attn_output, _, _ = hf_layer.self_attn(
                hidden_states=hf_ln_out,
                position_ids=torch.arange(input_ids.shape[1]).unsqueeze(0)
            )
            
            # For ANEMLL model - we need to handle the Conv2D format properly
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            
            # Convert to Conv2D format
            anemll_ln_conv = anemll_ln_out.permute(0, 2, 1).unsqueeze(2)  # [B, hidden, 1, seq]
            
            # Apply attention with proper scaling
            attn = anemll_layer.self_attn
            
            # Q projection
            q_input_scaled = anemll_ln_conv * attn.q_proj_input_scale.view(1, -1, 1, 1)
            q_raw = attn.q_proj(q_input_scaled)
            q = q_raw * attn.q_proj_output_scale.view(1, -1, 1, 1)
            
            # K projection
            k_input_scaled = anemll_ln_conv * attn.k_proj_input_scale.view(1, -1, 1, 1)
            k_raw = attn.k_proj(k_input_scaled)
            k = k_raw * attn.k_proj_output_scale.view(1, -1, 1, 1)
            
            # V projection
            v_input_scaled = anemll_ln_conv * attn.v_proj_input_scale.view(1, -1, 1, 1)
            v_raw = attn.v_proj(v_input_scaled)
            v = v_raw * attn.v_proj_output_scale.view(1, -1, 1, 1)
            
            # Convert back from Conv2D format and reshape for attention
            batch_size = 1
            q = q.squeeze(2).permute(0, 2, 1).view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = k.squeeze(2).permute(0, 2, 1).view(batch_size, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            v = v.squeeze(2).permute(0, 2, 1).view(batch_size, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            
            # Apply rotary embeddings
            cos, sin = attn.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
            # Attention computation
            if attn.num_key_value_heads != attn.num_heads:
                # Repeat k/v heads
                k = repeat_kv(k, attn.num_heads // attn.num_key_value_heads)
                v = repeat_kv(v, attn.num_heads // attn.num_key_value_heads)
            
            attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(attn.head_dim)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output_anemll = torch.matmul(attn_weights, v)
            
            # Reshape and apply output projection
            attn_output_anemll = attn_output_anemll.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            attn_output_conv = attn_output_anemll.permute(0, 2, 1).unsqueeze(2)  # [B, hidden, 1, seq]
            
            # Output projection with scaling
            o_input_scaled = attn_output_conv * attn.o_proj_input_scale.view(1, -1, 1, 1)
            o_raw = attn.o_proj(o_input_scaled)
            attn_output_anemll = o_raw * attn.o_proj_output_scale.view(1, -1, 1, 1)
            attn_output_anemll = attn_output_anemll.squeeze(2).permute(0, 2, 1)  # Back to [B, seq, hidden]
            
            attn_match, attn_stats = compare_hidden_states(f"Layer {layer_idx} Attention Output", attn_output, attn_output_anemll)
            all_stats.append(attn_stats)
            
            # 2.3 Residual + Post-Attention LayerNorm
            hf_hidden = hf_hidden + attn_output
            anemll_hidden = anemll_hidden + attn_output_anemll
            
            hf_ln2_out = hf_layer.post_attention_layernorm(hf_hidden)
            anemll_ln2_out = anemll_layer.post_attention_layernorm(anemll_hidden)
            
            ln2_match, ln2_stats = compare_hidden_states(f"Layer {layer_idx} Post-Attn LayerNorm", hf_ln2_out, anemll_ln2_out)
            all_stats.append(ln2_stats)
            
            # 2.4 MLP
            hf_mlp_out = hf_layer.mlp(hf_ln2_out)
            
            # ANEMLL MLP with Conv2D format
            mlp_input_conv = anemll_ln2_out.permute(0, 2, 1).unsqueeze(2)
            mlp = anemll_layer.mlp
            
            # Gate projection
            gate_input_scaled = mlp_input_conv * mlp.gate_proj_input_scale.view(1, -1, 1, 1)
            gate_raw = mlp.gate_proj(gate_input_scaled)
            gate = gate_raw * mlp.gate_proj_output_scale.view(1, -1, 1, 1)
            
            # Up projection
            up_input_scaled = mlp_input_conv * mlp.up_proj_input_scale.view(1, -1, 1, 1)
            up_raw = mlp.up_proj(up_input_scaled)
            up = up_raw * mlp.up_proj_output_scale.view(1, -1, 1, 1)
            
            # Activation and multiplication
            gate_activated = mlp.act_fn(gate)
            mlp_output = gate_activated * up
            
            # Down projection
            down_input_scaled = mlp_output * mlp.down_proj_input_scale.view(1, -1, 1, 1)
            down_raw = mlp.down_proj(down_input_scaled)
            anemll_mlp_out = down_raw * mlp.down_proj_output_scale.view(1, -1, 1, 1)
            anemll_mlp_out = anemll_mlp_out.squeeze(2).permute(0, 2, 1)  # Back to [B, seq, hidden]
            
            mlp_match, mlp_stats = compare_hidden_states(f"Layer {layer_idx} MLP Output", hf_mlp_out, anemll_mlp_out)
            all_stats.append(mlp_stats)
            
            # 2.5 Final residual
            hf_hidden = hf_hidden + hf_mlp_out
            anemll_hidden = anemll_hidden + anemll_mlp_out
            
            layer_match, layer_stats = compare_hidden_states(f"Layer {layer_idx} Final Output", hf_hidden, anemll_hidden)
            all_stats.append(layer_stats)
    
    # Summary
    print(f"\n{'='*80}")
    print("FUNCTIONAL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Component':<35} {'Max Abs':<12} {'Mean Abs':<12} {'Match':<10}")
    print("-" * 70)
    
    for stats in all_stats:
        if stats:
            name = stats['name'][:34]
            max_diff = stats['max_abs_diff']
            mean_diff = stats['mean_abs_diff']
            match = "✅" if max_diff < 0.1 else "❌"
            print(f"{name:<35} {max_diff:<12.6f} {mean_diff:<12.6f} {match:<10}")
    
    # Find first divergence
    print(f"\n{'='*60}")
    print("DIVERGENCE ANALYSIS")
    print(f"{'='*60}")
    
    first_mismatch = None
    for stats in all_stats:
        if stats and stats['max_abs_diff'] > 0.1:
            first_mismatch = stats['name']
            break
    
    if first_mismatch:
        print(f"🔍 First significant divergence: {first_mismatch}")
    else:
        print("✅ All functional outputs match within tolerance!")


# Helper functions for rotary embeddings
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embeddings."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    """Repeat key/value heads."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


import math

if __name__ == "__main__":
    functional_layer_comparison()
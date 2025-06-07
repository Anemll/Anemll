#!/usr/bin/env python3
"""Inspect the original Qwen model's internal implementations to understand differences."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import glob
import os

def inspect_original_qwen():
    """Inspect the original Qwen model's components."""
    
    print("🔍 Inspecting Original Qwen Model Internals")
    print("="*60)
    
    # Load original model
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    if not model_dirs:
        print("❌ Error: Qwen model not found in cache")
        return False
    
    model_dir = model_dirs[0]
    print(f"Loading model from: {model_dir}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16
    )
    
    print(f"\n📋 MODEL ARCHITECTURE:")
    print("-" * 40)
    print(f"Model type: {type(model)}")
    print(f"Model config: {type(model.config)}")
    print(f"Base model: {type(model.model)}")
    
    # Check model layers
    print(f"\n🏗️ LAYER STRUCTURE:")
    print("-" * 40)
    print(f"Number of layers: {len(model.model.layers)}")
    print(f"First layer type: {type(model.model.layers[0])}")
    
    # Inspect first layer components
    first_layer = model.model.layers[0]
    print(f"\n🔧 FIRST LAYER COMPONENTS:")
    print("-" * 40)
    print(f"Self attention: {type(first_layer.self_attn)}")
    print(f"MLP: {type(first_layer.mlp)}")
    print(f"Input norm: {type(first_layer.input_layernorm)}")
    print(f"Post attention norm: {type(first_layer.post_attention_layernorm)}")
    
    # Check attention components
    attn = first_layer.self_attn
    print(f"\n👁️ ATTENTION COMPONENTS:")
    print("-" * 40)
    print(f"Q projection: {type(attn.q_proj)}")
    print(f"K projection: {type(attn.k_proj)}")
    print(f"V projection: {type(attn.v_proj)}")
    print(f"O projection: {type(attn.o_proj)}")
    if hasattr(attn, 'rotary_emb'):
        print(f"Rotary embedding: {type(attn.rotary_emb)}")
    else:
        print("Rotary embedding: None (might be handled differently)")
    
    # Check if head norms exist
    if hasattr(attn, 'q_norm'):
        print(f"Q norm: {type(attn.q_norm)}")
    else:
        print("Q norm: None")
    
    if hasattr(attn, 'k_norm'):
        print(f"K norm: {type(attn.k_norm)}")
    else:
        print("K norm: None")
    
    # Check for other rotary-related attributes
    if hasattr(attn, 'rope_theta'):
        print(f"Rope theta: {attn.rope_theta}")
    if hasattr(attn, 'max_position_embeddings'):
        print(f"Max position embeddings: {attn.max_position_embeddings}")
    
    # List all attention attributes
    print(f"All attention attributes: {[attr for attr in dir(attn) if not attr.startswith('_')]}")
    
    # Check projection weights shapes
    print(f"Q proj weight shape: {attn.q_proj.weight.shape}")
    print(f"K proj weight shape: {attn.k_proj.weight.shape}")
    print(f"V proj weight shape: {attn.v_proj.weight.shape}")
    print(f"O proj weight shape: {attn.o_proj.weight.shape}")
    
    # Check MLP components
    mlp = first_layer.mlp
    print(f"\n🧠 MLP COMPONENTS:")
    print("-" * 40)
    print(f"Gate projection: {type(mlp.gate_proj)}")
    print(f"Up projection: {type(mlp.up_proj)}")
    print(f"Down projection: {type(mlp.down_proj)}")
    print(f"Activation: {mlp.act_fn if hasattr(mlp, 'act_fn') else 'Unknown'}")
    
    # Check MLP weight shapes
    print(f"Gate proj weight shape: {mlp.gate_proj.weight.shape}")
    print(f"Up proj weight shape: {mlp.up_proj.weight.shape}")
    print(f"Down proj weight shape: {mlp.down_proj.weight.shape}")
    
    # Check if MLP has activation function
    if hasattr(mlp, 'act_fn'):
        print(f"Activation function: {mlp.act_fn}")
    else:
        print("No explicit activation function found")
    
    # Check norm implementations
    input_norm = first_layer.input_layernorm
    print(f"\n📏 NORMALIZATION DETAILS:")
    print("-" * 40)
    print(f"Input norm class: {type(input_norm)}")
    print(f"Input norm attributes: {dir(input_norm)}")
    
    # Check if it's RMSNorm
    if hasattr(input_norm, 'weight'):
        print(f"Has weight parameter: True, shape: {input_norm.weight.shape}")
    if hasattr(input_norm, 'bias'):
        print(f"Has bias parameter: True")
    else:
        print(f"Has bias parameter: False")
    if hasattr(input_norm, 'eps'):
        print(f"Epsilon: {input_norm.eps}")
    
    # Test the normalization behavior
    print(f"\n🧪 NORMALIZATION TEST:")
    print("-" * 40)
    
    # Create test input
    test_input = torch.randn(1, 4, model.config.hidden_size, dtype=torch.float16)
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input range: [{test_input.min().item():.4f}, {test_input.max().item():.4f}]")
    
    # Apply normalization
    with torch.no_grad():
        normalized = input_norm(test_input)
        print(f"Normalized shape: {normalized.shape}")
        print(f"Normalized range: [{normalized.min().item():.4f}, {normalized.max().item():.4f}]")
        print(f"Normalized mean: {normalized.mean().item():.6f}")
        print(f"Normalized std: {normalized.std().item():.6f}")
    
    # Check what happens with mean-centered input vs non-mean-centered
    print(f"\n🔬 MEAN-CENTERING TEST:")
    print("-" * 40)
    
    # Test 1: Regular input
    test1 = torch.randn(1, 4, model.config.hidden_size, dtype=torch.float16)
    with torch.no_grad():
        norm1 = input_norm(test1)
    
    # Test 2: Mean-centered input  
    test2 = test1 - test1.mean(dim=-1, keepdim=True)
    with torch.no_grad():
        norm2 = input_norm(test2)
    
    print(f"Original input mean: {test1.mean(dim=-1)[0, 0].item():.6f}")
    print(f"Mean-centered input mean: {test2.mean(dim=-1)[0, 0].item():.6f}")
    print(f"Norm difference: {torch.abs(norm1 - norm2).max().item():.6f}")
    
    # Check final norm
    final_norm = model.model.norm
    print(f"\n🏁 FINAL NORM:")
    print("-" * 40)
    print(f"Final norm type: {type(final_norm)}")
    print(f"Final norm class: {final_norm.__class__}")
    
    # Get the source code of the normalization if possible
    try:
        import inspect
        print(f"\n📝 NORMALIZATION SOURCE:")
        print("-" * 40)
        norm_source = inspect.getsource(input_norm.forward)
        print(norm_source)
    except:
        print("Could not get source code")
    
    return True

if __name__ == "__main__":
    inspect_original_qwen() 
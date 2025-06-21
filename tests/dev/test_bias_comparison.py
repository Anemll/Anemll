#!/usr/bin/env python3
"""
Compare biases between standard Qwen2 model and GPTQ quantized version.
Shows how transformers handles different model structures.
"""

import torch
from transformers import AutoModelForCausalLM, Qwen2ForCausalLM, Qwen2Config
from huggingface_hub import snapshot_download
from safetensors import safe_open
import json


def check_model_architecture():
    """Check what the Qwen2 architecture expects for biases."""
    print("="*60)
    print("Qwen2 Architecture Definition")
    print("="*60)
    
    # Create a dummy config to see default architecture
    config = Qwen2Config(
        hidden_size=896,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864
    )
    
    # Create model to inspect
    model = Qwen2ForCausalLM(config)
    
    print("Checking which layers have bias parameters by default:")
    bias_params = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            bias_params.append(name)
    
    print(f"Total bias parameters in standard Qwen2: {len(bias_params)}")
    if bias_params:
        print("Bias parameters found:")
        for b in bias_params[:10]:
            print(f"  {b}")
    else:
        print("No bias parameters in standard Qwen2 architecture!")
    
    # Check specific layers
    print("\nChecking specific layer configurations:")
    layer0 = model.model.layers[0]
    
    # Check attention
    print(f"  layer0.self_attn.q_proj has bias: {hasattr(layer0.self_attn.q_proj, 'bias') and layer0.self_attn.q_proj.bias is not None}")
    print(f"  layer0.self_attn.k_proj has bias: {hasattr(layer0.self_attn.k_proj, 'bias') and layer0.self_attn.k_proj.bias is not None}")
    print(f"  layer0.self_attn.v_proj has bias: {hasattr(layer0.self_attn.v_proj, 'bias') and layer0.self_attn.v_proj.bias is not None}")
    print(f"  layer0.self_attn.o_proj has bias: {hasattr(layer0.self_attn.o_proj, 'bias') and layer0.self_attn.o_proj.bias is not None}")
    
    # Check MLP
    print(f"  layer0.mlp.gate_proj has bias: {hasattr(layer0.mlp.gate_proj, 'bias') and layer0.mlp.gate_proj.bias is not None}")
    print(f"  layer0.mlp.up_proj has bias: {hasattr(layer0.mlp.up_proj, 'bias') and layer0.mlp.up_proj.bias is not None}")
    print(f"  layer0.mlp.down_proj has bias: {hasattr(layer0.mlp.down_proj, 'bias') and layer0.mlp.down_proj.bias is not None}")
    
    return model


def compare_model_files():
    """Compare what's in the model files."""
    print("\n" + "="*60)
    print("Comparing Model Files")
    print("="*60)
    
    # Base Qwen model
    base_model = "Qwen/Qwen2.5-0.5B"
    print(f"\nBase model: {base_model}")
    base_path = snapshot_download(base_model)
    
    # Check base model structure
    base_config_path = f"{base_path}/config.json"
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    print(f"Base model config keys: {list(base_config.keys())}")
    
    # Check if config specifies bias
    if 'bias' in str(base_config):
        print("Config mentions bias settings")
    
    # Load base model state dict keys
    base_model_file = f"{base_path}/model.safetensors"
    base_keys = []
    with safe_open(base_model_file, framework="pt", device="cpu") as f:
        base_keys = list(f.keys())
    
    base_bias_keys = [k for k in base_keys if 'bias' in k]
    print(f"Base model bias tensors: {len(base_bias_keys)}")
    
    # GPTQ model
    gptq_model = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    print(f"\nGPTQ model: {gptq_model}")
    gptq_path = snapshot_download(gptq_model)
    
    # Load GPTQ model state dict keys
    gptq_model_file = f"{gptq_path}/model.safetensors"
    gptq_keys = []
    with safe_open(gptq_model_file, framework="pt", device="cpu") as f:
        gptq_keys = list(f.keys())
    
    gptq_bias_keys = [k for k in gptq_keys if 'bias' in k]
    print(f"GPTQ model bias tensors: {len(gptq_bias_keys)}")
    
    # Compare
    print("\n--- Comparison ---")
    print(f"Extra bias tensors in GPTQ: {len(gptq_bias_keys) - len(base_bias_keys)}")
    
    # Show what's different
    if len(gptq_bias_keys) > 0 and len(base_bias_keys) == 0:
        print("\nGPTQ added biases to layers that don't have them in base model!")
        print("Sample GPTQ bias keys:")
        for k in gptq_bias_keys[:5]:
            print(f"  {k}")


def test_loading_behavior():
    """Test how transformers handles loading models with different biases."""
    print("\n" + "="*60)
    print("Testing Loading Behavior")
    print("="*60)
    
    gptq_model = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    gptq_path = snapshot_download(gptq_model)
    
    # Load config without quantization info
    config_path = f"{gptq_path}/config.json"
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Remove quantization config
    if 'quantization_config' in config_dict:
        del config_dict['quantization_config']
    
    # Create config object
    config = Qwen2Config(**config_dict)
    
    print("Loading GPTQ model with Qwen2ForCausalLM...")
    
    # Option 1: Load with strict=False (default)
    print("\n1. Loading with default behavior (strict=False):")
    try:
        model1 = Qwen2ForCausalLM.from_pretrained(
            gptq_path,
            config=config,
            torch_dtype=torch.float32
        )
        
        # Check what got loaded
        loaded_biases = []
        for name, param in model1.named_parameters():
            if 'bias' in name and param is not None:
                loaded_biases.append(name)
        
        print(f"   ✓ Loaded successfully")
        print(f"   Bias parameters loaded: {len(loaded_biases)}")
        
        # Check if biases are actually used
        layer0 = model1.model.layers[0]
        has_q_bias = hasattr(layer0.self_attn.q_proj, 'bias') and layer0.self_attn.q_proj.bias is not None
        print(f"   q_proj bias exists and loaded: {has_q_bias}")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Show what happens
    print("\n--- How it works ---")
    print("1. Qwen2 architecture defines Linear layers WITHOUT bias by default")
    print("2. When loading, transformers sees bias tensors in the checkpoint")
    print("3. Since the Linear layers have bias=False, these bias tensors are ignored")
    print("4. The model runs but WITHOUT using the extra biases from GPTQ")
    
    # Verify by checking actual layer definition
    print("\n--- Layer Definition Check ---")
    print("Checking q_proj layer definition:")
    print(f"  Type: {type(layer0.self_attn.q_proj)}")
    print(f"  Has bias attribute: {hasattr(layer0.self_attn.q_proj, 'bias')}")
    if hasattr(layer0.self_attn.q_proj, 'bias'):
        print(f"  Bias is None: {layer0.self_attn.q_proj.bias is None}")
    
    # Check the Linear layer configuration
    if hasattr(layer0.self_attn.q_proj, 'bias'):
        print(f"  Linear layer bias setting: bias={layer0.self_attn.q_proj.bias is not None}")


def main():
    # First check architecture expectations
    check_model_architecture()
    
    # Compare model files
    compare_model_files()
    
    # Test loading behavior
    test_loading_behavior()


if __name__ == "__main__":
    main()
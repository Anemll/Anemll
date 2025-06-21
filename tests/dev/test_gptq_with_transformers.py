#!/usr/bin/env python3
"""
Test GPTQ quantized model with standard transformers.
This bypasses all SP quantization logic and uses the pre-quantized weights directly.
"""

import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from huggingface_hub import snapshot_download


def check_model_biases(model_path):
    """Check which layers have biases in the model."""
    print("="*60)
    print("Checking Model Biases")
    print("="*60)
    
    # List all files to see what we have
    import glob
    files = glob.glob(f"{model_path}/*")
    print("Files in model directory:")
    for f in sorted(files):
        print(f"  {os.path.basename(f)}")
    
    # Check if there's a pytorch model file
    pt_files = glob.glob(f"{model_path}/*.pt") + glob.glob(f"{model_path}/*.bin") + glob.glob(f"{model_path}/*.safetensors")
    print(f"\nModel files found: {[os.path.basename(f) for f in pt_files]}")
    
    # Load the state dict to check for biases
    if pt_files:
        print(f"\nLoading state dict from {os.path.basename(pt_files[0])}...")
        if pt_files[0].endswith('.safetensors'):
            from safetensors import safe_open
            state_dict_keys = []
            with safe_open(pt_files[0], framework="pt", device="cpu") as f:
                state_dict_keys = list(f.keys())
        else:
            state_dict = torch.load(pt_files[0], map_location='cpu')
            state_dict_keys = list(state_dict.keys())
        
        # Check for bias keys
        bias_keys = [k for k in state_dict_keys if 'bias' in k]
        print(f"\nBias tensors found: {len(bias_keys)}")
        if bias_keys:
            print("Bias keys (first 10):")
            for key in bias_keys[:10]:
                print(f"  {key}")
        else:
            print("No bias tensors found in the model!")
        
        # Check for scale keys (from GPTQ)
        scale_keys = [k for k in state_dict_keys if 'scale' in k.lower()]
        print(f"\nScale tensors found: {len(scale_keys)}")
        if scale_keys:
            print("Scale keys (first 10):")
            for key in scale_keys[:10]:
                print(f"  {key}")
                
        # Check what projection layers exist
        proj_keys = [k for k in state_dict_keys if any(proj in k for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])]
        print(f"\nProjection layers found: {len(proj_keys)}")
        print("Sample projection keys (first 5):")
        for key in proj_keys[:5]:
            print(f"  {key}")


def test_gptq_with_transformers(model_path):
    """Test GPTQ model with standard transformers."""
    print("\n" + "="*60)
    print("Testing GPTQ Model with Standard Transformers")
    print("="*60)
    
    # First, modify config.json to remove quantization_config
    config_path = f"{model_path}/config.json"
    print(f"Loading config from {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Remove quantization config if present
    if 'quantization_config' in config:
        print("Removing quantization_config from config.json")
        del config['quantization_config']
    
    # Save modified config
    modified_config_path = f"{model_path}/config_no_quant.json"
    with open(modified_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved modified config to {modified_config_path}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Try to load model with standard transformers
    print("\nLoading model with transformers (ignoring quantization)...")
    try:
        # Use the modified config
        model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            config=modified_config_path,
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True,  # In case quantized weights have different shapes
            device_map="cpu"
        )
        print("✓ Model loaded successfully!")
        
        # Check model structure
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Check for biases in loaded model
        bias_count = 0
        for name, param in model.named_parameters():
            if 'bias' in name:
                bias_count += 1
                if bias_count <= 5:  # Show first 5
                    print(f"  Found bias: {name} with shape {param.shape}")
        
        print(f"Total bias parameters in loaded model: {bias_count}")
        
    except Exception as e:
        print(f"✗ Failed to load with standard transformers: {e}")
        print("\nTrying alternative loading method...")
        
        # Try loading with AutoModelForCausalLM
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=modified_config_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
            print("✓ Model loaded with AutoModelForCausalLM!")
        except Exception as e2:
            print(f"✗ Also failed with AutoModelForCausalLM: {e2}")
            return
    
    # Test inference
    print("\n" + "-"*40)
    print("Testing Inference")
    print("-"*40)
    
    prompt = "Who are you?"
    print(f"Prompt: '{prompt}'")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"Token IDs: {inputs['input_ids'].tolist()}")
    
    # Generate
    try:
        print("\nGenerating response...")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        
        # Show first few generated tokens
        generated_ids = outputs[0][len(inputs["input_ids"][0]):]
        print("\nGenerated tokens:")
        for i, token_id in enumerate(generated_ids[:5]):
            token = tokenizer.decode([token_id])
            print(f"  Token {i+1}: '{token}' (ID: {token_id})")
            
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Get the locally stored model path
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    
    # Download or use cached version
    print(f"Using model: {model_name}")
    model_path = snapshot_download(model_name)
    print(f"Model path: {model_path}")
    
    # First check what's in the model
    check_model_biases(model_path)
    
    # Then test with transformers
    test_gptq_with_transformers(model_path)


if __name__ == "__main__":
    main()
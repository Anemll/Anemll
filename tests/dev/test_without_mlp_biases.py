#!/usr/bin/env python3
"""
Test ANEMLL model without MLP biases to match standard transformers behavior.
This helps isolate if the gibberish output is due to bias handling.
"""

import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download


def test_without_mlp_biases():
    """Test ANEMLL model loading GPTQ weights but skipping MLP biases."""
    print("="*60)
    print("Testing ANEMLL Without MLP Biases")
    print("="*60)
    
    # Use the GPTQ model
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    model_path = snapshot_download(model_name)
    
    # Load config
    config = Qwen25Config.from_json(f'{model_path}/config.json')
    
    # Create model
    print("Creating ANEMLL model...")
    model = Qwen25ForCausalLM(config, disable_kv_cache=False)
    
    # Monkey-patch the load function to skip MLP biases
    original_load = model.load_pretrained_weights
    
    def load_with_bias_filter(path):
        print("\n--- Custom weight loading (skipping MLP biases) ---")
        
        # First, let's see what biases the model expects
        expected_biases = []
        for name, param in model.named_parameters():
            if 'bias' in name:
                expected_biases.append(name)
        
        print(f"Model expects {len(expected_biases)} bias parameters")
        print("Expected biases (first 10):")
        for b in expected_biases[:10]:
            print(f"  {b}")
        
        # Load state dict
        from safetensors import safe_open
        model_file = f"{path}/model.safetensors"
        
        # Track what we're loading
        loaded_biases = []
        skipped_biases = []
        loaded_weights = []
        loaded_scales = []
        
        with safe_open(model_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                
                # Skip MLP biases
                if 'mlp' in key and 'bias' in key:
                    skipped_biases.append(key)
                    continue
                
                # Skip o_proj biases (not in standard Qwen2)
                if 'o_proj.bias' in key:
                    skipped_biases.append(key)
                    continue
                
                # Track what we're loading
                if 'bias' in key:
                    loaded_biases.append(key)
                elif 'scale' in key:
                    loaded_scales.append(key)
                elif 'weight' in key:
                    loaded_weights.append(key)
                
                # Try to load into model
                try:
                    # Remove 'model.' prefix if present
                    param_name = key
                    if param_name.startswith('model.'):
                        param_name = param_name[6:]
                    
                    # Get the parameter
                    if param_name in dict(model.named_parameters()):
                        param = dict(model.named_parameters())[param_name]
                        param.data.copy_(tensor.to(param.dtype))
                except Exception as e:
                    # Silently skip parameters that don't exist in the model
                    pass
        
        print(f"\nLoading summary:")
        print(f"  Loaded weights: {len(loaded_weights)}")
        print(f"  Loaded biases: {len(loaded_biases)}")
        print(f"  Skipped biases: {len(skipped_biases)}")
        print(f"  Scale tensors seen: {len(loaded_scales)}")
        
        print(f"\nLoaded biases:")
        for b in loaded_biases[:5]:
            print(f"  {b}")
        
        print(f"\nSkipped MLP/o_proj biases:")
        for b in skipped_biases[:5]:
            print(f"  {b}")
    
    # Load weights with bias filtering
    load_with_bias_filter(model_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test inference
    print("\n" + "="*60)
    print("Testing Inference")
    print("="*60)
    
    prompt = "Who are you?"
    print(f"Prompt: '{prompt}'")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Token IDs: {input_ids.tolist()}")
    
    # Generate tokens
    print("\nGenerating response...")
    generated_ids = input_ids[0].tolist()
    current_pos = len(generated_ids) - 1
    
    # Initialize position IDs and masks
    seq_len = len(generated_ids)
    position_ids = torch.arange(seq_len, dtype=torch.long)
    
    # Create causal mask
    causal_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float16)
    for i in range(seq_len):
        causal_mask[:, :, i, i+1:] = float('-inf')
    
    # Generate first 10 tokens
    for i in range(10):
        with torch.no_grad():
            # Prepare inputs
            input_tensor = torch.tensor([generated_ids], dtype=torch.long)
            update_mask = torch.zeros(1, dtype=torch.long)
            
            # Forward pass
            outputs = model(
                input_tensor,
                update_mask,
                position_ids[:len(generated_ids)],
                causal_mask[:, :, :len(generated_ids), :len(generated_ids)],
                torch.tensor(current_pos, dtype=torch.long)
            )
            
            # Get next token
            next_token_logits = outputs[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            # Add to generated sequence
            generated_ids.append(next_token_id)
            current_pos += 1
            
            # Update position IDs and mask
            new_position = torch.tensor([current_pos], dtype=torch.long)
            position_ids = torch.cat([position_ids, new_position])
            
            # Expand causal mask
            new_seq_len = len(generated_ids)
            new_causal_mask = torch.zeros((1, 1, new_seq_len, new_seq_len), dtype=torch.float16)
            new_causal_mask[:, :, :seq_len, :seq_len] = causal_mask
            for j in range(new_seq_len):
                new_causal_mask[:, :, j, j+1:] = float('-inf')
            causal_mask = new_causal_mask
            seq_len = new_seq_len
            
            # Show token
            token = tokenizer.decode([next_token_id])
            print(f"Token {i+1}: '{token}' (ID: {next_token_id})")
    
    # Decode full response
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\nFull response: {response}")


def compare_with_transformers():
    """Compare with standard transformers for reference."""
    print("\n" + "="*60)
    print("Reference: Standard Transformers")
    print("="*60)
    
    from transformers import AutoModelForCausalLM
    
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    
    # Load with transformers (which ignores extra biases)
    print("Loading with transformers...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Test
    prompt = "Who are you?"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=10,
            do_sample=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Transformers response: {response}")


def main():
    # Test ANEMLL without MLP biases
    test_without_mlp_biases()
    
    # Compare with transformers
    compare_with_transformers()


if __name__ == "__main__":
    # Disable SP quantization for this test
    os.environ['SKIP_SP_FORWARD'] = '1'
    print("Note: Running with SKIP_SP_FORWARD=1 (no quantization scaling)")
    main()
#!/usr/bin/env python3
"""
Test script to compare standard transformers inference with ANEMLL SP quantization.
This helps identify if issues are from quantization or model implementation.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def test_standard_transformers():
    """Test with standard HuggingFace transformers."""
    print("="*60)
    print("Testing Standard Transformers (Baseline)")
    print("="*60)
    
    model_name = "Qwen/Qwen2.5-0.5B"
    
    # Load standard transformers model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu"
    )
    model.eval()
    
    # Test prompt
    prompt = "Who are you?"
    print(f"\nPrompt: '{prompt}'")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"Token IDs: {inputs['input_ids'].tolist()}")
    
    # Generate
    print("\nGenerating response...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=tokenizer.eos_token_id
        )
    
    generation_time = time.time() - start_time
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse: {response}")
    print(f"Generation time: {generation_time:.2f}s")
    
    # Show token-by-token generation
    print("\n--- Token-by-token generation ---")
    generated_ids = outputs[0][len(inputs["input_ids"][0]):]
    for i, token_id in enumerate(generated_ids[:10]):  # Show first 10 tokens
        token = tokenizer.decode([token_id])
        print(f"Token {i+1}: '{token}' (ID: {token_id})")
    
    return response


def test_anemll_model():
    """Test with ANEMLL model implementation."""
    print("\n" + "="*60)
    print("Testing ANEMLL Model Implementation")
    print("="*60)
    
    # Set environment for SP quantization
    os.environ['ENABLE_SP_QUANT'] = '1'
    
    # Import ANEMLL model
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
    from huggingface_hub import snapshot_download
    
    model_name = "Qwen/Qwen2.5-0.5B"
    
    # Download and load model
    print(f"Loading model: {model_name}")
    model_path = snapshot_download(model_name)
    
    config = Qwen25Config.from_json(f'{model_path}/config.json')
    model = Qwen25ForCausalLM(config, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Check quantization scales
    scale_count = 0
    for name, module in model.named_modules():
        for attr in ['gate_proj_input_scale', 'up_proj_input_scale', 'down_proj_input_scale', 
                     'q_proj_input_scale', 'k_proj_input_scale', 'v_proj_input_scale', 'o_proj_input_scale']:
            if hasattr(module, attr):
                scale_count += 1
    
    print(f"Quantization scales found: {scale_count}")
    
    # Test prompt
    prompt = "Who are you?"
    print(f"\nPrompt: '{prompt}'")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Token IDs: {input_ids.tolist()}")
    
    # Generate token by token
    print("\nGenerating response...")
    start_time = time.time()
    
    generated_ids = input_ids[0].tolist()
    current_pos = len(generated_ids) - 1
    
    # Initialize position IDs and masks
    seq_len = len(generated_ids)
    position_ids = torch.arange(seq_len, dtype=torch.long)
    
    # Create causal mask
    causal_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float16)
    for i in range(seq_len):
        causal_mask[:, :, i, i+1:] = float('-inf')
    
    # Generate tokens
    max_new_tokens = 50
    for i in range(max_new_tokens):
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
            
            # Stop if EOS
            if next_token_id == tokenizer.eos_token_id:
                break
            
            # Show progress
            if i < 10:
                token = tokenizer.decode([next_token_id])
                print(f"Token {i+1}: '{token}' (ID: {next_token_id})")
    
    generation_time = time.time() - start_time
    
    # Decode full response
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\nResponse: {response}")
    print(f"Generation time: {generation_time:.2f}s")
    
    return response


def test_anemll_without_quantization():
    """Test ANEMLL model without quantization."""
    print("\n" + "="*60)
    print("Testing ANEMLL Model WITHOUT Quantization")
    print("="*60)
    
    # Disable quantization
    os.environ['SKIP_SP_FORWARD'] = '1'
    
    # Run the same test
    return test_anemll_model()


def main():
    print("Comparing Standard Transformers vs ANEMLL Implementation")
    print("="*80)
    
    # Test 1: Standard transformers (baseline)
    transformers_response = test_standard_transformers()
    
    # Test 2: ANEMLL without quantization
    anemll_no_quant_response = test_anemll_without_quantization()
    
    # Test 3: ANEMLL with quantization
    if 'SKIP_SP_FORWARD' in os.environ:
        del os.environ['SKIP_SP_FORWARD']
    anemll_quant_response = test_anemll_model()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Standard Transformers: {transformers_response[:100]}...")
    print(f"ANEMLL (no quant):     {anemll_no_quant_response[:100]}...")
    print(f"ANEMLL (with quant):   {anemll_quant_response[:100]}...")


if __name__ == "__main__":
    main()
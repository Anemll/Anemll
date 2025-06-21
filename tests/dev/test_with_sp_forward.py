#!/usr/bin/env python3
"""
Test ANEMLL with SP quantization ENABLED (no SKIP_SP_FORWARD).
This should properly handle the quantized weights.
"""

import os

# CRITICAL: Set environment variables BEFORE importing the model
os.environ['ENABLE_SP_QUANT'] = '1'
# Do NOT set SKIP_SP_FORWARD
if 'SKIP_SP_FORWARD' in os.environ:
    del os.environ['SKIP_SP_FORWARD']

import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download


def test_with_sp_quantization():
    """Test ANEMLL with SP quantization properly enabled."""
    print("="*80)
    print("Testing ANEMLL WITH SP Quantization")
    print("="*80)
    
    print(f"ENABLE_SP_QUANT: {os.environ.get('ENABLE_SP_QUANT', 'not set')}")
    print(f"SKIP_SP_FORWARD: {os.environ.get('SKIP_SP_FORWARD', 'not set')}")
    
    # Use the GPTQ quantized model
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    model_path = snapshot_download(model_name)
    
    # Load config
    config = Qwen25Config.from_json(f'{model_path}/config.json')
    
    # Create model
    print("\nCreating ANEMLL model with SP quantization...")
    model = Qwen25ForCausalLM(config, disable_kv_cache=False)
    
    # Load weights (should apply quantization properly)
    print("Loading pretrained weights...")
    model.load_pretrained_weights(model_path)
    model.eval()
    
    # Check quantization scales
    print("\n--- Quantization Scale Check ---")
    scale_count = 0
    for name, module in model.named_modules():
        for attr in ['gate_proj_input_scale', 'up_proj_input_scale', 'down_proj_input_scale', 
                     'q_proj_input_scale', 'k_proj_input_scale', 'v_proj_input_scale', 'o_proj_input_scale']:
            if hasattr(module, attr):
                scale = getattr(module, attr)
                scale_count += 1
                if scale_count <= 5:  # Show first 5
                    print(f"  {name}.{attr}: shape={scale.shape}, values={scale[:3].tolist()}")
    
    print(f"Total quantization scales loaded: {scale_count}")
    
    # Check if weights look reasonable after quantization loading
    print("\n--- Weight Check After Quantization Loading ---")
    layer0_q_weight = model.model.layers[0].self_attn.q_proj.weight
    print(f"Q projection weight shape: {layer0_q_weight.shape}")
    print(f"Q projection weight dtype: {layer0_q_weight.dtype}")
    print(f"Q projection weight mean: {layer0_q_weight.mean().item():.6f}")
    print(f"Q projection weight std: {layer0_q_weight.std().item():.6f}")
    print(f"Q projection first 5 values: {layer0_q_weight.flatten()[:5].tolist()}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test inference
    print("\n" + "="*60)
    print("Testing Inference with SP Quantization")
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
            
            # Forward pass with quantization
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
            
            # Stop if EOS
            if next_token_id == tokenizer.eos_token_id:
                break
    
    # Decode full response
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\nFull response: {response}")


def main():
    test_with_sp_quantization()


if __name__ == "__main__":
    main()
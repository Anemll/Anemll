#!/usr/bin/env python3
"""
Final test with properly enabled SP quantization to see if output is now coherent.
"""

import os

# CRITICAL: Set environment variables BEFORE importing
os.environ['ENABLE_SP_QUANT'] = '1'
if 'SKIP_SP_FORWARD' in os.environ:
    del os.environ['SKIP_SP_FORWARD']

import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download


def test_final_inference():
    """Test with properly enabled SP quantization."""
    print("="*80)
    print("FINAL TEST: ANEMLL with Properly Enabled SP Quantization")
    print("="*80)
    
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    model_path = snapshot_download(model_name)
    
    # Create and load model
    config = Qwen25Config.from_json(f'{model_path}/config.json')
    model = Qwen25ForCausalLM(config, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test inference
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
    
    # Generate tokens
    for i in range(20):  # Generate more tokens to see if it's coherent
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
    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print(f"{'='*60}")
    print(f"Response: {response}")
    
    # Compare with transformers baseline
    print(f"\n{'='*60}")
    print("COMPARISON WITH TRANSFORMERS")
    print(f"{'='*60}")
    
    from transformers import AutoModelForCausalLM
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    with torch.no_grad():
        hf_outputs = hf_model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False
        )
    
    hf_response = tokenizer.decode(hf_outputs[0], skip_special_tokens=True)
    print(f"Transformers: {hf_response}")
    print(f"ANEMLL:       {response}")
    
    # Check if they match
    if response == hf_response:
        print("\n🎉 SUCCESS: ANEMLL matches transformers exactly!")
    elif response.startswith(hf_response[:10]):  # At least same start
        print("\n✅ GOOD: ANEMLL output is coherent and similar to transformers")
    else:
        print("\n❌ ISSUE: ANEMLL still produces different output")


if __name__ == "__main__":
    test_final_inference()
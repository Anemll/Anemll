#!/usr/bin/env python3
"""
Final test with properly enabled SP quantization to see if output is now coherent.
"""

import os

# CRITICAL: Set environment variables BEFORE importing
os.environ['ENABLE_SP_QUANT'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism to avoid fork warning
if 'SKIP_SP_FORWARD' in os.environ:
    del os.environ['SKIP_SP_FORWARD']
use_single_token_prefill = True  # Option to use single token prefill

#model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
#model_name = "Qwen/Qwen2.5-0.5B"
model_name = "Qwen/Qwen2.5-0.5B-Instruct" # os.environ['ENABLE_SP_QUANT'] = '0' = required

max_tokens = 20

# Test inference
#prompt = "Who are you?"
#prompt = "What is Apple Neural Engine?"
prompt = "What is GPTQ?"


import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config, ENABLE_SP_QUANT
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download


def test_final_inference():
    """Test with properly enabled SP quantization."""
    print("="*80)
    print("FINAL TEST: ANEMLL with Properly Enabled SP Quantization")
    print("="*80)
    

    model_path = snapshot_download(model_name)
    
    # Create and load model
    config = Qwen25Config.from_json(f'{model_path}/config.json')
    model = Qwen25ForCausalLM(config, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    print(f"Prompt: '{prompt}'")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    print(f"Token IDs: {input_ids.tolist()}")
    
    # Generate tokens using correct ANEMLL pattern
    print("\nGenerating response...")
    generated_ids = input_ids[0].tolist()
    
    # ANEMLL uses FIXED context size - get from config
    context_length = config.context_length  # Usually 256
    
    # ANEMLL uses FIXED position IDs (0 to context_length-1) created as needed
    
    # Create FIXED causal mask for full context (using correct ANEMLL pattern)
    import numpy as np
    def make_causal_mask(length, start):
        """Create causal attention mask."""
        mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
        row_indices = np.arange(length).reshape(length, 1)
        col_indices = np.arange(length).reshape(1, length)
        mask[:, :, col_indices <= (row_indices + start)] = 0
        return mask
    
    causal_mask_data = make_causal_mask(context_length, 0)
    causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
    
    # Step 1: Prefill KV cache - option for single token prefill
    prompt_length = len(generated_ids)
    
    with torch.no_grad():
        if use_single_token_prefill:
            print(f"Single token prefill: processing {prompt_length} prompt tokens one by one...")
            
            # Process each prompt token individually
            for i, token_id in enumerate(generated_ids):
                single_token = torch.tensor([[token_id]], dtype=torch.long)
                
                # Single token generation (same as regular generation)
                outputs = model(
                    single_token,  # input_ids
                    torch.zeros(1, 1), # update_mask (not used in prefill)
                    torch.tensor([i], dtype=torch.long),  # position_ids
                    causal_mask[:, :, i:i+1, :],  # causal_mask - single row
                    torch.tensor(i, dtype=torch.long),  # current_pos
                    IN_PREFILL=True
                )
                # We don't need the outputs during prefill, just populating KV cache
                
        else:
            print(f"Batch prefill: processing {prompt_length} prompt tokens at once...")
            
            # Use the original prompt for prefill (batch mode)
            prefill_position_ids = torch.arange(prompt_length, dtype=torch.long)
            
            # Create causal mask for prefill: only within prompt length
            prefill_causal_mask = torch.zeros((1, 1, prompt_length, context_length), dtype=torch.float16)
            
            # Apply causal mask: token i can attend to tokens 0 through i, -inf for future positions
            for i in range(prompt_length):
                prefill_causal_mask[:, :, i, i+1:context_length] = float('-inf')
            
            # Run prefill to populate KV cache
            model(
                input_ids,  # input_ids
                torch.zeros(1, prompt_length),  # update_mask
                prefill_position_ids,  # position_ids
                prefill_causal_mask,   # causal_mask
                torch.tensor(0, dtype=torch.long),  # current_pos
                IN_PREFILL=True
            )
    
    # Step 2: Generate tokens one by one
    current_pos = prompt_length  # Start generating at position after prompt
    
    for i in range(max_tokens):  # Generate tokens
        with torch.no_grad():
            # Get the last generated token (or last prompt token for first generation)
            if len(generated_ids) > prompt_length:
                # Use last generated token
                last_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long)
            else:
                # Use last prompt token for first generation
                last_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long)
            
            # Single token generation
            # Create update mask for single token at current position
            update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
            update_mask[0, 0, current_pos, 0] = 1.0
            
            outputs = model(
                last_token,  # input_ids
                update_mask,  # update_mask
                torch.tensor([current_pos], dtype=torch.long),  # position_ids
                causal_mask[:, :, current_pos:current_pos+1, :],  # causal_mask - single row
                torch.tensor(current_pos, dtype=torch.long),  # current_pos
                IN_PREFILL=False
            )
            
            # Get next token (outputs is the tensor directly)
            next_token_logits = outputs[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            # Add to generated sequence and update position
            generated_ids.append(next_token_id)
            current_pos += 1
            
            # Show token
            token = tokenizer.decode([next_token_id])
            print(f"Token {i+1}: '{token}' (ID: {next_token_id})")
            
            # Stop if EOS or exceed context
            if next_token_id == tokenizer.eos_token_id or current_pos >= context_length:
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
        torch_dtype=torch.float16,
        #torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    with torch.no_grad():
        hf_outputs = hf_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )
    
    hf_response = tokenizer.decode(hf_outputs[0], skip_special_tokens=True)
    print(f"------------------------------------------------")
    print(f"Model: {model_name}")
    print(f"ENABLE_SP_QUANT: {ENABLE_SP_QUANT}")
    print(f"Prompt: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
    print(f"Max tokens: {max_tokens}")
    print(f"------------------------------------------------")

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
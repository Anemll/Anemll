#!/usr/bin/env python3
"""Test the converted CoreML Qwen model."""

import numpy as np
import torch
import coremltools as ct
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def test_coreml_model():
    print("Loading original model and tokenizer...")
    # Use repo name instead of local path
    model_name = "Qwen/Qwen2.5-0.5B"  # Use a similar working model for tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # We don't actually need the original model for this test, just the tokenizer

    print("Loading CoreML model...")
    #coreml_model = ct.models.MLModel("q3.mlpackage")
    compute_unit = ct.ComputeUnit.CPU_AND_NE
    coreml_model = ct.models.CompiledMLModel("../qwen-test/test_qwen.mlpackage", compute_unit)

    print("model loaded")

    
    print("Model inputs:", list(coreml_model.get_spec().description.input))
    print("Model outputs:", list(coreml_model.get_spec().description.output))
    
    # Use the same prompt as test_sequential_tokens.py
    prompt = "What is Apple Neural Engine?"
    print(f"\nTesting with prompt: '{prompt}'")
    
    # Tokenize the full prompt
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids
    prompt_tokens = input_ids[0].tolist()
    print(f"Tokenized to {len(prompt_tokens)} tokens: {prompt_tokens}")
    print(f"Token meanings: {[tokenizer.decode([t]) for t in prompt_tokens]}")
    
    # Generate 3 tokens like test_sequential_tokens.py
    max_new_tokens = 13
    generated_tokens = prompt_tokens.copy()
    
    batch_size = 1
    context_length = 256
    
    print(f"\n🚀 Generating {max_new_tokens} tokens with CoreML...")
    
    print(f"\n🔍 DEBUGGING: Comparing with PyTorch sequential test")
    print(f"PyTorch processes FULL SEQUENCE each time:")
    print(f"  - Step 6: input_ids = [[3838, 374, 8162, 60477, 8200, 30]] shape [1, 6]")
    print(f"  - position_ids = [0, 1, 2, 3, 4, 5] shape [6]")  
    print(f"  - current_pos = [5] shape [1]")
    print(f"  - Gets logits from output[0, -1, :] (last position)")
    print(f"  - Predicts next token: 2585 (' How')")
    print(f"\nCoreML approach: Single token inference with KV cache")
    
    for gen_step in range(max_new_tokens):
        print(f"\n--- Generation Step {gen_step + 1} ---")
        
        if gen_step == 0:
            # For first step, we need to process the LAST token of the prompt to get next prediction
            # This should match PyTorch step 6: processing token 30 at position 5
            current_token = generated_tokens[-1]  # Token 30 ('?')
            current_pos = len(generated_tokens) - 1  # Position 5
            
            print(f"🎯 CRITICAL FIRST STEP:")
            print(f"  Processing token {current_token} ('{tokenizer.decode([current_token])}') at position {current_pos}")
            print(f"  This should match PyTorch: token 30 at position 5")
            print(f"  Expected prediction: 2585 (' How')")
        else:
            # For subsequent steps, process the newly generated token
            current_token = generated_tokens[-1]
            current_pos = len(generated_tokens) - 1
            
            print(f"Processing token {current_token} at position {current_pos}")
        
        print(f"Current sequence: {generated_tokens}")
        print(f"Current sequence length: {len(generated_tokens)}")
        
        # Prepare inputs for CoreML (single token inference with KV cache format)
        input_ids = np.array([[current_token]], dtype=np.int32)  # [1, 1]
        position_ids = np.array([current_pos], dtype=np.int32)   # [1] - current position
        
        # Causal mask for current position - should allow attention to all previous positions
        causal_mask = np.ones((1, 1, 1, context_length), dtype=np.float16)  # [1, 1, 1, 256]
        
        # Current position
        current_pos_input = np.array([current_pos], dtype=np.int32)  # [1]
        
        # Update mask for KV cache
        update_mask = np.zeros((1, 1, context_length, 1), dtype=np.float16)  # [1, 1, 256, 1]
        update_mask[0, 0, current_pos, 0] = 1.0  # Update current position
        
        print(f"📊 CoreML inputs:")
        print(f"  input_ids: {input_ids} shape {input_ids.shape}")
        print(f"  position_ids: {position_ids} shape {position_ids.shape}")
        print(f"  current_pos: {current_pos_input} shape {current_pos_input.shape}")
        print(f"  causal_mask: shape {causal_mask.shape} (all ones)")
        print(f"  update_mask: shape {update_mask.shape} (1.0 at position {current_pos})")
        
        # Run CoreML inference
        try:
            coreml_inputs = {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'causal_mask': causal_mask,
                'current_pos': current_pos_input,
                'update_mask': update_mask
            }
            
            coreml_outputs = coreml_model.predict(coreml_inputs)
            
            # Concatenate all logits parts (16 parts)
            num_logits = 16
            logits_parts = []
            for i in range(1, num_logits + 1):
                key = f'logits{i}'
                if key in coreml_outputs:
                    logits_parts.append(coreml_outputs[key])
            
            coreml_logits = np.concatenate(logits_parts, axis=-1)  # [1, 1, 151936]
            logits_1d = coreml_logits[0, 0]  # [151936]
            
            # Get next token
            next_token = int(np.argmax(logits_1d))
            
            # Show top 5 predictions for debugging
            top_indices = np.argsort(logits_1d)[::-1][:5]
            print(f"🔍 Top 5 predictions:")
            for i, idx in enumerate(top_indices):
                idx_int = int(idx)
                token_text = tokenizer.decode([idx_int])
                print(f"  {i+1}. Token {idx_int}: '{token_text}' (logit: {logits_1d[idx]:.4f})")
            
            if gen_step == 0:
                print(f"\n🎯 COMPARISON:")
                print(f"  Expected (PyTorch): 2585 (' How')")
                print(f"  Got (CoreML): {next_token} ('{tokenizer.decode([next_token])}')")
                print(f"  Match: {next_token == 2585}")
                if next_token != 2585:
                    print(f"  ❌ MISMATCH! CoreML is not reproducing PyTorch results")
                else:
                    print(f"  ✅ MATCH! CoreML reproduces PyTorch results")
            
            generated_tokens.append(next_token)
            print(f"\nGenerated token: {next_token} ('{tokenizer.decode([next_token])}')")
            
        except Exception as e:
            print(f"❌ CoreML inference failed at step {gen_step + 1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Show final results
    print(f"\n📝 GENERATION RESULTS")
    print(f"=" * 50)
    print(f"Original prompt: '{prompt}'")
    
    # Decode the new tokens
    new_tokens = generated_tokens[len(prompt_tokens):]
    new_tokens_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print(f"Generated tokens: {new_tokens}")
    print(f"Generated text: '{new_tokens_text}'")
    print(f"Full text: '{full_text}'")
    
    print("\n✅ CoreML model test completed successfully!")

if __name__ == "__main__":
    test_coreml_model() 
#!/usr/bin/env python3
"""Simple test to see what original PyTorch model generates."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

def test_original_pytorch():
    print("🔍 Testing Original PyTorch Model")
    print("=" * 50)
    
    # Load original model (same as used for CoreML conversion)
    import os
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    print(f"Loading from local path: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()
    
    # Use the same prompt as our CoreML test
    prompt = "What is Apple Neural Engine?"
    print(f"Prompt: '{prompt}'")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids
    prompt_tokens = input_ids[0].tolist()
    print(f"Tokenized to {len(prompt_tokens)} tokens: {prompt_tokens}")
    
    # Decode tokens to see what they are
    print("Token meanings:")
    for i, token_id in enumerate(prompt_tokens):
        try:
            text = tokenizer.decode([token_id])
            print(f"  {i}: {token_id} -> '{text}'")
        except:
            print(f"  {i}: {token_id} -> <decode_error>")
    
    # Generate tokens one by one to match our CoreML approach
    print(f"\n🚀 Generating tokens step by step...")
    
    current_input_ids = input_ids.clone()
    generated_tokens = []
    
    for step in range(5):  # Generate 5 tokens
        print(f"\n--- Step {step + 1} ---")
        print(f"Current input shape: {current_input_ids.shape}")
        print(f"Current input: {current_input_ids[0].tolist()}")
        
        with torch.no_grad():
            outputs = model(current_input_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]
            
            # Get logits for the last position
            last_pos_logits = logits[0, -1, :]  # [vocab_size]
            
            print(f"Logits shape: {logits.shape}")
            print(f"Last position logits shape: {last_pos_logits.shape}")
            
            # Get next token
            next_token_id = torch.argmax(last_pos_logits).item()
            
            # Show top 5 predictions
            top_logits, top_indices = torch.topk(last_pos_logits, 5)
            print(f"Top 5 predictions:")
            for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
                try:
                    token_text = tokenizer.decode([idx.item()])
                    print(f"  {i+1}. Token {idx.item()}: '{token_text}' (logit: {logit.item():.4f})")
                except:
                    print(f"  {i+1}. Token {idx.item()}: <decode_error> (logit: {logit.item():.4f})")
            
            print(f"Selected token: {next_token_id}")
            generated_tokens.append(next_token_id)
            
            # Add the new token to input for next iteration
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
            current_input_ids = torch.cat([current_input_ids, next_token_tensor], dim=1)
    
    print(f"\n📝 FINAL RESULTS")
    print(f"=" * 30)
    print(f"Generated tokens: {generated_tokens}")
    
    # Try to decode the generated sequence
    try:
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Generated text: '{generated_text}'")
        
        # Full sequence
        full_tokens = prompt_tokens + generated_tokens
        full_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
        print(f"Full text: '{full_text}'")
    except Exception as e:
        print(f"Decode error: {e}")
        print(f"Generated tokens: {generated_tokens}")
    
    return generated_tokens

if __name__ == "__main__":
    expected_tokens = test_original_pytorch()
    print(f"\n🎯 Expected tokens for CoreML comparison: {expected_tokens}") 
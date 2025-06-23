#!/usr/bin/env python3
"""Debug script to examine token generation and find why it's repeating."""

import torch
import os
import sys
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config, ENABLE_SP_QUANT

def debug_generation():
    """Debug the token generation process step by step."""
    print("=" * 80)
    print("DEBUGGING TOKEN GENERATION")
    print("=" * 80)
    
    # Model setup (same as test script)
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    model_path = snapshot_download(model_name)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    config = Qwen25Config.from_json(f'{model_path}/config.json')
    model = Qwen25ForCausalLM(config, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    model.eval()
    
    # Prepare input
    prompt = "Who are you?"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Prompt: '{prompt}'")
    print(f"Input token IDs: {input_ids.tolist()}")
    print(f"Input tokens: {[tokenizer.decode([t]) for t in input_ids[0]]}")
    
    max_new_tokens = 10
    current_ids = input_ids.clone()
    
    print(f"\nGenerating {max_new_tokens} tokens...")
    print("-" * 60)
    
    for step in range(max_new_tokens):
        print(f"\nStep {step + 1}:")
        print(f"Current length: {current_ids.shape[1]}")
        print(f"Current tokens: {[tokenizer.decode([t]) for t in current_ids[0]]}")
        
        # Forward pass
        with torch.no_grad():
            # Create inputs for the model
            position_ids = torch.arange(current_ids.shape[1], dtype=torch.long)
            update_mask = torch.zeros((1, 1, 256, 1), dtype=torch.float16)
            causal_mask = torch.full((1, 1, current_ids.shape[1], 256), -torch.inf, dtype=torch.float16)
            for j in range(current_ids.shape[1]):
                causal_mask[:, :, j, :j+1] = 0
            current_pos = torch.tensor([current_ids.shape[1] - 1], dtype=torch.long)
            
            outputs = model(
                input_ids=current_ids,
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False
            )
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                # Multiple logits returned (CoreML mode)
                logits = torch.cat(outputs, dim=-1)
            else:
                logits = outputs
        
        # Get logits for the last token
        last_logits = logits[0, -1, :]  # Shape: [vocab_size]
        print(f"Logits shape: {logits.shape}")
        print(f"Last token logits range: [{last_logits.min().item():.3f}, {last_logits.max().item():.3f}]")
        
        # Get top 5 tokens and their probabilities
        probs = torch.softmax(last_logits, dim=-1)
        top_k = 5
        top_probs, top_indices = torch.topk(probs, top_k)
        
        print(f"Top {top_k} tokens:")
        for i in range(top_k):
            token_id = top_indices[i].item()
            prob = top_probs[i].item()
            token_str = tokenizer.decode([token_id])
            print(f"  {i+1}. Token {token_id} ('{token_str}'): {prob:.4f}")
        
        # Sample next token (greedy for debugging)
        next_token_id = torch.argmax(last_logits, dim=-1, keepdim=True)
        next_token_str = tokenizer.decode([next_token_id.item()])
        print(f"Selected token: {next_token_id.item()} ('{next_token_str}')")
        
        # Check if we're stuck in a loop
        if step > 0 and next_token_id.item() == prev_token_id:
            print(f"⚠️  WARNING: Same token generated as previous step!")
            print(f"⚠️  Previous logits range: [{prev_last_logits.min().item():.3f}, {prev_last_logits.max().item():.3f}]")
            print(f"⚠️  Current logits range: [{last_logits.min().item():.3f}, {last_logits.max().item():.3f}]")
            print(f"⚠️  Logits identical: {torch.allclose(prev_last_logits, last_logits, rtol=1e-5)}")
            break
        
        # Add to sequence
        current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0)], dim=1)
        
        # Store for loop detection
        prev_token_id = next_token_id.item()
        prev_last_logits = last_logits.clone()
        
        # Check for EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            print("EOS token generated, stopping.")
            break
    
    # Final output
    generated_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
    print(f"\n" + "=" * 60)
    print(f"Final generated text: '{generated_text}'")
    print(f"Final token sequence: {current_ids.tolist()}")

def check_model_state():
    """Check the model's internal state to identify issues."""
    print("CHECKING MODEL STATE")
    print("=" * 80)
    
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    model_path = snapshot_download(model_name)
    
    config = Qwen25Config.from_json(f'{model_path}/config.json')
    model = Qwen25ForCausalLM(config, disable_kv_cache=False)
    model.load_pretrained_weights(model_path)
    
    # Check some model parameters
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Check if lm_head weights are loaded correctly
    if hasattr(model, 'lm_head16_1'):
        print(f"LM head weight range: [{model.lm_head16_1.weight.min().item():.3f}, {model.lm_head16_1.weight.max().item():.3f}]")
        print(f"LM head weight shape: {model.lm_head16_1.weight.shape}")
    
    # Check embedding weights
    print(f"Embedding weight range: [{model.model.embed_tokens.weight.min().item():.3f}, {model.model.embed_tokens.weight.max().item():.3f}]")
    
    # Check a few layer weights
    for i in range(min(3, len(model.model.layers))):
        layer = model.model.layers[i]
        q_proj = layer.self_attn.q_proj.weight
        print(f"Layer {i} Q projection weight range: [{q_proj.min().item():.3f}, {q_proj.max().item():.3f}]")

if __name__ == "__main__":
    debug_generation()
    print("\n" + "=" * 80)
    check_model_state()
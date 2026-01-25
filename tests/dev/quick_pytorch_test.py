#!/usr/bin/env python3
"""
Quick test of PyTorch implementation on single sample
"""

import os
import sys
import torch
import numpy as np
from datasets import load_dataset

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add paths
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from transformers import AutoTokenizer
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def quick_test():
    """Quick test single sample"""
    
    # Load model (shortened)
    model_path = "Qwen/Qwen2.5-0.5B"
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        try:
            model_path = snapshot_download(repo_id=model_path, local_files_only=True)
        except:
            model_path = snapshot_download(repo_id=model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    import json
    with open(os.path.join(model_path, "config.json"), 'r') as f:
        hf_config = json.load(f)
    
    config = Qwen25Config(
        vocab_size=hf_config['vocab_size'],
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        state_length=512,
        rms_norm_eps=1e-6,
    )
    
    model = Qwen25ForCausalLM(config, disable_kv_cache=True)
    if not model.load_pretrained_weights(model_path):
        raise RuntimeError("Failed to load weights")
    
    print("Model loaded")
    
    # Get first BoolQ sample  
    dataset = load_dataset('boolq', split='validation')
    sample = dataset[0]
    
    # Create correct context
    context = f"{sample['passage']}\nQuestion: {sample['question']}?\nAnswer:"
    
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']} ({'yes' if sample['answer'] else 'no'})")
    
    # Test both choices
    choices = [" no", " yes"]  # BoolQ choice order: 0=no, 1=yes
    scores = []
    
    for i, choice in enumerate(choices):
        print(f"\nTesting choice {i}: '{choice}'")
        
        # Tokenize
        context_tokens = tokenizer.encode(context, add_special_tokens=True)
        choice_tokens = tokenizer.encode(choice, add_special_tokens=False)
        
        # Create full sequence
        full_sequence = torch.tensor(context_tokens + choice_tokens, dtype=torch.long)
        
        # Score just the choice token (simplified)
        pos = len(context_tokens) - 1
        last_token = torch.tensor([[context_tokens[-1]]], dtype=torch.long)
        
        # Create masks
        def make_causal_mask(length, start):
            mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
            row_indices = np.arange(length).reshape(length, 1)
            col_indices = np.arange(length).reshape(1, length)
            mask[:, :, col_indices <= (row_indices + start)] = 0
            return mask
        
        ctx_len = model.config.context_length
        causal_mask_data = make_causal_mask(ctx_len, 0)
        causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
        
        update_mask = torch.zeros((1, 1, ctx_len, 1), dtype=torch.float16)
        if pos < ctx_len:
            update_mask[0, 0, pos, 0] = 1.0
        
        with torch.no_grad():
            logits = model(
                last_token,
                update_mask,
                torch.tensor([pos], dtype=torch.long),
                causal_mask[:, :, pos:pos+1, :],
                torch.tensor(pos, dtype=torch.long),
                IN_PREFILL=False
            )
            
            # Extract logits - match ANE exactly
            if logits.dim() == 3:
                logits = logits[0, -1]  # [vocab]
            elif logits.dim() == 2:
                logits = logits[-1]    # [vocab]
            
            # Use exact same calculation as ANE
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Score the choice token
            choice_score = log_probs[choice_tokens[0]].item()
            scores.append(choice_score)
            
            print(f"  Choice token: {choice_tokens[0]}")
            print(f"  Log prob: {choice_score:.6f}")
    
    print(f"\n=== RESULTS ===")
    print(f"No score (choice 0): {scores[0]:.6f}")
    print(f"Yes score (choice 1): {scores[1]:.6f}")
    predicted_choice = 0 if scores[0] > scores[1] else 1
    correct_choice = int(sample['answer'])  # True=1, False=0
    print(f"Predicted: {predicted_choice} ({'no' if predicted_choice == 0 else 'yes'})")
    print(f"Correct: {correct_choice} ({'no' if correct_choice == 0 else 'yes'})")
    print(f"Match: {predicted_choice == correct_choice}")
    print(f"Accuracy: {1.0 if predicted_choice == correct_choice else 0.0}")

if __name__ == "__main__":
    quick_test()
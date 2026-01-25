#!/usr/bin/env python3
"""
Direct comparison with ANE methodology to find the systematic bias
"""

import os
import sys
import torch
import numpy as np

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add paths
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from transformers import AutoTokenizer
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def test_inference_differences():
    """Test what could cause systematic bias"""
    
    # Load model
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
    
    print("Model loaded for bias analysis")
    
    # Test simple contexts that should favor "yes" vs "no"
    test_cases = [
        ("Is the sky blue?\nAnswer:", "yes", "no"),
        ("Is water wet?\nAnswer:", "yes", "no"),
        ("Is ice cold?\nAnswer:", "yes", "no"),
        ("Is fire hot?\nAnswer:", "yes", "no"),
        ("Do dogs bark?\nAnswer:", "yes", "no"),
    ]
    
    def score_token(context, token_text):
        """Score a single token"""
        context_tokens = tokenizer.encode(context, add_special_tokens=True)
        token_ids = tokenizer.encode(token_text, add_special_tokens=False)
        
        if len(token_ids) != 1:
            return float('-inf')
        
        pos = len(context_tokens) - 1
        last_token = torch.tensor([[context_tokens[-1]]], dtype=torch.long)
        
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
            
            if logits.dim() == 3:
                logits = logits[0, -1]
            elif logits.dim() == 2:
                logits = logits[-1]
            
            log_probs = torch.log_softmax(logits, dim=-1)
            return log_probs[token_ids[0]].item()
    
    print("Testing simple cases for systematic bias:")
    print("Context".ljust(30) + "Yes Score".ljust(12) + "No Score".ljust(12) + "Prediction".ljust(12) + "Expected")
    print("-" * 80)
    
    yes_wins = 0
    no_wins = 0
    
    for context, expected_yes, expected_no in test_cases:
        yes_score = score_token(context, f" {expected_yes}")
        no_score = score_token(context, f" {expected_no}")
        
        predicted = "yes" if yes_score > no_score else "no"
        
        if predicted == "yes":
            yes_wins += 1
        else:
            no_wins += 1
        
        print(f"{context[:28]:30} {yes_score:8.3f}    {no_score:8.3f}    {predicted:12} {'yes':8}")
    
    print("-" * 80)
    print(f"Summary: Yes wins: {yes_wins}, No wins: {no_wins}")
    
    if no_wins > yes_wins:
        print("WARNING: Model shows systematic bias toward 'no' answers!")
        print("This could explain the poor BoolQ performance.")
    
    # Test if it's specifically about " yes" vs " no" tokens
    print(f"\nToken ID analysis:")
    yes_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode(" no", add_special_tokens=False)[0]
    print(f"' yes' token ID: {yes_id}")
    print(f"' no' token ID: {no_id}")
    
    # Test if tokenization affects this
    print(f"\nTokenization comparison:")
    alt_yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    alt_no_ids = tokenizer.encode("no", add_special_tokens=False)
    print(f"'yes' (no space): {alt_yes_ids}")
    print(f"'no' (no space): {alt_no_ids}")

if __name__ == "__main__":
    test_inference_differences()
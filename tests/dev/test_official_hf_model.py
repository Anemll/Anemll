#!/usr/bin/env python3
"""
Test the official HuggingFace model to see if it also has this bias
"""

import os
import sys
import torch

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM

def test_official_model():
    """Test official HF model for same bias"""
    
    model_path = "Qwen/Qwen2.5-0.5B"
    print("Loading official HuggingFace model...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    print("Official model loaded")
    
    def score_token_hf(context, token_text):
        """Score token using official HF model"""
        context_tokens = tokenizer.encode(context, add_special_tokens=True)
        full_text = context + token_text
        full_tokens = tokenizer.encode(full_text, add_special_tokens=True)
        
        # Get continuation tokens
        continuation_tokens = full_tokens[len(context_tokens):]
        if len(continuation_tokens) != 1:
            return float('-inf')
        
        # Forward pass
        input_ids = torch.tensor([full_tokens])
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab]
        
        # Get log prob for the continuation token at the right position
        pos = len(context_tokens) - 1  # Position where we predict the continuation
        log_probs = torch.log_softmax(logits[pos], dim=-1)
        return log_probs[continuation_tokens[0]].item()
    
    # Test same cases
    test_cases = [
        ("Is the sky blue?\nAnswer:", "yes", "no"),
        ("Is water wet?\nAnswer:", "yes", "no"),
        ("Is ice cold?\nAnswer:", "yes", "no"),
        ("Is fire hot?\nAnswer:", "yes", "no"),
        ("Do dogs bark?\nAnswer:", "yes", "no"),
    ]
    
    print("Testing official HF model:")
    print("Context".ljust(30) + "Yes Score".ljust(12) + "No Score".ljust(12) + "Prediction".ljust(12) + "Expected")
    print("-" * 80)
    
    yes_wins = 0
    no_wins = 0
    
    for context, expected_yes, expected_no in test_cases:
        yes_score = score_token_hf(context, f" {expected_yes}")
        no_score = score_token_hf(context, f" {expected_no}")
        
        predicted = "yes" if yes_score > no_score else "no"
        
        if predicted == "yes":
            yes_wins += 1
        else:
            no_wins += 1
        
        print(f"{context[:28]:30} {yes_score:8.3f}    {no_score:8.3f}    {predicted:12} {'yes':8}")
    
    print("-" * 80)
    print(f"Official HF model - Yes wins: {yes_wins}, No wins: {no_wins}")
    
    if no_wins > yes_wins:
        print("Official model ALSO shows bias toward 'no' - this might be expected model behavior")
    else:
        print("Official model does NOT show bias - our implementation has a bug!")

if __name__ == "__main__":
    test_official_model()
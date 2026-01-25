#!/usr/bin/env python3
"""
Debug BoolQ format exactly as lm-eval processes it
"""

import os
import sys
import torch
import numpy as np
from datasets import load_dataset

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add paths
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

from transformers import AutoTokenizer
from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def debug_boolq_exact_format():
    """Debug using exact BoolQ format from lm-eval"""
    
    # Load model (reuse from cache)
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
    
    # Load BoolQ sample
    dataset = load_dataset('boolq', split='validation')
    sample = dataset[0]
    
    print(f"Sample: {sample}")
    
    # Use EXACT BoolQ format from task config
    # doc_to_text='{{passage}}\nQuestion: {{question}}?\nAnswer:'
    # doc_to_choice=['no', 'yes']  
    # target_delimiter=' '
    
    context = f"{sample['passage']}\nQuestion: {sample['question']}?\nAnswer:"
    
    # BoolQ choices: 0='no', 1='yes' with target_delimiter=' '
    choice_0 = " no"  # Choice index 0
    choice_1 = " yes"  # Choice index 1
    
    print(f"\nBoolQ exact format:")
    print(f"Context ends with: ...{context[-50:]}")
    print(f"Choice 0: '{choice_0}'")
    print(f"Choice 1: '{choice_1}'") 
    print(f"Ground truth label: {sample['answer']} (0=False/no, 1=True/yes)")
    
    # Tokenize
    context_tokens = tokenizer.encode(context, add_special_tokens=True)
    choice_0_tokens = tokenizer.encode(choice_0, add_special_tokens=False)
    choice_1_tokens = tokenizer.encode(choice_1, add_special_tokens=False)
    
    print(f"\nTokenization:")
    print(f"Context tokens: {len(context_tokens)}")
    print(f"Choice 0 tokens: {choice_0_tokens}")
    print(f"Choice 1 tokens: {choice_1_tokens}")
    
    def score_choice_simple(context_tokens, choice_tokens):
        """Simple scoring"""
        if len(choice_tokens) == 0:
            return 0.0
            
        # Just score the first token of the choice for simplicity
        target_token = choice_tokens[0]
        context_length = model.config.context_length
        
        # Ensure we fit in context
        if len(context_tokens) >= context_length:
            context_tokens = context_tokens[-(context_length-1):]
        
        # Get position and token
        pos = len(context_tokens) - 1
        last_token = torch.tensor([[context_tokens[-1]]], dtype=torch.long)
        
        # Create masks
        def make_causal_mask(length, start):
            mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
            row_indices = np.arange(length).reshape(length, 1)
            col_indices = np.arange(length).reshape(1, length)
            mask[:, :, col_indices <= (row_indices + start)] = 0
            return mask
        
        causal_mask_data = make_causal_mask(context_length, 0)
        causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
        
        update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
        if pos < context_length:
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
            
            next_token_logits = logits[0, -1, :]
            log_probs = torch.log_softmax(next_token_logits.float(), dim=-1)
            
            return log_probs[target_token].item()
    
    # Score both choices
    score_0 = score_choice_simple(context_tokens, choice_0_tokens)  # "no"
    score_1 = score_choice_simple(context_tokens, choice_1_tokens)  # "yes"
    
    print(f"\nScores:")
    print(f"Choice 0 (no): {score_0:.6f}")
    print(f"Choice 1 (yes): {score_1:.6f}")
    
    # Prediction logic: higher score wins
    predicted_choice = 0 if score_0 > score_1 else 1
    correct_choice = int(sample['answer'])  # True=1, False=0
    
    print(f"\nResults:")
    print(f"Predicted choice: {predicted_choice} ({'no' if predicted_choice == 0 else 'yes'})")
    print(f"Correct choice: {correct_choice} ({'no' if correct_choice == 0 else 'yes'})")
    print(f"Match: {predicted_choice == correct_choice}")
    print(f"Accuracy: {1.0 if predicted_choice == correct_choice else 0.0}")

if __name__ == "__main__":
    debug_boolq_exact_format()
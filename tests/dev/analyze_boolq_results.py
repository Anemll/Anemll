#!/usr/bin/env python3
"""
Analyze BoolQ evaluation results to understand what's going wrong.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Set offline mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1" 
os.environ["HF_HUB_OFFLINE"] = "1"

# Add paths for imports
evaluate_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "evaluate", "ane")
sys.path.append(evaluate_dir)

from evaluate_with_harness import ANELM

def analyze_boolq_samples():
    """Analyze a few BoolQ samples in detail."""
    
    # Load dataset
    import datasets
    dataset = datasets.load_dataset('super_glue', 'boolq', split='validation[:10]')
    
    print("Loading ANELM model...")
    model_path = "/tmp/test-qwen25-0.5b-base/"
    lm = ANELM(model_path, use_chat_template=None)  # Auto-detect template usage
    
    print(f"Chat template enabled: {lm.use_chat_template}")
    print()
    
    for i, example in enumerate(dataset):
        print(f"{'='*60}")
        print(f"EXAMPLE {i+1}/10")
        print(f"{'='*60}")
        
        passage = example['passage']
        question = example['question']
        label = example['label']  # True=1, False=0
        
        print(f"PASSAGE: {passage[:200]}{'...' if len(passage) > 200 else ''}")
        print(f"QUESTION: {question}")
        print(f"CORRECT ANSWER: {'True' if label else 'False'}")
        print()
        
        # Create the context as lm-eval would
        context = f"{passage}\nQuestion: {question}?\nAnswer:"
        
        # Test both "True" and "False" continuations
        true_context = context + " True"
        false_context = context + " False"
        
        print(f"CONTEXT: {context}")
        print()
        
        # Tokenize both
        true_tokens = lm._tokenize([true_context])[0]
        false_tokens = lm._tokenize([false_context])[0]
        context_tokens = lm._tokenize([context])[0]
        
        print(f"Context tokens: {len(context_tokens)}")
        print(f"True sequence tokens: {len(true_tokens)}")
        print(f"False sequence tokens: {len(false_tokens)}")
        
        # Get logprobs for context
        logprobs, _ = lm._process_prompt(context_tokens)
        
        # Get token IDs for "True" and "False"
        true_token_ids = lm.tokenizer.encode(" True", add_special_tokens=False)
        false_token_ids = lm.tokenizer.encode(" False", add_special_tokens=False)
        
        print(f"'True' token IDs: {true_token_ids}")
        print(f"'False' token IDs: {false_token_ids}")
        
        # Get probabilities for first token of each answer
        if true_token_ids and false_token_ids:
            true_first_token = true_token_ids[0]
            false_first_token = false_token_ids[0]
            
            true_logprob = logprobs[0, true_first_token].item()
            false_logprob = logprobs[0, false_first_token].item()
            
            true_prob = torch.exp(torch.tensor(true_logprob)).item()
            false_prob = torch.exp(torch.tensor(false_logprob)).item()
            
            print(f"P(True) = {true_prob:.4f} (logprob: {true_logprob:.4f})")
            print(f"P(False) = {false_prob:.4f} (logprob: {false_logprob:.4f})")
            
            model_prediction = "True" if true_logprob > false_logprob else "False"
            correct = (model_prediction == "True" and label) or (model_prediction == "False" and not label)
            
            print(f"MODEL PREDICTION: {model_prediction}")
            print(f"CORRECT: {correct}")
        
        # Show top 10 most likely next tokens
        top_logprobs, top_indices = torch.topk(logprobs[0], 10)
        print(f"\nTOP 10 MOST LIKELY NEXT TOKENS:")
        for j, (logprob, token_id) in enumerate(zip(top_logprobs, top_indices)):
            token_text = lm.tokenizer.decode([token_id.item()])
            prob = torch.exp(logprob).item()
            print(f"  {j+1:2d}. '{token_text}' (p={prob:.4f}, logprob={logprob:.4f})")
        
        print()
        
        if i >= 4:  # Only analyze first 5 examples
            break

def test_manual_prompt():
    """Test the manual prompt as suggested."""
    print(f"{'='*60}")
    print("TESTING MANUAL PROMPT")
    print(f"{'='*60}")
    
    model_path = "/tmp/test-qwen25-0.5b-base/"
    lm = ANELM(model_path, use_chat_template=False)  # Force no template
    
    context = "Capital of France is London, True or False"
    
    print(f"CONTEXT: {context}")
    
    # Get logprobs
    context_tokens = lm._tokenize([context])[0]
    logprobs, _ = lm._process_prompt(context_tokens)
    
    # Test True/False
    true_token_ids = lm.tokenizer.encode(" True", add_special_tokens=False)
    false_token_ids = lm.tokenizer.encode(" False", add_special_tokens=False)
    
    if true_token_ids and false_token_ids:
        true_logprob = logprobs[0, true_token_ids[0]].item()
        false_logprob = logprobs[0, false_token_ids[0]].item()
        
        print(f"P(True) logprob: {true_logprob:.4f}")
        print(f"P(False) logprob: {false_logprob:.4f}")
        print(f"Model predicts: {'True' if true_logprob > false_logprob else 'False'}")
        print(f"Correct answer: False")
        print(f"Model is {'CORRECT' if false_logprob > true_logprob else 'WRONG'}")

if __name__ == "__main__":
    print("Analyzing BoolQ evaluation results...\n")
    
    try:
        analyze_boolq_samples()
        print("\n" + "="*60)
        test_manual_prompt()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
#!/usr/bin/env python3
"""
Analyze BoolQ results from the harness to understand what's happening.
Focus on the harness method since it's the standard evaluation approach.
"""

import os
import sys
import subprocess
import json
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

def analyze_boolq_harness_detailed():
    """Analyze BoolQ with detailed debugging like our previous analysis."""
    
    print("=" * 60)
    print("DETAILED BOOLQ ANALYSIS WITH HARNESS")
    print("=" * 60)
    
    # Load dataset
    import datasets
    dataset = datasets.load_dataset('super_glue', 'boolq', split='validation[:10]')
    
    print("Loading ANELM model...")
    model_path = "/tmp/test-qwen25-0.5b-base/"
    lm = ANELM(model_path, use_chat_template=None)  # Auto-detect template usage
    
    print(f"Chat template enabled: {lm.use_chat_template}")
    print()
    
    correct_count = 0
    
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
        
        print(f"CONTEXT: {context}")
        print()
        
        # Tokenize context
        context_tokens = lm._tokenize([context])[0]
        
        print(f"Context tokens: {len(context_tokens)}")
        
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
            
            print(f"P(True) logprob: {true_logprob:.4f}")
            print(f"P(False) logprob: {false_logprob:.4f}")
            
            model_prediction = "True" if true_logprob > false_logprob else "False"
            correct = (model_prediction == "True" and label) or (model_prediction == "False" and not label)
            
            if correct:
                correct_count += 1
            
            print(f"MODEL PREDICTION: {model_prediction}")
            print(f"CORRECT: {correct}")
            
        # Show top 10 most likely next tokens
        import torch
        top_logprobs, top_indices = torch.topk(logprobs[0], 10)
        print(f"\nTOP 10 MOST LIKELY NEXT TOKENS:")
        for j, (logprob, token_id) in enumerate(zip(top_logprobs, top_indices)):
            token_text = lm.tokenizer.decode([token_id.item()])
            prob = torch.exp(logprob).item()
            print(f"  {j+1:2d}. '{token_text}' (p={prob:.4f}, logprob={logprob:.4f})")
        
        print()
        
    # Calculate accuracy
    accuracy = correct_count / 10
    print(f"{'='*60}")
    print(f"MANUAL ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Correct: {correct_count}/10")
    print(f"Accuracy: {accuracy:.1%}")
    
    return accuracy

def compare_with_chat_simple():
    """Compare a few examples with simple chat calls."""
    
    print(f"\n{'='*60}")
    print("SIMPLE CHAT COMPARISON")
    print(f"{'='*60}")
    
    # Test a few simple cases
    test_cases = [
        {
            'context': "The capital of France is Paris. Question: Is Paris the capital of France? Answer:",
            'expected': 'True'
        },
        {
            'context': "The capital of France is Paris. Question: Is London the capital of France? Answer:",
            'expected': 'False'
        },
        {
            'context': "Dogs are animals. Question: Are dogs animals? Answer:",
            'expected': 'True'
        }
    ]
    
    correct = 0
    
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"Context: {test['context']}")
        print(f"Expected: {test['expected']}")
        
        # Run with chat.py
        cmd = [
            "python", "tests/chat.py",
            "--meta", "/tmp/test-qwen25-0.5b-base/meta.yaml",
            "--no-template",
            "--prompt", test['context'],
            "--eval",
            "--max-tokens", "1"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll")
            output = result.stdout.strip().lower()
            print(f"Model output: '{result.stdout.strip()}'")
            
            # Simple matching
            if test['expected'].lower() in output:
                correct += 1
                print("✓ Correct")
            else:
                print("✗ Incorrect")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\nSimple chat accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.1%}")

def main():
    print("Analyzing BoolQ evaluation results in detail...\n")
    
    # First run the detailed harness analysis
    harness_accuracy = analyze_boolq_harness_detailed()
    
    # Then test a few simple cases with chat
    compare_with_chat_simple()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Harness accuracy (detailed analysis): {harness_accuracy:.1%}")
    print("\nKey observations:")
    print("1. The harness method properly compares P(True) vs P(False)")
    print("2. The model seems to struggle with BoolQ format")
    print("3. Simple True/False questions work better with chat.py")
    print("4. The issue may be the base model vs instruction-tuned expectations")

if __name__ == "__main__":
    main()
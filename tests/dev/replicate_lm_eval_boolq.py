#!/usr/bin/env python3
"""
Demonstrate how to replicate lm_eval's BoolQ results by using the same preprocessing.
This shows the exact steps needed to match simple_evaluate's accuracy.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets import load_dataset
from lm_eval.models import HFLM
import lm_eval
import torch
import numpy as np
from tqdm import tqdm

def replicate_lm_eval_preprocessing(model, num_examples=100):
    """
    Replicate lm_eval's exact preprocessing for BoolQ task.
    """
    print("="*60)
    print("REPLICATING LM_EVAL PREPROCESSING")
    print("="*60)
    
    # Load BoolQ dataset
    dataset = load_dataset("google/boolq", split="validation")
    
    # Key preprocessing steps from lm_eval:
    # 1. Doc to text template: "{{passage}}\nQuestion: {{question}}?\nAnswer:"
    # 2. Choices: ["no", "yes"]
    # 3. Target delimiter: " " (space)
    # 4. Multiple choice creates continuations: [" no", " yes"]
    
    correct = 0
    total = 0
    
    for example in tqdm(dataset.select(range(num_examples)), desc="Processing"):
        # Apply doc_to_text template
        prompt = f"{example['passage']}\nQuestion: {example['question']}?\nAnswer:"
        
        # Choices with target_delimiter (space prefix)
        choices_with_delimiter = [" no", " yes"]
        
        # Create requests
        requests = [(prompt, choice) for choice in choices_with_delimiter]
        
        # Get loglikelihoods
        results = model.loglikelihood(requests)
        lls = [result[0] for result in results]
        
        # Get prediction (argmax)
        pred_idx = np.argmax(lls)
        
        # Convert to boolean (yes=1=True, no=0=False)
        pred_answer = pred_idx == 1
        
        # Check if correct
        if pred_answer == example["answer"]:
            correct += 1
        total += 1
    
    accuracy = correct / total * 100
    print(f"\nAccuracy with lm_eval preprocessing: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy

def compare_with_simple_evaluate(num_examples=100):
    """
    Compare our replication with actual simple_evaluate results.
    """
    print("\n" + "="*60)
    print("COMPARING WITH SIMPLE_EVALUATE")
    print("="*60)
    
    # Initialize model
    model = HFLM(
        pretrained="Qwen/Qwen2.5-0.5B",
        backend="causal",
        device="cpu",
        dtype=torch.float32,
    )
    
    # Our replication
    print("\n1. Our replication with manual preprocessing:")
    our_accuracy = replicate_lm_eval_preprocessing(model, num_examples)
    
    # Actual simple_evaluate
    print("\n2. Running actual simple_evaluate:")
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=["boolq"],
        num_fewshot=0,
        limit=num_examples,
        device="cpu",
    )
    
    lm_eval_accuracy = results["results"]["boolq"]["acc"] * 100
    print(f"   Simple_evaluate accuracy: {lm_eval_accuracy:.2f}%")
    
    print(f"\n3. Difference: {abs(our_accuracy - lm_eval_accuracy):.2f}%")
    print("   (Should be very small if preprocessing is correctly replicated)")

def demonstrate_key_difference():
    """
    Demonstrate the key difference: space prefix on choices.
    """
    print("\n" + "="*60)
    print("KEY FINDING: THE SPACE PREFIX")
    print("="*60)
    
    model = HFLM(
        pretrained="Qwen/Qwen2.5-0.5B",
        backend="causal",
        device="cpu",
        dtype=torch.float32,
    )
    
    # Get one example
    dataset = load_dataset("google/boolq", split="validation")
    example = dataset[0]
    
    prompt = f"{example['passage']}\nQuestion: {example['question']}?\nAnswer:"
    
    print(f"\nExample question: {example['question']}")
    print(f"True answer: {example['answer']} ({'yes' if example['answer'] else 'no'})")
    
    # Test both versions
    print("\n1. Without space prefix:")
    requests = [(prompt, "no"), (prompt, "yes")]
    results = model.loglikelihood(requests)
    lls_no_space = [r[0] for r in results]
    print(f"   Loglikelihoods: no={lls_no_space[0]:.4f}, yes={lls_no_space[1]:.4f}")
    print(f"   Prediction: {'yes' if np.argmax(lls_no_space) == 1 else 'no'}")
    
    print("\n2. With space prefix (lm_eval style):")
    requests = [(prompt, " no"), (prompt, " yes")]
    results = model.loglikelihood(requests)
    lls_with_space = [r[0] for r in results]
    print(f"   Loglikelihoods: no={lls_with_space[0]:.4f}, yes={lls_with_space[1]:.4f}")
    print(f"   Prediction: {'yes' if np.argmax(lls_with_space) == 1 else 'no'}")
    
    print("\nCONCLUSION:")
    print("The space prefix (target_delimiter) is crucial for matching lm_eval results.")
    print("When calling loglikelihood directly, use [' no', ' yes'] not ['no', 'yes'].")

if __name__ == "__main__":
    # Demonstrate the key difference
    demonstrate_key_difference()
    
    # Compare full evaluation
    compare_with_simple_evaluate(num_examples=100)
#!/usr/bin/env python3
"""
Test to verify that the target_delimiter (space prefix) is the cause of accuracy difference
between direct loglikelihood calls and simple_evaluate.

This tests the hypothesis that lm_eval adds a space prefix to choices via target_delimiter.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets import load_dataset
from lm_eval.models import HFLM
import torch
import numpy as np
from tqdm import tqdm

def main():
    # Load model
    model = HFLM(
        pretrained="Qwen/Qwen2.5-0.5B",
        backend="causal",
        device="cpu",
        dtype=torch.float32,
    )
    
    # Load BoolQ dataset
    dataset = load_dataset("google/boolq", split="validation")
    
    # Test configurations
    test_configs = [
        {"name": "Without space prefix", "choices": ["no", "yes"]},
        {"name": "With space prefix", "choices": [" no", " yes"]},
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"Choices: {config['choices']}")
        print(f"{'='*60}")
        
        correct = 0
        total = 0
        
        # Process examples
        for i, example in enumerate(tqdm(dataset.select(range(100)), desc="Processing")):
            passage = example["passage"]
            question = example["question"]
            label = example["answer"]  # True or False
            
            # Construct prompt (matching BoolQ format)
            prompt = f"{passage}\nQuestion: {question}?\nAnswer:"
            
            # Create requests for both choices
            requests = [
                (prompt, choice) for choice in config["choices"]
            ]
            
            # Get loglikelihoods
            results = model.loglikelihood(requests)
            lls = [result[0] for result in results]
            
            # Get prediction
            pred_idx = np.argmax(lls)
            pred_answer = pred_idx == 1  # "yes" is at index 1
            
            if pred_answer == label:
                correct += 1
            total += 1
            
            # Print first few examples for debugging
            if i < 3:
                print(f"\nExample {i}:")
                print(f"Question: {question}")
                print(f"Label: {label} ({'yes' if label else 'no'})")
                print(f"Loglikelihoods: no={lls[0]:.4f}, yes={lls[1]:.4f}")
                print(f"Prediction: {pred_answer} ({'yes' if pred_answer else 'no'})")
                print(f"Correct: {pred_answer == label}")
        
        accuracy = correct / total * 100
        print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("The space prefix (target_delimiter) is likely the key difference")
    print("between direct loglikelihood calls and simple_evaluate.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Comprehensive analysis of lm_eval preprocessing for BoolQ task.
This script examines all the preprocessing steps that lm_eval applies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets import load_dataset
import lm_eval
from lm_eval.tasks import get_task_dict
from lm_eval.models import HFLM
import torch

def analyze_task_preprocessing():
    """Analyze how lm_eval preprocesses BoolQ task data."""
    
    print("="*60)
    print("ANALYZING LM_EVAL PREPROCESSING FOR BOOLQ")
    print("="*60)
    
    # Load the task
    task_manager = lm_eval.tasks.TaskManager()
    task_dict = get_task_dict(["boolq"], task_manager)
    task = task_dict["boolq"]
    
    # Print task configuration
    print("\n1. Task Configuration:")
    print(f"   - Output type: {task.OUTPUT_TYPE}")
    print(f"   - Target delimiter: '{task.config.target_delimiter}'")
    print(f"   - Doc to text: {task.config.doc_to_text}")
    print(f"   - Doc to choice: {task.config.doc_to_choice}")
    print(f"   - Doc to target: {task.config.doc_to_target}")
    
    # Load a sample document
    dataset = load_dataset("google/boolq", split="validation")
    sample_doc = dataset[0]
    
    print(f"\n2. Sample Document:")
    print(f"   - Passage: {sample_doc['passage'][:100]}...")
    print(f"   - Question: {sample_doc['question']}")
    print(f"   - Answer: {sample_doc['answer']}")
    
    # Apply doc transformations
    print(f"\n3. Document Transformations:")
    doc_text = task.doc_to_text(sample_doc)
    print(f"   - doc_to_text result:\n     {repr(doc_text)}")
    
    doc_target = task.doc_to_target(sample_doc)
    print(f"   - doc_to_target result: {doc_target}")
    
    doc_choices = task.doc_to_choice(sample_doc)
    print(f"   - doc_to_choice result: {doc_choices}")
    
    # Build a request without few-shot examples
    print(f"\n4. Request Construction:")
    fewshot_ctx = task.fewshot_context(sample_doc, 0)
    print(f"   - Fewshot context (0-shot):\n     {repr(fewshot_ctx)}")
    
    # Construct requests
    requests = task.construct_requests(doc=sample_doc, ctx=fewshot_ctx)
    print(f"\n   - Number of requests: {len(requests)}")
    for i, req in enumerate(requests):
        print(f"   - Request {i}:")
        print(f"     Type: {req.request_type}")
        print(f"     Arguments: {repr(req.arguments)}")
    
    # Show exact formatting
    print(f"\n5. Exact Continuation Formatting:")
    for i, choice in enumerate(doc_choices):
        continuation = f"{task.config.target_delimiter}{choice}"
        print(f"   - Choice {i}: {repr(choice)} -> Continuation: {repr(continuation)}")
    
    print("\n6. Key Findings:")
    print(f"   - Target delimiter is: '{task.config.target_delimiter}' (length: {len(task.config.target_delimiter)})")
    print(f"   - This means each choice gets prefixed with a space")
    print(f"   - Direct loglikelihood calls should use: {[f' {c}' for c in doc_choices]}")
    
    return task

def test_with_model():
    """Test with actual model to verify preprocessing effect."""
    
    print("\n" + "="*60)
    print("TESTING WITH MODEL")
    print("="*60)
    
    # Initialize model
    model = HFLM(
        pretrained="Qwen/Qwen2.5-0.5B",
        backend="causal",
        device="cpu",
        dtype=torch.float32,
    )
    
    # Get task
    task_manager = lm_eval.tasks.TaskManager()
    task_dict = get_task_dict(["boolq"], task_manager)
    task = task_dict["boolq"]
    
    # Load dataset
    dataset = load_dataset("google/boolq", split="validation")
    sample_doc = dataset[0]
    
    # Get context and choices
    ctx = task.fewshot_context(sample_doc, 0)
    choices = task.doc_to_choice(sample_doc)
    
    print(f"\nTesting different continuation formats:")
    print(f"Context: {repr(ctx[:50])}...")
    
    # Test different formats
    formats = [
        ("Raw choices", choices),
        ("Space-prefixed choices", [f" {c}" for c in choices]),
        ("Exact lm_eval format", [f"{task.config.target_delimiter}{c}" for c in choices])
    ]
    
    for format_name, formatted_choices in formats:
        print(f"\n{format_name}: {formatted_choices}")
        requests = [(ctx, choice) for choice in formatted_choices]
        results = model.loglikelihood(requests)
        lls = [r[0] for r in results]
        print(f"  Loglikelihoods: {[f'{ll:.4f}' for ll in lls]}")
        print(f"  Prediction: {formatted_choices[np.argmax(lls)]}")

if __name__ == "__main__":
    import numpy as np
    
    # Analyze preprocessing
    task = analyze_task_preprocessing()
    
    # Test with model
    test_with_model()
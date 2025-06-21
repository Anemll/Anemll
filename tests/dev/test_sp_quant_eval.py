#!/usr/bin/env python3
"""
Evaluation script for sparse quantized models on standard benchmarks.
Similar to lm_eval but designed for ANEMLL SP quantization testing.
"""

import os
import sys
import json
import torch
import argparse
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np

# Set environment for SP quantization
os.environ['ENABLE_SP_QUANT'] = '1'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download


class SimpleEvaluator:
    """Simple evaluator for language model benchmarks."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = 'cpu'  # ANEMLL models run on CPU
        
    def get_loglikelihood(self, context: str, continuation: str) -> float:
        """Calculate log-likelihood of continuation given context."""
        # Tokenize full sequence
        full_text = context + continuation
        full_ids = self.tokenizer.encode(full_text, return_tensors='pt')
        context_ids = self.tokenizer.encode(context, return_tensors='pt')
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits[0]  # Remove batch dimension
            
        # Calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Get log likelihood of continuation tokens
        continuation_start = len(context_ids[0]) - 1  # -1 because we predict from previous token
        continuation_end = len(full_ids[0]) - 1
        
        total_log_likelihood = 0.0
        for i in range(continuation_start, continuation_end):
            token_id = full_ids[0, i + 1]
            total_log_likelihood += log_probs[i, token_id].item()
            
        return total_log_likelihood
    
    def evaluate_multiple_choice(self, question: str, choices: List[str]) -> int:
        """Evaluate multiple choice question by selecting highest likelihood choice."""
        scores = []
        for choice in choices:
            score = self.get_loglikelihood(question, choice)
            scores.append(score)
        
        return int(np.argmax(scores))
    
    def evaluate_binary(self, prompt: str, true_option: str, false_option: str) -> bool:
        """Evaluate binary classification task."""
        true_score = self.get_loglikelihood(prompt, true_option)
        false_score = self.get_loglikelihood(prompt, false_option)
        return true_score > false_score


# Sample evaluation tasks
class BenchmarkTasks:
    """Simple benchmark tasks for testing."""
    
    @staticmethod
    def get_arc_easy_samples():
        """Get sample ARC-Easy questions."""
        return [
            {
                "question": "Which of these is a mammal?",
                "choices": ["Fish", "Bird", "Dog", "Lizard"],
                "answer": 2
            },
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris", "Rome"],
                "answer": 2
            },
            {
                "question": "How many days are in a week?",
                "choices": ["5", "6", "7", "8"],
                "answer": 2
            }
        ]
    
    @staticmethod
    def get_piqa_samples():
        """Get sample PIQA (Physical IQ) questions."""
        return [
            {
                "question": "To cut a piece of paper in half,",
                "choices": ["use scissors", "use a spoon"],
                "answer": 0
            },
            {
                "question": "To keep food cold,",
                "choices": ["put it in the oven", "put it in the refrigerator"],
                "answer": 1
            }
        ]
    
    @staticmethod
    def get_boolq_samples():
        """Get sample BoolQ questions."""
        return [
            {
                "question": "Is the sky blue? Answer:",
                "true_option": " yes",
                "false_option": " no",
                "answer": True
            },
            {
                "question": "Can fish fly? Answer:",
                "true_option": " yes",
                "false_option": " no",
                "answer": False
            }
        ]
    
    @staticmethod
    def get_winogrande_samples():
        """Get sample WinoGrande questions."""
        return [
            {
                "question": "The trophy doesn't fit in the suitcase because",
                "choices": ["the trophy is too big", "the suitcase is too big"],
                "answer": 0
            },
            {
                "question": "The ball broke the window because",
                "choices": ["the ball was soft", "the ball was hard"],
                "answer": 1
            }
        ]


def run_evaluation(model_path: str, tasks: List[str], batch_size: int = 1):
    """Run evaluation on specified tasks."""
    
    print(f"\n{'='*60}")
    print(f"  SP Quantization Model Evaluation")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Batch size: {batch_size}")
    print(f"ENABLE_SP_QUANT: {os.environ.get('ENABLE_SP_QUANT', 'NOT SET')}")
    
    # Load model
    print("\nLoading model...")
    if model_path.startswith("Qwen/") or "/" in model_path:
        # Download from HuggingFace
        local_path = snapshot_download(model_path)
    else:
        local_path = model_path
    
    # Load config and model
    config = Qwen25Config.from_json(f'{local_path}/config.json')
    model = Qwen25ForCausalLM(config, disable_kv_cache=False)
    model.load_pretrained_weights(local_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create evaluator
    evaluator = SimpleEvaluator(model, tokenizer)
    
    # Run evaluations
    results = {}
    
    if "arc_easy" in tasks:
        print("\n--- Evaluating ARC-Easy ---")
        samples = BenchmarkTasks.get_arc_easy_samples()
        correct = 0
        for sample in tqdm(samples, desc="ARC-Easy"):
            pred = evaluator.evaluate_multiple_choice(sample["question"], sample["choices"])
            if pred == sample["answer"]:
                correct += 1
        accuracy = correct / len(samples)
        results["arc_easy"] = {"accuracy": accuracy, "correct": correct, "total": len(samples)}
        print(f"ARC-Easy Accuracy: {accuracy:.2%} ({correct}/{len(samples)})")
    
    if "piqa" in tasks:
        print("\n--- Evaluating PIQA ---")
        samples = BenchmarkTasks.get_piqa_samples()
        correct = 0
        for sample in tqdm(samples, desc="PIQA"):
            pred = evaluator.evaluate_multiple_choice(sample["question"], sample["choices"])
            if pred == sample["answer"]:
                correct += 1
        accuracy = correct / len(samples)
        results["piqa"] = {"accuracy": accuracy, "correct": correct, "total": len(samples)}
        print(f"PIQA Accuracy: {accuracy:.2%} ({correct}/{len(samples)})")
    
    if "boolq" in tasks:
        print("\n--- Evaluating BoolQ ---")
        samples = BenchmarkTasks.get_boolq_samples()
        correct = 0
        for sample in tqdm(samples, desc="BoolQ"):
            pred = evaluator.evaluate_binary(
                sample["question"], 
                sample["true_option"], 
                sample["false_option"]
            )
            if pred == sample["answer"]:
                correct += 1
        accuracy = correct / len(samples)
        results["boolq"] = {"accuracy": accuracy, "correct": correct, "total": len(samples)}
        print(f"BoolQ Accuracy: {accuracy:.2%} ({correct}/{len(samples)})")
    
    if "winogrande" in tasks:
        print("\n--- Evaluating WinoGrande ---")
        samples = BenchmarkTasks.get_winogrande_samples()
        correct = 0
        for sample in tqdm(samples, desc="WinoGrande"):
            pred = evaluator.evaluate_multiple_choice(sample["question"], sample["choices"])
            if pred == sample["answer"]:
                correct += 1
        accuracy = correct / len(samples)
        results["winogrande"] = {"accuracy": accuracy, "correct": correct, "total": len(samples)}
        print(f"WinoGrande Accuracy: {accuracy:.2%} ({correct}/{len(samples)})")
    
    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for task, result in results.items():
        print(f"{task:15} | Accuracy: {result['accuracy']:6.2%} | {result['correct']}/{result['total']}")
    
    # Check quantization scales
    print(f"\n{'='*60}")
    print("QUANTIZATION CHECK")
    print(f"{'='*60}")
    scale_count = 0
    for name, module in model.named_modules():
        for attr in ['gate_proj_input_scale', 'up_proj_input_scale', 'down_proj_input_scale', 
                     'q_proj_input_scale', 'k_proj_input_scale', 'v_proj_input_scale', 'o_proj_input_scale']:
            if hasattr(module, attr):
                scale_count += 1
    
    print(f"Total quantization scales found: {scale_count}")
    if scale_count > 0:
        # Show sample scale values
        for name, module in model.named_modules():
            if hasattr(module, 'q_proj_input_scale'):
                scale = module.q_proj_input_scale
                print(f"Sample scale from {name}: {scale.flatten()[:3].tolist()}")
                break
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate sparse quantized models")
    parser.add_argument("--model", type=str, required=True,
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--tasks", type=str, default="arc_easy,piqa,boolq,winogrande",
                        help="Comma-separated list of tasks")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--no_quant", action="store_true",
                        help="Disable quantization for comparison")
    
    args = parser.parse_args()
    
    if args.no_quant:
        os.environ['SKIP_SP_FORWARD'] = '1'
        print("Running WITHOUT SP quantization (SKIP_SP_FORWARD=1)")
    
    tasks = [t.strip() for t in args.tasks.split(",")]
    run_evaluation(args.model, tasks, args.batch_size)


if __name__ == "__main__":
    # If no arguments provided, run a default test
    if len(sys.argv) == 1:
        print("Running default evaluation test...")
        # Use a small model for testing
        test_model = "Qwen/Qwen2.5-0.5B"  # Base model, not GPTQ
        test_tasks = ["arc_easy", "piqa", "boolq", "winogrande"]
        run_evaluation(test_model, test_tasks, batch_size=1)
    else:
        main()
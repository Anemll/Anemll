#!/usr/bin/env python3
"""
Test script for PyTorch segmented evaluation using lm-eval's native HuggingFace support.
This uses the built-in HFLM wrapper from lm-eval.
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Set offline mode to prevent network calls
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

# Add paths for imports
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')

from simple_evaluate_with_segmentation import simple_evaluate_segmented
from datasets import load_dataset
from lm_eval.models.huggingface import HFLM

def test_pytorch_multi_segment_evaluation(model_path, segment_size):
    """Test PyTorch multi-segment evaluation using native HFLM"""
    print("=" * 80)
    print(f"Testing PyTorch Multi-Segment Evaluation (segment size: {segment_size})")
    print("=" * 80)
    
    # Load dataset to get total size
    dataset = load_dataset('boolq', split='validation')
    total_size = len(dataset)
    print(f"Total dataset size: {total_size}")
    
    # Calculate segments
    print(f"Segment size: {segment_size}")
    
    # Since we want full dataset as one segment
    num_segments = (total_size + segment_size - 1) // segment_size
    print(f"Number of segments: {num_segments}")
    
    all_results = []
    
    for segment_idx in range(num_segments):
        start_idx = segment_idx * segment_size
        end_idx = min((segment_idx + 1) * segment_size, total_size)
        current_segment_size = end_idx - start_idx
        
        print(f"\nProcessing PyTorch segment [{start_idx}..{end_idx-1}] ({current_segment_size} samples)")
        
        # Use lm-eval's native HuggingFace wrapper
        print(f"Creating HFLM wrapper for: {model_path}")
        lm = HFLM(
            pretrained=model_path,
            device="cpu",
            batch_size=1
        )
        
        # Run segmented evaluation with full lm-eval preprocessing
        results = simple_evaluate_segmented(
            model=lm,
            tasks=["boolq"],
            segment_start=start_idx,
            segment_size=current_segment_size,
            total_dataset_size=total_size
        )
        
        # Extract results
        if "results" in results and "boolq" in results["results"]:
            segment_results = {
                "segment_idx": segment_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "segment_size": current_segment_size,
                "accuracy": results["results"]["boolq"]["acc"],
                "stderr": results["results"]["boolq"].get("acc_stderr", 0),
                "num_examples": current_segment_size
            }
            all_results.append(segment_results)
            print(f"Segment {segment_idx} accuracy: {segment_results['accuracy']:.4f} ± {segment_results['stderr']:.4f}")
        else:
            print(f"Warning: No results found for segment {segment_idx}")
    
    # Calculate overall accuracy
    if all_results:
        total_correct = sum(r["accuracy"] * r["num_examples"] for r in all_results)
        total_examples = sum(r["num_examples"] for r in all_results)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        print(f"\nOverall PyTorch Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"Total examples evaluated: {total_examples}")
        
        # Save results
        output_file = f"tests/dev/segmented_results/pytorch_hf_segmented_results_size_{segment_size}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        results_data = {
            "model": model_path,
            "segment_size": segment_size,
            "total_size": total_size,
            "overall_accuracy": overall_accuracy,
            "segments": all_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo results collected!")

def main():
    parser = argparse.ArgumentParser(description='Test PyTorch segmented evaluation with HFLM')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B',
                       help='Model path or HuggingFace ID')
    parser.add_argument('--segment-size', type=int, default=100,
                       help='Size of each segment')
    
    args = parser.parse_args()
    
    test_pytorch_multi_segment_evaluation(args.model, args.segment_size)

if __name__ == "__main__":
    main()
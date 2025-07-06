#!/usr/bin/env python3
"""
Test script for ANE segmented evaluation using the full lm-eval preprocessing pipeline.
This uses the existing ANE evaluation script with segmentation support.
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
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/evaluate/ane')

from simple_evaluate_with_segmentation import simple_evaluate_segmented
from datasets import load_dataset
from evaluate_with_harness import ANELM

def test_ane_multi_segment_evaluation(model_path, segment_size, task='boolq', limit=None, skip=0, debug=False, log_incorrect_answers=False):
    """Test ANE multi-segment evaluation using existing ANELM class"""
    print("=" * 80)
    print(f"Testing ANE Multi-Segment Evaluation for {task} (segment size: {segment_size})")
    print("=" * 80)
    
    # Load dataset to get total size based on task
    if task == 'boolq':
        dataset = load_dataset('boolq', split='validation')
    elif task == 'arc_easy':
        dataset = load_dataset('ai2_arc', 'ARC-Easy', split='test')
    else:
        # For other tasks, we'll use the default split from lm-eval
        try:
            dataset = load_dataset(task, split='test')
        except:
            dataset = load_dataset(task, split='validation')
    
    dataset_size = len(dataset)
    print(f"Total dataset size: {dataset_size}")
    
    # Apply skip
    if skip >= dataset_size:
        print(f"Error: Skip value {skip} is >= dataset size {dataset_size}")
        return
    
    if skip > 0:
        print(f"Skipping first {skip} examples")
    
    # Calculate effective total size after skip
    available_size = dataset_size - skip
    
    # Apply limit if specified
    if limit is not None:
        total_size = min(limit, available_size)
        print(f"Limited to: {total_size} examples (from {skip} to {skip + total_size - 1})")
    else:
        total_size = available_size
        print(f"Evaluating {total_size} examples (from {skip} to {skip + total_size - 1})")
    
    # Calculate segments
    print(f"Segment size: {segment_size}")
    
    # Calculate number of segments based on (possibly limited) total size
    num_segments = (total_size + segment_size - 1) // segment_size
    print(f"Number of segments: {num_segments}")
    
    all_results = []
    
    for segment_idx in range(num_segments):
        # Calculate segment boundaries with skip offset
        segment_start_offset = segment_idx * segment_size
        segment_end_offset = min((segment_idx + 1) * segment_size, total_size)
        current_segment_size = segment_end_offset - segment_start_offset
        
        # Apply skip to get actual dataset indices
        actual_start = skip + segment_start_offset
        actual_end = skip + segment_end_offset
        
        print(f"\nProcessing ANE segment [{actual_start}..{actual_end-1}] ({current_segment_size} samples)")
        
        # Use existing ANELM class
        print(f"Creating ANELM wrapper for: {model_path}")
        lm = ANELM(
            model_path=model_path,
            max_tokens=2048,
            use_chat_template=False,
            verbose_output=debug,
            log_incorrect_answers=log_incorrect_answers,
            skip=actual_start  # Pass the actual start index for absolute question numbering
        )
        
        # Run segmented evaluation with full lm-eval preprocessing
        results = simple_evaluate_segmented(
            model=lm,
            tasks=[task],
            segment_start=actual_start,
            segment_size=current_segment_size,
            total_dataset_size=dataset_size,
            limit=None  # Don't pass limit here since we handle it with segment boundaries
        )
        
        # Extract results
        if "results" in results and task in results["results"]:
            task_results = results["results"][task]
            print(f"Available metrics for segment {segment_idx}: {list(task_results.keys())}")
            
            # Try different possible accuracy metric names
            accuracy_key = None
            for key in ["acc", "acc,none", "accuracy", "acc_norm", "acc_norm,none"]:
                if key in task_results:
                    accuracy_key = key
                    break
            
            if accuracy_key:
                segment_results = {
                    "segment_idx": segment_idx,
                    "start_idx": actual_start,
                    "end_idx": actual_end,
                    "segment_size": current_segment_size,
                    "accuracy": task_results[accuracy_key],
                    "stderr": task_results.get(f"{accuracy_key}_stderr", 0),
                    "num_examples": current_segment_size,
                    "accuracy_metric": accuracy_key
                }
                all_results.append(segment_results)
                print(f"Segment {segment_idx} accuracy ({accuracy_key}): {segment_results['accuracy']:.4f} ± {segment_results['stderr']:.4f}")
            else:
                print(f"Warning: No accuracy metric found for segment {segment_idx}. Available: {list(task_results.keys())}")
        else:
            print(f"Warning: No results found for segment {segment_idx}")
            if "results" in results:
                print(f"Available tasks: {list(results['results'].keys())}")
    
    # Calculate overall accuracy
    if all_results:
        total_correct = sum(r["accuracy"] * r["num_examples"] for r in all_results)
        total_examples = sum(r["num_examples"] for r in all_results)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        print(f"\nOverall ANE Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"Total examples evaluated: {total_examples}")
        
        # Save results
        output_file = f"tests/dev/segmented_results/ane_{task}_segmented_results_size_{segment_size}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        results_data = {
            "model": model_path,
            "task": task,
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
    parser = argparse.ArgumentParser(description='Test ANE segmented evaluation with existing ANELM')
    parser.add_argument('--model', type=str, default='/tmp/test-qwen25-0.5b-base2',
                       help='Path to ANE model directory')
    parser.add_argument('--task', type=str, default='boolq',
                       help='Task to evaluate (e.g., boolq, arc_easy, hellaswag)')
    parser.add_argument('--segment-size', type=int, default=100,
                       help='Size of each segment')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit total number of examples to evaluate (for testing)')
    parser.add_argument('--skip', type=int, default=0,
                       help='Skip first N examples and start evaluation from sample N')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output with detailed prompt information')
    parser.add_argument('--log-incorrect-answers', action='store_true',
                       help='Log detailed information for incorrect answers including context, options, selected answer, and correct answer')
    
    args = parser.parse_args()
    
    test_ane_multi_segment_evaluation(args.model, args.segment_size, args.task, args.limit, args.skip, args.debug, args.log_incorrect_answers)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test script to verify our copied simple_evaluate function works correctly
and then test segmentation.
"""

import argparse
import os
import sys

# Set offline mode to prevent network calls
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

# Add current directory to path
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev')

import mlx.core as mx
from mlx_true_official_copy import MLXLM, chat_template_fn
from simple_evaluate_with_segmentation import simple_evaluate_segmented
from datasets import load_dataset

def test_full_evaluation(model_path):
    """Test our copied simple_evaluate on full dataset"""
    print("=" * 80)
    print("Testing Full Evaluation (should match 62.26%)")
    print("=" * 80)
    
    # Initialize model
    lm = MLXLM(
        model_path,
        max_tokens=2048,
        use_chat_template=False,
    )
    MLXLM.apply_chat_template = chat_template_fn(**{})
    
    # Get total dataset size
    dataset = load_dataset("boolq", split="validation")
    total_size = len(dataset)
    print(f"Total dataset size: {total_size}")
    
    # Run full evaluation using our copied function
    results = simple_evaluate_segmented(
        model=lm,
        tasks=["boolq"],
        segment_start=None,  # None means evaluate all
        segment_size=None,
        total_dataset_size=total_size,
        random_seed=123,
        numpy_random_seed=123,
        torch_random_seed=123,
        fewshot_random_seed=123
    )
    
    # Extract accuracy
    accuracy = results["results"]["boolq"]["acc,none"]
    print(f"\nFull evaluation accuracy: {accuracy:.4f}")
    print(f"Expected: 0.6226 (62.26%)")
    print(f"Match: {'✓' if abs(accuracy - 0.6226) < 0.001 else '✗'}")
    
    return accuracy

def test_single_segment_evaluation(model_path):
    """Test segmented evaluation with all samples as one segment"""
    print("\n" + "=" * 80)
    print("Testing Single Segment (all samples) - should match full evaluation")
    print("=" * 80)
    
    # Initialize model
    lm = MLXLM(
        model_path,
        max_tokens=2048,
        use_chat_template=False,
    )
    MLXLM.apply_chat_template = chat_template_fn(**{})
    
    # Get total dataset size
    dataset = load_dataset("boolq", split="validation")
    total_size = len(dataset)
    
    # Run single segment evaluation (all samples)
    results = simple_evaluate_segmented(
        model=lm,
        tasks=["boolq"],
        segment_start=0,
        segment_size=total_size,
        total_dataset_size=total_size,
        random_seed=123,
        numpy_random_seed=123,
        torch_random_seed=123,
        fewshot_random_seed=123
    )
    
    # Extract accuracy
    accuracy = results["results"]["boolq"]["acc,none"]
    print(f"\nSingle segment accuracy: {accuracy:.4f}")
    print(f"Expected: 0.6226 (62.26%)")
    print(f"Match: {'✓' if abs(accuracy - 0.6226) < 0.001 else '✗'}")
    
    return accuracy

def test_multi_segment_evaluation(model_path, segment_size=100):
    """Test segmented evaluation with multiple segments"""
    print(f"\n" + "=" * 80)
    print(f"Testing Multi-Segment Evaluation (segment size: {segment_size})")
    print("=" * 80)
    
    # Create output files
    import json
    from pathlib import Path
    output_dir = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/tests/dev/segmented_results")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / f"mlx_segmented_results_size_{segment_size}.json"
    summary_file = output_dir / f"mlx_segmented_summary_size_{segment_size}.tsv"
    
    # Get total dataset size
    dataset = load_dataset("boolq", split="validation")
    total_size = len(dataset)
    
    print(f"Total dataset size: {total_size}")
    print(f"Segment size: {segment_size}")
    print(f"Number of segments: {(total_size + segment_size - 1) // segment_size}")
    
    segment_results = []
    overall_correct = 0
    overall_total = 0
    
    for start_idx in range(0, total_size, segment_size):
        current_segment_size = min(segment_size, total_size - start_idx)
        end_idx = start_idx + current_segment_size - 1
        
        print(f"\nProcessing segment [{start_idx}..{end_idx}] ({current_segment_size} samples)")
        
        # Initialize fresh model for each segment
        lm = MLXLM(
            model_path,
            max_tokens=2048,
            use_chat_template=False,
        )
        MLXLM.apply_chat_template = chat_template_fn(**{})
        
        # Run segmented evaluation
        results = simple_evaluate_segmented(
            model=lm,
            tasks=["boolq"],
            segment_start=start_idx,
            segment_size=current_segment_size,
            total_dataset_size=total_size,
            random_seed=123,
            numpy_random_seed=123,
            torch_random_seed=123,
            fewshot_random_seed=123
        )
        
        # Extract results
        accuracy = results["results"]["boolq"]["acc,none"]
        # Note: the "acc,none" value from segmented evaluation is already the accuracy for that segment
        # We need to track raw counts for overall accuracy calculation
        
        # Calculate raw correct count (accuracy * segment_size)
        correct_count = round(accuracy * current_segment_size)
        
        segment_results.append({
            'start': start_idx,
            'end': end_idx,
            'size': current_segment_size,
            'accuracy': accuracy,
            'correct': correct_count
        })
        
        overall_correct += correct_count
        overall_total += current_segment_size
        
        print(f"  Segment accuracy: {accuracy:.4f} ({correct_count}/{current_segment_size})")
    
    # Calculate overall accuracy
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    
    print(f"\n" + "=" * 80)
    print("Multi-Segment Results Summary")
    print("=" * 80)
    print(f"Overall accuracy: {overall_accuracy:.4f} ({overall_correct}/{overall_total})")
    print(f"Expected: 0.6226 (62.26%)")
    print(f"Match: {'✓' if abs(overall_accuracy - 0.6226) < 0.01 else '✗'}")
    
    # Show worst performing segments
    segment_results.sort(key=lambda x: x['accuracy'])
    print(f"\nWorst 5 segments:")
    for seg in segment_results[:5]:
        print(f"  [{seg['start']}..{seg['end']}]: {seg['accuracy']:.4f} ({seg['correct']}/{seg['size']})")
    
    # Save detailed results to JSON
    detailed_results = {
        'model_path': model_path,
        'segment_size': segment_size,
        'total_samples': overall_total,
        'overall_accuracy': overall_accuracy,
        'overall_correct': overall_correct,
        'segments': segment_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Save summary TSV
    with open(summary_file, 'w') as f:
        f.write("start\tend\tsize\taccuracy\tcorrect\n")
        for seg in segment_results:
            f.write(f"{seg['start']}\t{seg['end']}\t{seg['size']}\t{seg['accuracy']:.4f}\t{seg['correct']}\n")
    print(f"Summary TSV saved to: {summary_file}")
    
    return overall_accuracy, segment_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="MLX model path")
    parser.add_argument("--test", choices=["full", "single", "multi", "all"], default="all",
                       help="Which test to run")
    parser.add_argument("--segment-size", type=int, default=100, 
                       help="Segment size for multi-segment test")
    
    args = parser.parse_args()
    
    # Set random seed
    mx.random.seed(123)
    
    if args.test in ["full", "all"]:
        test_full_evaluation(args.model)
    
    if args.test in ["single", "all"]:
        test_single_segment_evaluation(args.model)
        
    if args.test in ["multi", "all"]:
        test_multi_segment_evaluation(args.model, args.segment_size)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Standalone MLX batch segment evaluation for BoolQ.
Runs all segments and exports results for comparison with ANE.
"""

import argparse
import json
from pathlib import Path
from mlx_lm.utils import load
from datasets import load_dataset
import mlx.core as mx
import mlx.nn as nn

def _pad_inputs(inputs):
    """Pad inputs to same length (from official MLX)"""
    import numpy as np
    lengths = np.array([len(x) for x in inputs])
    maxlen = lengths.max()
    padded = np.stack(
        [np.pad(x, (0, maxlen - len(x))) for x in inputs],
        axis=0,
    )
    return mx.array(padded), mx.array(lengths)

def _process_prompt(model, prompt, step_size=2048):
    """Process prompt and return cached logprobs (from official MLX)"""
    from mlx_lm.models.cache import make_prompt_cache
    
    prompt = mx.array(prompt)[None]
    cache = make_prompt_cache(model)
    for i in range(0, prompt.shape[1], step_size):
        logits = model(prompt[:, i : i + step_size], cache=cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()
    logprobs = nn.log_softmax(logits[:, -1, :].astype(mx.float32))
    return logprobs, cache

def _score_fn(model, inputs, cache=None, step_size=2048):
    """Score function matching official MLX exactly"""
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.base import create_causal_mask
    import copy
    
    inputs, lengths = _pad_inputs(inputs)
    inputs, targets = inputs[..., :-1], inputs[..., 1:]

    cache = cache or make_prompt_cache(model)
    lengths += cache[0].offset

    scores, is_greedy = [], []
    for i in range(0, inputs.shape[1], step_size):
        inp = inputs[:, i : i + step_size]
        T = inp.shape[1]

        offset = cache[0].offset
        mask = create_causal_mask(T, offset, lengths=lengths)

        logits = model(inp, cache=cache, mask=mask)
        log_probs = nn.log_softmax(logits.astype(mx.float32))

        score = mx.take_along_axis(
            log_probs, targets[:, i : i + step_size, mx.newaxis], axis=-1
        )[..., 0]
        ig = targets[:, i : i + step_size] == mx.argmax(logits, axis=-1)
        ig = mx.where(mx.arange(T) + offset < lengths[:, None], ig, False)

        mx.eval(score, ig)
        mx.clear_cache()

        is_greedy.append(ig)
        scores.append(score)

    scores = mx.concatenate(scores, axis=1)
    is_greedy = mx.concatenate(is_greedy, axis=1)

    return scores, lengths, is_greedy

def evaluate_mlx_segment(model, tokenizer, dataset, skip, limit):
    """Evaluate MLX model using official MLX evaluation logic exactly"""
    import copy
    
    correct = 0
    total = 0
    
    for i in range(skip, skip + limit):
        if i >= len(dataset):
            break
            
        item = dataset[i]
        question = item['question']
        passage = item['passage']
        ground_truth = item['answer']  # True/False
        
        # Format like BoolQ evaluation
        context = f"{passage}\nQuestion: {question}\nAnswer:"
        prefix = tokenizer.encode(context, add_special_tokens=True)
        
        # Define answer continuations
        continuations = [" no", " yes"]
        full_sequences = [tokenizer.encode(context + cont, add_special_tokens=True) for cont in continuations]
        
        # Process context and get cached logprobs (official MLX approach)
        try:
            logprobs, cache = _process_prompt(model, prefix)
            max_idx = mx.argmax(logprobs).item()
            
            scores = []
            is_greedy = []
            
            for s in full_sequences:
                inputs = s[len(prefix):]  # Get continuation tokens only
                
                # Score first continuation token from cached context logprobs
                scores.append(logprobs[0, inputs[0]].item())
                is_greedy.append((inputs[0] == max_idx))
                
                if len(inputs) == 1:
                    continue
                    
                # Score remaining continuation tokens using _score_fn with cache
                score, _, ig = _score_fn(model, mx.array(inputs)[None, :], cache=copy.deepcopy(cache))
                scores[-1] += mx.sum(score).item()
                is_greedy[-1] &= mx.all(ig).item()
        except Exception as e:
            # Fallback to simple method if official method fails
            print(f"Official method failed: {e}, falling back to simple scoring")
            scores = []
            for continuation in continuations:
                continuation_tokens = tokenizer.encode(continuation, add_special_tokens=False)
                full_tokens = prefix + continuation_tokens
                full_mx = mx.array([full_tokens])
                model.eval()
                logits = model(full_mx)
                
                log_likelihood = 0.0
                for j, token_id in enumerate(continuation_tokens):
                    position = len(prefix) + j - 1
                    if position >= 0 and position < logits.shape[1]:
                        token_logits = logits[0, position, :]
                        log_probs = nn.log_softmax(token_logits, axis=-1)
                        token_log_prob = log_probs[token_id].item()
                        log_likelihood += token_log_prob
                scores.append(log_likelihood)
        
        # Choose answer with highest log-likelihood
        mlx_prediction = "yes" if scores[1] > scores[0] else "no"  # [" no", " yes"]
        ground_truth_str = "yes" if ground_truth else "no"
        
        if mlx_prediction == ground_truth_str:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

def main():
    parser = argparse.ArgumentParser(description="MLX batch segment evaluation for BoolQ")
    parser.add_argument("--mlx-model", default="Qwen/Qwen2.5-0.5B", help="MLX model path (default: Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--step", type=int, default=200, help="Segment size")
    parser.add_argument("--total", "--limit", type=int, default=None, help="Total samples to evaluate (auto-detect if not provided)")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N samples and start evaluation from sample N")
    parser.add_argument("--worst", type=int, default=5, help="Number of worst segments to report")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Loading MLX model: {args.mlx_model}")
    model, tokenizer = load(args.mlx_model)
    
    print(f"Loading BoolQ dataset...")
    dataset = load_dataset("google/boolq", split="validation")
    
    # Determine total samples and apply skip
    dataset_size = len(dataset)
    if args.skip >= dataset_size:
        print(f"Error: Skip value {args.skip} is >= dataset size {dataset_size}")
        return
        
    if args.total is None:
        total_samples = dataset_size - args.skip
    else:
        total_samples = min(args.total, dataset_size - args.skip)
    
    print(f"Total BoolQ validation examples: {dataset_size}")
    if args.skip > 0:
        print(f"Skipping first {args.skip} samples")
    print(f"Evaluating {total_samples} samples (from {args.skip} to {args.skip + total_samples - 1})")
    print(f"Segment size: {args.step} examples; reporting {args.worst} worst segments")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    segment_results = []
    all_results = {"segments": []}
    
    # Process segments
    for segment_offset in range(0, total_samples, args.step):
        actual_skip = args.skip + segment_offset
        limit = min(args.step, total_samples - segment_offset)
        end = actual_skip + limit - 1
        
        print(f"Processing MLX segment [{actual_skip} .. {end}]")
        
        accuracy, correct, total_count = evaluate_mlx_segment(model, tokenizer, dataset, actual_skip, limit)
        
        segment_data = {
            "start": actual_skip,
            "end": end,
            "accuracy": accuracy,
            "correct": correct,
            "total": total_count
        }
        
        segment_results.append(segment_data)
        all_results["segments"].append(segment_data)
        
        print(f"  MLX: {accuracy:.4f} ({correct}/{total_count})")
        
        # Save individual segment result in format compatible with batch script
        segment_json = {
            "boolq": {
                "alias": "boolq", 
                "acc,none": accuracy,
                "acc_stderr,none": 0.0  # Placeholder
            }
        }
        
        segment_file = output_dir / f"mlx_window_{actual_skip}_{end}.json"
        segment_file.write_text(json.dumps(segment_json, indent=4))
    
    # Calculate overall statistics
    total_correct = sum(s["correct"] for s in segment_results)
    total_samples_evaluated = sum(s["total"] for s in segment_results)
    overall_accuracy = total_correct / total_samples_evaluated if total_samples_evaluated > 0 else 0
    
    all_results["overall"] = {
        "accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_samples": total_samples_evaluated
    }
    
    # Save summary files
    summary_tsv = output_dir / "mlx_boolq_segments_summary.tsv"
    with open(summary_tsv, 'w') as f:
        f.write("start\tend\tacc\n")
        for s in segment_results:
            f.write(f"{s['start']}\t{s['end']}\t{s['accuracy']}\n")
    
    # Save complete results
    results_json = output_dir / f"mlx_batch_results_{args.mlx_model.replace('/', '_')}.json"
    results_json.write_text(json.dumps(all_results, indent=4))
    
    # Print analysis
    print("\n" + "="*80)
    print("MLX Batch Evaluation Results")
    print("="*80)
    print(f"Overall MLX Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_samples_evaluated})")
    
    # Sort segments by accuracy
    sorted_segments = sorted(segment_results, key=lambda x: x['accuracy'])
    
    print(f"\nWorst {args.worst} MLX segments by accuracy (lowest first):")
    for i, seg in enumerate(sorted_segments[:args.worst]):
        print(f"{seg['start']}\t{seg['end']}\t{seg['accuracy']:.4f}")
    
    print(f"\nBest {args.worst} MLX segments by accuracy (highest first):")
    for i, seg in enumerate(sorted_segments[-args.worst:]):
        print(f"{seg['start']}\t{seg['end']}\t{seg['accuracy']:.4f}")
    
    print(f"\nResults saved to:")
    print(f"  Summary TSV: {summary_tsv}")
    print(f"  Complete results: {results_json}")
    print(f"  Individual segments: {output_dir}/mlx_window_*.json")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# Full perplexity evaluation with standard datasets
# Usage: python full_perplexity.py --dataset wikitext --chunk-size 400 --max-chunks 100

import os
import sys
import argparse
import math
import time
import signal
import json
import datetime
import torch
from tqdm import tqdm
from pathlib import Path

# Add current directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import ANE_Model
try:
    from ane_model import ANE_Model
except ImportError:
    print(f"Error: Failed to import ane_model from {current_dir}")
    sys.exit(1)

# Import tokenizer
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers package not found. Please install with: pip install transformers")
    sys.exit(1)

# Import dataset loading
try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets package not found. Please install with: pip install datasets")
    sys.exit(1)

# IMPORTANT NOTE ON STATE MANAGEMENT:
# When scoring tokens with CoreML models, we need to be careful about state mutations.
# For each position:
# 1. Read logits using compute_logprobs(current) to score target token
# 2. Write ground-truth target token using predict(target) to advance the state properly
# This ensures each position in the KV-cache contains the correct token with no duplicates
# or skipped positions.

# Set up timeout handler for prediction calls
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Prediction timeout")

# Load standard datasets
def load_wikitext(split="validation", subset_size=None, wikitext_version="wikitext-2"):
    version_name = "wikitext-2-raw-v1" if wikitext_version == "wikitext-2" else "wikitext-103-raw-v1"
    print(f"Loading {wikitext_version} {split} dataset...")
    dataset = load_dataset("wikitext", version_name, split=split)
 
#def load_wikitext(split="validation", subset_size=None):
#    print(f"Loading Wikitext-2 {split} dataset...")
#    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    # Use only a subset if specified
    if subset_size and subset_size > 0:
        dataset = dataset.select(range(min(subset_size, len(dataset))))
        print(f"Using only first {subset_size} examples")
    
    text = "\n".join(dataset["text"])
    return text

# A sample text for testing
SAMPLE_TEXT = """
The Apple Neural Engine (ANE) represents a specialized hardware implementation designed to accelerate neural network operations on Apple devices. By offloading computations from the CPU and GPU to a dedicated neural processing unit, the ANE significantly improves the efficiency of machine learning tasks, enabling complex models to run with lower power consumption and higher speed.

Language models present unique challenges in neural network design. Unlike computer vision tasks, which process spatial data, language models must process sequential data with complex dependencies across time. This led to the development of recurrent neural networks (RNNs), which maintain an internal state that can capture information from previous timesteps.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Full perplexity evaluation with standard datasets")
    parser.add_argument("--model", type=str, 
                       default=os.path.expandvars("$HOME/Models/ANE/anemll-Llama-3.2-1B-FP16-b64-ctx1024"),
                       help="Path to model directory")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug output")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Standard dataset to use (wikitext)")
    parser.add_argument("--wikitext-version", type=str, default="wikitext-2",
                       choices=["wikitext-2", "wikitext-103"],
                       help="WikiText version to use")
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split to use (test, validation, etc.)")
    parser.add_argument("--subset-size", type=int, default=100,
                       help="Number of examples to use from dataset (0 for all)")
    parser.add_argument("--chunk-size", type=int, default=256,
                       help="Size of each chunk in tokens")
    parser.add_argument("--max-chunks", type=int, default=10,
                       help="Maximum number of chunks to process")
    parser.add_argument("--prediction-timeout", type=int, default=10,
                       help="Timeout in seconds for each prediction call")
    parser.add_argument("--output-dir", type=str, default="evaluate/results",
                       help="Directory to save results")
    return parser.parse_args()

def get_model_name(model_path):
    """Extract a clean model name from the path."""
    path = Path(model_path)
    return path.name or path.parent.name

def main():
    args = parse_args()
    start_time = time.time()
    
    # Initialize model
    print(f"Loading model from {args.model}")
    model = ANE_Model(args.model)
    model.debug = 1 if args.debug else 0
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except:
        print("Falling back to Llama tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Use standard dataset if requested
    text = SAMPLE_TEXT
    dataset_name = "sample"
    if args.dataset == "wikitext":
        text = load_wikitext(args.split, args.subset_size, args.wikitext_version)
        dataset_name = f"{args.wikitext_version}-{args.split}"
        if args.subset_size:
            dataset_name += f"-{args.subset_size}examples"
        
    # Disable special tokens and manually add BOS if needed
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        tokens = [tokenizer.bos_token_id] + tokens
    print(f"Text tokenized to {len(tokens)} tokens")
    
    # Get context length from model name
    model_name = get_model_name(args.model)
    context_length = 1024  # Default to 1024
    if "ctx" in model_name:
        try:
            ctx_part = model_name.split("ctx")[1]
            context_length = int(ctx_part)
        except:
            pass
    print(f"Detected context length: {context_length}")
    
    # Create fixed-size chunks - ensure chunks aren't too large
    max_safe_chunk_size = context_length - 64  # Leave 64 tokens as safety margin
    chunk_size = min(args.chunk_size, max_safe_chunk_size)
    print(f"Using chunk size of {chunk_size} tokens (max safe: {max_safe_chunk_size})")
    
    chunks = []
    # Use non-overlapping chunks for cleaner evaluation
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        chunks.append(chunk)
        if len(chunks) >= args.max_chunks:
            break
    
    # Create a result structure to track all details
    result = {
        "model": get_model_name(args.model),
        "model_path": str(args.model),
        "dataset": dataset_name,
        "date": datetime.datetime.now().isoformat(),
        "parameters": {
            "chunk_size": chunk_size,
            "max_chunks": args.max_chunks,
            "prediction_timeout": args.prediction_timeout,
            "subset_size": args.subset_size
        },
        "chunks": [],
        "total_tokens": 0,
        "tokens_processed": 0,
        "perplexity": None,
        "duration_seconds": None
    }
    
    print(f"Processing {len(chunks)} chunks of size ~{chunk_size}")
    
    # Process each chunk
    total_log_likelihood = 0.0
    total_tokens = 0
    
    # Set up signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    
    # Create a progress bar for chunks
    pbar = tqdm(total=len(chunks), desc="Processing chunks", unit="chunk")
    
    for i, chunk in enumerate(chunks):
        chunk_start_time = time.time()
        
        if args.debug:
            print(f"\n--- Processing Chunk {i+1}/{len(chunks)} ---")
            print(f"Chunk has {len(chunk)} tokens")
        
        # Skip if chunk is too small
        if len(chunk) < 5:
            print(f"Skipping chunk {i+1} - too small ({len(chunk)} tokens)")
            pbar.update(1)
            continue
        
        # Initialize chunk result data
        chunk_result = {
            "index": i+1,
            "size": len(chunk),
            "tokens_processed": 0,
            "perplexity": None,
            "duration_seconds": None
        }
        
        # Calculate midpoint - only score second half of chunk for purer perplexity
        mid = len(chunk) // 2
        # Ensure we have enough tokens to score (at least 1)
        if mid < 1:
            mid = 1
        # Fix the sliding-window boundary: leave the last prefix token out of prefill
        prefix   = chunk[:mid-1]        # prefill   → rows 0 … k‑2
        currents = chunk[mid-1:-1]      # current   → T(k‑1) … T(n‑1)
        targets  = chunk[mid:]          # target    → T(k) …  T(n)
        
        if args.debug:
            print(f"Prefill: first {len(prefix)} tokens, Score: next {len(targets)} tokens (of {len(chunk)} total)")
        
        # Reset model and run prefill
        model.reset_state()  # Explicit reset for each chunk
        try:
            input_tensor = torch.tensor([prefix], dtype=torch.int32)
            _ = model.prefill(input_tensor)
        except Exception as e:
            print(f"Error during prefill: {str(e)}")
            pbar.update(1)
            continue
        
        # Score tokens
        chunk_scores = []
        
        # Create a progress bar for tokens if debugging
        token_pbar = None
        if args.debug:
            token_pbar = tqdm(total=len(targets), desc=f"Chunk {i+1} tokens", unit="token", leave=False)
        
        for j, (current, target) in enumerate(zip(currents, targets)):
            token_tensor = torch.tensor([[current]], dtype=torch.int32)
            
            # Set timeout for prediction
            signal.setitimer(signal.ITIMER_REAL, args.prediction_timeout)
            try:
                # 1) Read logits
                log_probs = model.compute_logprobs(token_tensor)
                signal.setitimer(signal.ITIMER_REAL, 0)  # Cancel timer
                
                if log_probs is not None:
                    score = log_probs[target].item()
                    chunk_scores.append(score)
                    
                    # 2) Write ground-truth token and advance
                    target_tensor = torch.tensor([[target]], dtype=torch.int32)
                    model.predict(target_tensor)
                    
                    # Show first few tokens in debug mode
                    if args.debug and j < 3:
                        curr_text = tokenizer.decode([current])
                        tgt_text = tokenizer.decode([target])
                        print(f"Token {j}: '{curr_text}' → '{tgt_text}', score: {score:.4f}")
                        print(f"  Current position after predict: {model.current_position}")
                
            except TimeoutError:
                print(f"\nTimeout during token {j} prediction. Skipping to next chunk.")
                break
            except Exception as e:
                print(f"\nError processing token {j}: {str(e)}")
                if j > 0:  # If we've processed at least one token, continue to next
                    continue
                else:  # Otherwise skip the chunk
                    break
                
            # Update token progress bar
            if token_pbar:
                token_pbar.update(1)
                
            # Only process a reasonable number of tokens per chunk to avoid slowdowns
            if j >= (chunk_size - 32) and len(chunk_scores) > 0:
                if args.debug:
                    print(f"Processed {j+1} tokens, moving to next chunk")
                break
        
        # Close token progress bar
        if token_pbar:
            token_pbar.close()
        
        # Calculate chunk perplexity
        if chunk_scores:
            chunk_ll = sum(chunk_scores) / len(chunk_scores)
            chunk_ppl = math.exp(-chunk_ll)
            
            # Update chunk results
            chunk_result["tokens_processed"] = len(chunk_scores)
            chunk_result["perplexity"] = chunk_ppl
            chunk_result["duration_seconds"] = time.time() - chunk_start_time
            
            if i % 2 == 0 or args.debug:  # Print every 2nd chunk or in debug mode
                print(f"Chunk {i+1} perplexity: {chunk_ppl:.4f} (tokens: {len(chunk_scores)})")
                # Sanity check: tokens in second half should be ~chunk_size/2
                expected_tokens = len(targets)
                if len(chunk_scores) > 0 and abs(len(chunk_scores) - expected_tokens) > 10:
                    print(f"  Note: Expected ~{expected_tokens} tokens scored, got {len(chunk_scores)}")
            
            # Add to totals
            total_log_likelihood += sum(chunk_scores)
            total_tokens += len(chunk_scores)
        else:
            print(f"Chunk {i+1}: No tokens were successfully scored")
            
        # Add chunk result to results
        result["chunks"].append(chunk_result)
        
        # Update chunk progress bar
        pbar.update(1)
        if total_tokens > 0:
            pbar.set_postfix(ppl=f"{math.exp(-total_log_likelihood/total_tokens):.4f}")
    
    # Close chunk progress bar
    pbar.close()
    
    # Calculate overall perplexity
    if total_tokens > 0:
        avg_ll = total_log_likelihood / total_tokens
        perplexity = math.exp(-avg_ll)
        
        # Update results
        result["perplexity"] = perplexity
        result["total_tokens"] = len(tokens)
        result["tokens_processed"] = total_tokens
        result["duration_seconds"] = time.time() - start_time
        
        print(f"\nFull evaluation results:")
        print(f"Dataset: {args.dataset if args.dataset else 'custom'}")
        print(f"Perplexity: {perplexity:.4f}")
        print(f"Total tokens scored: {total_tokens}")
        print(f"Chunks processed: {len(chunks)}")
    else:
        print("Error: No tokens were successfully scored")
    
    # Save results to file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model_name = get_model_name(args.model)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"perplexity_{model_name}_{dataset_name}_c{chunk_size}_n{args.max_chunks}_{timestamp}.json"
    
    output_path = output_dir / filename
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
if __name__ == "__main__":
    main() 
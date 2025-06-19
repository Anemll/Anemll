#!/usr/bin/env python3
# Simple perplexity test for ANE models
# This script demonstrates the chunking approach for perplexity calculation
# Usage: python simple_perplexity.py [--model MODEL_PATH]

import os
import sys
import argparse
import math
import torch
from pathlib import Path

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import ANE_Model
try:
    from ane_model import ANE_Model
except ImportError:
    print(f"Error: Failed to import ane_model from {current_dir}")
    sys.exit(1)

# Small sample text for testing
SAMPLE_TEXT = """
The Apple Neural Engine (ANE) is a specialized hardware component designed to accelerate machine learning tasks on Apple devices.
By offloading these computations to dedicated hardware, the ANE enables faster and more efficient processing of neural networks
while consuming less power than traditional CPU or GPU implementations.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Simple perplexity test for ANE models")
    parser.add_argument("--model", type=str, 
                       default=os.path.expandvars("$HOME/Models/ANE/anemll-Llama-3.2-1B-FP16-b64-ctx1024"),
                       help="Path to model directory")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    parser.add_argument("--chunk-size", type=int, default=100,
                       help="Size of each chunk in tokens")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize model
    print(f"Loading model from {args.model}")
    model = ANE_Model(args.model)
    model.debug = 1 if args.debug else 0
    
    # Use simple encoding for testing
    print("Encoding sample text")
    # Create a simple vocabulary for testing (ASCII characters)
    vocab = {chr(i): i for i in range(32, 127)}
    
    # Tokenize text (character-level for simplicity)
    text = SAMPLE_TEXT
    tokens = [vocab.get(c, 0) for c in text if c in vocab]
    print(f"Text encoded to {len(tokens)} tokens")
    
    # Create chunks
    chunk_size = args.chunk_size
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        chunks.append(chunk)
    
    print(f"Processing {len(chunks)} chunks of size {chunk_size}")
    
    # Process each chunk
    total_log_likelihood = 0.0
    total_tokens = 0
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Processing Chunk {i+1}/{len(chunks)} ---")
        print(f"Chunk has {len(chunk)} tokens")
        
        # Split into inputs and targets
        inputs, targets = chunk[:-1], chunk[1:]
        
        # Reset model for each chunk
        model.reset_state()
        print(f"Reset model state before chunk {i+1}")
        
        # Convert to tensor
        input_tensor = torch.tensor([inputs], dtype=torch.int32)
        
        # Run prefill for chunk
        print(f"Running prefill for chunk {i+1}")
        _ = model.prefill(input_tensor)
        
        # Process each token pair
        chunk_scores = []
        
        for j, (current, target) in enumerate(zip(inputs, targets)):
            # Create token tensor
            token_tensor = torch.tensor([[current]], dtype=torch.int32)
            
            # Get log probabilities
            log_probs = model.compute_logprobs(token_tensor)
            
            if log_probs is not None and target < len(log_probs):
                score = log_probs[target].item()
                chunk_scores.append(score)
                
                # Debug first few tokens
                if j < 5:
                    curr_char = chr(current) if 32 <= current < 127 else f"<{current}>"
                    tgt_char = chr(target) if 32 <= target < 127 else f"<{target}>"
                    print(f"Token {j}: '{curr_char}' → '{tgt_char}', score: {score:.4f}")
            
            # Update state for next token
            if j < len(inputs) - 1:
                _ = model.predict(token_tensor)
        
        # Calculate chunk perplexity
        if chunk_scores:
            chunk_log_likelihood = sum(chunk_scores)
            chunk_ppl = math.exp(-chunk_log_likelihood / len(chunk_scores))
            print(f"Chunk {i+1} perplexity: {chunk_ppl:.4f} (tokens: {len(chunk_scores)})")
            
            # Add to totals
            total_log_likelihood += chunk_log_likelihood
            total_tokens += len(chunk_scores)
        else:
            print(f"Chunk {i+1}: No tokens were scored")
    
    # Calculate overall perplexity
    if total_tokens > 0:
        perplexity = math.exp(-total_log_likelihood / total_tokens)
        print(f"\nOverall perplexity: {perplexity:.4f}")
        print(f"Total tokens scored: {total_tokens}")
    else:
        print("\nError: No tokens were scored")
    
if __name__ == "__main__":
    main()

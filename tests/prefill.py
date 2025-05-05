#!/usr/bin/env python3
# prefill.py
# Copyright (c) 2025 Anemll
# Licensed under the MIT License

import argparse
import os
import re
import glob
from pathlib import Path
import coremltools as ct
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import time
import yaml
import sys

# ANSI color codes
LIGHT_BLUE = "\033[94m"
DARK_BLUE = "\033[34m"
LIGHT_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

def parse_model_path(path):
    """Parse model path and return full path with .mlmodelc or .mlpackage extension."""
    path = Path(path)
    
    # If path exists exactly as specified, return it
    if path.exists():
        return str(path)
        
    # Try with both extensions
    candidates = [
        path,  # Original path
        path.with_suffix('.mlmodelc'),  # With .mlmodelc
        path.with_suffix('.mlpackage'),  # With .mlpackage
        Path(str(path) + '.mlmodelc'),  # Handle case where extension is included
        Path(str(path) + '.mlpackage')
    ]
    
    # Try all possible paths
    for candidate in candidates:
        if candidate.exists():
            print(f"Found model at: {candidate}")
            return str(candidate)
            
    # If we get here, no valid path was found
    print("\nError: Model not found. Tried following paths:")
    for candidate in candidates:
        print(f"  {candidate}")
    raise FileNotFoundError(f"Model not found: {path}")

def parse_ffn_filename(path):
    """Parse FFN model filename to extract chunk information."""
    path = Path(path)
    pattern = r'FFN_PF.*_chunk_(\d+)of(\d+)'
    match = re.search(pattern, path.name)
    
    if match:
        current_chunk = int(match.group(1))
        total_chunks = int(match.group(2))
        return current_chunk, total_chunks
    return None, None

def find_all_chunks(base_path):
    """Find all chunk files matching the base FFN path pattern."""
    path = Path(base_path)
    pattern = re.sub(r'_chunk_\d+of\d+', '_chunk_*', str(path))
    return sorted(glob.glob(pattern))

def load_model(path, function_name=None):
    """Load a CoreML model, handling both .mlmodelc and .mlpackage formats."""
    path = Path(path)
    compute_unit = ct.ComputeUnit.CPU_AND_NE
    
    try:
        if path.suffix == '.mlmodelc':
            # For compiled models (.mlmodelc), use CompiledMLModel
            if function_name:
                return ct.models.CompiledMLModel(str(path), compute_unit, function_name=function_name)
            else:
                return ct.models.CompiledMLModel(str(path), compute_unit)
        else:
            # For packages (.mlpackage)
            if function_name:
                return ct.models.MLModel(str(path), function_name=function_name)
            else:
                return ct.models.MLModel(str(path))
                
    except RuntimeError as e:
        if "valid manifest does not exist" in str(e):
            print(f"\nError: Could not load compiled model at {path}")
            print("This might be because:")
            print("1. The model is not properly compiled")
            print("2. The model was compiled for a different OS version")
            print("3. The model needs to be recompiled")
            print("\nTry using the .mlpackage version instead, or recompile the model.")
        raise

def load_metadata(model, args):
    # Extract metadata and config parameters
    metadata = {}
    if hasattr(model, 'user_defined_metadata'):
        meta = model.user_defined_metadata
        
        # Extract key parameters with defaults
        metadata['context_length'] = int(meta.get('com.anemll.context_length', 512))
        metadata['state_length'] = int(meta.get('com.anemll.state_length', metadata['context_length']))
        metadata['batch_size'] = int(meta.get('com.anemll.batch_size', 64))
        metadata['lut_bits'] = int(meta.get('com.anemll.lut_bits', 0))
        metadata['num_chunks'] = int(meta.get('com.anemll.num_chunks', 1))
        
        print("\nExtracted Parameters:")
        print(f"  Context Length: {metadata['context_length']}")
        print(f"  State Length: {metadata['state_length']}")
        print(f"  Prefill Batch Size: {metadata['batch_size']}")
        print(f"  LUT Bits: {metadata['lut_bits']}")
        print(f"  Number of Chunks: {metadata['num_chunks']}")
    else:
        print("\nWarning: No metadata found in model")

        # Check if model directory name contains context length pattern (ctxXXX)
        ctx_len = 512
        if args.context_length is None:
            import re
            ctx_match = re.search(r'ctx(\d+)', str(args.d))
            if ctx_match:
                ctx_len0 = int(ctx_match.group(1))
                if 512 <= ctx_len0 <= 8096:
                    ctx_len = ctx_len0
                    print(f"\nDetected context length {ctx_len} from directory name")
            else:
                print(f"\nWarning: No context length found in directory, using default {ctx_len}")
        else:
            ctx_len = args.context_length

        # Use defaults or values from args
        metadata['context_length'] = ctx_len
        metadata['state_length'] = ctx_len
        # Get batch size from args or use default
        metadata['batch_size'] = getattr(args, 'batch_size', 64)
        metadata['lut_bits'] = 4
        metadata['num_chunks'] = getattr(args, 'num_chunks', 4)
        print("\nUsing parameters:")
        print(f"  Context Length: {metadata['context_length']}")
        print(f"  State Length: {metadata['state_length']}")
        print(f"  Prefill Batch Size: {metadata['batch_size']}")
        print(f"  LUT Bits: {metadata['lut_bits']}")
        print(f"  Number of Chunks: {metadata['num_chunks']}")

    # Override with values from args if they exist
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        metadata['batch_size'] = args.batch_size
        print(f"\nOverriding batch size from args: {args.batch_size}")
    if hasattr(args, 'num_chunks') and args.num_chunks is not None:
        metadata['num_chunks'] = args.num_chunks
        print(f"\nOverriding num chunks from args: {args.num_chunks}")
    
    return metadata
    
def load_models(args, metadata):
    """Load all required models and extract metadata."""
    print("\nLoading models...")
    
    try:
        # Load embeddings model
        print("\nLoading embeddings model...")
        embed_path = parse_model_path(args.embed)
        print(f"Loading from: {embed_path}")
        embed_model = load_model(embed_path)
        print("Embeddings model loaded successfully")
        metadata = load_metadata(embed_model, args)
        
        # Load FFN model(s)
        print("\nLoading PREFILL functionality only...")
        ffn_path = parse_model_path(args.ffn)
        chunk_no, total_chunks = parse_ffn_filename(ffn_path)
        
        ffn_models = []
        if chunk_no and total_chunks:
            print(f"\nDetected chunked model with {total_chunks} chunks")
            # Find and load all chunks
            chunk_paths = find_all_chunks(ffn_path)
            if len(chunk_paths) != total_chunks:
                raise ValueError(f"Found {len(chunk_paths)} chunks but filename indicates {total_chunks} chunks")
                
            for chunk_path in chunk_paths:
                print(f"\nLoading PREFILL function from chunk: {Path(chunk_path).name}")
                try:
                    # For prefill testing, we only need the prefill function
                    prefill_model = load_model(chunk_path, function_name='prefill')
                    ffn_models.append(prefill_model)
                    print("Chunk loaded successfully (prefill only)")
                except Exception as e:
                    print(f"Error loading chunk {chunk_path}: {str(e)}")
                    raise
            metadata = load_metadata(ffn_models[0], args)
        else:
            print("\nLoading single model (prefill functionality only)...")
            ffn_models.append(load_model(ffn_path))
            print("Model loaded successfully")
        
        return embed_model, ffn_models, metadata
        
    except Exception as e:
        print(f"\nError loading models: {str(e)}")
        print("\nPlease ensure all model files exist and are accessible.")
        print("Expected files:")
        print(f"  Embeddings: {args.embed}")
        print(f"  FFN: {args.ffn}")
        raise

def initialize_tokenizer(model_path=None):
    """Initialize and configure the tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), 
            use_fast=False,
            trust_remote_code=True
        )
        
        print("\nTokenizer Configuration:")
        print(f"Tokenizer type: {type(tokenizer)}")
        print(f"Tokenizer name: {tokenizer.__class__.__name__}")
        print(f"Vocabulary size: {len(tokenizer)}")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("Set PAD token to EOS token")
        
        tokenizer.padding_side = "left"
        
        return tokenizer
        
    except Exception as e:
        print(f"\nError: Failed to load tokenizer from {model_path}")
        print(f"Error details: {str(e)}")
        raise

def make_causal_mask(length, start):
    """Create causal attention mask."""
    mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
    row_indices = np.arange(length).reshape(length, 1)
    col_indices = np.arange(length).reshape(1, length)
    mask[:, :, col_indices <= (row_indices + start)] = 0
    return mask

def initialize_causal_mask(context_length):
    """Initialize causal mask for transformer attention."""
    causal_mask = make_causal_mask(context_length, 0)
    causal_mask = torch.tensor(causal_mask, dtype=torch.float16)
    print(f"\nInitialized causal mask for context length {context_length}")
    return causal_mask

def run_prefill(embed_model, ffn_models, input_ids, context_pos, context_length, batch_size=64, state=None, causal_mask=None):
    """Run prefill on the input sequence."""
    # Use provided causal mask or create one if not provided
    if causal_mask is None:
        causal_mask = make_causal_mask(context_length, 0)
        causal_mask = torch.tensor(causal_mask, dtype=torch.float16)
    
    # Process in batches
    batch_pos = 0
    while batch_pos < context_pos:
        batch_end = min(batch_pos + batch_size, context_pos)
        current_batch_size = batch_end - batch_pos
        
        # Get current batch
        batch_input = input_ids[:, batch_pos:batch_end]
        
        # Always pad to full batch size for prefill
        batch_input = F.pad(
            batch_input,
            (0, batch_size - current_batch_size),
            value=0
        )
        
        # Generate position IDs for full batch size
        position_ids = torch.arange(batch_size, dtype=torch.int32)
        batch_causal_mask = causal_mask[:, :, :batch_size, :]
        
        # Run embeddings with proper batch size
        hidden_states = torch.from_numpy(
            embed_model.predict({
                'input_ids': batch_input.numpy(),
                'batch_size': np.array([batch_size], dtype=np.int32)
            })['hidden_states']
        )
        
        # Run through FFN chunks with state
        for ffn_model in ffn_models:
            # Handle both direct model and dictionary formats
            if isinstance(ffn_model, dict) and 'prefill' in ffn_model:
                # For backward compatibility with dictionary format
                prefill_model = ffn_model['prefill']
            else:
                # Direct access for models loaded with function_name='prefill'
                prefill_model = ffn_model
                
            inputs = {
                'hidden_states': hidden_states.numpy(),
                'position_ids': position_ids.numpy(),
                'causal_mask': batch_causal_mask.numpy(),
                'current_pos': np.array([batch_pos], dtype=np.int32)
            }
            output = prefill_model.predict(inputs, state)
            hidden_states = torch.from_numpy(output['output_hidden_states'])
        
        batch_pos = batch_end
    
    return torch.tensor([context_pos], dtype=torch.int32)

def create_unified_state(ffn_models, context_length):
    """Create unified KV cache state for transformer."""
    if hasattr(ffn_models[0], 'make_state'):
        # Direct access for models loaded with 'prefill' function_name
        state = ffn_models[0].make_state()
        print(f"\nCreated unified transformer state for {len(ffn_models)} chunks")
        return state
    else:
        # Fallback for dictionary-based models (for backward compatibility)
        if isinstance(ffn_models[0], dict) and 'prefill' in ffn_models[0]:
            state = ffn_models[0]['prefill'].make_state()
            print(f"\nCreated unified transformer state for {len(ffn_models)} chunks")
            return state
        else:
            state = ffn_models[0].make_state()
            print("\nCreated unified transformer state")
            return state

def test_prefill_speed(embed_model, ffn_models, tokenizer, batch_size, context_length, num_test_tokens, num_runs=20, test_single_chunk=True):
    """Test prefill speed with sample token sequences."""
    print(f"\n{LIGHT_GREEN}Testing prefill speed for {num_test_tokens} tokens (using internal batch size {batch_size}){RESET_COLOR}")
    print(f"Running {num_runs} iterations for warmup and measurement")

    # Create sample input sequence of exactly num_test_tokens tokens
    sample_text = "This is a test sequence. " * ((num_test_tokens + 4) // 5) # Ensure enough text
    input_ids = tokenizer(sample_text, return_tensors="pt").input_ids.to(torch.int32)

    # Trim or pad to exactly num_test_tokens tokens
    if input_ids.size(1) > num_test_tokens:
        input_ids = input_ids[:, :num_test_tokens]
    elif input_ids.size(1) < num_test_tokens:
        pad_length = num_test_tokens - input_ids.size(1)
        input_ids = F.pad(input_ids, (0, pad_length), value=tokenizer.pad_token_id)

    print(f"Sample input sequence length: {input_ids.size(1)} tokens")

    # Test with all chunks first
    print(f"\n{LIGHT_BLUE}Testing with all chunks ({len(ffn_models)} chunks){RESET_COLOR}")
    
    # Create unified state
    state_all_chunks = create_unified_state(ffn_models, context_length)
    
    # Initialize causal mask
    causal_mask = initialize_causal_mask(context_length)
    
    # Run prefill multiple times for warmup and testing
    all_chunks_times = []
    for i in range(num_runs):
        # Reset state for each run
        if i == 0:
            print("\nStarting warmup runs...")
        elif i == num_runs // 2:
            print("\nWarmup complete, starting measurement runs...")
        
        start_time = time.time()
        
        # Run prefill
        run_prefill(
            embed_model,
            ffn_models,
            input_ids,
            input_ids.size(1),  # context_pos
            context_length,
            batch_size, # Internal batching within run_prefill
            state_all_chunks,
            causal_mask
        )
        
        elapsed = time.time() - start_time
        all_chunks_times.append(elapsed)
        
        # Print progress
        if i < num_runs // 2:  # Warmup phase
            print(f"Warmup run {i+1}/{num_runs//2}: {elapsed:.4f}s ({batch_size/elapsed:.1f} tokens/s)")
        else:  # Measurement phase
            print(f"Run {i+1-num_runs//2}/{num_runs//2}: {elapsed:.4f}s ({batch_size/elapsed:.1f} tokens/s)")
    
    # Calculate and report statistics for all chunks (excluding warmup runs)
    all_chunks_measurement_times = all_chunks_times[num_runs // 2:]
    all_chunks_avg_time = sum(all_chunks_measurement_times) / len(all_chunks_measurement_times)
    all_chunks_min_time = min(all_chunks_measurement_times)
    all_chunks_max_time = max(all_chunks_measurement_times)
    all_chunks_tokens_per_sec = num_test_tokens / all_chunks_avg_time # Use num_test_tokens for speed calculation
    
    print(f"\n{LIGHT_BLUE}All Chunks Prefill Speed Results:{RESET_COLOR}")
    print(f"Number of Chunks: {len(ffn_models)}")
    print(f"Test Tokens: {num_test_tokens} tokens")
    print(f"Internal Batch Size: {batch_size} tokens")
    print(f"Context Size: {context_length} tokens")
    print(f"Average Time: {all_chunks_avg_time:.4f}s")
    print(f"Min Time: {all_chunks_min_time:.4f}s")
    print(f"Max Time: {all_chunks_max_time:.4f}s")
    print(f"Average Speed: {all_chunks_tokens_per_sec:.1f} tokens/second")
    print(f"Best Speed: {num_test_tokens / all_chunks_min_time:.1f} tokens/second") # Use num_test_tokens
    
    # Test with single chunk if requested and if multiple chunks exist
    single_chunk_tokens_per_sec = 0
    if test_single_chunk and len(ffn_models) > 1:
        print(f"\n{LIGHT_BLUE}Testing with single chunk (first chunk only){RESET_COLOR}")
        
        # Create a list with only the first chunk
        single_chunk_model = [ffn_models[0]]
        
        # Create unified state for single chunk
        state_single_chunk = create_unified_state(single_chunk_model, context_length)
        
        # Run prefill multiple times for single chunk
        single_chunk_times = []
        for i in range(num_runs):
            if i == 0:
                print("\nStarting single chunk warmup runs...")
            elif i == num_runs // 2:
                print("\nSingle chunk warmup complete, starting measurement runs...")
            
            start_time = time.time()
            
            # Run prefill with single chunk
            run_prefill(
                embed_model,
                single_chunk_model,
                input_ids,
                input_ids.size(1),  # context_pos
                context_length,
                batch_size, # Internal batching within run_prefill
                state_single_chunk,
                causal_mask
            )
            
            elapsed = time.time() - start_time
            single_chunk_times.append(elapsed)
            
            # Print progress
            if i < num_runs // 2:  # Warmup phase
                print(f"Single chunk warmup run {i+1}/{num_runs//2}: {elapsed:.4f}s ({batch_size/elapsed:.1f} tokens/s)")
            else:  # Measurement phase
                print(f"Single chunk run {i+1-num_runs//2}/{num_runs//2}: {elapsed:.4f}s ({batch_size/elapsed:.1f} tokens/s)")
        
        # Calculate and report statistics for single chunk
        single_chunk_measurement_times = single_chunk_times[num_runs // 2:]
        single_chunk_avg_time = sum(single_chunk_measurement_times) / len(single_chunk_measurement_times)
        single_chunk_min_time = min(single_chunk_measurement_times)
        single_chunk_max_time = max(single_chunk_measurement_times)
        single_chunk_tokens_per_sec = num_test_tokens / single_chunk_avg_time # Use num_test_tokens
        
        print(f"\n{LIGHT_BLUE}Single Chunk Prefill Speed Results:{RESET_COLOR}")
        print(f"Test Tokens: {num_test_tokens} tokens")
        print(f"Internal Batch Size: {batch_size} tokens")
        print(f"Context Size: {context_length} tokens")
        print(f"Average Time: {single_chunk_avg_time:.4f}s")
        print(f"Min Time: {single_chunk_min_time:.4f}s")
        print(f"Max Time: {single_chunk_max_time:.4f}s")
        print(f"Average Speed: {single_chunk_tokens_per_sec:.1f} tokens/second")
        print(f"Best Speed: {num_test_tokens / single_chunk_min_time:.1f} tokens/second") # Use num_test_tokens
        
        # Calculate overhead per chunk
        if len(ffn_models) > 1:
            chunk_overhead = (all_chunks_avg_time - single_chunk_avg_time) / (len(ffn_models) - 1)
            print(f"\n{LIGHT_GREEN}Chunk Overhead Analysis:{RESET_COLOR}")
            print(f"Single Chunk Time: {single_chunk_avg_time:.4f}s")
            print(f"All Chunks Time ({len(ffn_models)} chunks): {all_chunks_avg_time:.4f}s")
            print(f"Additional Time Per Chunk: {chunk_overhead:.4f}s")
            print(f"Overhead Percentage: {(all_chunks_avg_time/single_chunk_avg_time - 1)*100:.1f}%")
    
    return all_chunks_tokens_per_sec, single_chunk_tokens_per_sec

def parse_args():
    parser = argparse.ArgumentParser(description='Test prefill speed with CoreML LLaMA models (c) 2025 Anemll')
    
    # Add meta.yaml option
    parser.add_argument('--meta', type=str, help='Path to meta.yaml to load all parameters')
    
    # Model paths
    parser.add_argument('--d', '--dir', type=str, default='.',
                       help='Directory containing model files (default: current directory)')
    parser.add_argument('--embed', type=str, required=False,
                       help='Path to embeddings model (relative to --dir)')
    parser.add_argument('--ffn', type=str, required=False,
                       help='Path to FFN model (can be chunked, relative to --dir)')
    parser.add_argument('--tokenizer', type=str, required=False,
                       help='Path to tokenizer')
    
    # Test configuration
    parser.add_argument('--batch-size', type=int,
                       help='Batch size for prefill test (default: 64)')
    parser.add_argument('--runs', type=int, default=20,
                       help='Number of test runs (default: 20)')
    parser.add_argument('--context-length', type=int,
                       help='Context length for the model')
    parser.add_argument('--no-single-chunk', action='store_true',
                       help='Disable single chunk testing')
    parser.add_argument('--test-tokens', type=int,
                       help='Number of tokens to use for the speed test (default: batch_size)')
    parser.add_argument('--test-full-context', action='store_true',
                       help='Test prefill speed using the full context length (overrides --test-tokens)')
    
    args = parser.parse_args()
    
    # If meta.yaml is provided, load parameters from it
    if args.meta:
        try:
            with open(args.meta, 'r') as f:
                meta = yaml.safe_load(f)
            params = meta['model_info']['parameters']
            
            # Set model directory to meta.yaml directory if not specified
            if not args.d or args.d == '.':
                args.d = str(Path(args.meta).parent)
            
            # Build model paths based on parameters
            prefix = params.get('model_prefix', 'llama')
            lut_ffn = f"_lut{params['lut_ffn']}" if params['lut_ffn'] != 'none' else ''
            lut_embeddings = f"_lut{params['lut_embeddings']}" if params['lut_embeddings'] != 'none' else ''
            num_chunks = int(params['num_chunks'])
            
            # Set model paths if not specified
            if not args.embed:
                args.embed = f'{prefix}_embeddings{lut_embeddings}'
            if not args.ffn:
                args.ffn = f'{prefix}_FFN_PF{lut_ffn}_chunk_01of{num_chunks:02d}'
            if not args.tokenizer:
                args.tokenizer = args.d
            
            # Set other parameters if not overridden by command line
            if args.context_length is None:
                args.context_length = int(params['context_length'])
            if args.batch_size is None:
                args.batch_size = int(params['batch_size'])
            args.num_chunks = num_chunks
            
            print(f"\nLoaded parameters from {args.meta}:")
            print(f"  Context Length: {args.context_length}")
            print(f"  Batch Size: {args.batch_size}")
            print(f"  Num Chunks: {args.num_chunks}")
            print(f"  Models Directory: {args.d}")
            print(f"  Embeddings: {args.embed}")
            print(f"  FFN: {args.ffn}")
            
        except Exception as e:
            print(f"\nError loading meta.yaml: {str(e)}")
            sys.exit(1)
    
    return args

def main():
    args = parse_args()
    
    # Use default batch size if not specified
    if args.batch_size is None:
        args.batch_size = 64
        print(f"\nUsing default batch size: {args.batch_size}")
    
    # Convert directory to absolute path
    model_dir = Path(args.d).resolve()
    if not model_dir.exists():
        print(f"\nError: Model directory not found: {model_dir}")
        return 1
        
    print(f"\nUsing model directory: {model_dir}")
    
    try:
        # Update paths to be relative to model directory
        args.embed = str(model_dir / args.embed)
        args.ffn = str(model_dir / args.ffn)
        
        # Handle tokenizer path separately
        if args.tokenizer is None:
            args.tokenizer = str(model_dir)
        
        if not Path(args.tokenizer).exists():
            print(f"\nError: Tokenizer directory not found: {args.tokenizer}")
            return 1
    
        args.tokenizer = str(Path(args.tokenizer).resolve())
        print(f"Using tokenizer path: {args.tokenizer}")
        
        # Load models and extract metadata
        metadata = {}
        embed_model, ffn_models, metadata = load_models(args, metadata)
        
        # Override context length from command line if provided
        if args.context_length is not None:
            metadata['context_length'] = args.context_length
            metadata['state_length'] = args.context_length
            print(f"\nOverriding context length from command line: {args.context_length}")
        
        # Load tokenizer
        tokenizer = initialize_tokenizer(args.tokenizer)
        if tokenizer is None:
            raise RuntimeError("Failed to initialize tokenizer")
        
        # Determine number of tokens for the test
        if args.test_full_context:
            num_test_tokens = metadata['context_length']
            print(f"\nTesting with full context length: {num_test_tokens} tokens")
        elif args.test_tokens is not None:
            num_test_tokens = args.test_tokens
            print(f"\nTesting with specified tokens: {num_test_tokens} tokens")
        else:
            num_test_tokens = args.batch_size # Default to batch size
            print(f"\nTesting with default tokens (batch size): {num_test_tokens} tokens")

        # Ensure test tokens do not exceed context length
        if num_test_tokens > metadata['context_length']:
            print(f"\nWarning: Requested test tokens ({num_test_tokens}) exceed context length ({metadata['context_length']}).")
            print(f"Clamping test tokens to context length.")
            num_test_tokens = metadata['context_length']

        # Run prefill speed test
        test_prefill_speed(
            embed_model=embed_model,
            ffn_models=ffn_models,
            tokenizer=tokenizer,
            batch_size=args.batch_size, # Pass original batch_size for run_prefill internal logic
            context_length=metadata['context_length'],
            num_test_tokens=num_test_tokens, # Pass the number of tokens to actually test
            num_runs=args.runs,
            test_single_chunk=not args.no_single_chunk
        )
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
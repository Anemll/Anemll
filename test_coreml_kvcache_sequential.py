#!/usr/bin/env python3
"""Test CoreML KV cache with sequential token processing and state management."""

import time
import torch
import numpy as np
import argparse
import os
import glob
from pathlib import Path
import coremltools as ct
from transformers import AutoTokenizer
import sys

# ANSI colors for output
LIGHT_BLUE = "\033[94m"
DARK_BLUE = "\033[34m"
LIGHT_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

def compile_model_if_needed(mlpackage_path):
    """Compile .mlpackage to .mlmodelc if needed and return the compiled path."""
    mlpackage_path = Path(mlpackage_path)
    mlmodelc_path = mlpackage_path.with_suffix('.mlmodelc')
    
    # If .mlmodelc exists and is newer than .mlpackage, use it
    if mlmodelc_path.exists():
        if mlpackage_path.exists():
            if mlmodelc_path.stat().st_mtime >= mlpackage_path.stat().st_mtime:
                print(f"Using existing compiled model: {mlmodelc_path}")
                return str(mlmodelc_path)
            else:
                print(f"Compiled model is older than package, recompiling...")
        else:
            print(f"Using existing compiled model: {mlmodelc_path}")
            return str(mlmodelc_path)
    
    # Need to compile
    if not mlpackage_path.exists():
        raise FileNotFoundError(f"Model package not found: {mlpackage_path}")
    
    print(f"Compiling {mlpackage_path} to {mlmodelc_path}...")
    
    # Remove existing .mlmodelc if it exists
    if mlmodelc_path.exists():
        import shutil
        shutil.rmtree(mlmodelc_path)
        print(f"Removed existing {mlmodelc_path}")
    
    # Compile using xcrun coremlcompiler
    import subprocess
    try:
        cmd = ["xcrun", "coremlcompiler", "compile", str(mlpackage_path), str(mlpackage_path.parent)]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Successfully compiled to {mlmodelc_path}")
        return str(mlmodelc_path)
    except subprocess.CalledProcessError as e:
        print(f"❌ Compilation failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        print(f"Falling back to .mlpackage")
        return str(mlpackage_path)

def parse_model_path(path):
    """Parse model path and return full path, preferring compiled .mlmodelc."""
    path = Path(path)
    
    # If path exists exactly as specified, return it
    if path.exists():
        if path.suffix == '.mlpackage':
            # Try to compile to .mlmodelc
            return compile_model_if_needed(path)
        return str(path)
        
    # Try with both extensions, prefer .mlpackage for compilation
    candidates = [
        path.with_suffix('.mlpackage'),
        path.with_suffix('.mlmodelc'),
        Path(str(path) + '.mlpackage'),
        Path(str(path) + '.mlmodelc'),
        path
    ]
    
    for candidate in candidates:
        if candidate.exists():
            print(f"Found model at: {candidate}")
            if candidate.suffix == '.mlpackage':
                # Try to compile to .mlmodelc
                return compile_model_if_needed(candidate)
            return str(candidate)
            
    print(f"\nError: Model not found. Tried: {candidates}")
    raise FileNotFoundError(f"Model not found: {path}")

def load_coreml_model(path, function_name=None):
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
    except Exception as e:
        print(f"Error loading model {path}: {e}")
        raise

def extract_metadata(model):
    """Extract metadata from CoreML model."""
    metadata = {
        'context_length': 256,
        'state_length': 256,
        'batch_size': 64,
        'num_logits': 16
    }
    
    if hasattr(model, 'user_defined_metadata'):
        meta = model.user_defined_metadata
        metadata['context_length'] = int(meta.get('com.anemll.context_length', 256))
        metadata['state_length'] = int(meta.get('com.anemll.state_length', metadata['context_length']))
        metadata['batch_size'] = int(meta.get('com.anemll.batch_size', 64))
        
        print(f"Extracted metadata:")
        print(f"  Context Length: {metadata['context_length']}")
        print(f"  State Length: {metadata['state_length']}")
        print(f"  Batch Size: {metadata['batch_size']}")
    else:
        print(f"Using default metadata: {metadata}")
    
    return metadata

def make_causal_mask(length, start=0):
    """Create causal attention mask."""
    mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
    row_indices = np.arange(length).reshape(length, 1)
    col_indices = np.arange(length).reshape(1, length)
    mask[:, :, col_indices <= (row_indices + start)] = 0
    return mask

def single_token_prefill_coreml(model, token_id, position, causal_mask, state):
    """Process a single token with single CoreML model and update state."""
    # Prepare inputs
    input_ids = np.array([[token_id]], dtype=np.int32)
    position_ids = np.array([position], dtype=np.int32)
    current_pos = np.array([position], dtype=np.int32)
    
    # Create update mask for this position
    update_mask = np.zeros((1, 1, causal_mask.shape[-1], 1), dtype=np.float16)
    update_mask[0, 0, position, 0] = 1.0
    
    # Use single-token slice of causal mask
    single_causal_mask = causal_mask[:, :, position:position+1, :]
    
    print(f"  Processing token {token_id} at position {position}")
    
    # Run single unified model
    start_time = time.time()
    inputs = {
        'input_ids': input_ids,
        'update_mask': update_mask,
        'position_ids': position_ids,
        'causal_mask': single_causal_mask,
        'current_pos': current_pos
    }
    
    # Debug: Print input tensors for first few tokens
    if position <= 2:
        print(f"    🔍 DEBUG - Token {position} inputs:")
        print(f"      input_ids: {input_ids} (shape: {input_ids.shape})")
        print(f"      position_ids: {position_ids} (shape: {position_ids.shape})")
        print(f"      current_pos: {current_pos} (shape: {current_pos.shape})")
        print(f"      update_mask: shape {update_mask.shape}, sum: {update_mask.sum()}")
        print(f"      causal_mask: shape {single_causal_mask.shape}, min: {single_causal_mask.min():.1f}, max: {single_causal_mask.max():.1f}")
        print(f"      causal_mask first 10 values: {single_causal_mask.flatten()[:10]}")
    
    output = model.predict(inputs, state)
    total_time = time.time() - start_time
    
    print(f"    Model: {total_time*1000:.1f}ms")
    
    return output, total_time

def generate_next_token_coreml(model, token_id, position, causal_mask, metadata, state):
    """Generate next token using single CoreML model."""
    # Process current token
    output, processing_time = single_token_prefill_coreml(
        model, token_id, position, causal_mask, state
    )
    
    # Extract logits from output - handle multiple logits parts
    num_logits = metadata.get('num_logits', 16)
    
    # Try to extract and concatenate logits1-N (like in other tests)
    if 'logits1' in output:
        print(f"    Found chunked logits output (logits1-{num_logits})")
        logits_parts = []
        for i in range(1, num_logits + 1):
            key = f'logits{i}'
            if key in output:
                part = output[key]
                #print(f"      {key}: shape {part.shape}")
                logits_parts.append(part)
        
        if len(logits_parts) == num_logits:
            # Concatenate along vocab dimension  
            logits = np.concatenate(logits_parts, axis=-1)
            print(f"    Concatenated logits shape: {logits.shape}")
        else:
            raise ValueError(f"Expected {num_logits} logits parts, got {len(logits_parts)}")
            
    elif 'logits' in output:
        logits = output['logits']
        print(f"    Found single logits output: {logits.shape}")
    elif 'output_logits' in output:
        logits = output['output_logits']
        print(f"    Found output_logits: {logits.shape}")
    else:
        # Debug: print all available keys
        print(f"    Available output keys: {list(output.keys())}")
        # Try to find any logits output
        logits_keys = [k for k in output.keys() if 'logits' in k.lower()]
        if logits_keys:
            logits = output[logits_keys[0]]
            print(f"    Using fallback logits key '{logits_keys[0]}': {logits.shape}")
        else:
            raise ValueError(f"Could not find logits in model output. Available keys: {list(output.keys())}")
    
    # Get next token (greedy sampling) - use last position for single token output
    if logits.ndim == 3:
        # Shape: [batch, seq_len, vocab_size] - use last position
        next_token = np.argmax(logits[0, -1, :])
    elif logits.ndim == 2:
        # Shape: [batch, vocab_size] - direct indexing
        next_token = np.argmax(logits[0, :])
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    
    print(f"    Total: {processing_time*1000:.1f}ms")
    
    return int(next_token), processing_time

def test_coreml_kv_cache_sequential():
    """Test CoreML KV cache with sequential processing."""
    parser = argparse.ArgumentParser(description='Test CoreML KV cache sequential processing')
    parser.add_argument('--model', '-m', type=str, 
                       default='/tmp/qwen-test/full/test_qwen.mlpackage',
                       help='Path to CoreML model file')
    parser.add_argument('--tokenizer', type=str, 
                       default='Qwen/Qwen3-0.6B',
                       help='Hugging Face model identifier for Qwen tokenizer')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum number of tokens to generate (default: 100)')
    
    args = parser.parse_args()
    
    print(f"{LIGHT_GREEN}🚀 CoreML KV Cache Sequential Test{RESET_COLOR}")
    print("=" * 60)
    
    try:
        # Load single CoreML model
        print(f"\n📚 Loading CoreML Model...")
        model_path = parse_model_path(args.model)
        model = load_coreml_model(model_path)
        print(f"✅ Loaded model: {Path(model_path).name}")
        
        # Extract metadata
        metadata = extract_metadata(model)
        
        # Find model path for tokenizer
        print(f"\n🔧 Finding Qwen model for tokenizer...")
        model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
        model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
        if not model_dirs:
            print("❌ Error: Qwen model not found in cache")
            return False
        
        model_dir = model_dirs[0]
        print(f"Using model from: {model_dir}")
        
        # Load tokenizer from local model directory
        print(f"🔧 Loading Qwen tokenizer from local cache...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        print(f"✅ Loaded tokenizer: {tokenizer.__class__.__name__}")
        
        # Create unified state
        print(f"\n🔧 Creating CoreML state...")
        state = model.make_state()
        print(f"✅ Created CoreML state")
        
        # Initialize causal mask - override with actual model requirement (256)
        # The model description shows it expects (1 x 1 x 1 x 256) regardless of metadata
        actual_state_length = 256
        causal_mask = make_causal_mask(actual_state_length)
        print(f"✅ Created causal mask for actual state length: {actual_state_length} (model expects 256)")
        
        # Test prompt
        #prompt = "What is Apple Neural Engine"
        prompt = 'The capital of France is'
        print(f"\n🔥 Test prompt: '{prompt}'")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs.input_ids[0].tolist()
        print(f"Tokenized to {len(input_ids)} tokens: {input_ids}")
        print(f"Token meanings: {[tokenizer.decode([t]) for t in input_ids]}")
        
        # Prefill phase - process input tokens one by one
        print(f"\n⚡ Prefill Phase (Token-by-Token)")
        print("-" * 40)
        
        prefill_start = time.time()
        prefill_times = []
        
        for i, token_id in enumerate(input_ids):
            print(f"\nPrefill step {i+1}/{len(input_ids)}:")
            _, token_time = single_token_prefill_coreml(
                model, token_id, i, causal_mask, state
            )
            prefill_times.append(token_time)
        
        total_prefill_time = time.time() - prefill_start
        avg_prefill_time = sum(prefill_times) / len(prefill_times)
        prefill_tps = len(input_ids) / total_prefill_time
        
        print(f"\n📊 Prefill Results:")
        print(f"  Total time: {total_prefill_time*1000:.1f}ms")
        print(f"  Average per token: {avg_prefill_time*1000:.1f}ms")
        print(f"  Prefill speed: {prefill_tps:.1f} t/s")
        
        # Inference phase - generate new tokens
        print(f"\n🚀 Inference Phase (Generation)")
        print("-" * 40)
        
        max_new_tokens = args.max_tokens
        generated_tokens = []
        inference_times = []
        current_pos = len(input_ids)
        
        # Start with the last input token for the first generation step
        current_token = input_ids[-1] if input_ids else 0
        
        inference_start = time.time()
        
        for gen_step in range(max_new_tokens):
            print(f"\nGeneration step {gen_step+1}/{max_new_tokens}:")
            print(f"    Input token: {current_token} ('{tokenizer.decode([current_token])}')")
            
            next_token, token_time = generate_next_token_coreml(
                model, current_token, current_pos, causal_mask, metadata, state
            )
            
            generated_tokens.append(next_token)
            inference_times.append(token_time)
            
            # Update for next iteration: use the generated token as input for the next step
            current_token = next_token
            current_pos += 1
            
            decoded_token = tokenizer.decode([next_token])
            print(f"    Generated: {next_token} ('{decoded_token}')")
            
            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                print("    Reached EOS token, stopping")
                break
        
        total_inference_time = time.time() - inference_start
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        inference_tps = len(generated_tokens) / total_inference_time if total_inference_time > 0 else 0
        
        print(f"\n📊 Inference Results:")
        print(f"  Total time: {total_inference_time*1000:.1f}ms")
        print(f"  Average per token: {avg_inference_time*1000:.1f}ms")
        print(f"  Inference speed: {inference_tps:.1f} t/s")
        
        # Final results
        print(f"\n🎉 Final Results")
        print("=" * 50)
        print(f"Original prompt: '{prompt}'")
        
        if generated_tokens:
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"Generated text: '{generated_text}'")
            print(f"Full response: '{prompt}{generated_text}'")
        
        print(f"\n📈 Performance Summary:")
        print(f"  Prefill: {prefill_tps:.1f} t/s ({len(input_ids)} tokens)")
        print(f"  Inference: {inference_tps:.1f} t/s ({len(generated_tokens)} tokens)")
        print(f"  Total tokens: {len(input_ids) + len(generated_tokens)}")
        
        total_time = total_prefill_time + total_inference_time
        overall_tps = (len(input_ids) + len(generated_tokens)) / total_time
        print(f"  Overall: {overall_tps:.1f} t/s")
        
        print(f"\n✅ CoreML KV cache sequential test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_coreml_kv_cache_sequential()
    sys.exit(0 if success else 1) 
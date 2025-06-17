#!/usr/bin/env python3
"""Test CoreML KV cache with batch prefill using separate embeddings and prefill models."""

import time
import torch
import torch.nn.functional as F
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
        print(f"Loading CoreML model from: {path}")
        if path.suffix == '.mlmodelc':
            # For compiled models (.mlmodelc), use CompiledMLModel
            if function_name:
                model = ct.models.CompiledMLModel(str(path), compute_unit, function_name=function_name)
            else:
                model = ct.models.CompiledMLModel(str(path), compute_unit)
        else:
            # For packages (.mlpackage)
            if function_name:
                model = ct.models.MLModel(str(path), compute_units=compute_unit, function_name=function_name)
            else:
                model = ct.models.MLModel(str(path), compute_units=compute_unit)
        print(f"✅ Model loaded successfully")
        return model
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
        print(f"❌ Failed to load model: {e}")
        raise

def make_causal_mask(length, start=0):
    """Create causal attention mask."""
    mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
    row_indices = np.arange(length).reshape(length, 1)
    col_indices = np.arange(length).reshape(1, length)
    mask[:, :, col_indices <= (row_indices + start)] = 0
    return mask

def run_batch_prefill(embed_model, prefill_model, input_ids, context_pos, context_length, batch_size=64, state=None, causal_mask=None):
    """Run batch prefill using separate embeddings and prefill models."""
    print(f"\n🚀 Running batch prefill for {context_pos} tokens (batch_size={batch_size})")
    
    # Use provided causal mask or create one if not provided
    if causal_mask is None:
        causal_mask = make_causal_mask(context_length)
        causal_mask = torch.tensor(causal_mask, dtype=torch.float16)
    
    # Process in batches
    batch_pos = 0
    total_prefill_time = 0
    
    while batch_pos < context_pos:
        batch_end = min(batch_pos + batch_size, context_pos)
        current_batch_size = batch_end - batch_pos
        
        print(f"  📦 Processing batch: tokens {batch_pos} to {batch_end-1} (size: {current_batch_size})")
        
        # Get current batch
        batch_input = input_ids[:, batch_pos:batch_end]
        
        # Always pad to full batch size for prefill
        batch_input = F.pad(
            batch_input,
            (0, batch_size - current_batch_size),
            value=0
        )
        
        # Generate position IDs for full batch size
        position_ids = torch.arange(batch_pos, batch_pos + batch_size, dtype=torch.int32)
        batch_causal_mask = causal_mask[:, :, batch_pos:batch_pos + batch_size, :]
        
        start_time = time.time()
        
        # Step 1: Run embeddings model (input_ids -> hidden_states)
        print(f"    🔤 Running embeddings model...")
        embed_inputs = {
            'input_ids': batch_input.numpy().astype(np.int32),
        }
        embed_output = embed_model.predict(embed_inputs)
        hidden_states = torch.from_numpy(embed_output['hidden_states'])
        
        embed_time = time.time() - start_time
        print(f"      Embeddings: {embed_time*1000:.1f}ms")
        
        # Step 2: Run prefill model (hidden_states -> hidden_states + KV cache update)
        print(f"    🧠 Running prefill model...")
        prefill_start = time.time()
        
        prefill_inputs = {
            'hidden_states': hidden_states.numpy(),  # [1, batch_size, hidden_size]
            'position_ids': position_ids.numpy(),    # [batch_size]
            'causal_mask': batch_causal_mask.numpy(), # [1, 1, batch_size, context_length]
            'current_pos': np.array([batch_pos], dtype=np.int32)  # [1]
        }
        
        prefill_output = prefill_model.predict(prefill_inputs, state)
        prefill_time = time.time() - prefill_start
        print(f"      Prefill: {prefill_time*1000:.1f}ms")
        
        batch_time = time.time() - start_time
        batch_tps = current_batch_size / batch_time
        total_prefill_time += batch_time
        
        print(f"    ⚡ Batch {batch_pos//batch_size + 1}: {batch_time*1000:.1f}ms total, {batch_tps:.1f} t/s")
        
        batch_pos = batch_end
    
    overall_prefill_tps = context_pos / total_prefill_time
    print(f"  ✅ Prefill completed: {total_prefill_time*1000:.1f}ms total, {overall_prefill_tps:.1f} t/s")
    
    return torch.tensor([context_pos], dtype=torch.int32), total_prefill_time

def run_sequential_prefill_chat_style(full_model, input_ids, context_pos, context_length, state=None, causal_mask=None, no_debug=False):
    """Run sequential prefill using the chat.py approach - fixed window style."""
    print(f"\n🚀 Running sequential prefill (chat.py style) for {context_pos} tokens")
    
    # Use provided causal mask or create one if not provided  
    if causal_mask is None:
        # Create proper triangular causal mask like test_coreml_kvcache_sequential.py
        causal_mask_data = make_causal_mask(context_length)  # Dynamic triangular mask
        causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
    
    # Process tokens one by one using chat.py style
    total_prefill_time = 0
    prefill_times = []
    
    for i, token_id in enumerate(input_ids[0].tolist()):
        if not no_debug:
            print(f"  📦 Processing token {i+1}/{context_pos}: {token_id}")
        
        start_time = time.time()
        
        # Prepare inputs like chat.py generate_next_token (single token)
        current_token = torch.tensor([[token_id]], dtype=torch.int32)
        
        # Create masks like chat.py
        update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
        update_mask[0, 0, i, 0] = 1.0
        position_ids = torch.tensor([i], dtype=torch.int32)  # Single position like chat.py
        
        # Use single-token slice of causal mask like chat.py
        single_causal_mask = causal_mask[:, :, i:i+1, :]
        
        # Run full model for this token
        inputs = {
            'input_ids': current_token.numpy(),
            'update_mask': update_mask.numpy(),
            'position_ids': position_ids.numpy(),
            'causal_mask': single_causal_mask.numpy(),
            'current_pos': position_ids.numpy()
        }
        
        # Debug: Print input tensors for first few tokens
        if i <= 2 and not no_debug:
            print(f"    🔍 DEBUG - Token {i} inputs (chat.py style):")
            print(f"      input_ids: {inputs['input_ids']} (shape: {inputs['input_ids'].shape})")
            print(f"      position_ids: {inputs['position_ids']} (shape: {inputs['position_ids'].shape})")
            print(f"      current_pos: {inputs['current_pos']} (shape: {inputs['current_pos'].shape})")
            print(f"      update_mask: shape {inputs['update_mask'].shape}, sum: {inputs['update_mask'].sum()}")
            print(f"      causal_mask: shape {inputs['causal_mask'].shape}, min: {inputs['causal_mask'].min():.1f}, max: {inputs['causal_mask'].max():.1f}")
            print(f"      causal_mask first 10 values: {inputs['causal_mask'].flatten()[:10]}")
        
        output = full_model.predict(inputs, state)
        token_time = time.time() - start_time
        prefill_times.append(token_time)
        total_prefill_time += token_time
        
        if not no_debug:
            print(f"    ⚡ Token {i+1}: {token_time*1000:.1f}ms")
    
    overall_prefill_tps = context_pos / total_prefill_time
    print(f"  ✅ Sequential prefill completed: {total_prefill_time*1000:.1f}ms total, {overall_prefill_tps:.1f} t/s")
    
    return torch.tensor([context_pos], dtype=torch.int32), total_prefill_time

def single_token_inference_coreml(model, token_id, position, causal_mask, state, no_debug=False):
    """Generate single token using full CoreML model for inference."""
    # Prepare inputs
    input_ids = np.array([[token_id]], dtype=np.int32)
    position_ids = np.array([position], dtype=np.int32)
    current_pos = np.array([position], dtype=np.int32)
    
    # Create update mask for this position
    update_mask = np.zeros((1, 1, causal_mask.shape[-1], 1), dtype=np.float16)
    update_mask[0, 0, position, 0] = 1.0
    
    # Use single-token slice of causal mask
    single_causal_mask = causal_mask[:, :, position:position+1, :]
    
    # Run full model for inference
    start_time = time.time()
    
    # Convert causal mask to numpy (keep -inf values like chat.py)
    causal_mask_np = single_causal_mask.numpy() if hasattr(single_causal_mask, 'numpy') else single_causal_mask
    
    inputs = {
        'input_ids': input_ids,
        'update_mask': update_mask,
        'position_ids': position_ids,
        'causal_mask': causal_mask_np,
        'current_pos': current_pos
    }
    
    # Debug print for first inference
    if position == 5:
        print(f"    🔍 DEBUG - First inference inputs (chat.py style):")
        print(f"      causal_mask min: {inputs['causal_mask'].min():.1f}, max: {inputs['causal_mask'].max():.1f}")
        print(f"      causal_mask dtype: {inputs['causal_mask'].dtype}")
    
    output = model.predict(inputs, state)
    total_time = time.time() - start_time
    
    return output, total_time

def generate_next_token_coreml(full_model, token_id, position, causal_mask, metadata, state, no_debug=False):
    """Generate next token using full CoreML model."""
    # Process current token
    output, processing_time = single_token_inference_coreml(
        full_model, token_id, position, causal_mask, state, no_debug
    )
    
    # Extract logits from output - handle multiple logits parts
    num_logits = metadata.get('num_logits', 16)
    
    # Try to extract and concatenate logits1-N
    if 'logits1' in output:
        if not no_debug:
            print(f"    📊 Found chunked logits output (logits1-{num_logits})")
        logits_parts = []
        for i in range(1, num_logits + 1):
            key = f'logits{i}'
            if key in output:
                part = output[key]
                #print(f"      {key}: shape {part.shape}")
                logits_parts.append(part)
        
        if logits_parts:
            logits = np.concatenate(logits_parts, axis=-1)
            if not no_debug:
                print(f"    📈 Concatenated logits shape: {logits.shape}")
        else:
            raise ValueError("No logits parts found in output")
    else:
        # Single logits output
        if 'logits' in output:
            logits = output['logits']
        else:
            # Try first available output
            output_keys = list(output.keys())
            if output_keys:
                logits = output[output_keys[0]]
                if not no_debug:
                    print(f"    📊 Using output key: {output_keys[0]}")
            else:
                raise ValueError("No output found in model")
    
    # Get next token (greedy sampling) - use last position for single token output
    if logits.ndim == 3:
        # Shape: [batch, seq_len, vocab_size] - use last position
        next_token = np.argmax(logits[0, -1, :])
    elif logits.ndim == 2:
        # Shape: [batch, vocab_size] - direct indexing
        next_token = np.argmax(logits[0, :])
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    
    next_token = int(next_token)
    
    return next_token, processing_time

def test_coreml_kv_cache_prefill():
    """Test CoreML KV cache with batch prefill and inference."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test CoreML batch prefill with separate models")
    parser.add_argument("--embeddings", default="/tmp/qwen-test/embeddings/test_qwen_embeddings.mlpackage", 
                       help="Path to embeddings model")
    parser.add_argument("--prefill", default="/tmp/qwen-test/prefill/test_qwen_prefill.mlpackage", 
                       help="Path to prefill model")  
    parser.add_argument("--full", default="/tmp/qwen-test/full/test_qwen.mlpackage", 
                       help="Path to full model for inference")
    parser.add_argument("--tokenizer", default="~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/", 
                       help="Path to tokenizer")
    parser.add_argument("--prompt", default="The capital of France is", 
                       help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=20, 
                       help="Maximum tokens to generate")
    parser.add_argument("--batch-size", type=int, default=64, 
                       help="Batch size for prefill")
    parser.add_argument("--context-length", type=int, default=256, 
                       help="Context length")
    parser.add_argument("--sequential", action="store_true",
                       help="Use sequential token-by-token prefill instead of batch prefill")
    parser.add_argument("--no-debug", action="store_true",
                       help="Suppress detailed token processing debug output")
    parser.add_argument("--system", action="store_true",
                       help="Use system/default template for the prompt")
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting CoreML KV cache batch prefill test")
        print("=" * 60)
        
        # Load models
        print(f"\n📥 Loading models...")
        if args.sequential:
            # For sequential mode, only load the full model (like test_coreml_kvcache_sequential.py)
            print("Sequential mode: loading only full model")
            embeddings_model = None
            prefill_model = None
            full_model = load_coreml_model(parse_model_path(args.full))
        else:
            # For batch mode, load all separate models
            print("Batch mode: loading separate models")
            embeddings_model = load_coreml_model(parse_model_path(args.embeddings))
            prefill_model = load_coreml_model(parse_model_path(args.prefill))
            full_model = load_coreml_model(parse_model_path(args.full))
        
        # Load tokenizer
        print(f"\n🔤 Loading tokenizer from: {args.tokenizer}")
        tokenizer_path = os.path.expanduser(args.tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Setup
        prompt = args.prompt
        max_new_tokens = args.max_tokens
        batch_size = args.batch_size
        context_length = args.context_length
        
        # Apply system template if requested
        if args.system:
            print(f"\n📝 Original prompt: '{prompt}'")
            
            # Apply chat template - using default system prompt
            if hasattr(tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                templated_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                prompt = templated_prompt
                print(f"📝 Templated prompt: '{prompt}'")
            else:
                print("⚠️  Tokenizer does not support chat templates, using original prompt")
        
        print(f"\n⚙️  Configuration:")
        print(f"  Prompt: '{prompt}'")
        print(f"  Max tokens: {max_new_tokens}")
        print(f"  Batch size: {batch_size}")
        print(f"  Context length: {context_length}")
        
        # Tokenize prompt
        print(f"\n🔤 Tokenizing prompt...")
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        context_pos = input_ids.shape[1]
        
        print(f"  Input tokens: {input_ids.tolist()[0]}")
        print(f"  Context position: {context_pos}")
        
        # Create causal mask
        causal_mask = make_causal_mask(context_length)
        causal_mask = torch.tensor(causal_mask, dtype=torch.float16)
        
        # Create states for models
        if args.sequential:
            # For sequential mode, create state from the full model (like test_coreml_kvcache_sequential.py)
            shared_state = full_model.make_state()
        else:
            # For batch mode, create state from prefill model
            shared_state = prefill_model.make_state()
        
        # Debug: Check initial state
        print(f"\n🔍 Initial state type: {type(shared_state)}")
        print(f"🔍 Initial state: {shared_state}")
        if hasattr(shared_state, 'keys'):
            print(f"🔍 Initial state keys: {list(shared_state.keys())}")
            for key in list(shared_state.keys())[:3]:  # Show first 3 keys
                state_val = shared_state[key]
                if hasattr(state_val, 'shape'):
                    print(f"  {key}: shape {state_val.shape}")
                    print(f"  {key}: sum = {np.sum(state_val):.6f}")
        elif hasattr(shared_state, '__len__'):
            print(f"🔍 Initial state length: {len(shared_state)}")
        else:
            print(f"🔍 Initial state value: {shared_state}")
        
        # Run prefill (batch or sequential)
        if args.sequential:
            print(f"\n📦 Running sequential prefill...")
            current_pos, total_prefill_time = run_sequential_prefill_chat_style(
                full_model, input_ids, context_pos, 
                context_length, shared_state, causal_mask, args.no_debug
            )
        else:
            print(f"\n📦 Running batch prefill...")
            current_pos, total_prefill_time = run_batch_prefill(
                embeddings_model, prefill_model, input_ids, context_pos, 
                context_length, batch_size, shared_state, causal_mask
            )
        
        # Debug: Check state after prefill
        print(f"\n🔍 State after prefill:")
        if hasattr(shared_state, 'keys'):
            for key in list(shared_state.keys())[:3]:  # Show first 3 keys
                state_val = shared_state[key]
                if hasattr(state_val, 'shape'):
                    print(f"  {key}: shape {state_val.shape}")
                    print(f"  {key}: sum = {np.sum(state_val):.6f}")
        
        prefill_tps = context_pos / total_prefill_time
        print(f"✅ Prefill completed: {total_prefill_time*1000:.1f}ms, {prefill_tps:.1f} t/s")
        
        # Generate tokens
        print(f"\n🧠 Starting inference...")
        generated_tokens = []
        inference_times = []
        inference_start = time.time()
        current_pos = context_pos
        
        metadata = {'num_logits': 16}  # Default for Qwen
        
        for i in range(max_new_tokens):
            if not args.no_debug:
                print(f"\n  Token {i+1}/{max_new_tokens}:")
            
            # Use last token for next prediction
            last_token = input_ids[0, -1].item() if i == 0 else generated_tokens[-1]
            
            next_token, token_time = generate_next_token_coreml(
                full_model, last_token, current_pos, causal_mask, metadata, shared_state, args.no_debug
            )
            
            generated_tokens.append(next_token)
            inference_times.append(token_time)
            current_pos += 1
            
            if not args.no_debug:
                decoded_token = tokenizer.decode([next_token])
                print(f"    🎯 Generated: {next_token} ('{decoded_token}')")
            
            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                if not args.no_debug:
                    print("    🏁 Reached EOS token, stopping")
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
        print(f"  Prefill: {prefill_tps:.1f} t/s ({context_pos} tokens)")
        print(f"  Inference: {inference_tps:.1f} t/s ({len(generated_tokens)} tokens)")
        print(f"  Total tokens: {context_pos + len(generated_tokens)}")
        
        total_time = total_prefill_time + total_inference_time
        overall_tps = (context_pos + len(generated_tokens)) / total_time
        print(f"  Overall: {overall_tps:.1f} t/s")
        
        print(f"\n✅ CoreML KV cache batch prefill test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_coreml_kv_cache_prefill()
    sys.exit(0 if success else 1)
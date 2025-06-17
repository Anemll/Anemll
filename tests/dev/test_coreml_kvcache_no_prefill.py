#!/usr/bin/env python3
"""Test CoreML KV cache vs PyTorch baseline - should produce identical results."""

import numpy as np
import torch
import os
import glob
import coremltools as ct
from pathlib import Path
from transformers import AutoTokenizer
from anemll.models.qwen_model import QwenForCausalLM, QwenConfig
import warnings
warnings.filterwarnings('ignore')

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

def load_coreml_model(path):
    """Load CoreML model with compilation support and proper error handling."""
    path = Path(path)
    
    # Try different path variations, prefer .mlpackage for compilation
    candidates = [
        path.with_suffix('.mlpackage'),
        path.with_suffix('.mlmodelc'),
        Path(str(path) + '.mlpackage'),
        Path(str(path) + '.mlmodelc'),
        path.with_suffix(''),
        path
    ]
    
    for candidate in candidates:
        if candidate.exists():
            print(f"Found CoreML model at: {candidate}")
            try:
                # If it's a .mlpackage, try to compile to .mlmodelc first
                if candidate.suffix == '.mlpackage':
                    model_path = compile_model_if_needed(candidate)
                else:
                    model_path = str(candidate)
                
                final_path = Path(model_path)
                if final_path.suffix == '.mlmodelc':
                    return ct.models.CompiledMLModel(str(final_path), ct.ComputeUnit.CPU_AND_NE)
                else:
                    return ct.models.MLModel(str(final_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            except Exception as e:
                print(f"Failed to load {candidate}: {e}")
                continue
                
    raise FileNotFoundError(f"CoreML model not found. Tried: {candidates}")

def extract_coreml_metadata(model):
    """Extract metadata from CoreML model."""
    metadata = {
        'context_length': 256,
        'state_length': 256,
        'has_kv_cache': False
    }
    
    # Check for KV cache states
    if hasattr(model, 'get_spec'):
        spec = model.get_spec()
        if hasattr(spec.description, 'stateTypes') and spec.description.stateTypes:
            metadata['has_kv_cache'] = True
            print(f"✅ CoreML model has {len(spec.description.stateTypes)} KV cache state(s)")
        else:
            print("⚠️  CoreML model has no KV cache states")
            
        # Inspect input shapes
        print(f"\n🔍 CoreML Model Input Specifications:")
        for input_spec in spec.description.input:
            input_name = input_spec.name
            if hasattr(input_spec, 'type') and hasattr(input_spec.type, 'multiArrayType'):
                shape = input_spec.type.multiArrayType.shape
                print(f"  {input_name}: {[s for s in shape]}")
            else:
                print(f"  {input_name}: (unknown type)")
    
    # Extract user metadata if available
    if hasattr(model, 'user_defined_metadata'):
        meta = model.user_defined_metadata
        metadata['context_length'] = int(meta.get('com.anemll.context_length', 256))
        metadata['state_length'] = int(meta.get('com.anemll.state_length', metadata['context_length']))
        print(f"📋 CoreML metadata: context={metadata['context_length']}, state={metadata['state_length']}")
    
    return metadata

def test_coreml_vs_pytorch():
    print("🍎 CoreML vs PyTorch KV Cache Test")
    print("=" * 80)
    
    # Find model path
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    if not model_dirs:
        print("❌ Error: Qwen model not found in cache")
        return False
    
    model_dir = model_dirs[0]
    print(f"Using model from: {model_dir}")
    
    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    config = QwenConfig.from_json(os.path.join(model_dir, 'config.json'))
    
    # CoreML model path
    coreml_path = "/tmp/qwen-test/float32/test_qwen.mlpackage"
    
    # Check if CoreML model exists
    if not Path(coreml_path).exists():
        print(f"❌ CoreML model not found at {coreml_path}")
        print("Run export_coreml.py first to create the CoreML model")
        return False
    
    # CoreML-style fixed window parameters (matching our successful test)
    CONTEXT_LENGTH = 256  # Fixed context window size (matches model's state_length)
    PAD_TOKEN_ID = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    
    # Test prompt (same as our successful test)
    prompt_text = "What"
    
    print(f"\n📝 Testing prompt: '{prompt_text}' (CoreML vs PyTorch comparison)")
    print("-" * 60)
    
    # Tokenize prompt
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(TEST_DEVICE)
    original_tokens = input_ids[0].tolist()
    seq_len = len(original_tokens)
    
    print(f"  Original tokens: {original_tokens}")
    print(f"  Decoded: {[tokenizer.decode([t]) for t in original_tokens]}")
    print(f"  Sequence length: {seq_len}")
    
    # === CoreML-style Fixed Window Setup ===
    # Pad input to full context length
    padded_tokens = original_tokens + [PAD_TOKEN_ID] * (CONTEXT_LENGTH - seq_len)
    padded_input_ids = torch.tensor([padded_tokens], dtype=torch.long, device=TEST_DEVICE)
    
    # For single token processing (CoreML style), we only care about the last real token position
    position_ids = torch.tensor([seq_len - 1], dtype=torch.long, device=TEST_DEVICE)  # Single position
    
    # Current position is the last position of actual content
    current_pos = torch.tensor([seq_len - 1], dtype=torch.long, device=TEST_DEVICE)
    
    # Create single-row causal mask for single token (CoreML style)
    # For single token processing, we need [1, 1, 1, CONTEXT_LENGTH] shape
    causal_mask = torch.zeros((1, 1, 1, CONTEXT_LENGTH), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    # Allow attention to positions up to current position (seq_len - 1)
    # Block attention to future positions 
    for j in range(seq_len, CONTEXT_LENGTH):
        causal_mask[0, 0, 0, j] = float('-inf')
    
    # Create update mask for single token processing
    update_mask = torch.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    update_mask[0, 0, seq_len - 1, 0] = 1.0  # Only update the last real token position
    
    print(f"    Tensor shapes:")
    print(f"      input_ids: {padded_input_ids.shape}")
    print(f"      update_mask: {update_mask.shape}")
    print(f"      position_ids: {position_ids.shape}")
    print(f"      causal_mask: {causal_mask.shape}")
    print(f"      current_pos: {current_pos.shape}")
    
    # === Method 1: PyTorch Baseline (No KV Cache) ===
    print(f"\n🔥 Method 1: PyTorch Baseline (No KV Cache)")
    
    model_pytorch = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=True)
    model_pytorch.load_pretrained_weights(model_dir)
    model_pytorch.eval()
    
    with torch.no_grad():
        logits_pytorch = model_pytorch(
            input_ids=padded_input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
    
    # Get logits for the last real token position
    next_token_pytorch = torch.argmax(logits_pytorch[0, seq_len - 1, :]).item()
    print(f"  Result: {next_token_pytorch} ('{tokenizer.decode([next_token_pytorch])}')")

    # Print top 5 tokens and their probabilities
    logits_pytorch_last = logits_pytorch[0, seq_len - 1, :]
    probs_pytorch = torch.softmax(logits_pytorch_last, dim=-1)
    topk_pytorch = torch.topk(probs_pytorch, 5)
    print("  Top 5 tokens (PyTorch):")
    for rank, (idx, prob) in enumerate(zip(topk_pytorch.indices.tolist(), topk_pytorch.values.tolist()), 1):
        decoded = tokenizer.decode([idx])
        print(f"    {rank}. {idx} ('{decoded}') prob: {prob:.4f}")
    
    # === Method 2: CoreML Model ===
    print(f"\n🍎 Method 2: CoreML Model")
    
    try:
        # Load CoreML model
        coreml_model = load_coreml_model(coreml_path)
        metadata = extract_coreml_metadata(coreml_model)
        
        # Create CoreML state (always required)
        print(f"\n🔧 Creating CoreML state...")
        state = coreml_model.make_state()
        print(f"✅ Created CoreML state")
        
        # Determine input shapes based on CoreML model specifications
        # If model expects single token (shape 1), use single token approach
        # If model expects full context (shape 256), use fixed window approach
        
        # Check actual input specifications from the model
        use_single_token = True  # Default assumption
        if hasattr(coreml_model, 'get_spec'):
            spec = coreml_model.get_spec()
            for input_spec in spec.description.input:
                if input_spec.name == 'input_ids' and hasattr(input_spec, 'type'):
                    if hasattr(input_spec.type, 'multiArrayType'):
                        shape = input_spec.type.multiArrayType.shape
                        if len(shape) >= 2 and shape[1] > 1:  # Batch dimension and sequence dimension
                            use_single_token = False
                            expected_seq_len = shape[1]
                            print(f"📏 CoreML expects full context: sequence length {expected_seq_len}")
                            break
        
        if use_single_token:
            print(f"📏 CoreML expects single token inputs")
            # Prepare single token inputs (original approach)
            coreml_input_ids = np.array([[original_tokens[0]]], dtype=np.int32)  # Single token
            coreml_position_ids = np.array([0], dtype=np.int32)  # Position 0
            coreml_current_pos = np.array([0], dtype=np.int32)  # Current position 0
            # Single token causal mask: [1, 1, 1, CONTEXT_LENGTH] - allow attention to position 0 only
            coreml_causal_mask = np.zeros((1, 1, 1, CONTEXT_LENGTH), dtype=np.float16)
            for j in range(1, CONTEXT_LENGTH):
                coreml_causal_mask[0, 0, 0, j] = float('-inf')
            coreml_update_mask = np.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=np.float16)
            coreml_update_mask[0, 0, 0, 0] = 1.0  # Update only position 0
        else:
            print(f"📏 CoreML expects full context inputs")
            # Use the fixed window approach we already prepared
            coreml_input_ids = padded_input_ids.cpu().numpy().astype(np.int32)
            coreml_position_ids = position_ids.cpu().numpy().astype(np.int32)
            coreml_current_pos = current_pos.cpu().numpy().astype(np.int32)
            coreml_causal_mask = causal_mask.cpu().numpy().astype(np.float16)
            coreml_update_mask = update_mask.cpu().numpy().astype(np.float16)
        
        print(f"    CoreML input shapes:")
        print(f"      input_ids: {coreml_input_ids.shape}")
        print(f"      position_ids: {coreml_position_ids.shape}")
        print(f"      current_pos: {coreml_current_pos.shape}")
        print(f"      causal_mask: {coreml_causal_mask.shape}")
        print(f"      update_mask: {coreml_update_mask.shape}")
        
        # Prepare CoreML inputs
        coreml_inputs = {
            'input_ids': coreml_input_ids,
            'position_ids': coreml_position_ids,
            'causal_mask': coreml_causal_mask,
            'current_pos': coreml_current_pos,
            'update_mask': coreml_update_mask
        }
        
        print(f"    Running CoreML inference...")
        print(f"    KV cache enabled: {metadata['has_kv_cache']}")
        
        # Run CoreML model with state (always required)
        coreml_output = coreml_model.predict(coreml_inputs, state)
        
        # Extract logits from CoreML output
        coreml_logits_parts = []
        for i in range(1, 17):  # logits1 to logits16
            key = f'logits{i}'
            if key in coreml_output:
                part = coreml_output[key]
                if part.ndim == 3:
                    if use_single_token:
                        # Single token: use position 0
                        token_logits = part[0, 0, :]
                    else:
                        # Full context: use last real token position
                        token_logits = part[0, seq_len - 1, :]
                else:
                    # 2D output: assume [batch, vocab_size]
                    token_logits = part[0, :]
                coreml_logits_parts.append(token_logits)
        
        if len(coreml_logits_parts) == 16:
            # Concatenate all logits parts
            coreml_full_logits = np.concatenate(coreml_logits_parts)
            next_token_coreml = np.argmax(coreml_full_logits)
            
            print(f"  Result: {next_token_coreml} ('{tokenizer.decode([next_token_coreml])}')")
            
            # Print top 5 tokens for CoreML
            coreml_probs = np.exp(coreml_full_logits) / np.sum(np.exp(coreml_full_logits))
            top5_indices = np.argsort(coreml_probs)[-5:][::-1]
            print("  Top 5 tokens (CoreML):")
            for rank, idx in enumerate(top5_indices, 1):
                decoded = tokenizer.decode([idx])
                print(f"    {rank}. {idx} ('{decoded}') prob: {coreml_probs[idx]:.4f}")
            
        else:
            print(f"❌ Expected 16 logits parts from CoreML, got {len(coreml_logits_parts)}")
            next_token_coreml = None
            
    except Exception as e:
        print(f"❌ CoreML inference failed: {e}")
        import traceback
        traceback.print_exc()
        next_token_coreml = None
    
    # === Method 3: PyTorch with KV Cache (Reference) ===
    print(f"\n🔧 Method 3: PyTorch with KV Cache (Reference)")
    
    model_pytorch_kv = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=False)
    model_pytorch_kv.load_pretrained_weights(model_dir)
    model_pytorch_kv.eval()
    model_pytorch_kv.model.kv_cache_0.zero_()
    
    with torch.no_grad():
        logits_pytorch_kv = model_pytorch_kv(
            input_ids=padded_input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
    
    next_token_pytorch_kv = torch.argmax(logits_pytorch_kv[0, seq_len - 1, :]).item()
    print(f"  Result: {next_token_pytorch_kv} ('{tokenizer.decode([next_token_pytorch_kv])}')")
    
    # === Method 4: PyTorch Single Token (Fair Comparison with CoreML) ===
    print(f"\n🔬 Method 4: PyTorch Single Token (Fair Comparison with CoreML)")
    
    model_pytorch_single = QwenForCausalLM(config, enable_coreml=False, disable_kv_cache=True)
    model_pytorch_single.load_pretrained_weights(model_dir)
    model_pytorch_single.eval()
    
    # Use single token inputs (same as CoreML)
    single_input_ids = torch.tensor([[original_tokens[0]]], dtype=torch.long, device=TEST_DEVICE)  # Single token
    single_position_ids = torch.tensor([0], dtype=torch.long, device=TEST_DEVICE)  # Position 0
    single_current_pos = torch.tensor([0], dtype=torch.long, device=TEST_DEVICE)  # Current position 0
    # Single token causal mask should be [1, 1, 1, CONTEXT_LENGTH] for consistency
    single_causal_mask = torch.zeros((1, 1, 1, CONTEXT_LENGTH), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    # For first token, allow attention to position 0 only, block future positions
    for j in range(1, CONTEXT_LENGTH):
        single_causal_mask[0, 0, 0, j] = float('-inf')
    single_update_mask = torch.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
    single_update_mask[0, 0, 0, 0] = 1.0  # Update only position 0
    
    print(f"    PyTorch single token shapes:")
    print(f"      input_ids: {single_input_ids.shape}")
    print(f"      position_ids: {single_position_ids.shape}")
    print(f"      current_pos: {single_current_pos.shape}")
    print(f"      causal_mask: {single_causal_mask.shape}")
    print(f"      update_mask: {single_update_mask.shape}")
    
    with torch.no_grad():
        logits_pytorch_single = model_pytorch_single(
            input_ids=single_input_ids,
            update_mask=single_update_mask,
            position_ids=single_position_ids,
            causal_mask=single_causal_mask,
            current_pos=single_current_pos,
            IN_PREFILL=False
        )
    
    next_token_pytorch_single = torch.argmax(logits_pytorch_single[0, 0, :]).item()
    print(f"  Result: {next_token_pytorch_single} ('{tokenizer.decode([next_token_pytorch_single])}')")

    # Print top 5 tokens for PyTorch single token
    logits_pytorch_single_last = logits_pytorch_single[0, 0, :]
    probs_pytorch_single = torch.softmax(logits_pytorch_single_last, dim=-1)
    topk_pytorch_single = torch.topk(probs_pytorch_single, 5)
    print("  Top 5 tokens (PyTorch Single):")
    for rank, (idx, prob) in enumerate(zip(topk_pytorch_single.indices.tolist(), topk_pytorch_single.values.tolist()), 1):
        decoded = tokenizer.decode([idx])
        print(f"    {rank}. {idx} ('{decoded}') prob: {prob:.4f}")
    
    # === Compare Results ===
    print(f"\n📊 COMPARISON RESULTS")
    print("=" * 50)
    print(f"  Method 1 (PyTorch, no KV, padded):  {next_token_pytorch} ('{tokenizer.decode([next_token_pytorch])}')")
    if next_token_coreml is not None:
        print(f"  Method 2 (CoreML):                  {next_token_coreml} ('{tokenizer.decode([next_token_coreml])}')")
    else:
        print(f"  Method 2 (CoreML):                  FAILED")
    print(f"  Method 3 (PyTorch, KV, padded):     {next_token_pytorch_kv} ('{tokenizer.decode([next_token_pytorch_kv])}')")
    print(f"  Method 4 (PyTorch, single token):   {next_token_pytorch_single} ('{tokenizer.decode([next_token_pytorch_single])}')")
    
    # Check matches
    matches = []
    if next_token_pytorch == next_token_pytorch_kv:
        matches.append("Padded PyTorch no-KV == PyTorch KV")
    if next_token_coreml is not None and next_token_pytorch_single == next_token_coreml:
        matches.append("Single Token PyTorch == CoreML")
    if next_token_coreml is not None and next_token_pytorch == next_token_coreml:
        matches.append("Padded PyTorch == CoreML")
    
    if matches:
        print(f"\n✅ MATCHES: {', '.join(matches)}")
        if "Single Token PyTorch == CoreML" in matches:
            print(f"🎉 FAIR COMPARISON PASSED! Single token processing matches between PyTorch and CoreML!")
            if len(matches) >= 2:
                return True
        else:
            print(f"⚠️  Partial match - some methods differ")
            return False
    else:
        print(f"\n❌ NO MATCHES - All methods produce different results")
        print(f"\nThis indicates a problem with:")
        if next_token_pytorch != next_token_pytorch_kv:
            print(f"  - PyTorch KV cache implementation")
        if next_token_coreml is not None and next_token_pytorch_single != next_token_coreml:
            print(f"  - CoreML vs PyTorch single token processing")
        return False

if __name__ == "__main__":
    from anemll.models.qwen_model import TEST_DEVICE, MODEL_DTYPE
    success = test_coreml_vs_pytorch()
    if success:
        print(f"\n🎉 CoreML vs PyTorch test PASSED!")
    else:
        print(f"\n❌ CoreML vs PyTorch test FAILED!") 
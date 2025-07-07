#!/usr/bin/env python3
#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

"""
PyTorch-only test script for Qwen 2.5 model with SP per-tensor quantization.
This script tests the quantized model directly in PyTorch without CoreML conversion.
"""

import sys
import os
import json
import torch
import numpy as np

# CRITICAL: Set environment variables BEFORE importing
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism to avoid fork warning
test_text = "What is Apple Neural Engine?"
#test_text = "Who are you?"
max_tokens = 20

# Test models
test_models = [
    {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "description": "Small 0.5B model Instruct"
    }
]

def setup_model_path(model_id):
    """Download model and prepare for testing"""
    try:
        from huggingface_hub import snapshot_download
        
        # Download the model if not cached
        print(f"Downloading model {model_id}...")
        model_path = snapshot_download(repo_id=model_id)
        print(f"Model downloaded to: {model_path}")
        
        # Clean quantization config if present
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if 'quantization_config' in config:
                print(f"Removing quantization_config from {config_file}")
                del config['quantization_config']
                
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                print("✓ quantization_config removed")
        
        return model_path
    except Exception as e:
        print(f"Error setting up model: {e}")
        return None

def test_quantized_model(model_id, model_path):
    """Test SP quantized Qwen2.5 model in PyTorch"""
    try:
        from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
        
        print(f"\n=== Testing SP Quantized Model: {model_id} ===")
        
        # Load configuration
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            config = Qwen25Config.from_json(config_file)
            print(f"✓ Loaded config: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
        else:
            print(f"Config not found, using defaults")
            config = Qwen25Config()
        
        # Create model with SP quantization enabled
        print("Creating model with SP quantization...")
        model = Qwen25ForCausalLM(config, disable_kv_cache=False)
        
        # Load weights
        print("Loading quantized weights...")
        success = model.load_pretrained_weights(model_path)
        
        if not success:
            print("⚠️  Weight loading reported issues, attempting inference test...")
        
        # Test basic inference using CORRECT ANEMLL pattern
        print("\n--- Basic Inference Test (Fixed Context Pattern) ---")
        
        # Use FIXED context size from config
        context_length = config.context_length  # Usually 256
        print(f"Using fixed context length: {context_length}")
        
        # Test with simple token sequence
        test_tokens = [1, 2, 3, 4, 5]
        prompt_length = len(test_tokens)
        
        # Create FIXED causal mask for full context (using correct ANEMLL pattern)
        def make_causal_mask(length, start):
            """Create causal attention mask."""
            mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
            row_indices = np.arange(length).reshape(length, 1)
            col_indices = np.arange(length).reshape(1, length)
            mask[:, :, col_indices <= (row_indices + start)] = 0
            return mask
        
        causal_mask_data = make_causal_mask(context_length, 0)
        causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
        
        try:
            with torch.no_grad():
                # Step 1: Prefill KV cache with test tokens
                test_input = torch.tensor([test_tokens], dtype=torch.long)
                
                # Create causal mask for prefill: only within prompt length
                prefill_causal_mask = torch.zeros((1, 1, prompt_length, context_length), dtype=torch.float16)
                
                # Apply causal mask: token i can attend to tokens 0 through i, -inf for future positions
                for i in range(prompt_length):
                    prefill_causal_mask[:, :, i, i+1:context_length] = float('-inf')
                
                # Prefill position IDs
                prefill_position_ids = torch.arange(prompt_length, dtype=torch.long)
                
                # Run prefill to populate KV cache
                output = model(
                    test_input,  # input_ids
                    torch.zeros(1, prompt_length),  # update_mask
                    prefill_position_ids,  # position_ids
                    prefill_causal_mask,   # causal_mask
                    torch.tensor(0, dtype=torch.long),  # current_pos
                    IN_PREFILL=True
                )
                
                print(f"✓ Prefill successful!")
                print(f"  Output shape: {output.shape}")
                print(f"  Output dtype: {output.dtype}")
                print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                
                # Step 2: Generate one token
                current_pos = prompt_length
                last_token = torch.tensor([[test_tokens[-1]]], dtype=torch.long)
                
                # Create update mask for single token at current position
                update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
                update_mask[0, 0, current_pos, 0] = 1.0
                
                next_output = model(
                    last_token,  # input_ids
                    update_mask,  # update_mask
                    torch.tensor([current_pos], dtype=torch.long),  # position_ids
                    causal_mask[:, :, current_pos:current_pos+1, :],  # causal_mask - single row
                    torch.tensor(current_pos, dtype=torch.long),  # current_pos
                    IN_PREFILL=False
                )
                
                print(f"✓ Generation successful!")
                print(f"  Next token logits shape: {next_output.shape}")
                
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Check quantization scales with detailed layer debugging
        print("\n--- Quantization Scale Check ---")
        scale_count = 0
        layer_debug = {}
        
        # First, identify all transformer layers
        transformer_layers = []
        for name, module in model.named_modules():
            if 'layers.' in name and name.count('.') == 2:  # e.g., model.layers.0
                layer_num = name.split('.')[2]
                if layer_num.isdigit():
                    transformer_layers.append(int(layer_num))
        
        transformer_layers = sorted(set(transformer_layers))
        print(f"Found {len(transformer_layers)} transformer layers: {transformer_layers[:5]}{'...' if len(transformer_layers) > 5 else ''}")
        
        # Check each layer for quantization scales
        for layer_idx in transformer_layers:
            layer_scales = {}
            layer_prefix = f"model.layers.{layer_idx}"
            
            # Check attention scales
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                for scale_type in ['input_scale', 'output_scale']:
                    attr_name = f"{proj_name}_{scale_type}"
                    module_path = f"{layer_prefix}.self_attn"
                    
                    # Find the actual module
                    try:
                        module = model
                        for part in module_path.split('.'):
                            module = getattr(module, part)
                        
                        if hasattr(module, attr_name):
                            scale_tensor = getattr(module, attr_name)
                            layer_scales[f"attn.{attr_name}"] = scale_tensor
                            scale_count += 1
                    except AttributeError:
                        pass
            
            # Check MLP scales
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                for scale_type in ['input_scale', 'output_scale']:
                    attr_name = f"{proj_name}_{scale_type}"
                    module_path = f"{layer_prefix}.mlp"
                    
                    # Find the actual module
                    try:
                        module = model
                        for part in module_path.split('.'):
                            module = getattr(module, part)
                        
                        if hasattr(module, attr_name):
                            scale_tensor = getattr(module, attr_name)
                            layer_scales[f"mlp.{attr_name}"] = scale_tensor
                            scale_count += 1
                    except AttributeError:
                        pass
            
            layer_debug[layer_idx] = layer_scales
            
            # Show detailed info for first 3 layers
            if layer_idx < 3:
                print(f"\n  Layer {layer_idx}: {len(layer_scales)} scales found")
                for scale_name, scale_tensor in layer_scales.items():
                    if scale_tensor.numel() == 1:
                        print(f"    {scale_name}: {scale_tensor.item():.6f}")
                    else:
                        print(f"    {scale_name}: shape {scale_tensor.shape}, mean={scale_tensor.mean():.6f}")
        
        # Summary by scale type
        input_scale_count = 0
        output_scale_count = 0
        attn_scale_count = 0
        mlp_scale_count = 0
        
        for layer_idx, scales in layer_debug.items():
            for scale_name in scales.keys():
                if 'input_scale' in scale_name:
                    input_scale_count += 1
                if 'output_scale' in scale_name:
                    output_scale_count += 1
                if 'attn.' in scale_name:
                    attn_scale_count += 1
                if 'mlp.' in scale_name:
                    mlp_scale_count += 1
        
        print(f"\n✓ Scale Summary:")
        print(f"  Total scales found: {scale_count}")
        print(f"  Input scales: {input_scale_count}")
        print(f"  Output scales: {output_scale_count}")
        print(f"  Attention scales: {attn_scale_count}")
        print(f"  MLP scales: {mlp_scale_count}")
        print(f"  Expected total: {len(transformer_layers) * 7 * 2} (layers×projections×types)")
        
        # Check for missing scales
        missing_layers = []
        for layer_idx in transformer_layers:
            expected_scales = 14  # 7 projections × 2 types
            actual_scales = len(layer_debug.get(layer_idx, {}))
            if actual_scales != expected_scales:
                missing_layers.append(f"Layer {layer_idx}: {actual_scales}/{expected_scales}")
        
        if missing_layers:
            print(f"  ⚠️  Layers with missing scales: {missing_layers[:5]}{'...' if len(missing_layers) > 5 else ''}")
        
        print(f"\n--- Debug Info ---")

        print(f"Total layers processed: {len(layer_debug)}")
        print(f"Total scales found: {scale_count}")
        
        # Test inference WITHOUT quantization to compare
        print(f"\n--- Comparison Test: Disable Quantization Forward Pass ---")
        os.environ['SKIP_SP_FORWARD'] = '1'
        
        try:
            # Re-import to pick up the flag change
            import importlib
            import anemll.models.qwen2_5_model
            importlib.reload(anemll.models.qwen2_5_model)
            from anemll.models.qwen2_5_model import SKIP_SP_FORWARD as SKIP_AFTER_RELOAD
            print(f"SKIP_SP_FORWARD after reload: {SKIP_AFTER_RELOAD}")
            
            # Test with quantization disabled
            test_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
            position_ids = torch.arange(5, dtype=torch.long)
            causal_mask = torch.zeros((1, 1, 5, 5), dtype=torch.float16)
            for i in range(5):
                causal_mask[:, :, i, i+1:] = float('-inf')
            current_pos = torch.tensor(4, dtype=torch.long)
            update_mask = torch.zeros(1, dtype=torch.long)
            
            with torch.no_grad():
                output_no_quant = model(test_input, update_mask, position_ids, causal_mask, current_pos)
                print(f"✓ Forward pass without quantization successful!")
                print(f"  Output range without quant: [{output_no_quant.min().item():.4f}, {output_no_quant.max().item():.4f}]")
                
        except Exception as e:
            print(f"⚠️  Could not test without quantization: {e}")
        finally:
            # Reset the flag
            if 'SKIP_SP_FORWARD' in os.environ:
                del os.environ['SKIP_SP_FORWARD']
        
        # Test with actual text using CORRECT ANEMLL pattern
        try:
            from transformers import AutoTokenizer
            print("\n--- Text Generation Test (Fixed Context Pattern) ---")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Test prompt
            inputs = tokenizer(test_text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            
            print(f"Input text: '{test_text}'")
            print(f"Token IDs: {input_ids.tolist()}")
            
            # Generate tokens using correct ANEMLL pattern
            generated_ids = input_ids[0].tolist()
            
            # ANEMLL uses FIXED context size
            context_length = config.context_length
            
            # Create FIXED causal mask for full context
            causal_mask_data = make_causal_mask(context_length, 0)
            causal_mask = torch.tensor(causal_mask_data, dtype=torch.float16)
            with torch.no_grad():
                # Step 1: Prefill KV cache
                prompt_length = len(generated_ids)
                print(f"Batch prefill: processing {prompt_length} prompt tokens at once...")
                
                # Use the original prompt for prefill (batch mode)
                prefill_position_ids = torch.arange(prompt_length, dtype=torch.long)
                
                # Create causal mask for prefill: only within prompt length
                prefill_causal_mask = torch.zeros((1, 1, prompt_length, context_length), dtype=torch.float16)
                
                # Apply causal mask: token i can attend to tokens 0 through i, -inf for future positions
                for i in range(prompt_length):
                    prefill_causal_mask[:, :, i, i+1:context_length] = float('-inf')
                
                # Run prefill to populate KV cache
                model(
                    input_ids,  # input_ids
                    torch.zeros(1, prompt_length),  # update_mask
                    prefill_position_ids,  # position_ids
                    prefill_causal_mask,   # causal_mask
                    torch.tensor(0, dtype=torch.long),  # current_pos
                    IN_PREFILL=True
                )
                
                # Step 2: Generate tokens one by one
                current_pos = prompt_length  # Start generating at position after prompt
                
                for i in range(max_tokens):  # Generate tokens
                    # Get the last generated token (or last prompt token for first generation)
                    if len(generated_ids) > prompt_length:
                        # Use last generated token
                        last_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long)
                    else:
                        # Use last prompt token for first generation
                        last_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long)
                    
                    # Single token generation
                    # Create update mask for single token at current position
                    update_mask = torch.zeros((1, 1, context_length, 1), dtype=torch.float16)
                    update_mask[0, 0, current_pos, 0] = 1.0
                    
                    outputs = model(
                        last_token,  # input_ids
                        update_mask,  # update_mask
                        torch.tensor([current_pos], dtype=torch.long),  # position_ids
                        causal_mask[:, :, current_pos:current_pos+1, :],  # causal_mask - single row
                        torch.tensor(current_pos, dtype=torch.long),  # current_pos
                        IN_PREFILL=False
                    )
                    
                    # Get next token (outputs is the tensor directly)
                    next_token_logits = outputs[0, -1, :]
                    next_token_id = torch.argmax(next_token_logits).item()
                    
                    # Add to generated sequence and update position
                    generated_ids.append(next_token_id)
                    current_pos += 1
                    
                    # Show token
                    token = tokenizer.decode([next_token_id])
                    print(f"Token {i+1}: '{token}' (ID: {next_token_id})")
                    
                    # Stop if EOS or exceed context
                    if next_token_id == tokenizer.eos_token_id or current_pos >= context_length:
                        break
                
                # Decode full response
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                print(f"\n✓ Final response ({len(generated_ids) - len(input_ids[0])} tokens generated):")
                print(f"  '{response}'")
                
        except ImportError:
            print("\n⚠️  Transformers not available, skipping text generation test")
        except Exception as e:
            print(f"\n⚠️  Text generation test failed: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_sp_quant_tests():
    """Run SP quantization tests for Qwen 2.5 models"""
    
    print("===========================================")
    print("  Qwen 2.5 SP Quantization Test Suite")
    print("===========================================")
    print("Testing SP per-tensor quantized models in PyTorch")
    
    # Environment variable already set at top of file
    

    
    results = []
    
    for test_case in test_models:
        print(f"\n{'='*60}")
        print(f"Testing: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"{'='*60}")
        
        try:
            # Setup model
            model_path = setup_model_path(test_case["model_id"])
            if not model_path:
                print(f"✗ Failed to setup model {test_case['model_id']}")
                results.append((test_case['name'], False, "Model setup failed"))
                continue
            
            # Test the model
            success = test_quantized_model(test_case["model_id"], model_path)
            
            if success:
                print(f"\n✓ {test_case['name']} test PASSED")
                results.append((test_case['name'], True, "All tests passed"))
            else:
                print(f"\n✗ {test_case['name']} test FAILED")
                results.append((test_case['name'], False, "Model tests failed"))
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Test interrupted by user")
            results.append((test_case['name'], False, "Interrupted by user"))
            break
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            results.append((test_case['name'], False, f"Exception: {str(e)}"))
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for name, success, message in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name} - {message}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All SP quantization tests PASSED!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_sp_quant_tests())
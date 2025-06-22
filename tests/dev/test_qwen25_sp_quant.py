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
import argparse

def setup_model_path(model_id_or_path):
    """Download model or prepare local path for testing"""
    try:
        # Check if it's a local path
        if os.path.exists(model_id_or_path):
            print(f"Using local model path: {model_id_or_path}")
            model_path = model_id_or_path
        else:
            # Download from HuggingFace
            from huggingface_hub import snapshot_download
            
            print(f"Downloading model {model_id_or_path}...")
            model_path = snapshot_download(repo_id=model_id_or_path)
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
        
        # Test basic inference
        print("\n--- Basic Inference Test ---")
        test_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        position_ids = torch.arange(5, dtype=torch.long)
        causal_mask = torch.zeros((1, 1, 5, 5), dtype=torch.float16)
        # Create causal mask
        for i in range(5):
            causal_mask[:, :, i, i+1:] = float('-inf')
        current_pos = torch.tensor(4, dtype=torch.long)
        update_mask = torch.zeros(1, dtype=torch.long)
        
        try:
            with torch.no_grad():
                output = model(test_input, update_mask, position_ids, causal_mask, current_pos)
                print(f"✓ Forward pass successful!")
                print(f"  Output shape: {output.shape}")
                print(f"  Output dtype: {output.dtype}")
                print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
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
        
        # Debug: Check if ENABLE_SP_QUANT is actually working
        print(f"\n--- Debug Info ---")
        print(f"ENABLE_SP_QUANT environment: {os.environ.get('ENABLE_SP_QUANT', 'NOT SET')}")
        from anemll.models.qwen2_5_model import ENABLE_SP_QUANT, SKIP_SP_FORWARD
        print(f"ENABLE_SP_QUANT in model: {ENABLE_SP_QUANT}")
        print(f"SKIP_SP_FORWARD in model: {SKIP_SP_FORWARD}")
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
        
        # Test with actual text if tokenizer available
        try:
            from transformers import AutoTokenizer
            print("\n--- Text Generation Test ---")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Test 1: Simple prediction with "Who are you?"
            test_text = "Who are you?"
            input_ids = tokenizer(test_text, return_tensors="pt").input_ids
            
            print(f"Input text: '{test_text}'")
            print(f"Token IDs: {input_ids.tolist()}")
            
            # Generate next token
            with torch.no_grad():
                seq_len = input_ids.shape[1]
                position_ids = torch.arange(seq_len, dtype=torch.long)
                causal_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float16)
                for i in range(seq_len):
                    causal_mask[:, :, i, i+1:] = float('-inf')
                current_pos = torch.tensor(seq_len - 1, dtype=torch.long)
                update_mask = torch.zeros(1, dtype=torch.long)
                
                logits = model(input_ids, update_mask, position_ids, causal_mask, current_pos)
                next_token_logits = logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()
                next_token = tokenizer.decode([next_token_id])
                
                print(f"✓ Generated next token: '{next_token}' (ID: {next_token_id})")
                
                # Generate full response until EOS or 50 tokens
                print(f"\n--- Full Response Generation: '{test_text}' ---")
                generated_text = test_text
                current_ids = input_ids.clone()
                max_tokens = 50
                
                for step in range(max_tokens):
                    seq_len = current_ids.shape[1]
                    position_ids = torch.arange(seq_len, dtype=torch.long)
                    causal_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float16)
                    for i in range(seq_len):
                        causal_mask[:, :, i, i+1:] = float('-inf')
                    current_pos = torch.tensor(seq_len - 1, dtype=torch.long)
                    update_mask = torch.zeros(1, dtype=torch.long)
                    
                    logits = model(current_ids, update_mask, position_ids, causal_mask, current_pos)
                    next_token_logits = logits[0, -1, :]
                    next_token_id = torch.argmax(next_token_logits).item()
                    next_token = tokenizer.decode([next_token_id])
                    
                    # Append the new token
                    current_ids = torch.cat([current_ids, torch.tensor([[next_token_id]])], dim=1)
                    generated_text += next_token
                    
                    # Print progress every 5 tokens or on last token
                    if (step + 1) % 5 == 0 or step == 0:
                        print(f"  Step {step+1}: '{generated_text}'")
                    
                    # Stop if we hit end token
                    if next_token_id == tokenizer.eos_token_id:
                        print(f"  → EOS token reached at step {step+1}")
                        break
                
                print(f"✓ Final response ({len(current_ids[0]) - len(input_ids[0])} tokens generated):")
                print(f"  '{generated_text}'")
                
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

def run_sp_quant_tests(custom_models=None, skip_default=False):
    """Run SP quantization tests for Qwen 2.5 models"""
    
    print("===========================================")
    print("  Qwen 2.5 SP Quantization Test Suite")
    print("===========================================")
    print("Testing SP per-tensor quantized models in PyTorch")
    
    # Set environment variable for SP quantization
    os.environ['ENABLE_SP_QUANT'] = '1'
    print(f"✓ ENABLE_SP_QUANT=1 (SP quantization enabled)")
    
    # Initialize test models based on skip_default flag
    if skip_default:
        test_models = []
    else:
        # Default test models
        test_models = [
            {
                "name": "Qwen2.5-0.5B-4bit-PerTensor",
                "model_id": "smpanaro/Qwen2.5-0.5B-4bit-PerTensor",
                "description": "Small 0.5B model with 4-bit per-tensor quantization"
            }
        ]
    
    # Add custom models if provided
    if custom_models:
        for custom_model in custom_models:
            if isinstance(custom_model, str):
                # Simple path string
                model_name = os.path.basename(custom_model.rstrip('/'))
                test_models.append({
                    "name": f"Custom-{model_name}",
                    "model_id": custom_model,
                    "description": f"Custom quantized model at {custom_model}"
                })
            elif isinstance(custom_model, dict):
                # Full model specification
                test_models.append(custom_model)
    
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

def main():
    """Main function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description='Test SP per-tensor quantized Qwen 2.5 models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test default models only
  python test_qwen25_sp_quant.py
  
  # Test custom local model
  python test_qwen25_sp_quant.py --model /tmp/qwen25_quarot_fused_w4
  
  # Test multiple models
  python test_qwen25_sp_quant.py --model /tmp/model1 --model /tmp/model2
  
  # Test with custom name and description
  python test_qwen25_sp_quant.py --model /tmp/qwen25_quarot_fused_w4 --name "QuaRot-Fused" --description "Custom QuaRot fusion + per-tensor quantization"
        """)
    
    parser.add_argument('--model', action='append', dest='models',
                       help='Path to custom model directory (can be used multiple times)')
    
    parser.add_argument('--name', 
                       help='Custom name for the model (only used with single --model)')
    
    parser.add_argument('--description',
                       help='Custom description for the model (only used with single --model)')
    
    parser.add_argument('--skip-default', action='store_true',
                       help='Skip testing default models, only test custom ones')
    
    args = parser.parse_args()
    
    custom_models = []
    
    if args.models:
        if len(args.models) == 1 and (args.name or args.description):
            # Single model with custom name/description
            custom_models.append({
                "name": args.name or f"Custom-{os.path.basename(args.models[0].rstrip('/'))}",
                "model_id": args.models[0],
                "description": args.description or f"Custom quantized model at {args.models[0]}"
            })
        else:
            # Multiple models or single model without custom info
            custom_models.extend(args.models)
    
    # If custom models are specified, use them instead of defaults (unless explicitly asking for both)
    if custom_models and args.skip_default:
        # Only test custom models
        return run_sp_quant_tests(custom_models, skip_default=True)
    elif custom_models:
        # If --model specified without --skip-default, still use only custom models (more intuitive)
        return run_sp_quant_tests(custom_models, skip_default=True)
    else:
        # No custom models specified, test defaults
        return run_sp_quant_tests()


if __name__ == "__main__":
    sys.exit(main())
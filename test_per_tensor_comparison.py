#!/usr/bin/env python3
"""Per-tensor comparison between our qwen_model.py and official Qwen3 implementation."""

import numpy as np
import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from anemll.models.qwen_model import QwenForCausalLM, QwenConfig
import warnings
warnings.filterwarnings('ignore')

def compare_tensor_similarity(tensor1, tensor2, name, tolerance=1e-4):
    """Compare two tensors and return similarity metrics."""
    if tensor1.shape != tensor2.shape:
        print(f"❌ {name}: Shape mismatch - {tensor1.shape} vs {tensor2.shape}")
        return False
    
    # Convert to same device and dtype
    tensor1 = tensor1.cpu().float()
    tensor2 = tensor2.cpu().float()
    
    # Compute differences
    abs_diff = torch.abs(tensor1 - tensor2)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    rel_diff = (abs_diff / (torch.abs(tensor1) + 1e-8)).mean().item()
    
    # Check if tensors are similar
    similar = max_diff < tolerance
    
    status = "✅" if similar else "❌"
    print(f"{status} {name}:")
    print(f"    Shape: {tensor1.shape}")
    print(f"    Max diff: {max_diff:.6f}")
    print(f"    Mean diff: {mean_diff:.6f}")
    print(f"    Rel diff: {rel_diff:.6f}")
    
    if not similar:
        print(f"    Sample values T1: {tensor1.flatten()[:5]}")
        print(f"    Sample values T2: {tensor2.flatten()[:5]}")
    
    return similar

def load_models():
    """Load both our custom and official models."""
    print("🔄 Loading models for comparison...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    
    # Load official model
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    print(f"Loading official model from: {model_path}")
    official_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.float32  # Use float32 for better precision in comparison
    )
    official_model = official_model.to("cpu")
    official_model.eval()
    
    # Load our custom model
    print(f"Loading our custom model from: {model_path}")
    config = QwenConfig.from_json(os.path.join(model_path, "config.json"))
    config.context_length = 256
    custom_model = QwenForCausalLM(config, enable_coreml=False)  # Disable CoreML mode for fair comparison
    success = custom_model.load_pretrained_weights(model_path)
    if not success:
        raise RuntimeError("Failed to load custom model weights")
    custom_model.eval()
    
    print("✅ Both models loaded successfully")
    return tokenizer, official_model, custom_model

def prepare_test_inputs(tokenizer):
    """Prepare identical test inputs for both models."""
    # Use the same test token
    test_token = 3838  # "What"
    
    # Simple case: single token + padding
    CONTEXT_LENGTH = 256
    PAD_TOKEN = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # Official model inputs
    input_ids_official = torch.tensor([[test_token] + [PAD_TOKEN] * (CONTEXT_LENGTH - 1)], dtype=torch.long)
    attention_mask_official = torch.zeros_like(input_ids_official)
    attention_mask_official[0, 0] = 1  # Only first position is real
    
    # Custom model inputs
    input_ids_custom = input_ids_official.clone()
    position_ids_custom = torch.arange(CONTEXT_LENGTH, dtype=torch.long)
    
    # Create causal mask for custom model
    causal_mask_custom = torch.full((1, 1, CONTEXT_LENGTH, CONTEXT_LENGTH), -float('inf'), dtype=torch.float32)
    for i in range(CONTEXT_LENGTH):
        for j in range(i + 1):
            causal_mask_custom[0, 0, i, j] = 0
    
    current_pos_custom = torch.tensor([0], dtype=torch.long)
    update_mask_custom = torch.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=torch.float32)
    
    return {
        'official': {
            'input_ids': input_ids_official,
            'attention_mask': attention_mask_official,
        },
        'custom': {
            'input_ids': input_ids_custom,
            'position_ids': position_ids_custom,
            'causal_mask': causal_mask_custom,
            'current_pos': current_pos_custom,
            'update_mask': update_mask_custom,
        }
    }

def extract_intermediate_outputs(model, inputs, model_type):
    """Extract intermediate outputs from the model forward pass."""
    outputs = {}
    
    if model_type == 'official':
        # Hook into official model layers
        def create_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    outputs[name] = output[0].detach().clone()
                else:
                    outputs[name] = output.detach().clone()
            return hook
        
        # Register hooks
        hooks = []
        hooks.append(model.model.embed_tokens.register_forward_hook(create_hook('embeddings')))
        for i, layer in enumerate(model.model.layers[:3]):  # First 3 layers for manageable output
            hooks.append(layer.register_forward_hook(create_hook(f'layer_{i}')))
        hooks.append(model.model.norm.register_forward_hook(create_hook('final_norm')))
        
        # Forward pass
        with torch.no_grad():
            result = model(**inputs)
            outputs['final_logits'] = result.logits[0, 0, :].detach().clone()  # Extract position 0
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
    else:  # custom model
        # Hook into custom model layers
        def create_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    outputs[name] = output[0].detach().clone()
                else:
                    outputs[name] = output.detach().clone()
            return hook
        
        # Register hooks
        hooks = []
        hooks.append(model.model.embed_tokens.register_forward_hook(create_hook('embeddings')))
        for i, layer in enumerate(model.model.layers[:3]):  # First 3 layers
            hooks.append(layer.register_forward_hook(create_hook(f'layer_{i}')))
        hooks.append(model.model.norm.register_forward_hook(create_hook('final_norm')))
        
        # Forward pass
        with torch.no_grad():
            result = model(**inputs, IN_PREFILL=False)
            # Handle both tuple (CoreML mode) and single tensor output
            if isinstance(result, tuple):
                # Concatenate logits from 16 parts
                final_logits = torch.cat(result, dim=-1)[0, 0, :]
            else:
                final_logits = result[0, 0, :]
            outputs['final_logits'] = final_logits.detach().clone()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    return outputs

def compare_models():
    """Main comparison function."""
    print("🔬 Per-Tensor Comparison: Official vs Custom Qwen3")
    print("=" * 70)
    
    # Load models
    tokenizer, official_model, custom_model = load_models()
    
    # Prepare inputs
    inputs = prepare_test_inputs(tokenizer)
    print(f"\n📊 Test inputs prepared:")
    print(f"  Token: 3838 ('What')")
    print(f"  Context length: 256")
    print(f"  Input shape: {inputs['official']['input_ids'].shape}")
    
    # Extract intermediate outputs
    print(f"\n🔍 Extracting intermediate outputs...")
    official_outputs = extract_intermediate_outputs(official_model, inputs['official'], 'official')
    custom_outputs = extract_intermediate_outputs(custom_model, inputs['custom'], 'custom')
    
    print(f"Official model outputs: {list(official_outputs.keys())}")
    print(f"Custom model outputs: {list(custom_outputs.keys())}")
    
    # Compare each tensor
    print(f"\n🆚 TENSOR COMPARISONS:")
    print(f"=" * 50)
    
    all_similar = True
    tolerance = 1e-3  # Relaxed tolerance for float precision differences
    
    for key in ['embeddings', 'layer_0', 'layer_1', 'layer_2', 'final_norm', 'final_logits']:
        if key in official_outputs and key in custom_outputs:
            print(f"\n📈 Comparing {key}:")
            similar = compare_tensor_similarity(
                official_outputs[key], 
                custom_outputs[key], 
                key, 
                tolerance=tolerance
            )
            all_similar = all_similar and similar
        else:
            print(f"\n❌ Missing {key} in one of the models")
            all_similar = False
    
    # Final assessment
    print(f"\n📋 FINAL ASSESSMENT:")
    print(f"=" * 30)
    if all_similar:
        print(f"✅ Models are numerically equivalent (within tolerance {tolerance})")
        print(f"   → Issue is likely in CoreML conversion process")
    else:
        print(f"❌ Models have significant differences")
        print(f"   → Issue may be in our custom implementation")
        print(f"   → Need to investigate specific layer differences")
    
    # Compare final predictions
    print(f"\n🎯 PREDICTION COMPARISON:")
    print(f"  Official top 5 tokens: {torch.argsort(official_outputs['final_logits'], descending=True)[:5].tolist()}")
    print(f"  Custom top 5 tokens:   {torch.argsort(custom_outputs['final_logits'], descending=True)[:5].tolist()}")
    
    # Compare with previous results
    expected_tokens = [15846, 21806, 3405, 38297, 13355]  # From our previous tests
    official_top5 = torch.argsort(official_outputs['final_logits'], descending=True)[:5].tolist()
    custom_top5 = torch.argsort(custom_outputs['final_logits'], descending=True)[:5].tolist()
    
    print(f"\n🔍 CONSISTENCY CHECK:")
    print(f"  Expected tokens: {expected_tokens}")
    print(f"  Official tokens: {official_top5}")
    print(f"  Custom tokens:   {custom_top5}")
    print(f"  Official matches expected: {official_top5 == expected_tokens}")
    print(f"  Custom matches expected:   {custom_top5 == expected_tokens}")
    
    return all_similar

if __name__ == "__main__":
    try:
        success = compare_models()
        print(f"\n{'🎉 SUCCESS' if success else '🔧 INVESTIGATION NEEDED'}")
        if success:
            print("Models are equivalent → Focus on CoreML conversion debugging")
        else:
            print("Models differ → Fix custom implementation first")
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc() 
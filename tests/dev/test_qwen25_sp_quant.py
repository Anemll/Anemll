#!/usr/bin/env python3
"""Test script for Qwen 2.5 per-tensor quantization support."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub not installed. Please install it with:")
    print("  pip install huggingface_hub")
    sys.exit(1)

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config

def get_model_path():
    """Get model path, downloading from HuggingFace if not in cache."""
    model_id = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    
    # Check common cache locations
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    # Try to find the model in cache
    if cache_dir.exists():
        for item in cache_dir.iterdir():
            if item.is_dir() and "smpanaro" in str(item) and "Qwen2.5-0.5B-4bit-PerTensor" in str(item):
                # Found in cache, get the snapshots path
                snapshots_dir = item / "snapshots"
                if snapshots_dir.exists():
                    # Get the latest snapshot
                    snapshots = list(snapshots_dir.iterdir())
                    if snapshots:
                        model_path = snapshots[-1]
                        print(f"Found model in cache at: {model_path}")
                        return str(model_path)
    
    # Not in cache, download it using snapshot_download (not transformers)
    print(f"Model not found in cache. Downloading {model_id} from HuggingFace...")
    try:
        # Download just the files, don't load the model
        model_path = snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.h5", "*.ot", "*.msgpack"]  # Skip unnecessary files
        )
        print(f"Downloaded model files to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please ensure you have internet connection and huggingface_hub installed:")
        print("  pip install huggingface_hub")
        return None

def test_sp_quant_loading():
    """Test loading a per-tensor quantized Qwen 2.5 model."""
    
    # Get model path (download if needed)
    model_path = get_model_path()
    if not model_path:
        print("Failed to get model path")
        return
    
    # Load config
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        config = Qwen25Config.from_json(config_path)
    else:
        print(f"Config not found at {config_path}, using default config")
        config = Qwen25Config()
    
    # Create model
    model = Qwen25ForCausalLM(config)
    
    print("Loading quantized weights...")
    success = model.load_pretrained_weights(model_path)
    
    if success:
        print("✓ Model loaded successfully!")
        
        # Check if quantization scales were loaded
        print("\nChecking quantization scales:")
        for name, module in model.named_modules():
            if hasattr(module, 'gate_proj_output_scale'):
                print(f"  {name}.gate_proj_output_scale: {module.gate_proj_output_scale.item():.6f}")
            if hasattr(module, 'up_proj_output_scale'):
                print(f"  {name}.up_proj_output_scale: {module.up_proj_output_scale.item():.6f}")
            if hasattr(module, 'down_proj_output_scale'):
                print(f"  {name}.down_proj_output_scale: {module.down_proj_output_scale.item():.6f}")
            if hasattr(module, 'q_proj_output_scale'):
                print(f"  {name}.q_proj_output_scale: {module.q_proj_output_scale.item():.6f}")
            if hasattr(module, 'k_proj_output_scale'):
                print(f"  {name}.k_proj_output_scale: {module.k_proj_output_scale.item():.6f}")
            if hasattr(module, 'v_proj_output_scale'):
                print(f"  {name}.v_proj_output_scale: {module.v_proj_output_scale.item():.6f}")
            if hasattr(module, 'o_proj_output_scale'):
                print(f"  {name}.o_proj_output_scale: {module.o_proj_output_scale.item():.6f}")
        
        # Test a simple forward pass
        print("\nTesting forward pass...")
        test_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        position_ids = torch.arange(5, dtype=torch.long)
        causal_mask = torch.zeros((1, 1, 5, 5), dtype=torch.float16)
        causal_mask[:, :, torch.arange(5), torch.arange(5) + 1:] = float('-inf')
        current_pos = torch.tensor(4, dtype=torch.long)
        update_mask = torch.zeros(1, dtype=torch.long)
        
        with torch.no_grad():
            output = model(test_input, update_mask, position_ids, causal_mask, current_pos)
            print(f"  Output shape: {output.shape}")
            print(f"  Output dtype: {output.dtype}")
            print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        print("\n✓ Forward pass successful!")
        
        # Optional: Test with tokenizer if available
        try:
            from transformers import AutoTokenizer
            tokenizer_path = model_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            
            print("\nTesting with tokenizer...")
            test_text = "Hello, world!"
            input_ids = tokenizer(test_text, return_tensors="pt").input_ids
            print(f"  Input text: '{test_text}'")
            print(f"  Token IDs: {input_ids.tolist()}")
            
            # Generate one token
            with torch.no_grad():
                # Prepare inputs for single token generation
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
                
                print(f"  Next token ID: {next_token_id}")
                print(f"  Next token: '{next_token}'")
                print("\n✓ Tokenizer test successful!")
                
        except ImportError:
            print("\nTransformers not installed, skipping tokenizer test")
        except Exception as e:
            print(f"\nTokenizer test failed: {e}")
        
    else:
        print("✗ Failed to load model weights")

if __name__ == "__main__":
    test_sp_quant_loading()
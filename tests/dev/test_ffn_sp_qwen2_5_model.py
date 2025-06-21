#!/usr/bin/env python3
"""
Test script for FFN-only conversion of Qwen 2.5 with per-tensor quantization.
This script focuses only on the FFN conversion step to debug dtype issues.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add the anemll directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoConfig, AutoTokenizer
from anemll.models.qwen2_5_model import Qwen25Config, Qwen25ForCausalLM
from anemll.ane_converter.qwen2_5_converter import Qwen25Converter

def test_ffn_conversion():
    """Test FFN conversion with per-tensor quantization."""
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    output_dir = tempfile.mkdtemp(prefix="test-ffn-qwen25-sp-quant-")
    
    try:
        print(f"Testing FFN conversion for {model_name}")
        print(f"Output directory: {output_dir}")
        print(f"ENABLE_SP_QUANT: {os.environ.get('ENABLE_SP_QUANT', 'false')}")
        
        # Download model using huggingface_hub
        print(f"Downloading model {model_name}...")
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id=model_name)
        print(f"Model downloaded to: {model_dir}")
        
        # Load HuggingFace config and create our config
        print("Loading configuration...")
        hf_config = AutoConfig.from_pretrained(model_name)
        config = Qwen25Config(**hf_config.to_dict())
        
        # Create model instance
        print("Creating model...")
        model = Qwen25ForCausalLM(config, enable_coreml=True)
        
        # Load weights
        print("Loading pretrained weights...")
        if not model.load_pretrained_weights(model_dir):
            raise RuntimeError("Failed to load pretrained weights")
        
        print("Model loaded successfully!")
        
        # Set model to eval mode
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        print("Model set to eval mode and gradients disabled")
        
        # Create converter and test FFN conversion only
        print("Creating converter...")
        converter = Qwen25Converter(
            model=model,
            context_length=512
        )
        
        print("Starting FFN conversion...")
        try:
            # Convert only part 2 (FFN)
            mlmodel = converter.convert(part="2")
            print("✓ FFN conversion succeeded!")
            return True
            
        except Exception as e:
            print(f"✗ FFN conversion failed: {e}")
            if "dtype" in str(e):
                print("This appears to be a dtype mismatch error")
            return False
            
    except Exception as e:
        print(f"✗ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(output_dir):
            print(f"Cleaning up {output_dir}")
            shutil.rmtree(output_dir)

if __name__ == "__main__":
    # Ensure quantization is enabled
    os.environ['ENABLE_SP_QUANT'] = 'true'
    
    print("=" * 60)
    print("FFN-Only Qwen 2.5 Per-Tensor Quantization Test")
    print("=" * 60)
    
    success = test_ffn_conversion()
    
    if success:
        print("\n✓ FFN conversion test passed!")
        sys.exit(0)
    else:
        print("\n✗ FFN conversion test failed!")
        sys.exit(1)
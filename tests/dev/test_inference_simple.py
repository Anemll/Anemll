#!/usr/bin/env python3
"""
Simple inference test for quantized Qwen 2.5 models.
Tests text generation with custom prompts.
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer

def test_model_inference(model_path, prompt, max_tokens=50, temperature=1.0, backend="anemll"):
    """Test model inference with a custom prompt."""
    
    print(f"Testing model at: {model_path}")
    print(f"Backend: {backend.upper()}")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {max_tokens}")
    print("-" * 60)
    
    if backend == "transformers":
        return test_with_transformers(model_path, prompt, max_tokens, temperature)
    else:
        return test_with_anemll(model_path, prompt, max_tokens, temperature)


def test_with_transformers(model_path, prompt, max_tokens=50, temperature=1.0):
    """Test model inference using standard HuggingFace transformers."""
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("Loading model with HuggingFace transformers...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        
        print(f"✓ Model loaded successfully")
        
        # Tokenize input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        print(f"Input tokens: {input_ids.shape[1]} tokens")
        
        # Generate with transformers
        print("\n--- Generating Response with Transformers ---")
        
        with torch.no_grad():
            # Use transformers generate method
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode response
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
        print(f"✓ Generated text ({len(generated_ids[0]) - len(input_ids[0])} tokens):")
        print(f"'{generated_text}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Transformers inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_anemll(model_path, prompt, max_tokens=50, temperature=1.0):
    """Test model inference using ANEMLL."""
    
    try:
        # Import ANEMLL model
        from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
        
        # Load configuration
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            config = Qwen25Config.from_json(config_file)
            print(f"✓ Loaded config: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
        else:
            print("Config not found, using defaults")
            config = Qwen25Config()
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Create model with SP quantization
        print("Creating model with SP quantization...")
        model = Qwen25ForCausalLM(config, disable_kv_cache=False)
        
        # Load weights
        print("Loading quantized weights...")
        success = model.load_pretrained_weights(model_path)
        if not success:
            print("⚠️  Weight loading reported issues, continuing anyway...")
        
        # Tokenize input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        print(f"Input tokens: {input_ids.shape[1]} tokens")
        
        # Generate response
        print("\n--- Generating Response ---")
        generated_text = prompt
        current_ids = input_ids.clone()
        
        with torch.no_grad():
            for step in range(max_tokens):
                seq_len = current_ids.shape[1]
                position_ids = torch.arange(seq_len, dtype=torch.long)
                causal_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float16)
                
                # Create causal mask
                for i in range(seq_len):
                    causal_mask[:, :, i, i+1:] = float('-inf')
                
                current_pos = torch.tensor(seq_len - 1, dtype=torch.long)
                update_mask = torch.zeros(1, dtype=torch.long)
                
                # Forward pass
                logits = model(current_ids, update_mask, position_ids, causal_mask, current_pos)
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature and sample
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Get next token (greedy sampling for consistency)
                next_token_id = torch.argmax(next_token_logits).item()
                next_token = tokenizer.decode([next_token_id])
                
                # Append token
                current_ids = torch.cat([current_ids, torch.tensor([[next_token_id]])], dim=1)
                generated_text += next_token
                
                # Print progress every 10 tokens
                if (step + 1) % 10 == 0:
                    print(f"Step {step+1:2d}: '{generated_text}'")
                
                # Stop if EOS token
                if next_token_id == tokenizer.eos_token_id:
                    print(f"→ EOS token reached at step {step+1}")
                    break
        
        print(f"\n✓ Final output ({len(current_ids[0]) - len(input_ids[0])} tokens generated):")
        print(f"'{generated_text}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Simple inference test for quantized Qwen 2.5 models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with ANEMLL backend (default)
  python test_inference_simple.py --model /tmp/qwen25_quarot_fused_w4_fixed --prompt "What is Python?"
  
  # Test with HuggingFace transformers backend
  python test_inference_simple.py --model /tmp/qwen25_quarot_fused_w8 --prompt "What is Python?" --backend transformers
  
  # Compare backends with same prompt
  python test_inference_simple.py --model /tmp/qwen25_quarot_fused_w8 --prompt "Explain AI:" --backend anemll
  python test_inference_simple.py --model /tmp/qwen25_quarot_fused_w8 --prompt "Explain AI:" --backend transformers
  
  # Generate longer response
  python test_inference_simple.py --model /tmp/qwen25_quarot_fused_w4_fixed --prompt "Write a story:" --max-tokens 100 --backend anemll
        """)
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to quantized model directory')
    
    parser.add_argument('--prompt', type=str, default="Hello, how are you?",
                       help='Input prompt for text generation (default: "Hello, how are you?")')
    
    parser.add_argument('--max-tokens', type=int, default=50,
                       help='Maximum number of tokens to generate (default: 50)')
    
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (default: 1.0)')
    
    parser.add_argument('--backend', type=str, choices=['anemll', 'transformers'], default='anemll',
                       help='Inference backend: anemll (default) or transformers')
    
    args = parser.parse_args()
    
    # Set environment variable for SP quantization (only for ANEMLL backend)
    if args.backend == 'anemll':
        os.environ['ENABLE_SP_QUANT'] = '1'
        print(f"✓ ENABLE_SP_QUANT=1 (SP quantization enabled for ANEMLL)")
    else:
        print(f"✓ Using standard HuggingFace transformers backend")
    
    # Check if model path exists
    if not os.path.exists(args.model):
        print(f"Error: Model path does not exist: {args.model}")
        return 1
    
    # Run inference test
    success = test_model_inference(
        args.model, 
        args.prompt, 
        args.max_tokens, 
        args.temperature,
        args.backend
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
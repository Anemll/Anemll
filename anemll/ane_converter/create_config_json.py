#!/usr/bin/env python3
"""
Script to create a minimal config.json file required by iOS tokenizer.
"""

import os
import json
import argparse

def create_config_json(output_path, model_type="llama", tokenizer_class="LlamaTokenizer"):
    """
    Create a minimal config.json file required by iOS tokenizer.
    
    Args:
        output_path: Path where to save the config.json file
        model_type: The model type (default: "llama")
        tokenizer_class: The tokenizer class (default: "LlamaTokenizer")
    """
    config = {
        "tokenizer_class": tokenizer_class,
        "model_type": model_type
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write the config file
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created config.json at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create a minimal config.json file for iOS tokenizer")
    parser.add_argument("--output", "-o", required=True, help="Path to save the config.json file")
    parser.add_argument("--model-type", default="llama", help="Model type (default: llama)")
    parser.add_argument("--tokenizer-class", default="LlamaTokenizer", help="Tokenizer class (default: LlamaTokenizer)")
    
    args = parser.parse_args()
    create_config_json(args.output, args.model_type, args.tokenizer_class)

if __name__ == "__main__":
    main() 
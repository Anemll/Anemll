#!/usr/bin/env python3
"""
Modified version of fuse.py from MLX-LM that skips adapter loading when --de-quantize is used.
This allows the script to be used just for dequantization without needing adapters.
Saves only model.safetensors without redundant weight formats.
"""

import argparse
import glob
import shutil
from pathlib import Path
import os

import sys
sys.path.append("mlx-lm")

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

# Import from mlx_lm directly to avoid relative import issues
from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear
from mlx_lm.tuner.utils import dequantize, load_adapters
from mlx_lm.utils import (
    fetch_from_hub,
    get_model_path,
    save_config,
    save_model,
    upload_to_hub,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse fine-tuned adapters into the base model or dequantize a model."
    )
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--save-path",
        default="fused_model",
        help="The path to save the fused model.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="adapters",
        help="Path to the trained adapter weights and config (not needed for dequantization).",
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        default=None,
        help="Path to the original Hugging Face model. Required for upload if --model is a local directory.",
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--de-quantize",
        help="Generate a de-quantized model. No adapters needed.",
        action="store_true",
    )
    parser.add_argument(
        "--export-gguf",
        help="Export model weights in GGUF format.",
        action="store_true",
    )
    parser.add_argument(
        "--gguf-path",
        help="Path to save the exported GGUF format model weights. Default is ggml-model-f16.gguf.",
        default="ggml-model-f16.gguf",
        type=str,
    )
    return parser.parse_args()


def save_model_safetensors(save_path, model):
    """
    Save model weights only in safetensors format without redundant formats.
    """
    # Ensure save directory exists
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving weights to {save_path}")
    
    # Use save_model which is available in current MLX-LM
    print("Using save_model and then cleaning up redundant files")
    save_model(save_path, model)
    
    # Verify model.safetensors exists and is valid
    safetensors_path = save_path / "model.safetensors"
    if not safetensors_path.exists() or safetensors_path.stat().st_size < 1000:
        print("Error: model.safetensors is missing or invalid")
        return
    
    # Clean up redundant files
    weights_dir = save_path / "weights"
    weights_npz = save_path / "weights.npz"
    
    if weights_dir.exists():
        print(f"Removing redundant weights directory: {weights_dir}")
        shutil.rmtree(weights_dir)
        
    if weights_npz.exists():
        print(f"Removing redundant weights.npz file: {weights_npz}")
        os.remove(weights_npz)
    
    print("Cleanup complete - only model.safetensors remains")


def main() -> None:
    print("Loading pretrained model")
    args = parse_arguments()

    model_path = get_model_path(args.model)
    model, config, tokenizer = fetch_from_hub(model_path)

    model.freeze()
    
    # Skip adapter loading when only dequantizing
    if not args.de_quantize:
        print("Loading and fusing adapters...")
        model = load_adapters(model, args.adapter_path)

        fused_linears = [
            (n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")
        ]

        if fused_linears:
            model.update_modules(tree_unflatten(fused_linears))
    
    # Perform dequantization if requested
    if args.de_quantize:
        print("De-quantizing model")
        model = dequantize(model)

    save_path = Path(args.save_path)
    
    # Save only in safetensors format without redundant formats
    save_model_safetensors(save_path, model)
    
    # Copy Python files
    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, save_path)

    tokenizer.save_pretrained(save_path)

    if args.de_quantize:
        config.pop("quantization", None)
        config.pop("quantization_config", None)

    save_config(config, config_path=save_path / "config.json")

    if args.export_gguf:
        # Import here to avoid dependency for those not using this feature
        from mlx_lm.gguf import convert_to_gguf
        
        model_type = config["model_type"]
        if model_type not in ["llama", "mixtral", "mistral"]:
            raise ValueError(
                f"Model type {model_type} not supported for GGUF conversion."
            )
        convert_to_gguf(model_path, weights, config, str(save_path / args.gguf_path))

    if args.upload_repo is not None:
        hf_path = args.hf_path or (
            args.model if not Path(args.model).exists() else None
        )
        if hf_path is None:
            raise ValueError(
                "Must provide original Hugging Face repo to upload local model."
            )
        upload_to_hub(args.save_path, args.upload_repo, hf_path)


if __name__ == "__main__":
    main() 
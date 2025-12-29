#!/usr/bin/env python3
"""
Test A1F approach for Qwen transformer layers using integrated converter.

This test uses the proper convert_part_2 workflow with AQ1 checkpoint loading:
- Loads actual Qwen model from HuggingFace
- Uses QwenConverter with aq1_checkpoint parameter
- Replaces Conv2d with AnemllConv2d (custom op survives tracing)
- CoreML converter emits constexpr_lut_to_dense + matmul ops

Usage:
    # Test with 1-2 layers (for verification)
    ANEMLL_DYNAMIC_SCALES=1 python tests/dev/test_qwen_a1f_layer.py \
        --checkpoint /Users/anemll/Downloads/q2_pt_good1/snapped_lut/model_state_dict.pt \
        --model Qwen/Qwen3-0.6B \
        --layers 2

    # Full model test (all 28 layers)
    ANEMLL_DYNAMIC_SCALES=1 python tests/dev/test_qwen_a1f_layer.py \
        --checkpoint /Users/anemll/Downloads/q2_pt_good1/snapped_lut/model_state_dict.pt \
        --model Qwen/Qwen3-0.6B
"""

import os
import sys
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import coremltools as ct


def count_ops_by_type(mlmodel):
    """Count operations in CoreML model by type."""
    op_counts = {}
    spec = mlmodel.get_spec()

    if hasattr(spec, 'mlProgram'):
        prog_spec = spec.mlProgram
        for func_name, func in prog_spec.functions.items():
            for block_name, block in func.block_specializations.items():
                for op in block.operations:
                    op_type = op.type
                    op_counts[op_type] = op_counts.get(op_type, 0) + 1

    return op_counts


def print_ops(mlmodel, show_details=False):
    """Print operations in the model."""
    spec = mlmodel.get_spec()

    if hasattr(spec, 'mlProgram'):
        prog_spec = spec.mlProgram
        print("\n--- Model Operations ---")
        for func_name, func in prog_spec.functions.items():
            for block_name, block in func.block_specializations.items():
                for op in block.operations:
                    outputs = [out.name for out in op.outputs]
                    if show_details or op.type in ['constexpr_lut_to_dense', 'matmul', 'conv']:
                        print(f"  {op.type:30} -> {outputs[:2]}...")


def get_model_size(mlmodel):
    """Estimate model size from weights."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.mlpackage")
        mlmodel.save(path)
        total_size = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                total_size += os.path.getsize(os.path.join(root, f))
    return total_size


def main():
    parser = argparse.ArgumentParser(description='Test A1F integration with convert_part_2')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to AQ1 checkpoint with snapped weights + scales')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-0.6B',
                        help='HuggingFace model path')
    parser.add_argument('--layers', type=int, default=None,
                        help='Number of layers to convert (default: all)')
    parser.add_argument('--context', type=int, default=1024,
                        help='Context length')
    parser.add_argument('--output', type=str, default='/tmp/qwen_a1f_test.mlpackage',
                        help='Output path for converted model')
    parser.add_argument('--prefill', action='store_true',
                        help='Test prefill mode instead of generation mode')
    parser.add_argument('--chunk', type=int, default=0,
                        help='Chunk index to convert (0-based)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed operation info')
    args = parser.parse_args()

    # Check if dynamic mode is enabled
    if os.environ.get('ANEMLL_DYNAMIC_SCALES', '0') != '1':
        print("WARNING: ANEMLL_DYNAMIC_SCALES not set to 1!")
        print("Set ANEMLL_DYNAMIC_SCALES=1 to enable constexpr_lut_to_dense + matmul")
        print("Without it, weights will be pre-baked (no compression benefit)")
        print()

    print("=" * 60)
    print("A1F Integration Test - Qwen Converter with AQ1 Checkpoint")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model}")
    print(f"Layers: {args.layers or 'all'}")
    print(f"Context: {args.context}")
    print(f"Mode: {'prefill' if args.prefill else 'generation'}")
    print()

    # Import model and converter
    from anemll.models.qwen_model import QwenForCausalLM, QwenConfig, MODEL_DTYPE
    from anemll.ane_converter.qwen_converter import QwenConverter
    import glob
    from huggingface_hub import snapshot_download

    # Load model config from HuggingFace cache or download
    print(f"Loading Qwen model config from: {args.model}")

    # Try to find cached model first
    cache_pattern = os.path.expanduser(f'~/.cache/huggingface/hub/models--{args.model.replace("/", "--")}/snapshots/*')
    cached_dirs = glob.glob(cache_pattern)

    if cached_dirs:
        model_path = cached_dirs[0]
        print(f"  Found cached model: {model_path}")
    else:
        print(f"  Downloading model...")
        model_path = snapshot_download(args.model)
        print(f"  Downloaded to: {model_path}")

    # Load config from JSON file
    config_file = os.path.join(model_path, 'config.json')
    config = QwenConfig.from_json(config_file)
    config.context_length = args.context
    config.state_length = args.context

    print(f"  Config: hidden_size={config.hidden_size}, "
          f"num_layers={config.num_hidden_layers}, "
          f"num_heads={config.num_attention_heads}")

    # Create model - we don't load HF weights since AQ1 checkpoint will replace them
    print("\nCreating Qwen model...")
    model = QwenForCausalLM(config, enable_coreml=True)
    # Note: load_dynamic_weights_for_ane() will load the AQ1 checkpoint
    # which replaces the projection layers with AnemllConv2d
    model.eval()

    # Initialize converter with AQ1 checkpoint
    print("\nInitializing QwenConverter with AQ1 checkpoint...")
    converter = QwenConverter(
        model=model,
        context_length=args.context,
        batch_size=64,  # For prefill
        lut_bits=None,  # Don't apply standard LUT quantization
        aq1_checkpoint=args.checkpoint,  # Use AQ1 checkpoint
    )

    # Calculate chunks if layer limit is set
    total_layers = config.num_hidden_layers
    if args.layers and args.layers < total_layers:
        # Use chunking to convert specific layer range
        num_chunks = total_layers // args.layers
        chunk_idx = args.chunk
        total_chunks = num_chunks
        start_layer = chunk_idx * args.layers
        end_layer = start_layer + args.layers - 1
        print(f"\nConverting layers {start_layer}-{end_layer} (chunk {chunk_idx} of {num_chunks})...")
    else:
        chunk_idx = 0
        total_chunks = 1
        print(f"\nConverting all {total_layers} layers...")

    # Convert
    print("\nRunning CoreML conversion...")
    if args.prefill:
        mlmodel = converter.convert_part_2_prefill(model, chunk_idx=chunk_idx, total_chunks=total_chunks)
    else:
        mlmodel = converter.convert_part_2(model, chunk_idx=chunk_idx, total_chunks=total_chunks)

    print("\nConversion complete!")

    # Analyze the model
    print("\n" + "=" * 60)
    print("MODEL ANALYSIS")
    print("=" * 60)

    # Count operations
    op_counts = count_ops_by_type(mlmodel)
    print("\nOperation counts:")
    for op_type, count in sorted(op_counts.items()):
        if count > 0:
            print(f"  {op_type}: {count}")

    # Check for AQ1 operations
    lut_ops = op_counts.get('constexpr_lut_to_dense', 0)
    matmul_ops = op_counts.get('matmul', 0)
    conv_ops = op_counts.get('conv', 0)

    print(f"\nAQ1 Operations:")
    print(f"  constexpr_lut_to_dense: {lut_ops}")
    print(f"  matmul (A @ B scales): {matmul_ops}")
    print(f"  conv: {conv_ops}")

    # Expected ops calculation
    layers_converted = args.layers or total_layers
    # Per layer: 7 projections (gate, up, down, q, k, v, o)
    expected_lut_ops = layers_converted * 7
    # matmul ops: 7 projections * layers + some for attention
    print(f"\n  Expected LUT ops for {layers_converted} layers: ~{expected_lut_ops}")
    print(f"  Expected matmul ops (scales): ~{expected_lut_ops}")

    if lut_ops > 0:
        print(f"\n  ✓ constexpr_lut_to_dense ops found - LUT compression active!")
    else:
        print(f"\n  ✗ No constexpr_lut_to_dense ops - check ANEMLL_DYNAMIC_SCALES=1")

    # Print operations if verbose
    if args.verbose:
        print_ops(mlmodel, show_details=True)

    # Get model size
    model_size_bytes = get_model_size(mlmodel)
    model_size_mb = model_size_bytes / (1024 * 1024)
    print(f"\nModel size: {model_size_mb:.2f} MB")

    # Estimate compression ratio
    # Each layer has ~11M params (Qwen 0.6B has 28 layers, ~600M params)
    # FP16 baseline: 600M * 2 bytes = 1.2GB
    # With 2-bit LUT: 600M * 0.25 bytes + scales overhead
    params_per_layer = (
        config.hidden_size * config.intermediate_size * 3  # MLP
        + config.hidden_size * config.hidden_size * 4     # Attention
    )
    fp16_size = params_per_layer * layers_converted * 2 / (1024 * 1024)
    compression_ratio = fp16_size / model_size_mb if model_size_mb > 0 else 0
    print(f"FP16 baseline estimate: {fp16_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")

    # Save model
    print(f"\nSaving model to: {args.output}")
    mlmodel.save(args.output)
    print("Done!")

    # Test inference
    print("\n" + "=" * 60)
    print("INFERENCE TEST")
    print("=" * 60)

    try:
        # Create test inputs matching the model's expected format
        hidden_size = config.hidden_size
        context_length = args.context

        if args.prefill:
            batch_size = 64
            test_inputs = {
                'hidden_states': np.random.randn(1, batch_size, hidden_size).astype(np.float16),
                'position_ids': np.arange(batch_size).astype(np.int32),
                'causal_mask': np.zeros((1, 1, batch_size, context_length), dtype=np.float16),
                'current_pos': np.array([0], dtype=np.int32),
            }
        else:
            test_inputs = {
                'hidden_states': np.random.randn(1, 1, hidden_size).astype(np.float16),
                'position_ids': np.array([0], dtype=np.int32),
                'causal_mask': np.zeros((1, 1, 1, context_length), dtype=np.float16),
                'current_pos': np.array([0], dtype=np.int32),
            }

        print("Running inference with test inputs...")

        # Create state for stateful model (KV cache)
        state = mlmodel.make_state()
        print(f"  Created model state for KV cache")

        output = mlmodel.predict(test_inputs, state)

        print(f"  Input hidden_states: {test_inputs['hidden_states'].shape}")
        output_key = list(output.keys())[0]
        print(f"  Output {output_key}: {output[output_key].shape}")
        print("  ✓ Inference successful!")

    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

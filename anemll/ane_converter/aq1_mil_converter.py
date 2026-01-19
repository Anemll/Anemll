"""
AQ1 MIL-based Converter for CoreML

This module provides MIL-based conversion for ANEMLL-QUANT-1 models.
Unlike PyTorch tracing (which bakes constants), this builds the MIL
program directly to preserve dynamic A*B computation.

Key features:
- Uses constexpr_lut_to_dense for W_base (packed indices + LUT)
- Uses matmul for dynamic scale computation (A @ B at runtime)
- Uses pass_pipeline=ct.PassPipeline.EMPTY to prevent constant folding

Usage:
    from anemll.ane_converter.aq1_mil_converter import convert_aq1_ffn

    mlmodel = convert_aq1_ffn(
        checkpoint_path="path/to/checkpoint",
        context_length=256,
        batch_size=64,
        output_path="output.mlpackage"
    )
"""

import os
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

from ..models.aq1_mil_layer import (
    build_aq1_conv_layer,
    load_aq1_checkpoint,
    get_layer_aq1_data,
    compute_aq1_storage_size,
    _load_state_dict,
)


def build_aq1_mlp_block(
    x,  # MIL tensor input [batch, hidden_size, 1, 1]
    layer_idx: int,
    aq1_data: Dict[str, Dict[str, Any]],
    hidden_size: int,
    intermediate_size: int,
    name_prefix: str = "model.layers",
):
    """
    Build a complete MLP block with AQ1 quantization.

    Implements: output = down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        x: Input MIL tensor [batch, hidden_size, 1, 1]
        layer_idx: Layer index for naming
        aq1_data: Dict from load_aq1_checkpoint()
        hidden_size: Model hidden dimension
        intermediate_size: MLP intermediate dimension
        name_prefix: Prefix for layer names

    Returns:
        MIL tensor output [batch, hidden_size, 1, 1]
    """
    base_name = f"{name_prefix}.{layer_idx}.mlp"

    # Get AQ1 data for each projection
    gate_data = get_layer_aq1_data(aq1_data, f"{base_name}.gate_proj")
    up_data = get_layer_aq1_data(aq1_data, f"{base_name}.up_proj")
    down_data = get_layer_aq1_data(aq1_data, f"{base_name}.down_proj")

    if gate_data is None or up_data is None or down_data is None:
        raise ValueError(f"Missing AQ1 data for MLP layer {layer_idx}")

    # Gate projection
    gate = build_aq1_conv_layer(
        x=x,
        indices_packed=gate_data['indices_packed'],
        lut=gate_data['lut'],
        scale_A=gate_data['scale_A'],
        scale_B=gate_data['scale_B'],
        out_features=gate_data['out_features'],
        in_features=gate_data['in_features'],
        name=f"layer{layer_idx}_gate_proj"
    )

    # Up projection
    up = build_aq1_conv_layer(
        x=x,
        indices_packed=up_data['indices_packed'],
        lut=up_data['lut'],
        scale_A=up_data['scale_A'],
        scale_B=up_data['scale_B'],
        out_features=up_data['out_features'],
        in_features=up_data['in_features'],
        name=f"layer{layer_idx}_up_proj"
    )

    # SiLU activation on gate
    gate_silu = mb.silu(x=gate, name=f"layer{layer_idx}_gate_silu")

    # Multiply gate * up
    hidden = mb.mul(x=gate_silu, y=up, name=f"layer{layer_idx}_hidden")

    # Down projection
    output = build_aq1_conv_layer(
        x=hidden,
        indices_packed=down_data['indices_packed'],
        lut=down_data['lut'],
        scale_A=down_data['scale_A'],
        scale_B=down_data['scale_B'],
        out_features=down_data['out_features'],
        in_features=down_data['in_features'],
        name=f"layer{layer_idx}_down_proj"
    )

    return output


def build_aq1_attention_projections(
    hidden_states,  # [batch, hidden_size, 1, 1]
    layer_idx: int,
    aq1_data: Dict[str, Dict[str, Any]],
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    name_prefix: str = "model.layers",
):
    """
    Build attention Q/K/V/O projections with AQ1 quantization.

    Note: This only handles the projections, not the attention computation itself.
    The attention mechanism (rotary embedding, softmax, etc.) must be handled separately.

    Args:
        hidden_states: Input [batch, hidden_size, 1, 1]
        layer_idx: Layer index
        aq1_data: Dict from load_aq1_checkpoint()
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads (for GQA)
        head_dim: Dimension per head
        name_prefix: Prefix for layer names

    Returns:
        Tuple of (q, k, v, o_proj_data) where q/k/v are MIL tensors and o_proj_data
        is the AQ1 data dict for the output projection (to be applied after attention)
    """
    base_name = f"{name_prefix}.{layer_idx}.self_attn"

    # Get AQ1 data for projections
    q_data = get_layer_aq1_data(aq1_data, f"{base_name}.q_proj")
    k_data = get_layer_aq1_data(aq1_data, f"{base_name}.k_proj")
    v_data = get_layer_aq1_data(aq1_data, f"{base_name}.v_proj")
    o_data = get_layer_aq1_data(aq1_data, f"{base_name}.o_proj")

    if q_data is None or k_data is None or v_data is None:
        raise ValueError(f"Missing AQ1 data for attention layer {layer_idx}")

    # Q projection
    q = build_aq1_conv_layer(
        x=hidden_states,
        indices_packed=q_data['indices_packed'],
        lut=q_data['lut'],
        scale_A=q_data['scale_A'],
        scale_B=q_data['scale_B'],
        out_features=q_data['out_features'],
        in_features=q_data['in_features'],
        name=f"layer{layer_idx}_q_proj"
    )

    # K projection
    k = build_aq1_conv_layer(
        x=hidden_states,
        indices_packed=k_data['indices_packed'],
        lut=k_data['lut'],
        scale_A=k_data['scale_A'],
        scale_B=k_data['scale_B'],
        out_features=k_data['out_features'],
        in_features=k_data['in_features'],
        name=f"layer{layer_idx}_k_proj"
    )

    # V projection
    v = build_aq1_conv_layer(
        x=hidden_states,
        indices_packed=v_data['indices_packed'],
        lut=v_data['lut'],
        scale_A=v_data['scale_A'],
        scale_B=v_data['scale_B'],
        out_features=v_data['out_features'],
        in_features=v_data['in_features'],
        name=f"layer{layer_idx}_v_proj"
    )

    return q, k, v, o_data


def build_rmsnorm(
    x,  # Input tensor
    weight: np.ndarray,
    eps: float = 1e-6,
    name: str = "rmsnorm",
):
    """
    Build RMSNorm using MIL operations.

    ANE-compatible RMSNorm:
    1. Subtract mean first
    2. Use layer_norm with weight

    Args:
        x: Input MIL tensor
        weight: RMSNorm weight [hidden_size]
        eps: Epsilon for numerical stability
        name: Operation name

    Returns:
        Normalized MIL tensor
    """
    # Subtract mean (ANE compatibility)
    mean = mb.reduce_mean(x=x, axes=[-1], keep_dims=True, name=f"{name}_mean")
    x_centered = mb.sub(x=x, y=mean, name=f"{name}_centered")

    # Compute variance
    x_sq = mb.mul(x=x_centered, y=x_centered, name=f"{name}_sq")
    var = mb.reduce_mean(x=x_sq, axes=[-1], keep_dims=True, name=f"{name}_var")

    # Add epsilon and compute rsqrt
    var_eps = mb.add(x=var, y=np.float16(eps), name=f"{name}_var_eps")
    rsqrt = mb.rsqrt(x=var_eps, name=f"{name}_rsqrt")

    # Normalize
    normalized = mb.mul(x=x_centered, y=rsqrt, name=f"{name}_normalized")

    # Apply weight
    weight_const = mb.const(val=weight.astype(np.float16), name=f"{name}_weight")
    output = mb.mul(x=normalized, y=weight_const, name=f"{name}_output")

    return output


def create_aq1_mlp_only_program(
    checkpoint_path: str,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    is_prefill: bool = False,
    batch_size: int = 64,
    nbits_mlp: int = 2,
    verbose: bool = True,
):
    """
    Create a MIL program for MLP-only processing with AQ1 quantization.

    This is a simplified test that only includes MLP blocks, not full attention.
    Use this to verify AQ1 conversion before tackling full transformer layers.

    Args:
        checkpoint_path: Path to ANEMLL checkpoint
        hidden_size: Model hidden dimension
        intermediate_size: MLP intermediate dimension
        num_layers: Number of transformer layers
        is_prefill: If True, use batch_size sequence length
        batch_size: Batch/sequence size for prefill
        nbits_mlp: Bits for MLP quantization
        verbose: Print progress

    Returns:
        MIL program ready for ct.convert()
    """
    # Load AQ1 checkpoint data (only load needed layers)
    aq1_data = load_aq1_checkpoint(
        checkpoint_path,
        nbits_mlp=nbits_mlp,
        nbits_attn=4,  # Not used for MLP-only
        max_layers=num_layers,
        verbose=verbose,
    )

    # Determine input shape
    seq_len = batch_size if is_prefill else 1

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, hidden_size, 1, seq_len), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18  # iOS18 required for constexpr_lut_to_dense with sub-byte types
    )
    def prog(hidden_states):
        x = hidden_states

        # Process each layer's MLP
        for layer_idx in range(num_layers):
            # Build MLP block
            mlp_out = build_aq1_mlp_block(
                x=x,
                layer_idx=layer_idx,
                aq1_data=aq1_data,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )

            # Residual connection
            x = mb.add(x=x, y=mlp_out, name=f"layer{layer_idx}_residual")

        return x

    return prog


def convert_aq1_mlp_test(
    checkpoint_path: str,
    model_config: Dict[str, Any],
    output_path: str,
    is_prefill: bool = False,
    batch_size: int = 64,
    nbits_mlp: int = 2,
    verbose: bool = True,
) -> ct.models.MLModel:
    """
    Convert MLP layers with AQ1 quantization to CoreML.

    This is a test function that only converts MLP blocks.
    Use to verify AQ1 MIL conversion before full transformer.

    Args:
        checkpoint_path: Path to ANEMLL checkpoint
        model_config: Dict with hidden_size, intermediate_size, num_hidden_layers
        output_path: Path to save .mlpackage
        is_prefill: Use prefill shape
        batch_size: Batch/sequence size
        nbits_mlp: Bits for MLP quantization
        verbose: Print progress

    Returns:
        Converted CoreML model
    """
    if verbose:
        print("=" * 60)
        print("AQ1 MIL Converter - MLP Test")
        print("=" * 60)

    # Create MIL program
    prog = create_aq1_mlp_only_program(
        checkpoint_path=checkpoint_path,
        hidden_size=model_config['hidden_size'],
        intermediate_size=model_config['intermediate_size'],
        num_layers=model_config['num_hidden_layers'],
        is_prefill=is_prefill,
        batch_size=batch_size,
        nbits_mlp=nbits_mlp,
        verbose=verbose,
    )

    if verbose:
        print("\nConverting to CoreML...")

    # Convert with EMPTY pipeline to prevent constant folding
    model = ct.convert(
        prog,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,  # iOS18 for sub-byte constexpr_lut_to_dense
        pass_pipeline=ct.PassPipeline.EMPTY,  # Critical: prevent folding
    )

    if verbose:
        print(f"Saving to: {output_path}")

    model.save(output_path)

    # Verify structure
    if verbose:
        spec = model.get_spec()
        if hasattr(spec, 'mlProgram'):
            prog_spec = spec.mlProgram
            constexpr_count = 0
            matmul_count = 0
            for func_name, func in prog_spec.functions.items():
                for block_name, block in func.block_specializations.items():
                    for op in block.operations:
                        if op.type == 'constexpr_lut_to_dense':
                            constexpr_count += 1
                        elif op.type == 'matmul':
                            matmul_count += 1
            print(f"\nModel structure verification:")
            print(f"  constexpr_lut_to_dense ops: {constexpr_count}")
            print(f"  matmul ops (includes A@B): {matmul_count}")

    return model


# =============================================================================
# Integration with QwenConverter
# =============================================================================

def get_aq1_conversion_info(
    checkpoint_path: str,
    nbits_mlp: int = 2,
    nbits_attn: int = 4,
) -> Dict[str, Any]:
    """
    Analyze checkpoint and return AQ1 conversion info.

    Useful for checking if checkpoint is suitable for AQ1 conversion.

    Args:
        checkpoint_path: Path to ANEMLL checkpoint
        nbits_mlp: Bits for MLP layers
        nbits_attn: Bits for attention layers

    Returns:
        Dict with conversion info:
        - num_mlp_layers: Number of MLP projection layers found
        - num_attn_layers: Number of attention projection layers found
        - total_compression: Overall compression ratio
        - layer_details: Per-layer compression info
    """
    aq1_data = load_aq1_checkpoint(
        checkpoint_path,
        nbits_mlp=nbits_mlp,
        nbits_attn=nbits_attn,
        verbose=False,
    )

    mlp_layers = 0
    attn_layers = 0
    total_aq1_bytes = 0
    total_baked_bytes = 0
    layer_details = []

    mlp_proj_names = ['gate_proj', 'up_proj', 'down_proj']
    attn_proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    for name, data in aq1_data.items():
        storage = compute_aq1_storage_size(
            data['out_features'],
            data['in_features'],
            data['scale_A'].shape[1],
            data['nbits'],
        )
        total_aq1_bytes += storage['aq1_total_bytes']
        total_baked_bytes += storage['baked_total_bytes']

        if any(proj in name for proj in mlp_proj_names):
            mlp_layers += 1
        elif any(proj in name for proj in attn_proj_names):
            attn_layers += 1

        layer_details.append({
            'name': name,
            'out_features': data['out_features'],
            'in_features': data['in_features'],
            'nbits': data['nbits'],
            'compression': storage['compression_ratio'],
        })

    return {
        'num_mlp_layers': mlp_layers,
        'num_attn_layers': attn_layers,
        'total_layers': mlp_layers + attn_layers,
        'total_aq1_bytes': total_aq1_bytes,
        'total_baked_bytes': total_baked_bytes,
        'total_compression': total_baked_bytes / total_aq1_bytes if total_aq1_bytes > 0 else 0,
        'layer_details': layer_details,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test AQ1 MIL Conversion")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ANEMLL checkpoint")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden dimension")
    parser.add_argument("--intermediate-size", type=int, default=3072, help="MLP intermediate dimension")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of layers to convert")
    parser.add_argument("--output", type=str, default="/tmp/aq1_mlp_test.mlpackage", help="Output path")
    parser.add_argument("--nbits", type=int, default=2, help="Bits for MLP quantization")
    args = parser.parse_args()

    config = {
        'hidden_size': args.hidden_size,
        'intermediate_size': args.intermediate_size,
        'num_hidden_layers': args.num_layers,
    }

    convert_aq1_mlp_test(
        checkpoint_path=args.checkpoint,
        model_config=config,
        output_path=args.output,
        nbits_mlp=args.nbits,
        verbose=True,
    )

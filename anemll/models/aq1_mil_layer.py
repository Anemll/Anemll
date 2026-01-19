"""
AQ1 (ANEMLL-QUANT-1) MIL Layer Builder

This module provides functions to build CoreML MIL layers with:
- constexpr_lut_to_dense: Decompresses packed LUT indices to float16 weights
- Dynamic A*B: Computes scales = A @ B at runtime
- Final computation: W_base * scales -> conv

Usage:
    from anemll.models.aq1_mil_layer import build_aq1_conv_layer

    # In your MIL program:
    output = build_aq1_conv_layer(
        mb=mb,
        x=input_tensor,
        indices_packed=packed_indices,
        lut=lut_values,
        scale_A=scale_A,
        scale_B=scale_B,
        out_features=3072,
        in_features=1024,
        name="gate_proj"
    )

Checkpoint Loading:
    from anemll.models.aq1_mil_layer import load_aq1_checkpoint

    # Load checkpoint with snapped weights + scales
    layer_data = load_aq1_checkpoint(checkpoint_path)

    # Extract data for a specific layer
    data = layer_data['model.layers.0.mlp.gate_proj']
    indices_packed, lut, scale_A, scale_B, nbits = (
        data['indices_packed'], data['lut'], data['scale_A'], data['scale_B'], data['nbits']
    )
"""

import os
import re
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any
import torch

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


def pack_indices_to_bits(indices: np.ndarray, nbits: int) -> np.ndarray:
    """
    Pack integer indices into bit-packed uint8 array for constexpr_lut_to_dense.

    Args:
        indices: Integer indices array (any shape), values 0 to 2^nbits-1
        nbits: Number of bits per index (2 for 4 values, 4 for 16 values)

    Returns:
        Packed uint8 array where each byte holds multiple indices

    For 2-bit: each byte holds 4 values (indices 0-3)
    For 4-bit: each byte holds 2 values (indices 0-15)
    """
    flat = indices.flatten()
    n_elements = len(flat)

    elements_per_byte = 8 // nbits
    n_bytes = (n_elements + elements_per_byte - 1) // elements_per_byte

    packed = np.zeros(n_bytes, dtype=np.uint8)

    for i, idx in enumerate(flat):
        byte_idx = i // elements_per_byte
        bit_offset = (i % elements_per_byte) * nbits
        packed[byte_idx] |= (int(idx) & ((1 << nbits) - 1)) << bit_offset

    return packed


def build_aq1_conv_layer(
    x,  # MIL tensor input
    indices_packed: np.ndarray,
    lut: np.ndarray,
    scale_A: np.ndarray,
    scale_B: np.ndarray,
    out_features: int,
    in_features: int,
    name: str,
    bias: Optional[np.ndarray] = None,
):
    """
    Build an AQ1 conv layer using MIL operations.

    This creates the following computation graph:
        W_base = constexpr_lut_to_dense(indices, LUT)  # Decompress from packed bits
        scales = matmul(A, B)                           # Dynamic scale computation
        scales = clip(scales, 1e-8)                     # Clamp to positive
        W_eff = W_base * scales                         # Apply scales
        output = conv(x, W_eff)                         # Final convolution

    Args:
        x: Input MIL tensor (shape: [batch, in_features, 1, 1])
        indices_packed: Packed uint8 indices from pack_indices_to_bits()
        lut: LUT values array (float16)
        scale_A: Scale factor A [out_features, rank] (float16)
        scale_B: Scale factor B [rank, in_features] (float16)
        out_features: Output dimension
        in_features: Input dimension
        name: Layer name prefix
        bias: Optional bias array

    Returns:
        MIL tensor output of the convolution
    """
    # iOS18 constexpr_lut_to_dense format:
    # - indices: 4D [out_features, in_features, 1, 1] with sub-byte dtype
    # - lut: 6D [1, 1, 1, 1, lut_size, 1]
    # - no shape parameter (inferred from indices)

    # Determine nbits from LUT size
    lut_size = len(lut)
    nbits = int(np.ceil(np.log2(lut_size))) if lut_size > 1 else 1

    # Get proper sub-byte dtype for indices
    from coremltools.converters.mil.mil.types import np_uint2_dtype, np_uint4_dtype
    if nbits <= 2:
        indices_dtype = np_uint2_dtype
    elif nbits <= 4:
        indices_dtype = np_uint4_dtype
    else:
        indices_dtype = np.uint8

    # Convert indices to 4D with sub-byte dtype
    # indices_packed may be packed bytes or unpacked indices
    if isinstance(indices_packed, np.ndarray) and indices_packed.dtype == np.uint8 and len(indices_packed.shape) == 1:
        # Packed bytes - unpack first
        indices_per_byte = 8 // nbits
        total_elements = out_features * in_features
        mask = (1 << nbits) - 1

        unpacked = []
        for byte in indices_packed:
            for i in range(indices_per_byte):
                if len(unpacked) < total_elements:
                    unpacked.append((byte >> (i * nbits)) & mask)

        indices_2d = np.array(unpacked, dtype=np.uint8).reshape(out_features, in_features)
    else:
        # Already unpacked or 2D
        indices_2d = np.array(indices_packed, dtype=np.uint8).reshape(out_features, in_features)

    # Convert to 4D and proper sub-byte dtype
    indices_4d = indices_2d.reshape(out_features, in_features, 1, 1).astype(indices_dtype)

    # LUT to 6D format [1, 1, 1, 1, lut_size, 1]
    lut_6d = lut.astype(np.float16).reshape(1, 1, 1, 1, lut_size, 1)

    # constexpr_lut_to_dense outputs 4D [out, in, 1, 1]
    base_weights = mb.constexpr_lut_to_dense(
        indices=indices_4d,
        lut=lut_6d,
        name=f"{name}_base_weights"
    )

    # Scale factors as constants
    A = mb.const(val=scale_A.astype(np.float16), name=f"{name}_scale_A")
    B = mb.const(val=scale_B.astype(np.float16), name=f"{name}_scale_B")

    # Compute scales = A @ B at runtime -> 2D [out, in]
    scales = mb.matmul(x=A, y=B, name=f"{name}_scales")

    # Clamp scales to positive values (required for correct inference)
    scales_clamped = mb.clip(
        x=scales,
        alpha=np.float16(1e-8),
        beta=np.float16(65504),  # max fp16
        name=f"{name}_scales_clamped"
    )

    # Reshape scales to 4D [out, in, 1, 1] for element-wise mul with base_weights
    scales_4d = mb.reshape(
        x=scales_clamped,
        shape=[out_features, in_features, 1, 1],
        name=f"{name}_scales_4d"
    )

    # W_eff = W_base * scales (4D * 4D element-wise)
    weights_4d = mb.mul(x=base_weights, y=scales_4d, name=f"{name}_weights_4d")

    # Perform convolution with 4D weights
    if bias is not None:
        output = mb.conv(
            x=x,
            weight=weights_4d,
            bias=mb.const(val=bias.astype(np.float16), name=f"{name}_bias"),
            pad_type="valid",
            name=f"{name}_output"
        )
    else:
        output = mb.conv(
            x=x,
            weight=weights_4d,
            pad_type="valid",
            name=f"{name}_output"
        )

    return output


def build_factored_aq1_conv_layer(
    x,  # MIL tensor input
    indices_packed: np.ndarray,
    lut: np.ndarray,
    scale_A: np.ndarray,
    scale_B: np.ndarray,
    out_features: int,
    in_features: int,
    name: str,
    bias: Optional[np.ndarray] = None,
):
    """
    Build an AQ1 conv layer using FACTORED MIL operations.

    This avoids materializing the full [out_features, in_features] scale matrix.

    Instead of:
        scales = A @ B                    # Materializes [out, in] matrix
        W_eff = W_base * scales
        output = conv(x, W_eff)

    This computes:
        y = sum_k(A[k] * conv(W_base, B[k] * x))

    Where:
        - W_base = constexpr_lut_to_dense(indices, LUT)
        - B[k] is the k-th row of B: [1, in_features, 1, 1]
        - A[k] is the k-th column of A: [1, out_features, 1, 1]

    Benefits:
        - No A @ B matmul (avoids [out, in] intermediate)
        - Uses only mul, conv, add operations (ANE-friendly)
        - Same numerical result as original approach

    Args:
        x: Input MIL tensor (shape: [batch, in_features, 1, seq_len])
        indices_packed: Packed uint8 indices from pack_indices_to_bits()
        lut: LUT values array (float16)
        scale_A: Scale factor A [out_features, rank] (float16)
        scale_B: Scale factor B [rank, in_features] (float16)
        out_features: Output dimension
        in_features: Input dimension
        name: Layer name prefix
        bias: Optional bias array

    Returns:
        MIL tensor output of the convolution
    """
    # Determine nbits from LUT size
    lut_size = len(lut)
    nbits = int(np.ceil(np.log2(lut_size))) if lut_size > 1 else 1

    # Get proper sub-byte dtype for indices
    from coremltools.converters.mil.mil.types import np_uint2_dtype, np_uint4_dtype
    if nbits <= 2:
        indices_dtype = np_uint2_dtype
    elif nbits <= 4:
        indices_dtype = np_uint4_dtype
    else:
        indices_dtype = np.uint8

    # Convert indices to 4D with sub-byte dtype
    if isinstance(indices_packed, np.ndarray) and indices_packed.dtype == np.uint8 and len(indices_packed.shape) == 1:
        # Packed bytes - unpack first
        indices_per_byte = 8 // nbits
        total_elements = out_features * in_features
        mask = (1 << nbits) - 1

        unpacked = []
        for byte in indices_packed:
            for i in range(indices_per_byte):
                if len(unpacked) < total_elements:
                    unpacked.append((byte >> (i * nbits)) & mask)

        indices_2d = np.array(unpacked, dtype=np.uint8).reshape(out_features, in_features)
    else:
        indices_2d = np.array(indices_packed, dtype=np.uint8).reshape(out_features, in_features)

    # Convert to 4D and proper sub-byte dtype
    indices_4d = indices_2d.reshape(out_features, in_features, 1, 1).astype(indices_dtype)

    # LUT to 6D format [1, 1, 1, 1, lut_size, 1]
    lut_6d = lut.astype(np.float16).reshape(1, 1, 1, 1, lut_size, 1)

    # constexpr_lut_to_dense outputs 4D [out, in, 1, 1] - this is our Q (base weights)
    Q_weights = mb.constexpr_lut_to_dense(
        indices=indices_4d,
        lut=lut_6d,
        name=f"{name}_Q"
    )

    # Get rank from scale shapes
    # scale_A: [out_features, rank], scale_B: [rank, in_features]
    rank = scale_A.shape[1]

    # Transpose scales for iteration:
    # A_T: [rank, out_features] - each row k is A[:, k]
    # B stays as [rank, in_features] - each row k is B[k, :]
    scale_A_T = scale_A.T.astype(np.float16)  # [rank, out_features]
    scale_B_np = scale_B.astype(np.float16)   # [rank, in_features]

    y_accum = None

    for k in range(rank):
        # B[k]: shape [1, in_features, 1, 1] for broadcasting with x
        B_k = scale_B_np[k:k+1, :, np.newaxis, np.newaxis]  # [1, in_features, 1, 1]
        B_const = mb.const(val=B_k, name=f"{name}_B{k}")

        # A[k]: shape [1, out_features, 1, 1] for broadcasting with conv output
        A_k = scale_A_T[k:k+1, :, np.newaxis, np.newaxis]  # [1, out_features, 1, 1]
        A_const = mb.const(val=A_k, name=f"{name}_A{k}")

        # Step 1: x_scaled = x * B[k]
        # x is [batch, in_features, 1, seq], B_k is [1, in_features, 1, 1]
        x_scaled = mb.mul(x=x, y=B_const, name=f"{name}_xB{k}")

        # Step 2: y_k = conv(Q, x_scaled)
        # Q is [out, in, 1, 1], x_scaled is [batch, in, 1, seq]
        # Output is [batch, out, 1, seq]
        y_k = mb.conv(x=x_scaled, weight=Q_weights, pad_type="valid", name=f"{name}_conv{k}")

        # Step 3: y_k = y_k * A[k]
        # y_k is [batch, out, 1, seq], A_k is [1, out, 1, 1]
        y_k_scaled = mb.mul(x=y_k, y=A_const, name=f"{name}_yA{k}")

        # Step 4: Accumulate
        if y_accum is None:
            y_accum = y_k_scaled
        else:
            y_accum = mb.add(x=y_accum, y=y_k_scaled, name=f"{name}_add{k}")

    # Add bias if present
    if bias is not None:
        bias_4d = bias.astype(np.float16).reshape(1, out_features, 1, 1)
        bias_const = mb.const(val=bias_4d, name=f"{name}_bias")
        y_accum = mb.add(x=y_accum, y=bias_const, name=f"{name}_biased")

    return y_accum


def prepare_aq1_layer_data(
    weight: np.ndarray,
    scale_A: np.ndarray,
    scale_B: np.ndarray,
    lut: np.ndarray,
    nbits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare weight data for AQ1 layer from raw weight and scales.

    This function:
    1. Quantizes weights to LUT indices
    2. Packs indices into bit-field format
    3. Returns data ready for build_aq1_conv_layer()

    Args:
        weight: Float weight tensor [out_features, in_features]
        scale_A: Scale factor A [out_features, rank]
        scale_B: Scale factor B [rank, in_features]
        lut: LUT values (e.g., [-1, -0.33, 0.33, 1] for 2-bit)
        nbits: Number of bits for quantization (2 or 4)

    Returns:
        Tuple of (indices_packed, lut, scale_A, scale_B)
    """
    # Quantize weights to nearest LUT values
    weight_flat = weight.flatten()
    lut_np = np.array(lut)

    # Find nearest LUT index for each weight
    indices = np.abs(weight_flat[:, None] - lut_np[None, :]).argmin(axis=1)
    indices = indices.reshape(weight.shape).astype(np.uint8)

    # Pack indices
    indices_packed = pack_indices_to_bits(indices, nbits)

    return (
        indices_packed,
        lut_np.astype(np.float16),
        scale_A.astype(np.float16),
        scale_B.astype(np.float16),
    )


def compute_aq1_storage_size(
    out_features: int,
    in_features: int,
    rank: int,
    nbits: int,
) -> dict:
    """
    Compute storage sizes for AQ1 vs baked weights.

    Args:
        out_features: Output dimension
        in_features: Input dimension
        rank: Low-rank scale rank
        nbits: Number of bits for LUT indices

    Returns:
        Dict with storage sizes in bytes
    """
    n_weights = out_features * in_features

    # AQ1 storage
    lut_size = (2 ** nbits) * 2  # float16
    indices_size = (n_weights * nbits + 7) // 8  # packed bits
    scale_a_size = out_features * rank * 2  # float16
    scale_b_size = rank * in_features * 2  # float16
    aq1_total = lut_size + indices_size + scale_a_size + scale_b_size

    # Baked storage
    baked_total = n_weights * 2  # float16

    return {
        'lut_bytes': lut_size,
        'indices_bytes': indices_size,
        'scale_a_bytes': scale_a_size,
        'scale_b_bytes': scale_b_size,
        'aq1_total_bytes': aq1_total,
        'baked_total_bytes': baked_total,
        'compression_ratio': baked_total / aq1_total,
    }


# =============================================================================
# Checkpoint Loading and LUT Extraction
# =============================================================================

def extract_lut_and_indices(
    snapped_weight: np.ndarray,
    nbits: int = 2,
    lut: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract LUT and indices from snapped weights.

    The snapped weight contains quantized values from a LUT (e.g., [-1, -0.33, 0.33, 1]).
    This function:
    1. Identifies or uses the provided LUT values
    2. Maps each weight value to its LUT index
    3. Packs indices into bit-field format for constexpr_lut_to_dense

    Args:
        snapped_weight: Weight tensor with quantized values [out_features, in_features]
        nbits: Number of bits for quantization (2 or 4)
        lut: Optional pre-defined LUT values. If None, extracted from unique weight values.

    Returns:
        Tuple of (indices_packed, lut) where:
        - indices_packed: Packed uint8 array for constexpr_lut_to_dense
        - lut: The LUT values as float16 array
    """
    # Handle shape: weight may be [out, in] or [out, in, 1, 1]
    if snapped_weight.ndim == 4:
        snapped_weight = snapped_weight.squeeze(-1).squeeze(-1)

    # Get LUT values - either provided or extracted from unique values
    if lut is None:
        unique_vals = np.unique(snapped_weight)
        lut_size = 1 << nbits  # 2^nbits

        if len(unique_vals) <= lut_size:
            # Use actual unique values as LUT
            lut = np.sort(unique_vals).astype(np.float16)
            # Pad if fewer unique values than LUT size
            if len(lut) < lut_size:
                # Pad with evenly spaced values in the range
                min_val, max_val = lut.min(), lut.max()
                lut = np.linspace(min_val, max_val, lut_size, dtype=np.float16)
        else:
            # Too many unique values, use uniform LUT in [-1, 1]
            lut = np.linspace(-1.0, 1.0, lut_size, dtype=np.float16)
    else:
        lut = lut.astype(np.float16)

    # Map weights to LUT indices (nearest neighbor)
    weight_flat = snapped_weight.flatten()
    # Compute distance to each LUT value and find argmin
    indices = np.abs(weight_flat[:, None] - lut[None, :]).argmin(axis=1)
    indices = indices.reshape(snapped_weight.shape).astype(np.uint8)

    # Pack indices into bit-field format
    indices_packed = pack_indices_to_bits(indices, nbits)

    return indices_packed, lut


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load state dict from checkpoint file or directory."""
    if os.path.isdir(checkpoint_path):
        import safetensors.torch
        state_dict = {}
        for file in os.listdir(checkpoint_path):
            if file.endswith('.safetensors'):
                state_dict.update(
                    safetensors.torch.load_file(os.path.join(checkpoint_path, file))
                )
            elif file.endswith('.pt') or file.endswith('.bin'):
                state_dict.update(
                    torch.load(os.path.join(checkpoint_path, file), map_location='cpu', weights_only=False)
                )
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return state_dict


def load_aq1_checkpoint(
    checkpoint_path: str,
    nbits_mlp: int = 2,
    nbits_attn: int = 4,
    max_layers: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Load ANEMLL checkpoint and extract AQ1 data for quantized layers.

    Processes checkpoint with:
    - {layer}.weight: Snapped LUT values [out, in]
    - {layer}.scale_A: Low-rank factor A [out, rank]
    - {layer}.scale_B: Low-rank factor B [rank, in]
    - {layer}.lut: (optional) LUT values

    Returns data ready for build_aq1_conv_layer():
    - indices_packed: Packed bit-field indices
    - lut: LUT values
    - scale_A, scale_B: Scale factors
    - out_features, in_features, nbits

    Args:
        checkpoint_path: Path to checkpoint file or directory
        nbits_mlp: Bits for MLP layers (gate_proj, up_proj, down_proj)
        nbits_attn: Bits for attention layers (q_proj, k_proj, v_proj, o_proj)
        max_layers: Maximum number of transformer layers to load (None = all)
        verbose: Print loading progress

    Returns:
        Dict mapping layer names to their AQ1 data
    """
    state_dict = _load_state_dict(checkpoint_path)

    if verbose:
        print(f"Loading AQ1 checkpoint: {len(state_dict)} keys")

    # Identify quantized layers by looking for scale_A
    quantized_layers = {}
    for key in state_dict.keys():
        if key.endswith('.scale_A'):
            base_name = key[:-8]  # Remove '.scale_A'

            # Filter by max_layers if specified
            if max_layers is not None:
                # Extract layer index from name like "model.layers.5.mlp.gate_proj"
                match = re.search(r'layers\.(\d+)\.', base_name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx >= max_layers:
                        continue  # Skip layers beyond max_layers

            quantized_layers[base_name] = True

    if verbose:
        layer_msg = f" (limited to {max_layers} layers)" if max_layers else ""
        print(f"Found {len(quantized_layers)} quantized layers{layer_msg}")

    # Define which layers use which bit-width
    mlp_proj_names = ['gate_proj', 'up_proj', 'down_proj']
    attn_proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    result = {}
    for base_name in quantized_layers:
        weight_key = f'{base_name}.weight'
        scale_a_key = f'{base_name}.scale_A'
        scale_b_key = f'{base_name}.scale_B'
        lut_key = f'{base_name}.lut'

        if weight_key not in state_dict or scale_a_key not in state_dict:
            if verbose:
                print(f"  Skipping {base_name}: missing weight or scale_A")
            continue

        # Load tensors (convert BF16 to FP32 first since BF16 can't convert directly to numpy)
        weight_tensor = state_dict[weight_key]
        snapped_weight = weight_tensor.float().numpy() if weight_tensor.dtype == torch.bfloat16 else weight_tensor.numpy()

        scale_A_tensor = state_dict[scale_a_key]
        scale_A = scale_A_tensor.float().numpy() if scale_A_tensor.dtype == torch.bfloat16 else scale_A_tensor.numpy()

        if scale_b_key in state_dict:
            scale_B_tensor = state_dict[scale_b_key]
            scale_B = scale_B_tensor.float().numpy() if scale_B_tensor.dtype == torch.bfloat16 else scale_B_tensor.numpy()
        else:
            scale_B = None

        if scale_B is None:
            if verbose:
                print(f"  Skipping {base_name}: missing scale_B")
            continue

        # Get LUT if available (handle BF16)
        if lut_key in state_dict:
            lut_tensor = state_dict[lut_key]
            lut = lut_tensor.float().numpy() if lut_tensor.dtype == torch.bfloat16 else lut_tensor.numpy()
        else:
            lut = None

        # Determine bit-width: auto-detect from LUT size if available, else use config
        if lut is not None:
            # Auto-detect nbits from LUT size
            lut_size = len(lut)
            nbits = int(np.ceil(np.log2(lut_size))) if lut_size > 1 else 1
        elif any(proj in base_name for proj in mlp_proj_names):
            nbits = nbits_mlp
        elif any(proj in base_name for proj in attn_proj_names):
            nbits = nbits_attn
        else:
            # Auto-detect from unique weight values
            unique_vals = np.unique(snapped_weight)
            if len(unique_vals) <= 4:
                nbits = 2
            elif len(unique_vals) <= 16:
                nbits = 4
            else:
                nbits = nbits_mlp  # Default fallback

        # Extract indices and LUT
        indices_packed, lut = extract_lut_and_indices(snapped_weight, nbits, lut)

        # Handle weight shape
        if snapped_weight.ndim == 4:
            snapped_weight = snapped_weight.squeeze(-1).squeeze(-1)
        out_features, in_features = snapped_weight.shape

        result[base_name] = {
            'indices_packed': indices_packed,
            'lut': lut.astype(np.float16),
            'scale_A': scale_A.astype(np.float16),
            'scale_B': scale_B.astype(np.float16),
            'out_features': out_features,
            'in_features': in_features,
            'nbits': nbits,
        }

        if verbose:
            storage = compute_aq1_storage_size(out_features, in_features, scale_A.shape[1], nbits)
            print(f"  {base_name}: {out_features}x{in_features}, {nbits}-bit, "
                  f"rank={scale_A.shape[1]}, compression={storage['compression_ratio']:.1f}x")

    return result


def get_layer_aq1_data(
    aq1_data: Dict[str, Dict[str, Any]],
    layer_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Get AQ1 data for a specific layer, trying different key formats.

    Args:
        aq1_data: Dict from load_aq1_checkpoint()
        layer_name: Layer name to look up (e.g., 'model.layers.0.mlp.gate_proj')

    Returns:
        Dict with layer AQ1 data, or None if not found
    """
    # Try different key formats
    key_variants = [
        layer_name,
        layer_name.replace('model.model.', 'model.'),
        layer_name.replace('model.', ''),
        f'model.{layer_name}',
    ]

    for key in key_variants:
        if key in aq1_data:
            return aq1_data[key]

    return None


# =============================================================================
# MIL Building Utilities
# =============================================================================

def build_aq1_linear_layer(
    x,  # MIL tensor input [batch, seq, hidden] or [batch, hidden]
    indices_packed: np.ndarray,
    lut: np.ndarray,
    scale_A: np.ndarray,
    scale_B: np.ndarray,
    out_features: int,
    in_features: int,
    name: str,
    bias: Optional[np.ndarray] = None,
):
    """
    Build an AQ1 linear layer using MIL operations (matmul-based, not conv).

    This is an alternative to build_aq1_conv_layer for non-conv architectures.
    Uses matmul instead of conv2d.

    Args:
        x: Input MIL tensor
        indices_packed: Packed uint8 indices from pack_indices_to_bits()
        lut: LUT values array (float16)
        scale_A: Scale factor A [out_features, rank] (float16)
        scale_B: Scale factor B [rank, in_features] (float16)
        out_features: Output dimension
        in_features: Input dimension
        name: Layer name prefix
        bias: Optional bias array

    Returns:
        MIL tensor output
    """
    # Reconstruct base weights from LUT
    base_weights = mb.constexpr_lut_to_dense(
        indices=indices_packed,
        lut=lut.astype(np.float16),
        shape=np.array([out_features, in_features], dtype=np.uint32),
        name=f"{name}_base_weights"
    )

    # Scale factors
    A = mb.const(val=scale_A.astype(np.float16), name=f"{name}_scale_A")
    B = mb.const(val=scale_B.astype(np.float16), name=f"{name}_scale_B")

    # Compute scales = A @ B at runtime
    scales = mb.matmul(x=A, y=B, name=f"{name}_scales")

    # Clamp scales to positive values
    scales_clamped = mb.clip(
        x=scales,
        alpha=np.float16(1e-8),
        beta=np.float16(65504),
        name=f"{name}_scales_clamped"
    )

    # Multiply base_weights * scales (element-wise)
    weights_scaled = mb.mul(x=base_weights, y=scales_clamped, name=f"{name}_weights_scaled")

    # Transpose for matmul: [out, in] -> [in, out]
    weights_transposed = mb.transpose(
        x=weights_scaled,
        perm=[1, 0],
        name=f"{name}_weights_transposed"
    )

    # Perform matmul: x @ W^T
    output = mb.matmul(x=x, y=weights_transposed, name=f"{name}_output")

    # Add bias if present
    if bias is not None:
        bias_const = mb.const(val=bias.astype(np.float16), name=f"{name}_bias")
        output = mb.add(x=output, y=bias_const, name=f"{name}_biased")

    return output


# Example usage and test
if __name__ == "__main__":
    print("Testing AQ1 MIL Layer Builder")
    print("=" * 60)

    # Test dimensions
    out_features = 3072
    in_features = 1024
    rank = 32
    nbits = 2

    # Compute storage
    storage = compute_aq1_storage_size(out_features, in_features, rank, nbits)
    print(f"\nStorage comparison for {out_features}x{in_features} layer:")
    print(f"  LUT ({nbits}-bit):     {storage['lut_bytes']:,} bytes")
    print(f"  Indices (packed):      {storage['indices_bytes']:,} bytes")
    print(f"  Scale A (rank {rank}): {storage['scale_a_bytes']:,} bytes")
    print(f"  Scale B (rank {rank}): {storage['scale_b_bytes']:,} bytes")
    print(f"  AQ1 Total:             {storage['aq1_total_bytes']:,} bytes")
    print(f"  Baked (float16):       {storage['baked_total_bytes']:,} bytes")
    print(f"  Compression ratio:     {storage['compression_ratio']:.2f}x")

    # Test layer creation
    print("\nCreating test AQ1 layer...")

    # Create test data
    np.random.seed(42)
    lut = np.array([-1.0, -0.33, 0.33, 1.0], dtype=np.float16)
    indices_raw = np.random.randint(0, 4, size=(out_features, in_features)).astype(np.uint8)
    indices_packed = pack_indices_to_bits(indices_raw, nbits)
    scale_A = np.random.randn(out_features, rank).astype(np.float16) * 0.1
    scale_B = np.random.randn(rank, in_features).astype(np.float16) * 0.1

    print(f"  Indices raw: {indices_raw.shape}")
    print(f"  Indices packed: {indices_packed.shape} ({len(indices_packed)} bytes)")
    print(f"  Scale A: {scale_A.shape}")
    print(f"  Scale B: {scale_B.shape}")

    # Create MIL program
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, in_features, 1, 1), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18  # iOS18 required for constexpr_lut_to_dense with sub-byte types
    )
    def prog(x):
        return build_aq1_conv_layer(
            x=x,
            indices_packed=indices_packed,
            lut=lut,
            scale_A=scale_A,
            scale_B=scale_B,
            out_features=out_features,
            in_features=in_features,
            name="test_gate_proj"
        )

    # Convert
    model = ct.convert(
        prog,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,  # iOS18 for sub-byte constexpr_lut_to_dense
        pass_pipeline=ct.PassPipeline.EMPTY,
    )

    # Save
    output_path = "/tmp/aq1_layer_test.mlpackage"
    model.save(output_path)
    print(f"\nSaved test model to: {output_path}")

    # Show structure
    spec = model.get_spec()
    if hasattr(spec, 'mlProgram'):
        prog_spec = spec.mlProgram
        print("\n--- AQ1 Layer Operations ---")
        for func_name, func in prog_spec.functions.items():
            for block_name, block in func.block_specializations.items():
                for op in block.operations:
                    outputs = [out.name for out in op.outputs]
                    print(f"  {op.type:30} -> {outputs}")

    # Test inference
    print("\nTesting inference...")
    test_input = np.random.randn(1, in_features, 1, 1).astype(np.float16)
    output = model.predict({"x": test_input})

    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output['test_gate_proj_output'].shape}")
    print("\nAQ1 MIL Layer Builder test complete!")

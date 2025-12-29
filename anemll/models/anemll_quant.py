"""ANEMLL Quantization Module (ANEMLL-QUANT-1)

This module provides reusable quantization components for ANEMLL models.
It implements low-rank scale quantization where:
- Weights store LUT indices (packed for CoreML, unpacked for PyTorch)
- LUT contains the quantization levels (e.g., [-1, -0.33, 0.33, 1.0] for 2-bit)
- Scales are factorized as scale_A @ scale_B (low-rank)
- Effective weight = LUT[indices] * (scale_A @ scale_B)

For CoreML conversion:
- Uses constexpr_lut_to_dense for W_base (packed indices + LUT)
- Uses matmul for dynamic A @ B computation
- Preserves quantization structure in deployment

The conversion pipeline handles LUT+index packing for efficient storage.

Custom Op Approach:
- Defines torch.ops.anemll_quant.quant_conv custom op
- Op survives PyTorch tracing (torch.jit.trace)
- Registered coremltools converter emits constexpr_lut_to_dense + matmul
- Allows standard trace-based conversion workflow
"""

from __future__ import annotations

import os
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Default scale ranks for different layer types
DEFAULT_MLP_SCALE_RANK = 32
DEFAULT_ATTN_SCALE_RANK = 8


# =============================================================================
# Custom Op Registration (torch.library)
# =============================================================================

_CUSTOM_OP_REGISTERED = False
_ANEMLL_LIB = None  # Keep reference to prevent garbage collection

def _register_custom_op():
    """Register the anemll_quant::quant_conv custom op."""
    global _CUSTOM_OP_REGISTERED, _ANEMLL_LIB

    if _CUSTOM_OP_REGISTERED:
        return True

    try:
        from torch.library import Library, impl

        # Create library and keep reference
        _ANEMLL_LIB = Library("anemll_quant", "DEF")
        _ANEMLL_LIB.define(
            "quant_conv(Tensor x, Tensor indices, Tensor lut, Tensor scale_A, Tensor scale_B) -> Tensor"
        )

        # Implement for CPU
        @impl(_ANEMLL_LIB, "quant_conv", "CPU")
        def _quant_conv_cpu(x, indices, lut, scale_A, scale_B):
            """PyTorch implementation of quant_conv."""
            base = lut[indices.long()]
            scales = (scale_A @ scale_B).clamp(min=1e-8)
            out_features, in_features = indices.shape
            weight = (base * scales).view(out_features, in_features, 1, 1)
            return F.conv2d(x, weight.to(x.dtype))

        _CUSTOM_OP_REGISTERED = True
        return True

    except Exception as e:
        # torch.library may not be available in older PyTorch versions
        import traceback
        print(f"Warning: Could not register custom op: {e}")
        traceback.print_exc()
        return False


def _register_coreml_converter():
    """Register coremltools converter for anemll_quant::quant_conv."""
    try:
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs
        from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op

        @register_torch_op(torch_alias=["anemll_quant::quant_conv"])
        def anemll_quant_quant_conv(context, node):
            """Convert anemll_quant::quant_conv to MIL ops.

            Two modes based on ANEMLL_DYNAMIC_SCALES environment variable:

            DYNAMIC MODE (ANEMLL_DYNAMIC_SCALES=1):
            - Uses constexpr_lut_to_dense for W_base (compressed LUT storage, 4D)
            - Uses matmul for A @ B (computed at runtime)
            - Multiplies W_base * scales dynamically (all 4D tensors)
            - Runs on ANE with dynamic weight computation

            PRE-BAKED MODE (default):
            - Computes effective_weights = LUT[indices] * (A @ B) at CONVERSION time
            - Stores as static fp16 constant
            - Conv runs on ANE with static weights
            """
            inputs = _get_inputs(context, node, expected=5)
            x = inputs[0]
            indices = inputs[1]
            lut = inputs[2]
            scale_A = inputs[3]
            scale_B = inputs[4]

            # Get numpy arrays - these come from traced constants
            indices_val = indices.val
            lut_val = lut.val
            scale_A_val = scale_A.val
            scale_B_val = scale_B.val

            out_features, in_features = indices_val.shape

            # Check mode
            use_dynamic = os.environ.get('ANEMLL_DYNAMIC_SCALES', '0') == '1'

            # Option to skip LUT for attention layers (reduces constexpr_lut_to_dense count)
            # When ANEMLL_SKIP_ATTN_LUT=1, attention projections use pre-baked weights
            # This allows more layers per chunk (3 ops/layer instead of 7)
            # Detection: MLP uses 2-bit (4 LUT entries), Attention uses 4-bit (16 entries)
            skip_attn_lut = os.environ.get('ANEMLL_SKIP_ATTN_LUT', '0') == '1'
            lut_size = len(lut_val)
            is_attention = lut_size > 4  # Attention uses 4-bit (16 entries), MLP uses 2-bit (4 entries)

            if skip_attn_lut and is_attention and use_dynamic:
                use_dynamic = False  # Force pre-baked for attention
                if os.environ.get('ANEMLL_DEBUG_CONVERTER'):
                    print(f"  [SKIP_ATTN_LUT] {node.name}: using pre-baked (lut_size={lut_size})")

            # Debug: Check if values are being passed correctly
            if os.environ.get('ANEMLL_DEBUG_CONVERTER'):
                mode_str = "DYNAMIC" if use_dynamic else "PRE-BAKED"
                print(f"\n[CoreML Converter - {mode_str}] {node.name}:")
                print(f"  indices shape: {indices_val.shape}, unique: {len(np.unique(indices_val))}")
                print(f"  lut: {lut_val}")
                print(f"  scale_A: shape={scale_A_val.shape}, range=[{scale_A_val.min():.4f}, {scale_A_val.max():.4f}]")
                print(f"  scale_B: shape={scale_B_val.shape}, range=[{scale_B_val.min():.4f}, {scale_B_val.max():.4f}]")

            # Cast input to fp16
            x_fp16 = mb.cast(x=x, dtype="fp16", name=node.name + "_x_fp16")

            if use_dynamic:
                # ============= DYNAMIC MODE (iOS18) =============
                # Uses constexpr_lut_to_dense + matmul for dynamic A@B
                #
                # IMPORTANT: Use 4D shapes throughout for ANE compatibility.
                # ANE requires consistent 4D tensors for conv weight computation.

                # Determine nbits from lut size
                lut_size = len(lut_val)
                nbits = int(np.ceil(np.log2(lut_size))) if lut_size > 1 else 1

                # iOS18 format: LUT rank = indices rank + 2
                # For 4D indices [out, in, 1, 1], LUT must be 6D: [1, 1, 1, 1, num_levels, 1]
                lut_6d = lut_val.astype(np.float16).reshape(1, 1, 1, 1, lut_size, 1)

                # iOS18 requires proper sub-byte dtype for indices
                from coremltools.converters.mil.mil.types import (
                    np_uint2_dtype, np_uint4_dtype
                )
                if nbits <= 2:
                    indices_dtype = np_uint2_dtype
                elif nbits <= 4:
                    indices_dtype = np_uint4_dtype
                else:
                    indices_dtype = np.uint8

                # Use 4D indices [out, in, 1, 1] with proper sub-byte dtype for iOS18
                # This ensures constexpr_lut_to_dense outputs 4D directly
                indices_4d = indices_val.reshape(out_features, in_features, 1, 1)
                indices_subbyte = indices_4d.astype(np.uint8).astype(indices_dtype)

                # constexpr_lut_to_dense outputs 4D [out, in, 1, 1]
                base_weights = mb.constexpr_lut_to_dense(
                    indices=indices_subbyte,  # 4D [out_features, in_features, 1, 1]
                    lut=lut_6d,  # 6D [1, 1, 1, 1, lut_size, 1]
                    name=node.name + "_base_weights"
                )

                # Option to pre-compute A@B at conversion time (faster but larger model)
                prebake_scales = os.environ.get('ANEMLL_PREBAKE_SCALES', '0') == '1'

                if prebake_scales:
                    # Pre-compute scales at conversion time (faster inference, larger model)
                    scales_np = scale_A_val.astype(np.float32) @ scale_B_val.astype(np.float32)
                    scales_np = np.clip(scales_np, 1e-8, 65504).astype(np.float16)
                    scales_4d = mb.const(
                        val=scales_np.reshape(out_features, in_features, 1, 1),
                        name=node.name + "_scales_4d"
                    )
                    if os.environ.get('ANEMLL_DEBUG_CONVERTER'):
                        print(f"  [PREBAKE_SCALES] {node.name}: pre-computed A@B, scales shape={scales_np.shape}")
                else:
                    # Scale factors as constants (will be matmul'd at runtime)
                    # Factored scales: A @ B computed at runtime for compression benefit
                    A_fp16 = mb.const(val=scale_A_val.astype(np.float16), name=node.name + "_scale_A")
                    B_fp16 = mb.const(val=scale_B_val.astype(np.float16), name=node.name + "_scale_B")

                    # Dynamic A @ B (computed at runtime) - produces 2D [out, in]
                    scales_2d = mb.matmul(x=A_fp16, y=B_fp16, name=node.name + "_scales")

                    # Option to skip clip for speed (use if trained scales are already positive)
                    skip_clip = os.environ.get('ANEMLL_SKIP_CLIP', '0') == '1'

                    if skip_clip:
                        # Skip clipping - scales assumed positive from training
                        scales_for_reshape = scales_2d
                    else:
                        # Clamp scales to positive (still 2D)
                        scales_for_reshape = mb.clip(
                            x=scales_2d,
                            alpha=np.float16(1e-8),
                            beta=np.float16(65504),
                            name=node.name + "_scales_clamped"
                        )

                    # Reshape scales to 4D [out, in, 1, 1] for element-wise mul with 4D base_weights
                    scales_4d = mb.reshape(
                        x=scales_for_reshape,
                        shape=[out_features, in_features, 1, 1],
                        name=node.name + "_scales_4d"
                    )

                # W_eff = W_base * scales (4D * 4D element-wise)
                weights_4d = mb.mul(x=base_weights, y=scales_4d, name=node.name + "_weights_4d")

                # Conv with 4D weights directly (no reshape needed)
                output = mb.conv(
                    x=x_fp16,
                    weight=weights_4d,
                    pad_type="valid",
                    name=node.name
                )

                if os.environ.get('ANEMLL_DEBUG_CONVERTER'):
                    print(f"  -> Using constexpr_lut_to_dense (4D) + matmul (DYNAMIC, ANE-compatible)")

            else:
                # ============= PRE-BAKED MODE (ANE-compatible) =============
                # Bake weights at conversion time

                # Step 1: Reconstruct base weights from LUT indices
                base_weights_np = lut_val[indices_val.astype(np.int64)]  # [out, in]

                # Step 2: Compute scales = A @ B
                scales_np = scale_A_val.astype(np.float32) @ scale_B_val.astype(np.float32)

                # Step 3: Clamp scales to positive
                scales_np = np.clip(scales_np, 1e-8, 65504)

                # Step 4: Compute effective weights = base * scales
                effective_weights = (base_weights_np.astype(np.float32) * scales_np).astype(np.float16)

                # Debug: Show computed values
                if os.environ.get('ANEMLL_DEBUG_CONVERTER'):
                    print(f"  base_weights range: [{base_weights_np.min():.4f}, {base_weights_np.max():.4f}]")
                    print(f"  scales (A@B) range: [{scales_np.min():.6f}, {scales_np.max():.6f}]")
                    print(f"  effective_weights range: [{effective_weights.min():.4f}, {effective_weights.max():.4f}]")
                    print(f"  -> Using static pre-baked weights (ANE-compatible)")

                # Step 5: Reshape for conv [out, in] -> [out, in, 1, 1]
                effective_weights = effective_weights.reshape(out_features, in_features, 1, 1)

                # Create static weight constant (ANE-compatible)
                weight_const = mb.const(
                    val=effective_weights,
                    name=node.name + "_weight"
                )

                # Conv with STATIC weights -> runs on ANE
                output = mb.conv(
                    x=x_fp16,
                    weight=weight_const,
                    pad_type="valid",
                    name=node.name
                )

            context.add(output)

        return True

    except ImportError:
        # coremltools not installed
        return False
    except Exception as e:
        print(f"Warning: Could not register coreml converter: {e}")
        return False


# Register custom op at module import time
_register_custom_op()

# Flag to track if coreml converter is registered
_COREML_CONVERTER_REGISTERED = False

def ensure_coreml_converter_registered():
    """Ensure the coremltools converter is registered (call before conversion)."""
    global _COREML_CONVERTER_REGISTERED
    if not _COREML_CONVERTER_REGISTERED:
        _COREML_CONVERTER_REGISTERED = _register_coreml_converter()
    return _COREML_CONVERTER_REGISTERED


def pack_indices_to_bits(indices: np.ndarray, nbits: int) -> np.ndarray:
    """Pack integer indices into bit-packed uint8 array for constexpr_lut_to_dense.

    Args:
        indices: np.ndarray of shape [out_features, in_features] with integer indices
        nbits: Number of bits per index (e.g., 2 for 4-level LUT)

    Returns:
        np.ndarray of packed uint8 values
    """
    indices_flat = indices.flatten().astype(np.uint8)
    total_elements = len(indices_flat)
    indices_per_byte = 8 // nbits

    # Calculate padded length
    padded_len = ((total_elements + indices_per_byte - 1) // indices_per_byte) * indices_per_byte
    if padded_len > total_elements:
        indices_flat = np.pad(indices_flat, (0, padded_len - total_elements), constant_values=0)

    # Pack indices
    packed_len = padded_len // indices_per_byte
    packed = np.zeros(packed_len, dtype=np.uint8)

    for i in range(indices_per_byte):
        shift = i * nbits
        packed |= (indices_flat[i::indices_per_byte] << shift)

    return packed


class AnemllConv2d(nn.Module):
    """Conv2d with low-rank scales for ANEMLL quantization.

    This layer stores:
    - indices: LUT indices (unpacked for PyTorch), shape [out_features, in_features]
    - lut: LUT values, shape [lut_size] (e.g., [-1, -0.33, 0.33, 1.0] for 2-bit)
    - scale_A: Low-rank factor A, shape [out_features, scale_rank]
    - scale_B: Low-rank factor B, shape [scale_rank, in_features]
    - weight: (legacy) Reconstructed LUT[indices] for backward compat

    Forward computes: conv2d(x, LUT[indices] * clamp(scale_A @ scale_B, min=1e-8))

    For CoreML conversion:
    - get_packed_indices() returns bit-packed indices for constexpr_lut_to_dense
    - Custom converter emits: W_base = constexpr_lut_to_dense(packed_indices, lut)
    - Custom converter emits: scales = matmul(A, B), W = W_base * clip(scales)

    Args:
        in_features: Number of input features
        out_features: Number of output features
        scale_rank: Rank for low-rank scale factorization
        bias: Whether to include bias (default: False)
        dtype: Data type for parameters (default: torch.float16)
        lut_bits: Number of bits for LUT (default: 2 for 4-level quantization)
        use_custom_op: Use custom op for forward (enables trace-based CoreML conversion)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale_rank: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
        lut_bits: Optional[int] = 2,
        use_custom_op: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_rank = scale_rank
        self.lut_bits = lut_bits if lut_bits else 2
        self.lut_size = 1 << self.lut_bits
        self.use_custom_op = use_custom_op

        # Indices into LUT (unpacked for PyTorch, will pack for CoreML)
        self.register_buffer('indices', torch.zeros(out_features, in_features, dtype=torch.long))

        # LUT values (e.g., [-1, -0.33, 0.33, 1.0] for 2-bit)
        default_lut = torch.linspace(-1.0, 1.0, self.lut_size, dtype=dtype)
        self.register_buffer('lut', default_lut)

        # Legacy weight buffer for backward compatibility (reconstructed from lut[indices])
        self.register_buffer('weight', torch.zeros(out_features, in_features, 1, 1, dtype=dtype))

        # Low-rank scale factors
        self.scale_A = nn.Parameter(
            torch.zeros(out_features, scale_rank, dtype=dtype)
        )
        self.scale_B = nn.Parameter(
            torch.zeros(scale_rank, in_features, dtype=dtype)
        )

        # Cache for packed indices (computed lazily)
        self._indices_packed = None

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def bake_weights(self, clamp_scales: bool = True):
        """Pre-compute and store baked weights for testing."""
        with torch.no_grad():
            # Reconstruct base weights from LUT[indices]
            base_weights = self.lut[self.indices]  # [out, in]

            scales = self.scale_A.float() @ self.scale_B.float()
            if clamp_scales:
                scales = scales.clamp(min=1e-8)

            baked = (base_weights.float() * scales).to(self.scale_A.dtype)
            baked = baked.view(self.out_features, self.in_features, 1, 1)
            self.register_buffer('_baked_weight', baked)

    def get_packed_indices(self) -> np.ndarray:
        """Get bit-packed indices for CoreML constexpr_lut_to_dense.

        Returns:
            np.ndarray: Packed indices as uint8 array
        """
        if self._indices_packed is None:
            indices_np = self.indices.cpu().numpy().astype(np.uint8)
            self._indices_packed = pack_indices_to_bits(indices_np, self.lut_bits)
        return self._indices_packed

    def set_indices_from_weights(self, snapped_weights: torch.Tensor = None):
        """Extract indices from snapped weight values.

        Args:
            snapped_weights: Weight tensor with LUT values. If None, uses self.weight.
        """
        with torch.no_grad():
            if snapped_weights is None:
                # Use existing weight buffer
                w = self.weight.squeeze(-1).squeeze(-1)  # [out, in]
            else:
                w = snapped_weights.view(self.out_features, self.in_features)

            # Find nearest LUT value for each weight
            w_flat = w.flatten().float()
            lut_float = self.lut.float()
            distances = torch.abs(w_flat.unsqueeze(1) - lut_float.unsqueeze(0))
            indices = distances.argmin(dim=1)
            self.indices.copy_(indices.view(self.out_features, self.in_features))

            # Update weight buffer for backward compat
            self.weight.copy_(self.lut[self.indices].view(self.out_features, self.in_features, 1, 1))

            # Clear packed cache
            self._indices_packed = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with scale reconstruction.

        Computes: conv2d(x, LUT[indices] * clamp(scale_A @ scale_B, min=1e-8))

        If use_custom_op=True, uses torch.ops.anemll_quant.quant_conv which
        survives tracing and converts to constexpr_lut_to_dense + matmul in CoreML.
        """
        # Use custom op for trace-based conversion
        if self.use_custom_op and _CUSTOM_OP_REGISTERED:
            return torch.ops.anemll_quant.quant_conv(
                x, self.indices, self.lut, self.scale_A, self.scale_B
            )

        # Use pre-baked weight if available (for testing)
        if hasattr(self, '_baked_weight') and self._baked_weight is not None:
            w_eff = self._baked_weight
        else:
            # Reconstruct base weights from LUT[indices]
            base_weights = self.lut[self.indices]  # [out, in]

            # Reconstruct full scale (use fp16 for ANE compatibility)
            scales = self.scale_A @ self.scale_B

            # Clamp to positive values - required for correct inference
            scales = scales.clamp(min=1e-8)

            # Effective weight: LUT[indices] * scale
            w_eff = base_weights * scales

            # Reshape to conv format: [out, in, 1, 1]
            w_eff = w_eff.view(self.out_features, self.in_features, 1, 1)

        # Perform convolution
        return F.conv2d(x, w_eff.to(x.dtype), self.bias)

    def set_lut(self, lut: torch.Tensor):
        """Set the LUT tensor and derive lut_bits from its size."""
        self.lut = lut.to(self.scale_A.dtype)
        self.lut_size = len(lut)
        self.lut_bits = int(math.ceil(math.log2(self.lut_size))) if self.lut_size > 1 else 1
        # Clear packed cache
        self._indices_packed = None

    def set_lut_bits_from_weights(self):
        """Derive lut_bits from the loaded weight values.

        Assumes weights are quantized to values in [-1, 1] with uniform steps.
        The number of unique values determines lut_size and creates LUT.
        """
        with torch.no_grad():
            unique_vals = torch.unique(self.weight)
            self.lut_size = len(unique_vals)
            self.lut_bits = int(math.ceil(math.log2(self.lut_size))) if self.lut_size > 1 else 1

            # Create LUT from unique values (sorted)
            sorted_vals, _ = torch.sort(unique_vals)
            self.lut = sorted_vals.to(self.scale_A.dtype)

            # Extract indices from weights
            self.set_indices_from_weights()

        return self.lut_bits

    @classmethod
    def from_snapped_weights(
        cls,
        snapped_weight: torch.Tensor,  # [out, in] with LUT values
        scale_A: torch.Tensor,         # [out, rank]
        scale_B: torch.Tensor,         # [rank, in]
        lut: torch.Tensor = None,      # [lut_size]
        nbits: int = 2,
        bias: torch.Tensor = None,
    ) -> 'AnemllConv2d':
        """Create AnemllConv2d from snapped weights by extracting indices.

        Args:
            snapped_weight: Weight tensor with LUT values, shape [out, in] or [out, in, 1, 1]
            scale_A: Scale factor A, shape [out, rank]
            scale_B: Scale factor B, shape [rank, in]
            lut: LUT values. If None, creates uniform LUT in [-1, 1]
            nbits: Number of bits for LUT (default: 2)
            bias: Optional bias tensor

        Returns:
            AnemllConv2d with indices, lut, scale_A, scale_B populated
        """
        # Get dimensions
        if snapped_weight.dim() == 4:
            snapped_weight = snapped_weight.squeeze(-1).squeeze(-1)
        out_features, in_features = snapped_weight.shape
        scale_rank = scale_A.shape[1]
        dtype = scale_A.dtype

        # Create module
        module = cls(
            in_features=in_features,
            out_features=out_features,
            scale_rank=scale_rank,
            bias=bias is not None,
            dtype=dtype,
            lut_bits=nbits,
        )

        # Set LUT
        if lut is None:
            lut_size = 1 << nbits
            lut = torch.linspace(-1.0, 1.0, lut_size, dtype=dtype)
        module.set_lut(lut)

        # Copy scale factors
        module.scale_A.data.copy_(scale_A.to(dtype))
        module.scale_B.data.copy_(scale_B.to(dtype))

        # Extract indices from snapped weights
        module.set_indices_from_weights(snapped_weight)

        # Copy bias if present
        if bias is not None:
            module.bias.data.copy_(bias.to(dtype))

        return module

    def build_mil_conv(self, x, name: str = "anemll_conv"):
        """Build CoreML MIL ops for this conv layer using constexpr_lut_to_dense.

        This method creates the MIL graph structure (all 4D for ANE compatibility):
        - W_base = constexpr_lut_to_dense(indices_4d, lut_6d)  # 4D output
        - scales = matmul(scale_A, scale_B)  # 2D
        - scales_4d = reshape(clip(scales), [out, in, 1, 1])  # 4D
        - W = W_base * scales_4d  # 4D * 4D
        - output = conv(x, W)  # 4D weights

        Args:
            x: Input MIL tensor
            name: Name prefix for operations

        Returns:
            MIL tensor: conv output

        Note: Must import coremltools.converters.mil.Builder as mb before calling.
        """
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil.types import (
            np_uint2_dtype, np_uint4_dtype
        )

        # Get numpy arrays
        indices_np = self.indices.cpu().numpy().astype(np.uint8)
        lut_np = self.lut.cpu().numpy().astype(np.float16)
        scale_A_np = self.scale_A.data.cpu().numpy().astype(np.float16)
        scale_B_np = self.scale_B.data.cpu().numpy().astype(np.float16)

        # Determine proper sub-byte dtype
        if self.lut_bits <= 2:
            indices_dtype = np_uint2_dtype
        elif self.lut_bits <= 4:
            indices_dtype = np_uint4_dtype
        else:
            indices_dtype = np.uint8

        # Use 4D indices [out, in, 1, 1] for ANE compatibility
        indices_4d = indices_np.reshape(self.out_features, self.in_features, 1, 1)
        indices_subbyte = indices_4d.astype(indices_dtype)

        # LUT must be 6D for 4D indices: [1, 1, 1, 1, lut_size, 1]
        lut_6d = lut_np.reshape(1, 1, 1, 1, self.lut_size, 1)

        # Build constexpr_lut_to_dense for base weights (outputs 4D)
        base_weights = mb.constexpr_lut_to_dense(
            indices=indices_subbyte,
            lut=lut_6d,
            name=f"{name}_base_weights"
        )

        # Scale factors as constants
        A = mb.const(val=scale_A_np, name=f"{name}_scale_A")
        B = mb.const(val=scale_B_np, name=f"{name}_scale_B")

        # Dynamic A @ B computation (2D)
        scales_2d = mb.matmul(x=A, y=B, name=f"{name}_scales")

        # Clamp to positive values (2D)
        scales_clamped = mb.clip(
            x=scales_2d,
            alpha=np.float16(1e-8),
            beta=np.float16(65504),
            name=f"{name}_scales_clamped"
        )

        # Reshape scales to 4D for element-wise mul with 4D base_weights
        scales_4d = mb.reshape(
            x=scales_clamped,
            shape=[self.out_features, self.in_features, 1, 1],
            name=f"{name}_scales_4d"
        )

        # Apply scales: W = W_base * scales (4D * 4D)
        weights_4d = mb.mul(x=base_weights, y=scales_4d, name=f"{name}_weights_4d")

        # Conv2d operation with 4D weights
        if self.bias is not None:
            bias_np = self.bias.data.cpu().numpy().astype(np.float16)
            output = mb.conv(
                x=x,
                weight=weights_4d,
                bias=bias_np,
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

    def extra_repr(self) -> str:
        lut_info = f", lut_size={self.lut_size}" if self.lut is not None else ""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scale_rank={self.scale_rank}, lut_bits={self.lut_bits}{lut_info}, "
            f"bias={self.bias is not None}"
        )


class AnemllQuantConfig:
    """Configuration for ANEMLL quantization.

    Stores scale ranks and LUT bits for different layer types.
    """

    def __init__(
        self,
        mlp_scale_rank: int = DEFAULT_MLP_SCALE_RANK,
        attn_scale_rank: int = DEFAULT_ATTN_SCALE_RANK,
        lut_bits: Optional[int] = None,
    ):
        self.mlp_scale_rank = mlp_scale_rank
        self.attn_scale_rank = attn_scale_rank
        self.lut_bits = lut_bits

    def get_scale_rank(self, layer_name: str) -> int:
        """Get scale rank based on layer name."""
        if any(proj in layer_name for proj in ['gate_proj', 'up_proj', 'down_proj']):
            return self.mlp_scale_rank
        elif any(proj in layer_name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            return self.attn_scale_rank
        else:
            # Default to MLP rank for unknown layers
            return self.mlp_scale_rank


def load_anemll_weights(
    model: nn.Module,
    checkpoint_path: str,
    quant_config: Optional[AnemllQuantConfig] = None,
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[list, list]:
    """Load snapped ANEMLL weights into a model.

    Expects checkpoint to contain for each quantized layer:
    - {layer_name}.weight: LUT[idx] values in [-1, 1], shape [out, in]
    - {layer_name}.scale_A: shape [out, rank]
    - {layer_name}.scale_B: shape [rank, in]
    - {layer_name}.lut: (optional) LUT values, shape [lut_size]

    Args:
        model: Model with AnemllConv2d layers
        checkpoint_path: Path to checkpoint file or directory
        quant_config: Quantization configuration (optional)
        strict: Whether to require all keys to match
        verbose: Whether to print loading information

    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    if quant_config is None:
        quant_config = AnemllQuantConfig()

    # Load state dict
    if os.path.isdir(checkpoint_path):
        state_dict = _load_from_directory(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    missing_keys = []
    unexpected_keys = list(state_dict.keys())
    loaded_count = 0

    for name, module in model.named_modules():
        if not isinstance(module, AnemllConv2d):
            continue

        # Try different key formats to match checkpoint keys
        # Model module name: model.layers.0.self_attn.q_proj
        # Checkpoint key: model.layers.0.self_attn.q_proj.weight
        base_keys = [
            name,                           # e.g., model.layers.0.self_attn.q_proj
            name.replace('model.', ''),     # e.g., layers.0.self_attn.q_proj
            f'model.{name}',                # e.g., model.model.layers.0.self_attn.q_proj
            name.replace('model.model.', 'model.'),  # handle nested model prefix
        ]

        loaded = False
        for base_key in base_keys:
            weight_key = f'{base_key}.weight'
            scale_a_key = f'{base_key}.scale_A'
            scale_b_key = f'{base_key}.scale_B'
            lut_key = f'{base_key}.lut'

            if weight_key in state_dict:
                # Load weight [out, in] -> [out, in, 1, 1]
                w = state_dict[weight_key]
                if w.dim() == 2:
                    w = w.view(w.shape[0], w.shape[1], 1, 1)
                module.weight.data.copy_(w.to(module.weight.dtype))

                # Remove from unexpected
                if weight_key in unexpected_keys:
                    unexpected_keys.remove(weight_key)

                # Load scales if present
                if scale_a_key in state_dict:
                    module.scale_A.data.copy_(
                        state_dict[scale_a_key].to(module.scale_A.dtype)
                    )
                    if scale_a_key in unexpected_keys:
                        unexpected_keys.remove(scale_a_key)
                else:
                    # Initialize to identity-like if not present
                    _init_scale_identity(module)
                    if verbose:
                        print(f"  Warning: {scale_a_key} not found, using identity init")

                if scale_b_key in state_dict:
                    module.scale_B.data.copy_(
                        state_dict[scale_b_key].to(module.scale_B.dtype)
                    )
                    if scale_b_key in unexpected_keys:
                        unexpected_keys.remove(scale_b_key)

                # Load LUT if present and derive lut_bits from it
                if lut_key in state_dict:
                    lut_tensor = state_dict[lut_key]
                    module.set_lut(lut_tensor.to(module.weight.dtype))
                    if lut_key in unexpected_keys:
                        unexpected_keys.remove(lut_key)
                else:
                    # Derive LUT bits from weights if LUT not provided
                    module.set_lut_bits_from_weights()

                loaded = True
                loaded_count += 1

                if verbose:
                    lut_info = f", lut_size={module.lut_size}" if module.lut is not None else ""
                    print(f"  Loaded {base_key}: weight{list(w.shape)}, "
                          f"scale_A{list(module.scale_A.shape)}, "
                          f"scale_B{list(module.scale_B.shape)}, "
                          f"lut_bits={module.lut_bits}{lut_info}")
                break

        if not loaded:
            missing_keys.append(name)

    if verbose:
        print(f"\nLoaded {loaded_count} AnemllConv2d layers")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys and strict:
            print(f"Unexpected keys: {unexpected_keys[:10]}...")

    return missing_keys, unexpected_keys


def load_anemll_checkpoint_full(
    model: nn.Module,
    checkpoint_path: str,
    verbose: bool = True,
) -> Tuple[list, list]:
    """Load a full ANEMLL checkpoint including non-quantized layers.

    This loads both:
    - AnemllConv2d layers (with weight, scale_A, scale_B, lut)
    - Non-quantized layers (embeddings, norms, lm_head)

    Args:
        model: Model with AnemllConv2d layers
        checkpoint_path: Path to checkpoint file
        verbose: Whether to print loading information

    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    # Load state dict
    if os.path.isdir(checkpoint_path):
        state_dict = _load_from_directory(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if verbose:
        print(f"Loading checkpoint with {len(state_dict)} keys")

    # First, load AnemllConv2d layers
    missing_quant, unexpected = load_anemll_weights(model, checkpoint_path, verbose=verbose)

    # Handle lm_head weight splitting if needed
    lm_head_key = 'lm_head.weight'
    if lm_head_key in unexpected:
        lm_head_weight = state_dict[lm_head_key]

        # Check for 16-way split
        if hasattr(model, 'lm_head16_1'):
            vocab_size = lm_head_weight.shape[0]
            vocab_split = vocab_size // 16
            vocab_remainder = vocab_size % 16
            split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(16)]

            # Reshape to Conv2d format [out, in] -> [out, in, 1, 1]
            reshaped = lm_head_weight.view(vocab_size, -1, 1, 1)
            splits = torch.split(reshaped, split_sizes)

            for i, split in enumerate(splits):
                head = getattr(model, f'lm_head16_{i + 1}')
                head.weight.data.copy_(split.to(head.weight.dtype))

            unexpected.remove(lm_head_key)
            if verbose:
                print(f"  Loaded lm_head.weight split into 16 parts")

        # Check for 8-way split
        elif hasattr(model, 'lm_head8_1'):
            vocab_size = lm_head_weight.shape[0]
            vocab_split = vocab_size // 8
            vocab_remainder = vocab_size % 8
            split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(8)]

            reshaped = lm_head_weight.view(vocab_size, -1, 1, 1)
            splits = torch.split(reshaped, split_sizes)

            for i, split in enumerate(splits):
                head = getattr(model, f'lm_head8_{i + 1}')
                head.weight.data.copy_(split.to(head.weight.dtype))

            unexpected.remove(lm_head_key)
            if verbose:
                print(f"  Loaded lm_head.weight split into 8 parts")

        # Check for 2-way split
        elif hasattr(model, 'lm_head2_1'):
            vocab_size = lm_head_weight.shape[0]
            vocab_split = vocab_size // 2

            reshaped = lm_head_weight.view(vocab_size, -1, 1, 1)
            split1, split2 = torch.split(reshaped, [vocab_split, vocab_size - vocab_split])

            model.lm_head2_1.weight.data.copy_(split1.to(model.lm_head2_1.weight.dtype))
            model.lm_head2_2.weight.data.copy_(split2.to(model.lm_head2_2.weight.dtype))

            unexpected.remove(lm_head_key)
            if verbose:
                print(f"  Loaded lm_head.weight split into 2 parts")

        # Single lm_head
        elif hasattr(model, 'lm_head1'):
            reshaped = lm_head_weight.view(lm_head_weight.shape[0], -1, 1, 1)
            model.lm_head1.weight.data.copy_(reshaped.to(model.lm_head1.weight.dtype))
            unexpected.remove(lm_head_key)
            if verbose:
                print(f"  Loaded lm_head.weight")

        elif hasattr(model, 'lm_head'):
            reshaped = lm_head_weight.view(lm_head_weight.shape[0], -1, 1, 1)
            model.lm_head.weight.data.copy_(reshaped.to(model.lm_head.weight.dtype))
            unexpected.remove(lm_head_key)
            if verbose:
                print(f"  Loaded lm_head.weight")

    # Now load non-quantized layers
    loaded_other = 0
    for key in list(unexpected):
        # Skip keys that are for quantized layers (already handled)
        if any(x in key for x in ['.scale_A', '.scale_B', '.lut']):
            continue

        # Try to find the corresponding parameter in model
        parts = key.split('.')

        # Handle different key formats
        try:
            # Try direct assignment
            param = model
            for part in parts:
                if part.isdigit():
                    param = param[int(part)]
                else:
                    param = getattr(param, part)

            if isinstance(param, nn.Parameter):
                param.data.copy_(state_dict[key].to(param.dtype))
                unexpected.remove(key)
                loaded_other += 1
            elif isinstance(param, torch.Tensor) and not isinstance(param, nn.Parameter):
                # It's a buffer
                param.copy_(state_dict[key].to(param.dtype))
                unexpected.remove(key)
                loaded_other += 1
        except (AttributeError, IndexError, KeyError):
            pass

    if verbose:
        print(f"Loaded {loaded_other} non-quantized parameters")
        if unexpected:
            print(f"Remaining unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    return missing_quant, unexpected


def load_baked_weights_for_ane(
    model: nn.Module,
    checkpoint_path: str,
    verbose: bool = True,
) -> bool:
    """Load ANEMLL checkpoint with BAKED weights for ANE/CoreML conversion.

    This function bakes the weights (snapped * scale_A @ scale_B) and loads
    them into regular nn.Conv2d layers. Use this for CoreML conversion
    since CoreML cannot do dynamic scale computation.

    Args:
        model: Model with nn.Conv2d layers (not AnemllConv2d)
        checkpoint_path: Path to ANEMLL checkpoint with snapped weights + scales
        verbose: Whether to print loading information

    Returns:
        True if loading succeeded, False otherwise
    """
    # Load state dict
    if os.path.isdir(checkpoint_path):
        state_dict = _load_from_directory(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if verbose:
        print(f"Loading ANEMLL checkpoint for ANE conversion: {len(state_dict)} keys")

    # Identify quantized projection layer names
    quantized_proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    # Track loaded layers
    baked_count = 0
    other_count = 0

    # First pass: bake and load quantized projection weights
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module.kernel_size != (1, 1):
            continue
        if not any(proj in name for proj in quantized_proj_names):
            continue

        # Try different key formats
        base_keys = [
            name,
            name.replace('model.model.', 'model.'),
            name.replace('model.', ''),
            f'model.{name}',
        ]

        loaded = False
        for base_key in base_keys:
            weight_key = f'{base_key}.weight'
            scale_a_key = f'{base_key}.scale_A'
            scale_b_key = f'{base_key}.scale_B'

            if weight_key in state_dict and scale_a_key in state_dict and scale_b_key in state_dict:
                # Load snapped weights and scales
                snapped = state_dict[weight_key].to(torch.float32)
                scale_A = state_dict[scale_a_key].to(torch.float32)
                scale_B = state_dict[scale_b_key].to(torch.float32)

                # Compute baked weight: snapped * (scale_A @ scale_B)
                # Clamp scales to positive values (matching AnemllConv2d.forward behavior)
                scales = (scale_A @ scale_B).clamp(min=1e-8)

                # Handle shape: snapped is [out, in] or [out, in, 1, 1]
                if snapped.dim() == 4:
                    snapped = snapped.squeeze(-1).squeeze(-1)
                baked = snapped * scales

                # Reshape to Conv2d format [out, in, 1, 1]
                baked_4d = baked.view(baked.shape[0], baked.shape[1], 1, 1)

                # Load into module
                with torch.no_grad():
                    module.weight.data.copy_(baked_4d.to(module.weight.dtype))

                baked_count += 1
                loaded = True
                if verbose and baked_count <= 5:
                    print(f"  Baked {base_key}: snapped{list(snapped.shape)} * scales -> baked{list(baked_4d.shape)}")
                break

        if not loaded and verbose:
            # Check if this is a quantized layer that should have been loaded
            if any(proj in name for proj in quantized_proj_names):
                print(f"  Warning: Could not find checkpoint data for {name}")

    if verbose:
        print(f"  ...and {baked_count - 5} more projection layers" if baked_count > 5 else "")
        print(f"  Total baked: {baked_count} projection layers")

    # Second pass: load non-quantized weights (embed_tokens, norms, lm_head)
    # Handle lm_head weight splitting
    lm_head_key = 'lm_head.weight'
    if lm_head_key in state_dict:
        lm_head_weight = state_dict[lm_head_key]

        # Check for 16-way split
        if hasattr(model, 'lm_head16_1'):
            vocab_size = lm_head_weight.shape[0]
            vocab_split = vocab_size // 16
            vocab_remainder = vocab_size % 16
            split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(16)]
            reshaped = lm_head_weight.view(vocab_size, -1, 1, 1)
            splits = torch.split(reshaped, split_sizes)
            for i, split in enumerate(splits):
                head = getattr(model, f'lm_head16_{i + 1}')
                head.weight.data.copy_(split.to(head.weight.dtype))
            other_count += 1
            if verbose:
                print(f"  Loaded lm_head.weight split into 16 parts")

        # Check for 8-way split
        elif hasattr(model, 'lm_head8_1'):
            vocab_size = lm_head_weight.shape[0]
            vocab_split = vocab_size // 8
            vocab_remainder = vocab_size % 8
            split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(8)]
            reshaped = lm_head_weight.view(vocab_size, -1, 1, 1)
            splits = torch.split(reshaped, split_sizes)
            for i, split in enumerate(splits):
                head = getattr(model, f'lm_head8_{i + 1}')
                head.weight.data.copy_(split.to(head.weight.dtype))
            other_count += 1
            if verbose:
                print(f"  Loaded lm_head.weight split into 8 parts")

        # Check for 2-way split
        elif hasattr(model, 'lm_head2_1'):
            vocab_size = lm_head_weight.shape[0]
            vocab_split = vocab_size // 2
            reshaped = lm_head_weight.view(vocab_size, -1, 1, 1)
            split1, split2 = torch.split(reshaped, [vocab_split, vocab_size - vocab_split])
            model.lm_head2_1.weight.data.copy_(split1.to(model.lm_head2_1.weight.dtype))
            model.lm_head2_2.weight.data.copy_(split2.to(model.lm_head2_2.weight.dtype))
            other_count += 1
            if verbose:
                print(f"  Loaded lm_head.weight split into 2 parts")

        # Single lm_head
        elif hasattr(model, 'lm_head1'):
            reshaped = lm_head_weight.view(lm_head_weight.shape[0], -1, 1, 1)
            model.lm_head1.weight.data.copy_(reshaped.to(model.lm_head1.weight.dtype))
            other_count += 1
            if verbose:
                print(f"  Loaded lm_head.weight")

        elif hasattr(model, 'lm_head'):
            reshaped = lm_head_weight.view(lm_head_weight.shape[0], -1, 1, 1)
            model.lm_head.weight.data.copy_(reshaped.to(model.lm_head.weight.dtype))
            other_count += 1
            if verbose:
                print(f"  Loaded lm_head.weight")

    # Load embed_tokens
    embed_key = 'model.embed_tokens.weight'
    if embed_key in state_dict:
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                model.model.embed_tokens.weight.data.copy_(
                    state_dict[embed_key].to(model.model.embed_tokens.weight.dtype)
                )
                other_count += 1
                if verbose:
                    print(f"  Loaded embed_tokens.weight")
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to load embed_tokens: {e}")

    # Load layer norms
    norm_keys = [k for k in state_dict.keys() if 'norm' in k.lower() or 'layernorm' in k.lower()]
    for key in norm_keys:
        parts = key.split('.')
        try:
            param = model
            for part in parts:
                if part.isdigit():
                    param = param[int(part)]
                else:
                    param = getattr(param, part)
            if isinstance(param, nn.Parameter):
                param.data.copy_(state_dict[key].to(param.dtype))
                other_count += 1
        except (AttributeError, IndexError, KeyError):
            pass

    if verbose:
        print(f"  Loaded {other_count} non-projection weights (embed, norms, lm_head)")
        print(f"\nANE-ready weights loaded successfully!")

    return baked_count > 0


def load_dynamic_weights_for_ane(
    model: nn.Module,
    checkpoint_path: str,
    verbose: bool = True,
) -> bool:
    """Load ANEMLL checkpoint with DYNAMIC A*B computation for ANE/CoreML.

    This function replaces nn.Conv2d layers with AnemllConv2d layers that store
    scale_A, scale_B, and snapped weights separately. During tracing, the forward
    pass computes W * (A @ B) dynamically, and CoreML captures these operations.

    This preserves the quantization structure in the final CoreML model:
    - const A: [out_features, rank]
    - const B: [rank, in_features]
    - const W: [out_features, in_features] (snapped LUT values)
    - runtime: scales = matmul(A, B), W_effective = W * scales, output = conv(x, W_effective)

    Args:
        model: Model with nn.Conv2d layers (not AnemllConv2d)
        checkpoint_path: Path to ANEMLL checkpoint with snapped weights + scales
        verbose: Whether to print loading information

    Returns:
        True if loading succeeded, False otherwise
    """
    # Load state dict
    if os.path.isdir(checkpoint_path):
        state_dict = _load_from_directory(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if verbose:
        print(f"Loading ANEMLL checkpoint for DYNAMIC A*B conversion: {len(state_dict)} keys")

    # Identify quantized projection layer names
    quantized_proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    # Track loaded layers
    replaced_count = 0
    other_count = 0

    # First pass: replace nn.Conv2d with AnemllConv2d and load weights
    modules_to_replace = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module.kernel_size != (1, 1):
            continue
        if not any(proj in name for proj in quantized_proj_names):
            continue

        # Try different key formats to find checkpoint data
        base_keys = [
            name,
            name.replace('model.model.', 'model.'),
            name.replace('model.', ''),
            f'model.{name}',
        ]

        for base_key in base_keys:
            weight_key = f'{base_key}.weight'
            scale_a_key = f'{base_key}.scale_A'
            scale_b_key = f'{base_key}.scale_B'

            if weight_key in state_dict and scale_a_key in state_dict and scale_b_key in state_dict:
                scale_A = state_dict[scale_a_key]
                scale_rank = scale_A.shape[1]
                modules_to_replace.append((name, module, base_key, scale_rank))
                break

    # Check if dynamic mode is enabled (for constexpr_lut_to_dense + matmul)
    use_dynamic = os.environ.get('ANEMLL_DYNAMIC_SCALES', '0') == '1'
    if use_dynamic and verbose:
        print("  Dynamic mode enabled: using custom op for constexpr_lut_to_dense + matmul")

    # Replace modules
    for name, old_module, base_key, scale_rank in modules_to_replace:
        # Create AnemllConv2d with custom op enabled for dynamic mode
        new_module = AnemllConv2d(
            in_features=old_module.in_channels,
            out_features=old_module.out_channels,
            scale_rank=scale_rank,
            bias=old_module.bias is not None,
            dtype=old_module.weight.dtype,
            use_custom_op=use_dynamic,  # Enable custom op for dynamic tracing
        )

        # Load checkpoint data
        weight_key = f'{base_key}.weight'
        scale_a_key = f'{base_key}.scale_A'
        scale_b_key = f'{base_key}.scale_B'

        snapped = state_dict[weight_key].to(torch.float32)
        scale_A = state_dict[scale_a_key].to(torch.float32)
        scale_B = state_dict[scale_b_key].to(torch.float32)

        # Handle shape: snapped is [out, in] or [out, in, 1, 1]
        if snapped.dim() == 2:
            snapped = snapped.view(snapped.shape[0], snapped.shape[1], 1, 1)

        # Load into AnemllConv2d
        new_module.weight.data.copy_(snapped.to(new_module.weight.dtype))
        new_module.scale_A.data.copy_(scale_A.to(new_module.scale_A.dtype))
        new_module.scale_B.data.copy_(scale_B.to(new_module.scale_B.dtype))

        # Derive LUT bits from weights
        new_module.set_lut_bits_from_weights()

        # Replace in parent module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

        replaced_count += 1
        if verbose and replaced_count <= 5:
            print(f"  Replaced {base_key}: snapped{list(snapped.shape)}, "
                  f"scale_A{list(scale_A.shape)}, scale_B{list(scale_B.shape)}, "
                  f"lut_bits={new_module.lut_bits}")

    if verbose:
        if replaced_count > 5:
            print(f"  ...and {replaced_count - 5} more projection layers")
        print(f"  Total replaced: {replaced_count} Conv2d -> AnemllConv2d (dynamic A*B)")

    # Second pass: load non-quantized weights (embed_tokens, norms, lm_head)
    # Handle lm_head weight splitting
    lm_head_key = 'lm_head.weight'
    if lm_head_key in state_dict:
        lm_head_weight = state_dict[lm_head_key]

        # Check for 16-way split
        if hasattr(model, 'lm_head16_1'):
            vocab_size = lm_head_weight.shape[0]
            vocab_split = vocab_size // 16
            vocab_remainder = vocab_size % 16
            split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(16)]
            reshaped = lm_head_weight.view(vocab_size, -1, 1, 1)
            splits = torch.split(reshaped, split_sizes)
            for i, split in enumerate(splits):
                head = getattr(model, f'lm_head16_{i + 1}')
                head.weight.data.copy_(split.to(head.weight.dtype))
            other_count += 1
            if verbose:
                print(f"  Loaded lm_head.weight split into 16 parts")

        # Check for 8-way split
        elif hasattr(model, 'lm_head8_1'):
            vocab_size = lm_head_weight.shape[0]
            vocab_split = vocab_size // 8
            vocab_remainder = vocab_size % 8
            split_sizes = [vocab_split + (1 if i < vocab_remainder else 0) for i in range(8)]
            reshaped = lm_head_weight.view(vocab_size, -1, 1, 1)
            splits = torch.split(reshaped, split_sizes)
            for i, split in enumerate(splits):
                head = getattr(model, f'lm_head8_{i + 1}')
                head.weight.data.copy_(split.to(head.weight.dtype))
            other_count += 1
            if verbose:
                print(f"  Loaded lm_head.weight split into 8 parts")

        # Check for 2-way split
        elif hasattr(model, 'lm_head2_1'):
            vocab_size = lm_head_weight.shape[0]
            vocab_split = vocab_size // 2
            reshaped = lm_head_weight.view(vocab_size, -1, 1, 1)
            split1, split2 = torch.split(reshaped, [vocab_split, vocab_size - vocab_split])
            model.lm_head2_1.weight.data.copy_(split1.to(model.lm_head2_1.weight.dtype))
            model.lm_head2_2.weight.data.copy_(split2.to(model.lm_head2_2.weight.dtype))
            other_count += 1
            if verbose:
                print(f"  Loaded lm_head.weight split into 2 parts")

        # Single lm_head
        elif hasattr(model, 'lm_head1'):
            reshaped = lm_head_weight.view(lm_head_weight.shape[0], -1, 1, 1)
            model.lm_head1.weight.data.copy_(reshaped.to(model.lm_head1.weight.dtype))
            other_count += 1
            if verbose:
                print(f"  Loaded lm_head.weight")

        elif hasattr(model, 'lm_head'):
            reshaped = lm_head_weight.view(lm_head_weight.shape[0], -1, 1, 1)
            model.lm_head.weight.data.copy_(reshaped.to(model.lm_head.weight.dtype))
            other_count += 1
            if verbose:
                print(f"  Loaded lm_head.weight")

    # Load embed_tokens
    embed_key = 'model.embed_tokens.weight'
    if embed_key in state_dict:
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                model.model.embed_tokens.weight.data.copy_(
                    state_dict[embed_key].to(model.model.embed_tokens.weight.dtype)
                )
                other_count += 1
                if verbose:
                    print(f"  Loaded embed_tokens.weight")
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to load embed_tokens: {e}")

    # Load layer norms
    norm_keys = [k for k in state_dict.keys() if 'norm' in k.lower() or 'layernorm' in k.lower()]
    for key in norm_keys:
        parts = key.split('.')
        try:
            param = model
            for part in parts:
                if part.isdigit():
                    param = param[int(part)]
                else:
                    param = getattr(param, part)
            if isinstance(param, nn.Parameter):
                param.data.copy_(state_dict[key].to(param.dtype))
                other_count += 1
        except (AttributeError, IndexError, KeyError):
            pass

    if verbose:
        print(f"  Loaded {other_count} non-projection weights (embed, norms, lm_head)")
        print(f"\nDynamic A*B weights loaded successfully!")

    return replaced_count > 0


def _load_from_directory(path: str) -> Dict[str, torch.Tensor]:
    """Load state dict from a directory of safetensors/pt files."""
    import safetensors.torch

    state_dict = {}
    for file in os.listdir(path):
        if file.endswith('.safetensors'):
            state_dict.update(
                safetensors.torch.load_file(os.path.join(path, file))
            )
        elif file.endswith('.pt') or file.endswith('.bin'):
            state_dict.update(
                torch.load(os.path.join(path, file), map_location='cpu')
            )
    return state_dict


def _init_scale_identity(module: AnemllConv2d):
    """Initialize scale factors to approximate identity.

    Sets scale_A and scale_B such that scale_A @ scale_B ≈ ones.
    """
    with torch.no_grad():
        # Simple initialization: scale_A = ones/sqrt(rank), scale_B = ones*sqrt(rank)/in_features
        # This gives scale_A @ scale_B ≈ ones when summed
        rank = module.scale_rank
        module.scale_A.fill_(1.0 / math.sqrt(rank))
        module.scale_B.fill_(math.sqrt(rank) / module.in_features)


def convert_conv2d_to_anemll(
    module: nn.Conv2d,
    scale_rank: int,
    dtype: torch.dtype = torch.float16,
) -> AnemllConv2d:
    """Convert a standard Conv2d to AnemllConv2d.

    Useful for initializing from pretrained weights before quantization.

    Args:
        module: Standard nn.Conv2d with kernel_size=1
        scale_rank: Rank for scale factorization
        dtype: Target dtype

    Returns:
        AnemllConv2d with weights copied and scales initialized to identity
    """
    assert module.kernel_size == (1, 1), "Only kernel_size=1 supported"

    out_features = module.out_channels
    in_features = module.in_channels

    anemll_conv = AnemllConv2d(
        in_features=in_features,
        out_features=out_features,
        scale_rank=scale_rank,
        bias=module.bias is not None,
        dtype=dtype,
    )

    # Copy weight (already in correct shape)
    anemll_conv.weight.data.copy_(module.weight.data.to(dtype))

    # Initialize scales to identity-like (so weight * scale ≈ weight initially)
    _init_scale_identity(anemll_conv)

    # Copy bias if present
    if module.bias is not None:
        anemll_conv.bias.data.copy_(module.bias.data.to(dtype))

    return anemll_conv


# LUT utilities for conversion pipeline

def compute_lut_indices(
    weights: torch.Tensor,
    lut_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute LUT indices from quantized weights.

    Args:
        weights: Weight tensor with values in [-1, 1]
        lut_size: Number of LUT entries (e.g., 4 for 2-bit)

    Returns:
        Tuple of (indices, lut) where:
        - indices: Integer indices into LUT, same shape as weights
        - lut: The LUT values, shape [lut_size]
    """
    # Create uniform LUT in [-1, 1]
    lut = torch.linspace(-1.0, 1.0, lut_size, dtype=weights.dtype, device=weights.device)

    # Compute step size
    step = 2.0 / (lut_size - 1)

    # Quantize to indices: idx = round((weight + 1) / step)
    indices = torch.round((weights + 1.0) / step).long()
    indices = torch.clamp(indices, 0, lut_size - 1)

    return indices, lut


def reconstruct_from_lut(
    indices: torch.Tensor,
    lut: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct weights from LUT indices.

    Args:
        indices: Integer indices into LUT
        lut: The LUT values

    Returns:
        Reconstructed weights
    """
    return lut[indices]

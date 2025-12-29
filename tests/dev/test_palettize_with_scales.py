#!/usr/bin/env python3
"""
Test palettization with scales - palettize only snapped weights.

Architecture:
1. snapped weights → palettize with mode="unique" → LUT compressed
2. scales → keep as FP16 constants (NOT palettized)
3. At runtime: effective = snapped * scales, then conv

SOLUTION FOUND:
===============
1. PyTorch: torch.jit.freeze(traced, preserved_attrs=[...]) successfully
   preserves snapped and scales as separate prim::GetAttr nodes with aten::mul.

2. CoreML Frontend Problem: Creates mul op AND evaluates it because:
   - mul(const, const) has .val populated via value_inference()
   - promote_input_dtypes() creates _promoted const when var.val is not None
   ROOT CAUSE: elementwise_binary.py:47-49 @precondition(allow=VALUE) decorator
   triggers value_inference() for all elementwise binary ops including mul.

3. SOLUTION: Monkey-patch elementwise_binary.value_inference to return None
   for large tensors (>100K elements). This prevents mul from evaluating
   const*const, so mul.val is None, and promote_input_dtypes uses cast instead
   of creating a new const.

4. WORKFLOW:
   a) Patch value_inference before conversion
   b) Convert with PassPipeline.EMPTY
   c) Restore original value_inference
   d) Apply palettization with mode="unique" to snapped weights only
   e) Result: constexpr_lut_to_dense with 4-entry LUT + FP16 scales

COMPRESSION RESULTS:
====================
- Snapped weights: 2-bit LUT (8x compression)
- Scales: FP16 (no compression)
- Overall: ~1.8x compression (snapped is ~50% of weights)

ALTERNATIVE: Direct MIL construction (test_lut_mil_direct.py)
- Use constexpr_lut_to_dense directly (no .val → can't be folded)
- Use mb.const + mb.matmul for scale_A @ scale_B
- More control but requires rewriting model structure
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import coremltools as ct
import coremltools.optimize.coreml as cto


class FFNWithScales(nn.Module):
    """FFN where snapped and scales are stored as flat buffers (no Conv2d submodules).

    This avoids the nested GetAttr issue with coremltools.

    Scales are stored as factored A @ B matrices for compression:
    - scale_A: [out_features, rank]
    - scale_B: [rank, in_features]
    - At runtime: scales = scale_A @ scale_B
    """

    def __init__(self, gate_snapped, gate_scale_A, gate_scale_B,
                 up_snapped, up_scale_A, up_scale_B,
                 down_snapped, down_scale_A, down_scale_B):
        super().__init__()
        # Store snapped weights as buffers (NOT inside Conv2d modules)
        # This avoids nested GetAttr that coremltools can't handle
        self.register_buffer('gate_snapped', gate_snapped.reshape(gate_snapped.shape[0], gate_snapped.shape[1], 1, 1))
        self.register_buffer('up_snapped', up_snapped.reshape(up_snapped.shape[0], up_snapped.shape[1], 1, 1))
        self.register_buffer('down_snapped', down_snapped.reshape(down_snapped.shape[0], down_snapped.shape[1], 1, 1))

        # Store factored scales as separate A and B matrices
        # scale_A: [out_features, rank], scale_B: [rank, in_features]
        self.register_buffer('gate_scale_A', gate_scale_A)
        self.register_buffer('gate_scale_B', gate_scale_B)
        self.register_buffer('up_scale_A', up_scale_A)
        self.register_buffer('up_scale_B', up_scale_B)
        self.register_buffer('down_scale_A', down_scale_A)
        self.register_buffer('down_scale_B', down_scale_B)

    def forward(self, x):
        # Gate: scales = A @ B, effective = snapped * scales, then conv
        gate_scales = (self.gate_scale_A @ self.gate_scale_B).unsqueeze(-1).unsqueeze(-1)
        gate_eff = self.gate_snapped * gate_scales
        gate_out = F.conv2d(x, gate_eff)

        # Up
        up_scales = (self.up_scale_A @ self.up_scale_B).unsqueeze(-1).unsqueeze(-1)
        up_eff = self.up_snapped * up_scales
        up_out = F.conv2d(x, up_eff)

        # SiLU and multiply
        gate_silu = F.silu(gate_out)
        mlp = gate_silu * up_out

        # Down
        down_scales = (self.down_scale_A @ self.down_scale_B).unsqueeze(-1).unsqueeze(-1)
        down_eff = self.down_snapped * down_scales
        return F.conv2d(mlp, down_eff)


def create_synthetic_data(scale_rank=32):
    """Create synthetic snapped weights (4 unique values) and factored scales."""
    print(f"Creating synthetic data (4 unique LUT values, scale_rank={scale_rank})...")

    # 4 LUT values for 2-bit quantization
    lut_values = torch.tensor([-1.0, -0.333984375, 0.333984375, 1.0], dtype=torch.float16)

    data = {}
    shapes = {
        'gate_proj': (3072, 1024),
        'up_proj': (3072, 1024),
        'down_proj': (1024, 3072),
    }

    for proj, shape in shapes.items():
        out_features, in_features = shape

        # Create snapped weights with exactly 4 unique values
        indices = torch.randint(0, 4, shape)
        snapped = lut_values[indices]

        # Create factored scales: scale_A [out, rank] @ scale_B [rank, in]
        scale_A = torch.randn(out_features, scale_rank, dtype=torch.float16) * 0.1
        scale_B = torch.randn(scale_rank, in_features, dtype=torch.float16) * 0.1

        # Compute full scales for display
        scales_full = scale_A @ scale_B

        data[proj] = {'snapped': snapped, 'scale_A': scale_A, 'scale_B': scale_B}
        print(f"  {proj}: snapped unique={len(torch.unique(snapped))}, "
              f"scale_A={list(scale_A.shape)}, scale_B={list(scale_B.shape)}, "
              f"scales range=[{scales_full.min():.4f}, {scales_full.max():.4f}]")

    return data


def load_layer_data(ckpt_path, layer_idx=0):
    """Load snapped weights and factored scales from checkpoint."""
    print(f"Loading layer {layer_idx}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    data = {}
    for proj in ['gate_proj', 'up_proj', 'down_proj']:
        prefix = f'model.layers.{layer_idx}.mlp.{proj}'
        w = ckpt[f'{prefix}.weight'].to(torch.float16)
        scale_A = ckpt[f'{prefix}.scale_A'].to(torch.float16)
        scale_B = ckpt[f'{prefix}.scale_B'].to(torch.float16)

        # Compute full scales for display only
        scales_full = (scale_A.float() @ scale_B.float()).to(torch.float16)

        data[proj] = {'snapped': w, 'scale_A': scale_A, 'scale_B': scale_B}
        print(f"  {proj}: snapped unique={len(torch.unique(w))}, "
              f"scale_A={list(scale_A.shape)}, scale_B={list(scale_B.shape)}, "
              f"scales range=[{scales_full.min():.4f}, {scales_full.max():.4f}]")

    return data


# =============================================================================
# COREML CONSTANT FOLDING BYPASS
# =============================================================================
#
# PROBLEM: CoreML's MIL frontend evaluates const*const during conversion
# LOCATION: coremltools/converters/mil/mil/ops/defs/iOS15/elementwise_binary.py:47-49
#
#   @precondition(allow=VALUE)
#   def value_inference(self):
#       return self._cast_check_value_inferene(self.x.val, self.y.val)
#
# This decorator causes value_inference() to run when both inputs have .val,
# which results in the mul op's output also having .val (the evaluated result).
#
# Then in _utils.py:443-445, promote_input_dtypes() sees var.val is not None
# and creates a NEW const: mb.const(val=evaluated_result, name=var.name + "_promoted")
#
# SOLUTION: Monkey-patch value_inference to return None for large tensors
# =============================================================================

def patch_coreml_value_inference(threshold=100000):
    """
    Patch elementwise_binary.value_inference to skip evaluation for large tensors.

    This prevents CoreML from folding const*const, keeping snapped and scales
    as separate operations that can be individually optimized.

    Args:
        threshold: Skip evaluation if tensor size > threshold (default 100K elements)

    Returns:
        Original value_inference function (for restoration)
    """
    from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_binary import elementwise_binary

    original = elementwise_binary.value_inference

    def patched_value_inference(self):
        try:
            # Get shapes safely (may be symbolic for internal ops)
            x_shape = self.x.shape if hasattr(self.x, 'shape') else ()
            y_shape = self.y.shape if hasattr(self.y, 'shape') else ()

            # Compute sizes, handling symbolic shapes gracefully
            try:
                x_size = int(np.prod([int(s) for s in x_shape]))
            except (TypeError, ValueError):
                x_size = 0  # Symbolic shape, allow evaluation

            try:
                y_size = int(np.prod([int(s) for s in y_shape]))
            except (TypeError, ValueError):
                y_size = 0  # Symbolic shape, allow evaluation

            # Skip evaluation for large concrete tensors
            if x_size > threshold or y_size > threshold:
                return None

        except Exception:
            pass  # Fall through to original on any error

        return original(self)

    elementwise_binary.value_inference = patched_value_inference
    print(f"[PATCH] Bypassing CoreML const folding for tensors > {threshold} elements")
    return original


def restore_coreml_value_inference(original):
    """Restore original value_inference after conversion."""
    from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_binary import elementwise_binary
    elementwise_binary.value_inference = original
    print("[PATCH] Restored original CoreML value_inference")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', default=None, help='QAT checkpoint (optional, uses synthetic data if not provided)')
    parser.add_argument('--output', '-o', default='/tmp/test_palettize_scales')
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--no-patch', action='store_true', help='Skip patching to show baseline behavior')
    args = parser.parse_args()

    # Load data from checkpoint or create synthetic
    if args.checkpoint:
        data = load_layer_data(args.checkpoint, args.layer)
    else:
        data = create_synthetic_data()

    # Create model with factored scales (A and B separate)
    model = FFNWithScales(
        data['gate_proj']['snapped'], data['gate_proj']['scale_A'], data['gate_proj']['scale_B'],
        data['up_proj']['snapped'], data['up_proj']['scale_A'], data['up_proj']['scale_B'],
        data['down_proj']['snapped'], data['down_proj']['scale_A'], data['down_proj']['scale_B'],
    )
    model.eval()
    model = model.half()

    # Test PyTorch inference first
    print("\nTesting PyTorch inference...")
    x_pt = torch.randn(1, 1024, 64, 1, dtype=torch.float16)
    with torch.no_grad():
        out_pt = model(x_pt)
    print(f"  PyTorch output: shape={out_pt.shape}, range=[{out_pt.min():.4f}, {out_pt.max():.4f}]")

    # ==========================================================================
    # STEP 1: TRACE MODEL (PyTorch constant folding is prevented by not freezing)
    # ==========================================================================
    print("\nTracing model...")
    traced = torch.jit.trace(model, x_pt, strict=False)

    # ==========================================================================
    # STEP 2: APPLY COREML VALUE_INFERENCE PATCH
    # ==========================================================================
    # This prevents CoreML from evaluating const*const during conversion
    if not args.no_patch:
        original_value_inference = patch_coreml_value_inference()
    else:
        print("\n[NO-PATCH] Skipping patch to show baseline behavior (const folding)")

    # ==========================================================================
    # STEP 3: CONVERT TO COREML WITH iOS18 + ANE SETTINGS
    # ==========================================================================
    # Use iOS18 target with FP16 precision and ANE compute units
    # This matches the AQ1 approach in test_aq1_ffn_trace_convert.py
    # to avoid unnecessary cast operations from dtype promotion
    print("\nConverting to CoreML with iOS18 + FP16 + ANE settings...")

    # Use DEFAULT pipeline but remove const_elimination to keep snapped * scales separate
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(["common::const_elimination"])

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, 1024, 64, 1), dtype=np.float16)],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        pass_pipeline=pipeline,
    )

    # Restore original value_inference (needed for palettization passes)
    if not args.no_patch:
        restore_coreml_value_inference(original_value_inference)

    # ==========================================================================
    # STEP 4: ANALYZE MIL STRUCTURE
    # ==========================================================================
    print("\n=== MIL Structure Analysis ===")
    mil_prog = mlmodel._mil_program

    snapped_ops = []
    scale_ops = []
    promoted_ops = []

    if mil_prog:
        for func in mil_prog.functions.values():
            for op in func.operations:
                if op.op_type == 'const':
                    try:
                        val = op.val.val if hasattr(op.val, 'val') else None
                        if val is not None and hasattr(val, 'size') and val.size > 1000:
                            n_unique = len(np.unique(val))
                            name = op.outputs[0].name if op.outputs else op.name

                            if 'snapped' in name:
                                snapped_ops.append((name, val.shape, n_unique))
                            elif 'scale' in name:
                                scale_ops.append((name, val.shape, n_unique))
                            elif '_promoted' in name or 'eff' in name:
                                promoted_ops.append((name, val.shape, n_unique))
                    except:
                        pass

    print(f"\nSnapped consts ({len(snapped_ops)}):")
    for name, shape, n_unique in snapped_ops:
        print(f"  {name}: {shape}, unique={n_unique}")

    print(f"\nScale consts ({len(scale_ops)}):")
    for name, shape, n_unique in scale_ops:
        print(f"  {name}: {shape}, unique={n_unique}")

    print(f"\nPromoted/eff consts ({len(promoted_ops)}):")
    for name, shape, n_unique in promoted_ops:
        print(f"  {name}: {shape}, unique={n_unique}")

    # Check success
    if len(promoted_ops) == 0 and len(snapped_ops) > 0:
        print("\n✓ SUCCESS: No const folding! Snapped and scales are separate.")
    elif len(promoted_ops) > 0:
        print(f"\n✗ FOLDED: {len(promoted_ops)} promoted consts (mul was evaluated)")

    # Save FP16 model and test inference
    fp16_path = f"{args.output}_fp16.mlpackage"
    mlmodel.save(fp16_path)
    print(f"\nSaved FP16: {fp16_path}")

    x_np = x_pt.numpy()
    out_fp16 = mlmodel.predict({'x': x_np})['output']
    print(f"  CoreML FP16 output: range=[{out_fp16.min():.4f}, {out_fp16.max():.4f}]")

    # ==========================================================================
    # STEP 5: APPLY SELECTIVE PALETTIZATION (WITHOUT CONST ELIMINATION)
    # ==========================================================================
    # The standard palettize_weights() runs _mil_convert internally which uses
    # the DEFAULT pipeline including const_elimination - this folds our A @ B!
    #
    # Solution: Use _apply_graph_pass with return_pymil_prog=True to get the
    # palettized program, then convert manually with our custom pipeline.
    print("\n=== Applying Selective Palettization (preserving A @ B) ===")
    snapped_op_names = [name for name, _, _ in snapped_ops]

    if snapped_op_names:
        print(f"Palettizing {len(snapped_op_names)} snapped weight ops with mode='unique'...")

        # Create config to palettize only snapped weights
        op_name_configs = {}
        for op_name in snapped_op_names:
            op_name_configs[op_name] = cto.OpPalettizerConfig(
                mode="unique",
                granularity="per_tensor"
            )

        config = cto.OptimizationConfig(
            global_config=None,  # Don't palettize by default
            op_name_configs=op_name_configs
        )

        # Apply palettization pass but return pymil program (before _mil_convert)
        from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
        from coremltools.converters.mil.mil.passes.graph_pass import PassOption
        from coremltools.models.utils import _apply_graph_pass
        from coremltools.converters.mil.converter import mil_convert as _mil_convert

        weight_palettizer = PASS_REGISTRY["compression::palettize_weights"]
        weight_palettizer.set_options([PassOption("config", config), PassOption("joint_compression", False)])

        # Get palettized pymil program (before _mil_convert runs const_elimination)
        palettized_prog = _apply_graph_pass(mlmodel, weight_palettizer, return_pymil_prog=True)

        # Convert to mlmodel with our custom pipeline (NO const_elimination)
        pipeline = ct.PassPipeline.DEFAULT
        pipeline.remove_passes(["common::const_elimination"])

        spec = mlmodel.get_spec()
        palettized = _mil_convert(
            palettized_prog,
            convert_to="mlprogram",
            convert_from="milinternal",
            specification_version=spec.specificationVersion,
            compute_units=mlmodel.compute_unit,
            model_description=spec.description,
            skip_model_load=False,
            pass_pipeline=pipeline,
        )

        # Check LUT ops
        print("\nAnalyzing LUT ops after palettization...")
        mil_prog = palettized._mil_program
        lut_count = 0
        if mil_prog:
            for func in mil_prog.functions.values():
                for op in func.operations:
                    if op.op_type == 'constexpr_lut_to_dense':
                        lut_count += 1
                        lut_size = op.lut.val.shape[-1] if hasattr(op.lut, 'val') else '?'
                        print(f"  LUT: {op.name}, {lut_size} entries")
        print(f"Total LUT ops: {lut_count}")

        # Save LUT model
        lut_path = f"{args.output}_lut.mlpackage"
        palettized.save(lut_path)
        print(f"\nSaved LUT: {lut_path}")

        # Test LUT inference
        out_lut = palettized.predict({'x': x_np})['output']
        print(f"  CoreML LUT output: range=[{out_lut.min():.4f}, {out_lut.max():.4f}]")

        # Compare outputs
        diff = np.abs(out_fp16 - out_lut).max()
        print(f"\n  FP16 vs LUT max diff: {diff:.6f}")

        # Check file sizes for compression ratio
        fp16_weight = os.path.join(fp16_path, "Data/com.apple.CoreML/weights/weight.bin")
        lut_weight = os.path.join(lut_path, "Data/com.apple.CoreML/weights/weight.bin")
        if os.path.exists(fp16_weight) and os.path.exists(lut_weight):
            fp16_size = os.path.getsize(fp16_weight)
            lut_size = os.path.getsize(lut_weight)
            ratio = fp16_size / lut_size if lut_size > 0 else 0

            print(f"\n=== Compression Results ===")
            print(f"  FP16: {fp16_size/1024/1024:.2f} MB")
            print(f"  LUT:  {lut_size/1024/1024:.2f} MB")
            print(f"  Ratio: {ratio:.2f}x")

            if lut_count == len(snapped_op_names):
                print(f"\n✓ SUCCESS: All {lut_count} snapped weights converted to LUT!")
            else:
                print(f"\n⚠ PARTIAL: {lut_count}/{len(snapped_op_names)} ops converted to LUT")
    else:
        print("  No snapped weight candidates found - MIL optimizer may have folded them")
        print("  Try running with --no-patch to see the baseline behavior")

    print("\nDone!")


if __name__ == "__main__":
    main()

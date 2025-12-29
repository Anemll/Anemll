# AQ1 Constant Folding Issue - Complete Reference

## Overview

This document captures all key findings, problems, solutions, and code references for the constant folding issue encountered when converting QAT (Quantization-Aware Training) models with `snapped * scales` architecture to CoreML.

---

## Problem Statement

When converting a PyTorch model with the architecture:
```python
effective_weight = snapped_weight * scales  # snapped has 4 unique values
output = F.conv2d(x, effective_weight)
```

The multiplication is **evaluated at conversion time**, producing a single constant with millions of unique values. This prevents:
- `mode="unique"` palettization from detecting the 4-value pattern
- Selective compression of snapped weights (2-bit LUT)
- Storage optimization (should be ~8x compression on snapped)

---

## Root Cause Analysis

### Location 1: PyTorch JIT Constant Folding

**File**: PyTorch internal (`torch.jit.freeze`)

**Behavior**:
- `torch.jit.freeze()` inlines module buffers as `prim::Constant` tensors
- Constant propagation then collapses `aten::mul(const, const)` into single constant

**Solution**: Don't use freeze, or use `preserved_attrs`:
```python
frozen = torch.jit.freeze(traced, preserved_attrs=["snapped", "scales"])
```

**Result**: PyTorch graph shows `prim::GetAttr` + `aten::mul` instead of folded `prim::Constant`

---

### Location 2: CoreML MIL Value Inference (PRIMARY ISSUE)

**File**: `coremltools/converters/mil/mil/ops/defs/iOS15/elementwise_binary.py:47-49`

```python
class elementwise_binary(Operation):
    ...
    @precondition(allow=VALUE)
    def value_inference(self):
        return self._cast_check_value_inferene(self.x.val, self.y.val)
```

**Mechanism**:
1. `@precondition(allow=VALUE)` decorator checks if all inputs have `.val`
2. When both `x.val` and `y.val` are not None, `value_inference()` runs
3. Returns `x.val * y.val` as the mul output's `.val`
4. The mul op now has a concrete evaluated value

**Why This Matters**:
- Even with `PassPipeline.EMPTY`, this evaluation happens during frontend conversion
- It's not a MIL pass - it's part of op construction itself

---

### Location 3: CoreML Promote Input Dtypes

**File**: `coremltools/converters/mil/mil/ops/defs/_utils.py:439-446`

```python
def _promoted_var(var, promoted_dtype):
    if var.val is None:
        x = mb.cast(x=var, dtype=..., name=var.name + "_promoted")
    else:
        # THIS IS THE PROBLEM - creates NEW const with evaluated value
        const_value_after_cast = cast_op_class.get_cast_value(var, ...)
        x = mb.const(val=const_value_after_cast, name=var.name + "_promoted")
    return x
```

**Mechanism**:
1. When conv needs the mul output as weight, it calls `promote_input_dtypes()`
2. Since `mul.val is not None` (from value_inference), branch takes `else` path
3. Creates a NEW constant `eff_promoted` with the evaluated multiplication result
4. Conv uses `eff_promoted` instead of the mul op output

**Result**: Even though mul op exists, conv uses a folded constant

---

### Location 4: Precondition Decorator

**File**: `coremltools/converters/mil/mil/operation.py:45-80`

```python
def precondition(allow=ALL):
    """
    VALUE: value that can be materialized during compile time
    SYMBOL: value that cannot be materialized but exists as a symbol
    NONE: a None value
    """
    ALLOW_VALUE = allow & VALUE
    ...
    def decorator(func):
        def wrapper(self):
            HAS_VALUE = False
            for in_name, in_type in self._input_types.items():
                ...
                HAS_VALUE, HAS_SYMBOL, HAS_NONE = process(v, ...)

            # Only run value_inference if precondition is met
            if ALLOW_VALUE and HAS_VALUE:
                return func(self)
            return None
```

**Key Insight**: The decorator controls when `value_inference()` is called. For `mul`, it only runs when both inputs have concrete values.

---

## Solution: Monkey-Patch value_inference

### The Patch

```python
def patch_coreml_value_inference(threshold=100000):
    """
    Patch elementwise_binary.value_inference to skip evaluation for large tensors.
    """
    from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_binary import elementwise_binary

    original = elementwise_binary.value_inference

    def patched_value_inference(self):
        try:
            x_shape = self.x.shape if hasattr(self.x, 'shape') else ()
            y_shape = self.y.shape if hasattr(self.y, 'shape') else ()

            # Handle symbolic shapes gracefully
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
    return original


def restore_coreml_value_inference(original):
    """Restore original value_inference after conversion."""
    from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_binary import elementwise_binary
    elementwise_binary.value_inference = original
```

### Why It Works

1. When `value_inference()` returns `None`, the mul output has `.val = None`
2. In `_promoted_var()`, the `if var.val is None` branch is taken
3. This creates `mb.cast()` instead of `mb.const()` - maintains connection to mul
4. The snapped and scales remain as separate const ops
5. Palettization can now detect the 4-unique-value pattern in snapped

---

## Complete Workflow

```python
import numpy as np
import torch
import coremltools as ct
import coremltools.optimize.coreml as cto

# 1. Create/load model with snapped * scales architecture
model = FFNWithScales(snapped, scales)
model.eval()

# 2. Trace (don't freeze to avoid PyTorch folding)
x = torch.randn(1, 1024, 64, 1, dtype=torch.float16)
traced = torch.jit.trace(model, x)

# 3. Apply patch BEFORE conversion
original = patch_coreml_value_inference(threshold=100000)

# 4. Convert with iOS18 + FP16 + ANE settings (eliminates cast ops)
# Use DEFAULT pipeline but remove const_elimination
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

# 5. Restore original (needed for palettization passes)
restore_coreml_value_inference(original)

# 6. Apply selective palettization to snapped weights only
config = cto.OptimizationConfig(
    global_config=None,
    op_name_configs={
        "gate_snapped": cto.OpPalettizerConfig(mode="unique", granularity="per_tensor"),
        "up_snapped": cto.OpPalettizerConfig(mode="unique", granularity="per_tensor"),
        "down_snapped": cto.OpPalettizerConfig(mode="unique", granularity="per_tensor"),
    }
)
palettized = cto.palettize_weights(mlmodel, config)
```

---

## Results Comparison

| Metric | Without Patch | With Patch |
|--------|---------------|------------|
| Snapped consts | 3 (4 unique) | 3 (4 unique) |
| Scale consts | 3 (~1200 unique) | 3 (~1200 unique) |
| Promoted/eff consts | 3 (4337 unique) | **0** |
| LUT ops | 0 | **3 (4 entries)** |
| Model size | 72→36 MB | **36→20.25 MB** |
| Compression | 2x (MIL only) | **1.78x (real LUT)** |

---

## Alternative: Direct MIL Construction

For maximum control, build the model directly in MIL:

```python
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

@mb.program(
    input_specs=[mb.TensorSpec(shape=(1, 1024, 64, 1), dtype=types.fp16)],
    opset_version=ct.target.iOS17
)
def model(x):
    # constexpr_lut_to_dense has .val = None (evaluated at load time)
    snapped = mb.constexpr_lut_to_dense(
        indices=packed_indices,  # 2-bit packed uint8
        lut=lut_values,          # 4 FP16 values
        shape=weight_shape,
        name="snapped"
    )

    # Factored scales: scale_A @ scale_B computed at runtime
    scale_A = mb.const(val=scale_a_data, name="scale_A")
    scale_B = mb.const(val=scale_b_data, name="scale_B")
    scales = mb.matmul(x=scale_A, y=scale_B, name="scales")

    # Effective weight (not folded because snapped.val is None)
    effective = mb.mul(x=snapped, y=scales, name="effective")

    return mb.conv(x=x, weight=effective, name="output")
```

**Advantages**:
- `constexpr_lut_to_dense` inherently has `.val = None`
- No monkey-patching required
- Full control over model structure

**Disadvantages**:
- Requires rewriting model in MIL
- More complex for full model conversion

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `tests/dev/test_palettize_with_scales.py` | Complete working solution with patch |
| `tests/dev/test_skip_mul_palettize.py` | Simplified test with patch |
| `tests/dev/test_lut_mil_direct.py` | Direct MIL construction approach |
| `tests/dev/test_skip_mul_eval.py` | Initial patch testing |
| `tests/dev/COREML_CONST_FOLDING_BYPASS.md` | Summary documentation |

---

## CoreMLTools Source References

| File | Lines | Description |
|------|-------|-------------|
| `mil/ops/defs/iOS15/elementwise_binary.py` | 47-49 | `value_inference` with `@precondition` |
| `mil/ops/defs/iOS15/elementwise_binary.py` | 489-513 | `mul` class definition |
| `mil/ops/defs/_utils.py` | 439-446 | `_promoted_var` function |
| `mil/ops/defs/_utils.py` | 430-458 | `promote_input_dtypes` function |
| `mil/operation.py` | 45-80 | `precondition` decorator |
| `mil/operation.py` | 265, 382 | `type_value_inference`, `_auto_val` |

---

## Debugging Tips

### Check if folding occurred
```python
mil_prog = mlmodel._mil_program
for func in mil_prog.functions.values():
    for op in func.operations:
        if op.op_type == 'const':
            val = op.val.val
            if val is not None and val.size > 1000:
                n_unique = len(np.unique(val))
                print(f"{op.name}: unique={n_unique}")
                # If n_unique > 16, const was folded
```

### Check LUT ops created
```python
for func in mil_prog.functions.values():
    for op in func.operations:
        if op.op_type == 'constexpr_lut_to_dense':
            print(f"LUT: {op.name}, entries={op.lut.val.shape[-1]}")
```

### Trace patch execution
Add print statements to patched function:
```python
if x_size > threshold or y_size > threshold:
    print(f"[PATCH] Skipping value_inference for {self.name}")
    return None
```

---

## Correct Approach: Keep const_elimination, Use Size-Aware Patch

**IMPORTANT:** Do NOT remove `const_elimination` - it's needed for epsilon scalars in layer_norm!

The correct approach uses a **size-aware patch** that selectively blocks large tensor folding while allowing small scalars to become constants:

```python
# Apply size-aware patch BEFORE conversion
original = _patch_value_inference(threshold=100000)  # patches mul AND matmul

try:
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(...), dtype=np.float16)],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,      # All ops in FP16
        compute_units=ct.ComputeUnit.CPU_AND_NE,     # Enable ANE
        minimum_deployment_target=ct.target.iOS18,   # iOS18 for best FP16 support
        # NOTE: Keep DEFAULT pipeline - const_elimination needed for epsilon!
    )
finally:
    _restore_value_inference(original)
```

**How the size-aware patch works:**

| Tensor Size | value_inference returns | Effect |
|-------------|-------------------------|--------|
| Small (< 10K elements) | Normal result | Epsilon → const ✓ layer_norm works |
| Large (> 100K elements) | None | matmul(A,B) → stays as op ✓ |

**WRONG approaches (cause "epsilon must be const" error):**
```python
# ❌ WRONG - breaks layer_norm
pipeline = ct.PassPipeline.EMPTY

# ❌ WRONG - breaks layer_norm
pipeline = ct.PassPipeline.DEFAULT
pipeline.remove_passes(["common::const_elimination"])
```

**Key settings:**
- `compute_precision=ct.precision.FLOAT16` - All operations in FP16, no dtype promotion needed
- `compute_units=ct.ComputeUnit.CPU_AND_NE` - Enable ANE execution
- `minimum_deployment_target=ct.target.iOS18` - Latest FP16/ANE features
- **Keep DEFAULT pipeline** - `const_elimination` is needed for scalar constants

This approach is used in `qwen_converter.py` for AQ1 quantization.

---

## Preserving Factored Scales (A @ B) During Palettization

### The Problem

The standard `cto.palettize_weights()` function destroys the factored scale structure:

```python
# What we want to preserve:
scale_A: [out_features, rank]     # e.g., [3072, 32]
scale_B: [rank, in_features]      # e.g., [32, 1024]
matmul(scale_A, scale_B)          # Computed at runtime

# But palettize_weights() folds this into:
scales: [out_features, in_features]  # Pre-computed constant [3072, 1024]
```

**Root cause**: `palettize_weights()` internally calls `_mil_convert()` which runs the DEFAULT pipeline including `const_elimination`. This folds `matmul(const_A, const_B)` into a single constant.

### The Solution

Use `_apply_graph_pass()` with `return_pymil_prog=True` to get the palettized program BEFORE `_mil_convert` runs, then convert manually with a custom pipeline:

```python
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.mil.mil.passes.graph_pass import PassOption
from coremltools.models.utils import _apply_graph_pass
from coremltools.converters.mil.converter import mil_convert as _mil_convert

# Configure palettization
config = cto.OptimizationConfig(
    global_config=None,
    op_name_configs={
        "snapped_op_name": cto.OpPalettizerConfig(mode="unique", granularity="per_tensor")
    }
)

# Get the palettization pass
weight_palettizer = PASS_REGISTRY["compression::palettize_weights"]
weight_palettizer.set_options([
    PassOption("config", config),
    PassOption("joint_compression", False)
])

# Apply pass but return pymil program (BEFORE _mil_convert)
palettized_prog = _apply_graph_pass(mlmodel, weight_palettizer, return_pymil_prog=True)

# Convert with custom pipeline (NO const_elimination)
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
```

### Results

| Model | matmul | constexpr_lut_to_dense | Size | Compression |
|-------|--------|------------------------|------|-------------|
| FP16 (factored) | 3 | 0 | 18.75 MB | - |
| LUT (standard palettize) | 0 | 3 | 20.25 MB | 0.93x (worse!) |
| LUT (preserved A@B) | 3 | 3 | 3.00 MB | **6.25x** |

The preserved A @ B approach achieves **6.25x compression** by combining:
- **Snapped weights**: 2-bit LUT via `constexpr_lut_to_dense` (8x compression)
- **Factored scales**: `matmul(A, B)` at runtime (24x compression on scales)

---

## Known Limitations

1. **Conv with dynamic weights CAN run on ANE**: When using `constexpr_lut_to_dense + matmul` for weights, ANE CAN execute the conv. Pre-baked weights also work on ANE.

2. **Palettization internal pipeline**: The standard `palettize_weights()` runs `const_elimination` internally. Use the `_apply_graph_pass` + `_mil_convert` approach to avoid this.

3. **Threshold sensitivity**: Set threshold appropriately - too low skips small tensors, too high allows large tensor folding.

---

## Version Info

- **coremltools**: 8.2+
- **PyTorch**: 2.0+
- **Python**: 3.9
- **Target**: iOS17+

---

## Related Issues

- PyTorch JIT constant propagation
- CoreML skip_const_by_size (only affects MIL passes, not frontend)
- ANE static weight requirement for conv

---

## Related Documentation

For MIL Builder graph patterns, ANE data type requirements, KV cache state operations, and troubleshooting workflow, see:
- **[MB_GRAPH_ISSUES.md](MB_GRAPH_ISSUES.md)** - MIL Builder patterns and ANE compatibility

---

## Version Info

- **coremltools**: 8.2+
- **PyTorch**: 2.0+
- **Python**: 3.9
- **Target**: iOS18+

---

*Document created: 2025-12-28*
*Last updated: 2025-12-29*

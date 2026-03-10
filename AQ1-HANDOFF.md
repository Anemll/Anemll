# AQ1 (ANEMLL-QUANT-1) Handoff Document

## Overview

AQ1 is the first quantization scheme for ANEMLL, designed to run quantized LLMs on Apple Neural Engine (ANE) with CoreML. It uses **LUT-based weight quantization** with **low-rank factored scales** to achieve high compression while maintaining model quality.

**Core formula:**
```
effective_weight = LUT[indices] * (scale_A @ scale_B)
```

Where:
- `LUT[indices]` holds 2-bit or 4-bit quantized base weights (e.g., 4 values: `[-1, -0.33, 0.33, 1]`)
- `scale_A @ scale_B` is a low-rank factored scale matrix computed at runtime
- The final weight is used in `Conv2d` (ANE-friendly) operations

**Factored V2 forward pass:**
```
y = sum_k( g_k * (a_k . (Q @ (b_k . x))) )
```

---

## File Inventory

### Core Library (`anemll/`)

| File | Purpose |
|------|---------|
| `anemll/models/anemll_quant.py` | Reusable quantization module: `AnemllConv2d` custom layer, `torch.ops.anemll_quant.quant_conv` custom op, weight loading, LUT index computation, CoreML converter registration |
| `anemll/models/aq1_mil_layer.py` | MIL layer builder: `build_aq1_conv_layer()`, `build_factored_aq1_conv_layer()`, checkpoint loading (`load_aq1_checkpoint()`), LUT+index extraction, storage size computation |
| `anemll/models/qwenAQ1_model.py` | Qwen3 model implementation with AQ1 quantization (extends `qwen_model.py` with `AnemllConv2d` layers) |
| `anemll/models/__init__.py` | Exports `QwenAQ1Config`, `QwenAQ1ForCausalLM`, `QwenAQ1Model`, and all `anemll_quant` components |
| `anemll/ane_converter/aq1_mil_converter.py` | MIL-based CoreML converter: builds MLP blocks, attention projections, RMSNorm using MIL Builder directly (not PyTorch trace). Preserves `constexpr_lut_to_dense + matmul(A,B)` |
| `anemll/ane_converter/qwen_converter.py` | Full Qwen converter with AQ1 integration: `--part aq1_full`, `--part aq1_mlp`, `--part aq1_ffn` modes. Handles KV cache, attention, chunking |

### Test & Development (`tests/dev/`)

#### Conversion Scripts

| File | Purpose |
|------|---------|
| `convert_aq1_ffn_part2.py` | Convert FFN layers with AQ1 to CoreML |
| `convert_dynamic_scales.py` | Dynamic scale conversion utilities |
| `export_q4_baked.py` | Export baked (pre-computed) weights for fast inference |

#### Inference & Validation

| File | Purpose |
|------|---------|
| `test_qwenAQ1_load.py` | PyTorch inference test for AQ1 checkpoints |
| `test_qwenAQ1_load_rf.py` | V2 factored PyTorch inference (recommended) |
| `test_aq1_inference.py` | Standalone CoreML inference test with optional ANE comparison |
| `test_qwen_aq1_pytorch.py` | PyTorch-only baseline inference |
| `test_qwen_aq1_compare.py` | Interactive PyTorch vs CoreML comparison with divergence metrics |

#### Divergence & Stability Testing

| File | Purpose |
|------|---------|
| `qwen_aq1_divergence_harness.py` | Batch dataset generation for instability analysis |
| `analyze_runs.py` | Post-process harness outputs into train/val datasets |

#### MIL & Converter Tests

| File | Purpose |
|------|---------|
| `test_aq1_converter.py` | MIL conversion test |
| `test_aq1_dynamic_convert.py` | Dynamic conversion test |
| `test_aq1_ffn_converter.py` | FFN AQ1 conversion test |
| `test_aq1_ffn_trace_convert.py` | FFN trace-based conversion |
| `test_ffn_aq1_conversion.py` | FFN conversion validation |
| `test_qwen_a1f_layer.py` | PyTorch trace conversion with size-aware patch |
| `test_factored_conv_workflow.py` | Factored conv workflow test |
| `test_anemll_conv_constexpr.py` | Conv constexpr validation |
| `test_palettize_with_scales.py` | Palettization with factored scales |

### Documentation (`tests/dev/`)

| File | Purpose |
|------|---------|
| `QUANTIZED_HOW_TO_CONVERT.md` | Comprehensive conversion guide: 4 methods (PyTorch inference, baked, MIL graph, V2 factored) |
| `A1F_FOLDING_ISSUE.md` | Detailed analysis of CoreML constant folding problem and the size-aware `value_inference` patch |
| `DIVERGENCE_HARNESS.md` | Divergence detection and stability testing documentation |

---

## Architecture & Key Concepts

### 1. Weight Representation

AQ1 stores weights in three components:
- **LUT indices** (2-bit or 4-bit packed): Map each weight to a LUT entry
- **LUT values** (float16): Small table of quantization levels (4 or 16 entries)
- **Scale factors** (float16): Low-rank `scale_A [out, rank] @ scale_B [rank, in]`

Storage savings example (3072x1024 layer, 2-bit, rank=32):
- Baked FP16: 6,291,456 bytes
- AQ1: ~918,536 bytes (~6.8x compression)

### 2. Two Conversion Approaches

#### A. Baked Weights (Fast Inference, Larger Model)
Pre-computes `LUT[indices] * (scale_A @ scale_B)` at conversion time. Result is a standard FP16 constant. ~50 t/s inference.

#### B. Dynamic Scales (Smaller Model, Requires Careful Conversion)
Preserves factored structure in CoreML:
- `constexpr_lut_to_dense` for base weights (compressed storage)
- `matmul(A, B)` computed at runtime
- Requires **constant folding prevention** (see below)

### 3. Constant Folding Prevention (Critical)

CoreML's `value_inference` eagerly folds `const * const` during conversion, destroying the quantized structure. The solution is a **size-aware monkey patch**:

```python
# Patch BEFORE ct.convert()
original = _patch_value_inference(threshold=100000)
mlmodel = ct.convert(traced, ...)  # Keep DEFAULT pipeline
_restore_value_inference(original)
```

- Small tensors (< 10K elements): evaluated normally (needed for epsilon, layer_norm)
- Large tensors (> 100K elements): `value_inference` returns `None`, preventing folding

**Do NOT use `PassPipeline.EMPTY`** — it breaks `layer_norm` epsilon constants.

### 4. ANE Compatibility Requirements

- All dense layers must be `Conv2d` with kernel_size=1
- Weights must be 4D: `[out, in, 1, 1]`
- Spatial dimensions > 1 (use seq_len >= 2)
- Static weights required for ANE execution
- FP16 for both input and weights
- RMSNorm must subtract mean first (ANE-specific)

### 5. Custom Op Approach (`anemll_quant.py`)

For PyTorch trace-based conversion:
1. `torch.ops.anemll_quant.quant_conv` — custom op that survives `torch.jit.trace`
2. Registered coremltools converter emits `constexpr_lut_to_dense + matmul`
3. Enables standard trace → convert workflow

### 6. MIL Builder Approach (`aq1_mil_layer.py`)

For direct MIL construction (no PyTorch trace):
1. `build_aq1_conv_layer()` — standard approach: `W_base * (A @ B) → conv`
2. `build_factored_aq1_conv_layer()` — avoids materializing full scale matrix: `y = sum_k(A[k] * conv(Q, B[k] * x))`
3. Used by `aq1_mil_converter.py` and `qwen_converter.py --part aq1_full`

---

## Checkpoint Format

Quantized checkpoints contain:

```python
{
    # Standard (unquantized) weights
    'model.embed_tokens.weight': tensor,
    'lm_head.weight': tensor,
    'model.norm.weight': tensor,

    # Per-layer quantized weights
    'model.layers.{i}.mlp.gate_proj.weight': tensor,    # Snapped LUT values [out, in]
    'model.layers.{i}.mlp.gate_proj.scale_A': tensor,   # [out, rank]
    'model.layers.{i}.mlp.gate_proj.scale_B': tensor,   # [rank, in] or [rank, groups]
    'model.layers.{i}.mlp.gate_proj.lut': tensor,       # (optional) LUT values
    # ... gate_proj, up_proj, down_proj similarly

    # Attention (if quantized)
    'model.layers.{i}.self_attn.q_proj.weight': tensor,
    'model.layers.{i}.self_attn.q_proj.scale_A': tensor,
    'model.layers.{i}.self_attn.q_proj.scale_B': tensor,
    # ... k_proj, v_proj, o_proj similarly

    # LayerNorm
    'model.layers.{i}.input_layernorm.weight': tensor,
    'model.layers.{i}.post_attention_layernorm.weight': tensor,
}
```

Auto-config via `config.json` in checkpoint directory:
```json
{
    "model_id": "Qwen/Qwen3-0.6B",
    "snapped_mode": "lut",
    "lut_bits": 4,
    "attn_lut_bits": 4,
    "scale_rank": 4,
    "attn_scale_rank": 4
}
```

---

## Quick Start Commands

### PyTorch Inference Test (No Conversion)

```bash
# V2 factored inference (recommended)
python tests/dev/test_qwenAQ1_load_rf.py \
    --checkpoint /path/to/snapped/model_state_dict.pt \
    --interactive

# Basic inference test
python tests/dev/test_qwenAQ1_load.py \
    --checkpoint /path/to/model_state_dict.pt
```

### Baked Weights Conversion (Fast, Simple)

```bash
python tests/dev/export_baked_model.py \
    --checkpoint /path/to/model_state_dict.pt \
    --model Qwen/Qwen3-0.6B \
    --context 512 \
    --output /path/to/output
```

### MIL Graph Conversion (Full Control)

```bash
python -m anemll.ane_converter.qwen_converter \
    --model Qwen/Qwen3-0.6B \
    --aq1-checkpoint /path/to/checkpoint \
    --output /tmp/qwen_aq1 \
    --context 512 \
    --num-layers 8 \
    --part aq1_full
```

### PyTorch Trace Conversion (Dynamic Scales)

```bash
ANEMLL_DYNAMIC_SCALES=1 ANEMLL_SKIP_ATTN_LUT=1 \
python tests/dev/test_qwen_a1f_layer.py \
    --checkpoint /path/to/model_state_dict.pt \
    --model Qwen/Qwen3-0.6B \
    --layers 8 --context 256 \
    --output /tmp/qwen_aq1.mlpackage
```

### Divergence Testing

```bash
# Single prompt comparison (PyTorch vs CoreML)
python tests/dev/test_qwen_aq1_compare.py \
    /path/to/checkpoint.pt \
    /path/to/coreml_model \
    --prompt "What is AI?" --max-tokens 100 --driver coreml --no-think

# Batch stability harness
python tests/dev/qwen_aq1_divergence_harness.py \
    /path/to/checkpoint.pt /path/to/coreml_model \
    --dataset tests/dev/prompts.jsonl --out-dir runs/exp1 --no-think
```

---

## Known Issues & Gotchas

1. **Constant folding**: Must patch `value_inference` before `ct.convert()` for dynamic scales. See `A1F_FOLDING_ISSUE.md`.

2. **Qwen3 thinking mode**: Always use `enable_thinking=False` in `apply_chat_template()` for testing consistency. Do NOT use `/no_think` prefix.

3. **Conv2d static weights**: ANE rejects conv with dynamically-computed weights in some configurations. Pre-bake when possible for guaranteed ANE execution.

4. **Palettization pipeline**: Standard `cto.palettize_weights()` internally runs `const_elimination`, destroying factored scales. Use `_apply_graph_pass()` + `_mil_convert()` workaround (documented in `A1F_FOLDING_ISSUE.md`).

5. **BFloat16 handling**: Checkpoints may contain BF16 tensors. Always convert via `.float().numpy()` before processing.

6. **iOS18 requirement**: Sub-byte `constexpr_lut_to_dense` (2-bit/4-bit packed indices) requires `minimum_deployment_target=ct.target.iOS18`.

---

## Performance Summary

| Mode | Speed | Model Size | Notes |
|------|-------|------------|-------|
| Baked FP16 | ~50 t/s | Larger (FP16) | Pre-computed scales |
| Dynamic scales | ~6 t/s | Smaller (LUT+scales) | Runtime matmul |
| MIL with KV state | ~40-50 t/s | Smaller | Stateful decode |

---

## Dependencies

- Python 3.9 (strictly required)
- coremltools >= 8.2
- transformers >= 4.36.0
- torch >= 2.0
- numpy, safetensors
- macOS with Apple Neural Engine
- Xcode Command Line Tools

---

## Model Support

Currently tested with:
- Qwen3-0.6B (primary development target)
- Qwen3-8B (scaling validation)

Checkpoint variants tested: Q2_4 (2-bit LUT, rank-4), Q4_4 (4-bit LUT, rank-4), Q4_R32 (4-bit LUT, rank-32)

---

*Generated from branch `anemll-quant-1` — 2026-03-10*

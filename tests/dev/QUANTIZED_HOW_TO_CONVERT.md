# ANEMLL Quantized Model Conversion Guide

This document describes different methods for converting quantized models to CoreML format for Apple Neural Engine inference.

## Table of Contents

1. [Baked Weights Conversion](#1-baked-weights-conversion-quick-test) - Fast inference, pre-computed scales
2. [MIL Graph Conversion](#2-mil-graph-conversion) - Full control, stateful KV cache
3. [PyTorch Trace-Based Conversion](#3-pytorch-trace-based-conversion) - Dynamic scales with LUT compression

---

## 1. Baked Weights Conversion (Quick Test)

**Use case:** Quick quality testing of quantized checkpoints with maximum inference speed.

Baked quantization pre-computes the effective weights at conversion time by folding `W * (A @ B)` into a single constant. This results in ~8x faster inference compared to dynamic scales.

### When to Use Baked Mode

- Testing quantized checkpoint quality quickly
- Production deployment where model size is not a concern
- Maximum inference speed is required

### Limitations

- Larger model size (weights are expanded to FP16)
- No runtime scale adjustment possible
- Requires pre-quantized checkpoint with LUT indices and scales

### Prerequisites

1. A quantized checkpoint containing:
   - `model_state_dict.pt` - LUT indices and scale factors
   - 4-bit LUT quantization for FFN/Attention
   - Rank-4 factored scales (A, B matrices)

2. HuggingFace model downloaded:
   ```bash
   python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen3-0.6B')"
   ```

### Quick Start

```bash
python tests/dev/export_baked_model.py \
    --checkpoint /path/to/model_state_dict.pt \
    --model Qwen/Qwen3-0.6B \
    --context 512 \
    --output /path/to/output
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | required | Path to quantized checkpoint (.pt file) |
| `--model` | required | HuggingFace model ID (e.g., Qwen/Qwen3-0.6B) |
| `--context` | 512 | Context length |
| `--output` | required | Output directory for CoreML models |
| `--lut-lmhead` | 6 | LUT bits for LM head (4 or 6) |
| `--batch` | 64 | Batch size for prefill |
| `--prefix` | qwen | Model prefix for output files |
| `--skip-compile` | - | Skip CoreML compilation step |
| `--skip-test` | - | Skip inference test |

### Examples

```bash
# Q4_4 baked model (4-bit LUT, rank-4 scales)
python tests/dev/export_baked_model.py \
    --checkpoint /Users/anemll/Downloads/anemll_q4_a4_e2e_v2_scales_only/snapped/model_state_dict.pt \
    --model Qwen/Qwen3-0.6B \
    --context 512 \
    --output /Users/anemll/Models/ANE/AQ1_Q44_baked

# Q2_4 baked model (2-bit LUT, rank-4 scales)
python tests/dev/export_baked_model.py \
    --checkpoint /Users/anemll/Downloads/q2_pt_good1/snapped_lut/model_state_dict.pt \
    --model Qwen/Qwen3-0.6B \
    --context 512 \
    --output /Users/anemll/Models/ANE/AQ1_Q24_baked
```

### Pipeline Steps

The export script performs these steps automatically:

1. **Export Embeddings** - FP16, no quantization
2. **Export LM Head** - 6-bit LUT quantization (configurable)
3. **Export FFN (baked)** - 4-bit LUT with pre-computed scales
4. **Export Prefill** - Standard prefill attention
5. **Combine** - Merge FFN + Prefill into single model
6. **Compile** - CoreML compilation to .mlmodelc
7. **Create meta.yaml** - Configuration for chat.py
8. **Copy Tokenizer** - Tokenizer files from HuggingFace
9. **Test** - Verify inference works

### Output Structure

```
output/
├── qwen_embeddings.mlmodelc/           # Embeddings (FP16)
├── qwen_lm_head_lut6.mlmodelc/         # LM Head (6-bit LUT)
├── qwen_FFN_PF_chunk_01of01.mlmodelc/  # FFN+Prefill (baked FP16)
├── meta.yaml                           # Model configuration
├── tokenizer.json                      # Tokenizer files
└── ...
```

### Testing

```bash
# Interactive chat
python tests/chat.py --meta /path/to/output/meta.yaml

# Single prompt
echo "Hello, how are you?" | python tests/chat.py --meta /path/to/output/meta.yaml
```

### Performance

| Mode | Speed | Model Size | Notes |
|------|-------|------------|-------|
| Dynamic Scales | ~6 t/s | Smaller | Runtime matmul for scales |
| **Baked Weights** | ~50 t/s | Larger | Pre-computed at conversion |

---

## 2. MIL Graph Conversion

**Use case:** Full control over CoreML graph, stateful KV cache, ANE-optimized operations.

This method builds the CoreML model directly using MIL (Model Intermediate Language) Builder, giving complete control over operations, state management, and ANE compatibility.

### When to Use MIL Conversion

- Need stateful KV cache (single-token decode)
- Require specific ANE-compatible patterns
- Building custom transformer architectures
- Maximum control over model structure

### Advantages

| Advantage | Description |
|-----------|-------------|
| Full control | Direct access to CoreML graph structure |
| Stateful KV cache | Built-in state management for decode |
| ANE optimization | Can match traced model patterns exactly |
| Flexibility | Custom ops and architectures |

### Disadvantages

| Disadvantage | Description |
|--------------|-------------|
| Complexity | Requires MIL Builder knowledge |
| Manual work | Handle data types and shapes manually |
| Debugging | More effort to troubleshoot ANE dispatch |

### Quick Start

```bash
# Convert Qwen model with MIL graph (includes stateful KV cache)
python -m anemll.ane_converter.qwen_converter \
    --model Qwen/Qwen3-0.6B \
    --aq1-checkpoint /path/to/quantized_checkpoint \
    --output /tmp/qwen_aq1 \
    --context 1024 \
    --num-layers 1 \
    --part aq1_full
```

### Full Example

```bash
# Convert 8 layers with quantized checkpoint
python -m anemll.ane_converter.qwen_converter \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/... \
    --aq1-checkpoint /Users/anemll/Downloads/anemll_q4_a4_e2e_v2_scales_only \
    --output /tmp/qwen_aq1 \
    --context 512 \
    --num-layers 8 \
    --part aq1_full
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--model` | HuggingFace model ID or local path |
| `--aq1-checkpoint` | Path to quantized checkpoint directory |
| `--output` | Output directory for CoreML models |
| `--context` | Context length (default: 1024) |
| `--num-layers` | Number of layers to convert (default: all) |
| `--start-layer` | Starting layer index (default: 0) |
| `--part` | Conversion part: `aq1_full`, `embeddings`, `lm_head`, etc. |

### Output Structure

```
output/
├── qwen_aq1_full.mlpackage/     # MIL graph model (uncompiled)
├── qwen_aq1_full.mlmodelc/      # Compiled model for ANE
└── model.mil                     # MIL text representation
```

### Testing

```bash
# Run inference test
python tests/dev/test_aq1_converter.py \
    --model /tmp/qwen_mil/qwen_aq1_full.mlmodelc \
    --prompt "Hello, world!"
```

### Performance

| Mode | Speed | Model Size | Notes |
|------|-------|------------|-------|
| MIL with KV state | ~40-50 t/s | Smaller | Stateful decode |
| Baked weights | ~50 t/s | Larger | Pre-computed scales |

### Related Documentation

For detailed MIL Builder patterns, ANE compatibility requirements, and troubleshooting:
- **[MB_GRAPH_ISSUES.md](MB_GRAPH_ISSUES.md)** - KV cache patterns, data types, error fixes
- **[A1F_FOLDING_ISSUE.md](A1F_FOLDING_ISSUE.md)** - Constant folding prevention

### Related Files

| File | Description |
|------|-------------|
| `anemll/ane_converter/qwen_converter.py` | Full transformer MIL conversion |
| `tests/dev/test_aq1_converter.py` | MIL conversion test script |
| `tests/dev/convert_mil_dynamic.py` | Dynamic conversion utilities |

---

## 3. PyTorch Trace-Based Conversion

**Use case:** Convert PyTorch model with quantized checkpoint, preserving `constexpr_lut_to_dense` + `matmul(A, B)` for LUT compression with dynamic scales.

This method traces the PyTorch model with quantized weights and converts to CoreML while preserving the factored scale structure.

### When to Use PyTorch Conversion

- Have a quantized checkpoint from QAT (Quantization-Aware Training)
- Want to preserve `A @ B` scale computation at runtime
- Need KV cache state for single-token generation
- Testing different quantization configurations

### Key Concept: Size-Aware Value Inference Patch

The critical fix that makes this work:

```python
# WRONG approaches (cause "epsilon must be const" error):
# - PassPipeline.EMPTY
# - pipeline.remove_passes(["common::const_elimination"])

# CORRECT approach: Patch value_inference to be SIZE-AWARE
_patch_value_inference(threshold=100000)  # patches both mul AND matmul
mlmodel = ct.convert(traced, ...)  # Keep DEFAULT pipeline
_restore_value_inference(original)
```

**How the patch works:**

| Tensor Size | value_inference returns | Effect |
|-------------|-------------------------|--------|
| Small (< 10K) | Normal result | Epsilon → const ✓ |
| Large (> 100K) | None | matmul(A,B) → stays as op ✓ |

This allows `const_elimination` to run (needed for epsilon scalars) while preventing large tensor folding.

### Quick Start

```bash
# 2-bit LUT test (8 layers, context 256)
source env-anemll/bin/activate && \
ANEMLL_DYNAMIC_SCALES=1 ANEMLL_SKIP_ATTN_LUT=1 python tests/dev/test_qwen_a1f_layer.py \
    --checkpoint /path/to/snapped_lut/model_state_dict.pt \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<your_hash> \
    --layers 8 \
    --context 256 \
    --output /tmp/qwen_8layer_256.mlpackage
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANEMLL_DYNAMIC_SCALES` | 0 | Set to 1 to enable `constexpr_lut_to_dense + matmul` |
| `ANEMLL_SKIP_ATTN_LUT` | 0 | Set to 1 to skip attention LUT (use FP16 attention) |

### Full Examples

```bash
# Find your HuggingFace model path first
ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/
# Use the snapshot hash in --model below

# 2-bit LUT checkpoint
source env-anemll/bin/activate && \
ANEMLL_DYNAMIC_SCALES=1 ANEMLL_SKIP_ATTN_LUT=1 python tests/dev/test_qwen_a1f_layer.py \
    --checkpoint /Users/anemll/Downloads/q2_pt_good1/snapped_lut/model_state_dict.pt \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
    --layers 8 \
    --context 256 \
    --output /tmp/qwen_Q2_8layer_256.mlpackage

# 4-bit LUT checkpoint
source env-anemll/bin/activate && \
ANEMLL_DYNAMIC_SCALES=1 ANEMLL_SKIP_ATTN_LUT=1 python tests/dev/test_qwen_a1f_layer.py \
    --checkpoint /Users/anemll/Downloads/anemll_q4_a4_e2e_v2_scales_only/snapped/model_state_dict.pt \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
    --layers 8 \
    --context 256 \
    --output /tmp/qwen_Q4_8layer_256.mlpackage

# Full 28 layers
source env-anemll/bin/activate && \
ANEMLL_DYNAMIC_SCALES=1 ANEMLL_SKIP_ATTN_LUT=1 python tests/dev/test_qwen_a1f_layer.py \
    --checkpoint /Users/anemll/Downloads/q2_pt_good1/snapped_lut/model_state_dict.pt \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
    --context 512 \
    --output /tmp/qwen_full_512.mlpackage
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | required | Path to quantized checkpoint (.pt file) |
| `--model` | Qwen/Qwen3-0.6B | HuggingFace model ID |
| `--layers` | all | Number of layers to convert |
| `--context` | 1024 | Context length |
| `--output` | /tmp/qwen_a1f_test.mlpackage | Output path |
| `--prefill` | - | Test prefill mode |
| `--chunk` | 0 | Chunk index for partial conversion |
| `--verbose` | - | Show detailed operation info |

### Expected Output

```
============================================================
MODEL ANALYSIS
============================================================

Operation counts:
  constexpr_lut_to_dense: 56    # LUT compressed weights
  matmul: 72                     # A @ B scale computation
  conv: 56                       # Actual convolutions
  read_state: 17                 # KV cache reads
  write_state: 16                # KV cache writes
  ...

AQ1 Operations:
  constexpr_lut_to_dense: 56
  matmul (A @ B scales): 72
  conv: 56

  ✓ constexpr_lut_to_dense ops found - LUT compression active!

============================================================
INFERENCE TEST
============================================================
Running inference with test inputs...
  Created model state for KV cache
  Input hidden_states: (1, 1, 1024)
  Output output_hidden_states: (1, 1, 1024)
  ✓ Inference successful!
```

### Troubleshooting

**"Operation output 'hidden_states' shadows an earlier declaration":**
- Using `PassPipeline.EMPTY` causes naming collision
- Fix: Keep DEFAULT pipeline, use `_patch_value_inference` instead

**"epsilon must be const" or "No schema registered":**
- `const_elimination` was disabled
- Fix: Keep `const_elimination` in pipeline, patch handles large tensors only

**Model size larger than FP16:**
- Dynamic scales stores indices + LUT + scale_A + scale_B
- For smaller size, use baked weights (Section 1) instead

**"This model was not loaded with the Core ML Framework":**
- Model parse error due to naming collision
- Fix: Use DEFAULT pipeline with patch

### Related Documentation

- **[A1F_FOLDING_ISSUE.md](A1F_FOLDING_ISSUE.md)** - Detailed explanation of constant folding prevention
- **[MB_GRAPH_ISSUES.md](MB_GRAPH_ISSUES.md)** - KV cache patterns and ANE compatibility

### Related Files

| File | Description |
|------|-------------|
| [test_qwen_a1f_layer.py](test_qwen_a1f_layer.py) | PyTorch trace conversion test |
| [test_palettize_with_scales.py](test_palettize_with_scales.py) | Palettization with factored scales |
| [ANEMLL-quant.py](ANEMLL-quant.py) | Quantization simulation script |
| [ANE-group-quant.py](ANE-group-quant.py) | Group quantization simulation |

---

## Checkpoint Format Reference

Quantized checkpoints should contain:

```python
{
    # Standard weights
    'model.embed_tokens.weight': tensor,
    'lm_head.weight': tensor,
    'model.norm.weight': tensor,

    # Per-layer quantized weights
    'model.layers.{i}.mlp.gate_proj.lut_work': tensor,  # [out, in] uint8 indices
    'model.layers.{i}.mlp.gate_proj.scale_A': tensor,   # [out, rank]
    'model.layers.{i}.mlp.gate_proj.scale_B': tensor,   # [rank, groups]
    # ... up_proj, down_proj similarly

    # Attention (if quantized)
    'model.layers.{i}.self_attn.q_proj.lut_work': tensor,
    'model.layers.{i}.self_attn.q_proj.scale_A': tensor,
    'model.layers.{i}.self_attn.q_proj.scale_B': tensor,
    # ... k_proj, v_proj, o_proj similarly

    # LayerNorm weights
    'model.layers.{i}.input_layernorm.weight': tensor,
    'model.layers.{i}.post_attention_layernorm.weight': tensor,
}
```

---

## Troubleshooting

**"Model not found" error:**
- Ensure HuggingFace model is downloaded first
- Check the model ID matches exactly (case-sensitive)

**Context length mismatch:**
- All model parts must use the same context length
- Re-export with consistent `--context` value

**Slow inference (~6 t/s):**
- Verify baked mode is being used (not dynamic scales)
- Check that models are compiled (.mlmodelc not .mlpackage)

---

## Related Files

| File | Description |
|------|-------------|
| [export_baked_model.py](export_baked_model.py) | Baked weights export script |
| [test_qwen_a1f_layer.py](test_qwen_a1f_layer.py) | FFN baked conversion |
| [A1F_FOLDING_ISSUE.md](A1F_FOLDING_ISSUE.md) | Technical details on constant folding |

---

## Appendix: Common Paths

### HuggingFace Model Paths

```bash
# Qwen3-0.6B (find your snapshot hash)
ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/

# Full path example:
~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
```

### Quantized Checkpoint Paths

```bash
# Q4_A4 checkpoint (4-bit LUT, rank-4 scales)
/Users/anemll/Downloads/anemll_q4_a4_e2e_v2_scales_only

# Q2 checkpoint (2-bit LUT)
/Users/anemll/Downloads/q2_pt_good1/snapped_lut
```

### Copy-Paste Commands

**MIL Graph Conversion (8 layers):**
```bash
python -m anemll.ane_converter.qwen_converter \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
    --aq1-checkpoint /Users/anemll/Downloads/anemll_q4_a4_e2e_v2_scales_only \
    --output /tmp/qwen_aq1 \
    --context 512 \
    --num-layers 8 \
    --part aq1_full
```

**Baked Weights Conversion:**
```bash
python tests/dev/export_baked_model.py \
    --checkpoint /Users/anemll/Downloads/anemll_q4_a4_e2e_v2_scales_only/snapped/model_state_dict.pt \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
    --context 512 \
    --output /tmp/qwen_baked
```

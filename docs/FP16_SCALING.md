# FP16 Scaling Guide for Apple Neural Engine

## Overview

Apple Neural Engine (ANE) requires FP16 inference. Some model architectures produce residual activations that exceed FP16's representable range (±65,504), causing overflow and NaN propagation. This guide documents which architectures need scaling and which don't.

## FP16 Compatibility by Architecture

| Model | Peak Residual | FP16 Max | Ratio | Needs `--fp16-scale`? |
|-------|---------------|----------|-------|----------------------|
| **LLaMA 3.2 1B** | 410 | 65,504 | **0.01x** | No |
| **Qwen3 1.7B** | 14,858 | 65,504 | **0.23x** | No |
| **Gemma3 270M** | 104,162 | 65,504 | **1.6x** | **Yes** (α=0.48) |
| **Gemma3 1B** | 61,040 | 65,504 | **0.93x** | Recommended (α=0.82) |
| **Gemma3 4B QAT** | 292,969 | 65,504 | **4.5x** | **Yes** (α=0.1875) |

## Architecture Analysis

### LLaMA (No Scaling Needed)

LLaMA models are exceptionally FP16-safe:
- Peak activations at ~410 (only 1% of FP16 max)
- Pre-LayerNorm architecture naturally bounds residual stream
- 100% token match between FP16 and BF16

**Tested models:**
- LLaMA 3.2 1B-Instruct: Peak 410.3, Ratio 0.01x

### Qwen3 (No Scaling Needed)

Qwen3 models are FP16 compatible out of the box:
- Peak activations at ~15,000 (23% of FP16 max)
- Pre-LayerNorm with RMSNorm
- 100% token match between FP16 and BF16

**Tested models:**
- Qwen3 1.7B: Peak 14,858, Ratio 0.23x

**Observation:** Layer 2 MLP produces a large spike (~14,500) that establishes the baseline, with subsequent layers growing slowly. This is different from Gemma3's unbounded accumulation.

### Gemma3 (Scaling Required)

Gemma3 is the only architecture requiring FP16 scaling:
- Post-norm with `(1+w)` gain structure causes residual accumulation
- BF16 training assumed wider dynamic range (±3.4×10³⁸)
- QAT models have the most severe overflow (4.5x FP16 max)

**Tested models:**
- Gemma3 270M: Peak 104,162, α=0.48
- Gemma3 1B: Peak 61,040, α=0.82
- Gemma3 4B QAT: Peak 292,969, α=0.1875

See [GEMMA3_FP16_SCALING.md](GEMMA3_FP16_SCALING.md) for detailed Gemma3-specific guidance.

## Pre-Conversion Preflight (Recommended)

Before converting any model, run:

```bash
./anemll/utils/fp16_preflight.sh --model <model_id_or_path>
```

Behavior:
- Runs FP16 compatibility checks with clamp sweep by default
- Saves JSON report to `tests/dev/logs/fp16_preflight_<timestamp>.json`
- Auto-activates local envs (`env-anemll`, `.venv`, etc.) when possible

## Using `--fp16-scale`

### When to Use

Only use `--fp16-scale` for Gemma3 models:

```bash
# Gemma3 270M
./anemll/utils/convert_model.sh \
    --model google/gemma-3-270m-it \
    --output /path/to/output \
    --fp16-scale 0.48

# Gemma3 1B
./anemll/utils/convert_model.sh \
    --model google/gemma-3-1b-it \
    --output /path/to/output \
    --fp16-scale 0.82

# Gemma3 4B QAT
./anemll/utils/convert_model.sh \
    --model google/gemma-3-4b-it-qat-int4-unquantized \
    --output /path/to/output \
    --fp16-scale 0.1875

# Auto-detect (Gemma3 only)
./anemll/utils/convert_model.sh \
    --model google/gemma-3-270m-it \
    --output /path/to/output \
    --fp16-scale auto
```

### When NOT to Use

Do not use `--fp16-scale` for:
- **LLaMA models** - Already FP16 safe
- **Qwen models** - Already FP16 safe
- **DeepSeek models** - Based on LLaMA/Qwen, FP16 safe

```bash
# LLaMA - no --fp16-scale needed
./anemll/utils/convert_model.sh \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --output /path/to/output

# Qwen - no --fp16-scale needed
./anemll/utils/convert_model.sh \
    --model Qwen/Qwen3-1.7B \
    --output /path/to/output
```

## Recommended `--fp16-scale` Values

| Model | `--fp16-scale` |
|-------|----------------|
| google/gemma-3-270m-it | `0.48` |
| google/gemma-3-1b-it | `0.82` |
| google/gemma-3-4b-it | `0.5` |
| google/gemma-3-4b-it-qat-int4-unquantized | `0.1875` |
| All LLaMA models | Not needed |
| All Qwen models | Not needed |

## Diagnostic Tools

### Check FP16 Compatibility

For any model, run the residual scaling analysis:

```bash
# Generic (works for LLaMA, Qwen, Gemma3)
python tests/dev/compute_residual_scaling_qwen.py --model <model_path>

# Gemma3-specific with scaling computation
python tests/dev/compute_residual_scaling.py --model <model_path>
```

### FP16 Compatibility Check

```bash
python anemll/utils/fp16_compatibility_check.py --model <model_id> --sweep
```

## Why Gemma3 is Different

The root cause is **residual accumulation overflow**:

```
Layer 0:  hidden = embed(input)           ~1,000
Layer 1:  hidden = hidden + Δ₁            ~5,000
Layer 2:  hidden = hidden + Δ₂            ~15,000
...
Layer 5:  hidden = hidden + Δ₅            ~72,000  ← EXCEEDS FP16 MAX!
```

LLaMA and Qwen use **pre-LayerNorm** which normalizes before each sublayer, naturally bounding activations. Gemma3 uses **post-LayerNorm** with a `(1+w)` gain structure that allows unbounded growth.

## Test Results (2026-02-07)

### LLaMA 3.2 1B-Instruct
```
Peak layer output: 410.3
FP16 max: 65504.0
Ratio to FP16 max: 0.01x
FP16 generation: SUCCESS
BF16 generation: SUCCESS
FP16 vs BF16 token match: 100.0%
```

### Qwen3 1.7B
```
Peak layer output: 14858.1
FP16 max: 65504.0
Ratio to FP16 max: 0.23x
FP16 generation: SUCCESS
BF16 generation: SUCCESS
FP16 vs BF16 token match: 100.0%
```

## References

- [GEMMA3_FP16_SCALING.md](GEMMA3_FP16_SCALING.md) - Detailed Gemma3 scaling guide
- `tests/dev/compute_residual_scaling.py` - Gemma3 scaling computation
- `tests/dev/compute_residual_scaling_qwen.py` - Generic scaling analysis
- `anemll/utils/fp16_compatibility_check.py` - FP16 compatibility checker
- `anemll/utils/fp16_preflight.sh` - one-command FP16 pre-conversion sweep

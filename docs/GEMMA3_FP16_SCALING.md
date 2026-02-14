# Gemma3 FP16/ANE Scaling Guide for Apple Neural Engine

## Problem

Gemma3 models trained with BF16 can produce residual stream activations that exceed FP16's representable range (±65,504). FP16 inference is a hard requirement for ANE. This causes:
- Overflow to `inf` in FP16 computation
- NaN propagation through subsequent layers
- Complete model failure on ANE (which uses FP16)

This issue affects **all Gemma3 model sizes** (1B through 27B) as documented by [Unsloth](https://unsloth.ai/blog/gemma3).

## Root Cause

The issue is **residual accumulation overflow**, not individual tensor overflow:
- All sub-tensors (attention, MLP, norms) are within FP16 range
- The cumulative residual connection `output = input + attention + mlp` grows too large
- BF16 has range ±3.4×10³⁸, while FP16 only has ±65,504

```
Layer 0:  hidden = embed(input)           ~1,000
Layer 1:  hidden = hidden + Δ₁            ~5,000
Layer 2:  hidden = hidden + Δ₂            ~15,000
...
Layer 5:  hidden = hidden + Δ₅            ~72,000  ← EXCEEDS FP16 MAX!
Layer N:  hidden = hidden + Δₙ            ~293,000 (for QAT models)
```

## Solutions

We investigated two approaches. **Both work**, but have different trade-offs.

### Option A: Runtime Clamping (Quick Fix)

Apply `clip(hidden, -55000, 55000)` after residual additions.

**Pros:**
- Simple to implement
- No weight preprocessing required
- Validated: 100% token match at clamp=55,000

**Cons:**
- Adds runtime ops (one clamp per layer)
- May lose information for extreme activations
- Slightly reduces ANE efficiency

**When to use:** Quick prototyping, or as a safety net alongside scaling.

```python
# In forward pass, after each residual add:
hidden = hidden + attention_output
hidden = torch.clamp(hidden, -55000, 55000)
```

### Option B: Weight Scaling (Recommended)

Apply a **weight-only transformation** that shrinks the residual stream without adding runtime ops.

**Pros:**
- Zero runtime overhead
- Mathematically function-preserving
- 100% token match with BF16 baseline
- Better ANE efficiency

**Cons:**
- Requires weight preprocessing
- Must compute α per model

**When to use:** Production deployments, CoreML/ANE conversion.

#### The Transformation

For Gemma3's `(1 + weight)` gain structure:

```python
# 1. Scale embedding weights
embed_tokens.weight *= α

# 2. Transform post-norm weights
for layer in layers:
    post_attention_layernorm.weight = α * (1 + w_old) - 1
    post_feedforward_layernorm.weight = α * (1 + w_old) - 1
```

#### Why This Works

Let a block be:
- `y = RMSNorm(x)` - normalized input
- `Δ = f(y)` - attention + MLP branch outputs
- `x_next = x + Δ` - residual connection

If we scale both stream and update by α:
- `x' = α·x`, `Δ' = α·Δ`
- `x_next' = α·x + α·Δ = α·(x + Δ)`

Because RMSNorm is scale-invariant (eps is tiny), the internal computation stays the same. The final RMSNorm before the LM head also cancels the global scale.

#### Computing α

Formula: `α = target_max / observed_peak` where `target_max ≈ 50,000` (with headroom)

Convenient binary fractions:
- `α = 3/16 = 0.1875` - for ~293k peak → 54,932 (good for 4B QAT)
- `α = 1/8 = 0.125` - more aggressive, very safe

## Model-Specific Results

### google/gemma-3-270m

| Metric | Value |
|--------|-------|
| Original Peak | 104,162 (1.6x FP16 max) |
| First Overflow | Layer 7 |
| **Recommended α** | **0.48** |
| Scaled Peak | 50,000 |
| Token Match | 100% |

### google/gemma-3-1b-it

| Metric | Value |
|--------|-------|
| Original Peak | 61,040 (0.93x FP16 max) |
| First Overflow | None (but close to limit) |
| **Recommended α** | **0.82** (for headroom) |
| Scaled Peak | 50,000 |
| Token Match | 100% |

### google/gemma-3-4b-it-qat-int4-unquantized

| Metric | Value |
|--------|-------|
| Original Peak | 292,969 (4.5x FP16 max) |
| First Overflow | Layer 5 |
| **Recommended α** | **0.17-0.1875 (3/16)** |
| Scaled Peak | ~50,000-55,000 |
| Token Match | 100% |

## Usage with ANEMLL Conversion

### Using convert_model.sh

The `--fp16-scale` flag applies scaling automatically during conversion:

```bash
# For 4B QAT model (most aggressive scaling needed)
./anemll/utils/convert_model.sh \
    --model google/gemma-3-4b-it-qat-int4-unquantized \
    --output /path/to/output \
    --context 4096 \
    --chunk 2 \
    --lut2 4,0 \
    --fp16-scale 0.1875 \
    --skip-check

# For 1B model (minor scaling for headroom)
./anemll/utils/convert_model.sh \
    --model google/gemma-3-1b-it \
    --output /path/to/output \
    --context 4096 \
    --fp16-scale 0.82 \
    --skip-check

# Auto-detect scaling based on model name
./anemll/utils/convert_model.sh \
    --model google/gemma-3-270m-it \
    --output /path/to/output \
    --fp16-scale auto \
    --skip-check
```

### Recommended `--fp16-scale` Values

| Model | --fp16-scale |
|-------|--------------|
| google/gemma-3-270m-it | `0.48` |
| google/gemma-3-1b-it | `0.82` |
| google/gemma-3-4b-it | `0.5` |
| google/gemma-3-4b-it-qat-int4-unquantized | `0.1875` |

## Implementation

### Full Implementation (Weight Preprocessing)

```python
def apply_residual_scaling(model, alpha: float):
    """Apply residual stream scaling to Gemma3 model weights."""

    # Get model components
    if hasattr(model.model, 'language_model'):
        embed = model.model.language_model.embed_tokens
        layers = model.model.language_model.layers
    else:
        embed = model.model.embed_tokens
        layers = model.model.layers

    # 1. Scale embedding weights
    with torch.no_grad():
        embed.weight.mul_(alpha)

    # 2. Transform post-norm weights
    for layer in layers:
        for norm_name in ['post_attention_layernorm', 'post_feedforward_layernorm']:
            if hasattr(layer, norm_name):
                norm = getattr(layer, norm_name)
                if hasattr(norm, 'weight'):
                    with torch.no_grad():
                        # Gemma3 uses (1 + w) gain
                        norm.weight.data = alpha * (1 + norm.weight.data) - 1
```

### Minimal Variant (Scale from Layer N)

If overflow starts at layer 5, you can scale only from that point:

```python
def apply_partial_scaling(model, alpha: float, start_layer: int = 5):
    """Apply scaling only from a specific layer onward."""

    layers = model.model.layers  # or model.model.language_model.layers

    # Insert one hidden *= alpha between layers start_layer-1 and start_layer
    # (requires model modification or hook)

    # Transform post-norm weights only for layers >= start_layer
    for i, layer in enumerate(layers):
        if i >= start_layer:
            for norm_name in ['post_attention_layernorm', 'post_feedforward_layernorm']:
                norm = getattr(layer, norm_name, None)
                if norm and hasattr(norm, 'weight'):
                    with torch.no_grad():
                        norm.weight.data = alpha * (1 + norm.weight.data) - 1
```

### Hybrid: Scaling + Safety Clamp

For maximum robustness, combine both approaches:

```python
# 1. Apply weight scaling (preprocessing)
apply_residual_scaling(model, alpha=0.1875)  # 3/16

# 2. Add safety clamp (in forward, rarely triggers)
hidden = torch.clamp(hidden, -60000, 60000)
```

### Configurable Clamping in Gemma3Config

Runtime clamping is **OFF by default** (weight scaling is preferred). To enable clamping as a safety net:

```python
# In config
config = Gemma3Config(
    enable_residual_clamp=True,    # Enable clamping (default: False)
    residual_clamp_value=55000.0,  # Custom clamp value (default: 65504.0)
    ...
)

# Or via kwargs when loading
config_dict = json.load(open("config.json"))
config_dict["enable_residual_clamp"] = True
config_dict["residual_clamp_value"] = 55000.0
config = Gemma3Config(**config_dict)
```

## Comparison Summary

| Aspect | Runtime Clamping | Weight Scaling | Hybrid |
|--------|------------------|----------------|--------|
| Runtime ops | 1 clamp/layer | None | 1 clamp (safety) |
| Quality | ~100% match | 100% match | 100% match |
| ANE efficiency | Slightly reduced | Optimal | Near-optimal |
| Implementation | Easy | Moderate | Moderate |
| Robustness | Good | Good | Best |

**Recommendation**: Use weight scaling (Option B) for production. Add safety clamp if concerned about edge cases.

## Multimodal Considerations

If using Gemma3 with vision inputs:
- Vision tower can also produce large activations
- May need similar scaling/clamping at vision-to-language handoff
- Check `multi_modal_projector` outputs if inf/NaN appears with images

## Diagnostic Tools

### FP16 Compatibility Check
```bash
python anemll/utils/fp16_compatibility_check.py --model <model_id> --sweep
```

### Compute Scaling Factors
```bash
python tests/dev/compute_residual_scaling.py --model <model_id> --save scaling.json
```

### Detailed Tensor Analysis
```bash
python tests/dev/test_gemma3_qat_tensor_overflow_map.py
```

## References

- `anemll/utils/fp16_compatibility_check.py` - Universal FP16 compatibility checker
- `tests/dev/compute_residual_scaling.py` - Scaling factor computation
- `tests/dev/test_gemma3_qat_residual_scaling_fix.py` - Scaling fix validation
- `tests/dev/test_gemma3_qat_tensor_overflow_map.py` - Detailed tensor analysis
- [Unsloth Blog: Gemma 3](https://unsloth.ai/blog/gemma3) - FP16 overflow documentation

## Notes

1. **QAT Models**: The QAT (int4-unquantized) models have more severe overflow because they were trained expecting int4 quantization to constrain values.

2. **Per-Layer Scaling**: Uniform α is usually sufficient. Per-layer scaling only helps if early layers are well within range and you want to preserve their precision.

3. **Safety Margin**: Use `target_max = 50,000` instead of `65,504` to provide headroom for variation across different prompts.

4. **Binary Fractions**: Using α = 3/16 or 1/8 may be more efficient for hardware that handles power-of-2 operations well.

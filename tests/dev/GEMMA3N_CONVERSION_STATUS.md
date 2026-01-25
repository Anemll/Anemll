# Gemma3n CoreML Conversion - Status Report

**Date**: January 24, 2026
**Branch**: `dev_gemma3n`
**Model**: `gemma-3n-E2B-it` (Gemma 3n E2B Instruct)

---

## Executive Summary

The Gemma3n CoreML conversion is in progress. KV cache position indexing has been fixed to use scatter-based operations for dynamic tracing. The attention scaling has been corrected to match HuggingFace reference implementation. However, the model still produces degenerate output ("is is is is..." repetition pattern).

**Current Blocker**: Export process keeps getting OOM killed on the current system (25GB free disk, limited RAM). Need to continue on a system with more resources.

---

## Key Technical Discoveries

### 1. KV Cache Position Indexing (FIXED)

**Problem**: Using `int(current_pos)` or `current_pos.item()` in PyTorch becomes a constant during JIT tracing, causing all tokens to write to position 0.

**Solution**: Use non-in-place `scatter()` with position index tensors:

```python
# Create position index tensor for scatter: [1, heads, 1, dim]
pos_idx = current_pos.long().view(1, 1, 1, 1).expand(1, num_kv_heads, 1, head_dim)

# Use non-in-place scatter to create updated tensors
updated_keys = kv_key_slice.scatter(2, pos_idx, key_states.half())
updated_values = kv_value_slice.scatter(2, pos_idx, value_states.half())

# Write back to kv_cache using slice assignment (CoreML state update)
kv_cache[key_idx:key_idx + 1] = updated_keys
kv_cache[value_idx:value_idx + 1] = updated_values
```

**Status**: VERIFIED WORKING - KV cache positions are correctly updated at dynamic positions (tested via `coremltools` state read API).

### 2. Attention Scaling (FIXED in code, needs re-export)

**Problem**: Our implementation was applying TRIPLE scaling:
1. `query_states * self.query_pre_attn_scalar` (256)
2. `attn_weights / math.sqrt(self.head_dim)` (1/16)
3. Softcapping `torch.tanh(attn_weights / 30.0) * 30.0`

**Discovery**: HuggingFace Gemma3n uses:
- `scaling=1.0` (NO attention scaling)
- NO softcapping (`softcap=None`)
- QKV normalization (which keeps values bounded, making scaling unnecessary)

**Fix Applied** (in `gemma3n_model.py`):
- Removed `query_states * self.query_pre_attn_scalar` from `get_new_kv_cache()` and `get_new_kv_cache_prefill()`
- Changed `attn_weights = Q @ K.T / sqrt(head_dim)` to `attn_weights = Q @ K.T`
- Removed softcapping `torch.tanh(attn_weights / 30.0) * 30.0`

**Status**: Code fixed, needs re-export to test.

### 3. Dtype Requirements

**Requirement**: ANE buffers must be FP16. All model operations must use FP16.

**Status**: Implemented - all KV cache operations use `.half()` to ensure FP16.

---

## Files Modified

### Core Model (`anemll/models/gemma3n_model.py`)

| Function | Change | Lines |
|----------|--------|-------|
| `Gemma3nAttention.forward()` | Removed query_pre_attn_scalar, scaling, softcapping | ~347-381 |
| `Gemma3nTextAttention.get_new_kv_cache()` | Removed query_pre_attn_scalar | ~616-628 |
| `Gemma3nTextAttention.get_new_kv_cache_prefill()` | Removed query_pre_attn_scalar | ~642-653 |
| `Gemma3nTextAttention.forward_regular()` | Removed scaling/softcapping, scatter-based KV update | ~651-703 |
| `Gemma3nTextAttention.forward_prefill()` | Removed scaling/softcapping | ~705-748 |
| `_process_layer_regular()` | Scatter-based KV cache position updates | ~890-920 |

### Converter (`anemll/ane_converter/gemma3n_converter.py`)

- Dtype handling for FP16 consistency
- State type definitions for KV cache

---

## Conversion Commands

### Export Infer Models (Full Pipeline)

```bash
source env-anemll/bin/activate

python tests/dev/export_gemma3n.py \
  --model /Users/anemll/Models/Models/gemma-3n-E2B-it \
  --output /tmp/gemma3n-output \
  --part infer \
  --context-length 512 \
  --chunk 4
```

This creates:
- `gemma3n_infer_init.mlpackage` - Embeddings + per-layer inputs
- `gemma3n_infer_chunk_00of04.mlpackage` - Layers 0-7 (with KV cache state)
- `gemma3n_infer_chunk_01of04.mlpackage` - Layers 8-14
- `gemma3n_infer_chunk_02of04.mlpackage` - Layers 15-22
- `gemma3n_infer_chunk_03of04.mlpackage` - Layers 23-29
- `gemma3n_combine_streams.mlpackage` - AltUp stream combination
- `gemma3n_lm_head.mlpackage` - 16-way split LM head

### Test Chat Output

```bash
python tests/dev/test_gemma3n_ane_chat.py \
  --bundle /tmp/gemma3n-output/bundle \
  --use-infer \
  --prompt "The capital of France is" \
  --max-new-tokens 16 \
  --context-length 512 \
  --verbose
```

### Debug KV Cache State

```bash
python tests/dev/test_gemma3n_kv_state_debug.py \
  --bundle /tmp/gemma3n-output/bundle \
  --context-length 512
```

### Compare CoreML vs PyTorch

```bash
python tests/dev/test_gemma3n_coreml_vs_pytorch_chunks.py \
  --bundle /tmp/gemma3n-output/bundle \
  --model /Users/anemll/Models/Models/gemma-3n-E2B-it \
  --context-length 512 \
  --chunk 4 \
  --device cpu \
  --dtype float16
```

---

## Test Files Created

| File | Purpose |
|------|---------|
| `test_gemma3n_ane_chat.py` | End-to-end chat testing |
| `test_gemma3n_kv_state_debug.py` | KV cache state inspection using coremltools 9.0 API |
| `test_gemma3n_prefill_debug.py` | Multi-token prefill testing |
| `test_gemma3n_parts_debug.py` | Modular testing of individual parts |
| `test_gemma3n_coreml_vs_pytorch_chunks.py` | Numerical comparison |
| `test_gemma3n_attention_fixes.py` | Attention scaling verification |

---

## Known Issues

### 1. Degenerate Output (ACTIVE)

**Symptom**: Model outputs "is is is is..." or "aishowishie deepening deepening..." instead of coherent text.

**Root Cause Analysis**:
- KV cache position indexing: VERIFIED FIXED (positions are dynamic)
- Attention scaling: CODE FIXED (removed query_pre_attn_scalar, softcapping)
- Need to re-export and test with fixed attention

### 2. Export OOM Killed

**Symptom**: Export process killed with exit code 137 during chunk conversion.

**Cause**: Insufficient system resources (RAM, disk space ~25GB free).

**Workaround**: Continue on system with more resources.

### 3. HF Model Load Requires timm

The full HuggingFace model requires `timm` library for vision tower. For text-only testing, this can be worked around.

---

## Architecture Notes

### Gemma3n Model Architecture

- **Layers**: 30 transformer layers
- **Hidden Size**: 2048
- **Heads**: 8 attention heads, 2 KV heads (GQA)
- **Head Dim**: 256
- **Layer Types**: Mix of `sliding_attention` (512 window) and `full_attention`
- **Special Features**:
  - Per-Layer Embeddings (PLE): 7680 dims (30 layers x 256)
  - LAUREL: Low-rank adapters per layer
  - AltUp: 4-stream hidden states with unembed projections
  - QKV Normalization: RMSNorm on Q, K, V before attention

### Key Config Values

```python
head_dim = 256
num_attention_heads = 8
num_key_value_heads = 2
sliding_window = 512
scaling = 1.0  # NO attention scaling!
softcap = None  # NO softcapping!
vocab_size = 262400
final_logit_softcapping = 30.0  # For output logits only
```

---

## Next Steps

### Immediate (After System Switch)

1. **Re-export with attention fixes**:
   ```bash
   rm -rf /tmp/gemma3n-*
   python tests/dev/export_gemma3n.py \
     --model /path/to/gemma-3n-E2B-it \
     --output /tmp/gemma3n-fixed \
     --part infer \
     --context-length 512 \
     --chunk 4
   ```

2. **Test chat output**:
   ```bash
   python tests/dev/test_gemma3n_ane_chat.py \
     --bundle /tmp/gemma3n-fixed/bundle \
     --use-infer \
     --prompt "The capital of France is" \
     --max-new-tokens 20
   ```

3. **Compare numerics if still wrong**:
   ```bash
   python tests/dev/test_gemma3n_coreml_vs_pytorch_chunks.py \
     --bundle /tmp/gemma3n-fixed/bundle \
     --model /path/to/gemma-3n-E2B-it
   ```

### If Still Degenerate

1. **Check causal mask application** - verify mask is correctly applied at dynamic positions
2. **Check sliding window handling** - verify sliding attention layers use correct window
3. **Check AltUp stream combination** - verify 4-stream merge is correct
4. **Compare layer-by-layer outputs** - use `test_gemma3_layer_by_layer.py`

### Future Optimizations

1. Add LUT quantization (4-bit, 6-bit)
2. Optimize chunk boundaries for ANE efficiency
3. Add batched prefill support
4. Performance profiling on M-series chips

---

## Environment Requirements

- **Python**: 3.9 (strictly required)
- **coremltools**: >= 8.2 (9.0 recommended for state APIs)
- **transformers**: >= 4.57.0 (for Gemma3n support)
- **macOS**: Sequoia with ANE
- **Disk Space**: >= 50GB recommended
- **RAM**: >= 32GB recommended for export

---

## References

- [HuggingFace Gemma3n](https://huggingface.co/google/gemma-3n-e2b-it)
- [CoreML Tools State API](https://apple.github.io/coremltools/)
- [ANEMLL Project](https://github.com/anemll/anemll)

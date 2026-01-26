# Gemma3n CoreML Conversion - Status Report

**Date**: January 24, 2026
**Updated**: January 25, 2026 (CoreML infer pipeline FIXED)
**Branch**: `dev_gemma3n`
**Model**: `gemma-3n-E2B-it` (Gemma 3n E2B Instruct)

---

## Executive Summary

**STATUS: PYTORCH ✅ / COREML ✅** - The Gemma3n ANEMLL PyTorch implementation matches HuggingFace, and the CoreML infer pipeline now produces coherent output.

| Model | Top Prediction for "The capital of France is" | Probability | Cosine Similarity |
|-------|---------------------------------------------|-------------|-------------------|
| HuggingFace Reference | " Paris" | 90.68% | - |
| ANEMLL PyTorch (FIXED) | " Paris" | 90.42% | 1.0000 |

**Root Causes FIXED**:
1. **RMSNorm Implementation**: Changed to ANE-optimized doubled-tensor trick
2. **Rotary Cache Dtype**: Fixed to cast cos/sin to requested dtype (float16)

**Current Status**: CoreML infer models re-exported and validated. Chat now returns:  
`"Paris. Paris is known for its iconic landmarks like the Eiffel Tower and the Louvre"`

**Latest Run Artifacts (this session)**:
- Infer bundle: `/tmp/gemma3n-rotary/infer/`
- LM head: `/tmp/gemma3n-rotary/lm_head/gemma3n_lm_head.mlpackage`
- Tokenizer: `/tmp/gemma3n-rotary/tokenizer/` (copied into infer bundle)
- Combine streams: `/tmp/gemma3n-rotary/combine_streams/gemma3n_combine_streams.mlpackage` (copied into infer bundle)

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

### 4. RMSNorm Implementation (FIXED - Critical)

**Problem**: Original implementation used simple mean subtraction which doesn't match HuggingFace.

**Solution**: Use ANE-optimized doubled-tensor trick from LLaMA/Qwen:
```python
# concat([x, -x]) creates a tensor with mean=0, so LayerNorm's
# mean-subtraction becomes a no-op and we recover true RMSNorm statistics
doubled = torch.cat([x, -x], dim=-1)
normed = F.layer_norm(doubled, (2 * hidden_size,), None, None, eps)
normed = normed[..., :hidden_size]
return (normed * self.weight.to(normed.dtype)).to(input_dtype)
```

**Status**: FIXED and verified - cosine similarity 1.0000 with HuggingFace.

### 5. Rotary Cache Dtype (FIXED)

**Problem**: `create_rotary_cache()` computed in float32 and didn't cast to requested dtype.

**Solution**: Cast cos/sin to requested dtype at the end:
```python
if dtype is not None:
    cos = cos.to(dtype)
    sin = sin.to(dtype)
```

**Status**: FIXED - all operations now maintain float16 precision.

### 6. CoreML-Friendly Mask and Rotary Inputs (FIXED)

**Problem**: When `current_pos` is baked into the graph, both the causal mask gather and the rotary cache indexing collapse to constants in CoreML. Hidden states stopped depending on position even though the KV cache was updating.

**Solution**:
- Pass a pre-sliced causal mask row (`[1,1,ctx,ctx]`) from the host.
- Pass a one-hot tensor (`[1,1,ctx,1]`) that selects the KV cache slot to update.
- Pass the precomputed rotary embeddings (`cos/sin` for both local/global attention) as inputs instead of indexing inside the model.

**Implementation**:

### 7. Chunk Layer Splits (FIXED)

**Problem**: Layer chunking used integer division, which dropped remainder layers when `num_hidden_layers` is not divisible by `chunk_size`.

**Fix**: Use remainder-aware split (early chunks get +1 layer) so all layers are covered in both FFN and infer conversions.
- `anemll/models/gemma3n_model.py`: `_process_layer_regular` now accepts `position_one_hot` + rotary tensors.
- `anemll/ane_converter/gemma3n_converter.py`: Chunk wrappers & CoreML input specs updated.
- `tests/dev/gemma3n_coreml_inputs.py`: Shared helpers for mask/one-hot/rotary generation.
- `tests/dev/test_gemma3n_ane_chat_fixed.py` and other debug tools updated to feed the new tensors.

**Status**: Re-exported and validated. CoreML hidden states match PyTorch (cosine similarity ~0.99999).

### 7. CoreML Logit Concatenation Order (FIXED)

**Problem**: Split logits were concatenated using lexicographic key ordering (`logits_split_1, logits_split_10, ...`), which shuffled vocab blocks and produced degenerate output.

**Fix**: Sort logits by numeric suffix before concatenation.

**Status**: CoreML chat output is now coherent and matches PyTorch behavior.

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

### CoreML Utilities (`tests/dev/gemma3n_coreml_inputs.py`)

| File | Purpose |
|------|---------|
| `gemma3n_coreml_inputs.py` | Shared helpers for causal row masks, KV one-hot positions, and RoPE inputs |

### CoreML Scripts

| File | Purpose |
|------|---------|
| `chat_coreml_only.py` | CoreML-only chat runner (single infer or chunked infer) |
| `test_gemma3n_ane_chat_fixed.py` | End-to-end chat using infer chunks + KV state propagation |
| `test_gemma3n_kv_state_debug.py` | KV cache state inspection with `coremltools` state API |
| `test_gemma3n_kv_prefill_compare.py` | Compare CoreML vs PyTorch KV cache after prefill |
| `test_gemma3n_prefill_debug.py` | Token-by-token prefill checks |
| `test_gemma3n_coreml_vs_pytorch_chunks.py` | Chunk-level numeric diff (CoreML vs PyTorch) |

### Converter (`anemll/ane_converter/gemma3n_converter.py`)

- Dtype handling for FP16 consistency
- State type definitions for KV cache

---

## Conversion Commands

### Export Infer Models (Full Pipeline)

```bash
source env-anemll/bin/activate

MODEL_PATH=$(ls -td ~/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots/* | head -1)
echo "$MODEL_PATH"

python tests/dev/export_gemma3n.py \
  --model "$MODEL_PATH" \
  --output ~/Models/ANE/gemma3n \
  --part infer \
  --context-length 512 \
  --chunk 4
```

Single-step export script (all parts):

```bash
tests/dev/export_gemma3n_full.sh \
  --model "$MODEL_PATH" \
  --output ~/Models/ANE/gemma3n \
  --context-length 512 \
  --chunk 4
```

With LUT6 (FFN + LM head):

```bash
tests/dev/export_gemma3n_full.sh \
  --model "$MODEL_PATH" \
  --output ~/Models/ANE/gemma3n \
  --context-length 512 \
  --chunk 4 \
  --lut 6 \
  --lut-per-channel 8 \
  --lut-workers 1
```

LUT scope (limit palettization to specific CoreML ops):

```bash
tests/dev/export_gemma3n_full.sh \
  --model "$MODEL_PATH" \
  --output ~/Models/ANE/gemma3n_lut6 \
  --context-length 512 \
  --chunk 2 \
  --lut 6 \
  --lut-per-channel 8 \
  --lut-workers 1 \
  --lut-scope conv
```

FFN-only LUT (skip attention + router/gates):

```bash
tests/dev/export_gemma3n_full.sh \
  --model "$MODEL_PATH" \
  --output ~/Models/ANE/gemma3n_lut6 \
  --context-length 512 \
  --chunk 2 \
  --lut 6 \
  --lut-per-channel 8 \
  --lut-workers 1 \
  --lut-scope linear \
  --lut-include "gate_proj|up_proj|down_proj" \
  --lut-exclude "q_proj|k_proj|v_proj|o_proj|router|per_layer_input_gate" \
  --lut-report
```

With `--lut-report`, the converter prints:
- matched weights count (pre-quant)
- post-quant unique-value count (<= 2^bits)
- `weight.bin` size for each saved `.mlpackage` (helps verify compression)

Flat layout (all artifacts in one folder, no subdirs):

```bash
tests/dev/export_gemma3n_full.sh \
  --model "$MODEL_PATH" \
  --output ~/Models/ANE/gemma3n_lut6 \
  --context-length 512 \
  --chunk 2 \
  --lut 6 \
  --lut-per-channel 8 \
  --lut-workers 1 \
  --flat
```

If you export only `infer` (no tokenizer/LM head), pass a tokenizer path when testing:

```bash
python tests/dev/test_gemma3n_ane_chat_fixed.py \
  --bundle /Users/anemll/Models/ANE/gemma3n_lut8 \
  --tokenizer "$MODEL_PATH" \
  --prompt "History of ancient egypt" \
  --max-new-tokens 256 \
  --context-length 512 --verbose
```

> **Important**: This step requires access to POSIX shared memory (PyTorch imports abort with `OMP: Error #179` when `shm_open` is blocked). Run the export outside restricted sandboxing.

This creates:
- `gemma3n_infer_init.mlpackage` - Embeddings + per-layer inputs
- `gemma3n_infer_chunk_00of04.mlpackage` - Layers 0-7 (with KV cache state)
- `gemma3n_infer_chunk_01of04.mlpackage` - Layers 8-14
- `gemma3n_infer_chunk_02of04.mlpackage` - Layers 15-22
- `gemma3n_infer_chunk_03of04.mlpackage` - Layers 23-29
- `gemma3n_combine_streams.mlpackage` - AltUp stream combination
- `gemma3n_lm_head.mlpackage` - 16-way split LM head

### Export LM Head + Tokenizer (Required for Chat)

```bash
source env-anemll/bin/activate

python tests/dev/export_gemma3n.py \
  --model "$MODEL_PATH" \
  --output ~/Models/ANE/gemma3n \
  --part lm_head \
  --lut 6 \
  --lut-per-channel 8

python tests/dev/export_gemma3n.py \
  --model "$MODEL_PATH" \
  --output ~/Models/ANE/gemma3n \
  --part tokenizer

# Copy artifacts into the infer bundle for easy testing
cp -r ~/Models/ANE/gemma3n/lm_head/gemma3n_lm_head.mlpackage ~/Models/ANE/gemma3n/infer/
cp ~/Models/ANE/gemma3n/tokenizer/*.json ~/Models/ANE/gemma3n/infer/
cp ~/Models/ANE/gemma3n/tokenizer/tokenizer.model ~/Models/ANE/gemma3n/infer/
```

### Export Combine Streams (Optional Re-export)

```bash
source env-anemll/bin/activate

python tests/dev/export_gemma3n.py \
  --model "$MODEL_PATH" \
  --output ~/Models/ANE/gemma3n \
  --part combine_streams

cp -r ~/Models/ANE/gemma3n/combine_streams/gemma3n_combine_streams.mlpackage ~/Models/ANE/gemma3n/infer/
```

### Test Chat Output

```bash
python tests/dev/test_gemma3n_ane_chat_fixed.py \
  --bundle ~/Models/ANE/gemma3n/infer \
  --prompt "The capital of France is" \
  --max-new-tokens 16 \
  --context-length 512 \
  --verbose
```

CoreML-only runner (no PyTorch dependencies):

```bash
python tests/dev/chat_coreml_only.py \
  --bundle ~/Models/ANE/gemma3n/infer \
  --prompt "The capital of France is" \
  --max-new-tokens 16 \
  --context-length 512
```

Expected sample output (greedy):
`Paris. Paris is known for its iconic landmarks like the Eiffel Tower and the Louvre`

### Debug KV Cache State

```bash
python tests/dev/test_gemma3n_kv_state_debug.py \
  --bundle ~/Models/ANE/gemma3n/infer \
  --context-length 512
```

### Compare KV Cache (CoreML vs PyTorch)

```bash
python tests/dev/test_gemma3n_kv_prefill_compare.py \
  --bundle ~/Models/ANE/gemma3n_lut6/infer \
  --model "$MODEL_PATH" \
  --prompt "What is Apple Neural Engine?" \
  --context-length 512
```

### Compare CoreML vs PyTorch

```bash
python tests/dev/test_gemma3n_coreml_vs_pytorch_chunks.py \
  --bundle ~/Models/ANE/gemma3n/infer \
  --model "$MODEL_PATH" \
  --context-length 512 \
  --chunk 4 \
  --device cpu \
  --dtype float16
```

---

## Repro Workflow (Current Fixes)

1. **Export infer + lm_head + tokenizer** (see above) into `/tmp/gemma3n-output`.
2. **Copy artifacts into the infer bundle** so `test_gemma3n_ane_chat_fixed.py` can load everything from a single path.
3. **Run chat** via `test_gemma3n_ane_chat_fixed.py` to verify outputs.
4. **If output is degenerate**, run:
   - `test_gemma3n_kv_state_debug.py` to confirm KV updates
   - `test_gemma3n_kv_prefill_compare.py` to compare CoreML vs PyTorch KV cache after prefill
   - `test_gemma3n_coreml_vs_pytorch_chunks.py` for numerical diff per chunk
   - `test_gemma3n_prefill_debug.py` for token-by-token cache updates

---

## Quick Start (Current Snapshot Path)

```bash
source env-anemll/bin/activate

MODEL_PATH=/Users/anemll/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots/0330734afc91972a1aa1ba1bc4495e2723666854
OUT_DIR=/tmp/gemma3n-output

python tests/dev/export_gemma3n.py --model "$MODEL_PATH" --output "$OUT_DIR" --part infer --context-length 512 --chunk 4
python tests/dev/export_gemma3n.py --model "$MODEL_PATH" --output "$OUT_DIR" --part lm_head
python tests/dev/export_gemma3n.py --model "$MODEL_PATH" --output "$OUT_DIR" --part tokenizer
python tests/dev/export_gemma3n.py --model "$MODEL_PATH" --output "$OUT_DIR" --part combine_streams

cp -r "$OUT_DIR/lm_head/gemma3n_lm_head.mlpackage" "$OUT_DIR/infer/"
cp "$OUT_DIR/tokenizer/"*.json "$OUT_DIR/infer/"
cp "$OUT_DIR/tokenizer/tokenizer.model" "$OUT_DIR/infer/"
cp -r "$OUT_DIR/combine_streams/gemma3n_combine_streams.mlpackage" "$OUT_DIR/infer/"

python tests/dev/test_gemma3n_ane_chat_fixed.py \
  --bundle "$OUT_DIR/infer" \
  --prompt "The capital of France is" \
  --max-new-tokens 16 \
  --context-length 512 \
  --verbose
```

---

## Where Exports Land

- Infer bundle: `/tmp/gemma3n-output/infer/`
- LM head: `/tmp/gemma3n-output/lm_head/gemma3n_lm_head.mlpackage`
- Tokenizer: `/tmp/gemma3n-output/tokenizer/` (copy into infer bundle)
- Combine streams: `/tmp/gemma3n-output/combine_streams/gemma3n_combine_streams.mlpackage` (copy into infer bundle)

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

### 1. Degenerate Output (RESOLVED)

**Root Causes**:
1. Missing KV cache sharing for shared-KV layers in `gemma3n_model.py`.
2. Incorrect NumPy RoPE packing (`np.repeat`) which mismatched the PyTorch cache.
3. Lexicographic concatenation of split logits.

**Status**: All resolved. CoreML now generates the expected completion.

### 2. Export OOM Killed (RESOLVED)

**Status**: RESOLVED on system with 96GB RAM, 167GB disk.

### 3. HF Model Load Requires timm

The full HuggingFace model requires `timm` library for vision tower. Resolved with `pip install timm`.

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

### Immediate Priority: Debug PyTorch Implementation

The ANEMLL Gemma3n PyTorch model produces wrong output. Need to compare with HuggingFace layer-by-layer:

1. **Compare embeddings**:
   - Check `embed_tokens` scaling (√hidden_size)
   - Check PLE embeddings scaling (√256)
   - Compare inputs_embeds output

2. **Compare layer processing**:
   - Compare AltUp predict/correct/forward stages
   - Compare LAUREL block output
   - Compare attention output
   - Compare FFN output

3. **Compare final stages**:
   - Compare `_combine_streams` (AltUp unembed)
   - Compare final norm
   - Compare LM head output

### Debug Commands

```bash
# Compare HF vs ANEMLL at each stage
python tests/dev/test_gemma3n_layer_comparison.py \
  --model google/gemma-3n-E2B-it \
  --prompt "The capital of France is"
```

### After PyTorch Fix

1. Re-export CoreML models
2. Test chat output
3. Add LUT quantization
4. Performance profiling

### Completed

- ✅ Export pipeline works (all parts export successfully)
- ✅ KV cache state sharing between chunks (using read_state/write_state)
- ✅ CoreML numerically matches PyTorch source

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

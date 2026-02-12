# D-DAP: Deduplication-Aware Pipeline for combine_models.py

**D-DAP** = Deduplication-Aware Pipeline

## Problem Statement

When `combine_models.py` calls `ct.utils.save_multifunction()` to merge infer + prefill
(or 4-function rotate variants) into a single multifunction package, CoreMLTools' dedup
pass only shares weight blobs that are **byte-identical**. Our investigation shows:

- Infer vs prefill within the same context: only **~60%** of weights are byte-identical
- Cross-context same-mode (e.g., infer ctx512 vs infer ctx2048): **~82%** identical
- The remaining "different" weights encode the **exact same dequantized values**
  (cosine similarity = 1.0, mean_diff ~3e-8) via different LUT encodings

Root cause: MIL optimization passes produce graph-shape-dependent fp16 constants, causing
k-means to converge to different LUT centroids + index assignments for identical weights.

## Impact

| Combination | Identity Rate | Wasted Space |
|------------|---------------|-------------|
| 2-func (infer+prefill) | 60% | ~40% of weight bytes duplicated |
| 2-func split-rotate | 60% per file | ~40% per file |
| 4-func (infer+infer_rot+prefill+prefill_rot) | ~50-60% | ~40-50% duplicated |
| xstates cross-context infer-only | 82% | ~18% duplicated |
| xstates cross-context prefill-only | 82% | ~18% duplicated |

For a typical 1.5B model with 3 chunks, the wasted space per chunk is ~120-150 MB for
2-function combines, scaling with model size.

## Solution: Surgical Weight Replacement

**Utility**: `anemll/utils/dedup_weights.py`

Before calling `save_multifunction`, replace palettized weight blobs in non-anchor models
with the anchor model's blobs where dequantized values are verified identical. This forces
byte-identical const blobs so CoreMLTools dedup shares them automatically.

### Verified Safe

De-LUT verification confirms:
- `value = lut[group, index]` produces identical results (cos=1.0) across all mismatched tensors
- Replacement introduces **zero** additional quantization error
- We are standardizing on one of several equivalent LUT encodings

## Implementation Plan

### Phase 1: Core Utility (DONE)

- [x] `anemll/utils/dedup_weights.py` created with:
  - `find_replaceable_weights()` — compare anchor vs target, verify via dequantization
  - `_apply_replacements_to_mlpackage()` — load, replace const ops, save
  - `prepare_dedup_sources()` — context manager for combine pipeline integration
  - Standalone CLI for testing: `--anchor`, `--target`, `--dry-run`

### Phase 2: Standalone Validation

Test the utility on a single-context combine to verify correctness before integration.

#### 2.1 Dry-run comparison

```bash
source env-anemll/bin/activate

# Test infer vs prefill for ctx2048 chunk02
python3 -m anemll.utils.dedup_weights \
  --anchor /Volumes/Models/ANE/vibethinker_1.5b_ctx2048_L6_4_hybrid/qwen25_FFN_lut6_chunk_02of03.mlpackage \
  --target /Volumes/Models/ANE/vibethinker_1.5b_ctx2048_L6_4_hybrid/qwen25_prefill_lut6_chunk_02of03.mlpackage \
  --dry-run -v
```

Expected: reports N weight pairs replaceable, 0 skipped (cos < threshold).

#### 2.2 Full replacement + size comparison

```bash
# Create deduped prefill
python3 -m anemll.utils.dedup_weights \
  --anchor /Volumes/Models/ANE/vibethinker_1.5b_ctx2048_L6_4_hybrid/qwen25_FFN_lut6_chunk_02of03.mlpackage \
  --target /Volumes/Models/ANE/vibethinker_1.5b_ctx2048_L6_4_hybrid/qwen25_prefill_lut6_chunk_02of03.mlpackage \
  --output /tmp/claude/prefill_deduped.mlpackage -v

# Compare sizes
du -sh /Volumes/Models/ANE/vibethinker_1.5b_ctx2048_L6_4_hybrid/qwen25_prefill_lut6_chunk_02of03.mlpackage
du -sh /tmp/claude/prefill_deduped.mlpackage
```

#### 2.3 Combine with deduped sources + measure

```bash
# Combine original (baseline)
python3 -c "
import coremltools as ct
desc = ct.utils.MultiFunctionDescriptor()
desc.add_function('infer.mlpackage', 'main', 'infer')
desc.add_function('prefill.mlpackage', 'main', 'prefill')
desc.default_function_name = 'infer'
ct.utils.save_multifunction(desc, '/tmp/claude/combined_original.mlpackage')
"

# Combine deduped
python3 -c "
import coremltools as ct
desc = ct.utils.MultiFunctionDescriptor()
desc.add_function('infer.mlpackage', 'main', 'infer')
desc.add_function('/tmp/claude/prefill_deduped.mlpackage', 'main', 'prefill')
desc.default_function_name = 'infer'
ct.utils.save_multifunction(desc, '/tmp/claude/combined_deduped.mlpackage')
"

# Compare
du -sh /tmp/claude/combined_original.mlpackage /tmp/claude/combined_deduped.mlpackage
```

Expected: combined_deduped should be ~35-40% smaller than combined_original.

#### 2.4 Verify combined model correctness

```bash
# Load and run inference with deduped combined model
python3 tests/chat.py --meta /path/to/meta_with_deduped.yaml
```

### Phase 3: Integration into combine_models.py

Add `--dedup-weights` flag to `combine_models.py` that activates surgical dedup before
every `save_multifunction` call.

#### 3.1 Integration points (7 call sites)

| Function | Line | Functions Combined | Priority |
|----------|------|-------------------|----------|
| `combine_chunks()` | 174 | infer + prefill | HIGH |
| `combine_chunks_split_rotate()` | 287 | infer + prefill | HIGH |
| `combine_chunks_split_rotate()` | 302 | infer_rot + prefill_rot | HIGH |
| `combine_chunks_gemma3()` | 404 | 4 functions | MEDIUM |
| `combine_monolithic()` | 524 | infer + prefill | HIGH |
| `combine_monolithic_rotate()` | 627 | 4 functions | MEDIUM |
| `combine_models_custom()` | 82 | user-specified | LOW |

#### 3.2 Integration pattern

Each `save_multifunction` call site follows the same pattern. The integration wraps
the existing descriptor creation:

```python
# BEFORE (current code):
desc = ct.utils.MultiFunctionDescriptor()
desc.add_function(ffn_path, "main", "infer")
desc.add_function(prefill_path, "main", "prefill")
desc.default_function_name = "infer"
ct.utils.save_multifunction(desc, temp_path)

# AFTER (with dedup):
if dedup_weights:
    from anemll.utils.dedup_weights import prepare_dedup_sources
    sources = [(ffn_path, "main", "infer"), (prefill_path, "main", "prefill")]
    with prepare_dedup_sources(sources, verbose=verbose) as deduped:
        desc = ct.utils.MultiFunctionDescriptor()
        for path, src_fn, tgt_fn in deduped:
            desc.add_function(path, src_fn, tgt_fn)
        desc.default_function_name = "infer"
        ct.utils.save_multifunction(desc, temp_path)
else:
    # Original code path (unchanged)
    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(ffn_path, "main", "infer")
    desc.add_function(prefill_path, "main", "prefill")
    desc.default_function_name = "infer"
    ct.utils.save_multifunction(desc, temp_path)
```

#### 3.3 CLI changes

```python
# In parse_args():
parser.add_argument('--dedup-weights', action='store_true',
    help='Enable surgical weight deduplication before combining. '
         'Reduces combined package size by ~30-40%% for infer+prefill combines.')
parser.add_argument('--dedup-verbose', action='store_true',
    help='Show per-weight dedup details.')
parser.add_argument('--dedup-cos-threshold', type=float, default=0.9999,
    help='Cosine similarity threshold for dedup (default: 0.9999)')
```

#### 3.4 convert_model.sh changes

```bash
# In step 5 (combine), add --dedup-weights flag:
run_step 5 "Combining Models" "python3 \"$PROJECT_ROOT/anemll/utils/combine_models.py\" \
    --chunk $NUM_CHUNKS \
    $LUT2_PARAM \
    $SPLIT_ROTATE_FLAG \
    $GEMMA3_FLAG \
    --dedup-weights \
    --prefix \"$PREFIX\" \
    --input \"$OUTPUT_DIR\" \
    --output \"$OUTPUT_DIR\""
```

### Phase 4: Integration into xstates combine

Apply the same pattern to `tests/dev/combine_infer_context_exports.py`:

```python
# Before save_multifunction in split mode (infer package):
sources = [(str(path), src_fn, f"infer_ctx{ctx}") for ctx, path, _ in infer_sources]
with prepare_dedup_sources(sources, verbose=verbose) as deduped:
    infer_desc = ct.utils.MultiFunctionDescriptor()
    for path, src_fn, tgt_fn in deduped:
        infer_desc.add_function(path, src_fn, tgt_fn)
    ...
```

Expected improvement for xstates:
- Infer-only package: dedup ratio 0.85 -> ~0.25 (5 contexts)
- Prefill-only package: dedup ratio 0.85 -> ~0.25 (5 contexts)

### Phase 5: Performance Optimization (Optional)

If the dedup step is too slow (loading MIL programs + dequantization):

1. **Skip verification for known-safe models**: Add `--dedup-no-verify` that replaces
   all palettized weights without dequantization check (safe for same-architecture models)
2. **Parallel loading**: Load anchor and target MIL programs in parallel
3. **Caching**: Cache anchor weights across chunks (same model, different chunk indices)

## Validated Results (2026-02-10)

### Single-context combine: ctx512 chunk02 infer+prefill

**Test**: VibeThinker-1.5B, ctx512, chunk 02 (layers 10-18), LUT6

```
Dry-run: 54 weight pairs replaceable, 9 already identical, 0 skipped
All replacements verified: cosine similarity = 1.000000, mean_diff ~3e-8
```

| Metric | Before (original) | After (D-DAP) | Improvement |
|--------|-------------------|---------------|-------------|
| Combined package size | 523 MB | 309 MB | **-41%** |
| Dedup ratio | 0.847 | 0.500 | **41% smaller** |
| Single source size | 308 MB | 308 MB | (unchanged) |
| Ops replaced | — | 98 | 54 weight pairs |

The deduped combined (309 MB) is almost exactly 1x the single infer package (308 MB),
meaning nearly all weight blobs are now shared. The ~1 MB overhead is from non-weight
constants (causal masks, KV cache indices, graph structure) that differ between infer/prefill.

**Key technical detail**: Must use `PassPipeline.EMPTY` when re-saving the modified MIL
program, otherwise the default 92-pass optimization pipeline re-runs constant folding
which undoes the surgical replacements.

### Expected: xstates 5-context combine (same-mode)

| Metric | Before | After (projected) | Improvement |
|--------|--------|-------------------|-------------|
| Per-chunk package size | ~2.6 GB | ~0.7 GB | ~73% smaller |
| Weight identity rate | 82% | ~98%+ | +16% |
| Dedup ratio | ~0.85 | ~0.22 | 74% smaller |

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Dequantized values not actually identical | Very Low | cos_threshold=0.9999 with full dequant verification |
| MIL program modification breaks save | Low | Uses standard MIL API; fallback to original if save fails |
| Slow for large models | Medium | Add --dedup-no-verify for known-safe models; cache anchor |
| CoreMLTools version breaks MIL API | Low | Guarded import; graceful fallback to no-dedup path |

## Files

| File | Status | Purpose |
|------|--------|---------|
| `anemll/utils/dedup_weights.py` | CREATED | Core utility + CLI |
| `anemll/utils/combine_models.py` | TO MODIFY | Add --dedup-weights integration |
| `anemll/utils/convert_model.sh` | TO MODIFY | Pass --dedup-weights in step 5 |
| `tests/dev/combine_infer_context_exports.py` | TO MODIFY | xstates integration |

## Acceptance Criteria

1. Dry-run reports correct replacement counts for known test cases
2. Combined package with dedup is measurably smaller than without
3. Deduped combined model produces identical inference output
4. All existing combine modes work with --dedup-weights (chunked, monolithic, split-rotate, gemma3)
5. Fallback: without --dedup-weights, behavior is 100% unchanged
6. No regressions in convert_model.sh end-to-end pipeline

---

*Created: 2026-02-10*

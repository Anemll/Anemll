# ANEMLL-Dedup: Surgical Weight Deduplication for CoreML

## Overview

ANEMLL-Dedup is a surgical weight deduplication utility that reduces combined CoreML multifunction model sizes by ~50%. It works by replacing palettized weight blobs (LUT + indices) in non-anchor functions with the anchor function's blobs when the dequantized values are semantically identical.

## The Problem

When CoreML converts models with LUT palettization, the MIL optimization pipeline (constant folding, fp16 casting) produces microscopically different fp16 representations depending on graph shape (context length, sequence length). K-means then converges to different LUT centroids and index assignments that encode the **same** dequantized values.

CoreML's built-in dedup compares raw bytes, so these semantically-identical weights are not shared -- causing ~15-40% size bloat in multifunction packages.

## How It Works

1. **Load** anchor (e.g., infer) and target (e.g., prefill) MIL programs
2. **Extract** all const weight tensors from both
3. **Match** palettized weight pairs by normalized key name
4. **Verify** via dequantization: `value = lut[group, index]` proves identical semantics
5. **Replace** target's const op values with anchor's byte-exact blobs (using `_sym_val.val`)
6. **Re-save** modified MIL with `PassPipeline.EMPTY` (prevents default passes from undoing replacements)
7. **Combine** via `save_multifunction` -- CoreML's dedup now sees byte-identical blobs and shares them

## Results

### Monolithic Model (Qwen 2.5 0.5B, LUT6, ctx2048)

| Metric | Standard CoreML | ANEMLL-Dedup |
|--------|----------------|--------------|
| Combined size | 862 MB | **472 MB** |
| Saving vs naive | 8.5% | **49.9%** |

### Chunked Model (VibeThinker 1.5B, ctx512, chunk02, LUT6)

| Metric | Standard CoreML | ANEMLL-Dedup |
|--------|----------------|--------------|
| Combined size | 523 MB | **309 MB** |
| Saving vs naive | 15% | **41%** |

The deduped combined package is essentially the same size as a single function -- the theoretical optimum.

## Usage

### Automatic (Default)

ANEMLL-Dedup is **enabled by default** in the conversion pipeline. No action needed:

```bash
# Standard conversion -- dedup runs automatically in step 5
./anemll/utils/convert_model.sh --model <path> --output <dir>

# Monolithic conversion -- dedup runs automatically in step 3
./anemll/utils/convert_monolith.sh --model <path> --output <dir>
```

### Skip Dedup

To skip dedup (e.g., for debugging or if you want faster combine at the cost of larger output):

```bash
# Chunked conversion
./anemll/utils/convert_model.sh --model <path> --output <dir> --skip-anemll-dedup

# Monolithic conversion
./anemll/utils/convert_monolith.sh --model <path> --output <dir> --skip-anemll-dedup

# Direct combine_models.py usage
python3 anemll/utils/combine_models.py --monolithic --lut 6 --prefix qwen25 \
    --skip-anemll-dedup --input <dir> --output <dir>
```

### Standalone CLI

For advanced use, `dedup_weights.py` can be run standalone:

```bash
# Dry-run: see what would be replaced
python3 -m anemll.utils.dedup_weights \
    --anchor /path/to/infer.mlpackage \
    --target /path/to/prefill.mlpackage \
    --dry-run --verbose

# With diagnostics output
python3 -m anemll.utils.dedup_weights \
    --anchor /path/to/infer.mlpackage \
    --target /path/to/prefill.mlpackage \
    --diagnostics --verbose

# Strict mode (require max_abs <= 0.01)
python3 -m anemll.utils.dedup_weights \
    --anchor /path/to/infer.mlpackage \
    --target /path/to/prefill.mlpackage \
    --strict --output /path/to/deduped_prefill.mlpackage
```

### Python API

```python
from anemll.utils.dedup_weights import prepare_dedup_sources
import coremltools as ct

sources = [
    (ffn_path, "main", "infer"),
    (prefill_path, "main", "prefill"),
]

with prepare_dedup_sources(sources, verbose=True) as deduped:
    desc = ct.utils.MultiFunctionDescriptor()
    for path, src_fn, tgt_fn in deduped:
        desc.add_function(path, src_fn, tgt_fn)
    desc.default_function_name = "infer"
    ct.utils.save_multifunction(desc, output_path)
```

## Acceptance Criteria

ANEMLL-Dedup uses multi-metric acceptance to validate each weight replacement:

### Relaxed Mode (Default)

- **Cosine similarity** >= 0.9999
- **Mean absolute difference** <= 0.001

This is safe because max_abs outliers (0.01-0.014) in multi-million element tensors are k-means centroid placement artifacts, not real weight divergence.

### Strict Mode (`--strict`)

- **Cosine similarity** >= 0.9999
- **Max absolute difference** <= 0.01
- **Mean absolute difference** <= 0.001

Use strict mode if you want to be conservative and reject borderline weight pairs.

## Safety Features

### Preflight Checks

Before any replacements, ANEMLL-Dedup validates:
- **I/O signature compatibility**: Input/output names and shapes match
- **Weight count ratio**: Anchor and target have similar tensor counts (>0.8 ratio)
- **LUT configuration**: Palettized groups and centroids match

Disable with `--no-preflight` (not recommended).

### Diagnostics

Per-replacement diagnostics are available with `--diagnostics`:
- Tensor class (palettized_indices, palettized_lut, dense_weight, etc.)
- Reason code (identical, deq_close, rejected_threshold, etc.)
- Bytes saved per replacement
- Cosine similarity, max_abs, mean_abs metrics

## Technical Details

### Key Implementation Notes

- **`_sym_val.val`**: The writable path for MIL const op modification (`Var.val` is read-only)
- **`PassPipeline.EMPTY`**: Must be used when re-saving modified MIL. The default 92-pass pipeline re-runs constant folding which undoes the surgical replacements
- **Paired replacement**: Both `_palettized_indices` and `_palettized_lut` are always replaced together
- **Temp file workflow**: Modified mlpackages are saved to temp, then moved to final location

### File Locations

- **Core utility**: `anemll/utils/dedup_weights.py`
- **Integration**: `anemll/utils/combine_models.py` (all combine functions)
- **Shell scripts**: `anemll/utils/convert_model.sh`, `anemll/utils/convert_monolith.sh`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Output size same as without dedup | Verify `anemll-dedup weight dedup: enabled` in logs |
| Size regresses after re-save | Verify `PassPipeline.EMPTY` is used (not default pipeline) |
| Preflight check fails | Models may be from different architectures or chunk configs |
| ImportError for dedup_weights | Ensure `anemll` package is on PYTHONPATH |
| Slow combine step | Dedup adds ~30s per chunk for loading MIL + dequantization verification |

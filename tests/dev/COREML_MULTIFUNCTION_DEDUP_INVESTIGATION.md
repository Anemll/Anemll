# CoreML MultiFunction Dedup Investigation

## Problem Statement

When combining context-specific exports into a single multi-function chunk package, output size appears to grow more than expected. We need a reproducible way to determine whether `coremltools.utils.save_multifunction(...)` deduplicates shared constants across added functions.

Observed symptom:
- Adding many `infer_ctx*` and `prefill_ctx*` functions increases chunk package size close to additive growth.
- This suggests no (or limited) cross-function dedup in the saved package.

## Scope

Model family:
- `vibethinker_1.5b_ctx{context}_L6_4_hybrid` for contexts `512,1024,2048,3072,4096`.

Focus:
- Compare package size and compile behavior for:
1. Monolithic combined package (infer+prefill together).
2. Split combined packages (infer-only package + prefill-only package).
3. With/without alias functions (`infer`, `prefill`).

## Repro Matrix

Use `--no-compile` first to compare `.mlpackage` size only.

## Evaluation Step 1 (Export Provenance Control)

Before comparing combine modes, run the same combine matrix on two export sources:

1. `separate-runs` (current): each context exported in separate conversion runs/folders.
2. `single-run` (new control): all contexts exported in one export run/process from the same source model object/checkpoint load.

Keep everything else identical:
- same contexts (`512,1024,2048,3072,4096`)
- same chunk count (`3`)
- same LUT/palette settings
- same combine flags/modes

Goal:
- determine whether export provenance (separate files/runs vs one shared run) changes dedup behavior in `save_multifunction`.
- if size gap exists, treat provenance as a first-order factor in the dedup investigation.

### A) Monolithic (default)

```bash
scripts/combine_vibethinker_all_context_functions.sh \
  --contexts "512 1024 2048 3072 4096" \
  --max-context 4096 \
  --context-root /Volumes/Models/ANE \
  --context-name-template "vibethinker_1.5b_ctx{context}_L6_4_hybrid" \
  --output /Volumes/Models/ANE/vibethinker_1.5b_xstates_mono \
  --num-chunks 3 \
  --clean-output \
  --no-compile \
  --force
```

### B) Split infer/prefill

```bash
scripts/combine_vibethinker_all_context_functions.sh \
  --contexts "512 1024 2048 3072 4096" \
  --max-context 4096 \
  --context-root /Volumes/Models/ANE \
  --context-name-template "vibethinker_1.5b_ctx{context}_L6_4_hybrid" \
  --output /Volumes/Models/ANE/vibethinker_1.5b_xstates_split \
  --num-chunks 3 \
  --split-infer-prefill \
  --clean-output \
  --no-compile \
  --force
```

### C) Split + no aliases

```bash
scripts/combine_vibethinker_all_context_functions.sh \
  --contexts "512 1024 2048 3072 4096" \
  --max-context 4096 \
  --context-root /Volumes/Models/ANE \
  --context-name-template "vibethinker_1.5b_ctx{context}_L6_4_hybrid" \
  --output /Volumes/Models/ANE/vibethinker_1.5b_xstates_split_noalias \
  --num-chunks 3 \
  --split-infer-prefill \
  --no-alias-functions \
  --clean-output \
  --no-compile \
  --force
```

## Measurement Checklist

1. Package size:

```bash
du -sh /Volumes/Models/ANE/vibethinker_1.5b_xstates_mono/*.mlpackage
du -sh /Volumes/Models/ANE/vibethinker_1.5b_xstates_split/*.mlpackage
du -sh /Volumes/Models/ANE/vibethinker_1.5b_xstates_split_noalias/*.mlpackage
```

2. Manifest/function layout:

```bash
cat /Volumes/Models/ANE/vibethinker_1.5b_xstates_mono/state_transition_manifest.yaml
cat /Volumes/Models/ANE/vibethinker_1.5b_xstates_split/state_transition_manifest.yaml
cat /Volumes/Models/ANE/vibethinker_1.5b_xstates_split_noalias/state_transition_manifest.yaml
```

3. Optional compile-time comparison:

```bash
scripts/combine_vibethinker_all_context_functions.sh ... --output <out> --force
```

then compare `.mlmodelc` total size and compile duration.

## Hypotheses To Validate

1. `save_multifunction` does not deduplicate large constants across functions sourced from different input packages.
2. Alias functions have negligible size impact; most growth comes from context function bodies/constants.
3. Split mode may improve practical loading/compile behavior even if total bytes across both outputs remain similar.

## Expected Outputs

- `state_transition_manifest.yaml` should clearly indicate:
  - `split_infer_prefill`
  - `no_alias_functions`
  - per-chunk infer/prefill output names (in split mode)
- `meta.yaml` should include:
  - `state_transition_split_infer_prefill`
  - `state_transition_no_alias_functions`
  - (split mode) `ffn_prefill`, output-base tags.

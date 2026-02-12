# vibethinker_1.5b_xstates_split_noalias: Clean Creation Plan (D-DAP Updated)

## Objectives

1. Build one combined xstates model set from verified context folders:
   - `vibethinker_1.5b_ctx512_L6_4_hybrid`
   - `vibethinker_1.5b_ctx1024_L6_4_hybrid`
   - `vibethinker_1.5b_ctx2048_L6_4_hybrid`
   - `vibethinker_1.5b_ctx3072_L6_4_hybrid`
   - `vibethinker_1.5b_ctx4096_L6_4_hybrid`
2. Combine by chunk index, with **infer + prefill merged into the same package per chunk**.
3. Keep context-routed functions (`infer_ctx*`, `prefill_ctx*`) and disable aliases.
4. Apply D-DAP dedup path to maximize byte-identical sharing before `save_multifunction`.
5. Compile only final combined artifacts.

---

## What We Now Know

1. Raw CoreML dedup is partial for these exports because palettized blobs differ byte-wise across contexts.
2. De-LUT verification shows mismatched LUT/index blobs are semantically equivalent in many cases.
3. Surgical replacement (D-DAP) is effective:
   - validated sample: `523 MB -> 309 MB` for chunk02 infer+prefill (`-41%`)
   - dedup ratio improved from `0.847 -> 0.500`
   - replacements validated with cosine `1.0` on dequantized tensors.
4. Re-saving modified MIL must use `PassPipeline.EMPTY`; otherwise default optimization passes can undo replacements.

Reference:
- `anemll/utils/dedup_weights.py`
- `tests/dev/DEDUP_COMBINE_MODELS_PLAN.md`

---

## Recommended Strategy

Use **monolithic per-chunk combine**:
- `chunk_01`: one package with infer+prefill functions for all contexts
- `chunk_02`: one package with infer+prefill functions for all contexts
- `chunk_03`: one package with infer+prefill functions for all contexts

Do not split infer/prefill packages for the production xstates build.

---

## Preconditions

1. Verified context folders exist and include expected chunk artifacts.
2. Temporary compile path exists and has enough free space:
   - `/Volumes/Models/ANE/tmp_coreml_compile`
3. Causal mask audit passes before combine:
   - `tests/dev/audit_causal_mask_shapes.py`
4. D-DAP integration is available in combine path:
   - through `prepare_dedup_sources(...)`
   - or an equivalent CLI flag if exposed in this branch.

---

## Build Steps

### 1) Preflight check

```bash
for C in 512 1024 2048 3072 4096; do
  D="/Volumes/Models/ANE/vibethinker_1.5b_ctx${C}_L6_4_hybrid"
  echo "== ctx$C =="
  ls -1 "$D" | rg '^qwen25_(FFN_attn_fp32_chunk_01of03|FFN_lut6_chunk_0[123]of03|prefill_attn_fp32_chunk_01of03|prefill_lut6_chunk_0[123]of03|embeddings|lm_head)'
done
```

### 2) Causal mask audit (must pass)

```bash
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/env-anemll/bin/python3 \
  tests/dev/audit_causal_mask_shapes.py \
  --contexts-root /Volumes/Models/ANE \
  --name-template "vibethinker_1.5b_ctx{context}_L6_4_hybrid" \
  --contexts 512 1024 2048 3072 4096
```

Fast path (recommended for large runs):
- Use spec-only audit (reads `Data/com.apple.CoreML/model.mlmodel` via `ct.utils.load_spec`) to avoid slow `MLModel(...)` loads per package.
- This verifies causal-mask contract quickly without compiling/loading runtime proxies.

```bash
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/env-anemll/bin/python3 - <<'PY'
from pathlib import Path
import coremltools as ct
import re
contexts=[512,1024,2048,3072,4096]
root=Path('/Volumes/Models/ANE')
tmpl='vibethinker_1.5b_ctx{context}_L6_4_hybrid'
fails=0
for ctx in contexts:
    d=root / tmpl.format(context=ctx)
    for pkg in sorted([p for p in d.glob('*.mlpackage') if re.search(r'_chunk_\\d+of\\d+\\.mlpackage$', p.name)]):
        spec=ct.utils.load_spec(str(pkg/'Data/com.apple.CoreML/model.mlmodel'))
        cm=[list(i.type.multiArrayType.shape) for i in spec.description.input if i.name=='causal_mask']
        if not cm:
            continue
        if cm[0][-1] != ctx:
            fails += 1
            print('FAIL', ctx, pkg.name, cm[0])
print('STATUS: PASS' if fails==0 else f'STATUS: FAIL ({fails})')
raise SystemExit(0 if fails==0 else 1)
PY
```

### 3) Combine by chunk, merged infer+prefill (no split)

```bash
TMPDIR=/Volumes/Models/ANE/tmp_coreml_compile \
TMP=/Volumes/Models/ANE/tmp_coreml_compile \
TEMP=/Volumes/Models/ANE/tmp_coreml_compile \
scripts/combine_vibethinker_all_context_functions.sh \
  --contexts "512 1024 2048 3072 4096" \
  --max-context 4096 \
  --context-root /Volumes/Models/ANE \
  --context-name-template "vibethinker_1.5b_ctx{context}_L6_4_hybrid" \
  --output /Volumes/Models/ANE/vibethinker_1.5b_xstates_split_noalias \
  --num-chunks 3 \
  --no-alias-functions \
  --infer-kind FFN \
  --infer-chunk1-kind FFN_attn_fp32 \
  --prefill-kind prefill \
  --clean-output \
  --no-compile \
  --force
```

Notes:
1. Intentionally omit `--split-infer-prefill`.
2. Enable dedup mode if this branch exposes a switch for it (for example, `--dedup-weights`).
3. `--copy-source-chunks` is debug-only and should be avoided for production output size.

### 4) Validate outputs and metadata

```bash
OUT=/Volumes/Models/ANE/vibethinker_1.5b_xstates_split_noalias
for i in 01 02 03; do
  test -e "$OUT/qwen25_FFN_PF_statex_chunk_${i}of03.mlpackage" || { echo "missing chunk $i"; exit 1; }
done
test -e "$OUT/meta.yaml" || { echo "missing meta.yaml"; exit 1; }
test -e "$OUT/state_transition_manifest.yaml" || { echo "missing state_transition_manifest.yaml"; exit 1; }
echo "combined chunks + metadata present"
```

### 5) Compile final combined outputs only

```bash
OUT=/Volumes/Models/ANE/vibethinker_1.5b_xstates_split_noalias
for pkg in \
  "$OUT"/qwen25_FFN_PF_statex_chunk_0{1,2,3}of03.mlpackage \
  "$OUT"/qwen25_embeddings.mlpackage \
  "$OUT"/qwen25_lm_head_lut6.mlpackage
do
  [ -e "$pkg" ] || continue
  stem="$(basename "${pkg%.mlpackage}")"
  rm -rf "$OUT/${stem}.mlmodelc"
  xcrun coremlcompiler compile "$pkg" "$OUT"
done
```

### 6) Smoke test

```bash
TMPDIR=/Volumes/Models/ANE/tmp_coreml_compile \
TMP=/Volumes/Models/ANE/tmp_coreml_compile \
TEMP=/Volumes/Models/ANE/tmp_coreml_compile \
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/env-anemll/bin/python3 \
tests/dev/state_transition_growing_inference.py \
  --meta /Volumes/Models/ANE/vibethinker_1.5b_xstates_split_noalias/meta.yaml \
  --prompt "Tell me about Apple Neural Engine" \
  --max-tokens 96 \
  --prefill-mode token-infer \
  --compute-unit CPU_AND_NE \
  --sampling-mode greedy \
  --seed 123 \
  --progress-stream stdout
```

---

## Acceptance Criteria

1. Exactly 3 combined FFN_PF chunk packages exist (`chunk_01..03`) with infer+prefill functions inside each.
2. No split infer/prefill output packages are used for production.
3. Dedup path is enabled and measurable in logs.
4. Combined package sizes are materially below non-dedup baseline.
5. Runtime loads all contexts and generates coherent output.

---

## Troubleshooting

1. If output size regresses sharply:
   - verify dedup path is active,
   - verify modified MIL save uses `PassPipeline.EMPTY`.
2. If chunk1 quality regresses:
   - verify infer chunk1 uses `FFN_attn_fp32`,
   - verify prefill path policy (`token-infer` if batch-prefill chunk1 FP32 is unavailable).
3. If combine appears stuck:
   - wait for MIL save passes; chunk saves can be long.
4. If compiled model load fails with function-name errors:
   - check package is MLProgram multifunction,
   - recompile after freeing disk space.
5. If the causal-mask audit is too slow or appears stalled:
   - use the spec-only fast path in Step 2 (no runtime model loading).

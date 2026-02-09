# build_ctx_model

`scripts/build_ctx_model` is a step-driven wrapper for state-transition context builds.

Current defaults target VibeThinker paths/naming, but the script is intended to be reused for other models by overriding paths/templates/options.

It is designed to run in three explicit phases:

1. Build max context (4096 by default)
2. Build remaining contexts using static reuse from max context
3. Combine all contexts into one multi-context output

## Quick Start

Run from repo root:

```bash
scripts/build_ctx_model --step 1
scripts/build_ctx_model --step 2
scripts/build_ctx_model --step 3
```

Defaults:

- `state root`: `/Volumes/Models/ANE/vibethinker_1.5b_state_transition`
- `context root`: `/Volumes/Models/ANE/vibethinker_1.5b_state_transition/_contexts`
- `contexts`: `512 1024 2048 3072 4096`
- `max context`: `4096`

## Step Behavior

### Step 1

Builds max context with:

```bash
bash scripts/rebuild_vibethinker_hybrid.sh \
  --context 4096 \
  --output <context-root>/vibethinker_1.5b_ctx4096_fp16_hybrid \
  --force-clean \
  --skip-smoke
```

Includes FP32 hybrid patching (`chunk_01` first-layer attention split).
Also exports a standalone chunk-1 artifact:
`qwen25_FFN_attn_fp32_chunk_01of03.mlpackage`.

### Step 2

Loops over non-max contexts and runs:

```bash
bash scripts/rebuild_vibethinker_hybrid.sh \
  --context <C> \
  --output <context-root>/vibethinker_1.5b_ctx<C>_fp16_hybrid \
  --reuse-static-from <context-root>/vibethinker_1.5b_ctx4096_fp16_hybrid \
  --reuse-infer-only \
  --force-clean \
  --skip-smoke
```

Current behavior in `rebuild_vibethinker_hybrid.sh`:

- Reuses embeddings + lm_head from max context
- Runs conversion part 3 only (`ffn_infer`)
- Applies FP32 hybrid patch for each context output

### Step 3

Combines all context outputs:

```bash
bash scripts/combine_vibethinker_infer_contexts.sh \
  --contexts "512 1024 2048 3072 4096" \
  --max-context 4096 \
  --context-root <context-root> \
  --infer-kind FFN \
  --prefill-kind prefill \
  # If requested kind is missing, combiner falls back automatically \
  --clean-output \
  --output <output> \
  --force
```

Step 3 does not generate FP32 by itself. It combines whatever step 1/2 produced.

Step 3 now enforces that `<output>/meta.yaml` exists and contains:

- `state_transition_infer_contexts`
- `state_transition_prefill_context`

If these are missing, step 3 fails.

## Common Options

- `--state-root <dir>`: override base root
- `--context-root <dir>`: override per-context folder root
- `--output <dir>`: override combined output dir
- `--contexts "..."`: override context list
- `--max-context N`: set prefill/max context
- `--step all`: run 1,2,3 in sequence
- `--rebuild-extra-args "..."`: append args to rebuild calls
- `--combine-extra-args "..."`: append args to combine call

## Step/Part Labels for `convert_model.sh --only X`

Used in logs:

- `1`: embeddings
- `2`: lm_head
- `3`: ffn_infer
- `4`: prefill
- `5`: combine
- `6`: compile
- `7`: meta_tokenizer
- `8`: smoke_test

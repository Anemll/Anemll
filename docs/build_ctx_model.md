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
  --force-clean \
  --skip-smoke
```

Current behavior in `rebuild_vibethinker_hybrid.sh`:

- Reuses embeddings + lm_head from max context
- Default (`scripts/build_ctx_model`): standard reuse flow (converter parts `3..7`) + hybrid patch
- Optional fast mode (`--step2-reuse-infer-only`): converter part `3` only + standalone FP32 chunk1 export

### Step 3

Combines all context outputs:

```bash
bash scripts/combine_vibethinker_infer_contexts.sh \
  --contexts "512 1024 2048 3072 4096" \
  --max-context 4096 \
  --context-root <context-root> \
  --infer-kind FFN_PF \
  --infer-chunk1-kind auto \
  --prefill-kind FFN_PF \
  --clean-output \
  --output <output> \
  --force
```

Step 3 does not generate FP32 by itself. It combines whatever step 1/2 produced.

Step 3 now enforces that `<output>/meta.yaml` exists and contains:

- `state_transition_infer_contexts`
- `state_transition_prefill_context`

If these are missing, step 3 fails.

Default Step 3 function layout per chunk:

- `infer_ctx{512,1024,2048,3072,4096}`
- `infer` alias (from max context infer)
- `prefill` alias (from max context prefill)

To include context-specific prefill functions in the same combined chunks:

```bash
scripts/build_ctx_model --step 3 \
  --combine-extra-args "--prefill-all-contexts"
```

To split outputs into separate infer/prefill packages per chunk:

```bash
scripts/build_ctx_model --step 3 \
  --combine-extra-args "--prefill-all-contexts --split-infer-prefill"
```

To remove compatibility aliases (`infer`/`prefill`) and keep context-routed names only:

```bash
scripts/build_ctx_model --step 3 \
  --combine-extra-args "--prefill-all-contexts --no-alias-functions"
```

or directly:

```bash
bash scripts/combine_vibethinker_all_context_functions.sh \
  --contexts "512 1024 2048 3072 4096" \
  --max-context 4096 \
  --context-root <context-root> \
  --output <output> \
  --num-chunks 3 \
  --clean-output \
  --force
```

All-context prefill adds per chunk:

- `prefill_ctx{512,1024,2048,3072,4096}`
- keeps `prefill` alias from max context for compatibility

Metadata tags written for all-context prefill combines:

- `state_transition_all_context_prefill: true`
- `state_transition_prefill_contexts: [...]`
- `state_transition_prefill_function_template: "prefill_ctx{context}"`
- `state_transition_combined_functions_layout: "infer_ctx+prefill_ctx+aliases"`

Additional tags for split/no-alias combines:

- `state_transition_split_infer_prefill: true|false`
- `state_transition_no_alias_functions: true|false`
- `state_transition_infer_default_function`
- `state_transition_prefill_default_function`
- (split mode) `ffn_prefill`, `state_transition_infer_output_base`, `state_transition_prefill_output_base`

## Common Options

- `--state-root <dir>`: override base root
- `--context-root <dir>`: override per-context folder root
- `--output <dir>`: override combined output dir
- `--contexts "..."`: override context list
- `--max-context N`: set prefill/max context
- `--step all`: run 1,2,3 in sequence
- `--step2-reuse-infer-only`: switch step 2 to fast infer-only rebuild mode
- `--rebuild-extra-args "..."`: append args to rebuild calls
- `--combine-extra-args "..."`: append args to combine call
  Example: `--combine-extra-args "--prefill-all-contexts --split-infer-prefill --no-alias-functions"`

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

# Lazy Weights Conversion Framework

## Purpose

This framework reduces conversion memory pressure and conversion time by loading only the checkpoint tensors required for the current conversion target:

- `part 1` (embeddings): only embeddings
- `part 3` (lm head): only lm head (+ embed fallback if tied)
- `part 2` / `part 2_prefill` chunked: only layer weights for the active chunk (+ final norm only when needed)

Primary implementation target: **Qwen3**.  
Design goal: reusable for other decoder-only models.

## What Was Added

### 1) Generic lazy loader module

File: `anemll/models/lazy_weights.py`

Key components:

- `WeightLoadSpec`: declarative key filter
  - include/exclude prefixes and exact keys
  - optional layer-range filter
  - optional prefix stripping
- `LazySafeTensorLoader`
  - indexes safetensors keys once
  - loads only matching tensors using `safetensors.safe_open(...).get_tensor(key)`
  - returns load stats (matched tensors, files touched, MB loaded, elapsed time)
- `compute_chunk_layer_range(...)`
  - remainder-aware chunk boundaries used by converters

### 2) Qwen model integration

File: `anemll/models/qwen_model.py`

- `QwenModel.load_pretrained_weights(...)` now supports:
  - `weight_spec`
  - `allow_partial`
  - shared `loader`
- `QwenForCausalLM.load_pretrained_weights(...)` now supports:
  - separate base vs lm-head loading
  - selective specs for each
  - partial load mode for part/chunk conversion

Also added **lazy module construction** for conversion:

- optional constructor flags to skip unused modules:
  - `build_embeddings`
  - `build_layers`
  - `build_norm`
  - `build_kv_cache`
  - `build_lm_head`
- optional `layer_range=(start, end)` to instantiate only chunk-local layers.

For example, `part=2` + `chunk-no=1` can build only layers `[0:4)` instead of all model layers.

### 3) Converter integration (part/chunk aware)

File: `anemll/ane_converter/qwen_converter.py`

Added:

- `--lazy-weights` / `--no-lazy-weights`
- `--chunk-no` (1-based chunk targeting)
- part-aware weight specs:
  - embeddings-only
  - lm-head-only
  - active layer range for chunked part2/prefill
- chunked lazy reload path:
  - for `part 2` / `part 2_prefill` with chunks, weights are reloaded per chunk before tracing
  - for multi-chunk runs (no `--chunk-no`), converter rebuilds a chunk-local model per chunk so only active layers are instantiated in memory

## Qwen3 Weight Selection Rules

| Conversion target | Weights loaded |
|---|---|
| `part=1` | `embed_tokens` only |
| `part=3` | `lm_head.weight` (or `embed_tokens.weight` fallback for tied head) |
| `part=2` chunk `k` | `layers[start:end]` for chunk `k`, plus final `norm` only on last chunk |
| `part=2_prefill` chunk `k` | `layers[start:end]` for chunk `k` (no final norm) |
| `all/full/monolithic` | full base + lm head |

## Usage

### Qwen3 4B, chunk auto, no quantization (recommended workflow)

1) Resolve chunk count automatically:

```bash
python anemll/utils/calc_chunk_split.py \
  --auto --max-chunk-mb 950 --lut 16 --overhead 1.10 \
  /path/to/qwen3-4b-snapshot
```

2) Convert parts with lazy loading:

```bash
python -m anemll.ane_converter.qwen_converter --model /path/to/snapshot --part 1 --lut none --lazy-weights --output /tmp/out --prefix qwen4b
python -m anemll.ane_converter.qwen_converter --model /path/to/snapshot --part 3 --lut none --lazy-weights --output /tmp/out --prefix qwen4b
python -m anemll.ane_converter.qwen_converter --model /path/to/snapshot --part 2 --chunk 9 --lut none --lazy-weights --output /tmp/out --prefix qwen4b
python -m anemll.ane_converter.qwen_converter --model /path/to/snapshot --part 2_prefill --chunk 9 --lut none --lazy-weights --output /tmp/out --prefix qwen4b
```

3) Optional targeted chunk conversion:

```bash
python -m anemll.ane_converter.qwen_converter --model /path/to/snapshot --part 2 --chunk 9 --chunk-no 1 --lut none --lazy-weights --output /tmp/out --prefix qwen4b
```

## Validation Summary (Qwen3-4B-Instruct-2507)

Environment tested on `2026-02-26` with local snapshot:

`/Users/anemll/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`

Key observations:

- Lazy chunk load example (`part=2`, chunk 1/9):
  - `matched 44/398 tensors`
  - `loaded 770.0 MB`
- Chunk-local module build example (`part=2`, chunk 1/9):
  - model init overrides: `build_embeddings=False`, `build_lm_head=False`, `layer_range=(0, 4)`
  - no lm-head module allocation during FFN/prefill conversion
- Multi-chunk lazy example (`part=2`, `chunk=3`, Qwen3-0.6B):
  - chunk-local rebuilds observed for each chunk range: `[0:10)`, `[10:19)`, `[19:28)`
  - max RSS stayed around `~2.20 GiB` during full 3-chunk export
- Non-lazy full load path for same command:
  - effectively loads all model tensors for base + lm-head
- Measured conversion impact (Qwen3-4B, chunk 1):
  - `part=2` lazy ctor-opt: `~12.46s`, max RSS `~3.01 GiB`
  - `part=2_prefill` lazy ctor-opt: `~15.14s`, max RSS `~3.01 GiB`
- FFN and prefill chunk parity checks:
  - lazy vs non-lazy exported chunk outputs were identical (`max_abs_diff=0.0`)
- End-to-end smoke (single-step decode path):
  - prompt: `What is 2+2?`
  - generated first token: `7281` (`" Also"`)
  - HF reference next token for same prompt: `7281` (`" Also"`)

## How To Reuse For Other Models

### Step 1: Add/Reuse model-level selective load API

In the model loader (e.g. `llama_model.py`, `gemma3_model.py`):

- accept `weight_spec`, `allow_partial`, `loader`
- convert only loaded tensors into module state dict format
- tolerate missing keys in partial mode
- add constructor controls to avoid allocating modules not needed by the selected conversion part
  - for chunked conversion, support `layer_range` (global layer indices) and local/global index mapping for cache writes

### Step 2: Define part/chunk specs in converter

For each part (`1`, `2`, `2_prefill`, `3`, etc):

- map conversion target -> required key prefixes
- for chunked parts, apply `layer_range` with `compute_chunk_layer_range`
- include final norm only when actually used by that function

### Step 3: Reload per chunk for chunked exports

When converting all chunks in one run:

- before tracing each chunk, call model loader with chunk-specific spec
- keep `allow_partial=True`

### Step 4: Validate

Minimum checks:

- conversion succeeds for at least one chunk and all chunks
- lazy vs non-lazy exported outputs are numerically equal for sample inputs
- one inference smoke prompt runs and returns coherent text
- at least one token-level check against HF (next token) matches

## Operational Notes

- Disk pressure can fail package save (`Errno 28 No space left on device`) during large multi-chunk exports. If this happens:
  - clear old `/tmp` conversion artifacts
  - rerun failed chunks with `--chunk-no`
- `part 2_prefill` exports in this setup use fixed prefill batch shape from conversion batch size.
- Some CoreML runtime warnings about execution-plan creation can appear during load; verify by running a predict call on each required model.

# ANEMLL 0.3.5 Alpha Release Notes

Release type: `Alpha`
Version: `0.3.5`

## Summary

- Fully redesigned **ANEMLL Chat** iOS/macOS/visionOS reference app with voice input, AirDrop model sharing, local model import, and Markdown rendering.
- **Gemma 3** family support (270M, 1B, 4B QAT) with sliding-window attention, global attention, FP16 scaling, and up to 4K context.
- **Monolithic model** conversion and inference for all architectures (LLaMA, Qwen, Qwen 2.5, Gemma 3) — single-file deployment with reduced overhead.
- **Swift inference stability fixes**: IOSurface-backed buffers, serial prediction queue, and ping-pong/ring buffer patterns eliminate ANE race conditions on iOS.
- **ANEMLL-Dedup**: surgical weight deduplication achieving ~50% size reduction for multifunction CoreML models.
- New conversion tools: **ANE Profiler**, **auto chunk calculator**, **FP16 preflight**, and **real-time conversion monitor**.
- **Qwen 2.5 / Qwen 3** converter support with monolithic and chunked modes.
- Variable context size demo for reasoning models with dynamic KV-cache state transitions.

## Pre-converted Models

The following models are available for download from the [ANEMLL HuggingFace collection](https://huggingface.co/anemll):

| Model | Context | Size | Type | Notes |
|-------|---------|------|------|-------|
| Gemma 3 270M (M1-compatible) | 512 | 0.5 GB | Monolithic | Argmax, fast compact model |
| Gemma 3 270M | 4096 | 0.9 GB | Chunked | 4K context, SWA + global attention |
| Gemma 3 1B | 4096 | 1.5 GB | Chunked | 4K context, SWA + global attention |
| Gemma 3 4B QAT int4 | 4096 | 2.5 GB | Chunked | Quantization-aware training variant |
| LLaMA 3.2 1B | 1024 | 1.2 GB | Chunked | Stable, fast |
| Qwen 3 1.7B | 2048 | 1.6 GB | Chunked | Thinking mode support |
| Qwen 2.5 0.5B | 2048 | 0.5 GB | Monolithic | Ultra-compact |

## 1. Redesigned ANEMLL Chat Reference App (iOS + macOS + visionOS)

> Try it now on TestFlight: [Join Beta](https://testflight.apple.com/join/jrQq1D1C)

Major redesign of the reference app to reduce friction from model conversion to in-app validation. Initial design iteration built with [anemll_macOS_agent](https://github.com/Anemll/anemll_macOS_agent) for automated UI testing via iPhone Mirroring.

### New Features
- **Voice input** — Speech-to-text dictation via microphone button for hands-free operation.
- **AirDrop model sharing** — Share compiled models from Mac to iPhone/iPad with one tap. Models are packaged as `.anemllpkg` transfer bundles.
- **Local model import** — Drag-and-drop a compiled model folder on Mac to import (copy into app storage) or link (reference external folder).
- **Network/external drive linking** — Link models from network volumes or external drives without copying. Automatic 3-second timeout prevents app freeze on unreachable volumes.
- **Markdown rendering** — Full Markdown support for assistant responses: bold, italic, headings, code blocks with language labels, tables, numbered and bullet lists.
- **Thinking mode** — Folded thinking blocks for Qwen 3 models showing reasoning process before the final answer.
- **Monolithic model inference** — Full support for single-file monolithic models alongside chunked models.

### UI/UX Improvements
- Improved model management with download progress, speed, and ETA display.
- Model detail view with configuration, quantization, weight file details, and recommended sampling parameters.
- HuggingFace collection API integration with offline caching and custom model support via repo ID.
- Toast notifications instead of modal alerts for non-blocking error/status feedback.
- Scroll-to-bottom indicator with animated chevron during long streaming responses.
- Copy button on hover (macOS) and long-press context menu (iOS) for messages.
- Liquid Glass UI effects on macOS 26+ with material fallback on older versions.
- Larger controls for iPadOS and visionOS by default.
- Auto-load last model on startup (configurable in Settings).

### Device Compatibility
- M1 Macs and A14 iOS devices have limited Gemma 3 support due to ANE differences — constrained to 512-token context with monolithic models. The app detects older hardware and alerts users during download or load.
- Pre-download compatibility check fetches `meta.yaml` from HuggingFace to warn before downloading incompatible models.

### Known App Limitations
- Multi-turn conversations re-run prefill for previous context on each turn. This is optimizable but ANE prefill is fast enough that the impact is minimal.

## 2. Gemma 3 Support with SWA and Global Attention

Full Gemma 3 family support (270M, 1B, 4B) with the architecture's unique attention pattern.

### Architecture
Gemma 3 uses interleaved **sliding-window attention** (SWA, window=512) and **global full attention** at every 6th layer (layers 6, 12, 18 for 1B). This requires a split KV cache:
- `kv_cache_local`: For SWA layers (15 layers, size = sliding window = 512)
- `kv_cache_global`: For global attention layers (3 layers, size = full context length)

### Conversion Features
- **4-function models**: `infer`, `prefill`, `infer_rotate`, `prefill_rotate` — rotation functions enable KV cache shift-left-and-append for sequences beyond the sliding window.
- **Split-rotate mode** (`--split-rotate`): Creates separate `.mlmodelc` files for rotate/non-rotate functions as a workaround for CoreML multi-function loading limitation.
- **Single-cache mode** (`--single-cache`): Unified KV cache without rotation for simpler deployment on compatible hardware.
- **Monolithic mode**: All 4 functions in a single `.mlmodelc` file for smaller models (270M).

### FP16 Scaling
Gemma 3 models trained in bfloat16 can overflow when converted to float16 for ANE. ANEMLL applies a weight-only scaling transformation at conversion time:

| Model | Scaling Factor (α) | Peak Activation | vs FP16 Max |
|-------|-------------------|-----------------|-------------|
| Gemma 3 270M | 0.48 | 104K | 1.6× over |
| Gemma 3 1B | 0.82 | 61K | 0.93× (borderline) |
| Gemma 3 4B QAT | 0.1875 | 293K | 4.5× over |

Usage: `--fp16-scale auto` (auto-detects model) or `--fp16-scale 0.48` (explicit).

Alternative: `--clamp 55000` adds runtime clamping ops to the CoreML model.

See: [`docs/GEMMA3_FP16_SCALING.md`](GEMMA3_FP16_SCALING.md)

### QAT Support
Correct handling of Gemma 3 4B QAT (quantization-aware training) models with aggressive FP16 scaling (α=3/16) and safe bfloat16-to-float16 conversion with clamping.

See: [`docs/GEMMA3_FP16_SCALING.md` — QAT analysis](GEMMA3_FP16_SCALING.md#googlegemma-3-4b-it-qat-int4-unquantized) for per-layer overflow maps, recommended α values, and validation test scripts.

### M1/A14 Limitations
Gemma 3's split KV cache requires non-uniform state shapes (global can be up to 24K tokens while SWA is always 512). Both M1 and A14 ANE fail to compile models with non-uniform state shapes. Current workaround: 512-context monolithic models for these devices. Future improvement planned via quantized KV cache or unified cache approach.

### Gemma 3N (Preview)
Gemma 3N has been ported and runs on ANE but needs further optimization before release.

## 3. Swift Inference Stability and Performance Fixes

The 0.3.4 release had a race condition primarily on iOS where CoreML `predict()` and I/O operations competed for the same buffers. This is fully resolved in 0.3.5.

### IOSurface-Backed Buffers
All pixel buffer outputs now use IOSurface backing for proper ANE synchronization. This eliminates coherency hazards between the ANE hardware and CPU reads without large sleep delays.

### Ping-Pong Buffers for FFN Chaining
Dual output backing pools prevent buffer reuse during multi-chunk FFN inference. The runtime alternates buffers by chunk index so the ANE is never reading from a buffer that's being written to.

### Serial Prediction Queue
A single execution lane (`predictionQueue`) serializes all stateful predictions — `predict()` calls and KV state reads run on the same serial queue, eliminating thread-safety issues with `MLState` + ANE access.

### Ring Buffer for Monolithic Models
16-deep ring buffer for monolithic model outputs prevents buffer reuse during ANE execution. Slots are selected via `tokenCounter % 16`.

### Tokenizer Improvements
- Enhanced tokenizer configuration detection for different model families.
- Fixed hardcoded EOT token (was LLaMA 3's `128009`) — now correctly uses EOS tokens from the tokenizer configuration.
- Proper multi-turn conversation formatting for all templates: Gemma/Gemma 3, LLaMA 3, DeepSeek, Qwen/ChatML.

### Qwen 3 Multi-Chunk Inference Fix
Fixed Qwen 3 multi-chunk inference divergence caused by applying final RMSNorm on every FFN chunk. The final normalization layer must only run on the **last** chunk — applying it to intermediate chunks corrupts the hidden state passed to the next chunk, producing progressively degraded output. This fix applies to both `qwen_converter.py` and `qwen2_5_converter.py`.

### Monolithic Model Loading
Full support for monolithic models in the Swift inference pipeline:
- 4-function loading: `infer`, `prefill`, `infer_rotate`, `prefill_rotate`
- Graceful fallback to 2-function models if rotation functions are unavailable.
- Direct `input_ids → logits` path without separate embed/FFN/lmhead stages.
- `model_type: "monolithic"` auto-detection in `meta.yaml`.

## 4. Monolithic Model Conversion

Monolithic conversion bundles embeddings, FFN layers, and LM head into a **single `.mlmodelc` file** with multiple functions (`infer` + `prefill`), as opposed to the standard chunked approach which produces 3+ separate files.

### Advantages
- **Simpler deployment** — one file to manage instead of 3+ separate models.
- **Reduced ANE overhead** — single kernel launch reduces scheduling latency.
- **Smaller total size** — ANEMLL-Dedup eliminates redundant weights between functions, saving ~50%.
- **Easier debugging** — single model for direct output comparison.

### Supported Architectures
All architectures support monolithic conversion:
- **LLaMA** — 2-function: `infer` + `prefill`
- **Qwen 3** — 2-function: `infer` + `prefill`
- **Qwen 2.5** — 2-function: `infer` + `prefill`
- **Gemma 3** — 4-function: `infer` + `prefill` + `infer_rotate` + `prefill_rotate`

### Conversion Script
Dedicated script: `anemll/utils/convert_monolith.sh`. Key flags: `--lut`, `--lut-embeddings`, `--lut-lmhead` (per-component quantization), `--argmax` (in-model argmax), `--single-cache` (Gemma 3 unified cache), `--skip-anemll-dedup`.

### Limitations
- Best suited for smaller models (up to ~3B parameters). Larger models may exceed iOS file size limits or ANE memory constraints.
- Entire model loaded into memory simultaneously (chunked allows sequential loading).

## 5. Conversion Pipeline Improvements

### ANEMLL-Dedup — Weight Deduplication
Surgical weight deduplication for CoreML multifunction models. When `infer` and `prefill` functions share identical palettized weights (LUT + indices), dedup replaces duplicates with references to a single copy.
CoreML's built-in `ct.transform_weights` deduplication compares raw bytes, so semantically-identical palettized weights (same dequantized values but different LUT ordering or index encoding) are not recognized as duplicates — causing 15–40% unnecessary size bloat in multifunction models. ANEMLL-Dedup normalizes the LUT/index representation first, then re-saves using `PassPipeline.EMPTY` to prevent coremltools default optimization passes from undoing the surgical replacements.

**Results:**
- Qwen 2.5 0.5B monolithic (LUT6, ctx2048): 862 MB → **472 MB** (49.9% savings)
- VibeThinker 1.5B chunk 2 (LUT6, ctx512): 523 MB → **309 MB** (41% savings)

Enabled by default in all combine operations. Disable with `--skip-anemll-dedup`.

Multi-metric acceptance criteria: cosine similarity ≥ 0.9999 and mean absolute error ≤ 0.001.

See: [`docs/anemll-dedup.md`](anemll-dedup.md)

### Auto Chunk Calculator
`calc_chunk_split.py` automatically determines the optimal `--chunk` value based on model weight sizes and a target max chunk size. Also available as `--chunk auto` in `convert_model.sh`. Supports LLaMA, Qwen, Gemma, DeepSeek, and nested multimodal configurations.

See: [`docs/calc_chunk_split.md`](calc_chunk_split.md)

### ANE Profiler
Automated CoreML/ANE profiling without Xcode, using the coremltools 9.0+ `MLComputePlan` API. Use this tool when evaluating new model architectures or troubleshooting inference issues — it pinpoints which operations fall back to CPU/GPU instead of running on ANE, helping identify compatibility problems before deployment. Supports `--analyze` (op placement report), `--benchmark` (single compute unit), and `--benchmark-all` (compare ANE vs CPU vs GPU timing).

See: `anemll/utils/ANE_PROFILER.md`

### FP16 Preflight
One-command FP16 compatibility check before conversion (`fp16_preflight.sh`). Identifies overflow risks, problematic layers, and recommends scaling factors. Auto-activates local venv, runs clamp sweep tests, saves JSON report to `tests/dev/logs/`.

See: [`docs/FP16_SCALING.md`](FP16_SCALING.md)

### Real-Time Conversion Monitor
Live progress tracking for long-running conversions (`monitor_conversion.py`). Auto-detects running conversion processes, shows progress bars with step timing, estimated remaining time, and disk space warnings.

### In-Model Argmax (`--argmax`)
Moves the argmax operation from the host (Python/Swift) into the CoreML model itself. Instead of outputting the full logits tensor (vocab_size floats per token), the LM head computes `argmax` per chunk internally and returns only two small tensors: `argmax_idx` (local winner index per chunk, int32) and `argmax_val` (corresponding logit value per chunk, fp16). The host then selects the global winner across chunks.

**Why it matters:**
- Drastically reduces LM head output size — e.g., from 262K floats (Gemma 3 vocab) down to 2×16 scalars for 16 chunks.
- Less data transferred off ANE means lower latency per token.
- Simplifies the inference host code: no large logits buffer allocation, no host-side argmax scan.

**Future direction:** The same pattern can be extended to **top-k sampling** inside the CoreML model — each chunk returns its top-k candidates, and the host merges k×num_chunks entries instead of scanning the full vocabulary. This would enable efficient on-device sampling without transferring full logits.

Supported in both chunked and monolithic converters for all architectures (LLaMA, Qwen, Gemma 3). Recorded in `meta.yaml` as `argmax_in_model: true` and embedded in CoreML model metadata (`com.anemll.argmax_in_model`), so inference hosts auto-detect the output format.

### New convert_model.sh Flags

| Flag | Description |
|------|-------------|
| `--chunk auto` | Auto-calculate optimal chunk count based on weight sizes |
| `--max-chunk-mb <MB>` | Target max chunk size for auto mode (default: 950) |
| `--fp16-scale <factor>` | FP16 residual scaling for Gemma 3 (`auto`, `0.1875`, `0.48`, `0.82`) |
| `--clamp <value>` | Runtime clamping for FP16 overflow prevention |
| `--argmax` | Compute argmax inside LM head (outputs idx+val instead of logits) |
| `--split-rotate` | Separate files for rotate/non-rotate functions (Gemma 3) |
| `--single-cache` | Unified KV cache without rotation (Gemma 3) |
| `--skip-anemll-dedup` | Disable weight deduplication |
| `--dynamic-prefill-slice` | Dynamic slicing for prefill KV writes (default ON) |

## 6. ANE Conversion and Troubleshooting Workflows

Structured workflows for debugging ANE conversion and runtime issues:

- **Divergence analysis**: Compare PyTorch vs CoreML outputs layer-by-layer to identify where ANE lowering introduces numerical drift.
- **ANE lowering planning**: Use `ane_profiler.py --analyze` to identify which operations fall back to CPU/GPU and why.
- **FP16 preflight**: Run `fp16_preflight.sh` before conversion to catch overflow issues early.
- **Runtime profiling**: Use `ane_profiler.py --benchmark-all` to compare performance across compute units and identify bottlenecks.

Documented troubleshooting patterns in [`docs/troubleshooting.md`](troubleshooting.md).

### Chat CLI Improvements (`chat.py` / `chat_full.py`)

New flags and capabilities added in 0.3.5:

- **`--st` (single-token prefill)** in `chat_full.py` — processes one token at a time instead of batch prefill. Slower but invaluable for debugging multi-turn divergence and pinpointing exactly where batch vs sequential outputs differ.
- **`--cpu`** — force CPU-only execution to compare ANE vs CPU numerics for divergence analysis.
- **`--no-think`** — disable thinking mode for Qwen 3 models.
- **`--debug` / `--debug-argmax`** in `chat.py` — print position, state, and tensor shapes each step, or argmax indices and values from LM head output.
- **`--mem-report`** — print RSS memory usage after each major stage (model load, prefill, generation).
- **`--split-rotate`** / **`--sliding-window N`** — support for Gemma 3 split-rotate model files and SWA cache configuration.
- **Gemma 3 and SWA inference** — both tools now handle sliding-window attention models with rotation functions, including graceful skip when context fits within the window.
- **Robust stop-token detection** — architecture-aware EOS/EOT token collection replaces hardcoded token IDs, fixing premature or missing stops across model families.
- **Argmax LM head support** — `chat.py` handles in-model argmax with per-chunk local-to-global vocab index mapping, auto-resolved from `meta.yaml`.

## 7. Variable Context Size Demo for Reasoning Models (State Transitions)

Experimental infrastructure for dynamic context size switching during inference, targeting reasoning models (VibeThinker / Qwen 3.5 family).

### How It Works
The demo starts inference with a small context (512 tokens) and automatically transitions to larger contexts (1024 → 2048 → 3072 → 4096) as the generated token count grows. When the largest context is exhausted, a shift-refill policy keeps a sliding window of recent tokens.

### Implementation
- `anemll/utils/state_transition.py` — Core utility for KV-cache state transitions: resizes cache tensors, copies existing state, and pads for the new context size.
- `tests/dev/state_transition_growing_inference.py` — Full demo runner with batch/token-by-token prefill, sampling, time-limited generation.
- `tests/dev/test_state_transition.py` — 34-test suite validating transition correctness.

### Context Sizes Tested
512, 1024, 2048, 3072, 4096 tokens.

Works with both per-context chunked exports (separate directories per context size) and combined multi-context exports (single folder with `infer_ctx{N}` / `prefill_ctx{N}` functions).

### Try It

A standalone demo script and pre-converted VibeThinker 1.5B model are available:
- Demo: `examples/variable_context_demo.py` — single-command launcher with sensible defaults.
- Documentation: [`examples/VARIABLE_CONTEXT.md`](../examples/VARIABLE_CONTEXT.md) — full guide covering advantages, how it works, and all options.
- Pre-converted model: [anemll/anemll-vibethinker-1.5b-state-transition](https://huggingface.co/anemll/anemll-vibethinker-1.5b-state-transition) on HuggingFace.

See also: `tests/dev/STATE_TRANSITION_EXPERIMENTS.md`

## Model/Architecture Status in 0.3.5

| Architecture | Status | Chunked | Monolithic | Notes |
|-------------|--------|---------|------------|-------|
| LLaMA 3.1/3.2 | Stable | Yes | Yes | 1B, 8B tested |
| DeepSeek R1 | Stable | Yes | — | Via LLaMA converter (8B distilled) |
| DeepHermes | Stable | Yes | — | Via LLaMA converter (3B, 8B) |
| Qwen 3 | Stable | Yes | Yes | 0.6B, 1.7B, 8B; thinking mode |
| Qwen 2.5 | Stable | Yes | Yes | 0.5B tested |
| Gemma 3 | Stable | Yes | Yes | 270M, 1B, 4B QAT; SWA + global attention |

## Developer-Facing Changes

- Dependency baseline standardized to `coremltools>=9.0`.
- Added UV environment bootstrap: `create_uv_env.sh` (Rust-based `uv` for fast installs).
- Added Python 3.9 environment bootstrap: `create_python39_env.sh` with fallback to 3.10/3.11.
- Added FP16 preflight helper: `anemll/utils/fp16_preflight.sh`.
- Added monolithic conversion script: `anemll/utils/convert_monolith.sh`.
- Consolidated Gemma 3 FP16 scaling documentation at [`docs/GEMMA3_FP16_SCALING.md`](GEMMA3_FP16_SCALING.md).
- Comprehensive release test plan at `tests/dev/RELEASE_TEST_PLAN.md`.

## Upgrade Notes

- Recommended environment setup:
  - `./create_python39_env.sh` or `./create_uv_env.sh`
  - `./install_dependencies.sh`
- Before converting new model families:
  - `./anemll/utils/fp16_preflight.sh --model <model_id_or_path>`
- If chunk sizing is uncertain:
  - Start with `--chunk auto` and tune `--max-chunk-mb`
- For smallest model size:
  - Ensure ANEMLL-Dedup is enabled (default) — check for `--skip-anemll-dedup` if sizes look large.

## Known Constraints (Alpha)

- Quantization quality varies by architecture and LUT configuration.
- M1 Macs and A14 iOS devices limited to 512-context monolithic Gemma 3 models.
- Multi-turn conversations re-run prefill for full context history each turn.
- macOS 15+ and Apple Silicon required for ANE inference.
- iOS models with weight files >1 GB may fail to load on iPhone and non-M-series iPad.

## Documentation

| Document | Description |
|----------|-------------|
| [`convert_model.md`](convert_model.md) | Single-shot chunked conversion guide |
| [`convert_monolith.md`](convert_monolith.md) | Monolithic conversion guide |
| [`FP16_SCALING.md`](FP16_SCALING.md) | FP16 compatibility by architecture |
| [`GEMMA3_FP16_SCALING.md`](GEMMA3_FP16_SCALING.md) | Gemma 3 FP16 scaling details |
| [`anemll-dedup.md`](anemll-dedup.md) | Weight deduplication guide |
| [`calc_chunk_split.md`](calc_chunk_split.md) | Auto chunk calculator usage |
| [`ANE_PROFILER.md`](../anemll/utils/ANE_PROFILER.md) | ANE profiler documentation |
| [`troubleshooting.md`](troubleshooting.md) | Conversion and runtime troubleshooting |
| [`qwen3.md`](qwen3.md) | Qwen 3 conversion notes |
| [`sample_apps.md`](sample_apps.md) | iOS/macOS sample app guide |
| [`swift_cli.md`](swift_cli.md) | Swift CLI reference |

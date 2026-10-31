---
name: ane-lowering-plan
description: Create an ANE-legal CoreML/MIL lowering plan for decoder-only Transformers from Hugging Face or JAX/Flax, including architecture fingerprinting, KV-cache/state contracts, and legality constraints. Use when converting or debugging HF/JAX models for ANE (Core ML) deployment, especially when sliding window/global attention, chunked FFN, or monolithic multi-function models are involved. Also use to analyze PyTorch vs CoreML divergence using tests/dev compare harnesses. Do not use for encoder-decoder models or non-Transformer architectures.
---

# ANE Lowering Plan (HF/JAX -> CoreML/MIL)

## What this skill is for
Produce a repeatable, testable lowering plan that turns a decoder-only Transformer into ANE-compatible CoreML/MIL with a precise shape/state contract and legality checklist.

## When to use / when not to use
Use when:
- Converting HF Transformers or JAX/Flax decoder-only models to CoreML/MIL for ANE.
- Debugging chunked vs monolithic exports, sliding-window vs global attention, or KV-cache/state correctness.
- You need a strict, testable plan: shapes, state I/O, and legality rules.

Do not use when:
- Model is encoder-decoder, RNN/SSM, or non-decoder-only Transformer.
- Target is GPU-only or non-CoreML backends without ANE constraints.

## Required inputs
- Model config (HF `config.json` or JAX/Flax equivalent): layers, heads, hidden size, kv heads, rope, norm type.
- Weights (HF safetensors or JAX checkpoint) and tokenizer assets.
- Target context length + prefill batch size.
- Quantization plan (LUT bits, per-channel group size) if used.
- Attention pattern definition (full/global vs sliding/local) and window size.

## Step-by-step procedure
1) **Fingerprint the architecture**
   - Extract hidden size, num layers, num heads, num kv heads (GQA/MQA), intermediate size, norm type, RoPE base, vocab size.
   - Detect attention pattern: full/global vs sliding/local (and its interval or layer list).

2) **Lock fixed shapes**
   - Choose `context_length`, `state_length`, and `prefill_batch_size`.
   - Ensure all tensors are static and last-dim >= 64.

3) **Qualify FP16 compatibility (Gemma3 priority)**
   - Run FP16 compatibility check before ANE conversion.
   - If residual overflow risk exists, apply weight scaling (preferred) or safety clamp.
   - See `docs/GEMMA3_FP16_SCALING.md` and `anemll/utils/fp16_compatibility_check.py`.

4) **Define KV-cache/state contract**
   - Decide unified vs split cache (local/global) and the exact state layout.
   - Document prefill vs decode state update rules.

5) **Define attention masks**
   - Use one causal mask tensor; slice it consistently for prefill and decode.
   - For sliding window: ensure local attention window masking matches KV cache window.

6) **Lower to ANE-safe ops**
   - Use Conv2D/1x1 for linear projections.
   - Avoid dynamic reshape/transpose order that yields last-dim < 64.

7) **Chunking plan (if applicable)**
   - Split layers across chunks with remainder-aware indexing.
   - Validate each chunk's layer range and number of layers.

8) **Build function set**
   - Standard: `infer` and `prefill`.
   - For sliding-window rotation: include `infer_rotate` and `prefill_rotate`.

9) **Combine + metadata**
   - Combine into multi-function model if needed.
   - Generate `meta.yaml` with functions list, context/state length, LUT bits.

10) **Verify**
   - Run smoke tests and numeric sanity checks.
   - For probe/debug exports, print the traced wrapper output name and shape before `torch.jit.trace(...)`.
   - Compile suspect `.mlpackage` artifacts and inspect `model.mil` to confirm the real exported output path.

## Architecture fingerprint checklist
- Layers: `num_hidden_layers`
- Hidden size: `hidden_size`
- Attention heads: `num_attention_heads`
- KV heads: `num_key_value_heads` (GQA/MQA)
- Intermediate: `intermediate_size`
- Norm type: RMSNorm/LayerNorm
- RoPE: `rope_theta` and any local/global base
- Attention pattern: sliding/local window size + global layer indices

## Shape/state contract (prefill vs decode)

### Chunked or monolithic (unified cache)
| Phase | Inputs | Shapes (canonical) | Outputs/State |
|---|---|---|---|
| Prefill | `input_ids`, `position_ids`, `causal_mask`, `current_pos` | `input_ids:[1,B]` `position_ids:[B]` `causal_mask:[1,1,B,context]` `current_pos:[1]` | Updates state KV cache for positions `[current_pos, current_pos+B)` |
| Decode | `input_ids`, `position_ids`, `causal_mask`, `current_pos` | `input_ids:[1,1]` `position_ids:[1]` `causal_mask:[1,1,1,context]` `current_pos:[1]` | Updates state at `current_pos` and produces next logits |

**KV cache** (unified): `kv_cache:[2*num_layers, num_kv_heads, state_length, head_dim]`

### Split cache (sliding/local + global)
- Local cache: `kv_cache_local:[2*num_local_layers, num_kv_heads, sliding_window, head_dim]`
- Global cache: `kv_cache_global:[2*num_global_layers, num_kv_heads, state_length, head_dim]`
- Decode uses local window for local layers and full state_length for global layers.

### Rotation vs fill (sliding window)
- Apply only if the model supports sliding window attention; otherwise use standard full-context causal masking.
- **Fill mode** (`pos < sliding_window`): write sequentially.
- **Rotate mode** (`pos >= sliding_window`): rotate/shift window, update at end.
- Ensure causal mask matches the windowed key length used by each layer.

## ANE/MIL lowering rules (do / don't)
**Do**
- Use Conv2D (1x1) for linear projections (QKV, MLP, LM head).
- Keep last-dim >= 64 (pad/reshape if needed).
- Use static shapes and constant sizes in MIL graph.
- Slice causal masks instead of recomputing per op.
- For RMSNorm on ANE, use the ANEMLL RMS hack: `concat([x, -x]) -> layer_norm (no affine) -> slice back -> scale` (see `QwenRMSNorm`/`QwenHeadNorm` in `qwen_model`).

**Don't**
- Don't emit dynamic control flow in MIL.
- Don't rely on dynamic reshape that yields last-dim < 64.
- Don't use ops unsupported by ANE (dynamic gather/scatter with unknown shapes).

## Illegal-op avoidance checklist
- [ ] No dynamic shapes or control flow in exported graph
- [ ] No `Gather`/`Slice` with data-dependent indices in MIL
- [ ] Avoid casts that create unsupported types mid-graph
- [ ] Ensure `causal_mask` is static shape with correct dtype
- [ ] No last-dim < 64 in any Conv2D/MatMul path

## Common errors and fixes
| Error signature | Likely cause | Fix |
|---|---|---|
| `GemmaTokenizer requires the SentencePiece library` | Missing dependency | Install `sentencepiece` in env |
| `Provided key "update_mask" ... does not match any model input` | Model removed `update_mask` input | Only pass if present in model inputs |
| `tuple index out of range` when reshaping weights | Wrong tensor rank assumption | Check weight shapes before `view`; handle missing dims |
| Missing chunk files / identical layer norms | Chunking drops remainder layers | Use remainder-aware chunk split (last chunk gets extra) |
| `integer expression expected` in convert_monolith | Non-numeric step labels | Normalize step labels for comparison |
| `Infer model not found` during combine | Rotate functions not generated | Run rotate conversions for context > sliding_window |
| Palettize warning channel_group_size | channel count not divisible by group | Accept skip or adjust group size |

## Minimal verification protocol
1) **Smoke test**: run `tests/chat.py` and `tests/chat_full.py` with same prompt and max tokens; check responses are coherent.
2) **Numerical sanity**: check logits range (roughly -20 to +20) and no hidden-state explosions across chunks.
3) **Mask sanity**: confirm causal mask shapes match model inputs and window size.
4) **State sanity**: verify KV-cache shapes match contract and update positions are consistent.

## FP16 qualification (Gemma3 / ANE)
- **Required** for Gemma3: BF16-trained residuals can overflow FP16 on ANE.
- Use `anemll/utils/fp16_compatibility_check.py --model <id> --sweep` to detect overflow risk.
- Prefer **weight scaling** over runtime clamp; clamp only as a safety net.
- If scaling is needed, compute alpha from observed peak (`target_max ~= 50,000`) and apply via conversion flag (for example, `--fp16-scale`).
- Reference: `docs/GEMMA3_FP16_SCALING.md` and the generated FP16 preflight report for recommended next steps.

## ANE divergence analysis (PyTorch vs CoreML)
Use when outputs drift between CoreML/ANE and PyTorch or HF reference. Prefer deterministic runs (greedy, no-think).

### Workflow decision tree
1) **Pick comparison target**
   - Gemma-style via meta.yaml: `tests/dev/test_gemma3_compare.py`
   - Chunked CoreML vs PyTorch: `tests/dev/test_gemma3_coreml_chunks_vs_pytorch.py`

2) **Pick scope**
   - Single prompt: `*_compare.py`
   - Batch dataset: `tests/dev/gemma3_divergence_harness.py`

### Preflight checklist
- Match chat templates and special-token settings on both sides.
- Align `context_length` / `state_length` with `meta.yaml`.
- Use greedy decoding for parity (avoid sampling).
- Use `driver coreml` to mimic ANE; `driver pt` for baseline.

### Core commands (examples)
```bash
python tests/dev/test_gemma3_compare.py <coreml_dir_or_meta.yaml> \
  --hf-reference <hf_model_id> --prompt "..." --driver coreml --no-think

python tests/dev/test_gemma3_coreml_chunks_vs_pytorch.py <coreml_dir_or_meta.yaml> \
  --prompt "..." --no-think
```

### Interpret metrics
- Track **match_rate**, **KL**, **correlation**, **entropy**, **rep4**.
- Drift signals: correlation down, KL up, entropy collapse, rep4 spikes.

### Divergence triage
1) **Prompt-phase mismatch** -> verify tokenizer, template, and special-token handling first.
2) **Decode-only mismatch** -> shorten `--max-tokens`, compare `driver pt` vs `driver coreml`.
3) **Quantization vs conversion** -> HF reference vs CoreML to isolate quantization effects.
4) **Instability loops** -> inspect per-prompt outputs; check repetition/entropy/margin spikes.
5) **Probe package looks wrong** -> compile the `.mlpackage` and inspect `model.mil`; verify the traced wrapper return tensor before changing the lowering plan.

### Probe export sanity checks
- Before tracing a diagnostic wrapper, run one sample forward pass and print the output name and shape.
- Compile debug/probe packages with:

```bash
xcrun coremlcompiler compile <model.mlpackage> <out_dir>
```

- Inspect `model.mil` for:
  - final output name and shape
  - unexpected projections still on the output path (`o_proj`, LM head, final logits)
  - unexpected rank collapse or hidden-state-shaped outputs when a payload probe should stay 4D

### Required references
- `tests/dev/DIVERGENCE_HARNESS.md`
- `tests/dev/test_gemma3_compare.py`
- `tests/dev/gemma3_divergence_harness.py`
- `tests/dev/test_gemma3_coreml_chunks_vs_pytorch.py`
- `docs/GEMMA3_FP16_SCALING.md`
- `anemll/utils/fp16_compatibility_check.py`

## Example (Gemma3 sliding window)
- Model: Gemma3 1B, context 4096, sliding_window 512, global layers every 6.
- Functions: `infer`, `prefill`, `infer_rotate`, `prefill_rotate`.
- Split cache: local window cache for sliding layers + global cache for full-attention layers.
- Validation: prefill for first 512 with fill mode; rotate after 512; compare outputs across chunked vs monolithic.

## Version assumptions (freshness guards)
- coremltools 9.x (ML Program path) and coremlcompiler 35xx.
- macOS 15+ for ANE compilation.
- Python 3.9+ runtime.

## Likely-to-change warnings
- coremltools pass names and legality constraints can change between releases.
- Tokenizer special token IDs (EOS/EOT) may differ by model family.
- ANE backend constraints may tighten or expand with OS updates.

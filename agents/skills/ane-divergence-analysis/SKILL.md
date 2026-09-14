---
name: ane-divergence-analysis
description: Compare PyTorch vs CoreML/ANE and HuggingFace Transformers vs CoreML for released decoder-only LLM workflows in this repo. Use when debugging token/logit divergence, instability, or ANE lowering issues; when running tests/dev/*_compare.py or tests/dev/*_divergence_harness.py; or when you need prompt/decode parity metrics (KL, correlation, entropy, repetition).
---

# ANE Divergence Analysis

## Overview

Use the divergence harnesses under `tests/dev/` to measure parity, instability, and repetition between
PyTorch, HuggingFace Transformers, and CoreML/ANE exports. Keep runs deterministic (greedy decoding,
`--no-think`) while diagnosing, then scale to batch harnesses for instability datasets.

## Workflow Decision Tree

1) Pick the comparison target:
- **Generic HF vs CoreML via meta.yaml (Gemma-style exports)** -> `tests/dev/test_gemma3_compare.py`
- **Chunked CoreML vs PyTorch** -> `tests/dev/test_gemma3_coreml_chunks_vs_pytorch.py`

2) Choose scope:
- **Single prompt** -> use `*_compare.py`
- **Batch / instability dataset** -> `tests/dev/gemma3_divergence_harness.py`

## Preflight Checklist

- Use the same chat template and special-token settings on both sides.
- Align `context_length` / `state_length` with CoreML `meta.yaml` (override only when intentional).
- Use greedy decoding for parity; avoid sampling while debugging divergence.
- Prefer `driver coreml` to mimic ANE behavior; use `driver pt` for parity baselines.
- For diagnostic exports, run the wrapper once before `torch.jit.trace(...)` and print the traced output name and shape.
- For suspect `.mlpackage` exports, compile them and inspect `model.mil` before blaming backend math.

## Core Commands

### Gemma/HF vs CoreML (generic meta.yaml)

```bash
python tests/dev/test_gemma3_compare.py <coreml_dir_or_meta.yaml> \
  --hf-reference <hf_model_id> --prompt "..." --driver coreml --no-think
```

### Batch HF vs CoreML (Gemma-style)

```bash
python tests/dev/gemma3_divergence_harness.py <coreml_dir_or_meta.yaml> \
  --hf-reference <hf_model_id> --dataset <prompts.jsonl> --out-dir runs/<name>
```

### Chunked CoreML vs PyTorch

```bash
python tests/dev/test_gemma3_coreml_chunks_vs_pytorch.py <coreml_dir_or_meta.yaml> \
  --prompt "..." --no-think
```

## Export Probe Checks

Use these when a diagnostic/CoreML probe appears to expose the wrong tensor, wrong shape, or unexpected downstream ops.

### Trace-sample print before `torch.jit.trace(...)`

Run the wrapper once on representative sample inputs and print:
- output name
- output shape

This catches wrapper-return mistakes early, especially when you intend to export a pre-projection payload but the wrapper still returns post-projection hidden states.

### Compile-check the exported package

```bash
xcrun coremlcompiler compile <model.mlpackage> <out_dir>
```

Then inspect:

```bash
find <out_dir> -name model.mil -print
sed -n '1,240p' <out_dir>/<compiled>.mlmodelc/model.mil
```

Check `model.mil` for:
- final output name and shape
- unexpected `output_hidden_states` vs payload output names
- unexpected `o_proj`/LM-head/final projection still present on the output path
- rank regressions (for example, payload expected as 4D but exported as hidden-size 3D)

## Interpret Metrics

- Use **match_rate**, **KL**, and **correlation** to measure parity; lower correlation and higher KL indicate drift.
- Watch instability signals: stuck token, phrase repeat, rep4 spike, entropy collapse, margin explosion.

## Pseudocode: Divergence Calculations

```text
# Inputs per step:
#   pt_logits[vocab], cm_logits[vocab]
# Outputs per step:
#   kl, corr, entropy_cm, match

softmax(x):
  x = x - max(x)
  return exp(x) / sum(exp(x))

entropy(p):
  p = clip(p, eps, 1)
  return -sum(p * log(p))

kl_divergence(p, q):
  p = clip(p, eps, 1)
  q = clip(q, eps, 1)
  return sum(p * log(p / q))

corrcoef(a, b):
  a_mean = mean(a); b_mean = mean(b)
  a_std = std(a); b_std = std(b)
  if a_std < eps or b_std < eps: return NaN
  return mean((a - a_mean) * (b - b_mean)) / (a_std * b_std)

pt_probs = softmax(pt_logits)
cm_probs = softmax(cm_logits)

kl = kl_divergence(pt_probs, cm_probs)   # KL(PT || CM)
corr = corrcoef(pt_logits, cm_logits)
entropy_cm = entropy(cm_probs)
match = (argmax(pt_logits) == argmax(cm_logits)) ? 1 : 0

# Aggregate over steps:
kl_mean = mean(kl_list)
corr_mean = mean(corr_list)
entropy_mean = mean(entropy_cm_list)
match_rate = mean(match_list)
```

## Divergence Triage

1) **Prompt-phase mismatch**: Verify tokenizer, chat template, and special-token settings before blaming backend math.
2) **Decode-only mismatch**: Flip `--driver` between `pt` and `coreml` to isolate ANE behavior; shorten `--max-tokens`.
3) **Quantization vs conversion**: Compare HF vs CoreML first; if chunked PT vs CoreML is also close, suspect conversion-specific drift elsewhere in the pipeline.
4) **Instability loops**: Use the batch harness and inspect per-prompt NPZ/JSON for rep4/entropy/margin spikes.
5) **Diagnostic export mismatch**: If a probe package looks wrong in Xcode or Netron, compile it and inspect `model.mil`; verify the traced wrapper return tensor before changing model math.

## Required References

- Read `tests/dev/DIVERGENCE_HARNESS.md` for released harness usage and dataset guidance.
- Inspect `tests/dev/test_gemma3_compare.py` and `tests/dev/gemma3_divergence_harness.py` for generic HF vs CoreML comparisons.
- Inspect `tests/dev/test_gemma3_coreml_chunks_vs_pytorch.py` for chunked CoreML vs PyTorch comparisons.

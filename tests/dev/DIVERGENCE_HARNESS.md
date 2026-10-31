# ANEMLL Divergence Harness

Tools for detecting and analyzing divergence or instability between PyTorch/HuggingFace reference runs and CoreML/ANE exports in the released `main` workflow.

## Supported entry points

| File | Purpose |
|---|---|
| `test_gemma3_compare.py` | Interactive HF vs CoreML comparison with token/logit metrics |
| `gemma3_divergence_harness.py` | Batch dataset evaluation for instability analysis |
| `test_gemma3_coreml_chunks_vs_pytorch.py` | Chunked CoreML vs PyTorch comparison |

## What to measure

- **Token divergence**: next-token disagreement between reference and CoreML
- **Logit drift**: KL increase, correlation drop, margin changes
- **Entropy collapse**: CoreML becomes overconfident too early
- **Repetition loops**: rep4/phrase repetition spikes over decode
- **State drift**: chunked or cached decode diverges over longer sequences

## Quick start

### Single prompt: HF vs CoreML

```bash
python tests/dev/test_gemma3_compare.py <coreml_dir_or_meta.yaml> \
  --hf-reference <hf_model_id> \
  --prompt "What is the capital of France?" \
  --driver coreml \
  --no-think
```

### Batch dataset: instability scan

```bash
python tests/dev/gemma3_divergence_harness.py <coreml_dir_or_meta.yaml> \
  --hf-reference <hf_model_id> \
  --dataset <prompts.jsonl> \
  --out-dir runs/<name>
```

### Chunked CoreML vs PyTorch

```bash
python tests/dev/test_gemma3_coreml_chunks_vs_pytorch.py <coreml_dir_or_meta.yaml> \
  --prompt "Explain rotary embeddings." \
  --no-think
```

## Preflight

- Match tokenizer/chat-template behavior on both sides.
- Match `context_length` and `state_length` with `meta.yaml`.
- Use greedy decoding while debugging divergence.
- Prefer short prompts first, then longer prompts once parity is understood.
- If a debug export looks suspicious, compile the `.mlpackage` and inspect `model.mil`.

## Suggested prompt dataset fields

JSONL is recommended:

```json
{
  "id": "prompt_001",
  "prompt": "What is the capital of France?",
  "category": "factual",
  "risk": "low",
  "target_len": "short"
}
```

Useful `risk` buckets:
- `low`
- `repetition`
- `reasoning`
- `long_context`

## Instability signals

| Signal | Meaning |
|---|---|
| `match_rate` down | Token-level drift is accumulating |
| `KL` up | Distribution mismatch is widening |
| `correlation` down | Logit shape diverges even if argmax still matches |
| `entropy` collapse | Model becomes overconfident or stuck |
| `rep4` spike | Repetition loop likely |

## Practical triage

1. Verify prompt formatting and tokenizer behavior.
2. Compare HF vs CoreML on a short prompt.
3. If close, compare chunked CoreML vs PyTorch to isolate state/chunk drift.
4. Run the batch harness on a prompt set to identify unstable categories.
5. If a probe export looks wrong, inspect compiled `model.mil` before changing model math.

## Outputs worth keeping

- Per-prompt JSON summaries
- NPZ arrays for per-step metrics
- A CSV or summary table sorted by worst `match_rate`, highest `KL`, and lowest `correlation`
- The exact `meta.yaml` and command line used for the run

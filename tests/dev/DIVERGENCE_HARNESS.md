# ANEMLL Divergence Harness & Stability Testing

Tools for detecting and analyzing instability in quantized LLM models running on Apple Neural Engine.

## Overview

These tools compare PyTorch reference models against CoreML/ANE models to detect:
- **Token divergence**: When models predict different next tokens
- **Entropy collapse**: When the model becomes overly confident (low entropy)
- **Repetition loops**: When the model gets stuck repeating phrases
- **Numerical drift**: Accumulated errors in KV cache over long sequences

## Files

### Core Testing Tools

| File | Purpose |
|------|---------|
| `test_qwen_aq1_compare.py` | Interactive PyTorch vs CoreML comparison with detailed metrics |
| `qwen_aq1_divergence_harness.py` | Batch dataset generation for instability analysis |
| `analyze_runs.py` | Post-process harness outputs into train/val datasets |

### Supporting Files

| File | Purpose |
|------|---------|
| `test_aq1_inference.py` | Basic CoreML inference testing |
| `test_qwen_aq1_pytorch.py` | PyTorch-only inference for baseline |
| `prompts.jsonl` | 51 test prompts with risk categories |
| `short_prompts.jsonl` | Quick smoke test prompts |

#### test_aq1_inference.py

Standalone inference test for AQ1 quantized models with optional ANE comparison.

```bash
# Basic inference
python tests/dev/test_aq1_inference.py ~/Downloads/snapped_step1800.pt \
    --prompt "What is AI?" --no-think --max-tokens 100

# Compare with CoreML/ANE
python tests/dev/test_aq1_inference.py ~/Downloads/snapped_step1800.pt \
    --prompt "What is AI?" --no-think --compare-ane \
    --coreml-model /Users/anemll/Models/ANE/q4_r32_lut_ka
```

**Key options:** `--no-think`, `--compare-ane`, `--coreml-model`, `--temperature`, `--max-tokens`

#### test_qwen_aq1_pytorch.py

PyTorch-only inference using QwenModel (same model class used for ANE conversion).

```bash
# Basic PyTorch inference
python tests/dev/test_qwen_aq1_pytorch.py ~/Downloads/snapped_step1800.pt \
    --prompt "What is AI?" --no-think --max-tokens 100

# Verbose with streaming disabled
python tests/dev/test_qwen_aq1_pytorch.py ~/Downloads/snapped_step1800.pt \
    --prompt "History of England" -v --no-stream
```

**Key options:** `--no-think`, `--temperature`, `--max-tokens`, `--verbose`, `--no-stream`

---

## Quick Start

### 1. Single Prompt Comparison

```bash
# Compare PyTorch vs CoreML on a single prompt
python tests/dev/test_qwen_aq1_compare.py \
    ~/Downloads/snapped_step1800.pt \
    /Users/anemll/Models/ANE/q4_r32_lut_ka \
    --prompt "What is the capital of France?" \
    --max-tokens 100 \
    --driver coreml \
    --no-think
```

**Key options:**
- `--driver coreml` - Use CoreML predictions (realistic ANE behavior)
- `--driver pt` - Use PyTorch predictions (parity testing)
- `--prefill-mode token` - Single-token stepping (better for analysis)
- `--prefill-mode batch` - Batch prefill (faster, matches chat.py)
- `--no-think` - Disable Qwen3 thinking mode (uses `enable_thinking=False`)
- `--hf-reference MODEL_ID` - Use HuggingFace model as reference instead of baked checkpoint

### 1b. HuggingFace Reference Comparison

```bash
# Compare vanilla HuggingFace Qwen3 vs CoreML (isolates quantization effects)
python tests/dev/test_qwen_aq1_compare.py \
    --hf-reference Qwen/Qwen3-0.6B \
    /Users/anemll/Models/ANE/q4_r32_lut_ka \
    --prompt "What is the capital of France?" \
    --max-tokens 100 \
    --driver coreml \
    --no-think
```

This mode compares the original HuggingFace weights against the quantized CoreML model, useful for:
- Isolating quantization-induced divergence from training/baking effects
- Establishing baseline accuracy vs vanilla HF model
- Debugging weight conversion issues

### 2. Batch Dataset Generation

```bash
# Run harness over multiple prompts
python tests/dev/qwen_aq1_divergence_harness.py \
    ~/Downloads/snapped_step1800.pt \
    /Users/anemll/Models/ANE/q4_r32_lut_ka \
    --dataset tests/dev/prompts.jsonl \
    --out-dir runs/exp1 \
    --max-new-tokens 256 \
    --driver coreml \
    --no-think
```

### 3. Analyze Results

```bash
# Generate train/val instability datasets
python tests/dev/analyze_runs.py runs/exp1 --output runs/exp1/analysis
```

**Output files:**
- `train_instability.jsonl` - Prompts that triggered instability
- `val_instability.jsonl` - Held-out unstable prompts (10%)
- `stable_control.jsonl` - Stable prompts (>=98% match)
- `near_boundary.jsonl` - Borderline prompts (95-98% match)
- `diverged_unflagged.jsonl` - Low match but no instability flags
- `metrics.csv` - All metrics for sorting/analysis

---

## Prompt Dataset Format

JSONL with fields:

```json
{
  "id": "prompt_001",
  "prompt": "What is the capital of France?",
  "category": "factual",
  "risk": "low",
  "expected_repetition": false,
  "target_len": "short"
}
```

**Risk categories:**
- `low` - Normal prompts, unlikely to trigger issues
- `repetition` - May cause some repetition
- `high_repetition` - Designed to produce repetitive output (lists, tables)
- `extreme_repetition` - Highly repetitive by design ("count to 100")
- `entropy_collapse` - May cause model to become overly confident

Prompts with `high_repetition` or `extreme_repetition` have relaxed thresholds to avoid false positives.

---

## Instability Detection

### Multi-Signal Detection

The harness uses multiple signals to detect instability:

| Signal | Description | Default Threshold |
|--------|-------------|-------------------|
| `stuck_token` | Same token repeated N times | 10 consecutive |
| `phrase_repeat` | Exact phrase repeated | 8+ tokens, 3+ times |
| `rep4_spike` | 4-gram repetition jumps suddenly | 0.15 → 0.40 in 32 tokens |
| `entropy_collapse` | Very low output entropy | < 0.1 for 16+ steps |
| `margin_explosion` | Top-1 probability too high | > 20 for 4+ steps |

### Status Classification

| Status | Match Rate | Meaning |
|--------|------------|---------|
| `OK` | >= 98% | Models agree, stable |
| `NEAR` | 95-98% | Borderline, worth watching |
| `DIVERGED` | < 95% | Significant disagreement |

**Short decode guard:** Sequences < 64 tokens with <= 1 mismatch are classified as `OK` regardless of rate.

---

## Output Formats

### NPZ Arrays (per prompt)

```python
data = np.load("runs/exp1/prompt_001.npz")

# Token sequences
data["prompt_tokens"]      # Input prompt tokens
data["driver_tokens"]      # Tokens fed to both models
data["pt_argmax"]          # PyTorch predicted tokens
data["cm_argmax"]          # CoreML predicted tokens

# Per-step metrics
data["prompt_kl"]          # KL divergence during prompt
data["decode_kl"]          # KL divergence during decode
data["decode_entropy_cm"]  # CoreML entropy per step
data["decode_margin_cm"]   # CoreML top-1 margin per step
data["decode_maxlogit_cm"] # CoreML max logit per step
data["decode_correlation"] # Logit correlation per step
data["decode_rep4"]        # 4-gram repetition score per step
```

### JSON Summary (per prompt)

```json
{
  "id": "prompt_001",
  "prompt_len": 42,
  "decode_len": 100,
  "decode_match_rate": 0.97,
  "decode_mismatches": 3,
  "stop_reason": "max_tokens",
  "first_stuck_step": null,
  "first_phrase_step": null,
  "first_instability_global_pos": null
}
```

---

## Qwen3 Template Notes

### Thinking Mode

Qwen3 has a "thinking" mode that outputs reasoning in `<think>...</think>` tags.

**With `--no-think` (recommended for testing):**
```
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
<think>

</think>

```
Uses `enable_thinking=False` which pre-fills empty think tags (17 tokens).

**Without `--no-think`:**
```
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
```
Model generates `<think>...</think>` naturally (13 tokens).

### Sampling Recommendations (from Qwen3 docs)

| Mode | Temperature | TopP | TopK |
|------|-------------|------|------|
| Thinking enabled | 0.6 | 0.95 | 20 |
| Thinking disabled | 0.7 | 0.8 | 20 |

**Note:** Greedy decoding (argmax) is appropriate for parity testing but not recommended for production inference.

---

## Common Issues

### 1. Repetition Loops

**Symptom:** Model outputs "English-speaking English-speaking English-speaking..."

**Cause:** Model behavior, not CoreML-specific. Both PyTorch and CoreML agree on the repetition.

**Solutions:**
- Use sampling (temperature > 0) instead of greedy
- Add repetition penalty
- Fine-tune with anti-repetition objective

### 2. High False Positive Rate

**Symptom:** Many prompts flagged as unstable

**Solutions:**
- Check `risk` field in prompts - high_repetition prompts should be expected
- Increase entropy/rep4 thresholds
- Use `--entropy-threshold 0.1` (default) to only flag near-zero entropy

### 3. Template Mismatch

**Symptom:** Different outputs between scripts

**Check:** Verify both use same `enable_thinking=False` approach:
```python
tokenizer.apply_chat_template(messages, enable_thinking=False, ...)
```

---

## Metrics Reference

### Stability Metrics

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `kl_divergence` | KL between PT and CM distributions | < 0.01 |
| `correlation` | Logit correlation | > 0.99 |
| `entropy` | Output distribution entropy | > 0.5 (varies) |
| `top1_margin` | Gap between top-1 and top-2 logits | < 15 |
| `rep4` | 4-gram repetition ratio | < 0.30 |

### Match Rates

| Rate | Interpretation |
|------|----------------|
| 100% | Perfect agreement |
| 95-99% | Minor numerical differences |
| 90-95% | Some divergence, may be concerning |
| < 90% | Significant divergence |

---

## Example Workflow

```bash
# 1. Run smoke test
python tests/dev/qwen_aq1_divergence_harness.py \
    ~/Downloads/snapped_step1800.pt \
    /Users/anemll/Models/ANE/q4_r32_lut_ka \
    --dataset tests/dev/short_prompts.jsonl \
    --out-dir runs/smoke \
    --max-new-tokens 64

# 2. Analyze results
python tests/dev/analyze_runs.py runs/smoke

# 3. Full dataset run
python tests/dev/qwen_aq1_divergence_harness.py \
    ~/Downloads/snapped_step1800.pt \
    /Users/anemll/Models/ANE/q4_r32_lut_ka \
    --dataset tests/dev/prompts.jsonl \
    --out-dir runs/full \
    --max-new-tokens 256

# 4. Generate training data
python tests/dev/analyze_runs.py runs/full --output datasets/instability_v1

# 5. Investigate specific prompt
python tests/dev/test_qwen_aq1_compare.py \
    ~/Downloads/snapped_step1800.pt \
    /Users/anemll/Models/ANE/q4_r32_lut_ka \
    --prompt "History of England and UK" \
    --max-tokens 400 \
    --driver coreml \
    --no-think \
    --prefill-mode batch
```

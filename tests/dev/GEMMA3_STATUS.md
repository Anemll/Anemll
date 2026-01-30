# Gemma3 CoreML status

Last updated: 2025-01-30

## Summary
- Gemma3 4B QAT with FP16 scaling (α=0.1875) now achieves excellent parity with HuggingFace BF16.
- Previous issues with divergence were resolved by implementing proper FP16 overflow prevention.
- 700-token test shows 99.86% match rate with KL divergence of 0.0006.

---

## FP16 Scaling for Gemma3 4B QAT (RECOMMENDED)

### Background
Gemma3 4B QAT models experience FP16 overflow on ANE due to large intermediate activations.
Solution: Apply scaling factor α=0.1875 to embeddings and inverse scaling to LM head logits.

### Scaling Implementation
```
embeddings_scaled = embeddings * 0.1875
logits = lm_head(hidden_states) / 0.1875
```

### Divergence Test Results (2025-01-30)

**Model**: `google/gemma-3-4b-it-qat-int4-unquantized` (FP16 scaled)
**Reference**: HuggingFace BF16 (unscaled)
**Test**: `tests/dev/test_gemma3_compare.py`

| Tokens | KL Divergence | Correlation | Match Rate | First Divergence |
|--------|---------------|-------------|------------|------------------|
| 10     | 0.0032        | 0.9984      | 100%       | -                |
| 100    | 0.0044        | 0.9940      | 99%        | -                |
| 700    | 0.0006        | 0.9954      | 99.86%     | Token 6          |

**Test Command (700 tokens)**:
```bash
python3 tests/dev/test_gemma3_compare.py \
    /Volumes/Models/ANE/gemma3_4b_qat4_scale \
    --hf-reference google/gemma-3-4b-it-qat-int4-unquantized \
    --prompt "History of Ancient Egypt" \
    --max-tokens 700 \
    --driver coreml \
    --no-think
```

**Assessment**: Excellent parity achieved. KL < 0.01 and match rate > 99% indicate the FP16 scaling approach works well for ANE deployment.

---

## LUT Quantization Testing (2025-01-30)

Testing LUT quantization on top of FP16 scaling for Gemma3 4B QAT.

### LUT4 Per-Tensor (`4,0`) - 8 chunks

**Config**: `--lut2 "4,0"` (FFN only, embeddings/lm_head unquantized)
**Output**: `/Volumes/Models/ANE/gemma3_4b_qat4_scale_lut4_tensor`
**Speed**: ~20 t/s

| Tokens | KL Divergence | Correlation | Match Rate | First Divergence |
|--------|---------------|-------------|------------|------------------|
| 100    | 0.693         | 0.753       | 83%        | Token 0          |

**Assessment**: Per-tensor quantization too aggressive. Immediate divergence from token 0. Quality unacceptable.

### LUT4 Per-Channel (`4,8`) FFN + Various LM Head Configs - 2 chunks

**Base Config**: `--lut2 "4,8"` (FFN with per-channel group size 8)
**Output**: `/Volumes/Models/ANE/gemma3_4b_qat4_scale_lut4x8`

#### FFN LUT4,8 Only (no LM head quantization)

| Tokens | KL Divergence | Correlation | Match Rate | First Divergence |
|--------|---------------|-------------|------------|------------------|
| 100    | 0.284         | 0.971       | 87%        | Token 0          |

**Assessment**: Acceptable baseline for LUT. The QAT model already has INT4 weights, so additional LUT quantization compounds errors.

#### FFN LUT4,8 + LM Head LUT6,0 (per-tensor)

| Tokens | KL Divergence | Correlation | Match Rate | First Divergence |
|--------|---------------|-------------|------------|------------------|
| 100    | 0.438         | 0.967       | 81%        | Token 0          |

**Assessment**: Per-tensor LM head quantization degrades quality.

#### FFN LUT4,8 + LM Head LUT4,4 (per-channel)

| Tokens | KL Divergence | Correlation | Match Rate | First Divergence |
|--------|---------------|-------------|------------|------------------|
| 100    | 0.432         | 0.962       | 72%        | Token 0          |

**Assessment**: 4-bit LM head with per-channel still degrades significantly.

#### FFN LUT4,8 + LM Head LUT6,4 (per-channel)

| Tokens | KL Divergence | Correlation | Match Rate | First Divergence |
|--------|---------------|-------------|------------|------------------|
| 100    | 0.279         | 0.970       | 86%        | Token 0          |

**Assessment**: 6-bit per-channel is the best LM head LUT option for 4,8 FFN.

### LUT4 Per-Channel (`4,4`) FFN + LM Head LUT6,4 - 2 chunks - **BEST LUT CONFIG**

**Config**: `--lut2 "4,4"` (FFN with per-channel group size 4) + LM head LUT6,4
**Output**: `/Volumes/Models/ANE/gemma3_4b_qat4_scale_lut4x4`

| Tokens | KL Divergence | Correlation | Match Rate | First Divergence |
|--------|---------------|-------------|------------|------------------|
| 100    | 0.196         | 0.959       | 90%        | Token 0          |

**Assessment**: Per-channel group size 4 outperforms group size 8. This is the best full-LUT configuration for Gemma3 4B QAT, achieving 90% match rate. However, still significantly below FP16 baseline (99.86%).

### LUT Summary for Gemma3 4B QAT

| Configuration | KL | Corr | Match | Recommendation |
|--------------|-----|------|-------|----------------|
| FP16 baseline (no LUT) | 0.0006 | 0.995 | **99.86%** | **RECOMMENDED** |
| FFN LUT4,4 + LM LUT6,4 | 0.196 | 0.959 | **90%** | **Best LUT option** |
| FFN LUT4,8 only | 0.284 | 0.971 | 87% | Acceptable if size critical |
| FFN LUT4,8 + LM LUT6,4 | 0.279 | 0.970 | 86% | Good LUT option |
| FFN LUT4,8 + LM LUT6,0 | 0.438 | 0.967 | 81% | Not recommended |
| FFN LUT4,8 + LM LUT4,4 | 0.432 | 0.962 | 72% | Not recommended |
| FFN LUT4 per-tensor | 0.693 | 0.753 | 83% | Not recommended |

**Key Finding**: QAT models (already INT4 optimized) do not tolerate additional LUT quantization well. For Gemma3 4B QAT, the FP16 scaled baseline without LUT provides the best quality-to-performance ratio.

---

## Historical Results (Pre-Scaling)

### Latest divergence results (ctx1024) - Gemma3 270M (unscaled)
Batch run with 3 short prompts (driver=coreml):
- KL mean: 0.18599
- Logit correlation mean: 0.98459
- Entropy mean: 1.22222
- Match rate: 0.93229

Single prompt example:
- First divergence at step 19 on "What is the capital of France?"

## Model locations
- Local conversion outputs:
  - /tmp/gemma3_convert_model
  - /tmp/gemma3_convert_model_ctx1024
  - /tmp/gemma3_convert_model_ctx4096
- Persistent copies:
  - ~/Models/ANE/gemma3_convert_model
  - ~/Models/ANE/gemma3_convert_model_ctx1024
  - ~/Models/ANE/gemma3_convert_model_ctx4096

## Conversion workflow
Use the standard converter script:
```
./anemll/utils/convert_model.sh \
  --model /path/to/hf/model \
  --output /tmp/gemma3_convert_model_ctx1024 \
  --context 1024 \
  --batch 64 \
  --chunk 1 \
  --prefix gemma3 \
  --skip-check
```
Notes:
- Gemma3 uses split_lm_head=16 due to large vocab.
- Tokenizer files should be copied into the output directory (tokenizer.json, tokenizer.model, tokenizer_config.json).

## Regenerate meta.yaml only
If models already exist, regenerate meta without reconversion:
```
python3 anemll/utils/generate_meta_yaml.py \
  google-gemma-3-270m-it \
  1024 64 none none none 1 gemma3 gemma3 /tmp/gemma3_convert_model_ctx1024
```
This ensures `split_lm_head=16` and correct filenames.

## Quick sanity test
```
TOKENIZERS_PARALLELISM=false \
python tests/chat.py \
  --meta /tmp/gemma3_convert_model_ctx1024/meta.yaml \
  --prompt "What is the capital of France?" \
  --max-tokens 128
```

## Divergence harness
Single prompt:
```
TOKENIZERS_PARALLELISM=false \
python tests/dev/test_gemma3_compare.py \
  --hf-reference google/gemma-3-270m-it \
  /tmp/gemma3_convert_model_ctx1024 \
  --prompt "What is the capital of France?" \
  --max-tokens 100 \
  --driver coreml \
  --no-think
```

Batch run:
```
TOKENIZERS_PARALLELISM=false \
python tests/dev/gemma3_divergence_harness.py \
  --hf-reference google/gemma-3-270m-it \
  /tmp/gemma3_convert_model_ctx1024 \
  --dataset tests/dev/short_prompts.jsonl \
  --out-dir runs/hf_vs_ane_ctx1024 \
  --max-new-tokens 256 \
  --driver coreml \
  --no-think
```

Metric guidance:
- KL divergence: healthy < 0.01, concerning > 0.1
- Logit correlation: healthy > 0.99, concerning < 0.95
- Match rate: healthy > 98%, concerning < 90%
- Entropy: healthy > 0.5, collapse risk < 0.1

## Known issues (Historical - mostly resolved with scaling)
- ~~Divergence appears after multiple tokens even without prefill.~~ Resolved with FP16 scaling.
- Near-tie logits can flip argmax at FP16, which compounds over steps (minor effect with scaling).
- ~~Long generations may repeat or collapse.~~ Resolved with FP16 scaling.

## Next steps
- ~~Test quantized (LUT4/LUT6) versions of scaled 4B model.~~ **DONE** - LUT not recommended for QAT models
- Validate sliding window rotation for context > 1024.
- Benchmark ANE inference speed vs HuggingFace.
- Test LUT quantization on non-QAT models (e.g., base Gemma3 or LLaMA) where LUT may perform better.

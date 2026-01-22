# Gemma3 CoreML status

Last updated: 2025-01-22

## Summary
- CoreML Gemma3 models run and generate, but diverge from HF over time.
- Divergence is not prefill-only; token-by-token inference diverges as well.
- KV cache write positions are correct; values drift and accumulate over tokens.
- First divergence on a short prompt often appears within ~20 steps.

## Latest divergence results (ctx1024)
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

## Known issues
- Divergence appears after multiple tokens even without prefill.
- Near-tie logits can flip argmax at FP16, which compounds over steps.
- Long generations may repeat or collapse.

## Next steps
- Run the divergence harness on a higher-memory machine for longer prompts.
- Add top-k logit delta logging around the first divergence step.
- Explore stability tweaks (e.g., logit scaling or sampling) after parity is understood.

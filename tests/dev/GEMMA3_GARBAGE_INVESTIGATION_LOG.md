# Gemma3 Garbage Output Investigation Log

Date: 2026-02-11
Branch: `qwen3.5`

## Scope
Investigate multilingual/garbled outputs from Gemma3 270M conversions when running:
- `tests/chat.py`
- `tests/chat_full.py`
- Swift runtime (`anemll-swift-cli/anemllcli`)

## Confirmed Findings
1. The issue is **not only Python chat loop**:
   - Reproduced in `tests/chat.py`
   - Reproduced in `tests/chat_full.py`
   - Reproduced in Swift CLI (`anemllcli`)

2. The issue is **not fixed by disabling anemll-dedup at combine time**:
   - Recombined `/tmp/test-gemma3-270m` with `--skip-anemll-dedup`
   - Recompiled model
   - Output stayed the same (garbled)

3. Argmax mapping path looked internally coherent (no obvious index range bug):
   - `tests/chat.py --debug-argmax` showed valid chunk-local/global mapping and in-range indices

4. Current likely suspect remains conversion/export quality path (quantization and/or model path), not `chat_full` formatting logic.

## Commands Executed (key)

### A/B combine with dedup OFF (no default code changes)
```bash
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/env-anemll/bin/python3 \
  anemll/utils/combine_models.py --monolithic --lut 4 --prefix gemma3 \
  --input /tmp/test-gemma3-270m --output /tmp/test-gemma3-270m --skip-anemll-dedup

/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/env-anemll/bin/python3 \
  anemll/utils/compile_models.py monolithic --lut 4 --prefix gemma3 \
  --input /tmp/test-gemma3-270m --output /tmp/test-gemma3-270m
```

### Repro runs (still garbled)
```bash
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/env-anemll/bin/python3 \
  tests/chat.py --meta /tmp/test-gemma3-270m/meta.yaml --prompt "who are you?" \
  --max-tokens 64 --no-think

/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/env-anemll/bin/python3 \
  tests/chat_full.py --meta /tmp/test-gemma3-270m/meta.yaml --prompt "who are you?" \
  --max-tokens 64 --no-think

cd anemll-swift-cli
swift run anemllcli --meta /tmp/test-gemma3-270m/meta.yaml \
  --prompt "who are you?" --max-tokens 64 --temperature 0.0
```

## Disk/Space Status at Stop
- `/System/Volumes/Data` almost full (about 3.4 GiB free)
- `/Volumes/Models` had more free space (about 47 GiB)

## Partial Outputs Present
- ` /tmp/test-gemma3-270m` (LUT4 combined/compiled path complete)
- ` /tmp/test-gemma3-270m-fp16` (FP16/no-LUT partially generated)
- ` /Volumes/Models/ANE/test-gemma3-270m-fp16` (FP16/no-LUT infer package only)

## Next Step After Reboot/Cleanup
Goal: run a clean **no-LUT (FP16)** end-to-end build on `/Volumes/Models/ANE` and compare quality.

### Recommended clean rerun (all steps explicitly)
```bash
source /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-0.3.5/env-anemll/bin/activate

export TMPDIR=/Volumes/Models/ANE/tmp_coreml_compile
export TMP=/Volumes/Models/ANE/tmp_coreml_compile
export TEMP=/Volumes/Models/ANE/tmp_coreml_compile

MODEL=/Users/anemll/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3
OUT=/Volumes/Models/ANE/test-gemma3-270m-fp16

rm -rf "$OUT"
mkdir -p "$OUT"

bash anemll/utils/convert_monolith.sh --model "$MODEL" --output "$OUT" --context 512 --batch 64 --lut none --prefix gemma3 --argmax --skip-check --skip-anemll-dedup --only 1
bash anemll/utils/convert_monolith.sh --model "$MODEL" --output "$OUT" --context 512 --batch 64 --lut none --prefix gemma3 --argmax --skip-check --skip-anemll-dedup --only 2
bash anemll/utils/convert_monolith.sh --model "$MODEL" --output "$OUT" --context 512 --batch 64 --lut none --prefix gemma3 --argmax --skip-check --skip-anemll-dedup --only 3
bash anemll/utils/convert_monolith.sh --model "$MODEL" --output "$OUT" --context 512 --batch 64 --lut none --prefix gemma3 --argmax --skip-check --skip-anemll-dedup --only 4
bash anemll/utils/convert_monolith.sh --model "$MODEL" --output "$OUT" --context 512 --batch 64 --lut none --prefix gemma3 --argmax --skip-check --skip-anemll-dedup --only 5
```

### Validation commands after rebuild
```bash
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/env-anemll/bin/python3 \
  tests/chat.py --meta /Volumes/Models/ANE/test-gemma3-270m-fp16/meta.yaml \
  --prompt "who are you?" --max-tokens 64 --no-think

/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/env-anemll/bin/python3 \
  tests/chat_full.py --meta /Volumes/Models/ANE/test-gemma3-270m-fp16/meta.yaml \
  --prompt "who are you?" --max-tokens 64 --no-think

cd anemll-swift-cli
swift run anemllcli --meta /Volumes/Models/ANE/test-gemma3-270m-fp16/meta.yaml \
  --prompt "who are you?" --max-tokens 64 --temperature 0.0
```

## Important Constraint Followed
- No change made to make dedup disabled by default.
- All dedup-off tests were done by passing CLI flag (`--skip-anemll-dedup`) only.

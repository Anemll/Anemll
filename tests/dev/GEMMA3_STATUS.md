# Gemma3n Status (ANEMLL)

Date: 2026-01-22

## Snapshot
- Code under test: `anemll/models/gemma3n_model.py`
- Local model path used: `/Users/anemll/Models/Models/gemma-3n-E2B-it`

## ✅ PyTorch Parity (Text-only)
- **Test**: `env-anemll/bin/python tests/dev/test_gemma3n_vs_hf_pytorch.py --model /Users/anemll/Models/Models/gemma-3n-E2B-it --device mps --dtype float16`
- **Result** (prompt: “The capital of France is”):
  - cosine: **0.9975**
  - mse: **2.0963**
  - top‑20 overlap: **0.85**
  - HF top‑1: **' Paris'**
  - ANEMLL top‑1: **' Paris'**

## Key Fixes Applied (since last status)
- Implemented HF‑style **4‑stream AltUp pipeline** across layers.
- Added **per‑layer projection path** (`per_layer_model_projection`, `per_layer_projection_norm`, scaling).
- Implemented HF‑style **LAUREL residual add** inside the LAUREL block.
- Added **altup_unembed_projections** + proper unembed aggregation.
- Added text‑only HF attention/MLP layers and updated weight loading mappings.
- Fixed RoPE caching slicing/device alignment and removed shadowed helpers.
- RMSNorm stays **ANE‑compatible** (mean‑sub + `F.layer_norm`).
- Switched internal `position_ids` generation to **int32** to avoid CoreML dtype warnings.

## Converter Status (Text‑only)
- `anemll/utils/convert_model.sh` **routes Gemma3n** to `anemll.ane_converter.gemma3n_converter`.
- `anemll/utils/check_dependencies.sh` accepts `gemma3n`.
- `gemma3n_converter.py`:
  - Uses repo venv python when available (`env-anemll/bin/python`).
  - Added flags: `--disable-laurel`, `--disable-per-layer-embeddings`, `--enable-multimodal`, `--disable-sparsity`.
  - Added `disable_sparsity` handling (zeroes activation sparsity pattern).
  - Adjusted FFN conversion to handle Conv2d layout when LAUREL disabled.
  - Fixed attention prefill conversion to use `input_layernorm` and `post_attention_layernorm`.

## Known Gaps / Future Work
- Attention prefill conversion can OOM at `--context 512` on 16‑32GB RAM systems; needs more RAM or attention‑chunking.
- Converter still uses `Gemma3nLaurelBlock` for export (not the HF‑parity text path).
- Dense layers are still `nn.Linear` in the PyTorch text‑only path (Conv2d refactor pending for ANE).

## Dev Test Commands
### PyTorch parity (HF vs ANEMLL)
- `env-anemll/bin/python tests/dev/test_gemma3n_vs_hf_pytorch.py --model /Users/anemll/Models/Models/gemma-3n-E2B-it --device mps --dtype float16`

### Conversion (full)
- `./anemll/utils/convert_model.sh --model /Users/anemll/Models/Models/gemma-3n-E2B-it --output /tmp/gemma3n-converted --context 512 --batch 1 --chunk 2 --disable-laurel --disable-per-layer-embeddings --disable-sparsity`

### Conversion (part-by-part, recommended for debugging)
- `env-anemll/bin/python tests/dev/export_gemma3n.py --model /Users/anemll/Models/Models/gemma-3n-E2B-it --output /tmp/gemma3n-test --context-length 128 --batch-size 1 --chunk 2 --disable-laurel --disable-per-layer-embeddings --part embeddings`
- `env-anemll/bin/python tests/dev/export_gemma3n.py --model /Users/anemll/Models/Models/gemma-3n-E2B-it --output /tmp/gemma3n-test --context-length 128 --batch-size 1 --chunk 2 --disable-laurel --disable-per-layer-embeddings --part ffn`
- `env-anemll/bin/python tests/dev/export_gemma3n.py --model /Users/anemll/Models/Models/gemma-3n-E2B-it --output /tmp/gemma3n-test --context-length 128 --batch-size 1 --chunk 2 --disable-laurel --disable-per-layer-embeddings --part attention`
- `env-anemll/bin/python tests/dev/export_gemma3n.py --model /Users/anemll/Models/Models/gemma-3n-E2B-it --output /tmp/gemma3n-test --part lm_head`
- `env-anemll/bin/python tests/dev/export_gemma3n.py --model /Users/anemll/Models/Models/gemma-3n-E2B-it --output /tmp/gemma3n-test --part tokenizer`

## Next Steps (per plan)
1. Run conversion on a higher‑RAM system at `--context 512` (attention prefill can OOM).
2. Decide on attention‑chunking strategy if OOM persists.
3. Reconcile converter layer class with HF‑parity text path once CoreML conversion is stable.

# Chunk Split Calculator

The `calc_chunk_split.py` utility calculates weight sizes and recommends optimal `--chunk` values for model conversion. It reads `config.json` from any HuggingFace model and computes per-layer weight sizes at FP16 and various LUT quantization levels.

## Supported Models

Works with any standard HuggingFace decoder-only LLM:
- LLaMA 3.x (1B, 8B)
- Qwen 2.5, Qwen 3 (0.6B–14B)
- Gemma, Gemma 2, Gemma 3 (270M–4B), Gemma 3n
- DeepSeek, DeepHermes
- Any model with a standard `config.json`

Handles nested multimodal configs (e.g., Gemma3 `text_config`), per-layer `intermediate_size` arrays (Gemma3n), tied embeddings, attention/MLP bias variations, and GQA/MQA.

## Usage

### Interactive Report

```bash
# Full report with all chunk options
python anemll/utils/calc_chunk_split.py Qwen/Qwen3-4B

# With target max chunk size
python anemll/utils/calc_chunk_split.py --max-chunk-mb 950 --lut 6 google/gemma-3-4b-it-qat-int4-unquantized

# From a local path
python anemll/utils/calc_chunk_split.py /path/to/model/directory
```

### Auto Mode (Scripting)

Returns a single integer — the recommended `--chunk` value:

```bash
# Get optimal chunk count for LUT6, max 950MB per chunk
python anemll/utils/calc_chunk_split.py --auto --lut 6 Qwen/Qwen3-4B
# Output: 4

# Small model → monolithic (1 chunk)
python anemll/utils/calc_chunk_split.py --auto --lut 6 google/gemma-3-1b-it
# Output: 1
```

### With convert_model.sh

Use `--chunk auto` to automatically determine the optimal chunk count:

```bash
./anemll/utils/convert_model.sh \
    --model google/gemma-3-4b-it-qat-int4-unquantized \
    --output /Volumes/Models/ANE/gemma3-4b-qat-ctx4096 \
    --context 4096 \
    --lut2 6,4 \
    --chunk auto
# Auto-detected: --chunk 3 (LUT6, max 950MB per chunk with 10% overhead)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | (required) | Path to model dir, HF cache path, or HF model ID |
| `--max-chunk-mb` | 950 | Target max chunk size in MB |
| `--lut` | show all | LUT quantization bits (4, 6, 8, or 16 for FP16) |
| `--overhead` | 1.10 | Safety overhead multiplier (10% by default) |
| `--auto` | off | Output only the recommended chunk count |

## How It Works

1. **Reads `config.json`** from the model (local path, HF cache, or model ID)
2. **Calculates per-layer weight sizes** based on architecture parameters (hidden_size, intermediate_size, num_heads, etc.)
3. **Simulates balanced chunk splits** using the same algorithm as the converters
4. **Applies LUT compression ratio** (e.g., LUT6 = 6/16 = 37.5% of FP16 size)
5. **Applies safety overhead** (default 10%) to account for metadata, alignment, and quantization overhead
6. **Recommends minimum chunks** where the largest chunk fits within the target size

### Overhead

The 10% default overhead accounts for:
- CoreML model metadata and structure overhead
- Weight alignment padding in `.mlpackage` files
- Quantization lookup tables
- Minor size variations between estimated and actual weights

## Example Output

```
======================================================================
Model Weight Size Calculator
======================================================================
  Architecture:    Qwen3ForCausalLM
  Model type:      qwen3
  Hidden size:     2560
  Intermediate:    9728
  Layers:          36
  Attention heads: 32 (KV: 8)
  Head dim:        128
  Vocab size:      151936
  Tied embeddings: True
  Overhead:        110%

Component Sizes (FP16):
  Embeddings:        741.9 MB
  LM Head:         (tied with embeddings)
  FFN Body:            6.8 GB  (36 layers)
  Total:               7.5 GB  (4.02B params)

Chunk Split Options:
  (sizes include 110% overhead)

  Chunks                Layout    FP16 max    LUT8 max    LUT6 max    LUT4 max
       1                    36      7.4 GB      3.7 GB      2.8 GB      1.9 GB
       2                 18+18      3.7 GB      1.9 GB      1.4 GB    952.9 MB
       3              12+12+12      2.5 GB      1.2 GB    952.9 MB    635.3 MB
       4               9+9+9+9      1.9 GB    952.9 MB    714.7 MB    476.5 MB

Recommendation (max chunk: 950 MB, overhead: 110%):
   FP16: --chunk  9
   LUT8: --chunk  5
   LUT6: --chunk  4
   LUT4: --chunk  3
```

## Model Input Formats

The `model` argument accepts:

1. **HuggingFace model ID**: `Qwen/Qwen3-4B` (looks up HF cache automatically)
2. **Local directory**: `/path/to/model/` (must contain `config.json`)
3. **HF cache path**: `~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/<hash>`
4. **Direct config path**: `/path/to/config.json`

## See Also

- [convert_model.sh Documentation](convert_model.md) — `--chunk auto` integration
- [ANEMLL-Dedup](anemll-dedup.md) — Weight deduplication for combined models
- [FP16 Scaling](FP16_SCALING.md) — Gemma3 FP16 overflow handling

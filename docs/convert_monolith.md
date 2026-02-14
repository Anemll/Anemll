# Monolithic Model Conversion Script Documentation

The `convert_monolith.sh` script converts LLM models to a single CoreML file containing embeddings, FFN (transformer layers), and LM head combined into one monolithic model.

## Overview

Unlike the standard chunked conversion (`convert_model.sh`) which produces multiple model files, monolithic conversion creates a single unified model package. This approach:

- **Simplifies deployment**: Single model file to manage
- **Reduces overhead**: No inter-model communication
- **Smaller total size**: Eliminates redundant metadata and structure
- **Reduced kernel launch overhead**: Single model execution reduces ANE scheduling latency
- **Easier debugging**: Enables direct comparison of inference outputs for detecting divergence or errors

> **Note**: Monolithic models are best suited for smaller models (1B-3B parameters). For larger models (8B+), chunked conversion may still be preferred due to iOS file size limits.

## Usage

```bash
./anemll/utils/convert_monolith.sh --model <path_to_model> --output <output_directory> [options]
```

## Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `--model` | Path to the model directory (HuggingFace format) |
| `--output` | Output directory for converted models |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--context` | Context length | 512 |
| `--batch` | Batch size for prefill | 64 |
| `--lut` | LUT bits for all components (see format below) | 4 |
| `--prefix` | Prefix for model names | auto-detect |
| `--restart` | Restart from specific step (1-6) | 1 |
| `--only` | Run only specified step and exit (1-6) | none |
| `--skip-check` | Skip dependency check | false |

### LUT Quantization Format

The `--lut` argument supports two formats:

1. **Simple format**: `--lut 4`
   - Specifies only the number of quantization bits
   - Uses default per_channel group size of 8

2. **Advanced format**: `--lut 4,16`
   - First value (4): Number of quantization bits
   - Second value (16): Per-channel group size for grouped quantization

## Examples

### Basic Monolithic Conversion
```bash
./anemll/utils/convert_monolith.sh \
    --model ./models/Qwen3-0.6B \
    --output ./converted \
    --lut 4
```

### With Custom Per-Channel Group Size
```bash
./anemll/utils/convert_monolith.sh \
    --model ./models/Qwen3-0.6B \
    --output ./converted \
    --lut 4,16
```

### High Quality 6-bit Conversion
```bash
./anemll/utils/convert_monolith.sh \
    --model ./models/Meta-Llama-3.2-1B \
    --output ./converted \
    --context 1024 \
    --lut 6
```

### Resume from Step 4
```bash
./anemll/utils/convert_monolith.sh \
    --model ./models/Qwen3-0.6B \
    --output ./converted \
    --restart 4
```

## Conversion Steps

The script performs these steps in sequence:

| Step | Description | Output |
|------|-------------|--------|
| 1 | Convert Monolithic Inference Model | `{prefix}_monolithic.mlpackage` |
| 2 | Convert Monolithic Prefill Model | `{prefix}_monolithic_prefill.mlpackage` |
| 3 | Combine into Multi-Function Model | `{prefix}_monolithic_combined.mlpackage` |
| 4 | Compile to .mlmodelc | `{prefix}_monolithic_combined.mlmodelc/` |
| 5 | Copy tokenizer & create meta.yaml | `tokenizer.json`, `meta.yaml` |
| 6 | Test with chat.py | Validation output |

## Supported Architectures

The script auto-detects the model architecture from `config.json`:

| Architecture | Converter Used | Default Prefix |
|--------------|----------------|----------------|
| LLaMA / DeepSeek | `llama_converter` | `llama` |
| Qwen 2.5 | `qwen2_5_converter` | `qwen25` |
| Qwen 3 | `qwen_converter` | `qwen` |

## Output Files

After successful conversion, your output directory will contain:

```
converted/
├── tokenizer.json
├── tokenizer_config.json
├── config.json
├── meta.yaml
├── {prefix}_monolithic.mlpackage
├── {prefix}_monolithic_prefill.mlpackage
├── {prefix}_monolithic_combined.mlpackage
└── {prefix}_monolithic_combined.mlmodelc/
```

## meta.yaml Configuration

The script generates a `meta.yaml` file with `model_type: monolithic`:

```yaml
model_info:
  name: anemll-ModelName-ctx512
  version: 0.3.0
  parameters:
    model_type: monolithic
    context_length: 512
    batch_size: 64
    lut_ffn: 4
    model_prefix: qwen
```

This configuration allows chat interfaces to automatically detect and load monolithic models:

```bash
# Basic chat with monolithic model
python ./tests/chat.py --meta ./converted/meta.yaml

# Full conversation mode
python ./tests/chat_full.py --meta ./converted/meta.yaml
```

## Comparison: Monolithic vs Chunked

| Feature | Monolithic | Chunked |
|---------|------------|---------|
| Number of files | 1 model package | Multiple (embed + LM head + N chunks) |
| Best for | Small models (1B-3B) | Large models (8B+) |
| iOS compatibility | Good for small models | Better for large models |
| Deployment complexity | Simple | More complex |
| Memory usage | All at once | Chunked loading possible |

## Troubleshooting

If conversion fails:

1. **Check error messages** for the failing step
2. **Use `--restart N`** to resume from step N
3. **Verify permissions** on input/output directories
4. **Ensure disk space** is available
5. **Validate model files** exist in the source directory

### Common Issues

| Issue | Solution |
|-------|----------|
| "Model directory does not exist" | Check the `--model` path |
| "Unknown architecture" | Ensure `config.json` has valid `model_type` |
| Step 3 fails | Verify both inference and prefill models exist |
| Step 4 fails | Check Xcode Command Line Tools are installed |

## Manual Conversion Steps

For fine-grained control, you can run individual converter commands:

### Step 1: Convert Monolithic Inference Model
```bash
python -m anemll.ane_converter.llama_converter \
    --part monolithic \
    --lut 4 \
    --context-length 512 \
    --batch-size 64 \
    --model ./models/Meta-Llama-3.2-1B \
    --output ./converted
```

### Step 2: Convert Monolithic Prefill Model
```bash
python -m anemll.ane_converter.llama_converter \
    --part monolithic_prefill \
    --lut 4 \
    --context-length 512 \
    --batch-size 64 \
    --model ./models/Meta-Llama-3.2-1B \
    --output ./converted
```

### Step 3: Combine Models
```bash
python ./anemll/utils/combine_models.py \
    --monolithic \
    --lut 4 \
    --prefix llama \
    --input ./converted \
    --output ./converted
```

### Step 4: Compile
```bash
python ./anemll/utils/compile_models.py \
    monolithic \
    --lut 4 \
    --prefix llama \
    --input ./converted \
    --output ./converted
```

## See Also

- [Standard Model Conversion](convert.md) - Chunked conversion workflow
- [Convert Model Script](convert_model.md) - Detailed chunked conversion options
- [Combine Models](combine_models.md) - Model combining utility
- [Compile Models](compile_models.md) - Model compilation details

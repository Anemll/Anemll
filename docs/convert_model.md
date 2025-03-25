# Convert Model Script Documentation

The `convert_model.sh` script automates the conversion of LLAMA models to CoreML format for Apple Neural Engine.

## Usage

```bash
./anemll/utils/convert_model.sh --model <path_to_model> --output <output_directory> [options]
```

## Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `--model` | Path to the model directory |
| `--output` | Output directory for converted models |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--context` | Context length | 512 |
| `--batch` | Batch size for prefill | 64 |
| `--lut1` | LUT bits for embeddings | none |
| `--lut2` | LUT bits for FFN/prefill | 4 |
| `--lut3` | LUT bits for LM head | 6 |
| `--restart` | Restart from specific step | 1 |
| `--only` | Run only specified step and exit | none |
| `--chunk` | Number of chunks to split FFN/prefill | 2 |
| `--prefix` | Prefix for model names | llama |

## New Arguments

- `--skip-check`: Skip the dependency check step.

## Examples

### Run Single Step
```bash
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models \
    --only 2    # Only run step 2 (Convert LM Head)
```

### Basic Conversion
```bash
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models
```

### Single Chunk Conversion
```bash
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models \
    --chunk 1
```

### High Quality Conversion
```bash
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models \
    --context 1024 \
    --lut2 6 \
    --lut3 6
```

### Standard Conversion (6-bit LUT)
```bash
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models \
    --context 1024 \
    --batch 64 \
    --lut2 6 \
    --lut3 6 \
    --chunk 2
```

### Fast Model (4-bit LUT)
```bash
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models_fast \
    --context 512 \
    --batch 64 \
    --lut2 4 \
    --lut3 4 \
    --chunk 2
```
> Note: 4-bit quantization provides faster inference at the cost of slightly reduced quality

### No Quantization
```bash
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models_fp16 \
    --context 1024 \
    --batch 64 \
    --lut1 "" \
    --lut2 "" \
    --lut3 "" \
    --chunk 2
```
> Note: No LUT quantization uses FP16 precision, providing best quality but slower inference and larger model size

### DeepSeek Model Conversion
```bash
./anemll/utils/convert_model.sh \
    --model ../DeepSeekR1-8B \
    --output ./converted_models_deepseek \
    --context 1024 \
    --batch 64 \
    --lut2 6 \
    --lut3 6 \
    --chunk 8 \
    --prefix DeepSeek
```
> Note: The --prefix parameter changes output filenames from llama_* to DeepSeek_*

## Control Options

### Using --restart
The `--restart` option lets you resume conversion from a specific step e.g when you run the script and it fails part way through:
```bash
# Resume from step 5 (Combine Models)
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models \
    --restart 5
```

### Using --only
The `--only` option runs a single step and exits:
```bash
# Only run step 6 (Compile Models)
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models \
    --only 6
```

Available steps:
1. Convert Embeddings
2. Convert LM Head
3. Convert FFN
4. Convert Prefill
5. Combine Models
6. Compile Models
7. Copy Tokenizer & Create meta.yaml
8. Test Conversion

## Conversion Steps

The script performs these steps in sequence:

1. **Convert Embeddings (Part 1)**
   - Creates `llama_embeddings.mlpackage`
   - Optional LUT quantization via `--lut1`

2. **Convert LM Head (Part 3)**
   - Creates `llama_lm_head_lutX.mlpackage`
   - Uses LUT quantization specified by `--lut3`

3. **Convert FFN (Part 2)**
   - Creates `llama_FFN_lutX_chunk_NNofMM.mlpackage` files
   - Uses LUT quantization specified by `--lut2`
   - Splits into chunks based on `--chunk` parameter

4. **Convert Prefill Models**
   - Creates prefill models for KV cache optimization
   - Uses same settings as FFN conversion

5. **Combine Models**
   - Merges FFN and prefill chunks
   - Creates combined `llama_FFN_PF_lutX_chunk_NNofMM.mlpackage` files

6. **Compile Models**
   - Converts all parts to MLModelC format
   - Creates device-ready inference models

7. **Copy Tokenizer Files**
   - Copies required tokenizer files
   - Creates meta.yaml configuration file containing:
     - Model name and version
     - Context length and batch size
     - LUT quantization settings
     - Number of chunks
     - Model prefix for filenames

8. **Test Conversion**
   - Runs basic chat test
   - Prints command for running chat.py

## Using meta.yaml

The script generates a meta.yaml file containing all model configuration:

```yaml
model_info:
  name: anemll-ModelName-ctx1024
  version: 0.3.0
  parameters:
    context_length: 1024
    batch_size: 64
    lut_embeddings: none
    lut_ffn: 6
    lut_lmhead: 6
    num_chunks: 2
    model_prefix: llama
```

You can use this file to run chat without specifying individual model paths:

```bash
# Basic chat
python ./tests/chat.py --meta ./converted_models/meta.yaml

# Full conversation mode
python ./tests/chat_full.py --meta ./converted_models/meta.yaml
```

This automatically loads all settings and model paths based on the meta.yaml configuration.

## Output Files

After successful conversion, your output directory will contain:

```
converted_models/
├── tokenizer.json
├── tokenizer_config.json
├── meta.yaml
├── llama_embeddings.mlpackage
├── llama_embeddings.mlmodelc/
├── llama_lm_head_lutX.mlpackage
├── llama_lm_head_lutX.mlmodelc/
├── llama_FFN_PF_lutX_chunk_NNofMM.mlpackage
└── llama_FFN_PF_lutX_chunk_NNofMM.mlmodelc/
```

Where:
- `X` is the LUT bits value used
- `NN` is the chunk number
- `MM` is the total number of chunks

## Troubleshooting

If conversion fails:

For troubleshooting common issues, please refer to the [Troubleshooting Guide](./troubleshooting.md).

1. Check error messages for the failing step
2. Use `--restart N` to resume from a specific step
3. Verify input/output directory permissions
4. Ensure enough disk space is available
5. Check that all required files exist in model directory

## See Also

- [Model Conversion Guide](convert.md)
- [DeepSeek Model Guide](ConvertingDeepSeek.md)
- [Model Compilation](compile_models.md)
- [Model Combining](combine_models.md) 
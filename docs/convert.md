# ANEMLL Convert Workflow

> [!Important]
> This guide explains how to convert and run LLAMA models using ANEMLL.
>
> **Source Models:**
> - Note, only LLAMA architecture is supported im 0.3.0-alpha release
> - Quantized models like GUFF or MLX are not supported yet
> 
> **Size Limits:**
> - iOS models are limited to 1GB per file
> - macOS supports up to ~2GB per file
> - Models are automatically split during conversion to meet these limits
> 
> **System Requirements:**
> - Requires macOS 15 or later
> 
> **Performance vs Quality:**
> - Default is 6-bit LUT quantization (better quality, slower)
> - 4-bit LUT quantization is faster but slightly lower quality
> - For fastest performance, use 4-bit LUT with 512 context length

# Quick Conversion Using Script

For convenience, you can use the provided conversion script to perform all steps in one go. For detailed information about batch conversion, see [Convert Model Script](convert_model.md).

```bash
./anemll/utils/convert_model.sh --model <path_to_original_model_directory> --output <output_directory> [options]
```

Basic usage example:
```bash
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models
```

> See [Convert Model Guide](convert_model.md) for detailed parameters and examples.

## Model Size and Chunks

Choose the number of chunks based on your model size. Here are the recommended configurations:

- 1B parameter models: 1 or more chunks
  ```bash
  --chunk 1  # Single chunk for basic use and LUT 6 or lower
  ```
- 3B parameter models: 2 or more chunks
  ```bash
  --chunk 2  # Minimum recommended for basic use and LUT 6 or lower
  ```
- 8B parameter models: 8 or more chunks
  ```bash
  --chunk 8  # Minimum recommended for basic use and LUT 6 or lower
  ```

Choose chunk count based on your model size and available system memory.

Complete examples for different model sizes:
```bash
# 1B model with single chunk (basic)
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models \
    --chunk 1 \
    --lut2 6 \
    --lut3 6

# 3B model with recommended chunks
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-3B \
    --output ./converted_models \
    --chunk 2 \
    --lut2 6 \
    --lut3 6

# 8B model with recommended chunks
./anemll/utils/convert_model.sh \
    --model ../DeepSeek-8B \
    --output ./converted_models \
    --chunk 8 \
    --lut2 6 \
    --lut3 6 \
    --prefix DeepSeek
```


## Changing Context Size

To change the context size during conversion, use the `--context` option. This allows you to specify the maximum context length for the model.

Example:
```bash
# Convert with a context size of 1024
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models \
    --context 1024
```

## Restarting and Running Single Steps

### Restarting from a Specific Step
If the conversion process fails or you need to resume from a specific step, use the `--restart` option. This allows you to continue the conversion from the specified step without starting over.

Example:
```bash
# Resume from step 5 (Combine Models)
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models \
    --restart 5
```

### Running a Single Step
To execute only a specific step and then exit, use the `--only` option. This is useful for testing or when you only need to perform one part of the conversion process.

Example:
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

## Troubleshooting

If conversion fails:

1. Check error messages for the failing step.
2. Use `--restart N` to resume from a specific step.
3. Verify input/output directory permissions.
4. Ensure enough disk space is available.
5. Check that all required files exist in the model directory.

Convert Model Script check dependencies on startup. If you're confident the dependency check is incorrectly failing, you can skip it:
```bash
# Skip dependency checks (use with caution)
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models \
    --skip-check
```

For more detailed troubleshooting, please refer to the [Troubleshooting Guide](./troubleshooting.md).

## Preparing for Hugging Face Upload

For detailed instructions on uploading models to Hugging Face, see [Model Upload Guide](./upload_model.md).

Quick example:
```bash
# Prepare model for Hugging Face
./anemll/utils/prepare_hf.sh \
    --input ./converted_models \
    --output ./hf_model \
    --org anemll
```


After conversion, test your model using the chat interface:
```bash
# Basic chat interface
python ./tests/chat.py --meta ./converted_models/meta.yaml

# Full chat interface with conversation history
python ./tests/chat_full.py --meta ./converted_models/meta.yaml
```

See [Chat Guide](./chat.md) for detailed usage of chat interfaces.

## See Also
- [Manual Conversion Steps](manual_conversion.md) - For detailed conversion steps
- [Converting DeepSeek Models](ConvertingDeepSeek.md) - For larger model conversion
- [Compile Models Documentation](compile_models.md) - Details about compilation
- [Combine Models Documentation](combine_models.md) - Details about model combining

    


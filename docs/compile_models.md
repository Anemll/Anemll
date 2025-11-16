# Compile Models Documentation

The `compile_models.py` utility is the final step in the ANEMLL workflow, responsible for compiling MLPackage models into MLModelC format for on-device inference.

## Purpose

This tool compiles the converted and combined models into an optimized format that can be efficiently executed on Apple Neural Engine (ANE) hardware. The MLModelC format is specifically designed for optimal performance on Apple devices.

## Location
```
./anemll/utils/compile_models.py
```

## Usage

Basic command structure:
```bash
python ./anemll/utils/compile_models.py PART [OPTIONS]
```

### Command Line Arguments

- `PART`: Required. Specifies which model part to compile:
  - `1`: Embedding model
  - `2`: FFN model
  - `3`: LM Head model
- `--lut`: LUT quantization configuration
  - Simple format: `--lut 6` (6 bits, uses default per_channel group size of 8)
  - Advanced format: `--lut 6,4` (6 bits with per_channel group size of 4)
  - Note: Only the bits value affects file naming; per_channel is used during model conversion
- `--chunk`: Number of chunks (for FFN models)
- `--prefix`: Model name prefix (default: 'llama')
- `--input-dir`: (Optional) Input directory containing MLPackage files
- `--output-dir`: (Optional) Output directory for compiled models

## Example Usage

### Using Default Prefix ('llama')
```bash
# Compile Embedding Model
python ./anemll/utils/compile_models.py 1

# Compile LM Head with 6-bit Quantization
python ./anemll/utils/compile_models.py 3 --lut 6

# Compile FFN with Chunking
python ./anemll/utils/compile_models.py 2 --lut 6 --chunk 2
```

### Using Custom Prefix
```bash
# Compile Embedding Model with custom prefix
python ./anemll/utils/compile_models.py 1 --prefix mymodel

# Compile LM Head with custom prefix
python ./anemll/utils/compile_models.py 3 --lut 6 --prefix mymodel

# Compile FFN with custom prefix
python ./anemll/utils/compile_models.py 2 --lut 6 --chunk 2 --prefix mymodel
```

## Input Files

The utility expects MLPackage files with the following naming conventions:

For default prefix 'llama':
- Embedding: `llama_embeddings.mlpackage`
- LM Head: `llama_lm_head_lut{N}.mlpackage`
- FFN: `llama_FFN_PF_lut{N}_chunk_{X}of{Y}.mlpackage`

For custom prefix:
- Embedding: `{prefix}_embeddings.mlpackage`
- LM Head: `{prefix}_lm_head_lut{N}.mlpackage`
- FFN: `{prefix}_FFN_PF_lut{N}_chunk_{X}of{Y}.mlpackage`

Where:
- `prefix` is your custom model name prefix
- `N` is the LUT bits
- `X` is the current chunk number
- `Y` is the total number of chunks

## Output Files

The tool generates compiled models in MLModelC format, maintaining similar naming conventions but with optimized internals for deployment.

## Compilation Process

1. Loads the specified MLPackage model
2. Performs optimization passes
3. Generates device-specific code
4. Creates compiled MLModelC output

## Performance Considerations

### Compilation Options
- The compiler automatically selects optimal settings for the target device
- Includes various optimization levels
- Generates device-specific code paths

### Memory Usage
- Compiled models are optimized for runtime memory usage
- Memory layout is arranged for efficient ANE access
- Reduced loading time compared to uncompiled models

## Integration in Workflow

Compilation is the final step before deployment:

1. Convert model parts using ANE_converter
2. Combine FFN and Prefill chunks using combine_models
3. Compile all model parts using compile_models
4. Deploy compiled models in your application

## Verification

After compilation, you can verify the models using the test chat application:

```bash
python ./tests/chat.py --embed llama_embeddings \
                      --lmhead llama_lm_head_lut6 \
                      --ffn llama_FFN_PF_lut6_chunk_01of02 \
                      --tokenizer PATH_TO_TOKENIZER \
                      --context-length 1024
```

## Notes

1. Compilation is device-specific - compile on the same OS version as the target device
2. Ensure sufficient disk space for compilation process
3. Compilation can take several minutes depending on model size
4. Keep original MLPackage files as backup
5. Verify compiled models before deployment

### Xcode
- Xcode 15.0 or later must be installed
- Command Line Tools for Xcode must be installed and selected
- You can verify and select the correct Xcode version using:

## Troubleshooting

Common issues and solutions:

1. **Compilation Errors**
   - Verify input MLPackage files exist
   - Check LUT and chunk parameters match original conversion
   - Ensure sufficient disk space

2. **Performance Issues**
   - Verify OS version compatibility
   - Check model parameters match device capabilities
   - Monitor system resources during compilation

## Related Tools

- [ANE_converter.py](ANE_converter.md): Creates initial MLPackage files
- [combine_models.py](combine_models.md): Combines FFN and Prefill models
- [chat.py](chat.md): Test application for compiled models

For the complete conversion workflow, refer to the [convert.md](convert.md) documentation. 
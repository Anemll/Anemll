# Combine Models Documentation

The `combine_models.py` utility is a crucial component in the ANEMLL workflow that optimizes model storage by merging Feed Forward Network (FFN) and Prefill model chunks into Multi-Function Chunks.

## Purpose

The primary purpose of this tool is to reduce the overall model weight size by approximately 50% by combining FFN and KV pre-fill models that share the same weights into unified Multi-Function Chunks.

## Location
```
./anemll/utils/combine_models.py
```

## Usage

Basic command structure:
```bash
python ./anemll/utils/combine_models.py [OPTIONS]
```

### Command Line Arguments

- `--lut`: LUT quantization configuration
  - Simple format: `--lut 6` (6 bits, uses default per_channel group size of 8)
  - Advanced format: `--lut 6,4` (6 bits with per_channel group size of 4)
  - Note: Only the bits value affects file naming; per_channel is used during model conversion
- `--chunk`: Number of chunks the model is split into
- `--input-dir`: (Optional) Input directory containing the MLPackage files
- `--output-dir`: (Optional) Output directory for combined MLPackage files
- `--monolithic`: Combine monolithic models instead of chunked FFN/prefill models
- `--prefix`: Model prefix (default: `llama`)

## Example Usage

### Chunked Models (Standard)

Basic usage with 6-bit quantization and 2 chunks:
```bash
python ./anemll/utils/combine_models.py --lut 6 --chunk 2
```

### Monolithic Models

For monolithic model conversion, use the `--monolithic` flag:
```bash
python ./anemll/utils/combine_models.py \
    --monolithic \
    --lut 4 \
    --prefix qwen \
    --input ./converted \
    --output ./converted
```

This combines `{prefix}_monolithic.mlpackage` and `{prefix}_monolithic_prefill.mlpackage` into a single `{prefix}_monolithic_combined.mlpackage` with both inference and prefill functions.

## Input Files

The utility expects the following MLPackage files to be present:
- FFN chunks: `llama_FFN_lut{N}_chunk_{X}of{Y}.mlpackage`
- Prefill chunks: `llama_prefill_lut{N}_chunk_{X}of{Y}.mlpackage`

Where:
- `N` is the LUT bits
- `X` is the current chunk number
- `Y` is the total number of chunks

## Output Files

The tool generates combined MLPackage files with the following naming convention:
```
llama_FFN_PF_lut{N}_chunk_{X}of{Y}.mlpackage
```

For example, with 6-bit LUT and 2 chunks:
- `llama_FFN_PF_lut6_chunk_01of02.mlpackage`
- `llama_FFN_PF_lut6_chunk_02of02.mlpackage`

## Process Flow

1. Loads the corresponding FFN and Prefill chunks
2. Combines the models while maintaining their respective functionalities
3. Optimizes the shared weights
4. Saves the combined models as new MLPackage files

## Benefits

1. **Storage Optimization**: Reduces the total model size by approximately 50%
2. **Memory Efficiency**: Eliminates redundant weight storage
3. **Performance**: No impact on inference performance
4. **iOS Compatibility**: Helps maintain file size under iOS 1GB limit

## Integration in Workflow

This utility is typically used after converting the individual model parts and before compiling the models for deployment:

1. Convert model parts using ANE_converter
2. Combine FFN and Prefill chunks using combine_models
3. Compile final models using compile_models

## Notes

1. Ensure all input MLPackage files are present before running the combination
2. The number of chunks specified must match the original conversion
3. LUT quantization bits must match the original conversion
4. Backup original MLPackage files before combining

## Related Tools

- [ANE_converter.py](ANE_converter.md): Creates initial MLPackage files
- [compile_models.py](compile_models.md): Compiles combined models for deployment
- [convert_monolith.md](convert_monolith.md): Monolithic model conversion workflow

For the complete conversion workflow, refer to the [convert.md](convert.md) documentation. 
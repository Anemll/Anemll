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

- `--lut`: LUT quantization bits for FFN models (typically 4 or 6)
- `--lut-prefill`: LUT quantization bits for Prefill models (if different from FFN)
- `--chunk`: Number of chunks the model is split into
- `--input`: (Optional) Input directory containing the MLPackage files
- `--output`: (Optional) Output directory for combined MLPackage files
- `--prefix`: (Optional) Prefix for model names (default: llama)

## Example Usage

Basic usage with 6-bit quantization for both FFN and Prefill, split into 2 chunks:
```bash
python ./anemll/utils/combine_models.py --lut 6 --chunk 2
```

Using different LUT quantization for FFN and Prefill:
```bash
python ./anemll/utils/combine_models.py --lut 4 --lut-prefill 6 --chunk 2
```

## Input Files

The utility expects the following MLPackage files to be present:

When using the same LUT for both:
- FFN chunks: `llama_FFN_lut{N}_chunk_{X}of{Y}.mlpackage`
- Prefill chunks: `llama_prefill_lut{N}_chunk_{X}of{Y}.mlpackage`

When using different LUTs:
- FFN chunks: `llama_FFN_lut{N1}_chunk_{X}of{Y}.mlpackage`
- Prefill chunks: `llama_prefill_lut{N2}_chunk_{X}of{Y}.mlpackage`

Where:
- `N`, `N1`, `N2` are the LUT bits
- `X` is the current chunk number
- `Y` is the total number of chunks

## Output Files

The tool generates combined MLPackage files with the following naming convention:

```
llama_FFN_PF_lut{N}_chunk_{X}of{Y}.mlpackage
```

Where `N` is the LUT bits value used for FFN models (even when Prefill uses a different LUT value).

For example:
- With 6-bit LUT for FFN: `llama_FFN_PF_lut6_chunk_01of02.mlpackage`
- With 4-bit LUT for FFN (regardless of Prefill LUT): `llama_FFN_PF_lut4_chunk_01of02.mlpackage`

Note: For backward compatibility, the output filename only includes the FFN LUT value, even when Prefill uses a different LUT.

## Process Flow

1. Loads the corresponding FFN and Prefill chunks (which may have different LUT quantization)
2. Combines the models while maintaining their respective functionalities
3. Optimizes the shared weights
4. Saves the combined models as new MLPackage files

## Benefits

1. **Storage Optimization**: Reduces the total model size by approximately 50%
2. **Memory Efficiency**: Eliminates redundant weight storage
3. **Performance**: No impact on inference performance
4. **iOS Compatibility**: Helps maintain file size under iOS 1GB limit
5. **Flexible Quantization**: Allows optimizing FFN and Prefill separately for better quality/performance tradeoffs

## Integration in Workflow

This utility is typically used after converting the individual model parts and before compiling the models for deployment:

1. Convert model parts using ANE_converter
2. Combine FFN and Prefill chunks using combine_models
3. Compile final models using compile_models

## Notes

1. Ensure all input MLPackage files are present before running the combination
2. The number of chunks specified must match the original conversion
3. LUT quantization bits must match the original conversion
4. If `--lut-prefill` is not specified, it will use the same value as `--lut`
5. Backup original MLPackage files before combining

## Related Tools

- [ANE_converter.py](ANE_converter.md): Creates initial MLPackage files
- [compile_models.py](compile_models.md): Compiles combined models for deployment
- [convert_model.sh](convert_model.md): Complete conversion workflow

For the complete conversion workflow, refer to the [convert.md](convert.md) documentation. 
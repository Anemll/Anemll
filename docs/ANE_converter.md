# ANE Converter Documentation

The ANE (Apple Neural Engine) Converter is a core component of ANEMLL that handles the conversion of Large Language Models into a format compatible with Apple's Neural Engine. This tool specifically manages the creation of MLPackages for different parts of the model architecture.

## Overview

The converter splits the model into three main components:
1. **Embedding Layer** (Part 1)
2. **Feed Forward Network (FFN)** (Part 2)
3. **Language Model Head** (Part 3)

Additionally, it handles the creation of "Prefill" models for KV cache operations using Apple's Stateful API (introduced in iOS 18 / macOS 15).

## Usage

The converter can be invoked using the following command structure:

```bash
python -m anemll.ane_converter.llama_converter [OPTIONS]
```

### Command Line Arguments

- `--part`: Specifies which part of the model to convert
  - `1`: Embedding layer
  - `2`: Feed Forward Network
  - `2_prefill`: FFN Prefill model
  - `3`: LM Head
- `--model`: Path to the source model
- `--lut`: LUT quantization configuration
  - Simple format: `--lut 6` (6 bits, default per_channel group size of 8)
  - Advanced format: `--lut 6,4` (6 bits with per_channel group size of 4)
  - First value: quantization bits (typically 4, 6, or 8)
  - Second value (optional): per_channel group size for grouped quantization
- `--chunk`: Number of chunks to split the FFN into
- `--context-length`: Maximum context length (e.g., 1024, 2048, 4096)
- `--batch-size`: Batch size for prefill operations

## Examples

### Converting Embedding Layer
```bash
python -m anemll.ane_converter.llama_converter --part 1 --model "path/to/model"
```

### Converting LM Head with 6-bit Quantization
```bash
python -m anemll.ane_converter.llama_converter --part 3 --lut 6 --model "path/to/model"
```

### Converting FFN with Chunking
```bash
python -m anemll.ane_converter.llama_converter --part 2 --lut 6 --chunk 2 --context-length 1024 --batch-size 64 --model "path/to/model"
```

### Converting Prefill Model
```bash
python -m anemll.ane_converter.llama_converter --part 2_prefill --lut 6 --chunk 2 --context-length 1024 --batch-size 64 --model "path/to/model"
```

## Output Files

The converter generates MLPackage files based on the conversion parameters:

- Embedding: `llama_embeddings.mlpackage`
- LM Head: `llama_lm_head_lut{N}.mlpackage` (where N is the LUT bits)
- FFN: `llama_FFN_PF_lut{N}_chunk_{X}of{Y}.mlpackage` (where X is the current chunk and Y is total chunks)
- Prefill: Similar naming to FFN but with prefill-specific configurations

## Technical Details

### Quantization
- The converter supports Look-Up Table (LUT) quantization
- Typically uses 6-bit quantization for optimal performance/quality trade-off
- Embedding layer usually remains unquantized for better accuracy

### Chunking
- FFN layers can be split into multiple chunks
- Helps manage iOS file size limitations (1GB per file)
- Enables support for larger models on device

### State Management
- Utilizes Apple's Stateful API for KV cache operations
- Manages model state between inference steps
- Optimizes memory usage during text generation

## Notes

1. iOS devices have a 1GB file size limitation for neural network models
2. macOS supports larger file sizes (~2GB)
3. Chunking is recommended for larger models
4. The converter is optimized for LLaMA-style architectures
5. Always verify model performance after conversion

## Related Tools

- `combine_models.py`: Merges FFN and prefill chunks
- `compile_models.py`: Compiles models to MLModelC format
- `chat.py`: Test application for converted models

For more detailed information about the complete workflow, refer to the [convert.md](convert.md) documentation. 
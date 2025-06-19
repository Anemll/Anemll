# Conversion Tests

This directory contains scripts and tests for model conversion workflows.

## Purpose
- Test conversion scripts and pipelines
- Validate converted model outputs
- Compare conversion results across different configurations
- Store conversion-related utilities and helpers

## Available Test Scripts

### test_qwen_simple.sh
Tests Qwen model conversion and inference.
- **Model**: Qwen3-0.6B from HuggingFace cache
- **Conversion**: FP16 (no quantization), single chunk
- **Tests**: Python chat.py and Swift CLI inference
- **Usage**: `./test_qwen_simple.sh`

### test_llama_simple.sh
Tests LLaMA model conversion and inference.
- **Model**: Meta-Llama-3.2-1B from local path
- **Conversion**: FP16 (no quantization), single chunk
- **Tests**: Python chat.py and Swift CLI inference
- **Usage**: `./test_llama_simple.sh`

## Common Features
- Uses `/tmp` for output to avoid cluttering project directories
- Provides timestamped output directories for multiple test runs
- Tests both Python and Swift inference paths
- Includes cleanup instructions
- Verifies model existence before conversion

## Organization
- `test_*.sh` - Model conversion and inference test scripts
- `test_*.py` - Python-based conversion test scripts (future)
- `validate_*.py` - Model validation scripts (future)
- `compare_*.py` - Comparison utilities (future)
- `utils/` - Helper functions for conversion testing (future)
# Conversion Tests

This directory contains scripts and tests for model conversion workflows.

## Purpose
- Test conversion scripts and pipelines
- Validate converted model outputs
- Compare conversion results across different configurations
- Store conversion-related utilities and helpers

## Available Test Scripts

### test_hf_model.sh - Generic HuggingFace Model Testing
**NEW**: Universal test script for any HuggingFace model.

```bash
# Usage: ./test_hf_model.sh [model_name] [output_dir] [num_chunks]

# Test any model with automatic naming
./test_hf_model.sh meta-llama/Llama-3.2-1B-Instruct

# Test with custom output directory
./test_hf_model.sh Qwen/Qwen2.5-0.5B-Instruct /tmp/my-test

# Test larger models with chunks
./test_hf_model.sh meta-llama/Llama-3.2-8B-Instruct /tmp/llama8b 4
```

**Features**:
- **Universal**: Works with any HuggingFace model (LLaMA, Qwen, etc.)
- **Auto-downloads**: Downloads model and tokenizer from HuggingFace
- **HF Authentication**: Automatically uses your HF token for gated models
- **Flexible chunking**: Specify number of chunks for larger models
- **Smart naming**: Auto-constructs output directory from model name
- **Tests**: Python chat.py and Swift CLI inference

## Python Test Scripts

### test_llama_model.py
Python wrapper that uses test_hf_model.sh to test LLaMA models:
- LLaMA 3.2 1B Instruct

**Usage**: `python tests/test_llama_model.py`

### test_qwen_model.py
Python wrapper that uses test_hf_model.sh to test Qwen models:
- Qwen3 0.6B

**Usage**: `python tests/test_qwen_model.py`

## Common Features
- **Auto-downloads models**: No need for manual model setup
- **Virtual environment aware**: Automatically activates env-anemll or anemll-env if present
- **HuggingFace Authentication**: Uses cached HF tokens for gated models
- **Portable**: Works on any system with HuggingFace transformers
- **Clean testing**: Uses `/tmp` for output to avoid cluttering project directories
- **Timestamped outputs**: Provides timestamped output directories for multiple test runs
- **End-to-end validation**: Tests both Python and Swift inference paths
- **Cleanup instructions**: Shows how to clean up test outputs
- **Model caching**: Reuses downloaded models for subsequent runs

## Prerequisites
- Python virtual environment with ANEMLL dependencies (env-anemll or anemll-env)
- Run `./install_dependencies.sh` to set up environment
- Internet connection for model downloads (first run only)
- HuggingFace CLI login for gated models: `huggingface-cli login`

## Recommended Testing Workflow

1. **Quick Test**: Use the generic script for any model
   ```bash
   ./tests/conv/test_hf_model.sh meta-llama/Llama-3.2-1B-Instruct
   ```

2. **Architecture Testing**: Use Python wrappers for comprehensive testing
   ```bash
   python tests/test_llama_model.py
   python tests/test_qwen_model.py
   ```

3. **Custom Models**: Use test_hf_model.sh with your own models
   ```bash
   ./tests/conv/test_hf_model.sh your-org/your-model /tmp/custom-test 2
   ```

## Organization
- `test_hf_model.sh` - Universal HuggingFace model test script (recommended)
- Python test scripts in `/tests/` use the shell scripts for actual testing
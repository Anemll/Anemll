# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ANEMLL (pronounced "animal") is an open-source project for accelerating Large Language Models (LLMs) on Apple Neural Engine (ANE). The project converts Hugging Face models to CoreML format for on-device inference on Apple devices.

## Development Commands

### Environment Setup
```bash
# Create Python 3.9 virtual environment (required)
python -m venv anemll-env
source anemll-env/bin/activate
pip install -r requirements.txt

# Install Xcode Command Line Tools (required for CoreML compilation)
xcode-select --install
xcrun --find coremlcompiler  # Verify installation
```

**Important**: Always activate the virtual environment before running any Python scripts in this repository:
```bash
source env-anemll/bin/activate  # or anemll-env/bin/activate depending on your setup
```

You can verify the environment is active by checking:
- The prompt should show `(env-anemll)` or `(anemll-env)`
- `which python` should point to the virtual environment's Python
- `python --version` should show Python 3.9.x

### Model Conversion
```bash
# Single-shot model conversion script
./anemll/utils/convert_model.sh --model <path_to_model> --output <output_directory>

# With additional options
./anemll/utils/convert_model.sh \
    --model ./models/llama-3.1-1b \
    --output ./converted_models \
    --context 512 \
    --batch 64 \
    --lut2 4 \
    --lut3 6 \
    --chunk 2
```

### Testing and Chat Interfaces
```bash
# Basic chat interface (quick testing)
python ./tests/chat.py --meta ./converted_models/meta.yaml

# Advanced chat with conversation history
python ./tests/chat_full.py --meta ./converted_models/meta.yaml

# Manual model specification
python ./tests/chat.py \
    --embed llama_embeddings \
    --lmhead llama_lm_head_lut6 \
    --ffn llama_FFN_PF_lut4_chunk_01of02 \
    --tokenizer ./converted_models \
    --context-length 512 \
    --d ./converted_models
```

### Swift CLI Development
```bash
# Build Swift CLI
cd anemll-swift-cli
swift build

# Run Swift CLI
swift run anemllcli --help

# Run tests
swift test
```

### Development Tools
```bash
# Code formatting (Python)
black anemll/ tests/ examples/
flake8 anemll/ tests/ examples/

# Install development dependencies
pip install -e ".[dev]"
```

## Architecture Overview

### Core Components

1. **ANE Converter Pipeline** (`anemll/ane_converter/`)
   - `base_converter.py`: Abstract base class for model converters
   - `llama_converter.py`: LLaMA/DeepSeek model conversion
   - `qwen_converter.py`: Qwen model conversion
   - `deepseek_converter.py`: DeepSeek-specific optimizations
   - Converts models in 3 parts: embeddings (part 1), FFN/prefill (part 2), LM head (part 3)

2. **Model Implementations** (`anemll/models/`)
   - `base_model.py`: Abstract base model with weight loading interface
   - `llama_model.py`: LLaMA architecture implementation
   - `qwen_model.py`: Qwen architecture implementation
   - `deepseek_model.py`: DeepSeek architecture implementation

3. **Utilities** (`anemll/utils/`)
   - `combine_models.py`: Combines chunked FFN models
   - `compile_models.py`: CoreML compilation with LUT quantization
   - `convert_model.sh`: Main conversion orchestration script

4. **Swift Implementation** (`anemll-swift-cli/`)
   - `AnemllCore`: Core inference engine for Swift applications
   - `InferenceManager.swift`: Manages model inference pipeline
   - `ModelLoader.swift`: Loads and manages CoreML models
   - `Tokenizer.swift`: Tokenization handling
   - `YAMLConfig.swift`: Configuration file parsing

5. **iOS/macOS Sample App** (`anemll-chatbot/`)
   - SwiftUI-based chat interface
   - Model management and downloading
   - Core ML inference integration

### Conversion Pipeline

The model conversion follows an 8-step process:
1. Convert embeddings (part 1) with optional LUT quantization
2. Convert LM head (part 3) with optional LUT quantization
3. Convert FFN layers (part 2) with chunking and optional LUT quantization
4. Convert prefill attention (part 2_prefill)
5. Combine chunked models
6. Compile all parts to CoreML format
7. Copy tokenizer files and create meta.yaml configuration
8. Test with chat interface

### Key Design Patterns

- **Multi-part Architecture**: Models are split into 3 main parts for ANE optimization
- **Chunking Strategy**: FFN layers are chunked to fit ANE memory constraints
- **LUT Quantization**: Lookup table quantization for different model parts (4-bit, 6-bit)
- **Meta Configuration**: YAML-based model configuration for easy deployment

### Testing Infrastructure

The project includes extensive testing files (test_*.py) focusing on:
- KV cache implementations and correctness
- CoreML vs PyTorch output comparison
- Sequential token generation validation
- Attention mechanism testing
- Single vs multi-token inference verification

These tests are primarily for development validation rather than CI/CD.

## Requirements

- **System**: macOS Sequoia with Apple Neural Engine
- **Memory**: Minimum 16GB RAM
- **Python**: 3.9 (strictly required)
- **Dependencies**: coremltools>=8.2, transformers>=4.36.0, numpy>=1.24.0, scikit-learn<=1.5.1
- **Tools**: Xcode Command Line Tools for coremlcompiler

## Model Support

Currently supports:
- LLaMA 3.1/3.2 (1B, 8B variants)
- Qwen 3 (0.6B, 8B)
- DeepSeek R1 (8B distilled)
- DeepHermes (3B, 8B)

Pre-converted models available at https://huggingface.co/anemll

#QWEN TEST
export_coreml.py is a test file for Qwen export development
test_coreml_kvcache_sequential.py is a test file for Qwen inference development
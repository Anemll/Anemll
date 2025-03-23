# anemll-swift-cli

A Swift-based CLI tool for running LLM inference with CoreML models. This package provides high-performance inference capabilities for transformer models on Apple Silicon devices.

## Requirements

- macOS 15 (Sonoma) or later
- iOS 18 or later
- Apple Silicon Mac (M1/M2/M3)
- Xcode 15.0 or later
- Swift 6.0

## Features

- High-performance CoreML inference
- Support for chunked model execution
- Batch processing capabilities
- Interactive chat mode
- Progress reporting during model loading
- Detailed debugging options
- Support for various model versions (including 0.1.1)
- Metal-accelerated tensor operations

## Example Models

Pre-converted CoreML models are available in the [ANEMLL iOS Collection](https://huggingface.co/collections/anemll/anemll-ios-67bdea29e45a1bf4b47d8623) on Hugging Face:

- [anemll-llama-3.2-1B-iOSv2.0](https://huggingface.co/anemll/anemll-llama-3.2-1B-iOSv2.0) - Lightweight 1B parameter model
- [anemll-DeepSeek_ctx1024_iOS.0.1.2](https://huggingface.co/anemll/anemll-DeepSeek_ctx1024_iOS.0.1.2) - DeepSeek model with 1024 context
- [anemll-Hermes-3.2-3B-iOS-0.1.1](https://huggingface.co/anemll/anemll-Hermes-3.2-3B-iOS-0.1.1) - 3B parameter Hermes model

These models are pre-optimized for iOS and macOS devices with Apple Silicon processors.

## Installation

### Swift Package Manager

Add the following to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/Anemll/Anemll.git", exact: "0.1.1")
    // OR
    .package(url: "https://github.com/Anemll/Anemll.git", from: "0.1.1")
    // OR for development
    .package(url: "https://github.com/Anemll/Anemll.git", branch: "swift-inference")
]
```

Choose the dependency declaration that best suits your needs:
- `exact: "0.1.1"` - Use exactly version 0.1.1
- `from: "0.1.1"` - Use version 0.1.1 or higher
- `branch: "swift-inference"` - Use the latest development version

## Usage

### Command Line Options

```bash
USAGE: anemllcli [options]

OPTIONS:
  --meta <path>           Path to meta.yaml config file
  --prompt <text>         Single prompt to process and exit
  --system <text>         System prompt to use
  --max-tokens <number>   Maximum number of tokens to generate
  --temperature <float>   Temperature for sampling (0 for deterministic)
  --template <style>      Template style (default, deephermes)
  --debug-level <number>  Debug level (0=disabled, 1=basic, 2=hidden states)
  --thinking-mode         Enable thinking mode with detailed reasoning
  --show-special-tokens   Show special tokens in output
  --show-loading-progress Show detailed model loading progress
```

### Example Usage

Single prompt mode:
```bash
anemllcli --meta /path/to/model/meta.yaml --prompt "What is the capital of France?"
```

Interactive chat mode:
```bash
anemllcli --meta /path/to/model/meta.yaml
```

### Model Configuration

The package expects a `meta.yaml` file with the following structure:

```yaml
model_info:
  version: "0.2.2"  # Model version
  parameters:
    model_prefix: "model_name"
    context_length: 2048
    batch_size: 32
    num_chunks: 3
    lut_bits: 4
    lut_ffn: -1
    lut_lmhead: -1
```

## Components

### AnemllCore

Core library providing:
- Model loading and configuration
- Tokenization
- Inference management
- YAML configuration parsing

### ANEMLLCLI

Command-line interface providing:
- Interactive chat mode
- Single prompt processing
- Progress reporting
- Debug output

## Development

### Building from Source

1. Clone the repository:
```bash
git clone https://github.com/Anemll/Anemll.git
cd Anemll
```

2. Build the package:
```bash
swift build
```

3. Run tests:
```bash
swift test
```

## Dependencies

- [swift-argument-parser](https://github.com/apple/swift-argument-parser) - CLI argument parsing
- [swift-transformers](https://github.com/huggingface/swift-transformers) - Transformers with tokenizers
- [Yams](https://github.com/jpsim/Yams.git) - YAML parsing
- [Stencil](https://github.com/stencilproject/Stencil.git) - Template processing

## License

This project is part of the Anemll framework. See the LICENSE file for details. 
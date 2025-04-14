# ANEMLL Swift CLI Reference implementation 

A Swift-based CLI tool for running LLM inference with CoreML models. This package provides high-performance inference capabilities for transformer models on Apple Silicon devices.

Swift pakage is located in ./anemll-swift-cli folder

## Requirements

- macOS 15 (Sonoma) or later
- iOS 18 or later ( as a package)
- Apple Silicon Mac (M1/M2/M3), A12 Bionic or later ( iPhone 12 and later) 
- Xcode 15.0 or later
- Swift 6.0

## Features

- High-performance Apple Neural Engine (ANE) acceleration
- CoreML Models
- Support for chunked model execution
- Batch Prefill and KV-cache support
- Interactive chat mode
- Progress reporting during model loading
- Detailed debugging options
- Support for various model versions (including 0.1.1)

## Quick Start

Test the CLI with our sample models:

1. Download a test model:
```bash
# Create models directory
mkdir -p ~/Documents/anemll-models
cd ~/Documents/anemll-models

# Using Hugging Face CLI to download model
huggingface-cli download anemll/anemll-llama-3.2-1B-iOSv2.0 --local-dir .
```

> [!Note]
> If you don't have the Hugging Face CLI installed, you can install it with:
> ```bash
> pip install --upgrade huggingface_hub
> ```

2. Build and run the CLI:
```bash
# Build CLI
cd anemll-swift-cli
swift build -c release

# Test with single prompt
swift run -c release anemllcli --meta ~/Documents/anemll-models/meta.yaml --prompt "What is quantum computing?"

# Or start interactive chat
swift run -c release anemllcli --meta ~/Documents/anemll-models/meta.yaml
```

> [!Tip]
> For the best performance, use release build (`-c release`).
> First model load may take a few seconds while CoreML optimizes for your device.

## Example Models

Pre-converted CoreML models are available in the [ANEMLL iOS Collection](https://huggingface.co/collections/anemll/anemll-ios-67bdea29e45a1bf4b47d8623) on Hugging Face:

- [anemll-llama-3.2-1B-iOSv2.0](https://huggingface.co/anemll/anemll-llama-3.2-1B-iOSv2.0) - Lightweight 1B parameter model
- [anemll-DeepSeek_ctx1024_iOS.0.1.2](https://huggingface.co/anemll/anemll-DeepSeek_ctx1024_iOS.0.1.2) - DeepSeek model with 1024 context
- [anemll-Hermes-3.2-3B-iOS-0.1.1](https://huggingface.co/anemll/anemll-Hermes-3.2-3B-iOS-0.1.1) - 3B parameter Hermes model

## Installation

### Current Development Setup (Pre-Release)

Currently, the package should be added as a local package in Xcode:

1. In Xcode, go to your project
2. Select File > Add Packages...
3. Click on "Add Local..."
4. Navigate to the `anemll-swift-cli` directory and select the `Package.swift` file

### Sample Chat Bot Application
For a complete example of integrating ANEMLL in a SwiftUI application, see our sample chat bot:
[iOS/macOS Sample Applications](../docs/sample_apps.md)

> [!Important]
> Make sure to select the Package.swift from the CLI folder (`anemll-swift-cli`), not the root ANEMLL directory.
> Package manager support via remote URL will be available in the final release. Currently, use local package integration for development.

### Future Release Installation (Coming Soon)

The following installation method will be available in the final release:

```swift
dependencies: [
    .package(url: "https://github.com/Anemll/Anemll.git", from: "1.0.0")
]
```

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
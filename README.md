# ANEMLL

ANEMLL (pronounced like "animal") is an open-source project focused on accelerating the porting of Large Language Models (LLMs) to tensor processors, starting with the Apple Neural Engine (ANE).

## 🚀 Version 0.3.3 Alpha Release - Initial Support for QWEN 3 Architecture

### 🆕 **New Features**
- **QWEN 3 Architecture Support**: Added support for Qwen3-0.6B and Qwen3-8B models with custom converter optimizations
- **Streamlined Installation**: One-command setup with automatic virtual environment detection
- **Automated Testing Framework**: End-to-end validation scripts for conversion and inference workflows
- **Enhanced Developer Experience**: Improved error handling, better feedback, and simplified testing

### 📦 **Quick Start (New Simplified Workflow)**
```bash
# 1. Setup environment (one-time)
./create_python39_env.sh

# 2. Install dependencies (auto-detects virtual environment)
./install_dependencies.sh

# 3. Test conversion pipeline
python tests/test_qwen_model.py     # Test Qwen models
python tests/test_llama_model.py    # Test LLaMA models

# 4. Convert your own models
./anemll/utils/convert_model.sh --model <path> --output <dir>
```

## Goals
> The goal is to provide a fully open-source pipeline from model conversion to inference for common LLM architectures running on ANE.
> This enables seamless integration and on-device inference for low-power applications on edge devices, ensuring maximum privacy and security.
> This is critical for autonomous applications, where models run directly on the device without requiring an internet connection.
>
> We aim to:
> - Provide flexible and easy to use library/framework to port LLMs to ANE directly from Hugging Face models
> - Provide on-device examples for iOS and macOS swift or C/C++ Applications

See update [Roadmap.md](./docs/Roadmap.md) for more details

## Main Components in 0.3.3 Alpha Release

ANEMLL provides five main components for Apple Neural Engine inference development:

1. [LLM Conversion Tools](./docs/convert.md) - Scripts and code to convert models directly from Hugging Face weights
   - [Single-shot Conversion Script](./docs/convert_model.md)

2. [Swift Reference Implementation](./docs/swift_cli.md) - Optimized inference code for Swift applications
   - Sample CLI application in `anemll-swift-cli`
   - Core inference engine implementation

3. [Python Sample Code](./docs/chat.md) - Reference implementation and testing tools
   - Basic chat interface (`chat.py`)
   - Advanced conversation management (`chat_full.py`)

4. [iOS/macOS Sample Applications](./docs/sample_apps.md) - Ready-to-use example applications (Alpha, now on TestFlight)
   - SwiftUI Chat interface
   - Model Downloads and integration example
   - Conversation management

5. [ANEMLL-BENCH](https://github.com/anemll/anemll-bench) - Apple Neural Engine Benchmarking
   - Performance testing and comparison
   - Model optimization metrics
   - Hardware-specific benchmarks
   - [GitHub Repository](https://github.com/anemll/anemll-bench)

### Pre-converted Models

We provide sample converted models ready for use:
- **LLAMA 3.1/3.2** (1B and B variants) including iOS "friendly builds"
- **🆕 Qwen 3** (0.6B and 4B) - **New in 0.3.3!** Initial support with custom converter
- **DeepSeek** distilled models
- **DeepHermes** distilled models

> [!NOTE]
> Please note that Quantization should be improved. LUT4 quality is fairly low due to lack of Block Quantization on Apple Neural Engine.

### 🧪 **New Testing Infrastructure**

#### Quick Model Testing
- **Generic HF Model Testing**: `./tests/conv/test_hf_model.sh [model_name] [output_dir] [chunks]`
- **LLaMA Testing**: `python tests/test_llama_model.py`
- **Qwen Testing**: `python tests/test_qwen_model.py`

#### Test Any HuggingFace Model
```bash
# Test any model with automatic naming
./tests/conv/test_hf_model.sh meta-llama/Llama-3.2-1B-Instruct

# Test with custom output directory
./tests/conv/test_hf_model.sh Qwen/Qwen2.5-0.5B-Instruct /tmp/my-test

# Test larger models with chunks
./tests/conv/test_hf_model.sh meta-llama/Llama-3.2-8B-Instruct /tmp/llama8b 4
```

#### Features
- **Auto-downloads models**: No manual setup required, downloads models from HuggingFace
- **Fast validation**: Uses unquantized FP16 conversion for quick pipeline testing
- **Virtual environment aware**: Automatically activates env-anemll if present
- **End-to-end validation**: Tests cover conversion → Python inference → Swift CLI inference
- **Clean testing**: Uses `/tmp` directories to avoid cluttering your workspace
- **HuggingFace Authentication**: Automatically uses your HF token for gated models
> Some GPTQ and Spin Quant should greatly improve LUT4 models.

Visit our [Hugging Face repository](https://huggingface.co/anemll) for the latest converted models.

### ⚠️ **Important Alpha Release Notes**
> This is **Alpha Release 0.3.3** - **QWEN 3 support is experimental**
> - **Breaking Change**: `install_dependencies.sh` moved to project root
> - **Enhanced Python Support**: Now supports Python 3.9-3.13 (recommended: 3.9-3.11)
> - **New Architecture**: Initial Qwen 3 support with custom converter optimizations
> - **Improved Testing**: Automated validation scripts for conversion workflows
> 
> Please visit https://huggingface.co/anemll for pre-converted models and follow [@anemll](https://x.com/anemll) for updates
> 
> ⭐ **Please star this repo to support the project!**

### 🔄 **What's New in 0.3.3**
- **🆕 Qwen 3 Architecture Support** - Initial implementation with custom converter
- **📦 Streamlined Installation** - Auto-detecting virtual environment setup
- **🧪 Automated Testing** - End-to-end validation scripts
- **🛠 Enhanced Developer Experience** - Better error handling and feedback
- **🔧 Bug Fixes** - Resolved Swift CLI tensor shape mismatches


### Sample iOS/macOS Applications
- Downloads reference or custom models from HuggingFace
- Inference / chat implementation use Swift Library
- Sample TestFlight App for a quick test
- See [iOS/macOS Sample Applications Guide](./docs/sample_apps.md) for details

> [!Tip]
> Try our TestFlight app: [Join Beta](https://testflight.apple.com/join/jrQq1D1C)

## Swift CLI Reference Implementation

The Swift CLI provides a reference implementation for running models on Apple Neural Engine. For detailed documentation, see [Swift CLI Guide](./docs/swift_cli.md).

### Quick Start

1. Download a model from [Hugging Face](https://huggingface.co/anemll)
2. Convert the model using our single-shot conversion script:
```bash
./anemll/utils/convert_model.sh --model <path_to_model> --output <output_directory>
```
3. Run the model using our sample code:
```bash
python ./tests/chat.py --meta <output_directory>/meta.yaml
```

For detailed conversion steps and advanced options, see:
- [Model Conversion Guide](./docs/convert.md)
- [Single-shot Conversion Script](./docs/convert_model.md)
- [DeepSeek Model Guide](./docs/ConvertingDeepSeek.md)

## Testing with Python

We provide two chat interfaces:
- `chat.py` - Basic chat interface for quick testing
- `chat_full.py` - Advanced chat with conversation history management

Features of chat_full.py:
- Maintains full conversation history within context window
- Automatically truncates older messages when needed
- Shifts context window dynamically during long responses
- Shows generation speed and token statistics
- Better handles multi-turn conversations

### Quick Testing with Conversion Scripts

```bash
# Test complete pipeline: download → convert → inference
./tests/conv/test_qwen_simple.sh    # Tests Qwen3-0.6B conversion
./tests/conv/test_llama_simple.sh   # Tests meta-llama/Llama-3.2-1B (requires HF access)
```

> **📝 Note:** Test scripts use small models (0.6B-1B parameters) with unquantized FP16 conversion for faster testing and validation. For production models with quantization (LUT4/LUT6), use the full conversion script with your preferred model size.

### Manual Chat Testing

```bash
# Basic chat
python ./tests/chat.py --meta ./converted_models/meta.yaml

# Full conversation mode
python ./tests/chat_full.py --meta ./converted_models/meta.yaml
```
See [chat.md](./docs/chat.md) for more details 

> [Note]
>The first time the model loads, macOS will take some time to place it on the device. Subsequent loads will be instantaneous. Use Ctrl-D to exit, Ctrl-C to interrupt inference.




## Installation

### System Requirements
- **macOS Sequoia** with Apple Neural Engine (Apple Silicon recommended)
- **Minimum 16GB RAM** (32GB recommended for 8B models)
- **Python 3.9-3.11** (Python 3.9 strongly recommended for best compatibility)
- **Xcode Command Line Tools** (for CoreML compiler)
- Dependencies: coremltools>=8.2, transformers>=4.36.0, numpy>=1.24.0, scikit-learn<=1.5.1

### 📦 Installation (New Streamlined Process)

**🚀 One-Command Setup:**
```bash
# 1. Create Python environment with correct version (auto-detects Python 3.9/3.10/3.11)
./create_python39_env.sh

# 2. Install all dependencies (auto-detects and activates virtual environment)
./install_dependencies.sh

# 3. Verify installation with automated tests (downloads models automatically)
./tests/conv/test_qwen_simple.sh    # Test Qwen conversion (auto-downloads ~2.4GB)
./tests/conv/test_llama_simple.sh   # Test LLaMA conversion (auto-downloads ~500MB)
```

**🔧 Manual Setup (if needed):**
```bash
# Create virtual environment with Python 3.9 (recommended)
python3.9 -m venv env-anemll
source env-anemll/bin/activate

# Install dependencies
./install_dependencies.sh
```

> **📝 Note on Test Scripts:** The automated test scripts will automatically download required models from HuggingFace:
> - `test_qwen_simple.sh` downloads `Qwen/Qwen3-0.6B` (2.4GB) - tiny model, unquantized FP16
> - `test_llama_simple.sh` downloads `HuggingFaceTB/SmolLM-135M` (500MB) - tiny model, unquantized FP16
> 
> **First run may take longer due to model downloads. Models are cached for subsequent runs.**
> These use small models with no quantization for fast validation - ideal for testing the pipeline.
> 
> **Alternative: Test with your own models:**
> ```bash
> # Convert any HuggingFace model
> ./anemll/utils/convert_model.sh --model <your_model_path> --output /tmp/test-model
> python3 tests/chat.py --meta /tmp/test-model/meta.yaml --prompt "Hello!"
> ```

### ✅ **Verification Steps**

The installation script automatically verifies:
- ✅ Python version compatibility (3.9-3.11 supported, 3.9 recommended)
- ✅ Xcode Command Line Tools (`xcode-select --install` if missing)
- ✅ CoreML compiler (`xcrun --find coremlcompiler`)
- ✅ PyTorch with MPS support
- ✅ CoreML Tools compatibility
- ✅ Apple Neural Engine availability

**Manual verification commands:**
```bash
# Check CoreML compiler
xcrun --find coremlcompiler

# Verify Python environment
python --version  # Should show 3.9.x - 3.11.x
pip list | grep -E "(torch|coremltools|transformers)"

# Test Apple Neural Engine
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```


## 🤖 Model Support

### ✅ **Fully Supported Architectures**

**🦙 LLaMA Family (Stable)**
- **Meta LLaMA 3.1/3.2** (1B, 8B) - Production ready
- **DeepSeek R1** (8B distilled) - Based on LLaMA architecture
- **DeepHermes** (3B, 8B) - LLaMA-based fine-tuned models
- **Context lengths**: Up to 2048 tokens (512-1024 recommended for optimal ANE performance, 4K verified)

**🆕 Qwen Family (Alpha - New in 0.3.3!)**
- **Qwen 3** (0.6B, 1.7B, 4B) - Initial support with custom converter
- **Architecture**: Transformer with RMSNorm, SwiGLU, and RoPE
- **Context lengths**: Up to 32K (512-2048 recommended for ANE, 4K verified)
- **Status**: Experimental - please report issues, needs TopK and Temperature support

### 🔧 **Model Specifications**

| Model Family | Sizes | Context | ANE Optimized | Status |
|-------------|-------|---------|---------------|---------|
| LLaMA 3.1/3.2 | 1B, 8B | 512-2048 | ✅ Yes | 🟢 Stable |
| DeepSeek R1 | 8B | 512-1024 | ✅ Yes | 🟢 Stable |
| DeepHermes | 3B, 8B | 512-1024 | ✅ Yes | 🟢 Stable |
| Qwen 3 | 0.6B, 4B | 512-2048 | ⚠️ Experimental | 🟡 Alpha |

### 🎯 **ANE Performance Notes**
- **Recommended context**: 512-1024 tokens for best performance
- **Memory requirements**: 16GB+ RAM for 1B models, 32GB+ for 8B models
- **Quantization**: LUT4 (FFN) + LUT6 (LM Head) for optimal speed/quality balance
- **Chunking**: Automatic chunking for large models to fit ANE constraints

### 🚀 **Coming Soon**
- **Additional Qwen variants** (1.5B, 3B)
- **Mistral family** support
- **Gemma models**
- **Enhanced quantization** (GPTQ, SpinQuant integration)
- **Larger context lengths** (4K, 8K optimization)

### 📥 **Pre-converted Models**
Ready-to-use models available at [Hugging Face](https://huggingface.co/anemll):
- iOS-friendly builds (unzipped .mlmodelc)
- Standard builds for macOS development
- Multiple quantization levels (FP16, LUT4, LUT6)

## Acknowledgements

### Core Technologies
- Thanks to [@apple](https://apple.com) for developing the Apple Neural Engine 
- Thanks to Apple CoreML Tools team for providing the tools https://github.com/apple/coremltools
- Thanks to [@huggingface](https://huggingface.co) for providing the transformers library and models

### Inspirations, feedback and other resources
- Stephen Panaro https://x.com/flat for feedback and coreml-llm-cli https://github.com/smpanaro/coreml-llm-cli 
- Seba https://x.com/CulStory for inspiration with fast ANE models. https://huggingface.co/seba
- Maynard Handley https://x.com/handleym99 For indepth ANE resources https://github.com/name99-org/AArch64-Explore/blob/main/vol7%20ANE.nb.pdf and feedback

## Contributing

> [!Note]
> We welcome contributions! Please read our contributing guidelines before submitting PRs.

Feel free to submit issues and pull requests to improve **ANEMLL**!

> [!Note]
> If you're using ANEMLL in your project, please submit a PR to add it to this list.
> We love to showcase how the community is using ANEMLL!
### Third-Party Applications Using ANEMLL

### Open Source Projects
- [anemll-server](https://github.com/alexgusevski/anemll-server) - Server implementation of ANEMLL inference

> [!Note]
> If you're using ANEMLL in your project, please submit a PR to add it to this list.
> We love to showcase how the community is using ANEMLL!

### Integration Examples
For examples of how to integrate ANEMLL into your projects, see:
- [iOS Integration Guide](./docs/sample_apps.md)
- [Swift CLI Reference](./docs/swift_cli.md)
- [Python Sample Code](./docs/chat.md)

## Links & Resources

- 🌐 Website: [anemll.com](https://anemll.com)
- 🤗 Models: [huggingface.co/anemll](https://huggingface.co/anemll)
- 📱 X: [@anemll](https://x.com/anemll)
- 💻 GitHub: [github.com/anemll](https://github.com/anemll)

## Contact

For any questions or support, reach out to us at [realanemll@gmail.com](mailto:realanemll@gmail.com)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Anemll/Anemll&type=Date)](https://star-history.com/#Anemll/Anemll&Date)

## License

ANEMLL is licensed under the MIT License.
https://opensource.org/license/mit

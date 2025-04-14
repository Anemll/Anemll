# ANEMLL

ANEMLL (pronounced like "animal") is an open-source project focused on accelerating the porting of Large Language Models (LLMs) to tensor processors, starting with the Apple Neural Engine (ANE).

## Goals
> The goal is to provide a fully open-source pipeline from model conversion to inference for common LLM architectures running on ANE.
> This enables seamless integration and on-device inference for low-power applications on edge devices, ensuring maximum privacy and security.
> This is critical for autonomous applications, where models run directly on the device without requiring an internet connection.
>
> We aim to:
> - Provide flexible and easy to use library/framework to port LLMs to ANE directly from Hugging Face models
> - Provide on-device examples for iOS and macOS swift or C/C++ Applications

See update [Roadmap.md](./docs/Roadmap.md) for more details

## Main Components in 0.3.0 Release

ANEMLL provides five main components for Apple Neural Engine inference development:

1. [LLM Conversion Tools](./docs/convert.md) - Scripts and code to convert models directly from Hugging Face weights
   - [Single-shot Conversion Script](./docs/convert_model.md)

2. [Swift Reference Implementation](./docs/swift_cli.md) - Optimized inference code for Swift applications
   - Sample CLI application in `anemll-swift-cli`
   - Core inference engine implementation

3. [Python Sample Code](./docs/chat.md) - Reference implementation and testing tools
   - Basic chat interface (`chat.py`)
   - Advanced conversation management (`chat_full.py`)

4. [iOS/macOS Sample Applications](./docs/sample_apps.md) - Ready-to-use example applications
   - SwiftUI Chat interface
   - Conversation management
   - Model integration examples

5. [ANEMLL-BENCH](https://github.com/anemll/anemll-bench) - Apple Neural Engine Benchmarking
   - Performance testing and comparison
   - Model optimization metrics
   - Hardware-specific benchmarks

### Pre-converted Models

We provide sample converted models ready for use:
- LLAMA 3.1 (1B and 8B variants) including iOS "friendly builds"
- DeepSeek Coder distilled models
- DeepHermes distilled models

Visit our [Hugging Face repository](https://huggingface.co/anemll) for the latest converted models.

> [!Important]
> This is Alpha Release 0.3.0 for the library. It is designed to process Model Weights directly from Hugging Face models and convert them to the CoreML format for Apple Neural Engine (ANE for short).
> This is Alpha Release 0.3.0 for the library. It is designed to process Model Weights directly from Hugging Face models and convert them to the CoreML format for Apple Neural Engine (ANE for short).
> - This release only supports LLAMA models including DeepSeek and DeepHermes distilled models on LLaMA 3.1 architecture
> - The future release will add support for more models and architectures
> - Please visit https://huggingface.co/anemll where we upload the latest models and X: [@anemll](https://x.com/anemll) for updates
> - Please star this repo to support the project!

### New in 0.3.0 🚀

Swift UI Sample Code
- Sample iOS/macOS inference Chat-Bot App
- Updates to Model conversion scripts 

> Reference implementation for Swift inference:
> ```bash
> cd anemll-swift-cli
> ```
> # To Build
> ```bash
> swift build -c release
> ```
> # To Run:
> ```bash
> swift run -c release anemllcli --meta <path_to_anemall_model_folder>/meta.yaml
> # optional parms: --prompt "Who are you?"
> ```

### Sample iOS/macOS Applications
- Downloads reference or custom models from HuggingFace
- Inference / chat implementation use Swift Library
- Sample TestFlight App for a quick test
- See [iOS/macOS Sample Applications Guide](./docs/sample_apps.md) for details

> [!Tip]
> Try our TestFlight app: [Join Beta](https://testflight.apple.com/join/jrQq1D1C)

## Basic Workflow

See [Model Conversion Guide](./docs/convert.md) and [DeepSeek Model Conversion Guide](./docs/ConvertingDeepSeek.md) and Single-shot model conversion with [Convert Model Script](./docs/convert_model.md) for more details.

1. Download the model from Hugging Face
2. Convert the model to the CoreML format using ANEMLL
3. Run the model on the Apple Neural Engine using provided example code `chat.py`

### Conversion Process Overview
- ANE models on iOS are limited to 1GB file size. macOS will work with ~2GB
- We split models during the conversion process to avoid this limit

### Model Components
There are 3 parts for LLM:
1. Embeddings
2. Feed Forward Network/layers 
3. LM Head

> LLaMA Model ANE optimized implementation is in `./anemll/models/llama_model.py`

For FFN, we can split it into multiple chunks to allow for big models (like 8GB LLaMA/DeepSeek)

### Conversion Steps explained

1. **ANE_converter**:
   - `./anemll/ane_converter/llama_converter.py` creates MLPackages for each part
   - We also create "Prefill" models for KV cache
   - This implementation uses Stateful API for ANE, introduced in iOS 18 / macOS 15

2. **Combine Models**:
   - After creating MLPackages, we merge FFN and prefill chunks into Multi-Function Chunks
   - This reduces weight size by 50% as KV pre-fill and FFN use the same weights
   - Processed by `./anemll/utils/combine_models.py`

3. **Compile Models**:
   - Convert to MLModelC format for on-device inference
   - Done via `./anemll/utils/compile_models.py`

Additional Documentation:
- See Single-shot model conversion with [convert_model.sh](./docs/convert_model.md)
- See Automated Hugging Face Model Distribution preparation with [prepare_hf.sh](./docs/prepare_hf.md)
- See [Model Conversion Documentation](./docs/convert.md) for more details


## Testing

We provide two chat interfaces:
- `chat.py` - Basic chat interface for quick testing
- `chat_full.py` - Advanced chat with conversation history management

Features of chat_full.py:
- Maintains full conversation history within context window
- Automatically truncates older messages when needed
- Shifts context window dynamically during long responses
- Shows generation speed and token statistics
- Better handles multi-turn conversations

Example running Chats:

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
- macOS Sequoia with Apple Neural Engine
- Minimum 16GB RAM
- Python 3.9

### Setup Steps

1. Install ANEMLL:
We recommend creating a new virtual environment for this project.
```bash
python -m venv anemll-env
source anemll-env/bin/activate
pip install -r requirements.txt
# pip install anemll
# due to Alpha Release, we do not recommend installing ANEMLL as a package yet
```
CoreML compiler is required to compile the model. It is part of the Xcode command line tools.
- Ensure that Xcode Command Line Tools are installed, as they include `coremlcompiler`.
- You can install them by running `xcode-select --install`.
- Verify that the `xcrun` command is available and correctly configured in your PATH.
- Use `xcrun --find coremlcompiler` to verify the installation.
- If above fails, please try following steps:
- Download Xcode from the App Store.
- Run `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer/` to set the path.
- Use `xcrun --find coremlcompiler` to verify the installation.
- Run `sudo xcodebuild -license` and agree to the license.


## Model Support

Currently optimized for:
- Meta's LLaMA 3.2 1B and 8B (1024 context) model including DeepSeek R1 8GB distilled model
- More models are coming soon

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

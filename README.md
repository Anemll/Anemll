# ANEMLL

ANEMLL (pronounced like "animal") is an open-source project focused on accelerating the porting of Large Language Models (LLMs) to tensor processors, starting with the Apple Neural Engine (ANE).

## Version 0.3.5 Alpha Release

For complete release notes, see [`docs/RELEASE_NOTES_0.3.5.md`](./docs/RELEASE_NOTES_0.3.5.md).

### What's New in 0.3.5
- **ANEMLL Chat redesign** — Fully rebuilt iOS/macOS/visionOS reference app with voice input, AirDrop model sharing, local model import/linking, Markdown rendering, and thinking mode. [TestFlight Beta](https://testflight.apple.com/join/jrQq1D1C)
- **Gemma 3 family** — Full support for 270M, 1B, 4B QAT with sliding-window + global attention, FP16 scaling, and up to 4K context.
- **Monolithic models** — Single-file conversion and inference for all architectures (LLaMA, Qwen, Qwen 2.5, Gemma 3) with ANEMLL-Dedup for ~50% size reduction.
- **In-model argmax** (`--argmax`) — Moves argmax into the CoreML LM head, outputting per-chunk winner index+value instead of full logits. Drastically reduces ANE-to-host data transfer. Extensible to top-k sampling. Recorded in `meta.yaml` as `argmax_in_model: true`.
- **Swift inference stability** — IOSurface-backed buffers, serial prediction queue, ping-pong/ring buffer patterns eliminate ANE race conditions on iOS.
- **ANEMLL-Dedup** — Surgical weight deduplication for multifunction CoreML models (~50% savings). [Documentation](./docs/anemll-dedup.md)
- **Qwen 3 multi-chunk fix** — Fixed inference divergence caused by applying final RMSNorm on every FFN chunk instead of only the last.
- **New conversion tools** — ANE Profiler ([docs](./anemll/utils/ANE_PROFILER.md)), auto chunk calculator ([docs](./docs/calc_chunk_split.md)), FP16 preflight, real-time conversion monitor.
- **Chat CLI improvements** — New `--st` (single-token prefill for debugging), `--cpu`, `--debug-argmax`, `--mem-report`, `--split-rotate`, `--sliding-window` flags. Architecture-aware stop-token detection.
- **Auto-activate venv** — `convert_model.sh` and `check_dependencies.sh` auto-activate project venvs. Override with `ANEMLL_VENV` or disable with `ANEMLL_AUTO_VENV=0`.

### 🔄 **What's New in 0.3.4**
- **📊 lm-evaluation-harness Support** - Model evaluation with standard benchmarks (BoolQ, ARC Challenge, etc.) - [Documentation](./evaluate/ane/README.md)
- **🎯 New RMSN-orm Implementation** - Precise calculation with ANE hardware ops
- **🐛 Fixed RoPE Tensor Size Bug** - Resolved random overflows (existing pre-0.3.4 models should be re-converted)

#### Example ANE vs HF on MPS backend 

| Task        | HF-FP16 | ANEMLL-FP16 | DIFF % |
|-------------|---------|--------------|--------|
| arc_challenge | 31.66%  | 30.97%       | -0.69% |
| arc_easy      | 60.65%  | 60.94%       | +0.29% |
| boolq         | 63.91%  | 64.68%       | +0.77% |
| piqa          | 66.81%  | 67.74%       | +0.93% |
| winogrande    | 56.43%  | 56.67%       | +0.24% |
| **Average**   | **55.89%** | **56.60%**     | **+0.71%** 

✅ DIFF = ANEMLL-FP16 - HF-FP16, where positive values indicate ANEMLL outperforms HuggingFace on that metric.

🆕 New 0.3.4.models with benchmarks are [here](https://huggingface.co/collections/anemll/anemll-034-686c21c2cb05c715eb3f6a26)


### Quick Start
```bash
# 1. Setup environment (uv recommended)
brew install uv              # one-time
./create_uv_env.sh           # creates env-anemll with Python 3.9
source env-anemll/bin/activate
./install_dependencies.sh

# 2. Test conversion pipeline
python tests/test_gemma3_model.py     # Gemma 3 270M (monolithic + argmax)
python tests/test_qwen_model.py       # Qwen 3
python tests/test_llama_model.py      # LLaMA

# 3. Convert your own models
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

See update [Roadmap.md](./Roadmap.MD) for more details

## Main Components in 0.3.5 Alpha Release

ANEMLL provides six main components for Apple Neural Engine inference development:

1. [LLM Conversion Tools](./docs/convert.md) - Scripts and code to convert models directly from Hugging Face weights
   - [Single-shot Conversion Script](./docs/convert_model.md)

2. [ANE Profiler](./anemll/utils/ANE_PROFILER.md) - CoreML/ANE profiling without Xcode (analyze compute plan, benchmark all units, compatibility reports). Requires CoreMLTools 9.0+ and macOS 15+.

3. [Swift Reference Implementation](./docs/swift_cli.md) - Optimized inference code for Swift applications
   - Sample CLI application in `anemll-swift-cli`
   - Core inference engine implementation

4. [Python Sample Code](./docs/chat.md) - Reference implementation and testing tools
   - Basic chat interface (`chat.py`)
   - Advanced conversation management (`chat_full.py`)

5. [iOS/macOS Sample Applications](./docs/sample_apps.md) - Redesigned ANEMLL Chat app with voice input, AirDrop sharing, Markdown, and thinking mode. [TestFlight Beta](https://testflight.apple.com/join/jrQq1D1C)
   - SwiftUI Chat interface (iOS, macOS, visionOS)
   - HuggingFace model downloads, local import, network drive linking
   - Conversation management with streaming and performance metrics

6. [ANEMLL-BENCH](https://github.com/anemll/anemll-bench) - Apple Neural Engine Benchmarking
   - Performance testing and comparison
   - Model optimization metrics
   - Hardware-specific benchmarks
   - [GitHub Repository](https://github.com/anemll/anemll-bench)

### Pre-converted Models

We provide sample converted models ready for use:
- **Gemma 3** (270M, 1B, 4B QAT) — SWA + global attention, up to 4K context, monolithic and chunked
- **LLaMA 3.1/3.2** (1B, 8B) — including iOS "friendly builds"
- **Qwen 3** (0.6B, 1.7B) — thinking mode support
- **Qwen 2.5** (0.5B) — monolithic available
- **DeepSeek R1** (8B distilled) — via LLaMA converter
- **DeepHermes** (3B, 8B) — LLaMA-based fine-tuned models

> [!NOTE]
> Please note that Quantization should be improved. LUT4 quality is fairly low due to lack of Block Quantization on Apple Neural Engine.

### 🧪 **New Testing Infrastructure**

#### Quick Model Testing
- **Generic HF Model Testing**: `./tests/conv/test_hf_model.sh [model_name] [output_dir] [chunks]`
- **LLaMA Testing**: `python tests/test_llama_model.py`
- **Qwen 3 Testing**: `python tests/test_qwen_model.py`
- **Qwen 2.5 Testing**: `python tests/test_qwen2.5_model.py`
- **Gemma 3 Testing**: `python tests/test_gemma3_model.py`

#### Test Any HuggingFace Model
```bash
# Test any model with automatic naming
./tests/conv/test_hf_model.sh meta-llama/Llama-3.2-1B-Instruct

# Test with custom output directory
./tests/conv/test_hf_model.sh Qwen/Qwen2.5-0.5B-Instruct /tmp/my-test

# Test larger models with chunks
./tests/conv/test_hf_model.sh meta-llama/Llama-3.2-8B-Instruct /tmp/llama8b 4
```

#### Gemma 3 Model Conversion
Gemma 3 models use a split KV cache architecture with interleaved local (sliding window) and global attention layers.

> **Note:** The conversion script now auto-detects HuggingFace model names and downloads them automatically!

```bash
# Convert Gemma 3 270M (small, good for testing)
./anemll/utils/convert_model.sh \
    --model google/gemma-3-270m-it \
    --output /path/to/output/gemma3_270m \
    --context 512 \
    --batch 64 \
    --lut2 4 \
    --lut3 6 \
    --chunk 1

# Convert Gemma 3 1B with LUT6 and 4K context (single chunk)
./anemll/utils/convert_model.sh \
    --model google/gemma-3-1b-it \
    --output /path/to/output/gemma3_1b_lut6_ctx4096 \
    --context 4096 \
    --batch 64 \
    --lut1 6 \
    --lut2 6 \
    --lut3 6 \
    --chunk 1

# Test the converted model
python3 tests/chat.py --meta /path/to/output/gemma3_270m/meta.yaml --prompt "Hello!"
```

**Gemma 3 Notes:**
- HuggingFace model names (e.g., `google/gemma-3-1b-it`) are auto-detected and downloaded
- **270M model**: Uses monolithic format (single CoreML file) with argmax - ideal for quick testing
- **1B model**: Uses standard chunked format (separate embeddings, FFN, LM head)
- Uses split KV cache: local layers (sliding window 512) + global layers (full context)
- For context > 512: 4-function models (infer, infer_rotate, prefill, prefill_rotate) enable automatic cache rotation
- Recommended: `--chunk 1` for all Gemma 3 models (1B fits in single chunk)
- Supports context lengths up to 4096 (512-2048 recommended for optimal ANE performance)
- Large vocabulary (262K tokens) uses 16-way LM head splitting
- Requires HuggingFace login for gated models: `hf login`

> **⚠️ FP16 Overflow Warning**: Gemma 3 models can produce activations exceeding FP16 range (65,504). See [FP16 Compatibility](#fp16-compatibility-for-ane) below.

#### Features
- **Auto-downloads models**: No manual setup required, downloads models from HuggingFace
- **Fast validation**: Uses unquantized FP16 conversion for quick pipeline testing
- **Virtual environment aware**: Automatically activates env-anemll if present
- **End-to-end validation**: Tests cover conversion → Python inference → Swift CLI inference
- **Clean testing**: Uses `/tmp` directories to avoid cluttering your workspace
- **HuggingFace Authentication**: Automatically uses your HF token for gated models
> Some GPTQ and Spin Quant should greatly improve LUT4 models.

Visit our [Hugging Face repository](https://huggingface.co/anemll) for the latest converted models.

### Important Alpha Release Notes
> This is **Alpha Release 0.3.5** — Gemma 3, monolithic models, in-model argmax, ANEMLL Chat redesign, and ANE stability fixes.
> - **Breaking Change**: `install_dependencies.sh` moved to project root
> - **Dependency baseline**: `coremltools>=9.0`
> - **Stable architectures**: LLaMA 3.1/3.2, DeepSeek R1, DeepHermes, Qwen 3, Qwen 2.5, Gemma 3
> - **New conversion modes**: Monolithic (`convert_monolith.sh`), in-model argmax (`--argmax`), per-component LUT (`--lut-embeddings`, `--lut-lmhead`)
>
> Please visit https://huggingface.co/anemll for pre-converted models and follow [@anemll](https://x.com/anemll) for updates
>
> Star this repo to support the project!




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
- Dependencies: coremltools>=9.0, transformers>=4.36.0, numpy>=1.24.0, scikit-learn<=1.5.1

### Installation

**Recommended: UV Setup (fast, reproducible):**
```bash
# Install uv (once)
brew install uv

# Create env-anemll with Python 3.9 and install dependencies
./create_uv_env.sh
source env-anemll/bin/activate
./install_dependencies.sh

# Verify
python --version           # Should show 3.9.x
python -c "import coremltools; print(coremltools.__version__)"
```

**Alternative: Standard venv:**
```bash
./create_python39_env.sh
source env-anemll/bin/activate
./install_dependencies.sh
```

**Test the pipeline:**
```bash
./tests/conv/test_qwen_simple.sh    # Qwen3-0.6B end-to-end (auto-downloads ~2.4GB)
./tests/conv/test_llama_simple.sh   # SmolLM-135M end-to-end (auto-downloads ~500MB)
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

**Qwen Family (Stable)**
- **Qwen 3** (0.6B, 1.7B, 8B) — chunked and monolithic, thinking mode support
- **Qwen 2.5** (0.5B, 1.5B, 3B, 7B) — chunked and monolithic
- **Context lengths**: Up to 4K (512-2048 recommended for ANE)

**Gemma 3 Family (Stable)**
- **Gemma 3** (270M, 1B, 4B QAT) — split KV cache with sliding-window + global attention
- **Context lengths**: Up to 4096 tokens (512-2048 recommended for ANE)
- **Special features**: SWA + global attention, FP16 scaling, in-model argmax, 4-function rotation support
- **M1/A14 limitation**: Constrained to 512-context monolithic models due to ANE non-uniform state shape restrictions

### 🔧 **Model Specifications**

| Model Family | Sizes | Context | Chunked | Monolithic | Status |
|-------------|-------|---------|---------|------------|--------|
| LLaMA 3.1/3.2 | 1B, 8B | 512-2048 | Yes | Yes | Stable |
| DeepSeek R1 | 8B | 512-1024 | Yes | — | Stable |
| DeepHermes | 3B, 8B | 512-1024 | Yes | — | Stable |
| Qwen 3 | 0.6B, 1.7B, 8B | 512-4096 | Yes | Yes | Stable |
| Qwen 2.5 | 0.5B, 1.5B, 3B, 7B | 512-2048 | Yes | Yes | Stable |
| Gemma 3 | 270M, 1B, 4B QAT | 512-4096 | Yes | Yes | Stable |

### 🎯 **ANE Performance Notes**
- **Recommended context**: 512-1024 tokens for best performance
- **Memory requirements**: 16GB+ RAM for 1B models, 32GB+ for 8B models
- **Quantization**: LUT4 (FFN) + LUT6 (LM Head) for optimal speed/quality balance
- **Chunking**: Automatic chunking for large models to fit ANE constraints

### 🚀 **Coming Soon**
- **Additional Qwen 2.5 variants** (14B, 32B)
- **Mistral family** support
- **Enhanced quantization** (GPTQ, SpinQuant integration)
- **Larger context lengths** (8K, 16K optimization)

### 📥 **Pre-converted Models**
Ready-to-use models available at [Hugging Face](https://huggingface.co/anemll):
- iOS-friendly builds (unzipped .mlmodelc)
- Standard builds for macOS development
- Multiple quantization levels (FP16, LUT4, LUT6)

## FP16 Compatibility for ANE

Apple Neural Engine (ANE) operates in FP16 precision, which can only represent values up to ±65,504. Some models (particularly Gemma 3) produce activations that exceed this range, causing NaN/Inf failures.

### The Problem

Models trained in BF16 (range ±3.4×10³⁸) may have:
- **Residual accumulation overflow**: The cumulative `hidden = hidden + attention + mlp` grows too large
- **All sub-tensors within range**: Individual attention, MLP, and norm outputs are fine
- **Overflow in layer outputs**: Combined residual stream exceeds FP16 max

This affects **all Gemma 3 sizes** (270M through 27B) - see [Unsloth's analysis](https://unsloth.ai/blog/gemma3).

### FP16 Compatibility Check Tool

Check any HuggingFace model for ANE compatibility:

```bash
# Quick check
python anemll/utils/fp16_compatibility_check.py --model google/gemma-3-1b-it

# Full analysis with clamp sweep
python anemll/utils/fp16_compatibility_check.py --model google/gemma-3-4b-it-qat-int4-unquantized --sweep
```

Recommended one-command pre-conversion sweep:

```bash
./anemll/utils/fp16_preflight.sh --model <model_id_or_path>
```

This runs the sweep by default and writes a JSON report to `tests/dev/logs/`.

The tool reports:
- Weight analysis (are weights within FP16 range?)
- Precision tests (BF16, FP16, FP16→FP32)
- Residual accumulation analysis
- Recommended scaling factor (α)

### Solutions

We support two approaches:

| Approach | Pros | Cons |
|----------|------|------|
| **Weight Scaling** (Recommended) | Zero runtime overhead, 100% quality match | Requires preprocessing |
| **Runtime Clamping** | Simple to implement | Adds ops per layer |

#### Weight Scaling (Recommended)

For Gemma 3 models, apply a weight-only transformation:

```python
alpha = 0.1875  # 3/16, adjust based on model

# 1. Scale embedding weights
embed_tokens.weight *= alpha

# 2. Transform post-norm weights (Gemma uses (1+w) gain)
for layer in layers:
    post_attention_layernorm.weight = alpha * (1 + w_old) - 1
    post_feedforward_layernorm.weight = alpha * (1 + w_old) - 1
```

#### Model-Specific α Values

| Model | Peak Activation | α Recommended | Status |
|-------|-----------------|---------------|--------|
| gemma-3-270m | 104,162 (1.6x) | 0.48 | 100% match |
| gemma-3-1b-it | 61,040 (0.93x) | 0.82 | 100% match |
| gemma-3-4b-it-qat | 292,969 (4.5x) | 0.17-0.1875 | 100% match |

### Documentation

- [GEMMA3_FP16_SCALING.md](./docs/GEMMA3_FP16_SCALING.md) - Detailed scaling guide
- [fp16_compatibility_check.py](./anemll/utils/fp16_compatibility_check.py) - Diagnostic tool
- [fp16_preflight.sh](./anemll/utils/fp16_preflight.sh) - One-command FP16 pre-conversion sweep

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

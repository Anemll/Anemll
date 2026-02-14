# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ANEMLL (pronounced "animal") is an open-source project for accelerating Large Language Models (LLMs) on Apple Neural Engine (ANE). The project converts Hugging Face models to CoreML format for on-device inference on Apple devices.

## Command Allowlist (Claude Code)

Approved commands for Claude Code in this repo:
```bash
osascript -e 'tell application "System Events" to key code 121'  # page down
sleep 0.5
export ANEMLL_HOST="http://127.0.0.1:8765"
curl ...
```

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

# With additional options (default per_channel group size of 8)
./anemll/utils/convert_model.sh \
    --model ./models/llama-3.1-1b \
    --output ./converted_models \
    --context 512 \
    --batch 64 \
    --lut2 4 \
    --lut3 6 \
    --chunk 2

# With custom per_channel group sizes
# Format: --lutX bits,per_channel (e.g., --lut2 6,4 means 6 bits with group size 4)
./anemll/utils/convert_model.sh \
    --model ./models/llama-3.1-1b \
    --output ./converted_models \
    --lut2 6,4 \
    --lut3 6,16
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

5. **iOS/macOS Sample Apps**
   - `anemll-chatbot/`: SwiftUI-based chat interface for iOS/macOS
   - `ANEMLLChat/`: macOS-specific chat application with enhanced UI
   - Both apps share `AnemllCore` library for inference
   - Model management, downloading, and Core ML inference integration

### ANEMLLChat App Architecture

The macOS `ANEMLLChat` app uses:
- **InferenceService**: Manages model loading and text generation
- **ChatViewModel**: Handles conversation state and UI updates
- **StorageService**: Persists settings and conversations to UserDefaults

**Key Settings** (stored in `com.anemll.chat` UserDefaults):
- `systemPrompt`: Default is empty (matches CLI behavior)
- `repetitionDetectionEnabled`: Default is `false` (matches CLI behavior)
- `temperature`, `maxTokens`, `debugLevel`

**History Trimming**: The app trims conversation history when it exceeds `stateLength - 100` tokens, matching CLI behavior. Old message pairs (user + assistant) are removed to fit within context.

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

### ANE-Specific Implementation Requirements

**CRITICAL**: When implementing models for ANE (Apple Neural Engine) compatibility:

1. **RMSNorm Implementation**: Always use ANE-aware RMSNorm that:
   - Subtracts the mean first: `hidden_states = hidden_states - mean`
   - Then uses `F.layer_norm()` instead of manual computation
   - Example:
   ```python
   def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
       mean = hidden_states.mean(-1, keepdim=True)
       hidden_states = hidden_states - mean
       return F.layer_norm(hidden_states, self.weight.shape, self.weight, bias=None, eps=float(self.eps))
   ```
   - This is REQUIRED for ANE compatibility - standard RMSNorm without mean subtraction will fail on ANE

2. **Conv2d Layers**: All dense layers must be expressed as `nn.Conv2d` with `kernel_size=1`

3. **Weight Reshaping**: Weights from HuggingFace models need proper reshaping for Conv2d format

4. **KV-Cache State Updates Must Use Static Slicing Only**:

   Any slice bounds that depend on runtime values (`current_pos`, dynamic `seq_len`) will compile into
   `slice_by_index` with unresolved parameters, causing ANE failure: "Failed to retrieve parameter end."

   ```python
   # ✅ OK - Static slices with fixed bounds
   cache[:, :, 0:seq_len, :]           # seq_len is fixed at trace time (e.g., batch_size=64)
   torch.narrow(cache, dim=2, start=1, length=sw-1)  # constant start/length

   # ✅ OK - Rotation using shift-left + append with constant bounds
   shifted = cache[:, :, 1:, :]        # constant slice
   new_cache = torch.cat([shifted, new_kv], dim=2)

   # ❌ NOT OK - Dynamic slice bounds
   cache[:, :, current_pos:current_pos+1, :]      # dynamic start
   cache[:, :, current_pos:current_pos+seq_len, :]  # dynamic start and end
   cache[:, :, :current_pos+1, :]                 # dynamic end

   # ❌ NOT OK - Mask/gather logic with int32 ops (blocks ANE)
   mask = torch.arange(sw) >= current_pos  # greater_equal
   result = torch.where(mask, new_val, cache)  # logical ops
   ```

   **Correct Semantics for KV Cache**:
   - **Normal prefill** (< sliding_window): Left-fill from position 0 with static `[0:batch_size]`
   - **Rotate prefill** (>= sliding_window): Shift-left + append using static `narrow()` + `cat()`
   - **Single token infer**: Left-fill with static slice
   - **Single token infer_rotate**: Shift-left + append with static bounds

   If you need dynamic position writes, use separate compiled functions with fixed bounds for each case.

### Testing Infrastructure

The project includes extensive testing files (test_*.py) focusing on:
- KV cache implementations and correctness
- CoreML vs PyTorch output comparison
- Sequential token generation validation
- Attention mechanism testing
- Single vs multi-token inference verification

These tests are primarily for development validation rather than CI/CD.

### IMPORTANT: CoreML Testing Guidelines

**ALWAYS use Apple Neural Engine for testing** - NEVER use `CPU_ONLY`:
```python
# CORRECT - Always use ANE
compute_unit = ct.ComputeUnit.CPU_AND_NE  # or ct.ComputeUnit.ALL

# WRONG - Never use this for testing
compute_unit = ct.ComputeUnit.CPU_ONLY  # Models may work on CPU but fail on ANE!
```

This is critical because:
- Models are optimized specifically for ANE
- CPU_ONLY may work but doesn't validate actual ANE compatibility
- Production deployment targets ANE, so testing must use ANE

**Known Issue**: Multi-function compiled models (.mlmodelc) with 4 functions may fail to load
prefill functions on ANE with error: "function_name must be nil unless model type is ML Program".
This is a CoreML limitation. Workaround: use `--split-rotate` to create separate files for
rotate and non-rotate functions.

## Development Guidelines

### Test and Debug File Organization

**IMPORTANT**: Always create test, debug, and development files in `./tests/dev/` to keep the root directory clean.

When working on:
- **Bug fixes**: Create debug scripts in `./tests/dev/debug_<issue_name>.py`
- **New architecture support**: Create test files in `./tests/dev/test_<arch>_<feature>.py`
- **Model validation**: Create comparison scripts in `./tests/dev/test_<model>_vs_<reference>.py`
- **Development utilities**: Place tools in `./tests/dev/` with descriptive names

**Never** create test or debug files directly in the root directory. This keeps the project structure clean and professional.

See `./tests/dev/README.md` for a complete catalog of existing development files organized by architecture and purpose.

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

## ANEMLLChat vs CLI Parity

The macOS ANEMLLChat app should match CLI (`anemllcli`) behavior:

| Feature | CLI | App |
|---------|-----|-----|
| System prompt | Only if `--system` provided | Only if configured in Settings |
| Repetition detection | None | Off by default (toggle in Settings) |
| History trimming | Trims when > stateLength-100 | Same logic |
| Context display | `[History: N tokens]` | `N ctx` (input + output tokens) |

**Common Issues**:
- If app gives different output than CLI, check Settings → System Prompt is "No Prompt"
- If generation stops early, check Settings → Repetition Detection is OFF
- Context mismatch: App now shows `historyTokens` (input + output) matching CLI

#QWEN TEST
export_coreml.py is a test file for Qwen export development
test_coreml_kvcache_sequential.py is a test file for Qwen inference development

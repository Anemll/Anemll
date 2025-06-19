# Test and Debug Files for Qwen Development

This document catalogs all test files created during Qwen model development and debugging, organized by purpose and functionality.

## 🎯 Model Comparison Tests

### test_pytorch_vs_original.py
**Purpose**: Compare ANEMLL PyTorch implementation with original HuggingFace Transformers  
**Parameters**: 
- `--model`: Model path (optional, used within script)
**Usage**: `python test_pytorch_vs_original.py --model ~/Models/HF/qwen3_1.7B`  
**Description**: Comprehensive comparison including full sequence and single token generation

### test_fair_single_token_comparison.py  
**Purpose**: Fair single-token comparison using KV cache prefill  
**Parameters**:
- `--model`: Model path (optional, used within script)
**Usage**: `python test_fair_single_token_comparison.py --model ~/Models/HF/Qwen3-0.6B-MLX-dequantized-2`  
**Description**: Processes context first, then compares single token generation

### test_hf_comparison.py
**Purpose**: Test multiple prompts with original HuggingFace model  
**Parameters**: 
- `model_path`: Model path (function parameter)
**Usage**: Called from other scripts  
**Description**: Establishes baseline behavior with official model

### test_step_by_step_comparison.py
**Purpose**: Multi-token step-by-step comparison  
**Parameters**: None (hardcoded paths)  
**Usage**: `python test_step_by_step_comparison.py`  
**Description**: Detailed token-by-token generation analysis

## 🔧 CoreML Testing

### test_coreml_kvcache_sequential.py ⭐
**Purpose**: Test CoreML KV cache with sequential token processing  
**Parameters**:
- `--model`, `-m`: CoreML model path (default: `/tmp/qwen-test/full/test_qwen.mlpackage`)
- `--tokenizer`: HuggingFace tokenizer model (default: `Qwen/Qwen3-0.6B`)
- `--max-tokens`: Maximum tokens to generate (default: 100)
**Usage**: 
```bash
python test_coreml_kvcache_sequential.py \
    --model /path/to/model.mlpackage \
    --tokenizer Qwen/Qwen3-0.6B \
    --max-tokens 50
```
**Description**: Main CoreML testing script with full parameter support

### test_coreml_kvcache_prefill.py
**Purpose**: Test CoreML with batch prefill using separate models  
**Parameters**:
- `--embeddings-model`: Embeddings CoreML model path
- `--prefill-model`: Prefill CoreML model path  
- `--inference-model`: Inference CoreML model path
- `--tokenizer`: HuggingFace tokenizer model
- `--max-tokens`: Maximum tokens to generate
**Usage**:
```bash
python test_coreml_kvcache_prefill.py \
    --embeddings-model embed.mlpackage \
    --prefill-model prefill.mlpackage \
    --inference-model inference.mlpackage
```
**Description**: Tests multi-part CoreML model architecture

### test_coreml_vs_pytorch.py
**Purpose**: Direct CoreML vs PyTorch output comparison  
**Parameters**: None (hardcoded paths)  
**Usage**: `python test_coreml_vs_pytorch.py`  
**Description**: 4-token window comparison

### test_pytorch_vs_coreml.py
**Purpose**: Compare PyTorch vs CoreML implementations  
**Parameters**: None (hardcoded paths)  
**Usage**: `python test_pytorch_vs_coreml.py`  
**Description**: Single token comparison with 16-part logits

## 🏗️ Model Conversion Tests

### test_part2_conversion.py
**Purpose**: Test FFN (part 2) conversion  
**Parameters**: None (hardcoded paths)  
**Usage**: `python test_part2_conversion.py`  
**Description**: Tests FFN layer conversion with 2-chunk splitting

### test_part3_conversion.py
**Purpose**: Test LM head (part 3) conversion  
**Parameters**: None (hardcoded paths)  
**Usage**: `python test_part3_conversion.py`  
**Description**: Tests LM head conversion with output suppression

## 🔍 Minimal Validation Tests

### test_minimal_official_qwen3.py
**Purpose**: Minimal test with official HuggingFace Qwen3  
**Parameters**: None  
**Usage**: `python test_minimal_official_qwen3.py`  
**Description**: Establishes ground truth baseline

### test_minimal_pytorch_qwen.py
**Purpose**: Minimal test with custom PyTorch implementation  
**Parameters**: None  
**Usage**: `python test_minimal_pytorch_qwen.py`  
**Description**: Tests ANEMLL QwenForCausalLM with KV cache disabled

### test_original_qwen.py
**Purpose**: Test original Transformers model  
**Parameters**: None  
**Usage**: `python test_original_qwen.py`  
**Description**: Baseline text generation with official model

## 🧪 Development and Debug Tests

### test_single_token_generation.py
**Purpose**: Debug single token generation  
**Parameters**: None  
**Usage**: `python test_single_token_generation.py`  
**Description**: Focused single token testing

### test_tensor_shapes.py
**Purpose**: Debug tensor shapes and dimensions  
**Parameters**: None  
**Usage**: `python test_tensor_shapes.py`  
**Description**: Validates tensor shapes throughout pipeline

### test_kv_cache_prefill.py
**Purpose**: Test KV cache prefill functionality  
**Parameters**: None  
**Usage**: `python test_kv_cache_prefill.py`  
**Description**: KV cache state management testing

### test_intermediate_layers.py
**Purpose**: Debug intermediate layer outputs  
**Parameters**: None  
**Usage**: `python test_intermediate_layers.py`  
**Description**: Layer-by-layer output analysis

## 📱 Chat Interface Tests

### tests/chat.py ⭐
**Purpose**: Basic chat interface for converted models  
**Parameters**:
- `--meta`: Path to meta.yaml configuration
- `--embed`: Embeddings model name
- `--lmhead`: LM head model name  
- `--ffn`: FFN model name
- `--tokenizer`: Tokenizer path
- `--context-length`: Context length
- `--d`: Model directory
**Usage**:
```bash
# Using meta.yaml
python tests/chat.py --meta ./converted_models/meta.yaml

# Manual model specification  
python tests/chat.py \
    --embed llama_embeddings \
    --lmhead llama_lm_head_lut6 \
    --ffn llama_FFN_PF_lut4_chunk_01of02 \
    --tokenizer ./converted_models \
    --context-length 512 \
    --d ./converted_models
```

### tests/chat_full.py
**Purpose**: Advanced chat with conversation history  
**Parameters**: Same as chat.py  
**Usage**: Same as chat.py  
**Description**: Enhanced chat interface with memory

## 🔬 Unit Tests (tests/ directory)

### tests/test_qwen_inference.py
**Purpose**: Compare HuggingFace vs custom Qwen inference  
**Parameters**: None (pytest)  
**Usage**: `pytest tests/test_qwen_inference.py`  
**Description**: Detailed comparison with cosine similarity

### tests/test_qwen_model.py
**Purpose**: Unit tests for Qwen model forward pass  
**Parameters**: None (pytest)  
**Usage**: `pytest tests/test_qwen_model.py`  
**Description**: Small model validation tests

### tests/test_qwen_converter_parts.py
**Purpose**: Test Qwen conversion pipeline  
**Parameters**: None (pytest, skipped in CI)  
**Usage**: `pytest tests/test_qwen_converter_parts.py`  
**Description**: Conversion parts 1, 2, 3 testing

## 📊 Analysis and Comparison Tests

### test_per_tensor_comparison.py
**Purpose**: Per-tensor detailed comparison  
**Parameters**: None  
**Usage**: `python test_per_tensor_comparison.py`  
**Description**: Tensor-level analysis

### test_direct_comparison.py
**Purpose**: Direct model output comparison  
**Parameters**: None  
**Usage**: `python test_direct_comparison.py`  
**Description**: Head-to-head model comparison

### test_sequential_tokens.py
**Purpose**: Sequential token processing test  
**Parameters**: None  
**Usage**: `python test_sequential_tokens.py`  
**Description**: Multi-token sequence validation

## 🏷️ Test Categories

- **⭐ Primary Tests**: Main testing scripts with full CLI support
- **🎯 Comparison**: Model-vs-model validation  
- **🔧 CoreML**: ANE-specific testing
- **🏗️ Conversion**: Model format conversion
- **🔍 Minimal**: Simple validation tests
- **🧪 Debug**: Development debugging tools
- **📱 Interactive**: Chat interfaces
- **🔬 Unit**: Automated test suite
- **📊 Analysis**: Detailed output analysis

## 🚀 Getting Started

For new users testing Qwen models:

1. **Start with**: `test_minimal_official_qwen3.py` for baseline
2. **Compare implementation**: `test_pytorch_vs_original.py --model <path>`  
3. **Test CoreML**: `test_coreml_kvcache_sequential.py --model <path>`
4. **Interactive testing**: `tests/chat.py --meta <meta.yaml>`

## 📝 Notes

- Most recent development focused on KV cache and CoreML testing
- Tests with `--model` parameter support different model paths
- Chat interfaces support both meta.yaml and manual model specification
- Some tests use hardcoded paths and may need modification for your setup
# Development and Testing Files

This directory contains development, testing, and debugging files organized to support ANEMLL development and architecture expansion.

## Organization by Architecture

### LLaMA Architecture Support
- `test_llama_model.py` - LLaMA model testing
- `debug_compare_llama.py` - LLaMA comparison debugging
- `test_pytorch_vs_original.py` - PyTorch vs original LLaMA comparison

### Qwen Architecture Support
- `export_coreml.py` - **Qwen export development test file**
- `test_coreml_kvcache_sequential.py` - **Qwen inference development test file**
- `official_modeling_qwen3.py` - Reference Qwen3 implementation
- `test_minimal_official_qwen3.py` - Minimal Qwen3 testing
- `test_original_qwen.py` - Original Qwen model testing

### General Model Testing
- `test_hf_comparison.py` - Hugging Face model comparison
- `test_pytorch_vs_coreml.py` - PyTorch vs CoreML validation
- `test_coreml_vs_pytorch.py` - CoreML vs PyTorch validation
- `test_fair_single_token_comparison.py` - Fair single token comparison

## Core Component Testing

### KV Cache Testing
- `test_kv_cache_prefill.py` - KV cache prefill testing
- `test_coreml_kvcache.py` - CoreML KV cache testing
- `test_kvcache_no_prefill.py` - KV cache without prefill
- `debug_kv_*.py` - Various KV cache debugging scripts

### Attention Mechanism Testing
- `debug_attention_*.py` - Attention mechanism debugging
- `debug_rotary_emb.py` - Rotary embedding debugging
- `debug_causal_mask.py` - Causal mask debugging

### Sequential Generation Testing
- `test_sequential_tokens.py` - Sequential token generation
- `test_single_token_generation.py` - Single token generation
- `test_coreml_sequential.py` - CoreML sequential testing

## Development Guidelines

### Adding New Architecture Support
1. Create test files: `test_<arch>_<feature>.py`
2. Add debug scripts: `debug_<arch>_<issue>.py`
3. Include comparison tests: `test_<arch>_vs_<reference>.py`
4. Update this README with new file locations

### Debugging Issues
1. Create focused debug scripts in this directory
2. Use descriptive naming: `debug_<component>_<issue>.py`
3. Document findings in script comments
4. Keep related debug files together

### File Naming Conventions
- `test_*.py` - Test and validation scripts
- `debug_*.py` - Debugging and investigation tools
- `check_*.py` - Configuration and setup validation
- `inspect_*.py` - Model inspection utilities

## Usage

These files are for developers working on ANEMLL internals. Most users don't need these files.

For regular usage:
- `../chat.py` - Main chat interface
- `../chat_full.py` - Full conversation chat interface
- See main project documentation

## Note

Files may have dependencies on specific models or configurations. They are preserved for development reference and debugging purposes. When adding new architecture support, use existing files as templates and follow the established patterns.
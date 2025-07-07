# ANEMLL Model Conversion Guide for AI Agents

This guide provides step-by-step instructions for AI agents to generate model conversion command lines for ANEMLL.

## Overview

ANEMLL converts HuggingFace models to CoreML format optimized for Apple Neural Engine (ANE) with LUT quantization support.

## Quick Reference

### Basic Command Structure
```bash
./anemll/utils/convert_model.sh \
    --model "<model_path>" \
    --output "<output_directory>" \
    --lut1 <bits> \
    --lut2 <bits> \
    --lut3 <bits> \
    --chunk <num_chunks>
```

### Direct Python Converter
```bash
python -m anemll.ane_converter.<model_type>_converter \
    --model "<model_name_or_path>" \
    --output "<output_directory>" \
    --lut <bits> \
    --part "all"
```

## Step-by-Step Process

### 1. Determine Model Path

**Option A: Use HuggingFace Cache Path (Recommended)**
```bash
# Find exact snapshot path
find ~/.cache/huggingface/hub/ -name "*<ModelName>*" -type d | grep -E "(<model_type>|<ModelName>)" | head -5

# Get snapshot directory
ls ~/.cache/huggingface/hub/models--<Org>--<ModelName>/snapshots/

# Use full path
--model "~/.cache/huggingface/hub/models--<Org>--<ModelName>/snapshots/<commit_hash>/"
```

**Option B: Use Python Converter with HF Name**
```bash
# Python converter supports HF names directly
--model "Org/ModelName"
```

### 2. Determine Model Type and Converter

Check model architecture from config.json:
```bash
cat <model_path>/config.json | python3 -m json.tool | grep -E "(model_type|architectures)"
```

**Converter Mapping:**
- `qwen3` → `qwen_converter.py`
- `qwen2` → `qwen2_5_converter.py` 
- `llama` → `llama_converter.py`

### 3. Calculate Model Size and Chunking

**Get safetensors size:**
```bash
wc -c <model_path>/model.safetensors
```

**Calculate FFN size and chunks:**
```python
# Extract from config.json
hidden_size = config["hidden_size"]
intermediate_size = config["intermediate_size"] 
num_layers = config["num_hidden_layers"]
dtype_size = 2  # bfloat16/float16

# FFN weights per layer
ffn_weights_per_layer = (hidden_size * intermediate_size * 2) + (hidden_size * intermediate_size) + (intermediate_size * hidden_size)
ffn_weights_per_layer_bytes = ffn_weights_per_layer * dtype_size

# Total FFN weights
total_ffn_bytes = ffn_weights_per_layer_bytes * num_layers

# With LUT quantization
quantized_ffn_bytes = total_ffn_bytes / (16 / lut_bits)

# Chunks needed for 1GB limit
chunks_needed = max(1, int(quantized_ffn_bytes / (1024**3)) + 1)
```

### 4. Determine LUT Quantization Settings

**Default LUT Settings:**
- `--lut1`: Embeddings (usually none/empty)
- `--lut2`: FFN/prefill layers (default: 4, recommended: 6)
- `--lut3`: LM head (default: 6)

**LUT Bit Options:**
- `4`: 4-bit quantization (smaller, less accurate)
- `6`: 6-bit quantization (balanced)
- `8`: 8-bit quantization (larger, more accurate)

### 5. Set Other Parameters

**Context Length:**
- Default: 512
- Range: 256-4096 (depending on model)

**Batch Size:**
- Default: 64
- For prefill operations

**Prefix:**
- Auto-detected based on model type
- `qwen` for Qwen models
- `llama` for Llama models

## Examples

### Example 1: Qwen3-0.6B with 6-bit LUT
```bash
# Model: Qwen3-0.6B (1.4GB, 28 layers, 1024 hidden, 3072 intermediate)
# FFN size: 0.66GB → 0.25GB with 6-bit LUT → 1 chunk needed

./anemll/utils/convert_model.sh \
    --model "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/" \
    --output "./qwen3-0.6b-anemll" \
    --lut1 6 \
    --lut2 6 \
    --lut3 6 \
    --context 512 \
    --batch 64 \
    --chunk 1
```

### Example 2: Qwen3-4B with 6-bit LUT
```bash
# Model: Qwen3-4B (larger model)
# FFN size: ~2.6GB → ~1GB with 6-bit LUT → 2 chunks needed

./anemll/utils/convert_model.sh \
    --model "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/<commit_hash>/" \
    --output "./qwen3-4b-anemll" \
    --lut1 6 \
    --lut2 6 \
    --lut3 6 \
    --context 512 \
    --batch 64 \
    --chunk 2
```

### Example 3: Using Python Converter
```bash
python -m anemll.ane_converter.qwen_converter \
    --model "Qwen/Qwen3-0.6B" \
    --output "./qwen3-0.6b-anemll" \
    --prefix "qwen3" \
    --lut 6 \
    --part "all" \
    --context-length 512 \
    --batch-size 64 \
    --chunk 1
```

## Model Size Guidelines

### Small Models (< 1B parameters)
- **Chunks**: 1
- **LUT**: 6-bit recommended
- **Context**: 512-1024

### Medium Models (1B-7B parameters)
- **Chunks**: 1-3 (calculate based on FFN size)
- **LUT**: 6-bit recommended
- **Context**: 512-2048

### Large Models (7B+ parameters)
- **Chunks**: 3+ (calculate based on FFN size)
- **LUT**: 4-bit for size, 6-bit for quality
- **Context**: 512-4096

## Troubleshooting

### Common Issues

1. **Model not found**: Use exact snapshot path from HF cache
2. **Quantized model error**: Use unquantized models (avoid FP8, GPTQ, AWQ)
3. **Chunking issues**: Calculate FFN size and use appropriate chunk count
4. **Memory errors**: Reduce batch size or context length
5. **Embeddings LUT naming**: Embeddings with LUT quantization are correctly named with `_lut<X>` suffix (fixed in v0.3.4+)

### Validation Commands

```bash
# Check model exists
ls -la <model_path>/config.json

# Check model size
wc -c <model_path>/model.safetensors

# Check quantization
cat <model_path>/config.json | grep -i quant

# Test conversion
./anemll/utils/convert_model.sh --model <path> --output <output> --only 1
```

## Best Practices

1. **Always check model quantization status** - avoid pre-quantized models
2. **Calculate chunking requirements** - use FFN size, not total model size
3. **Use 6-bit LUT for quality** - 4-bit for size constraints
4. **Test with small context first** - then increase if needed
5. **Use exact snapshot paths** - avoid HF names in shell script
6. **Monitor memory usage** - adjust batch size if needed

## Conversion Process Steps

The conversion process includes:
1. **Part 1**: Embeddings conversion
2. **Part 2**: FFN/prefill conversion (chunked)
3. **Part 3**: LM head conversion
4. **Compilation**: CoreML model compilation
5. **Metadata**: Create meta.yaml and config files
6. **Testing**: Basic inference test

## Output Structure

```
output_directory/
├── <prefix>_embeddings_lut<X>.mlpackage
├── <prefix>_FFN_lut<X>_chunk_01of<Y>.mlpackage
├── <prefix>_prefill_lut<X>_chunk_01of<Y>.mlpackage
├── <prefix>_lm_head_lut<X>.mlpackage
├── config.json
├── tokenizer.json
├── meta.yaml
└── ...
```

## Usage After Conversion

```bash
# Test with chat.py
python tests/chat.py --meta <output>/meta.yaml

# Test with Swift CLI
cd anemll-swift-cli && swift run anemllcli --meta <output>/meta.yaml
```

## Preparing for Hugging Face Upload

### Prerequisites

1. **Hugging Face Account and Token**
   ```bash
   # Install HF CLI
   pip install huggingface_hub
   
   # Login (choose one method)
   huggingface-cli login
   # OR set environment variable
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

### Step 1: Prepare Distribution Files

**For Standard Distribution:**
```bash
./anemll/utils/prepare_hf.sh \
    --input "<output_directory>" \
    --output "<hf_dist_directory>" \
    --org "<your_org>"
```

**For iOS-Ready Distribution:**
```bash
./anemll/utils/prepare_hf.sh \
    --input "<output_directory>" \
    --output "<hf_dist_directory>" \
    --org "<your_org>" \
    --ios
```

**For Both Standard and iOS:**
```bash
./anemll/utils/prepare_hf.sh \
    --input "<output_directory>" \
    --output "<hf_dist_directory>" \
    --org "<your_org>" \
    --ios
```

### Step 2: Create HF Repository

1. Go to https://huggingface.co/new
2. Create repository with name: `anemll-<ModelName>-ctx<ContextLength>_<Version>`
   - Example: `anemll-Qwen-Qwen3-0.6B-ctx512_0.3.3`
3. Set visibility (public/private)

### Step 3: Upload to Hugging Face

**Upload Standard Distribution:**
```bash
huggingface-cli upload <org>/<model-name>_<version> <hf_dist_directory>/standard
```

**Upload iOS Distribution:**
```bash
huggingface-cli upload <org>/<model-name>_<version> <hf_dist_directory>/ios
```

**Example Commands:**
```bash
# For Qwen3-0.6B conversion
./anemll/utils/prepare_hf.sh \
    --input "./qwen3-0.6b-anemll" \
    --output "./qwen3-0.6b-hf-dist" \
    --org "anemll" \
    --ios

# Upload standard version
huggingface-cli upload anemll/anemll-Qwen-Qwen3-0.6B-ctx512_0.3.3 ./qwen3-0.6b-hf-dist/standard

# Upload iOS version
huggingface-cli upload anemll/anemll-Qwen-Qwen3-0.6B-ctx512_0.3.3 ./qwen3-0.6b-hf-dist/ios
```

### Distribution Structure

**Standard Distribution:**
```
hf_dist_directory/standard/
├── meta.yaml
├── tokenizer.json
├── tokenizer_config.json
├── chat.py
├── chat_full.py
├── README.md
├── <prefix>_embeddings_lut<X>.mlmodelc.zip
├── <prefix>_lm_head_lut<X>.mlmodelc.zip
└── <prefix>_FFN_PF_lut<X>_chunk_XXofYY.mlmodelc.zip
```

**Note**: When LUT quantization is applied to embeddings (LUT1), the file is correctly named with `_lut<X>` suffix (e.g., `qwen_embeddings_lut6.mlpackage`).

**iOS Distribution:**
```
hf_dist_directory/ios/
├── meta.yaml
├── tokenizer.json
├── tokenizer_config.json
├── config.json           # Required for iOS offline tokenizer
├── chat.py
├── chat_full.py
├── <prefix>_embeddings_lut<X>.mlmodelc/  # Uncompressed
├── <prefix>_lm_head_lut<X>.mlmodelc/     # Uncompressed
└── <prefix>_FFN_PF_lut<X>_chunk_NNofMM.mlmodelc/  # Uncompressed
```

### Troubleshooting Upload

1. **Token Issues:**
   ```bash
   echo $HUGGING_FACE_HUB_TOKEN
   huggingface-cli login
   ```

2. **Permission Issues:**
   - Ensure organization membership
   - Check repository visibility settings
   - Verify write permissions

3. **Size Issues:**
   - HF has file size limits
   - Large models may need Git LFS
   - Contact HF support for increased limits

## Manual iOS Installation for Anemll-Chatbot

### Step 1: Locate iOS Distribution Files

After running `prepare_hf.sh` with `--ios` flag, locate the iOS distribution:
```bash
# iOS files are in the ios/ subdirectory
ls -la <hf_dist_directory>/ios/
```

### Step 2: Copy to Anemll-Chatbot Models Directory

```bash
# Create Models directory if it doesn't exist
mkdir -p ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/Models/

# Copy the entire model directory
cp -r <hf_dist_directory>/ios/* ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/Models/<model_name>/

# Example for Qwen3-0.6B:
cp -r ./qwen3-0.6b-hf-dist/ios/* ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/Models/qwen3-0.6b-lut6/
```

### Step 3: Update models.json

**Check existing models.json:**
```bash
cat ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/models.json
```

**Add new model entry:**
```bash
# Backup existing file
cp ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/models.json ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/models.json.backup

# Add new model entry (example for Qwen3-0.6B)
cat >> ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/models.json << 'EOF'
  {
    "name": "Qwen3-0.6B LUT6 v0.3.4",
    "isDownloaded": true,
    "id": "anemll-qwen-qwen3-0.6b-ctx512_0.3.4",
    "description": "Qwen3-0.6B model converted with 6-bit LUT quantization",
    "size": 0,
    "downloadURL": "anemll/anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4"
  }
EOF
```

**Complete models.json example:**
```json
[
  {
    "name": "Qwen3-0.6B LUT6 v0.3.4",
    "isDownloaded": true,
    "id": "anemll-qwen-qwen3-0.6b-ctx512_0.3.4",
    "description": "Qwen3-0.6B model converted with 6-bit LUT quantization",
    "size": 0,
    "downloadURL": "anemll/anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4"
  },
  {
    "name": "Llama-3.2-1B LUT6 v0.3.3",
    "isDownloaded": true,
    "id": "anemll-meta-llama-3.2-1b-ctx512_0.3.3",
    "description": "Llama 3.2 1B model converted with 6-bit LUT quantization",
    "size": 0,
    "downloadURL": "anemll/anemll-Meta-Llama-3.2-1B-ctx512_0.3.3"
  }
]
```

### Step 4: Verify Installation

**Check file structure:**
```bash
ls -la ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/Models/<model_name>/
```

**Expected structure:**
```
Models/<model_name>/
├── meta.yaml
├── tokenizer.json
├── tokenizer_config.json
├── config.json
├── chat.py
├── chat_full.py
├── <prefix>_embeddings_lut<X>.mlmodelc/
├── <prefix>_lm_head_lut<X>.mlmodelc/
└── <prefix>_FFN_PF_lut<X>_chunk_XXofYY.mlmodelc/
```

### Step 5: Restart Anemll-Chatbot

1. Quit Anemll-Chatbot application
2. Relaunch the application
3. The new model should appear in the model selection dropdown

### Troubleshooting Manual Installation

1. **Permission Issues:**
   ```bash
   # Check container permissions
   ls -la ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/
   
   # Fix permissions if needed
   chmod 755 ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/Models/
   ```

2. **JSON Format Issues:**
   ```bash
   # Validate JSON syntax
   python3 -m json.tool ~/Library/Containers/anemll.anemll-chat.demo/Data/Documents/models.json
   ```

3. **Model Not Appearing:**
   - Verify `isDownloaded` is set to `true`
   - Check model directory name matches the `id` field
   - Ensure all required files are present
   - Restart the application

### Model Entry Template

Use this template for new models:

```json
{
  "name": "<ModelName> LUT<Bits> v<Version>",
  "isDownloaded": true,
  "id": "anemll-<model-name>-ctx<context>_<version>",
  "description": "<ModelName> model converted with <Bits>-bit LUT quantization",
  "size": 0,
  "downloadURL": "anemll/anemll-<ModelName>-ctx<Context>_<Version>"
}
```

**Example for different models:**
```json
// Qwen3-4B
{
  "name": "Qwen3-4B LUT6 v0.3.4",
  "isDownloaded": true,
  "id": "anemll-qwen-qwen3-4b-ctx512_0.3.4",
  "description": "Qwen3-4B model converted with 6-bit LUT quantization",
  "size": 0,
  "downloadURL": "anemll/anemll-Qwen-Qwen3-4B-ctx512_0.3.4"
}

// Llama-3.2-3B
{
  "name": "Llama-3.2-3B LUT6 v0.3.3",
  "isDownloaded": true,
  "id": "anemll-meta-llama-3.2-3b-ctx512_0.3.3",
  "description": "Llama 3.2 3B model converted with 6-bit LUT quantization",
  "size": 0,
  "downloadURL": "anemll/anemll-Meta-Llama-3.2-3B-ctx512_0.3.3"
}
```

---

**Note**: This guide is designed for AI agents to systematically generate conversion commands. Always validate model paths and parameters before execution. 
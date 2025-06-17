# MLX Quantized Model Dequantization Guide

This guide explains how to dequantize MLX quantized models for use with ANEMLL CoreML conversion.

## Overview

MLX (Apple's ML framework) provides quantized versions of popular models to reduce memory usage and improve performance. However, ANEMLL's CoreML conversion pipeline requires unquantized models. This document explains how to properly dequantize MLX models for ANEMLL compatibility.

## Why Dequantization is Needed

- **CoreML Requirements**: ANEMLL's conversion pipeline expects full-precision weights
- **Quantization Artifacts**: MLX quantization may introduce artifacts that affect CoreML conversion
- **Architecture Differences**: MLX quantized models may have different configurations than original models

## Prerequisites

1. **MLX-LM Installation**:
   ```bash
   pip install mlx-lm
   ```

2. **Model Availability**: MLX quantized model (e.g., from Hugging Face)

## Method 1: Using ANEMLL's MLX Dequantization Script

### Step 1: Download MLX Quantized Model

```bash
# Using Hugging Face CLI (recommended)
huggingface-cli download mlx-community/Qwen3-0.6B-MLX-4bit --local-dir ~/Models/MLX/Qwen3-0.6B-MLX-4bit

# Or using git (alternative)
git clone https://huggingface.co/mlx-community/Qwen3-0.6B-MLX-4bit ~/Models/MLX/Qwen3-0.6B-MLX-4bit
```

### Step 2: Dequantize Using ANEMLL Script

```bash
python utils/dequantize_mlx.py \
    --model ~/Models/MLX/Qwen3-0.6B-MLX-4bit \
    --save-path ~/Models/HF/Qwen3-0.6B-dequantized \
    --de-quantize
```

**Output**: The script will create a dequantized model in HuggingFace format at the specified save path.

## Method 2: Using HuggingFace Dequantization (Alternative)

For broader compatibility, you can also use the HuggingFace-based dequantization script:

```bash
python utils/dequantize_model.py \
    --input ~/Models/MLX/Qwen3-0.6B-MLX-4bit \
    --output ~/Models/HF/Qwen3-0.6B-dequantized-hf
```

## ⚠️ Important Limitations

### FP8 Quantized Models Not Supported

**MLX dequantization cannot handle FP8 quantized models**. If you encounter errors like:

```
ValueError: Received parameters not in model: model.layers.*.weight_scale_inv
```

This indicates an FP8 quantized model. MLX only supports 4-bit quantization formats.

**For FP8 models:**
1. Use the original unquantized model instead
2. Try the HuggingFace dequantization script (`utils/dequantize_model.py`)
3. Download the base model without quantization

## Post-Dequantization Configuration Fixes

MLX dequantization may produce configuration files that differ from the original model. Common issues and fixes:

### 1. Missing/Incorrect Configuration Parameters

The dequantized model's `config.json` may have:
- Missing `bos_token_id`
- Incorrect `max_position_embeddings`
- Wrong `use_cache` setting
- Extra parameters like `use_qk_norm` not in original architecture

### 2. Automatic Configuration Correction

ANEMLL automatically handles most configuration issues, but you can manually verify:

```python
import json

# Load and check config
with open('~/Models/HF/Qwen3-0.6B-dequantized/config.json', 'r') as f:
    config = json.load(f)

# Verify key parameters
print("bos_token_id:", config.get('bos_token_id'))
print("tie_word_embeddings:", config.get('tie_word_embeddings'))
print("use_cache:", config.get('use_cache'))
```

## Testing the Dequantized Model

### 1. Test with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '~/Models/HF/Qwen3-0.6B-dequantized'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Test generation
prompt = "What is Apple Neural Engine?"
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 2. Test with ANEMLL

```bash
python test_fair_single_token_comparison.py --model ~/Models/HF/Qwen3-0.6B-dequantized
```

This will compare the dequantized model's outputs with the ANEMLL PyTorch implementation.

## Common Issues and Solutions

### 1. Missing lm_head.weight

**Problem**: `Warning: lm_head.weight not found in model weights`

**Solution**: ANEMLL automatically uses tied embeddings. This is normal for models with `tie_word_embeddings=True`.

### 2. Config Parameter Mismatches

**Problem**: Dequantized model has different config than original

**Solution**: ANEMLL handles most differences automatically. Critical parameters are corrected during weight loading.

### 3. QK Norm Layers

**Problem**: MLX models may include `q_norm` and `k_norm` layers not in original architecture

**Solution**: These are typically ignored during ANEMLL conversion and don't affect functionality.

### 4. System Prompts in MLX Generation

**Problem**: MLX-LM adds system prompts that change model behavior

**Explanation**: 
- MLX-LM: Uses system prompts → reasoning output with `<think>` tags
- ANEMLL/Transformers: Raw model → direct continuation

This is expected behavior and doesn't indicate an issue with dequantization.

## Converting to CoreML

After successful dequantization, you can convert the model to CoreML:

```bash
./anemll/utils/convert_model.sh \
    --model ~/Models/HF/Qwen3-0.6B-dequantized \
    --output ~/converted_models/qwen3_0.6b \
    --context 512 \
    --batch 64
```

## Verification Steps

1. **Model Loading**: Verify the dequantized model loads without errors
2. **Output Quality**: Compare outputs with original model (accounting for quantization differences)  
3. **Config Correctness**: Check that essential config parameters are present
4. **CoreML Conversion**: Ensure successful conversion to CoreML format

## Best Practices

1. **Use Original Models When Possible**: If available, prefer the original unquantized model over dequantized versions
2. **Verify Outputs**: Always test dequantized models before CoreML conversion
3. **Check Configurations**: Manually verify critical config parameters
4. **Document Sources**: Keep track of which MLX model was used for dequantization

## Troubleshooting

### MLX Import Errors

```bash
# Ensure MLX is properly installed
pip install mlx-lm
python -c "import mlx.core as mx; print('MLX OK')"
```

### Permission Errors

```bash
# Ensure proper permissions for model directories
chmod -R 755 ~/Models/
```

### Memory Issues

Large models may require significant RAM during dequantization. Ensure sufficient memory is available (16GB+ recommended).

## Related Documentation

- [Model Conversion Guide](./convert.md)
- [Troubleshooting Guide](./troubleshooting.md)
- [Supported Models](./supported_models.md)

## Examples

See the `examples/` directory for complete dequantization workflows with different model families.
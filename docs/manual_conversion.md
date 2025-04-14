# Manual Conversion Steps

This guide is for LLAMA 3.2 1B Conversion and testing with chat.py app

1. Download the model from Hugging Face:
   URL: https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main   

2. Convert the model to the CoreML format using ANEMLL
3. Run the model on the Apple Neural Engine using provide example code chat.py

ANE models on iOS are limited to 1GB file size. macOS will work with ~2GB.
We split models during conversion process to avoid this limit.

Generally there are 3 models' parts for LLM:  embedding, Feed Forward Network/layers and LM Head.
We call it part: 1, 2 and 3 respectively.
LLama Model ANE optimized implementation is in ./anemall/models/llama_model.py
For FFN, we can split it in multiple chunks to allow for big models (like 8GB LLama/DeepSeek)

## Conversion Steps

### 1. Convert Embeddings (Part 1)
> Creates `llama_embeddings.mlpackage`

```bash
python -m anemll.ane_converter.llama_converter \
    --part 1 \
    --model "../Meta-Llama-3.2-1B"
```

### 2. Convert LM Head (Part 3)
> Creates `llama_lm_head_lut6.mlpackage` with 6-bit LUT quantization
```bash
python -m anemll.ane_converter.llama_converter \
    --part 3 \
    --lut 6 \
    --model "../Meta-Llama-3.2-1B"
```

### 3. Convert FFN (Part 2)
> Convert FFN part (2) splitting it into 2 chunks and 6-bit LUT quantization, 
it creates `llama_FFN_PF_lut6_chunk_[01-02]of02.mlpackage`
```bash
python -m anemll.ane_converter.llama_converter \
    --part 2 \
    --lut 6 \
    --chunk 2 \
    --context-length 1024 \
    --batch-size 64 \
    --model "../Meta-Llama-3.2-1B"
```

### 4. Convert Prefill (Part 2_prefill)
> Creates prefill model chunks for KV cache optimization
```bash
python -m anemll.ane_converter.llama_converter \
    --part 2_prefill \
    --lut 6 \
    --chunk 2 \
    --context-length 1024 \
    --batch-size 64 \
    --model "../Meta-Llama-3.2-1B"
```

### 5. Combine Models
After we have MLpackages, we merge FFN and prefill chunks into Multi-Function Chunks.
This allows us to reduce weight size by 50% because KV pre-fill and FFN are using the same weights.

```bash
# Basic usage (in models directory)
python ./anemll/utils/combine_models.py \
    --lut 6 \
    --chunk 2

# With explicit input/output directories
python ./anemll/utils/combine_models.py \
    --lut 6 \
    --chunk 2 \
    --input ./models \
    --output ./combined_models
```

### 6. Compile Models
> Converts to MLModelC format for device inference
```bash
python ./anemll/utils/compile_models.py 1
python ./anemll/utils/compile_models.py 3 --lut 6
python ./anemll/utils/compile_models.py 2 --lut 6 --chunk 2
```

## Step 7: Create meta.yaml

Create a `meta.yaml` file in your output directory to store model configuration:

```python
# Create meta.yaml with your parameters
MODEL_NAME="Meta-Llama-3.2-1B"  # Your model name
CONTEXT="1024"                  # Context length used
BATCH="64"                      # Batch size used
LUT_EMB="none"                 # LUT bits for embeddings
LUT_FFN="6"                    # LUT bits for FFN/prefill
LUT_LMH="6"                    # LUT bits for LM head
NUM_CHUNKS="2"                 # Number of chunks used
PREFIX="llama"                 # Model prefix used

meta = f'''model_info:
  name: anemll-{MODEL_NAME}-ctx{CONTEXT}
  version: 0.3.0
  description: |
    Demonstrates running {MODEL_NAME} on Apple Neural Engine
    Context length: {CONTEXT}
    Batch size: {BATCH}
    Chunks: {NUM_CHUNKS}
  license: MIT
  author: Anemll
  framework: Core ML
  language: Python
  parameters:
    context_length: {CONTEXT}
    batch_size: {BATCH}
    lut_embeddings: {LUT_EMB}
    lut_ffn: {LUT_FFN}
    lut_lmhead: {LUT_LMH}
    num_chunks: {NUM_CHUNKS}
    model_prefix: {PREFIX}
'''

# Write to file
with open('converted_models/meta.yaml', 'w') as f:
    f.write(meta)
```


### 8. Test with chat.py
> There are two options for testing:

#### Standard Chat (chat.py)

 Option 1 - Using meta.yaml (recommended):
 ```bash
 python ./tests/chat.py \
     --meta ./converted_models/meta.yaml
 ```
 
 Option 2 - Manual configuration:
 ```bash
 python ./tests/chat.py \
     --embed llama_embeddings \
     --lmhead llama_lm_head_lut6 \
     --ffn llama_FFN_PF_lut6_chunk_01of02 \
     --tokenizer ../Meta-Llama-3.2-1B \
     --context-length 1024
 ```

> Note: If you used [convert_model.sh](convert_model.md) to convert the model, you can also run chat using the generated meta.yaml: 

This file is used by the chat interfaces to automatically configure model parameters.
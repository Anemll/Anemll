# Chat Interfaces

Two chat interfaces are provided for interacting with converted models:

## Basic Chat (chat.py)

Simple chat interface for basic interactions and testing:

```bash
python ./tests/chat.py \
    --embed llama_embeddings \
    --lmhead llama_lm_head_lut6 \
    --ffn llama_FFN_PF_lut6_chunk_01of02 \
    --tokenizer ./models \
    --context-length 1024
```

### Features
- Single message interactions
- Basic generation statistics
- Minimal memory usage
- Suitable for quick testing

## Full Chat (chat_full.py)

Advanced chat interface with conversation management:

```bash
python ./tests/chat_full.py \
    --embed llama_embeddings \
    --lmhead llama_lm_head_lut6 \
    --ffn llama_FFN_PF_lut6_chunk_01of02 \
    --tokenizer ./models \
    --context-length 1024
```

### Features
- Full conversation history
- Automatic history truncation
- Dynamic context window shifting
- Detailed generation statistics:
  - Tokens per second (t/s)
  - Time to first token (TTFT)
  - Total tokens generated
- **Thinking Mode**: Toggle thinking mode with `/t` command to enable deep reasoning processes for DeepHermes
- **No Warmup Option**: Use `--nw` flag to skip the warmup phase for faster startup
- **Debug Mode**: Use `--debug-level` to control debug output:
  - Level 0: Normal operation (default)
  - Level 1: Print full prompts before generation
  - Level 2: Reserved for more verbose debugging
We use warm up models to prevent CoreML/Python GIL race conditions. You can skip the warmup phase for bigger models.

## Using meta.yaml

If you used [convert_model.sh](convert_model.md) to convert your model, you can run either interface using the generated meta.yaml:

```bash
# Basic chat
python ./tests/chat.py --meta ./converted_models/meta.yaml

# Full conversation mode
python ./tests/chat_full.py --meta ./converted_models/meta.yaml
```

## Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--d, --dir` | Directory containing model files | current directory |
| `--embed` | Path to embeddings model | required |
| `--ffn` | Path to FFN model | required |
| `--lmhead` | Path to LM head model | required |
| `--tokenizer` | Path to tokenizer | model directory |
| `--context-length` | Context length for model | 512 |
| `--meta` | Path to meta.yaml for automatic configuration | none |
| `--prompt` | Run once with this prompt and exit | none |
| `--nw` | Skip warmup phase | false |
| `--debug-level` | Debug output level (0=none, 1=print prompts, 2=more verbose) | 0 |

## Model Paths

Model paths can be specified in several ways:

1. **Relative to --dir**
```bash
python ./tests/chat.py \
    --d ./models \
    --embed llama_embeddings \
    --lmhead llama_lm_head_lut6
```

2. **Absolute paths**
```bash
python ./tests/chat.py \
    --embed /path/to/models/llama_embeddings \
    --lmhead /path/to/models/llama_lm_head_lut6
```

3. **Using meta.yaml**
```bash
python ./tests/chat.py --meta ./models/meta.yaml
```

## See Also
- [Model Conversion Guide](convert.md)
- [DeepSeek Models Guide](ConvertingDeepSeek.md) 
# Variable Context Size: Long-Form Generation on Apple Neural Engine

## The Problem

LLMs on Apple Neural Engine run with a fixed KV cache size set at conversion time. A model converted with context length 512 can only process 512 tokens before it must stop. For reasoning models that produce long outputs (code generation, multi-step math, essays), this is a hard limit that cuts off generation mid-sentence.

Increasing the context at conversion time helps, but larger KV caches consume more memory, load more slowly, and reduce throughput. A 4096-context model is significantly slower to prefill and uses more ANE bandwidth than a 512-context model.

## The Solution: Dynamic Context Growth

ANEMLL's variable context system solves this by converting the model at multiple context sizes and switching between them at runtime. The model starts with the smallest context that fits the prompt, then grows the KV cache on demand as generation progresses:

    512 tokens  -->  1024 tokens  -->  2048 tokens  -->  4096 tokens

Each transition copies the valid KV cache entries from the smaller state into the larger one and continues generating without interruption. The user sees a continuous stream of tokens with no visible pause.

When the largest context fills up, the system compacts the state by keeping recent tokens (and optionally the original prompt) in a sliding window, rebuilds the KV cache via re-prefill, and continues generating. This enables generating outputs far longer than any single context window.

### Advantages

- **Fast startup**: Prefill runs on the smallest context that fits the prompt, so short prompts start generating immediately even when the model supports 4096 tokens.

- **Memory efficiency**: Early tokens are generated with minimal KV cache memory. The system only allocates larger caches when needed.

- **Unlimited output length**: The shift-refill overflow policy enables generating 24,000+ tokens from a 4096-context model by periodically compacting the cache and continuing.

- **Prompt preservation**: During compaction, the original prompt tokens are preserved as a fixed prefix, maintaining the model's awareness of the original instruction throughout long generations.

- **On-device privacy**: Everything runs locally on Apple Neural Engine. No tokens leave the device.

## Quick Start

```bash
python examples/variable_context_demo.py \
  --meta ~/Models/ANE/vibethinker_xstates/meta.yaml
```

This runs with the default prompt (write a Tic Tac Toe game in Python), generates up to 24,000 tokens, and demonstrates context transitions and overflow compaction live in the terminal.

### Custom Prompt

```bash
python examples/variable_context_demo.py \
  --meta ~/Models/ANE/vibethinker_xstates/meta.yaml \
  --prompt "Explain the history of the Roman Empire in detail"
```

### Time-Limited Generation

Generate for 5 minutes instead of a fixed token count:

```bash
python examples/variable_context_demo.py \
  --meta ~/Models/ANE/vibethinker_xstates/meta.yaml \
  --max-time 300
```

### Greedy Decoding (Deterministic Output)

```bash
python examples/variable_context_demo.py \
  --meta ~/Models/ANE/vibethinker_xstates/meta.yaml \
  --sampling-mode greedy
```

## How It Works

### Context Transitions

The system loads model chunks for each context size (e.g., 512, 1024, 2048, 4096). Each context has its own set of infer/prefill functions compiled into CoreML. The context sizes and function names are recorded in `meta.yaml`.

When the current position reaches the current context's capacity, the runner:
1. Reads the KV cache state from the current model
2. Creates a new state buffer sized for the next context
3. Copies valid cache entries into the larger buffer (zero-pads the rest)
4. Continues decoding with the larger context

This transition takes a few milliseconds and is invisible to the end user.

### Overflow and Shift-Refill

When the largest context fills up, shift-refill handles the overflow:

1. Keep the original prompt tokens as a fixed prefix
2. Keep recent generated tokens (controlled by `--overflow-reserve-batches`)
3. Discard intermediate tokens that are no longer in the attention window
4. Create fresh KV cache state
5. Re-prefill with the preserved prompt + recent tokens
6. Continue generating from where it left off

The `--overflow-reserve-batches` parameter (default: 9) controls how much recent context to keep. Higher values preserve more recent tokens at the cost of longer compaction prefills.

### Live Events

When `--live-events` is enabled (default in the demo script), you see transition and compaction events inline with the generated text:

```
[transition] ctx512 -> ctx1024 at tokens=512 (2.3 ms, avg decode 45.2 t/s)
[transition] ctx1024 -> ctx2048 at tokens=1024 (3.1 ms, avg decode 44.8 t/s)
[compact] ctx4096 drop=3200 keep=896 (1250.3 ms, avg decode 42.1 t/s)
```

### Performance Summary

At the end of generation, the runner prints a summary with per-context decode speed, transition overhead, and compaction statistics:

```
=== Summary ===
prompt_tokens=42
stop_reason=max-tokens
prefill=125.3ms (335.4 t/s) context=512
decode_tokens=24000 decode_tps=44.2 final_context=4096
transitions:
  ctx512->ctx1024 at token_count=512 (2.3 ms)
  ctx1024->ctx2048 at token_count=1024 (3.1 ms)
  ctx2048->ctx4096 at token_count=2048 (4.2 ms)
compactions:
  ctx4096 drop=3200 keep=896 (1250.3 ms)
  ctx4096 drop=3200 keep=896 (1245.1 ms)
  ...
```

## Model Requirements

The demo requires a model exported with multi-context state transition support. This means the conversion produced separate infer/prefill functions for each context size, recorded in `meta.yaml` under `state_transition_infer_contexts`.

Example `meta.yaml` entries:

```yaml
model_info:
  parameters:
    state_transition_infer_contexts: [512, 1024, 2048, 3072, 4096]
    state_transition_infer_function_template: "infer_ctx{context}"
    state_transition_prefill_function_template: "prefill_ctx{context}"
    state_transition_no_alias_functions: true
```

Pre-converted models with state transition support are available on [HuggingFace/anemll](https://huggingface.co/anemll).

## Demo Script Options

| Option | Default | Description |
|--------|---------|-------------|
| `--meta` | (required) | Path to meta.yaml of multi-context model |
| `--prompt` | Tic Tac Toe game | Input prompt for generation |
| `--max-tokens` | 24000 | Maximum tokens to generate |
| `--max-time` | (none) | Time limit in seconds (overrides --max-tokens) |
| `--max-context-size` | 4096 | Cap for context growth |
| `--sampling-mode` | auto | `auto` (from meta.yaml) or `greedy` |
| `--seed` | 123 | Random seed for reproducible output |
| `--overflow-reserve-batches` | 9 | Batch slots to reserve during compaction |
| `--no-think` | on | Disable thinking mode (Qwen3/VibeThinker) |
| `--think` | off | Enable thinking mode |
| `--quiet` | off | Suppress progress logs |

## Dependencies

The demo depends on the core ANEMLL state transition infrastructure:

- `anemll/utils/state_transition.py` — KV cache resize and copy logic
- `tests/dev/state_transition_growing_inference.py` — Full inference runner with transitions, overflow, and sampling

Both are included in the ANEMLL repository. Install the Python environment with:

```bash
./create_uv_env.sh
source env-anemll/bin/activate
./install_dependencies.sh
```

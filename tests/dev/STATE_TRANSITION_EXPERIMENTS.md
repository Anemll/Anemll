# State Transition Experiments

Documentation for KV cache state transition and attention window size experiments for ANEMLL.

---

## Overview

This document describes **two distinct experiments** for dynamic model management in CoreML LLM inference:

| Experiment | What Changes | Purpose |
|------------|--------------|---------|
| **1. State Transition** | KV cache size (e.g., 256 ↔ 512 tokens) | Resize cache to handle more/fewer tokens |
| **2. Attention Window** | Attention window size | Change how many tokens the model attends to |

These are **different optimizations**:
- **State Transition** = Physically resize the KV cache tensor
- **Attention Window** = Change the attention computation scope (may use same state size)

---

## Experiment 1: State Transition (KV Cache Resizing)

### What It Does
Physically resizes the KV cache tensor between models with different state sizes.

**Example:**
- Model A: State size 256 → `[36, 1, 256, 256]`
- Model B: State size 512 → `[36, 1, 512, 256]`

When transitioning from A to B, we copy the valid KV cache entries and pad with zeros.

### Use Cases
1. **Expansion (256 → 512)**: Start with smaller model, expand when more context needed
2. **Compaction (512 → 256)**: Prefill with larger model, compact for efficient generation

### Key Constraint
- Expansion: Always works (target >= source)
- Compaction: Only works when `current_position <= target_size`

---

## Experiment 2: Attention Size Transition (NOT YET IMPLEMENTED)

### What It Does
Changes the **attention size** (how much of the KV cache to attend to) without changing the state size.

**Important**: This is NOT sliding window attention. The window doesn't slide - we change the attention computation scope.

**Example:**
- Same state size (512 tokens stored in cache)
- Attention size changes from 256 to 512
- Model attends to more/fewer tokens from the cache

### Use Cases
1. **Adaptive Attention Size**: Start with small attention (fast), expand when needed
2. **Memory-Efficient Generation**: Store full cache but attend to subset

### Status
- [ ] Not yet implemented
- [ ] See `tests/dev/test_adaptive_attention.py` for experimental code

---

## Comparison Table

| Feature | State Transition | Attention Window |
|---------|------------------|------------------|
| **Changes** | KV cache tensor size | Attention computation scope |
| **State Shape** | Changes (e.g., 256 → 512) | May stay same |
| **Memory Impact** | Direct (larger cache = more RAM) | Indirect (computation only) |
| **Implementation** | Copy + pad tensor | Modify attention mask |
| **File** | `state_transition.py` | `test_adaptive_attention.py` |
| **Status** | ✅ Implemented & tested | 🔄 Experimental |

---

## Files Summary

### Experiment 1: State Transition (IMPLEMENTED)

### Experiment 2: Attention Window (EXPERIMENTAL)

| File | Description |
|------|-------------|
| `tests/dev/test_adaptive_attention.py` | Attention window transition experiments |

---

## Detailed Documentation

---

# EXPERIMENT 1: State Transition (KV Cache Resizing)

## Files Created

### Core Utility

| File | Description |
|------|-------------|
| `anemll/utils/state_transition.py` | Main utility module for KV cache state transitions |

**Functions provided:**
- `transition_kv_state()` - Expand state from smaller to larger (256 → 512)
- `compact_kv_state()` - Compact state from larger to smaller (512 → 256)
- `transition_coreml_state()` - CoreML convenience function for expansion
- `compact_coreml_state()` - CoreML convenience function for compaction
- `get_transition_info()` - Debug helper for transition details
- `validate_state_shapes()` - Validate shape compatibility
- `StateTransitionManager` - High-level manager class

### Test Files

| File | Description |
|------|-------------|
| `tests/dev/test_state_transition.py` | Comprehensive test suite (34 tests) with demo |
| `tests/dev/test_adaptive_attention.py` | Attention window transition experiments |

---

## State Shape Format

```
[num_layers, num_kv_heads, state_length, head_dim]
```

**Examples:**
- 256 state: `Float16 [36, 1, 256, 256]`
- 512 state: `Float16 [36, 1, 512, 256]`

---

## Use Cases

### 1. Expansion (Small → Large)

**Scenario**: Start with a smaller, faster model and transition to a larger model when more context is needed.

```python
from anemll.utils.state_transition import transition_kv_state

# After processing 200 tokens with 256-state model
small_state = state_256.read_state(name="kv_cache")  # [36, 1, 256, 256]

# Expand to 512 state
large_state = transition_kv_state(
    source_state=small_state,
    target_seq_length=512,
    current_position=200
)

state_512.write_state(name="kv_cache", value=large_state)
```

### 2. Compaction (Large → Small)

**Scenario**: Prefill with a larger model (efficient batch processing), then compact to a smaller model for token-by-token generation.

```python
from anemll.utils.state_transition import compact_kv_state

# After prefilling 200 tokens with 512-state model
large_state = state_512.read_state(name="kv_cache")  # [36, 1, 512, 256]

# Compact to 256 state (200 <= 256, so this works)
small_state = compact_kv_state(
    source_state=large_state,
    target_seq_length=256,
    current_position=200
)

state_256.write_state(name="kv_cache", value=small_state)
```

**Key Constraint**: Compaction only works when `current_position <= target_seq_length`

---

## Full Workflow Example

```
1. [PREFILL]   Use 512-state model for efficient batch prefill (180 tokens)
2. [COMPACT]   Switch to 256-state model (180 fits in 256)
3. [GENERATE]  Generate tokens until approaching 256 limit
4. [EXPAND]    If needed, transition back to 512-state model
```

---

## Test Results

### Test Suite: `test_state_transition.py`

```
Ran 34 tests in ~2 seconds

OK
```

**Test Categories:**
- `TestTransitionKVState` - NumPy expansion tests (12 tests)
- `TestCompactKVState` - NumPy compaction tests (7 tests)
- `TestTransitionKVStateTorch` - PyTorch expansion tests (3 tests)
- `TestCompactKVStateTorch` - PyTorch compaction tests (2 tests)
- `TestGetTransitionInfo` - Info helper tests (1 test)
- `TestValidateStateShapes` - Shape validation tests (4 tests)
- `TestStateTransitionManager` - Manager class tests (3 tests)
- `TestEdgeCases` - Edge case tests (3 tests)

### Demo Output

```bash
python tests/dev/test_state_transition.py --demo
```

```
======================================================================
PART 1: EXPANSION (256 -> 512)
======================================================================
Source shape: (36, 1, 256, 256)
Target shape: (36, 1, 512, 256)
- Valid positions (0:200) match: True
- Padded positions (200:512) are zeros: True

Memory Usage (Expansion):
- Source state (256): 4.50 MB
- Target state (512): 9.00 MB
- Memory INCREASE: 4.50 MB

======================================================================
PART 2: COMPACTION (512 -> 256)
======================================================================
Source shape (512 state): (36, 1, 512, 256)
Target shape (256 state): (36, 1, 256, 256)
- Valid positions (0:200) match: True
- Remaining positions (200:256) are zeros: True

Memory Usage (Compaction):
- Source state (512): 9.00 MB
- Target state (256): 4.50 MB
- Memory SAVINGS: 4.50 MB (50%)
```

---

## Achievements

1. **Universal State Transition Utility**
   - Works with NumPy arrays and PyTorch tensors
   - Preserves dtype (Float16) and device (CPU/MPS)
   - Flexible size combinations (256→400, 512→700, etc.)

2. **Bidirectional Transitions**
   - Expansion: Small state → Large state (always works)
   - Compaction: Large state → Small state (when tokens fit)

3. **CoreML 9.0 Compatibility**
   - Designed for `MLState.read_state()` / `write_state()` APIs
   - Convenience functions for direct CoreML state manipulation

4. **Comprehensive Testing**
   - 34 unit tests covering all functions
   - Edge cases (zero tokens, boundary conditions, invalid inputs)
   - PyTorch MPS device support tested

5. **Documentation**
   - Full usage examples in docstrings
   - Swift implementation guide included
   - Demo script with visual output

---

## Known Issues / Limitations

1. **Compaction Constraint**
   - Cannot compact if `current_position > target_seq_length`
   - Must plan transitions to ensure tokens fit

2. **CoreML State Read/Write**
   - Requires coremltools 9.0+ for `read_state()`/`write_state()` APIs
   - Earlier versions only support implicit state updates during `predict()`

3. **Memory Overhead**
   - Transition creates a new array (not in-place)
   - Temporary memory usage = source + target during transition

4. **Model Loading**
   - Utility handles state transition only
   - Model loading/switching must be handled separately

---

## Reproducing with Other Models

### Step 1: Determine State Shape

Check your model's KV cache state shape:
```python
import coremltools as ct

model = ct.models.MLModel("your_model.mlpackage")
state = model.make_state()

# Read state to see shape
kv_cache = state.read_state(name="kv_cache")  # or your state name
print(f"State shape: {kv_cache.shape}")
# Expected: [num_layers, num_kv_heads, state_length, head_dim]
```

### Step 2: Convert Models with Different State Sizes

```bash
# Convert model with 256 state
./anemll/utils/convert_model.sh \
    --model ./models/your_model \
    --output ./converted/model_256 \
    --context 256

# Convert model with 512 state
./anemll/utils/convert_model.sh \
    --model ./models/your_model \
    --output ./converted/model_512 \
    --context 512
```

### Step 3: Use State Transition

```python
from anemll.utils.state_transition import (
    transition_kv_state,
    compact_kv_state,
    transition_coreml_state,
    compact_coreml_state
)

# Load both models
model_256 = ct.models.MLModel("model_256.mlpackage")
model_512 = ct.models.MLModel("model_512.mlpackage")

# Create states
state_256 = model_256.make_state()
state_512 = model_512.make_state()

# ... run inference and transition as needed ...
```

### Step 4: Validate Transitions

```python
# Verify state integrity after transition
source_kv = state_source.read_state(name="kv_cache")
target_kv = state_target.read_state(name="kv_cache")

# Check valid positions match
import numpy as np
assert np.allclose(
    source_kv[:, :, :current_pos, :],
    target_kv[:, :, :current_pos, :]
), "State mismatch after transition!"
```

---

## Swift Implementation

See `tests/dev/test_state_transition.py` lines 213-358 for complete Swift code including:
- `transitionKVState()` function
- `compactKVState()` function
- `transitionCoreMLState()` convenience wrapper
- Integration example for `InferenceManager.swift`

---

## Commands Reference

```bash
# Run all tests
python tests/dev/test_state_transition.py

# Run demo
python tests/dev/test_state_transition.py --demo

# Run specific test class
python -m pytest tests/dev/test_state_transition.py::TestCompactKVState -v

# Run utility demo
python anemll/utils/state_transition.py
```

---

# EXPERIMENT 2: Attention Window Transition (EXPERIMENTAL)

## Status: 🔄 In Progress / Experimental

This experiment explores changing the attention window size dynamically during inference.

## Concept

Unlike state transition (which resizes the KV cache tensor), attention window transition changes **how much context the model attends to** during the attention computation.

### Key Differences from State Transition

| Aspect | State Transition | Attention Window |
|--------|------------------|------------------|
| **Tensor Size** | Changes | May stay same |
| **What's Modified** | KV cache shape | Attention mask / computation |
| **Memory** | More/less RAM needed | Same RAM, different computation |
| **Tokens Stored** | More/fewer in cache | Same in cache, fewer attended |

## Use Cases

1. **Variable Attention Size**
   - Fixed state size (e.g., 512)
   - Change attention size: attend to last N tokens
   - Enables memory-efficient generation with bounded computation

2. **Adaptive Attention**
   - Start with small attention window (fast)
   - Expand window for complex reasoning tasks
   - Contract window when context is less important

## Files

| File | Description |
|------|-------------|
| `tests/dev/test_adaptive_attention.py` | Experimental attention window code |

## Implementation Approach (TODO)

```python
# Conceptual - not yet implemented

def create_attention_size_mask(
    seq_length: int,
    attention_size: int,
    current_position: int
) -> np.ndarray:
    """
    Create attention mask for variable attention size.

    Only attend to tokens in range:
    [max(0, current_position - attention_size), current_position]
    """
    mask = np.full((1, 1, seq_length, seq_length), float('-inf'))

    for pos in range(seq_length):
        start = max(0, pos - attention_size)
        mask[0, 0, pos, start:pos+1] = 0.0

    return mask
```

## Challenges

1. **Causal Mask Integration**: Must modify causal mask generation
2. **Prefill vs. Inference**: Different mask handling for batch vs. single token
3. **Model Compatibility**: Requires attention implementation to support variable attention sizes
4. **Accuracy Impact**: Smaller attention sizes may reduce generation quality

## Future Work

- [ ] Implement attention size mask generation
- [ ] Integrate with existing causal mask in InferenceManager
- [ ] Benchmark variable attention size vs. full attention
- [ ] Test accuracy impact on various prompts
- [ ] Combine with state transition for optimal performance

---

## Combined Strategy (Future Vision)

The ultimate goal is to combine both experiments:

```
1. [PREFILL]     Large state (512), full attention
2. [COMPACT]     Compact to small state (256)
3. [GENERATE]    Small state, reduced attention size (128)
4. [EXPAND]      When needed, expand state back to 512
```

This would give:
- Fast prefill with large context
- Memory-efficient generation
- Adaptive attention for quality/speed tradeoff

---

## Summary: Two Experiments

| # | Experiment | Status | Purpose |
|---|------------|--------|---------|
| 1 | **State Transition** | ✅ Complete | Resize KV cache tensors |
| 2 | **Attention Window** | 🔄 Experimental | Change attention scope |

**Remember**: These are complementary, not mutually exclusive!

---

## Future Work (Combined)

- [ ] Implement Swift version of state transition in `AnemllCore`
- [ ] Implement attention window transition
- [ ] Benchmark performance impact of state transitions
- [ ] Test with real CoreML models and measure accuracy
- [ ] Add support for multiple named states (not just "kv_cache")
- [ ] Implement rolling buffer for extended context generation
- [ ] Combine state transition + attention window for optimal inference

---

## References

- CoreML Stateful Models: https://apple.github.io/coremltools/docs-guides/source/stateful-models.html
- coremltools 9.0 Release Notes (state read/write APIs)
- ANEMLL Project: https://github.com/anemll/anemll

---
## Post-LUT Multi-Context Chunk Export (Inference-Only)

When you already have context-specific exports (and LUT/palettization already done),
use the exporter below to avoid re-running LUT for every state size.

What it creates per chunk:
- `infer_ctx512`, `infer_ctx1024`, `infer_ctx2048`, `infer_ctx3072`, `infer_ctx4096`
- `infer` alias (points to max context infer)
- `prefill` from max context only (single prefill state size)

Script:
- `tests/dev/export_state_transition_chunks.py`

Example:

```bash
python tests/dev/export_state_transition_chunks.py \
  --contexts \
    512=/Volumes/Models/ANE/vibethinker_1.5b_ctx0512_fp16_hybrid \
    1024=/Volumes/Models/ANE/vibethinker_1.5b_ctx1024_fp16_hybrid \
    2048=/Volumes/Models/ANE/vibethinker_1.5b_ctx2048_fp16_hybrid \
    3072=/Volumes/Models/ANE/vibethinker_1.5b_ctx3072_fp16_hybrid \
    4096=/Volumes/Models/ANE/vibethinker_1.5b_ctx4096_fp16_hybrid \
  --output-dir /Volumes/Models/ANE/vibethinker_1.5b_state_transition \
  --compile \
  --force
```

Notes:
- This is an export/combine step only; it does not re-convert weights.
- Current scope is inference state transitions. Prefill remains max-context only.
- Output writes `state_transition_manifest.yaml` to the output folder.

One-command builder (creates context subfolders, verifies inference for each context, then combines):

```bash
bash scripts/build_vibethinker_state_transition.sh \
  --output /Volumes/Models/ANE/vibethinker_1.5b_state_transition \
  --contexts "512 1024 2048 3072 4096" \
  --force-output
```

Export-first flow (no combine/compile), with max context first:

```bash
bash scripts/export_vibethinker_infer_contexts.sh \
  --contexts "512 1024 2048 3072 4096" \
  --max-context 4096 \
  --context-root /Volumes/Models/ANE/vibethinker_1.5b_state_transition/_contexts
```

This export-first script does exactly:
- Context `4096`: steps `1,2,3,4` only (embeddings + lm_head + FFN infer chunks + prefill chunks)
- Other contexts: step `3` only (FFN infer chunks only)
- Never runs combine/compile/test steps (`5/6/8`)

After that, combine into state-transition chunk packages (no compile):

```bash
bash scripts/combine_vibethinker_infer_contexts.sh \
  --contexts "512 1024 2048 3072 4096" \
  --max-context 4096 \
  --context-root /Volumes/Models/ANE/vibethinker_1.5b_state_transition/_contexts \
  --output /Volumes/Models/ANE/vibethinker_1.5b_state_transition \
  --no-compile \
  --force
```

This combine script (default mode) creates per chunk:
- `infer_ctx512`, `infer_ctx1024`, `infer_ctx2048`, `infer_ctx3072`, `infer_ctx4096`
- `infer` alias (from 4096 infer)
- `prefill` (from 4096 prefill)

To include context-specific prefill functions in the same chunk packages:

```bash
bash scripts/combine_vibethinker_all_context_functions.sh \
  --contexts "512 1024 2048 3072 4096" \
  --max-context 4096 \
  --context-root /Volumes/Models/ANE/vibethinker_1.5b_state_transition/_contexts \
  --output /Volumes/Models/ANE/vibethinker_1.5b_state_transition \
  --force
```

Equivalent using the base combine script:

```bash
bash scripts/combine_vibethinker_infer_contexts.sh \
  --contexts "512 1024 2048 3072 4096" \
  --max-context 4096 \
  --context-root /Volumes/Models/ANE/vibethinker_1.5b_state_transition/_contexts \
  --output /Volumes/Models/ANE/vibethinker_1.5b_state_transition \
  --prefill-all-contexts \
  --force
```

Key combine parameters (current):
- `--infer-fn infer` (default): infer source function in each input chunk.
- `--prefill-fn prefill` (default): prefill source function in each input chunk.
- `--prefill-all-contexts`: include `prefill_ctx{N}` functions for all listed contexts.
- `--split-infer-prefill`: write two outputs per chunk (`*_infer_chunk_*` and `*_prefill_chunk_*`).
- `--no-alias-functions`: do not emit compatibility aliases `infer`/`prefill`; keep only `infer_ctx*` / `prefill_ctx*`.
- `--no-compile`: skip `.mlmodelc` compile (combine/export only).

All-context prefill combine adds per chunk:
- `prefill_ctx512`, `prefill_ctx1024`, `prefill_ctx2048`, `prefill_ctx3072`, `prefill_ctx4096`
- `prefill` alias (still points to 4096 prefill for compatibility)

Split mode (`--split-infer-prefill`) outputs per chunk:
- infer package: `infer_ctx*` (+ `infer` alias unless `--no-alias-functions`)
- prefill package: `prefill_ctx*` (or max-only `prefill_ctx4096`) (+ `prefill` alias unless `--no-alias-functions`)

New metadata tags (all-context prefill mode):
- `state_transition_all_context_prefill: true`
- `state_transition_prefill_contexts: [512, 1024, 2048, 3072, 4096]`
- `state_transition_prefill_function_template: "prefill_ctx{context}"`
- `state_transition_combined_functions_layout: "infer_ctx+prefill_ctx+aliases"`

Additional metadata tags (split/no-alias modes):
- `state_transition_split_infer_prefill: true|false`
- `state_transition_no_alias_functions: true|false`
- `state_transition_infer_default_function`
- `state_transition_prefill_default_function`
- (split mode) `ffn_prefill`, `state_transition_infer_output_base`, `state_transition_prefill_output_base`

For a reproducible package-size/dedup test plan, see:
- `tests/dev/COREML_MULTIFUNCTION_DEDUP_INVESTIGATION.md`

Default intermediate context folders are created under:
- `<output>/_contexts/`
for example:
- `/Volumes/Models/ANE/vibethinker_1.5b_state_transition/_contexts/vibethinker_1.5b_ctx512_fp16_hybrid`

Final combined chunks are written directly in:
- `/Volumes/Models/ANE/vibethinker_1.5b_state_transition`

Per-context inference verification (before combine):
- Builder runs a smoke inference (`tests/chat.py --meta ... --prompt ...`) for each context.
- A PASS marker is written at `<context_dir>/.infer_smoke_ok`.
- On later runs, verification is skipped automatically for contexts that already have the marker.
- Use `--verify-always` to force re-test, or `--no-verify-infer` to skip checks.

Quantization behavior:
- `--context-lut2` controls per-context rebuild quantization (slow if enabled for many contexts)
- `--post-lut2` / `--lut2` quantizes once per **combined chunk package**
- Recommended for multi-context export: `--context-lut2 none --lut2 <bits[,per_channel]>`

## Growing Inference Benchmark (2026-02-10)

Test script:
- `tests/dev/state_transition_growing_inference.py`

### Parameter Reference (Current)

Core routing:
- `--contexts "512,1024,2048,3072,4096"`: Ordered context set used for state growth.
- `--max-context-size N` (alias `--max-active-context N`): Cap active contexts to `<= N` (e.g., `2048`, `3072`).
- `--contexts-root /path`: Root containing per-context export folders.
- `--name-template "vibethinker_1.5b_ctx{context}_L6_4_hybrid"`: Context folder naming template.
- `--context-dirs 512=/path 1024=/path ...`: Optional explicit context-to-folder mapping.
- `--tokenizer /path`: Optional tokenizer override (defaults to max-context folder).

Decode/prefill:
- `--prompt "..."`: Prompt text.
- `--max-tokens N`: Maximum tokens to decode.
- `--max-time SECONDS`: Maximum wall-clock runtime from prefill start (hard stop).
  - If `--max-time` is set and `--max-tokens` is not explicitly provided, token cap is treated as unbounded and time limit governs stop.
- `--prefill-mode batch-prefill|token-infer`: Prefill via batch function or token-by-token infer path.
- `--compute-unit ALL|CPU_ONLY|CPU_AND_GPU|CPU_AND_NE`: CoreML compute unit.
- `--allow-mlpackage-fallback`: Allow loading `.mlpackage` when `.mlmodelc` is missing.
- `--state-name NAME`: Optional explicit KV state name override.
- `--no-think`: Disable template thinking mode (`enable_thinking=False`).
- `--no-eos-stop`: Continue generation past EOS (stress testing mode).
- `--progress-stream stderr|stdout|none`: Route system/progress logs (default: `stderr`); decode tokens always stream on `stdout`.
- `--live-events`: Print transition/compact/stop lines during decode (default: off, summary-only for cleaner answer stream).

Sampling:
- `--sampling-mode auto|greedy`:
  - `auto`: Uses `meta.yaml` `recommended_sampling.do_sample` (+ temperature).
  - `greedy`: Argmax decoding.
- `--temperature X`: Sampling temperature override for `auto`.
- `--seed INT`: Sampling seed for reproducible sampled runs.

Overflow/compaction:
- `--overflow-policy stop|shift-refill`:
  - `stop`: Stop when max context capacity is exceeded.
  - `shift-refill`: Keep recent tail, rebuild state via prefill, continue decoding.
- `--overflow-reserve-batches N`: Shift/refill aggressiveness; higher values drop more tokens per compact (fewer compactions).
- `--overflow-preserve-prompt`: In shift/refill mode, preserve original prompt tokens as fixed prefix on each compact/refill.

### Example: Baseline Growing Decode

```bash
TMPDIR=/Volumes/Models/ANE/tmp_coreml_compile \
TMP=/Volumes/Models/ANE/tmp_coreml_compile \
TEMP=/Volumes/Models/ANE/tmp_coreml_compile \
python3 tests/dev/state_transition_growing_inference.py \
  --contexts "512,1024,2048,3072,4096" \
  --contexts-root /Volumes/Models/ANE \
  --name-template "vibethinker_1.5b_ctx{context}_L6_4_hybrid" \
  --prompt "Fix this function (minimal changes) so it returns the correct median for odd/even lengths, does not modify the input list, and handles an empty list: def median(xs): xs.sort(); n=len(xs); mid=n//2; return (xs[mid]+xs[mid+1])/2 if n%2==1 else xs[mid]." \
  --max-tokens 4096 \
  --max-time 120 \
  --prefill-mode token-infer \
  --compute-unit CPU_AND_NE \
  --sampling-mode auto
```

### Example: Deterministic (Greedy)

```bash
python3 tests/dev/state_transition_growing_inference.py \
  --contexts "512,1024,2048,3072,4096" \
  --contexts-root /Volumes/Models/ANE \
  --name-template "vibethinker_1.5b_ctx{context}_L6_4_hybrid" \
  --prompt "Tell me about Apple Neural Engine" \
  --max-tokens 2000 \
  --prefill-mode batch-prefill \
  --compute-unit CPU_AND_NE \
  --sampling-mode greedy
```

### Example: Cap Growth At 2048

```bash
python3 tests/dev/state_transition_growing_inference.py \
  --contexts "512,1024,2048,3072,4096" \
  --max-context-size 2048 \
  --contexts-root /Volumes/Models/ANE \
  --name-template "vibethinker_1.5b_ctx{context}_L6_4_hybrid" \
  --prompt "Tell me about Apple Neural Engine" \
  --max-tokens 4000 \
  --prefill-mode batch-prefill \
  --compute-unit CPU_AND_NE \
  --sampling-mode auto
```

### Example: Cap Growth At 3072

```bash
python3 tests/dev/state_transition_growing_inference.py \
  --contexts "512,1024,2048,3072,4096" \
  --max-context-size 3072 \
  --contexts-root /Volumes/Models/ANE \
  --name-template "vibethinker_1.5b_ctx{context}_L6_4_hybrid" \
  --prompt "Tell me about Apple Neural Engine" \
  --max-tokens 6000 \
  --prefill-mode batch-prefill \
  --compute-unit CPU_AND_NE \
  --sampling-mode auto \
  --overflow-policy shift-refill \
  --overflow-reserve-batches 8
```

### Example: Long Run With Optional Shift/Refill

```bash
python3 tests/dev/state_transition_growing_inference.py \
  --contexts "512,1024,2048,3072,4096" \
  --contexts-root /Volumes/Models/ANE \
  --name-template "vibethinker_1.5b_ctx{context}_L6_4_hybrid" \
  --prompt "Fix this function (minimal changes) so it returns the correct median for odd/even lengths, does not modify the input list, and handles an empty list: def median(xs): xs.sort(); n=len(xs); mid=n//2; return (xs[mid]+xs[mid+1])/2 if n%2==1 else xs[mid]." \
  --max-tokens 24000 \
  --prefill-mode batch-prefill \
  --compute-unit CPU_AND_NE \
  --sampling-mode auto \
  --overflow-policy shift-refill \
  --overflow-reserve-batches 8 \
  --overflow-preserve-prompt
```

Observed summary (baseline run):

```text
prompt_tokens=97
prefill=764.4ms (126.9 t/s) context=512
decode_tokens=4000 decode_tps=17.0 final_context=4096
transitions:
  ctx512->ctx1024 at token_count=512 (21.1 ms)
  ctx1024->ctx2048 at token_count=1024 (59.1 ms)
  ctx2048->ctx3072 at token_count=2048 (96.6 ms)
  ctx3072->ctx4096 at token_count=3072 (139.1 ms)
per-context decode:
  ctx512: tokens=415 tps=42.5
  ctx1024: tokens=512 tps=31.4
  ctx2048: tokens=1024 tps=19.5
  ctx3072: tokens=1024 tps=14.9
  ctx4096: tokens=1025 tps=11.6
```

Notes:
- Throughput declines as context grows, as expected.
- Transition overhead remained small relative to decode time at each stage.
- Script now uses context-specific causal masks per active context, avoiding prompt-conditioning regressions.
- Shift/refill compaction is optional and only active with `--overflow-policy shift-refill`.
- `--overflow-preserve-prompt` keeps the original prompt as anchor during compact/refill.

---

---

## Hybrid Chunk1 Export (`proto_qwen25_chunk1_fp32.py`)

Script: `tests/dev/proto_qwen25_chunk1_fp32.py`

### Architecture

The hybrid chunk1 approach splits layer 0 into two separate CoreML models.
Artifact names depend on `--lut2` setting (suffix is `_lutN` or absent for `none`):

| Artifact (example with `--lut2 6,4`) | Contents | Precision | Compute |
|---------------------------------------|----------|-----------|---------|
| `{prefix}_FFN_attn_fp32_chunk_01ofNN` | Layer 0 attention only (residual) | FP32 | CPU_ONLY |
| `{prefix}_prefill_attn_fp32_chunk_01ofNN` | Layer 0 attention only (prefill) | FP32 | CPU_ONLY |
| `{prefix}_FFN_lut6_chunk_01ofNN` | Layer 0 MLP + layers 1-E (infer) | FP16+LUT | CPU_AND_NE |
| `{prefix}_prefill_lut6_chunk_01ofNN` | Layer 0 MLP + layers 1-E (prefill) | FP16+LUT | CPU_AND_NE |

Where `{prefix}` = `model_prefix` from meta.yaml (e.g. `qwen25`), `NN` = `num_chunks`, `E` = end layer for chunk 1 partition. The `_lut6` suffix changes based on `--lut2` (e.g. `_lut4` for `--lut2 4`, absent for `--lut2 none`).

Data flow (3-chunk hybrid, 28 layers, LUT6 example):
```
embeddings -> hidden_states
  -> FP32 attn chunk:  layer0 attention only (CPU)
  -> ANE chunk 01:     layer0 MLP + layers[1:10) (ANE, LUT6)
  -> ANE chunk 02:     layers[10:19) (ANE, LUT6)
  -> ANE chunk 03:     layers[19:28) + final_norm (ANE, LUT6)
  -> lm_head -> logits
```

The FP32 attention chunk runs layer 0 attention at full precision on CPU to avoid ANE quantization artifacts on the first layer, which is most sensitive to precision loss.

### Operating Modes

#### 1. `--rebuild-hybrid-chunk1` (Recommended)

Rebuilds all 4 hybrid chunk1 artifacts in one shot. Reads config from `meta.yaml` in `--out-dir`. No `--source-dir` needed.

```bash
python3 tests/dev/proto_qwen25_chunk1_fp32.py \
  --model-path ~/.cache/huggingface/hub/models--WeiboAI--VibeThinker-1.5B \
  --out-dir /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
  --rebuild-hybrid-chunk1
```

Produces:
- `qwen25_FFN_attn_fp32_chunk_01of03.mlpackage` + `.mlmodelc`
- `qwen25_prefill_attn_fp32_chunk_01of03.mlpackage` + `.mlmodelc`
- `qwen25_FFN_lut6_chunk_01of03.mlpackage` + `.mlmodelc`
- `qwen25_prefill_lut6_chunk_01of03.mlpackage` + `.mlmodelc`

All other files (chunks 02, 03, embeddings, lm_head, meta.yaml, tokenizer) are preserved.

#### 2. `--infer-only` / `--prefill-only`

Build only the FP32 attention chunk (infer, prefill, or both).

```bash
# FP32 attention infer only
python3 tests/dev/proto_qwen25_chunk1_fp32.py \
  --model-path ~/.cache/huggingface/hub/models--WeiboAI--VibeThinker-1.5B \
  --source-dir /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
  --out-dir /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
  --reuse-out-dir \
  --context-length 512 --batch-size 64 --num-chunks 3 \
  --lut2 6,4 \
  --infer-only \
  --infer-only-out-base qwen25_FFN_attn_fp32

# FP32 attention prefill only
python3 tests/dev/proto_qwen25_chunk1_fp32.py \
  --model-path ~/.cache/huggingface/hub/models--WeiboAI--VibeThinker-1.5B \
  --source-dir /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
  --out-dir /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
  --reuse-out-dir \
  --context-length 512 --batch-size 64 --num-chunks 3 \
  --lut2 6,4 \
  --prefill-only \
  --infer-only-out-base qwen25_FFN_attn_fp32

# Both FP32 infer + prefill
python3 tests/dev/proto_qwen25_chunk1_fp32.py \
  --model-path ~/.cache/huggingface/hub/models--WeiboAI--VibeThinker-1.5B \
  --source-dir /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
  --out-dir /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
  --reuse-out-dir \
  --context-length 512 --batch-size 64 --num-chunks 3 \
  --lut2 6,4 \
  --infer-only --prefill-only \
  --infer-only-out-base qwen25_FFN_attn_fp32
```

Output naming with `--infer-only-out-base qwen25_FFN_attn_fp32`:
- Infer: `qwen25_FFN_attn_fp32_chunk_01of03`
- Prefill: `qwen25_prefill_attn_fp32_chunk_01of03`

#### 3. `--rebuild-ffn-chunk N`

Rebuild a single ANE FFN+prefill chunk (1-indexed) with layer0 attention excluded.

```bash
# Rebuild chunk 1 (layer0 MLP + layers 1-9, infer + prefill)
python3 tests/dev/proto_qwen25_chunk1_fp32.py \
  --model-path ~/.cache/huggingface/hub/models--WeiboAI--VibeThinker-1.5B \
  --source-dir /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
  --out-dir /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
  --reuse-out-dir \
  --context-length 512 --batch-size 64 --num-chunks 3 \
  --lut2 6,4 \
  --rebuild-ffn-chunk 1
```

Produces:
- `qwen25_FFN_lut6_chunk_01of03.mlpackage` + `.mlmodelc`
- `qwen25_prefill_lut6_chunk_01of03.mlpackage` + `.mlmodelc`

For chunk 1 specifically: uses `RemainingSegmentInferWrapper(start=1, end=10, include_layer0_mlp=True)`.

#### 4. `--rebuild-prefill-chunk N`

Rebuild only the prefill for a single ANE chunk (1-indexed). Skips FFN infer.

```bash
# Rebuild prefill chunk 1 only (layer0 MLP + layers 1-9)
python3 tests/dev/proto_qwen25_chunk1_fp32.py \
  --model-path ~/.cache/huggingface/hub/models--WeiboAI--VibeThinker-1.5B \
  --source-dir /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
  --out-dir /Volumes/Models/ANE/vibethinker_1.5b_ctx512_L6_4_hybrid \
  --reuse-out-dir \
  --context-length 512 --batch-size 64 --num-chunks 3 \
  --lut2 6,4 \
  --rebuild-prefill-chunk 1
```

Produces only: `qwen25_prefill_lut6_chunk_01of03.mlpackage` + `.mlmodelc`

#### 5. Full Hybrid Build (default, no special flags)

Clones `--source-dir` to `--out-dir`, then rebuilds all chunks from scratch.

```bash
python3 tests/dev/proto_qwen25_chunk1_fp32.py \
  --model-path ~/.cache/huggingface/hub/models--WeiboAI--VibeThinker-1.5B \
  --source-dir /Volumes/Models/ANE/vibethinker_1.5b_ctx4096_fp16_hybrid \
  --out-dir /Volumes/Models/ANE/vibethinker_1.5b_ctx4096_L6_4_hybrid \
  --context-length 4096 --batch-size 64 \
  --remaining-chunks 2 \
  --lut1 none --lut2 6,4 --lut3 6,4 \
  --reuse-lm-head \
  --argmax-in-model false \
  --recommended-do-sample true \
  --recommended-temperature 0.6 \
  --recommended-top-p 0.95 \
  --recommended-top-k 0
```

**WARNING**: The full build deletes all existing `*_FFN_PF*_chunk_*` files before rebuilding. Use `--reuse-out-dir` to avoid re-cloning, but chunks are still cleaned.

**NOTE**: `--no-compile` is only honored in `--infer-only`, `--prefill-only`, `--rebuild-ffn-chunk`, `--rebuild-prefill-chunk`, and `--rebuild-hybrid-chunk1` modes. The full hybrid build path always compiles (LM head, chunk1, and post-attention chunks).

### All Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model-path` | str | `~/.cache/.../VibeThinker-1.5B` | HuggingFace model path |
| `--source-dir` | str | has default** | Existing converted model dir to clone from |
| `--out-dir` | str | has default** | Output directory |
| `--context-length` | int | from meta*** | Context length |
| `--batch-size` | int | from meta*** | Prefill batch size |
| `--num-chunks` | int | from meta*** | Total chunk count (for `--infer-only`/`--rebuild-*`) |
| `--remaining-chunks` | int | 3 | Post-attention segments for full hybrid mode |
| `--reuse-out-dir` | flag | false | Don't re-clone source dir |
| `--infer-only` | flag | false | Build FP32 attention infer only |
| `--prefill-only` | flag | false | Build FP32 attention prefill only |
| `--infer-only-out-base` | str | auto | Output base stem (e.g. `qwen25_FFN_attn_fp32`) |
| `--rebuild-ffn-chunk` | int | none | Rebuild single ANE FFN+prefill chunk (1-indexed) |
| `--rebuild-prefill-chunk` | int | none | Rebuild single ANE prefill chunk (1-indexed) |
| `--rebuild-hybrid-chunk1` | flag | false | Rebuild all 4 hybrid chunk1 artifacts |
| `--no-compile` | flag | false | Skip `.mlmodelc` compilation |
| `--no-quantize-ffn` | flag | false | Skip LUT quantization for FFN chunks |
| `--no-quantize-chunk2` | flag | false | Legacy alias for `--no-quantize-ffn` |
| `--copy-tail-from-source` | flag | false | Reuse source tail chunk |
| `--lut1` | str | from meta | LUT for embeddings (e.g. `none`, `6,4`) |
| `--lut2` | str | from meta | LUT for FFN chunks (e.g. `none`, `6,4`) |
| `--lut3` | str | from meta | LUT for lm_head (e.g. `none`, `6,4`) |
| `--reuse-lm-head` | flag | false | Reuse existing lm_head |
| `--argmax-in-model` | choice | auto | Argmax mode: `auto`, `true`, `false` |
| `--recommended-do-sample` | choice | auto | Sampling in meta: `auto`, `true`, `false` |
| `--recommended-temperature` | float | from meta | Temperature in meta |
| `--recommended-top-p` | float | from meta | Top-p in meta |
| `--recommended-top-k` | int | from meta | Top-k in meta |

*`--source-dir` not needed for `--rebuild-hybrid-chunk1` (reads meta from `--out-dir`).

**Script has hardcoded defaults for `--source-dir` and `--out-dir` but you should always specify them explicitly.

***In rebuild modes (`--rebuild-hybrid-chunk1`, `--rebuild-ffn-chunk`, `--rebuild-prefill-chunk`), `--context-length`, `--batch-size`, and `--num-chunks` are auto-read from `meta.yaml` when not explicitly passed on the CLI. In full hybrid mode or `--infer-only`/`--prefill-only`, CLI defaults apply (2048, 64, 3). Always verify the `[meta]` log lines confirm correct values.

### File Preservation

| Mode | Preserves existing files? |
|------|--------------------------|
| `--rebuild-hybrid-chunk1` | Yes, only overwrites 4 chunk1 artifacts |
| `--infer-only` / `--prefill-only` | Yes, only overwrites named chunk. **WARNING**: without `--infer-only-out-base`, writes to the regular FFN/prefill stem (e.g. `qwen25_FFN_lut6_chunk_01of03`), replacing ANE chunks with FP32 attention-only artifacts. Always use `--infer-only-out-base qwen25_FFN_attn_fp32` to write to separate files. |
| `--rebuild-ffn-chunk N` | Yes, only overwrites specific FFN + prefill chunk |
| `--rebuild-prefill-chunk N` | Yes, only overwrites specific prefill chunk |
| Full hybrid (default) | **No** - deletes all `*_FFN_PF*_chunk_*` before rebuild |

### Layer Partition (VibeThinker-1.5B, 28 layers, 3 chunks)

Standard converter partition:
- Chunk 1: layers 0-9 (10 layers)
- Chunk 2: layers 10-18 (9 layers)
- Chunk 3: layers 19-27 (9 layers) + final RMS norm

Hybrid chunk 1 split:
- FP32 attention: layer 0 attention only
- ANE chunk 01: layer 0 MLP + layers 1-9

---

*Last updated: 2026-02-10*

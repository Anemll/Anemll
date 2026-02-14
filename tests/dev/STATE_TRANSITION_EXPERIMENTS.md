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

*Last updated: 2026-01-24*

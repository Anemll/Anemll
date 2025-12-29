# MIL Builder (MB) Graph Issues and ANE Compatibility

This document covers MIL Builder patterns, ANE data type requirements, and troubleshooting workflow for building CoreML models that run on Apple Neural Engine.

For constant folding issues during model conversion, see [A1F_FOLDING_ISSUE.md](A1F_FOLDING_ISSUE.md).

---

## Table of Contents

1. [KV Cache State Pattern](#kv-cache-state-pattern)
2. [State Definition in MIL Program](#state-definition-in-mil-program)
3. [ANE Data Type Requirements](#ane-data-type-requirements)
4. [Gather Index Types](#gather-index-types)
5. [Position Preprocessing for RoPE](#position-preprocessing-for-rope)
6. [Common Errors and Fixes](#common-errors-and-fixes)
7. [Troubleshooting Workflow](#troubleshooting-workflow)

---

## KV Cache State Pattern

When building KV cache operations directly in MIL for ANE execution, the pattern matters significantly. ANE has specific requirements for how state operations must be chained.

### Incorrect Pattern (CPU/BNNS Fallback)

```python
# This pattern causes slice_update to run on CPU/BNNS
k_state = mb.read_state(input=k_state_input, name="k_read")
v_state = mb.read_state(input=v_state_input, name="v_read")

# ... compute k_new and v_new ...

# slice_update not directly connected to write_state
k_updated = mb.slice_update(x=k_state, update=k_new, begin=[0,0,pos,0], end=[...])
v_updated = mb.slice_update(x=v_state, update=v_new, begin=[0,0,pos,0], end=[...])

# Using updated values BEFORE write_state
k_for_attn = k_updated  # Problem: using slice_update output directly
```

**Error**: "Cannot support standalone slice_update" - runs on BNNS

### Correct Pattern (ANE Compatible)

**Option A: Separate K and V States**
```python
# For EACH state (K and V), the pattern must be:
# read_state -> slice_update -> write_state -> read_state

# K cache update
k_state = mb.read_state(input=k_state_input, name="k_read")
k_updated = mb.slice_update(x=k_state, update=k_new, begin=[0,0,pos,0], end=[...], name="k_slice_update")
mb.coreml_update_state(state=k_state_input, value=k_updated, name="k_write")
k_for_attn = mb.read_state(input=k_state_input, name="k_read_for_attn")

# V cache update (same pattern)
v_state = mb.read_state(input=v_state_input, name="v_read")
v_updated = mb.slice_update(x=v_state, update=v_new, begin=[0,0,pos,0], end=[...], name="v_slice_update")
mb.coreml_update_state(state=v_state_input, value=v_updated, name="v_write")
v_for_attn = mb.read_state(input=v_state_input, name="v_read_for_attn")
```

**Option B: Combined KV State (Recommended)**

Use a single state tensor with shape `[2, num_kv_heads, context_length, head_dim]` where index 0 = K, index 1 = V:

```python
# Combined KV cache: [2, num_kv_heads, context_length, head_dim]
# Index 0 = K cache, Index 1 = V cache

# Read current state
kv_state = mb.read_state(input=kv_cache_state, name="kv_read")

# Update K at index 0: begin=[0, 0, pos, 0], end=[1, 0, pos+1, 0] (with end_mask for heads/dim)
k_updated = mb.slice_update(x=kv_state, update=k_new,
    begin=[0, 0, pos, 0], end=[1, 0, pos+1, 0],
    end_mask=[False, True, False, True],  # True = use full extent
    name="k_cache_update")
mb.coreml_update_state(state=kv_cache_state, value=k_updated, name="k_write")

# Re-read after K update
kv_after_k = mb.read_state(input=kv_cache_state, name="kv_read_after_k")

# Update V at index 1: begin=[1, 0, pos, 0], end=[2, 0, pos+1, 0]
v_updated = mb.slice_update(x=kv_after_k, update=v_new,
    begin=[1, 0, pos, 0], end=[2, 0, pos+1, 0],
    end_mask=[False, True, False, True],
    name="v_cache_update")
mb.coreml_update_state(state=kv_cache_state, value=v_updated, name="v_write")

# Re-read after V update for attention
kv_after_v = mb.read_state(input=kv_cache_state, name="kv_read_after_v")

# Extract K and V for attention
k_for_attn = mb.slice_by_index(x=kv_after_v, begin=[0,0,0,0], end=[1,num_kv_heads,ctx,head_dim])
v_for_attn = mb.slice_by_index(x=kv_after_v, begin=[1,0,0,0], end=[2,num_kv_heads,ctx,head_dim])
```

### Key Requirements

1. `slice_update` output MUST flow directly into `coreml_update_state` (write_state)
2. For attention, read the state AGAIN after write_state
3. Each K and V update must follow the complete chain independently
4. The chain must be: `read` -> `slice_update` -> `write` -> `read`
5. Combined state is more memory efficient (single state vs two separate states)

### Flow Diagram: Combined KV State (Option B)

```
                              ┌─────────────────────────────────────┐
                              │         kv_cache_state              │
                              │   [2, num_kv_heads, ctx, head_dim]  │
                              │   index 0 = K, index 1 = V          │
                              └──────────────┬──────────────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    ▼                        │                        │
           ┌────────────────┐                │                        │
           │  read_state    │                │                        │
           │  "kv_read"     │                │                        │
           └───────┬────────┘                │                        │
                   │                         │                        │
                   ▼                         │                        │
           ┌────────────────┐                │                        │
           │  slice_update  │◄── k_new       │                        │
           │  begin=[0,...]  │   (computed)   │                        │
           │  "k_cache_upd" │                │                        │
           └───────┬────────┘                │                        │
                   │                         │                        │
                   ▼                         │                        │
           ┌────────────────┐                │                        │
           │  write_state   │────────────────┘                        │
           │  "k_write"     │  (updates state)                        │
           └───────┬────────┘                                         │
                   │                                                  │
                   ▼                                                  │
           ┌────────────────┐                                         │
           │  read_state    │                                         │
           │  "kv_after_k"  │                                         │
           └───────┬────────┘                                         │
                   │                                                  │
                   ▼                                                  │
           ┌────────────────┐                                         │
           │  slice_update  │◄── v_new                                │
           │  begin=[1,...]  │   (computed)                            │
           │  "v_cache_upd" │                                         │
           └───────┬────────┘                                         │
                   │                                                  │
                   ▼                                                  │
           ┌────────────────┐                                         │
           │  write_state   │─────────────────────────────────────────┘
           │  "v_write"     │  (updates state)
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐
           │  read_state    │
           │  "kv_after_v"  │
           └───────┬────────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
   ┌──────────────┐  ┌──────────────┐
   │slice_by_index│  │slice_by_index│
   │ begin=[0,..] │  │ begin=[1,..] │
   │ "k_for_attn" │  │ "v_for_attn" │
   └──────┬───────┘  └──────┬───────┘
          │                 │
          ▼                 ▼
      K for Attn        V for Attn
```

### Flow Diagram: Separate K/V States (Option A)

```
   ┌───────────────┐              ┌───────────────┐
   │  k_cache_state│              │  v_cache_state│
   └───────┬───────┘              └───────┬───────┘
           │                              │
           ▼                              ▼
   ┌───────────────┐              ┌───────────────┐
   │  read_state   │              │  read_state   │
   │   "k_read"    │              │   "v_read"    │
   └───────┬───────┘              └───────┬───────┘
           │                              │
           ▼                              ▼
   ┌───────────────┐              ┌───────────────┐
   │ slice_update  │◄── k_new     │ slice_update  │◄── v_new
   │"k_slice_upd"  │              │"v_slice_upd"  │
   └───────┬───────┘              └───────┬───────┘
           │                              │
           ▼                              ▼
   ┌───────────────┐              ┌───────────────┐
   │  write_state  │──┐           │  write_state  │──┐
   │   "k_write"   │  │           │   "v_write"   │  │
   └───────────────┘  │           └───────────────┘  │
                      │                              │
           ┌──────────┘                   ┌──────────┘
           │                              │
           ▼                              ▼
   ┌───────────────┐              ┌───────────────┐
   │  read_state   │              │  read_state   │
   │"k_for_attn"   │              │"v_for_attn"   │
   └───────┬───────┘              └───────┬───────┘
           │                              │
           ▼                              ▼
       K for Attn                     V for Attn
```

### Critical: What NOT to Do

```
   ┌───────────────┐
   │  kv_cache_state│
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │  read_state   │
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │ slice_update  │◄── k_new
   └───────┬───────┘
           │
           ├──────────────────┐
           │                  │
           ▼                  ▼
   ┌───────────────┐   ┌──────────────┐
   │  write_state  │   │ USE DIRECTLY │  ◄── WRONG! Causes BNNS fallback
   └───────────────┘   │ for attention│
                       └──────────────┘
```

The `slice_update` output must go ONLY to `write_state`, then you must `read_state` again to get the updated values for attention.

---

## State Definition in MIL Program

**Option A: Separate K and V States**
```python
from coremltools.converters.mil.mil import types

@mb.program(
    input_specs=[
        mb.TensorSpec(shape=(1, 1, hidden_size), dtype=types.fp16),
        mb.TensorSpec(shape=(1,), dtype=types.int32),
        mb.StateTensorSpec(shape=(1, num_kv_heads, context_length, kv_head_dim), dtype=types.fp16),  # K state
        mb.StateTensorSpec(shape=(1, num_kv_heads, context_length, kv_head_dim), dtype=types.fp16),  # V state
        mb.TensorSpec(shape=(1, 1, 1, context_length), dtype=types.fp16),
    ],
    opset_version=ct.target.iOS18
)
def transformer(hidden_states, position_ids, k_cache_state, v_cache_state, causal_mask):
    # ... implementation using Option A pattern above ...
```

**Option B: Combined KV State (Recommended)**
```python
from coremltools.converters.mil.mil import types

@mb.program(
    input_specs=[
        mb.TensorSpec(shape=(1, hidden_size, 1, 1), dtype=types.fp16),  # Conv2d format
        mb.TensorSpec(shape=(1,), dtype=types.int32),
        mb.StateTensorSpec(shape=(2, num_kv_heads, context_length, kv_head_dim), dtype=types.fp16),  # Combined KV
        mb.TensorSpec(shape=(1, 1, 1, context_length), dtype=types.fp16),
    ],
    opset_version=ct.target.iOS18
)
def transformer(hidden_states, position_ids, kv_cache_state, causal_mask):
    # ... implementation using Option B pattern above ...
```

**Benefits of Combined State**:
- Single state tensor for both K and V caches
- First dimension indexes: 0 = K, 1 = V
- More memory efficient
- Simpler state management

---

## ANE Data Type Requirements

### Integer Types on ANE

ANE has limited support for integer operations. Key constraints:

| Type | Supported on ANE | Notes |
|------|-----------------|-------|
| int32 | No | Falls back to BNNS/CPU |
| int16 | Limited | Some ops only |
| uint16 | Yes | Preferred for indices |
| fp16 | Yes | Preferred for data |

---

## Gather Index Types

The `mb.gather()` operation requires specific index types for ANE execution:

### int32 Indices (CPU Fallback)

```python
# Error: "Unsupported tensor data type: int32"
cos_pos = mb.gather(x=cos_table, indices=position_ids, axis=1)  # position_ids is int32
```

### int16 Indices (May Still Fall Back)

```python
# Error: "Unsupported gather index type" on some ANE versions
pos_int16 = mb.cast(x=position_ids, dtype="int16")
cos_pos = mb.gather(x=cos_table, indices=pos_int16, axis=1)  # May not work
```

### uint16 Indices (ANE Compatible)

```python
# Use uint16 for ALL gather indices
pos_uint16 = mb.cast(x=position_ids, dtype="uint16")
cos_pos = mb.gather(x=cos_table, indices=pos_uint16, axis=1, name="cos_gather")  # ANE!
sin_pos = mb.gather(x=sin_table, indices=pos_uint16, axis=1, name="sin_gather")  # ANE!
```

---

## Position Preprocessing for RoPE

When casting from int32 to uint16, add preprocessing to match traced model patterns:

```python
# Position preprocessing chain (matches traced model pattern)
# Done ONCE before layer loop for efficiency

# 1. Check if position >= 0
pos_ge_zero = mb.greater_equal(x=position_ids, y=np.int32(0), name="pos_ge_zero")

# 2. Compute fallback position (pos + 2*context_length for negative positions)
pos_plus_2ctx = mb.add(x=position_ids, y=np.int32(context_length * 2), name="pos_plus_2ctx")

# 3. Select: if pos >= 0, use pos; else use pos + 2*ctx
pos_selected = mb.select(a=position_ids, b=pos_plus_2ctx, cond=pos_ge_zero, name="pos_selected")

# 4. Cast to uint16 for gather
pos_uint16 = mb.cast(x=pos_selected, dtype="uint16", name="pos_uint16")

# Now use pos_uint16 in all layers
for layer_idx in range(num_layers):
    cos_pos = mb.gather(x=cos_table, indices=pos_uint16, axis=1, name=f"layer{layer_idx}_cos")
    sin_pos = mb.gather(x=sin_table, indices=pos_uint16, axis=1, name=f"layer{layer_idx}_sin")
```

**Note**: The `cast`, `greater_equal`, `add`, and `select` operations will run on BNNS (CPU). This is unavoidable because ANE doesn't support int32. However, the actual `gather` operations will run on ANE.

---

## Common Errors and Fixes

### Error: "Cannot support standalone slice_update"

**Symptom**: KV cache updates fall back to BNNS

**Cause**: `slice_update` output is used directly instead of going through write_state first

**Fix**: Use the correct chain: `read_state` -> `slice_update` -> `write_state` -> `read_state`

---

### Error: "Unsupported tensor data type: int32"

**Symptom**: Gather or other int operations fall back to BNNS

**Cause**: ANE doesn't support int32 operations

**Fix**: Cast to int16 or uint16 before the operation

---

### Error: "Unsupported gather index type"

**Symptom**: Gather with int16 indices still falls back

**Cause**: Some gather implementations require uint16

**Fix**: Use `mb.cast(x=..., dtype="uint16")` for gather indices

---

### Error: Cast operations on CPU despite uint16 usage

**Symptom**: Cast from int32 runs on BNNS, gather runs on ANE

**Cause**: ANE cannot execute int32 operations including cast FROM int32

**Status**: This is expected behavior. The cast will always run on BNNS, but subsequent operations using the uint16 result can run on ANE.

**Optimization**: Do the cast ONCE before layer loops, not inside each layer.

---

## Troubleshooting Workflow

### Step 1: Generate Performance Report

Run your model with profiling to identify which ops run where:

```bash
# Use Xcode Instruments or coremltools prediction with compute_unit logging
python -c "
import coremltools as ct
model = ct.models.MLModel('your_model.mlpackage', compute_units=ct.ComputeUnit.CPU_AND_NE)
# Run prediction and check Instruments
"
```

### Step 2: Compare with Traced Model

Create a traced reference model using `torch.jit.trace`:

```python
# Trace the PyTorch model
traced = torch.jit.trace(model, example_inputs)

# Convert to CoreML
mlmodel_traced = ct.convert(
    traced,
    inputs=[...],
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT16,
)

# Save and examine MIL
mlmodel_traced.save("/tmp/traced_reference.mlpackage")
```

### Step 3: Compare MIL Files

Extract and compare the MIL from both models:

```bash
# Find and compare model.mil files
diff /tmp/traced_reference.mlmodelc/model.mil /tmp/your_model.mlmodelc/model.mil
```

Look for differences in:
- Operation order
- Data types (int32 vs uint16)
- State operation chaining
- Op names and connectivity

### Step 4: Match Traced Model Patterns

When you find a difference, update your hand-built MIL to match the traced model's pattern exactly:

| What to Check | Traced Model | Your Model | Fix |
|---------------|--------------|------------|-----|
| Gather index type | uint16 | int32 | Add cast to uint16 |
| State chain | read->slice->write->read | read->slice->use | Add write_state + second read |
| Position preprocessing | greater_equal->add->select | direct cast | Add preprocessing chain |
| Loop invariants | Outside loop | Inside loop | Move outside loop |

### Step 5: Verify with Instruments

After fixes, run Instruments "Core ML" template to verify:

1. Open Instruments
2. Select "Core ML" template
3. Run your app/test
4. Check the "Neural Engine" track for operation placement

Expected results after fixes:
- `slice_update`, `gather`, `matmul`, `conv`: ANE
- `cast` (from int32), `greater_equal`, `add`, `select`: BNNS (unavoidable)

---

## Version Info

- **coremltools**: 8.2+
- **PyTorch**: 2.0+
- **Python**: 3.9
- **Target**: iOS18+

---

*Document created: 2025-12-29*
*Split from A1F_FOLDING_ISSUE.md*

# Debug Scripts Usage Guide

## Custom Prompt Support

All debug scripts now support custom prompts via command line arguments.

### ANEMLL Debugging
```bash
# Default: single token "What" (lightweight, no BOS token)
python debug_tensor_comparison_simple.py

# Custom prompt (with BOS token and full tokenization)
python debug_tensor_comparison_simple.py "The weather today is"
python debug_tensor_comparison_simple.py "Hello world"
python debug_tensor_comparison_simple.py "Explain quantum physics"
```

### HuggingFace Debugging
```bash
# Default: single token "What" (lightweight, no BOS token)
python debug_hf_model.py

# Custom prompt (with BOS token and full tokenization)
python debug_hf_model.py "The weather today is"
python debug_hf_model.py "Hello world"
python debug_hf_model.py "Explain quantum physics"
```

### Tensor Comparison
```bash
# Default: single token "What" comparison (lightweight)
python compare_tensor_debug_outputs.py

# Custom prompt comparison (same prompt for both models)
python compare_tensor_debug_outputs.py "The weather today is"
python compare_tensor_debug_outputs.py "Hello world"
python compare_tensor_debug_outputs.py "Explain quantum physics"
```

## Example Output

### Default (Lightweight)
```bash
$ python debug_hf_model.py
Loading HuggingFace model for debug trace...
🎯 Using default: single token 'What' (ID: 3689) - lightweight debugging
💡 For custom prompts, use: python debug_hf_model.py "Your prompt here"
==================================================
🔬 TRACING HF FORWARD PASS FOR LAYER 0
==================================================
```

### Custom Prompt
```bash
$ python debug_hf_model.py "Hello world"
Loading HuggingFace model for debug trace...
🎯 Using custom prompt: 'Hello world'
📝 Token IDs: [2, 15496, 2146]
==================================================
🔬 TRACING HF FORWARD PASS FOR LAYER 0
==================================================
```

## Key Features

1. **Lightweight Default**: Single token "What" without BOS for focused layer debugging
2. **Automatic Tokenization**: Custom prompts are automatically converted to token IDs
3. **Consistent Prompts**: Same prompt used for both ANEMLL and HF models in comparison
4. **Token ID Display**: Shows actual token IDs being processed for transparency
5. **No Heavy Attention**: Default mode avoids complex multi-token attention patterns

## Prompt Selection Tips

- **Default (no args)**: Single token "What" - best for layer debugging without attention complexity
- **Short prompts** (2-5 tokens) for focused debugging with minimal attention
- **Common phrases** for typical model behavior analysis
- **Domain-specific prompts** for testing specific capabilities
- **Multi-word prompts** for testing tokenization edge cases

## Debugging Workflow

### Quick Start (Recommended)
```bash
# Start with lightweight comparison
python compare_tensor_debug_outputs.py

# If you see issues, test with your specific prompt
python compare_tensor_debug_outputs.py "Your problematic prompt here"
```

### Detailed Analysis
```bash
# Examine individual tensor flows
python debug_tensor_comparison_simple.py "Your prompt"
python debug_hf_model.py "Your prompt"
```

### Progressive Debugging
1. **Start with default** (single token) to establish lightweight baseline behavior
2. **Test specific prompts** that show issues in your application  
3. **Compare single vs multi-token** to understand attention impact
4. **Use domain-specific prompts** for targeted debugging

### Script Behavior Summary

| Script | Purpose | Default Behavior | With Prompt |
|--------|---------|-----------------|-------------|
| `debug_tensor_comparison_simple.py` | ANEMLL tensor tracing | Single token "What" | Full tokenization |
| `debug_hf_model.py` | HF tensor tracing | Single token "What" | Full tokenization |
| `compare_tensor_debug_outputs.py` | Automated comparison | Runs both with default | Runs both with prompt |

**Note**: Individual debug scripts show detailed tensor traces but NO model outputs or predictions. For model predictions, use `test_simple_gemma_arch.py`.

---

## The Debugging & Comparison Workflow

The primary goal of this debug suite is to find the first point of numerical divergence between our local model (`SimpleGemma3nModel`) and the official Hugging Face reference implementation.

The process revolves around three core scripts and two tensor files:
- **Reference Tensors**: `tests/dev/debug_tensors/hf_tensors.pt`
- **Our Tensors**: `tests/dev/debug_tensors/anemll_tensors.pt`

### The 3-Step Process

1.  **Generate Reference Tensors**:
    ```bash
    python tests/dev/debug_hf_model.py
    ```
    This script loads the official Hugging Face model, runs a forward pass, and saves the captured intermediate tensors to `hf_tensors.pt`. You typically only need to run this once unless you change the fundamental model structure or input prompt.

2.  **Generate ANEMLL Tensors**:
    ```bash
    python tests/dev/debug_anemll_model.py
    ```
    This runs our local `SimpleGemma3nModel` from `test_simple_gemma_arch.py` and saves its intermediate tensors to `anemll_tensors.pt`.

3.  **Compare Tensors**:
    ```bash
    python tests/dev/compare_outputs.py
    ```
    This is the main script you will run repeatedly. It loads both `.pt` files and compares each tensor one-by-one, following the execution order defined in the script. It will print `✅ MATCH` or `🔴 MISMATCH` for each tensor and stop at the first point of failure.

### The Iterative Cycle

The typical debugging loop is as follows:
1.  Run `python tests/dev/compare_outputs.py`.
2.  Identify the **first** tensor that reports a `🔴 MISMATCH`.
3.  Analyze the code in `test_simple_gemma_arch.py` that is responsible for generating that specific tensor.
4.  Propose a fix in `test_simple_gemma_arch.py`.
5.  Re-run `python tests/dev/compare_outputs.py` (it will automatically re-generate the ANEMLL tensors before comparing).
6.  Repeat until all tensors match.

---

## How to Add New Debugger Hooks

If you need to inspect a tensor that isn't already being captured, follow these four steps to add a new hook.

### Step 1: Instrument the Model (`test_simple_gemma_arch.py`)

Find the spot in the model's `forward` pass where your tensor of interest is calculated. Add a hook call, making sure to use a unique string for the hook name.

**Example**: Capturing the output of an activation function in the MLP.
```python
# In SimpleGemma3nMLP.forward()...
gate_proj = self.act_fn(gate_proj)
# Add the hook right after the calculation
if 'mlp_gate_after_act' in self.debugger_hooks:
    self.debugger_hooks['mlp_gate_after_act'](gate_proj)
```

### Step 2: Register the Hook in the ANEMLL Debug Script (`debug_anemll_model.py`)

Now, tell the ANEMLL debug script to listen for this new hook.

**Example**:
```python
# In main() of debug_anemll_model.py, find the layer you hooked.
# For a hook in a specific layer (e.g., the MLP in layer 0):
layer = model.layers[0]
layer.mlp.debugger_hooks['mlp_gate_after_act'] = get_hook('mlp_gate_after_act')

# For a hook in the main model body:
model.debugger_hooks['my_main_model_hook'] = get_hook('my_main_model_hook')
```

### Step 3: Register the Hook in the HuggingFace Debug Script (`debug_hf_model.py`)

This script uses monkey-patching to capture tensors. You need to find the equivalent module in the HF model and attach a forward hook.

**Example**:
```python
# In capture_hf_tensors() of debug_hf_model.py...
# Find the target module in the HuggingFace model hierarchy.
# This may require printing/inspecting the `hf_model` object.
target_module = hf_model.model.language_model.layers[0].mlp.gate_proj
# The hook captures the *output* of the module.
# Since we want the output *after* the activation, we hook the gate_proj
# and perform the activation manually inside the hook.
# (This is an advanced case; often you just hook the output directly).
target_module.register_forward_hook(
    get_activation('mlp_gate_after_act', apply_act=True)
)
```

### Step 4: Add the Hook to the Comparison Order (`compare_outputs.py`)

Finally, add the unique hook name to the `TENSOR_ORDER` list in the comparison script. This ensures it's checked in the correct sequence.

**Example**:
```python
# In compare_outputs.py
TENSOR_ORDER = [
    # ...
    'mlp_gate_proj',
    'mlp_gate_after_act', # Add the new hook name here
    'mlp_up_proj',
    # ...
]
```
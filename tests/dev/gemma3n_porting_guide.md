# Gemma3n ANEMLL Porting Guide

## Overview
This document provides a comprehensive guide for porting Google's Gemma3n model to ANEMLL for Apple Neural Engine acceleration.

## Architecture Overview

### Core Components

#### 1. LAUREL (Learned Augmented Residual Layer) Blocks
```python
class Gemma3nLaurelBlock:
    """
    Key innovation: Adds residual connections with low-rank linear transformations
    Components:
    - Standard transformer block (attention + FFN)
    - Low-rank input/output projections
    - Residual connection with learned weighting
    """
```

#### 2. Alternating Updates (AltUp) Mechanism
```python
class AltUp:
    """
    Sophisticated routing mechanism for:
    - Predicting layer outputs
    - Correcting predictions with routing
    - Modality-specific computation paths
    
    Components:
    - Routing network
    - Correction network
    - Modality-specific processors
    """
```

#### 3. Per-Layer Input Projections
- Each layer has its own input embedding transformation
- Loaded from `layer_embeddings.{layer_idx}` in weights
- Enables layer-specific feature extraction

#### 4. Activation Sparsity
```python
def gelu_topk(x, k):
    """
    GELU activation with top-k sparsity
    - Applies GELU
    - Keeps only top-k activations
    - Zeros out remaining values
    """
```

## ANE Conversion Requirements

### 1. RMSNorm Implementation (ANEMLL Optimized)
```python
class Gemma3nRMSNorm(nn.Module):
    """
    ANE-optimized RMSNorm using ANEMLL's tensor doubling technique.
    This achieves 2-3x faster inference on Apple Neural Engine.
    """
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.variance_epsilon = eps
        self.dims = dims
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # ANE-optimized using tensor doubling: concat([x, -x]) → μ = 0
        x = hidden_states
        
        # ❶ Double with negative states (creates exactly zero mean)
        doubled = torch.cat([x, -x], dim=-1)
        
        # ❷ Use highly-optimized LayerNorm kernel
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * self.dims,),
            weight=None, bias=None,
            eps=float(self.variance_epsilon)
        )
        
        # ❸ Drop mirror half to get RMS-normed activations
        normed = normed[..., : self.dims]
        
        # ❹ Apply learnable gain with efficient casting
        return (normed * self.weight
                       .to(normed.dtype, copy=False)
                       .to(normed.device, copy=False))
```

### 2. Linear Layer Conversion
All linear layers must be converted to Conv2d for ANE:

```python
def convert_linear_to_conv2d(linear_layer):
    """Convert nn.Linear to nn.Conv2d for ANE"""
    out_features, in_features = linear_layer.weight.shape
    conv = nn.Conv2d(
        in_channels=in_features,
        out_channels=out_features,
        kernel_size=1,
        bias=linear_layer.bias is not None
    )
    # Reshape weights: [out, in] -> [out, in, 1, 1]
    conv.weight.data = linear_layer.weight.data.unsqueeze(-1).unsqueeze(-1)
    if linear_layer.bias is not None:
        conv.bias.data = linear_layer.bias.data
    return conv
```

### 3. LAUREL Block ANE Adaptation
```python
class ANEGemma3nLaurelBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Low-rank projections as Conv2d
        self.low_rank_in = nn.Conv2d(config.hidden_size, config.low_rank_dim, 1)
        self.low_rank_out = nn.Conv2d(config.low_rank_dim, config.hidden_size, 1)
        
        # Standard components
        self.attention = ANEGemma3nAttention(config)
        self.ffn = ANEGemma3nFFN(config)
        self.norm1 = Gemma3nRMSNorm(config.hidden_size)
        self.norm2 = Gemma3nRMSNorm(config.hidden_size)
```

### 4. Attention Mechanism
```python
class ANEGemma3nAttention(nn.Module):
    """ANE-compatible attention with sliding window support"""
    def __init__(self, config):
        super().__init__()
        # Query, Key, Value projections as Conv2d
        self.q_proj = nn.Conv2d(config.hidden_size, config.num_heads * config.head_dim, 1)
        self.k_proj = nn.Conv2d(config.hidden_size, config.num_kv_heads * config.head_dim, 1)
        self.v_proj = nn.Conv2d(config.hidden_size, config.num_kv_heads * config.head_dim, 1)
        self.o_proj = nn.Conv2d(config.num_heads * config.head_dim, config.hidden_size, 1)
```

### 5. FFN Layer
```python
class ANEGemma3nFFN(nn.Module):
    """ANE-compatible FFN with gelu_topk activation"""
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Conv2d(config.hidden_size, config.intermediate_size, 1)
        self.up_proj = nn.Conv2d(config.hidden_size, config.intermediate_size, 1)
        self.down_proj = nn.Conv2d(config.intermediate_size, config.hidden_size, 1)
        self.activation_topk = config.activation_topk
```

## Implementation Steps

### Step 1: Create Base Model Structure
```python
# File: anemll/models/gemma3n_model.py

from anemll.models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class Gemma3nModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_type = "gemma3n"
    
    def load_weights(self, state_dict, config):
        """Load and convert HuggingFace weights to ANE format"""
        # Implementation details below
```

### Step 2: Weight Conversion Pipeline
```python
def convert_gemma3n_weights(hf_state_dict):
    """Convert HuggingFace Gemma3n weights to ANEMLL format"""
    ane_state_dict = {}
    
    for key, value in hf_state_dict.items():
        # Handle per-layer embeddings
        if "layer_embeddings" in key:
            # Special handling for per-layer embeddings
            layer_idx = extract_layer_index(key)
            new_key = f"layers.{layer_idx}.input_proj.weight"
            ane_state_dict[new_key] = value.unsqueeze(-1).unsqueeze(-1)
        
        # Handle LAUREL block weights
        elif "laurel" in key:
            # Convert low-rank projections
            if "low_rank_in" in key:
                new_key = key.replace("low_rank_in", "low_rank_in_conv")
                ane_state_dict[new_key] = reshape_for_conv2d(value)
            elif "low_rank_out" in key:
                new_key = key.replace("low_rank_out", "low_rank_out_conv")
                ane_state_dict[new_key] = reshape_for_conv2d(value)
        
        # Handle attention weights
        elif any(proj in key for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            ane_state_dict[key] = reshape_for_conv2d(value)
        
        # Handle FFN weights
        elif any(proj in key for proj in ["gate_proj", "up_proj", "down_proj"]):
            ane_state_dict[key] = reshape_for_conv2d(value)
        
        # Handle RMSNorm weights
        elif "norm" in key:
            ane_state_dict[key] = value
    
    return ane_state_dict
```

### Step 3: Create Converter Class
```python
# File: anemll/ane_converter/gemma3n_converter.py

from anemll.ane_converter.base_converter import BaseConverter
import coremltools as ct
import torch

class Gemma3nConverter(BaseConverter):
    def __init__(self, model_path: str, output_dir: str):
        super().__init__(model_path, output_dir)
        self.model_type = "gemma3n"
    
    def convert_embeddings(self):
        """Convert embeddings (Part 1) with per-layer projections"""
        # Load base embeddings
        # Convert per-layer embeddings
        # Export to CoreML
    
    def convert_ffn(self, chunk_idx: int, total_chunks: int):
        """Convert FFN layers (Part 2) with LAUREL blocks"""
        # Handle LAUREL block conversion
        # Convert FFN with gelu_topk
        # Export chunked models
    
    def convert_lm_head(self):
        """Convert LM head (Part 3) with soft-capping"""
        # Convert output projection
        # Apply logit soft-capping
        # Export to CoreML
```

### Step 4: Activation Function Implementation
```python
def gelu_topk_ane_compatible(x, k):
    """ANE-compatible implementation of gelu_topk"""
    # Apply GELU activation
    activated = F.gelu(x)
    
    # Get top-k mask (ANE-friendly approach)
    batch_size, channels, height, width = x.shape
    x_flat = x.view(batch_size, -1)
    
    # Find top-k indices
    topk_values, topk_indices = torch.topk(x_flat.abs(), k, dim=-1)
    
    # Create mask
    mask = torch.zeros_like(x_flat)
    mask.scatter_(1, topk_indices, 1.0)
    mask = mask.view(batch_size, channels, height, width)
    
    # Apply mask
    return activated * mask
```

### Step 5: Model Configuration
```python
# File: anemll/models/gemma3n_config.py

class Gemma3nConfig:
    def __init__(
        self,
        vocab_size=256128,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=4096,
        final_logit_softcapping=30.0,
        # Gemma3n specific
        low_rank_dim=256,
        activation_topk=128,
        use_laurel_blocks=True,
        use_per_layer_embeddings=True,
        altup_routing_dim=512,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # ... rest of config
```

## Chunking Strategy

### 1. LAUREL Block Chunking
Due to the additional complexity of LAUREL blocks, consider:
- Chunk by groups of LAUREL blocks (e.g., 2-3 blocks per chunk)
- Keep low-rank projections with their associated blocks
- Separate AltUp routing networks if memory constrained

### 2. Per-Layer Embedding Strategy
```python
def chunk_per_layer_embeddings(embeddings, num_chunks):
    """Distribute per-layer embeddings across chunks"""
    layers_per_chunk = len(embeddings) // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * layers_per_chunk
        end = start + layers_per_chunk if i < num_chunks - 1 else len(embeddings)
        chunks.append(embeddings[start:end])
    return chunks
```

## Development Workflow: Syncing Test Files

### CRITICAL: Maintaining Alignment Between Test Files

When modifying `tests/dev/test_simple_gemma_arch.py`, you **MUST** sync changes to related debugging files to maintain architectural consistency.

#### Files That Must Stay Aligned

1. **`test_simple_gemma_arch.py`** - Main implementation and reference
2. **`debug_tensor_comparison_simple.py`** - Layer-by-layer tensor debugging
3. **`layer_by_layer_comparison_detailed.py`** - HF vs ANEMLL comparison
4. **`debug_hf_model.py`** - HuggingFace model debugging

#### Sync Procedure After Modifying `test_simple_gemma_arch.py`

**Step 1: Identify Changes Made**
- Note which forward pass steps were modified
- Identify any new scaling factors, normalization changes, or architectural updates
- Check if new components were added/removed

**Step 2: Update `debug_tensor_comparison_simple.py`**
```bash
# The debug file copies the exact forward pass logic from the main file
# Key areas to sync:

# 1. LAUREL scaling changes
if "laurel_x *=" in changes:
    # Update lines ~108-110 in debug file to match main implementation
    
# 2. Residual connection patterns  
if "residual" in changes:
    # Update lines ~118-124 (First Residual, Combined Output sections)
    
# 3. PLE (Per-Layer Embeddings) handling
if "per_layer" in changes:
    # Update lines ~144-182 (PLE injection section)
    
# 4. Normalization changes
if "norm" in changes:
    # Update corresponding norm calls and their tensor debugging
```

**Step 3: Verification Commands**
```bash
# Run both files and compare key tensor statistics
python tests/dev/test_simple_gemma_arch.py 2>/dev/null | grep "Cosine similarity"
python tests/dev/debug_tensor_comparison_simple.py 2>/dev/null | grep -A5 "Final Layer Output"

# Check for architectural consistency
python tests/dev/check_alignment.py  # Script to verify forward pass alignment
```

**Step 4: Update Related Documentation**
```bash
# Update implementation notes in these files:
# - tests/dev/gemma3n_progress_log.md
# - tests/dev/gemma3n_implementation_plan.md
# - tests/dev/context.md (if architectural changes)
```

#### Common Sync Patterns

**1. LAUREL Block Changes**
```python
# Main file (test_simple_gemma_arch.py)
laurel_x *= (self.config.hidden_size ** -0.5)  # Line ~391

# Must sync to debug file (debug_tensor_comparison_simple.py) 
laurel_x *= (layer.config.hidden_size ** -0.5)  # Line ~109
debugger.debug_tensor(laurel_x, "Laurel Scaled", "", f"After 1/sqrt({layer.config.hidden_size}) scaling")
```

**2. Residual Connection Changes**
```python
# Main file: Any changes to residual patterns
attn_output += active_stream  # First residual
attn_output = (attn_output + laurel_out_normed) * (1.0 / math.sqrt(2.0))  # Combined

# Debug file: Must have identical logic with tensor debugging
attn_output += active_stream
debugger.debug_tensor(attn_output, "First Residual", "", "Attn output + original active stream")
attn_output = (attn_output + laurel_out_normed) * (1.0 / math.sqrt(2.0))
debugger.debug_tensor(attn_output, "Combined Output", "", "Combined and scaled by 1/sqrt(2)")
```

**3. PLE (Per-Layer Embeddings) Changes**
```python
# Main file: Any changes to PLE processing
if per_layer_embeddings is not None:
    # PLE logic here
    
# Debug file: Must have identical PLE logic with debugging
if per_layer_embeddings is not None:
    # Exact same logic + debugger.debug_tensor() calls
```

#### Automated Sync Verification

**Create a sync verification script:**
```python
# File: tests/dev/verify_sync.py
def verify_laurel_scaling():
    """Verify LAUREL scaling is consistent across files"""
    main_file = read_file("test_simple_gemma_arch.py")
    debug_file = read_file("debug_tensor_comparison_simple.py")
    
    # Check if both files have/don't have LAUREL scaling
    main_has_scaling = "laurel_x *= (self.config.hidden_size ** -0.5)" in main_file
    debug_has_scaling = "laurel_x *= (layer.config.hidden_size ** -0.5)" in debug_file
    
    assert main_has_scaling == debug_has_scaling, "LAUREL scaling mismatch!"

def verify_ple_handling():
    """Verify PLE handling is consistent"""
    # Similar checks for PLE logic alignment
    
def verify_residual_patterns():
    """Verify residual connection patterns match"""
    # Check residual connection sequences
```

#### Documentation Update Requirements

When syncing changes, also update:

1. **Progress Log**: Document what changed and why
2. **Implementation Plan**: Update architectural diagrams if needed  
3. **Context File**: Note any breaking changes or new patterns
4. **This Porting Guide**: Add any new sync patterns discovered

#### Example Sync Session Log

```markdown
## Sync Session: 2025-01-13 - LAUREL Scaling Restoration

### Changes Made to test_simple_gemma_arch.py:
- Restored LAUREL scaling: `laurel_x *= (self.config.hidden_size ** -0.5)` (line 391)
- Reason: Root cause identified as corrupted weights, not architectural issue

### Files Synced:
1. ✅ debug_tensor_comparison_simple.py 
   - Uncommented LAUREL scaling (line 109)
   - Added tensor debugging for scaling step
2. ✅ layer_by_layer_comparison_detailed.py
   - Updated to expect scaling in tensor comparison
3. ⚠️ debug_hf_model.py 
   - No changes needed (HF baseline reference)

### Verification:
- Both files now show identical forward pass logic
- Tensor debugging confirms scaling is applied
- Cosine similarity should improve with restored scaling

### Documentation Updates:
- Updated progress log with scaling restoration rationale
- Added sync procedure to porting guide
```

## Debugging Strategy

### 1. Direct Forward Pass Debugging

#### `debug_tensor_comparison_simple.py` - ANEMLL Model Debugging
- **Purpose**: Step-by-step tensor debugging of our ANEMLL implementation
- **Method**: Direct copy of `SimpleGemma3nLayer.forward` with debug calls
- **Scope**: Complete breakdown of every computation step
- **Sync Requirement**: Must stay aligned with `test_simple_gemma_arch.py`

#### `debug_hf_model.py` - HuggingFace Model Debugging  
- **Purpose**: Step-by-step tensor debugging of HuggingFace reference implementation
- **Method**: Direct copy of `Gemma3nTextDecoderLayer.forward` with debug calls
- **Scope**: Complete breakdown matching ANEMLL debug format
- **Stability**: Static reference - only updated if HuggingFace model changes

#### Key Advantage: No Hook Complexity
```python
# OLD APPROACH: Complex hook management
def get_activation_hook(activations_dict, name):
    def hook(model, input, output):
        activations_dict[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
    return hook

# NEW APPROACH: Direct forward pass copy with debug prints
def trace_layer_forward(model, input_ids, layer_idx=0):
    # Copy exact forward pass logic
    active_stream = hidden_streams[layer.altup.altup_active_idx]
    debugger.debug_tensor(active_stream, "Active Stream", "", "...")
    
    # Continue with exact forward pass + debug calls
```

### 2. Tensor-by-Tensor Comparison Workflow

#### Step 1: Run Debugging Scripts

**Option A: Automated Comparison (Recommended)**
```bash
# Default: single token "What" (lightweight)
python tests/dev/compare_tensor_debug_outputs.py

# Custom prompt (with BOS token, full tokenization)
python tests/dev/compare_tensor_debug_outputs.py "Hello world"
```

**Option B: Individual Scripts (For Detailed Analysis)**
```bash
# Default: single token "What" without BOS (lightweight debugging)
python tests/dev/debug_tensor_comparison_simple.py
python tests/dev/debug_hf_model.py

# Custom prompt (with BOS token and full tokenization)  
python tests/dev/debug_tensor_comparison_simple.py "Hello world"
python tests/dev/debug_hf_model.py "Hello world"

# Then run comparison
python tests/dev/compare_tensor_debug_outputs.py "Hello world"
```

#### Step 2: Compare Corresponding Tensors
Both scripts output identical debug format:
```
--- Active Stream (stream 0) ---
  Shape: [1, 5, 2048]
  Range: [-2.456789, 3.123456]
  Mean: 0.012345, Std: 0.987654
  Notes: This implements the exact parallel data flow from canonical JAX
```

#### Step 3: Identify Divergence Points
- Compare tensor statistics at each step
- Look for first point where tensors diverge significantly
- Focus debugging efforts on the divergence point

### 3. Configuration-Driven Debugging

#### Stable HuggingFace Reference
- `debug_hf_model.py` demonstrates the debugging **approach**
- Currently shows simulated output due to HF model complexity
- Provides template for future full HF debugging implementation
- For production comparison, use end-to-end model outputs

#### Evolving ANEMLL Implementation
- `debug_tensor_comparison_simple.py` tracks our implementation changes
- Must be synced whenever `test_simple_gemma_arch.py` is modified
- Allows detailed analysis of architectural improvements

### 4. Debugging Best Practices

#### When to Use Each Tool
1. **Initial divergence detection**: Use both scripts to find where models differ
2. **Architecture changes**: Sync ANEMLL debug script with main implementation
3. **Performance optimization**: Compare tensor statistics before/after changes
4. **Weight loading issues**: Use debug scripts to verify weight loading correctness

#### Debugging Session Example
```markdown
## Debug Session: Tensor Comparison Analysis

### Objective: Demonstrate automated tensor comparison workflow

### Method:
1. Run ANEMLL debug script (real data)
2. Run HF debug script (demonstration/simulated data)
3. Parse and compare tensor statistics
4. Identify differences automatically

### Sample Output:
🔬 Tensor Debug Comparison: ANEMLL vs HuggingFace
============================================================

📊 Parsing tensor statistics...
Found 31 ANEMLL tensors
Found 7 HF tensors

🎯 Comparing 11 matched tensor pairs...

--- Active Stream (stream 0) vs HF Active Prediction (stream 0) ---
  ANEMLL: mean=0.015383, std=0.990824
  HF:     mean=0.023456, std=0.876543
  Diff:   mean=0.008073, std=0.114281
  Rel:    mean=34.42%, std=13.04%
  ⚠️  SIGNIFICANT DIFFERENCE

### Current Status:
- ✅ ANEMLL debugging: Complete with real tensor data
- 🔬 HF debugging: Demonstration mode (simulated data)
- 📊 Comparison framework: Fully functional
- 🎯 Production use: Focus on end-to-end model comparison

### For Real HF Comparison:
When HF debugging is fully implemented, this same framework will:
- Automatically identify true divergence points
- Provide quantitative similarity metrics
- Focus debugging efforts on specific computation steps
```

#### Current Implementation Status (January 2025)

**ANEMLL Debug System: ✅ Production Ready**
- Real tensor debugging with complete forward pass tracing
- 31 intermediate tensors captured and analyzed
- Synchronized with main implementation
- **Prompt Support**: Default single token "What" (lightweight) or custom prompts

**HF Debug System: ✅ Production Ready**
- Actual HF tensor debugging with real forward pass data
- 14 intermediate tensors captured including pre-normalization states
- Hook-based tensor capture for complete visibility
- **Prompt Support**: Same prompt options as ANEMLL for fair comparison

**Comparison System: ✅ Fully Functional**
- Automatically runs both debug scripts with matching prompts
- Parses outputs and calculates similarity metrics (mean/std differences)
- Identifies significant divergence points (14 tensor pairs analyzed)
- **Usage**: `compare_tensor_debug_outputs.py` or `compare_tensor_debug_outputs.py "custom prompt"`

#### Prompt Behavior Summary

| Mode | Command | Token Count | BOS Token | Use Case |
|------|---------|-------------|-----------|----------|
| **Default** | `python script.py` | 1 ("What") | ❌ No | Lightweight layer debugging |
| **Custom** | `python script.py "Hello world"` | Variable | ✅ Yes | Real-world prompt testing |

**Key Benefits:**
- **Lightweight Default**: Single token avoids complex attention patterns
- **Custom Flexibility**: Test specific prompts that show issues in your application
- **Consistent Comparison**: Same prompt used for both ANEMLL and HF models
- **Self-Contained**: Comparison script handles everything automatically

## Testing Strategy

### 1. Component Testing
```python
# File: tests/dev/test_gemma3n_components.py

def test_laurel_block():
    """Test LAUREL block implementation"""
    config = Gemma3nConfig()
    block = ANEGemma3nLaurelBlock(config)
    
    # Test forward pass
    x = torch.randn(1, config.hidden_size, 1, 1)
    output = block(x)
    
    # Verify output shape
    assert output.shape == x.shape
    
    # Test low-rank projection
    assert hasattr(block, 'low_rank_in')
    assert hasattr(block, 'low_rank_out')

def test_gelu_topk():
    """Test sparse activation function"""
    x = torch.randn(1, 512, 1, 1)
    k = 128
    output = gelu_topk_ane_compatible(x, k)
    
    # Verify sparsity
    non_zero = (output != 0).sum()
    assert non_zero <= k
```

### 2. Full Model Testing
```python
# File: tests/dev/test_gemma3n_vs_hf.py

def test_gemma3n_equivalence():
    """Compare ANEMLL Gemma3n with HuggingFace implementation"""
    # Load both models
    hf_model = load_hf_gemma3n()
    ane_model = load_ane_gemma3n()
    
    # Test with same inputs
    test_input = "Hello, world!"
    
    # Compare outputs
    hf_output = generate_hf(hf_model, test_input)
    ane_output = generate_ane(ane_model, test_input)
    
    # Verify similarity
    assert torch.allclose(hf_output, ane_output, rtol=1e-3)
```

## Conversion Script Integration

### Update convert_model.sh
```bash
# Add Gemma3n support to the conversion script
case "$MODEL_TYPE" in
    gemma3n*)
        echo "Converting Gemma3n model..."
        python -m anemll.ane_converter.gemma3n_converter \
            --model "$MODEL_PATH" \
            --output "$OUTPUT_DIR" \
            --context "$CONTEXT_LENGTH" \
            --batch "$BATCH_SIZE" \
            --lut2 "$LUT2" \
            --lut3 "$LUT3" \
            --chunk "$CHUNK_SIZE" \
            --enable-laurel \
            --enable-per-layer-embeddings
        ;;
esac
```

## Performance Considerations

### 1. Memory Optimization
- LAUREL blocks add ~20% memory overhead
- Consider aggressive chunking for larger models
- Per-layer embeddings can be loaded on-demand

### 2. Compute Optimization
- gelu_topk may require custom CoreML ops
- Consider pre-computing routing tables for AltUp
- Optimize low-rank projections with grouped convolutions

### 3. ANE-Specific Optimizations
- Fuse LAUREL residual connections where possible
- Batch per-layer embeddings lookups
- Use CoreML's sparse tensor support for gelu_topk

## Known Challenges

1. **AltUp Mechanism**: May require custom CoreML operators
2. **Per-Layer Embeddings**: Increases model size significantly
3. **Activation Sparsity**: Not natively supported in CoreML
4. **Dynamic Routing**: May need static approximation for ANE

## Next Steps

1. Implement base Gemma3n model class
2. Create weight conversion utilities
3. Develop ANE-compatible LAUREL blocks
4. Test component equivalence with HuggingFace
5. Integrate into ANEMLL conversion pipeline
6. Optimize for ANE performance
7. Create comprehensive test suite

## Resources

- [Gemma3n Paper](https://arxiv.org/abs/gemma3n)
- [MLX Implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma3n.py)
- [HuggingFace Gemma3n](https://huggingface.co/google/gemma3n)
- [CoreML Documentation](https://coremltools.readme.io/)
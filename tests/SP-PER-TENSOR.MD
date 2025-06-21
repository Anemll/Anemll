# Per-Tensor Quantization Support for Qwen 2.5

## PRD Summary

### Objective
Add per-tensor quantization support to the Qwen 2.5 model implementation to enable loading and inference with 4-bit per-tensor quantized models from HuggingFace (specifically `smpanaro/Qwen2.5-0.5B-4bit-PerTensor`).

### Requirements
1. Support loading quantized weights where `stored_weight = weight × input_scale × output_scale`
2. Apply quantization scaling during inference when `ENABLE_SP_QUANT=True`
3. Maintain compatibility with non-quantized models
4. **Correct bias handling**: `loaded_bias = stored_bias / output_scale` (since bias is added before output scaling)
5. **Correct weight handling**: `loaded_weight = stored_weight / (output_scale * input_scale)` (de-fuse)
6. Apply input scales for Conv2D : `x = Conv2D(x*  × input_scale)`
7 Apply output scales after Conv2D operations: `x = Conv2D(x) × output_scale`
7. Exclude LM_HEAD layers from quantization
8. Handle quantized model format compatibility (codebooks, bias differences)

9. store output_scale and inputscale in class buffer for conv2D
## Implementation Changes

### 1. Added Utility Function (`unbake_output_scale`)
```python
def unbake_output_scale(weight: torch.Tensor, output_scale: torch.Tensor) -> torch.Tensor:
    """Remove output scales from weights for per-tensor quantization.
    
    Quantized weights are stored as: weight = weight * input_scale * output_scale
    Since we apply output_scale after Conv2D, we need: weight = weight / output_scale
    """
    return weight / output_scale
```

### 2. Modified Qwen25MLP Class
- Added output scale buffers: `gate_proj_output_scale`, `up_proj_output_scale`, `down_proj_output_scale`
- **Updated bias configuration**: Enabled bias when `ENABLE_SP_QUANT=True` for all MLP projections
- Applied output scales after each Conv2D operation in forward pass

### 3. Modified Qwen25Attention Class
- Added output scale buffers: `q_proj_output_scale`, `k_proj_output_scale`, `v_proj_output_scale`, `o_proj_output_scale`
- **Updated bias configuration**: Enabled bias for `o_proj` when `ENABLE_SP_QUANT=True`
- Applied output scales after projections in both single-token and prefill methods
- Maintained existing bias configuration (q/k/v always have bias, o has bias only when quantized)

### 4. Updated Weight Loading Logic
Enhanced three-pass loading process:
1. **First pass**: Collect all `input_scales` and `output_scales` from state dict
2. **Second pass**: Process weights and biases
   - **Skip codebook tensors**: Ignore quantization codebooks that aren't needed
   - **Weight processing**: Remove output scales from weights using `unbake_output_scale` (since weights are stored as `weight * input_scale * output_scale`)
   - **Bias processing**: Remove output scales from biases using `unbake_output_scale` (since biases need to be unscaled before output scaling is applied)
   - Load all projection layers (including o_proj when quantized)
3. **Third pass**: Load output scales into model buffers
   - **Shape handling**: Convert per-channel scales to scalars using mean
   - Convert key format from `.output_scales` to `_output_scale`
   - Handle missing scales gracefully

### 5. Updated Dependency Checker
Modified `check_dependencies.sh` to allow quantized models when `ENABLE_SP_QUANT=true`:
- Added environment variable check for `ENABLE_SP_QUANT`
- Skip quantization rejection when per-tensor quantization is enabled
- Added informational message when quantized models are allowed

### 6. Enhanced Test Infrastructure
- **`tests/dev/test_qwen25_sp_quant.py`**: Standalone test with auto-download from HuggingFace
- **`tests/test_sp_qwen2.5_model.py`**: Enhanced integration test with:
  - Automatic output directory cleanup
  - Quantization config removal from downloaded models
  - Environment variable setup for `ENABLE_SP_QUANT=true`

## Key Design Decisions
1. **Conditional Quantization**: Only active when `ENABLE_SP_QUANT=True`
2. **Input Scale Baking**: Input scales are multiplied into weights during loading (not stored separately)
3. **Output Scale Storage**: Output scales are stored as model buffers and applied during forward pass
4. **Enhanced Bias Support**: Both MLP and attention layers have conditional bias support based on quantization mode
5. **Codebook Handling**: Quantization codebooks are ignored during loading (not needed for inference)
6. **Shape Flexibility**: Handle different scale tensor shapes (per-channel vs scalar) automatically
7. **Compatibility**: Non-quantized models continue to work unchanged

## Testing
The implementation can be tested with:
```bash
# Full conversion pipeline test (with environment variable)
ENABLE_SP_QUANT=true 

=

#CONVERT ALL 
python tests/test_sp_qwen2.5_model.py

# Standalone test with auto-download
python tests/dev/test_qwen25_sp_quant.py

#compare output vs transformers
python tests/dev/test_final_inference.py

# TEST CONVERT ONLY  FFN ONLY (part5_)
python tests/dev/test_ffn_sp_qwen2_5_model.py 
```

Both tests will download the quantized model from HuggingFace if not cached locally and verify that quantization scales are properly loaded and applied during inference.

## Model Format
The per-tensor quantized model format includes:
- `model.layers.{i}.mlp.{gate,up,down}_proj.weight`: Pre-dequantized weights
- `model.layers.{i}.mlp.{gate,up,down}_proj.input_scales`: Input scales (baked into weights)
- `model.layers.{i}.mlp.{gate,up,down}_proj.output_scales`: Output scales (applied after Conv2D)
- `model.layers.{i}.mlp.{gate,up,down}_proj.bias`: Bias terms (when quantized)
- `model.layers.{i}.mlp.{gate,up,down}_proj.codebook`: Quantization codebooks (ignored)
- Similar structure for attention layers (q_proj, k_proj, v_proj, o_proj)

## Implementation Notes
1. **Codebooks are ignored**: The quantization codebooks are not used in our implementation
2. **Scale shapes**: Output scales may be per-channel or scalar - automatically handled
3. **Bias differences**: Quantized models may have different bias configurations than unquantized
4. **Environment dependency**: Requires `ENABLE_SP_QUANT=true` environment variable for dependency checker

## Critical Bug Fix: Scale Tensor Broadcasting Issue

### Problem
When applying scales during forward pass, PyTorch was creating sparse COO tensors due to improper broadcasting between scale tensors and Conv2D outputs:
- Scale tensors from HF checkpoints: shapes like `[4864, 1]` or `[896, 1]` 
- Conv2D outputs: shapes like `[1, 4864, 1, 1]` or `[1, 896, 1, 1]`
- Operations like `tensor * scale.unsqueeze(0).unsqueeze(-1)` caused dimension mismatches
- PyTorch created sparse COO tensors which failed on `permute()` operations during CoreML conversion

### Root Cause  
Broadcasting `[1, 4864, 1, 1] * [4864, 1]` after unsqueeze operations created sparse COO tensors because:
1. The scale tensor shapes from checkpoints vary: `(out_dim,)` or `(out_dim, 1)`
2. When unsqueezed incorrectly, they don't match 4D Conv2D output dimensions
3. PyTorch falls back to sparse representation for efficiency
4. Sparse tensors fail on operations expecting dense tensors (like `permute(0,2,1)`)

### Solution
**Always use `.view(1, -1, 1, 1)` for all scale applications** instead of `.unsqueeze()` operations:

```python
# ❌ WRONG - Creates sparse COO tensors
a = conv_output * scale.unsqueeze(0).unsqueeze(-1)

# ✅ CORRECT - Ensures proper 4D dense tensor
a = conv_output * scale.view(1, -1, 1, 1)
```

### Implementation Pattern
Apply this fix to ALL scale operations in the model:
- MLP forward: `gate_proj_output_scale`, `up_proj_output_scale`, `down_proj_output_scale`
- Attention forward: `q_proj_input_scale`, `q_proj_output_scale`, `k_proj_input_scale`, etc.
- All helper methods: `get_new_kv_cache`, `get_new_kv_cache_prefill`, `forward_regular`, `forward_prefill`

### Error Symptom
```
permute(sparse_coo): number of dimensions in the tensor input does not match 
the length of the desired ordering of dimensions i.e. input.dim() = 4 is not equal to len(dims) = 3
```

This error occurs during CoreML conversion when the trace encounters sparse tensors in operations expecting dense tensors.

## Reference
- Model: https://huggingface.co/smpanaro/Qwen2.5-0.5B-4bit-PerTensor
- Colab: https://gist.github.com/smpanaro/5890838e424b2970a287e4a05f9049b6
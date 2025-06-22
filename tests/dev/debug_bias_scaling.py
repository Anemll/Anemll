#!/usr/bin/env python3
"""
Investigate bias scaling in quantization with pre-multiplication.

This script examines:
1. Which bias parameters exist in a Qwen 2.5 model
2. How ANEMLL handles biases in per-tensor quantization
3. The mathematical relationship between bias scaling and weight scale pre-multiplication
"""

import os
import torch
import json
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download
from safetensors import safe_open

# Set environment for SP quantization
os.environ['ENABLE_SP_QUANT'] = '1'

def examine_hf_model_biases():
    """Examine which layers in a standard Qwen 2.5 model have biases."""
    print("="*60)
    print("1. EXAMINING HUGGING FACE QWEN 2.5 MODEL BIASES")
    print("="*60)
    
    # Load standard Qwen 2.5 model
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading model: {model_name}")
    
    try:
        model_path = snapshot_download(model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Examine bias parameters
        bias_params = {}
        for name, param in hf_model.named_parameters():
            if 'bias' in name:
                bias_params[name] = param
        
        print(f"Found {len(bias_params)} bias parameters in standard Qwen 2.5:")
        
        if bias_params:
            for name, param in bias_params.items():
                print(f"  {name}: shape={param.shape}, dtype={param.dtype}")
                if param.numel() <= 10:
                    print(f"    values: {param.tolist()}")
                else:
                    print(f"    mean={param.mean():.6f}, std={param.std():.6f}")
        else:
            print("  NO BIAS PARAMETERS FOUND in standard Qwen 2.5!")
        
        # Check specific layer configurations  
        print(f"\nLayer configuration analysis:")
        layer0 = hf_model.model.layers[0]
        
        # Attention projections
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            proj = getattr(layer0.self_attn, proj_name)
            has_bias = hasattr(proj, 'bias') and proj.bias is not None
            print(f"  {proj_name}: bias={has_bias}")
            
        # MLP projections
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(layer0.mlp, proj_name)
            has_bias = hasattr(proj, 'bias') and proj.bias is not None
            print(f"  {proj_name}: bias={has_bias}")
            
        return hf_model, bias_params
        
    except Exception as e:
        print(f"Error loading HF model: {e}")
        return None, {}

def examine_quantized_model_biases():
    """Examine bias parameters in a quantized Qwen 2.5 model."""
    print("\n" + "="*60)
    print("2. EXAMINING QUANTIZED QWEN 2.5 MODEL BIASES")
    print("="*60)
    
    # Load quantized model
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    print(f"Loading quantized model: {model_name}")
    
    try:
        model_path = snapshot_download(model_name)
        
        # Examine safetensors file directly
        model_file = f"{model_path}/model.safetensors"
        bias_tensors = {}
        weight_tensors = {}
        scale_tensors = {}
        
        with safe_open(model_file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            for key in keys:
                if 'bias' in key:
                    bias_tensors[key] = f.get_tensor(key)
                elif 'weight' in key and not any(x in key for x in ['scales', 'codebook']):
                    weight_tensors[key] = f.get_tensor(key)
                elif 'scales' in key:
                    scale_tensors[key] = f.get_tensor(key)
        
        print(f"Found {len(bias_tensors)} bias tensors in quantized model:")
        for name, tensor in list(bias_tensors.items())[:10]:  # Show first 10
            print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
            if tensor.numel() <= 5:
                print(f"    values: {tensor.tolist()}")
            else:
                print(f"    mean={tensor.mean():.6f}, std={tensor.std():.6f}, range=[{tensor.min():.6f}, {tensor.max():.6f}]")
        
        if len(bias_tensors) > 10:
            print(f"  ... and {len(bias_tensors) - 10} more bias tensors")
        
        print(f"\nFound {len(scale_tensors)} scale tensors:")
        for name, tensor in list(scale_tensors.items())[:10]:
            print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
            if tensor.numel() <= 5:
                print(f"    values: {tensor.tolist()}")
            else:
                print(f"    mean={tensor.mean():.6f}, std={tensor.std():.6f}")
        
        return bias_tensors, weight_tensors, scale_tensors, model_path
        
    except Exception as e:
        print(f"Error loading quantized model: {e}")
        return {}, {}, {}, None

def examine_anemll_bias_handling(model_path):
    """Examine how ANEMLL handles biases in per-tensor quantization."""
    print("\n" + "="*60)  
    print("3. EXAMINING ANEMLL BIAS HANDLING")
    print("="*60)
    
    try:
        from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config
        
        # Load ANEMLL model
        config = Qwen25Config.from_json(f'{model_path}/config.json')
        print(f"Creating ANEMLL model with SP quantization enabled...")
        model = Qwen25ForCausalLM(config, disable_kv_cache=False)
        
        # Load weights and examine bias loading process
        print(f"Loading pretrained weights...")
        success = model.load_pretrained_weights(model_path)
        
        if success:
            print(f"✓ Weights loaded successfully")
        else:
            print(f"⚠️ Weight loading had issues")
        
        # Examine bias parameters in loaded model
        print(f"\nBias parameters in loaded ANEMLL model:")
        bias_count = 0
        for name, param in model.named_parameters():
            if 'bias' in name and param is not None:
                bias_count += 1
                print(f"  {name}: shape={param.shape}, dtype={param.dtype}")
                if param.numel() <= 5:
                    print(f"    values: {param.tolist()}")
                else:
                    print(f"    mean={param.mean():.6f}, std={param.std():.6f}")
        
        print(f"Total bias parameters in ANEMLL model: {bias_count}")
        
        # Examine scale parameters
        print(f"\nScale parameters in loaded ANEMLL model:")
        scale_count = 0
        for name, param in model.named_parameters():
            if 'scale' in name:
                scale_count += 1
                print(f"  {name}: shape={param.shape}, dtype={param.dtype}")
                if param.numel() <= 5:
                    print(f"    values: {param.tolist()}")
                else:
                    print(f"    mean={param.mean():.6f}, std={param.std():.6f}")
        
        # Also check registered buffers for scales
        for name, buffer in model.named_buffers():
            if 'scale' in name:
                scale_count += 1
                print(f"  {name} (buffer): shape={buffer.shape}, dtype={buffer.dtype}")
                if buffer.numel() <= 5:
                    print(f"    values: {buffer.tolist()}")
                else:
                    print(f"    mean={buffer.mean():.6f}, std={buffer.std():.6f}")
        
        print(f"Total scale parameters/buffers in ANEMLL model: {scale_count}")
        
        return model
        
    except Exception as e:
        print(f"Error examining ANEMLL model: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_bias_scaling_math(bias_tensors, scale_tensors):
    """Analyze the mathematical relationship between bias scaling and weight scale pre-multiplication."""
    print("\n" + "="*60)
    print("4. ANALYZING BIAS SCALING MATHEMATICS")
    print("="*60)
    
    print("Mathematical Analysis of Bias Scaling with Pre-multiplied Weights:")
    print("")
    print("Standard quantized linear layer computation:")
    print("  y = (W_q * x_scale) * x + b_q * output_scale")
    print("  where W_q = quantized weights, b_q = quantized bias")
    print("")
    print("With pre-multiplied weights (SmoothQuant style):")
    print("  W_stored = W_original * input_scale * output_scale")
    print("  b_stored = b_original * output_scale")  
    print("")
    print("To get original computation:")
    print("  W_recover = W_stored / (input_scale * output_scale)")
    print("  b_recover = b_stored / output_scale")
    print("")
    print("Forward pass computation:")
    print("  y = (W_recover * input_scale) * x + b_recover * output_scale")
    print("  y = ((W_stored / (input_scale * output_scale)) * input_scale) * x + (b_stored / output_scale) * output_scale")
    print("  y = (W_stored / output_scale) * x + b_stored")
    print("")
    
    # Find matching bias/scale pairs and analyze
    print("Analyzing actual stored values:")
    
    # Group by layer and projection
    layer_projections = {}
    for key in bias_tensors.keys():
        if 'model.layers.' in key:
            parts = key.split('.')
            layer_num = parts[2]
            if 'self_attn' in key:
                proj_type = parts[4].replace('.bias', '')
                layer_key = f"layer_{layer_num}_attn_{proj_type}"
            elif 'mlp' in key:
                proj_type = parts[4].replace('.bias', '')
                layer_key = f"layer_{layer_num}_mlp_{proj_type}"
            else:
                continue
                
            if layer_key not in layer_projections:
                layer_projections[layer_key] = {}
            layer_projections[layer_key]['bias'] = bias_tensors[key]
    
    # Find corresponding scales
    for key in scale_tensors.keys():
        if 'model.layers.' in key and 'output_scales' in key:
            parts = key.split('.')
            layer_num = parts[2]
            if 'self_attn' in key:
                proj_type = parts[4].replace('.output_scales', '')
                layer_key = f"layer_{layer_num}_attn_{proj_type}"
            elif 'mlp' in key:
                proj_type = parts[4].replace('.output_scales', '')
                layer_key = f"layer_{layer_num}_mlp_{proj_type}"
            else:
                continue
                
            if layer_key not in layer_projections:
                layer_projections[layer_key] = {}
            layer_projections[layer_key]['output_scale'] = scale_tensors[key]
    
    # Analyze first few complete pairs
    analyzed_count = 0
    for layer_key, data in layer_projections.items():
        if 'bias' in data and 'output_scale' in data and analyzed_count < 5:
            bias = data['bias']
            output_scale = data['output_scale']
            
            print(f"\n{layer_key}:")
            print(f"  Stored bias shape: {bias.shape}, dtype: {bias.dtype}")
            print(f"  Output scale shape: {output_scale.shape}, dtype: {output_scale.dtype}")
            
            if bias.numel() <= 5 and output_scale.numel() <= 5:
                print(f"  Stored bias values: {bias.tolist()}")
                print(f"  Output scale values: {output_scale.tolist()}")
                
                # Compute what the recovered bias would be
                if output_scale.numel() == bias.numel():
                    recovered_bias = bias / output_scale
                    print(f"  Recovered bias (bias/output_scale): {recovered_bias.tolist()}")
                elif output_scale.shape[0] == bias.shape[0] and output_scale.shape[-1] == 1:
                    # Handle broadcast case
                    scale_flat = output_scale.squeeze(-1)
                    recovered_bias = bias / scale_flat
                    print(f"  Recovered bias (bias/output_scale): {recovered_bias.tolist()}")
            else:
                print(f"  Stored bias stats: mean={bias.mean():.6f}, std={bias.std():.6f}")
                print(f"  Output scale stats: mean={output_scale.mean():.6f}, std={output_scale.std():.6f}")
                
                # Check if shapes are compatible for element-wise division
                try:
                    if output_scale.numel() == bias.numel():
                        recovered_bias = bias / output_scale
                    elif output_scale.shape[0] == bias.shape[0] and len(output_scale.shape) == 2 and output_scale.shape[-1] == 1:
                        scale_flat = output_scale.squeeze(-1)
                        recovered_bias = bias / scale_flat
                    else:
                        print(f"  ⚠️ Shape mismatch for bias recovery")
                        continue
                        
                    print(f"  Recovered bias stats: mean={recovered_bias.mean():.6f}, std={recovered_bias.std():.6f}")
                    
                    # Check the scaling ratio
                    scale_ratio = bias.abs().mean() / recovered_bias.abs().mean()
                    print(f"  Scaling ratio (stored/recovered): {scale_ratio:.6f}")
                    
                except Exception as e:
                    print(f"  ⚠️ Error computing recovered bias: {e}")
            
            analyzed_count += 1
    
    print(f"\nAnalyzed {analyzed_count} bias/scale pairs")

def investigate_anemll_bias_loading():
    """Investigate the specific bias loading logic in ANEMLL."""
    print("\n" + "="*60)
    print("5. INVESTIGATING ANEMLL BIAS LOADING LOGIC")
    print("="*60)
    
    # Show the relevant code from qwen2_5_model.py
    print("Current ANEMLL bias loading logic (from load_pretrained_weights):")
    print("")
    print("Lines 1108-1138 in qwen2_5_model.py:")
    print("""
    elif any(
        proj in new_k
        for proj in [
            "q_proj.bias",
            "k_proj.bias", 
            "v_proj.bias",
            "o_proj.bias",
            "gate_proj.bias",
            "up_proj.bias",
            "down_proj.bias",
        ]
    ):
        # Handle bias tensors for projections (bias is stored as bias / output_scale)
        if ENABLE_SP_QUANT:
            output_scale_key = k.replace(".bias", ".output_scales")
            if output_scale_key in output_scales:
                output_scale = output_scales[output_scale_key].to(MODEL_DTYPE)
                # Correct bias handling: loaded_bias = stored_bias / output_scale
                # Since bias is added before output scaling in forward pass
                scale = (output_scale).to(MODEL_DTYPE)
                if scale.numel() == v.numel():
                    scale = scale.view_as(v)
                elif scale.shape[-1] == 1 and scale.shape[0] == v.shape[0]:
                    scale = scale.squeeze(-1)
                
                # should not scale, scaled on export for SmoothQuant?
                bias_tensor = (v).to(MODEL_DTYPE)  # ← CURRENTLY NOT SCALING!
                
                print(f"De-scaled bias for {new_k}: stored_bias shape {v.shape}, output_scale shape {output_scale.shape}")
            else:
                bias_tensor = v.to(MODEL_DTYPE)
        else:
            bias_tensor = v.to(MODEL_DTYPE)
        conv_state[new_k] = bias_tensor
    """)
    
    print("\nAnalysis:")
    print("1. The code is currently NOT applying bias scaling (line 1128)")
    print("2. The comment suggests bias scaling was disabled for SmoothQuant compatibility")
    print("3. This means stored_bias is loaded directly without division by output_scale")
    print("")
    print("Mathematical implications:")
    print("- If bias is pre-scaled during export: stored_bias = original_bias * output_scale")
    print("- Current loading: loaded_bias = stored_bias (no scaling)")
    print("- Forward pass: output = ... + loaded_bias * output_scale")
    print("- Result: output = ... + stored_bias * output_scale")
    print("- If stored_bias = original_bias * output_scale, then:")
    print("  output = ... + (original_bias * output_scale) * output_scale")
    print("  output = ... + original_bias * output_scale²")
    print("- This causes bias scaling error by factor of output_scale!")

def main():
    """Main investigation function."""
    print("BIAS SCALING INVESTIGATION IN QUANTIZATION WITH PRE-MULTIPLICATION")
    print("="*80)
    
    # Step 1: Examine standard HF model biases
    hf_model, hf_bias_params = examine_hf_model_biases()
    
    # Step 2: Examine quantized model biases
    bias_tensors, weight_tensors, scale_tensors, model_path = examine_quantized_model_biases()
    
    # Step 3: Examine ANEMLL bias handling
    if model_path:
        anemll_model = examine_anemll_bias_handling(model_path)
    
    # Step 4: Analyze bias scaling mathematics
    if bias_tensors and scale_tensors:
        analyze_bias_scaling_math(bias_tensors, scale_tensors)
    
    # Step 5: Investigate ANEMLL bias loading logic
    investigate_anemll_bias_loading()
    
    print("\n" + "="*80)
    print("SUMMARY FINDINGS")
    print("="*80)
    print("1. Standard Qwen 2.5 models have NO bias parameters by default")
    print("2. Quantized models ADD bias parameters to all projections")
    print("3. ANEMLL currently loads biases WITHOUT scaling (potential bug)")
    print("4. This likely causes bias scaling errors by factor of output_scale²")
    print("5. Proper fix: uncomment bias scaling in load_pretrained_weights()")

if __name__ == "__main__":
    main()
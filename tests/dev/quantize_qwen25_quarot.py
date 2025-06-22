#!/usr/bin/env python3
"""
CPU-only QuaRot quantization for Qwen 2.5 models.
Adapted from QuaRot's fake_quant functionality.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file
# Import QuaRot fusion functionality
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from quarot_fusion import apply_quarot_fusion_to_model, verify_fusion_quality


class QuantConfig:
    """Quantization configuration similar to QuaRot args."""
    def __init__(self):
        # Weight quantization
        self.w_bits = 4
        self.w_groupsize = -1
        self.w_asym = False
        self.w_rtn = True  # Use RTN instead of GPTQ for simplicity
        
        # Activation quantization  
        self.a_bits = 16
        self.a_groupsize = -1
        self.a_asym = False
        self.a_clip_ratio = 1.0
        
        # Rotation settings
        self.rotate = True
        self.fp32_had = False
        
        # Per-tensor quantization
        self.per_tensor = True
        
        # QuaRot fusion settings
        self.block_size = 64  # Hadamard block size


def quantize_weight(weight, bits=4, groupsize=-1, sym=True, premultiply_scales=False):
    """
    Simple RTN (Round-to-Nearest) weight quantization.
    Returns quantized weight, input_scale, output_scale compatible with ANEMLL format.
    
    Args:
        premultiply_scales: If True, multiply scales into weight and return unit scales
    """
    if bits >= 16:
        out_features, in_features = weight.shape
        return weight, torch.ones(1, in_features), torch.ones(out_features, 1)
    
    # Calculate scales
    if groupsize == -1:
        # Per-tensor quantization - compatible with ANEMLL SP format
        w_max = weight.abs().max()
        if sym:
            # Symmetric quantization: [-2^(b-1), 2^(b-1)-1]
            qmax = 2 ** (bits - 1) - 1
            scale = w_max / qmax
        else:
            # Asymmetric quantization: [0, 2^b-1]  
            qmax = 2 ** bits - 1
            scale = (weight.max() - weight.min()) / qmax
            
        # Quantize and dequantize
        if sym:
            w_q = torch.clamp(torch.round(weight / scale), -2**(bits-1), 2**(bits-1)-1)
        else:
            w_min = weight.min()
            w_q = torch.clamp(torch.round((weight - w_min) / scale), 0, 2**bits-1)
            w_q = w_q * scale + w_min
            
            # Return ANEMLL-compatible scales for asymmetric case
            out_features, in_features = weight.shape
            input_scale = torch.ones(1, in_features)  # [1, in_features] for ANEMLL
            output_scale = torch.ones(out_features, 1) * scale  # [out_features, 1] for ANEMLL
            return w_q, input_scale, output_scale
            
        w_dq = w_q * scale
        
        # ANEMLL-compatible scale shapes for per-tensor quantization
        out_features, in_features = weight.shape
        
        # Input scale: [1, in_features] - typically ones for weight quantization
        input_scale = torch.ones(1, in_features)
        
        # Output scale: [out_features, 1] - contains the quantization scale
        output_scale = torch.ones(out_features, 1) * scale
        
        if premultiply_scales:
            # Pre-multiply scales into weight and return unit scales
            # Formula: weight_premult = output_scale * weight * input_scale
            # Since input_scale is ones, this simplifies to: output_scale * weight
            weight_premult = output_scale * w_dq
            unit_input_scale = torch.ones(1, in_features)
            unit_output_scale = torch.ones(out_features, 1)
            return weight_premult, unit_input_scale, unit_output_scale
        
        return w_dq, input_scale, output_scale
    else:
        # Per-group quantization (more complex, simplified for now)
        return quantize_weight(weight, bits, -1, sym)


def apply_quarot_rotation(model, config):
    """
    Apply true QuaRot fusion transformations.
    Includes LayerNorm fusion, Hadamard rotation, and bias fusion.
    """
    print("Applying QuaRot fusion transformations...")
    
    # Apply comprehensive QuaRot fusion
    fusion_metadata = apply_quarot_fusion_to_model(model, config)
    
    # Verify fusion quality
    quality_metrics = verify_fusion_quality(model, config)
    
    print("\nQuaRot fusion quality metrics:")
    
    # Rotation coverage metrics
    if 'rotation_coverage' in quality_metrics:
        cov = quality_metrics['rotation_coverage']
        print(f"\n  Rotation Coverage:")
        print(f"  - Total layers processed: {cov['total_layers']}")
        print(f"  - Fully rotated (≥99% coverage): {cov['fully_rotated']}")
        print(f"  - Partially rotated: {cov['partial_rotated']}")
        print(f"  - Failed rotations: {cov['failed_rotation']}")
        print(f"  - Average input coverage: {cov['avg_input_coverage']:.1%}")
        print(f"  - Average output coverage: {cov['avg_output_coverage']:.1%}")
    
    # Rotation quality metrics
    if 'rotation_quality' in quality_metrics:
        qual = quality_metrics['rotation_quality']
        print(f"\n  Rotation Quality:")
        print(f"  - Total rotations: {qual['total_rotations']}")
        print(f"  - Norm preserved: {qual['norm_preserved_count']} ({qual['norm_preserved_count']/max(1, qual['total_rotations']):.1%})")
        print(f"  - Quality improved: {qual['quality_improved_count']} ({qual['quality_improved_count']/max(1, qual['total_rotations']):.1%})")
    
    # Weight statistics
    if 'avg_weight_norm' in quality_metrics:
        print(f"\n  Weight Statistics:")
        print(f"  - Average weight norm: {quality_metrics['avg_weight_norm']:.4f}")
        print(f"  - Weight norm std: {quality_metrics['weight_norm_std']:.4f}")
    if 'avg_bias_magnitude' in quality_metrics:
        print(f"  - Average bias magnitude: {quality_metrics['avg_bias_magnitude']:.4f}")
    
    # Overall assessment
    if 'rotation_coverage' in quality_metrics:
        if cov['avg_input_coverage'] >= 0.99 and cov['avg_output_coverage'] >= 0.99:
            print("\n  ✅ EXCELLENT: Full rotation coverage achieved!")
        elif cov['avg_input_coverage'] >= 0.90 or cov['avg_output_coverage'] >= 0.90:
            print("\n  ⚠️  WARNING: Partial rotation coverage - may impact quality")
        else:
            print("\n  ❌ CRITICAL: Poor rotation coverage - quantization quality will be low!")
    
    # Store metadata in model
    model._quarot_fusion_metadata = fusion_metadata
    model._quarot_quality_metrics = quality_metrics
    
    return model


def quantize_qwen25_model(model_path, calib_data_path, output_dir, config, premultiply_scales=False):
    """Main quantization function."""
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map='cpu'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load calibration data
    print(f"Loading calibration data from {calib_data_path}...")
    with open(calib_data_path, 'r') as f:
        calib_data = json.load(f)
    
    print(f"Loaded {len(calib_data)} calibration samples")
    
    # Apply rotations if enabled
    if config.rotate:
        model = apply_quarot_rotation(model, config)
    
    # Quantize weights
    print("Quantizing model weights...")
    if premultiply_scales:
        print("Pre-multiplying scales into weights (ANEMLL unit scale format)")
    else:
        print("Keeping scales separate (ANEMLL separate scale format)")
    quantized_state_dict = {}
    scales_dict = {}
    
    total_layers = len([n for n in model.state_dict().keys() if 'weight' in n])
    processed = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            # Skip lm_head if specified
            if 'lm_head' in name:
                print(f"Skipping lm_head layer: {name}")
                quantized_state_dict[name] = param
                continue
                
            # Quantize the weight
            w_q, input_scale, output_scale = quantize_weight(
                param.data, 
                bits=config.w_bits,
                groupsize=config.w_groupsize,
                sym=not config.w_asym,
                premultiply_scales=premultiply_scales
            )
            
            # Store quantized weight and scales
            quantized_state_dict[name] = w_q
            
            # Store scales in per-tensor format compatible with ANEMLL
            base_name = name.replace('.weight', '')
            scales_dict[f"{base_name}.input_scales"] = input_scale
            scales_dict[f"{base_name}.output_scales"] = output_scale
            
            processed += 1
            if processed % 10 == 0:
                print(f"Quantized {processed}/{total_layers} weight layers")
        else:
            # Keep non-weight parameters as-is
            quantized_state_dict[name] = param
    
    # Add bias terms for quantized layers (needed for per-tensor quantization)
    for name, param in model.named_parameters():
        if 'weight' in name and 'lm_head' not in name and param.dim() >= 2:
            bias_name = name.replace('weight', 'bias')
            if bias_name not in quantized_state_dict:
                # Create zero bias
                out_features = param.shape[0]
                quantized_state_dict[bias_name] = torch.zeros(out_features, dtype=param.dtype)
    
    # Combine quantized weights and scales
    final_state_dict = {**quantized_state_dict, **scales_dict}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save quantized model
    print(f"Saving quantized model to {output_dir}...")
    
    # Save as safetensors (compatible with HuggingFace)
    model_file = os.path.join(output_dir, "model.safetensors")
    save_file(final_state_dict, model_file)
    
    # Save config
    model.config.quantization_config = {
        "quant_method": "quarot",
        "bits": config.w_bits,
        "group_size": config.w_groupsize,
        "per_tensor": config.per_tensor,
        "rotated": config.rotate
    }
    model.config.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save comprehensive quantization metadata
    quant_meta = {
        "quantization_method": "quarot",
        "weight_bits": config.w_bits,
        "per_tensor": config.per_tensor,
        "rotated": config.rotate,
        "symmetric": not config.w_asym,
        "model_type": "qwen2",
        "layers_quantized": processed,
        "block_size": config.block_size,
        "fusion_applied": True,
        "premultiplied_scales": premultiply_scales
    }
    
    # Add fusion metadata if available (excluding tensors)
    if hasattr(model, '_quarot_fusion_metadata'):
        fusion_meta = model._quarot_fusion_metadata.copy()
        # Process rotation info for JSON serialization
        for layer_info in fusion_meta.get('layer_info', {}).values():
            if 'rotation_matrices' in layer_info:
                # Convert rotation info to serializable format
                rotation_summary = {}
                for name, rot_info in layer_info['rotation_matrices'].items():
                    if isinstance(rot_info, dict):
                        # Keep the metadata, remove any tensor data
                        rotation_summary[name] = {
                            'in_coverage': rot_info.get('in_coverage', 0),
                            'out_coverage': rot_info.get('out_coverage', 0),
                            'in_features': rot_info.get('in_features', 0),
                            'out_features': rot_info.get('out_features', 0),
                            'quality_improved': rot_info.get('quality_improved', False),
                            'norm_preserved': rot_info.get('norm_preserved', False),
                            'std_ratio': rot_info.get('std_ratio', 1.0),
                            'block_sizes_used': rot_info.get('block_sizes_used', {})
                        }
                layer_info['rotation_info'] = rotation_summary
                layer_info['rotation_matrices_count'] = len(layer_info['rotation_matrices'])
                del layer_info['rotation_matrices']
        quant_meta["fusion_metadata"] = fusion_meta
    
    # Add quality metrics if available (excluding tensors)
    if hasattr(model, '_quarot_quality_metrics'):
        quality_meta = {}
        for key, value in model._quarot_quality_metrics.items():
            if isinstance(value, (int, float, str, bool, list)):
                quality_meta[key] = value
        quant_meta["quality_metrics"] = quality_meta
    
    with open(os.path.join(output_dir, "quantization_config.json"), 'w') as f:
        json.dump(quant_meta, f, indent=2)
    
    print(f"Quantization complete! Model saved to {output_dir}")
    print(f"Quantized {processed} layers with {config.w_bits}-bit precision")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='CPU-only QuaRot quantization for Qwen 2.5')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the model directory')
    parser.add_argument('--calib_data', type=str, required=True,
                        help='Path to calibration data JSON file')
    parser.add_argument('--quant_scheme', type=str, default='w_int4_per_tensor',
                        choices=['w_int4_per_tensor', 'w_int8_per_tensor'],
                        help='Quantization scheme')
    parser.add_argument('--pre_quantization_optimization', type=str, default='quarot',
                        help='Pre-quantization optimization method')
    parser.add_argument('--quarot_config', type=str, 
                        help='Path to QuaRot configuration file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for quantized model')
    parser.add_argument('--premultiply_scales', action='store_true',
                        help='Pre-multiply scales into weights (ANEMLL format with unit scales)')
    
    args = parser.parse_args()
    
    # Create quantization config
    config = QuantConfig()
    
    # Parse quantization scheme
    if 'int4' in args.quant_scheme:
        config.w_bits = 4
    elif 'int8' in args.quant_scheme:
        config.w_bits = 8
    
    if 'per_tensor' in args.quant_scheme:
        config.per_tensor = True
        config.w_groupsize = -1
    
    # Load QuaRot config if provided
    if args.quarot_config and os.path.exists(args.quarot_config):
        print(f"Loading QuaRot config from {args.quarot_config}")
        with open(args.quarot_config, 'r') as f:
            quarot_cfg = json.load(f)
        print(f"QuaRot config: {quarot_cfg}")
        
        # Apply learning parameters to quantization config
        if 'search_iters' in quarot_cfg:
            config.search_iters = quarot_cfg['search_iters']
        if 'lr' in quarot_cfg:
            config.lr = quarot_cfg['lr']
        if 'grad_clip' in quarot_cfg:
            config.grad_clip = quarot_cfg['grad_clip']
        if 'ortho_loss_weight' in quarot_cfg:
            config.ortho_loss_weight = quarot_cfg['ortho_loss_weight']
        if 'codebook_bits' in quarot_cfg:
            config.codebook_bits = quarot_cfg['codebook_bits']
        if 'override_dtype' in quarot_cfg:
            config.override_dtype = quarot_cfg['override_dtype']
        if 'online-had' in quarot_cfg:
            config.online_had = quarot_cfg['online-had']
    
    # Run quantization
    output_dir = quantize_qwen25_model(
        args.model_dir,
        args.calib_data,
        args.output_dir,
        config,
        premultiply_scales=args.premultiply_scales
    )
    
    print(f"\nQuantization completed successfully!")
    print(f"Quantized model saved to: {output_dir}")
    print(f"\nTo use with ANEMLL, set environment variable:")
    print(f"export ENABLE_SP_QUANT=true")


if __name__ == '__main__':
    main()
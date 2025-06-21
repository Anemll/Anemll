#!/usr/bin/env python3
"""
Focused test to identify why Q projection weights are completely different
between transformers and ANEMLL despite loading from the same model.
"""

import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config


def compare_raw_weights():
    """Compare raw weights loaded from the checkpoint."""
    print("="*80)
    print("WEIGHT MISMATCH ANALYSIS")
    print("="*80)
    
    model_name = "smpanaro/Qwen2.5-0.5B-4bit-PerTensor"
    model_path = snapshot_download(model_name)
    
    # Disable quantization
    os.environ['SKIP_SP_FORWARD'] = '1'
    
    # Load transformers model
    print("\n1. Loading transformers model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    hf_model.eval()
    
    # Load ANEMLL model
    print("2. Loading ANEMLL model...")
    config = Qwen25Config.from_json(f'{model_path}/config.json')
    anemll_model = Qwen25ForCausalLM(config, disable_kv_cache=False)
    anemll_model.load_pretrained_weights(model_path)
    anemll_model.eval()
    
    # Compare first layer Q projection weights
    print("\n3. Comparing Q projection weights...")
    
    # HF weights
    hf_q_weight = hf_model.model.layers[0].self_attn.q_proj.weight  # [896, 896]
    print(f"HF Q weight shape: {hf_q_weight.shape}")
    print(f"HF Q weight dtype: {hf_q_weight.dtype}")
    print(f"HF Q weight mean: {hf_q_weight.mean().item():.6f}")
    print(f"HF Q weight std: {hf_q_weight.std().item():.6f}")
    
    # ANEMLL weights (Conv2d format)
    anemll_q_weight_raw = anemll_model.model.layers[0].self_attn.q_proj.weight  # [896, 896, 1, 1]
    anemll_q_weight = anemll_q_weight_raw.squeeze(-1).squeeze(-1).t()  # Convert to [896, 896]
    print(f"ANEMLL Q weight raw shape: {anemll_q_weight_raw.shape}")
    print(f"ANEMLL Q weight converted shape: {anemll_q_weight.shape}")
    print(f"ANEMLL Q weight dtype: {anemll_q_weight.dtype}")
    print(f"ANEMLL Q weight mean: {anemll_q_weight.mean().item():.6f}")
    print(f"ANEMLL Q weight std: {anemll_q_weight.std().item():.6f}")
    
    # Compare raw checkpoint data
    print("\n4. Loading raw checkpoint data...")
    from safetensors import safe_open
    
    with safe_open(f"{model_path}/model.safetensors", framework="pt", device="cpu") as f:
        # Check the actual stored weight
        raw_q_weight = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
        print(f"Raw checkpoint Q weight shape: {raw_q_weight.shape}")
        print(f"Raw checkpoint Q weight dtype: {raw_q_weight.dtype}")
        print(f"Raw checkpoint Q weight mean: {raw_q_weight.mean().item():.6f}")
        print(f"Raw checkpoint Q weight std: {raw_q_weight.std().item():.6f}")
        
        # Show first few values
        print(f"Raw checkpoint first 5 values: {raw_q_weight.flatten()[:5].tolist()}")
        print(f"HF model first 5 values: {hf_q_weight.flatten()[:5].tolist()}")
        print(f"ANEMLL model first 5 values: {anemll_q_weight.flatten()[:5].tolist()}")
    
    # Check if they match the raw data
    print("\n5. Checking matches with raw data...")
    hf_matches_raw = torch.allclose(hf_q_weight, raw_q_weight, rtol=1e-4, atol=1e-4)
    anemll_matches_raw = torch.allclose(anemll_q_weight, raw_q_weight, rtol=1e-4, atol=1e-4)
    
    print(f"HF matches raw: {hf_matches_raw}")
    print(f"ANEMLL matches raw: {anemll_matches_raw}")
    
    if not hf_matches_raw:
        diff = torch.abs(hf_q_weight - raw_q_weight)
        max_diff = diff.max().item()
        print(f"HF vs raw max diff: {max_diff:.6f}")
        
    if not anemll_matches_raw:
        diff = torch.abs(anemll_q_weight - raw_q_weight)
        max_diff = diff.max().item()
        print(f"ANEMLL vs raw max diff: {max_diff:.6f}")
    
    # Check direct comparison
    print("\n6. Direct comparison HF vs ANEMLL...")
    direct_match = torch.allclose(hf_q_weight, anemll_q_weight, rtol=1e-4, atol=1e-4)
    print(f"HF matches ANEMLL: {direct_match}")
    
    if not direct_match:
        diff = torch.abs(hf_q_weight - anemll_q_weight)
        max_diff = diff.max().item()
        max_idx = torch.argmax(diff)
        max_pos = np.unravel_index(max_idx.item(), diff.shape)
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"At position {max_pos}:")
        print(f"  HF: {hf_q_weight[max_pos].item():.6f}")
        print(f"  ANEMLL: {anemll_q_weight[max_pos].item():.6f}")
        print(f"  Raw: {raw_q_weight[max_pos].item():.6f}")
    
    # Check for GPTQ scales affecting the weights
    print("\n7. Checking for quantization scale effects...")
    
    # Check if ANEMLL model has any scales loaded
    layer0_attn = anemll_model.model.layers[0].self_attn
    has_q_input_scale = hasattr(layer0_attn, 'q_proj_input_scale')
    has_q_output_scale = hasattr(layer0_attn, 'q_proj_output_scale')
    
    print(f"ANEMLL has q_proj_input_scale: {has_q_input_scale}")
    print(f"ANEMLL has q_proj_output_scale: {has_q_output_scale}")
    
    if has_q_input_scale:
        q_input_scale = getattr(layer0_attn, 'q_proj_input_scale')
        print(f"Q input scale shape: {q_input_scale.shape}")
        print(f"Q input scale values: {q_input_scale[:5].tolist()}")
        
    if has_q_output_scale:
        q_output_scale = getattr(layer0_attn, 'q_proj_output_scale')
        print(f"Q output scale shape: {q_output_scale.shape}")
        print(f"Q output scale values: {q_output_scale[:5].tolist()}")
    
    # Check if quantization was applied despite SKIP_SP_FORWARD
    print("\n8. Environment check...")
    print(f"SKIP_SP_FORWARD: {os.environ.get('SKIP_SP_FORWARD', 'not set')}")
    print(f"ENABLE_SP_QUANT: {os.environ.get('ENABLE_SP_QUANT', 'not set')}")


def main():
    compare_raw_weights()


if __name__ == "__main__":
    main()
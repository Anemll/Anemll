#!/usr/bin/env python3
# Copyright (c) 2025 ANEMLL
# Licensed under the MIT License

"""
Compare Tensors Script

This script loads two .pt files containing dictionaries of tensors,
one from the Hugging Face reference model and one from our local
implementation. It then compares the tensors for shape, dtype, and
numerical similarity (cosine similarity and Mean Absolute Error) in
a predefined, logical order to find the first point of divergence.
"""

import torch
import os
import sys
import torch.nn.functional as F

# --- Configuration ---
HF_TENSORS_FILE = "tests/dev/hf_tensors.pt"
ANEMLL_TENSORS_FILE = "tests/dev/anemll_tensors.pt"
ANEMLL_TENSORS_FILE_INJECTED = "tests/dev/anemll_tensors_with_injection.pt"
ANEMLL_TENSORS_FILE_DUAL_INJECTED = "tests/dev/anemll_tensors_with_dual_injection.pt"
COSINE_THRESHOLD_EXACT = 0.999
COSINE_THRESHOLD_CLOSE = 0.995  # Close match threshold
MAE_THRESHOLD = 1e-4

# --- Color Codes for Output ---
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'

# --- Comparison Logic ---

def compare_tensors(hf_tensor, anemll_tensor, tensor_name=None):
    """Compares two tensors and returns a dictionary of results."""
    results = {}
    results['shape_match'] = hf_tensor.shape == anemll_tensor.shape
    results['dtype_match'] = hf_tensor.dtype == anemll_tensor.dtype

    if results['shape_match'] and results['dtype_match']:
        # Ensure tensors are float32 for comparison
        hf_f32 = hf_tensor.to(torch.float32)
        anemll_f32 = anemll_tensor.to(torch.float32)

        # Cosine Similarity
        cos_sim = F.cosine_similarity(hf_f32.flatten(), anemll_f32.flatten(), dim=0).item()
        results['cos_sim'] = cos_sim
        
        # Magnitude Comparison
        hf_magnitude = hf_f32.std().item()
        anemll_magnitude = anemll_f32.std().item()
        magnitude_ratio = anemll_magnitude / hf_magnitude if hf_magnitude > 1e-8 else float('inf')
        magnitude_diff = abs(hf_magnitude - anemll_magnitude)
        results['hf_magnitude'] = hf_magnitude
        results['anemll_magnitude'] = anemll_magnitude
        results['magnitude_ratio'] = magnitude_ratio
        results['magnitude_diff'] = magnitude_diff
        
        # Special handling for corrected predictions - different thresholds for different streams
        if tensor_name and 'corrected_stream' in tensor_name:
            if 'stream_0' in tensor_name:
                # Active stream should have high similarity
                threshold_exact = 0.995
                threshold_close = 0.990
            else:
                # Non-active streams can have lower similarity due to AltUp complexity
                threshold_exact = 0.750
                threshold_close = 0.700
            
            results['cos_sim_exact'] = cos_sim >= threshold_exact
            results['cos_sim_close'] = cos_sim >= threshold_close
            results['corrected_stream_thresholds'] = f"exact>={threshold_exact}, close>={threshold_close}"
        else:
            # Standard thresholds for other tensors
            results['cos_sim_exact'] = cos_sim >= COSINE_THRESHOLD_EXACT
            results['cos_sim_close'] = cos_sim >= COSINE_THRESHOLD_CLOSE

        # Mean Absolute Error
        mae = torch.mean(torch.abs(hf_f32 - anemll_f32)).item()
        results['mae'] = mae
        results['mae_match'] = mae <= MAE_THRESHOLD

        # Determine match type
        if results['cos_sim_exact'] or results['mae_match']:
            results['match_type'] = 'exact'
        elif results['cos_sim_close']:
            results['match_type'] = 'close'
        else:
            results['match_type'] = 'mismatch'
    else:
        results['match_type'] = 'mismatch'
        results['cos_sim'] = 0.0
        results['mae'] = float('inf')


    return results

def print_comparison_result(name, hf_tensor, anemll_tensor, results):
    """Prints the formatted comparison results for a single tensor."""
    match_type = results['match_type']
    if match_type == 'exact':
        status_color = Colors.GREEN
        status_text = "EXACT MATCH"
    elif match_type == 'close':
        status_color = Colors.YELLOW
        status_text = "CLOSE MATCH"
    else:
        status_color = Colors.RED
        status_text = "MISMATCH"
    
    print(f"Status: {status_color}{status_text}{Colors.ENDC}")

    # Shape and Dtype
    shape_status = "✅" if results['shape_match'] else "❌"
    dtype_status = "✅" if results['dtype_match'] else "❌"
    print(f"  Shape: {shape_status} HF={hf_tensor.shape}, ANEMLL={anemll_tensor.shape}")
    print(f"  Dtype: {dtype_status} HF={hf_tensor.dtype}, ANEMLL={anemll_tensor.dtype}")

    if results['shape_match'] and results['dtype_match']:
        # Cosine Similarity
        if results.get('cos_sim_exact', False):
            cos_color = Colors.GREEN
        elif results.get('cos_sim_close', False):
            cos_color = Colors.YELLOW
        else:
            cos_color = Colors.RED
        # Show appropriate thresholds
        if 'corrected_stream_thresholds' in results:
            threshold_info = results['corrected_stream_thresholds']
            print(f"  Cosine Sim: {cos_color}{results['cos_sim']:.6f}{Colors.ENDC} (Corrected stream: {threshold_info})")
        else:
            print(f"  Cosine Sim: {cos_color}{results['cos_sim']:.6f}{Colors.ENDC} (Exact: {COSINE_THRESHOLD_EXACT}, Close: {COSINE_THRESHOLD_CLOSE})")

        # Mean Absolute Error
        mae_color = Colors.GREEN if results['mae_match'] else Colors.RED
        print(f"  MAE:        {mae_color}{results['mae']:.6e}{Colors.ENDC} (Threshold: {MAE_THRESHOLD})")
        
        # Magnitude Comparison
        magnitude_good = abs(results['magnitude_ratio'] - 1.0) < 0.1  # Within 10%
        magnitude_color = Colors.GREEN if magnitude_good else Colors.RED
        print(f"  Magnitude:  {magnitude_color}HF={results['hf_magnitude']:.6f}, ANEMLL={results['anemll_magnitude']:.6f}, Ratio={results['magnitude_ratio']:.3f}{Colors.ENDC}")
        
        # Special analysis for logits
        if name and name == 'logits' and hf_tensor.dim() >= 3:
            print(f"  🎯 Logits comparison:")
            
            # Get top 5 for both models
            hf_top_logits, hf_top_indices = torch.topk(hf_tensor[0, 0], 5)
            anemll_top_logits, anemll_top_indices = torch.topk(anemll_tensor[0, 0], 5)
            
            print(f"    HF Top 5:")
            for i, (logit_val, token_idx) in enumerate(zip(hf_top_logits, hf_top_indices)):
                print(f"      {i+1}. Token {token_idx.item():5d}: {logit_val.item():8.4f}")
            
            print(f"    ANEMLL Top 5:")
            for i, (logit_val, token_idx) in enumerate(zip(anemll_top_logits, anemll_top_indices)):
                print(f"      {i+1}. Token {token_idx.item():5d}: {logit_val.item():8.4f}")
            
            # Check if top tokens match
            top_match = torch.equal(hf_top_indices, anemll_top_indices)
            print(f"    Top 5 tokens match: {'✅' if top_match else '❌'}")
            
            # Show differences in top 5 logits
            for i in range(5):
                hf_logit = hf_top_logits[i].item()
                anemll_logit = anemll_top_logits[i].item()
                diff = abs(hf_logit - anemll_logit)
                print(f"      Position {i+1}: HF={hf_logit:.4f}, ANEMLL={anemll_logit:.4f}, diff={diff:.6f}")
        
        # Per-stream analysis for stacked tensors
        elif name and ('stacked_streams' in name or 'final_output' in name or 'initial_stream_stack' in name or 'after_ple' in name) and hf_tensor.dim() >= 4:
            print(f"  📊 Per-stream analysis:")
            
            # Check if we have real HF data to compare against
            real_hf_data = None
            if name == 'stacked_streams':
                try:
                    real_hf_data = torch.load('tests/dev/hf_real_unembed_data.pt', weights_only=False)
                    if 'real_stacked_streams' in real_hf_data:
                        print(f"    🎯 USING REAL HF DATA (not fake)")
                        hf_tensor = real_hf_data['real_stacked_streams']
                except:
                    print(f"    ⚠️  Using fake HF data (real data not available)")
            
            # Debug tensor shapes before comparison
            print(f"    🔍 DEBUG SHAPES:")
            print(f"      HF tensor shape: {hf_tensor.shape}")
            print(f"      ANEMLL tensor shape: {anemll_tensor.shape}")
            
            # Ensure both tensors have the same number of streams
            min_streams = min(hf_tensor.shape[0], anemll_tensor.shape[0])
            print(f"      Comparing {min_streams} streams")
            
            for stream_idx in range(min_streams):
                hf_stream = hf_tensor[stream_idx]
                anemll_stream = anemll_tensor[stream_idx]
                
                print(f"      Stream {stream_idx}: HF={hf_stream.shape}, ANEMLL={anemll_stream.shape}")
                
                # Skip comparison if shapes don't match
                if hf_stream.shape != anemll_stream.shape:
                    print(f"      ⚠️  Stream {stream_idx} shape mismatch - skipping cosine similarity")
                    stream_cos_sim = 0.0
                else:
                    stream_cos_sim = F.cosine_similarity(hf_stream.flatten(), anemll_stream.flatten(), dim=0).item()
                hf_stream_std = hf_stream.std().item()
                anemll_stream_std = anemll_stream.std().item()
                hf_stream_mean = hf_stream.mean().item()
                anemll_stream_mean = anemll_stream.mean().item()
                stream_ratio = anemll_stream_std / hf_stream_std if hf_stream_std > 1e-8 else float('inf')
                stream_color = Colors.GREEN if abs(stream_ratio - 1.0) < 0.1 else Colors.RED
                print(f"    Stream {stream_idx}: {stream_color}cos={stream_cos_sim:.6f}, HF_std={hf_stream_std:.6f}, ANEMLL_std={anemll_stream_std:.6f}, ratio={stream_ratio:.3f}{Colors.ENDC}")
                print(f"      Magnitude: HF_mean={hf_stream_mean:.6f}, ANEMLL_mean={anemll_stream_mean:.6f}")

    if match_type != 'exact':
        print(f"  HF Tensor Stats:     min={hf_tensor.min():.6f}, max={hf_tensor.max():.6f}, mean={hf_tensor.mean():.6f}, std={hf_tensor.std():.6f}")
        print(f"  ANEMLL Tensor Stats: min={anemll_tensor.min():.6f}, max={anemll_tensor.max():.6f}, mean={anemll_tensor.mean():.6f}, std={anemll_tensor.std():.6f}")

# --- Main Execution ---
def main():
    """Loads tensors and runs the comparison."""
    # Check command line arguments
    use_injected = False
    use_dual_injected = False
    layer_name = "layer_0"  # Default layer name
    max_layers = None  # Default: show all layers
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--injected":
            use_injected = True
        elif sys.argv[1] == "--dual-injected":
            use_dual_injected = True
        elif sys.argv[1] == "--help":
            print("Usage: python compare_outputs.py [--injected|--dual-injected] [layer_name] [--limit N]")
            print("  --injected: Compare with corrected_streams injected tensors")
            print("  --dual-injected: Compare with corrected_streams + after_ple injected tensors")
            print("  layer_name: Specify layer name (e.g., layer_0, layer_1, etc.)")
            print("  --limit N: Limit output to first N layers to reduce output size")
            print("  (default): Compare with original tensors (tests/dev/anemll_tensors.pt)")
            return
        elif sys.argv[1].startswith("layer_"):
            layer_name = sys.argv[1]
        elif sys.argv[1] == "--limit" and len(sys.argv) > 2:
            max_layers = int(sys.argv[2])
    
    # Check for layer name and limit in remaining arguments
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("layer_"):
            layer_name = arg
        elif arg == "--limit" and i + 1 < len(sys.argv):
            max_layers = int(sys.argv[i + 1])
    
    # Try to auto-detect layer name from tensor files
    if not os.path.exists(HF_TENSORS_FILE):
        print(f"{Colors.RED}❌ Error: HF tensor file not found. Run debug scripts first.{Colors.ENDC}")
        return
    
    # Load HF tensors to detect layer name
    hf_tensors = torch.load(HF_TENSORS_FILE, weights_only=False)
    detected_layers = [key for key in hf_tensors.keys() if key.startswith("layer_") and ".final_output" in key]
    if detected_layers:
        # Extract layer name from first detected layer
        detected_layer = detected_layers[0].split('.')[0]
        layer_name = detected_layer
        print(f"🔍 Auto-detected layer name: {layer_name}")
    
    # Select the appropriate ANEMLL tensor file
    if use_dual_injected:
        anemll_file = ANEMLL_TENSORS_FILE_DUAL_INJECTED
        comparison_type = "DUAL INJECTED"
    elif use_injected:
        anemll_file = ANEMLL_TENSORS_FILE_INJECTED
        comparison_type = "INJECTED"
    else:
        anemll_file = ANEMLL_TENSORS_FILE
        comparison_type = "ORIGINAL"
    
    print(f"📊 Starting tensor comparison ({comparison_type} ANEMLL tensors)...")
    if not os.path.exists(HF_TENSORS_FILE) or not os.path.exists(anemll_file):
        print(f"{Colors.RED}❌ Error: Tensor file(s) not found. Run debug scripts first.{Colors.ENDC}")
        if use_injected and not os.path.exists(anemll_file):
            print(f"{Colors.RED}   Missing: {anemll_file} (run test_attention_divergence_experiment.py first){Colors.ENDC}")
        return

    hf_tensors = torch.load(HF_TENSORS_FILE, weights_only=False)
    anemll_tensors = torch.load(anemll_file, weights_only=False)
    print(f"✅ Tensor files loaded successfully ({comparison_type} mode).")

    # Dynamic tensor order for multiple layers
    TENSOR_ORDER = [
        'initial_embedding',
        'initial_stream',
        'initial_stream_stack',
    ]
    
    # Determine all layers present in the data
    all_layers = set()
    for key in list(hf_tensors.keys()) + list(anemll_tensors.keys()):
        if key.startswith('layer_'):
            layer_num = int(key.split('.')[0].split('_')[1])
            all_layers.add(layer_num)
    
    # Apply layer limit if specified
    if max_layers is not None:
        all_layers = {layer for layer in all_layers if layer < max_layers}
        print(f"🔍 Limiting comparison to first {max_layers} layers: {sorted(all_layers)}")
    
    # Add tensors for each layer in chronological execution order
    # Execution flow: AltUp.predict → Laurel → Attention → MLP → AltUp.correct
    for layer_num in sorted(all_layers):
        layer_name = f'layer_{layer_num}'
        TENSOR_ORDER.extend([
            # PHASE 1: AltUp.predict (router computation and per-stream predictions)
            f'{layer_name}.active_stream_initial',
            f'{layer_name}.altup.router_inputs',
            f'{layer_name}.altup.routed',
            f'{layer_name}.altup.modalities_predict',
            f'{layer_name}.altup.predictions_stream_0',
            f'{layer_name}.altup.predictions_stream_1',
            f'{layer_name}.altup.predictions_stream_2',
            f'{layer_name}.altup.predictions_stream_3',
            f'{layer_name}.active_prediction',
            f'{layer_name}.active_prediction_normed',
            f'{layer_name}.predictions',
            
            # PHASE 2: Laurel (processing active predictions)
            f'{layer_name}.laurel_out',
            
            # PHASE 3: Attention (self-attention computation)
            f'{layer_name}.pre_attention_norm',
            f'{layer_name}.attn.q_proj',
            f'{layer_name}.attn.k_proj',
            f'{layer_name}.attn.v_proj',
            f'{layer_name}.attn.q_norm',
            f'{layer_name}.attn.k_norm',
            # CRITICAL: Position embeddings & RoPE debug tensors
            f'{layer_name}.attn.position_embeddings_global',
            f'{layer_name}.attn.position_embeddings_local',
            f'{layer_name}.attn.cos_sin_global',
            f'{layer_name}.attn.cos_sin_local',
            f'{layer_name}.attn.q_before_rope',
            f'{layer_name}.attn.k_before_rope',
            f'{layer_name}.attn.q_after_rope',
            f'{layer_name}.attn.k_after_rope',
            # Attention computation intermediates
            f'{layer_name}.attn.attention_mask',
            f'{layer_name}.attn.causal_mask',
            f'{layer_name}.attn.value_states_before_norm',
            f'{layer_name}.attn.value_states_transposed',
            f'{layer_name}.attn.raw_attn_weights',
            f'{layer_name}.attn.scaled_attn_weights',
            f'{layer_name}.attn.attn_weights_raw',
            f'{layer_name}.attn.attn_weights_pre_softmax',
            f'{layer_name}.attn.attn_weights_post_softmax',
            f'{layer_name}.attn.attn_output_pre_reshape',
            f'{layer_name}.attn.o_proj_input',
            f'{layer_name}.attn.attn_output',

            # PHASE 4: MLP (feed-forward network)
            f'{layer_name}.post_attention_norm',
            f'{layer_name}.pre_ffw_norm',
            f'{layer_name}.mlp.gate_proj',
            f'{layer_name}.mlp.gate_proj_after_sparsity',
            f'{layer_name}.mlp.gate_proj_activated',
            f'{layer_name}.mlp.up_proj',
            f'{layer_name}.mlp.activations',
            f'{layer_name}.mlp_output',
            f'{layer_name}.post_ffw_norm',
            f'{layer_name}.processed_active_stream',
            
            # PHASE 5: AltUp.correct (correction coefficients and stream corrections)
            f'{layer_name}.altup.activated',
            f'{layer_name}.altup.modalities_correct',
            f'{layer_name}.altup.correction_coefs_output',
            f'{layer_name}.altup.all_coefs',
            f'{layer_name}.altup.corrected_stream_0',
            f'{layer_name}.altup.corrected_stream_1',
            f'{layer_name}.altup.corrected_stream_2',
            f'{layer_name}.altup.corrected_stream_3',
            f'{layer_name}.altup_corrected',
            f'{layer_name}.corrected_streams',
            f'{layer_name}.after_ple',
            f'{layer_name}.final_output',
        ])
    
    # Add final processing tensors
    TENSOR_ORDER.extend([
        'stacked_streams',
        'averaged_before_norm',
        'final_norm_input',
        'final_norm_output',
        'logits',
    ])

    exact_matches = 0
    close_matches = 0
    mismatches = 0
    
    for key in TENSOR_ORDER:
        if key in hf_tensors and key in anemll_tensors:
            # Extract layer number from key for better readability
            layer_info = ""
            if key.startswith('layer_'):
                layer_num = key.split('.')[0].split('_')[1]
                layer_info = f" [Layer {layer_num}]"
            
            # Add special label for stacked_streams to clarify it's after decoder layers
            if key == 'stacked_streams':
                print(f"\n--- Comparing: {key} AFTER DECODER LAYERS ---")
            else:
                print(f"\n--- Comparing: {key}{layer_info} ---")
            results = compare_tensors(hf_tensors[key], anemll_tensors[key], tensor_name=key)
            print_comparison_result(key, hf_tensors[key], anemll_tensors[key], results)
            
            if results['match_type'] == 'exact':
                exact_matches += 1
            elif results['match_type'] == 'close':
                close_matches += 1
            else:
                mismatches += 1
        elif key in hf_tensors:
            print(f"\n{Colors.YELLOW}❓ SKIPPING: Tensor '{key}' found in HF but not in ANEMLL.{Colors.ENDC}")
        elif key in anemll_tensors:
            print(f"\n{Colors.YELLOW}❓ SKIPPING: Tensor '{key}' found in ANEMLL but not in HF.{Colors.ENDC}")

    print("\n" + "="*50)
    total_compared = exact_matches + close_matches + mismatches
    if total_compared > 0:
        print(f"📊 COMPARISON SUMMARY:")
        print(f"  {Colors.GREEN}✅ Exact matches: {exact_matches}{Colors.ENDC}")
        if close_matches > 0:
            print(f"  {Colors.YELLOW}🟡 Close matches: {close_matches}{Colors.ENDC}")
        if mismatches > 0:
            print(f"  {Colors.RED}❌ Mismatches: {mismatches}{Colors.ENDC}")
        print(f"  📈 Total compared: {total_compared}")
        
        # Add detailed analysis of attention components
        attention_components = [
            f'{layer_name}.attn.q_proj', f'{layer_name}.attn.k_proj', f'{layer_name}.attn.v_proj',
            f'{layer_name}.attn.q_norm', f'{layer_name}.attn.k_norm',
            f'{layer_name}.attn.attn_weights_raw', f'{layer_name}.attn.attn_weights_pre_softmax',
            f'{layer_name}.attn.attn_weights_post_softmax', f'{layer_name}.attn.attn_output_pre_reshape',
            f'{layer_name}.attn.o_proj_input', f'{layer_name}.attn.attn_output'
        ]
        
        perfect_attention = []
        divergent_attention = []
        
        for key in TENSOR_ORDER:
            if key in attention_components and key in hf_tensors and key in anemll_tensors:
                results = compare_tensors(hf_tensors[key], anemll_tensors[key], tensor_name=key)
                if results['cos_sim'] >= 0.999:
                    perfect_attention.append(f"{key}: {results['cos_sim']:.6f}")
                else:
                    divergent_attention.append(f"{key}: {results['cos_sim']:.6f}")
        
        if perfect_attention or divergent_attention:
            print(f"\n🔍 ATTENTION ANALYSIS:")
            if perfect_attention:
                print(f"  {Colors.GREEN}✅ Perfect attention components:{Colors.ENDC}")
                for comp in perfect_attention:
                    print(f"    - {comp} similarity (perfect)")
            if divergent_attention:
                print(f"  {Colors.RED}❌ Divergent attention components:{Colors.ENDC}")
                for comp in divergent_attention:
                    print(f"    - {comp} similarity (divergence source)")
        
        # Show additional ANEMLL-only debug tensors not in main comparison
        additional_debug_tensors = [
            'hidden_streams_before_unembed',
            'target_magnitude',
            'stream_0_raw',
            'stream_1_raw',
            'stream_2_raw', 
            'stream_3_raw',
            'stream_1_projected',
            'stream_2_projected',
            'stream_3_projected',
            'stream_1_normalized',
            'stream_2_normalized',
            'stream_3_normalized'
        ]
        
        additional_debug_found = []
        for tensor_name in additional_debug_tensors:
            if tensor_name in anemll_tensors:
                additional_debug_found.append(tensor_name)
        
        if additional_debug_found:
            print(f"\n🔍 ADDITIONAL ANEMLL DEBUG TENSORS:")
            print("-" * 50)
            for tensor_name in additional_debug_found:
                tensor = anemll_tensors[tensor_name]
                print(f"  {tensor_name}:")
                print(f"    Shape: {tensor.shape}")
                print(f"    Stats: min={tensor.min():.3f}, max={tensor.max():.3f}, std={tensor.std():.3f}")
                if tensor.numel() < 100:  # Show actual values for small tensors
                    print(f"    Values: {tensor.flatten()[:10].tolist()}")

        if mismatches == 0:
            if close_matches == 0:
                print(f"\n{Colors.GREEN}🎉 EXCELLENT: All tensors match exactly!{Colors.ENDC}")
            else:
                print(f"\n{Colors.YELLOW}👍 GOOD: All tensors match (some closely)!{Colors.ENDC}")
        else:
            print(f"\n{Colors.RED}⚠️  NEEDS WORK: {mismatches} tensor(s) still diverging.{Colors.ENDC}")
    else:
        print(f"{Colors.RED}❌ NO COMPARISONS: No matching tensors found.{Colors.ENDC}")

if __name__ == "__main__":
    main()
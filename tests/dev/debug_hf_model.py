#!/usr/bin/env python3
# Copyright (c) 2025 ANEMLL
# Licensed under the MIT License

"""
X Debug script for the Hugging Face Gemma3n model.

This script loads the reference Gemma3n model from Hugging Face,
runs a single-token input through it, and uses forward hooks to
capture intermediate activation tensors at various points. The
captured tensors are saved to a file for comparison against our
local implementation.
"""

import torch
import torch.nn.functional as F
import os
import math
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

# --- Configuration ---
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots/0330734afc91972a1aa1ba1bc4495e2723666854/"
)
OUTPUT_FILE = "tests/dev/hf_tensors.pt"
TEST_PROMPT = "What"
TEST_PROMPT = "What is the capital of France? Answer:"

# --- Main Execution ---
def capture_hf_tensors(prompt=TEST_PROMPT, num_layers=1):
    """
    Loads the HF model, registers hooks, runs inference, and saves tensors.
    """
    print(f"🔬 Capturing Hugging Face Gemma3n reference tensors...")
    print(f"   Prompt: '{prompt}'")
    print(f"   Processing layers: {num_layers}")

    # Ensure model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model path not found: {MODEL_PATH}")
        print("   Please ensure you have the correct model downloaded.")
        return

    # --- Model and Tokenizer Loading ---
    print(f"   tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    print(f"   model from: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="cpu",
        low_cpu_mem_usage=True
    ).eval()
    
    # Set debug layer limit
    model.model.language_model._debug_layer_limit = num_layers
    print(f"✅ Model and tokenizer loaded. Debug layer limit set to {num_layers}.")

    # --- Hook Setup ---
    captured_tensors = {}
    hooks = []

    def get_activation(name, output_index=None):
        def hook(module, input, output):
            tensor_to_save = output[output_index] if output_index is not None and isinstance(output, tuple) else output
            captured_tensors[name] = tensor_to_save.detach().cpu().to(torch.float32)
            print(f"  [HOOK] Captured '{name}': shape={captured_tensors[name].shape}, dtype={captured_tensors[name].dtype}")
        return hook

    def get_input_activation(name):
        def hook(module, input, output):
            # Input is a tuple, we want the first element
            tensor_to_save = input[0]
            captured_tensors[name] = tensor_to_save.detach().cpu().to(torch.float32)
            print(f"  [HOOK] Captured '{name}' (input): shape={captured_tensors[name].shape}, dtype={captured_tensors[name].dtype}")
        return hook

    # Capture tensors from all layers being processed
    layers_to_capture = list(range(num_layers))
    
    print(f"   🔍 DEBUG: Will capture tensors from layers {layers_to_capture}")
    
    # Define the exact, ordered list of tensors we need
    tensor_keys = ['initial_embedding']
    
    # Add tensors for each layer
    for layer_idx in layers_to_capture:
        layer_name = f"layer_{layer_idx}"
        tensor_keys.extend([
            f'{layer_name}.pre_attention_norm',
            f'{layer_name}.attn.q_proj', f'{layer_name}.attn.k_proj', f'{layer_name}.attn.v_proj',
            f'{layer_name}.attn.q_norm', f'{layer_name}.attn.k_norm',
            # CRITICAL: Position embeddings & RoPE debug hooks
            f'{layer_name}.attn.position_embeddings_global',
            f'{layer_name}.attn.position_embeddings_local',
            f'{layer_name}.attn.cos_sin_global',
            f'{layer_name}.attn.cos_sin_local',
            f'{layer_name}.attn.q_before_rope',
            f'{layer_name}.attn.k_before_rope',
            f'{layer_name}.attn.q_after_rope',
            f'{layer_name}.attn.k_after_rope',
            # Attention computation steps  
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
            f'{layer_name}.attn.attn_output',
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
            f'{layer_name}.altup_corrected',
            f'{layer_name}.corrected_streams',
            f'{layer_name}.after_ple',
            f'{layer_name}.final_output',
        ])
    
    # Add final tensors
    tensor_keys.append('final_norm_output')
    
    module_map = {
        'initial_embedding': model.model.language_model.embed_tokens,
        'initial_stream': 'CUSTOM_HOOK',  # Will capture inputs_embeds after embed_scale
        'initial_stream_stack': 'CUSTOM_HOOK',  # Will capture the 4-stream equivalent
        'final_norm_output': model.model.language_model.norm,
    }
    
    # Add module mappings for all layers
    for layer_idx in layers_to_capture:
        layer_name = f"layer_{layer_idx}"
        layer_module = model.model.language_model.layers[layer_idx]
        
        module_map.update({
            f'{layer_name}.pre_attention_norm': layer_module.input_layernorm,
            f'{layer_name}.attn.q_proj': layer_module.self_attn.q_proj,
            f'{layer_name}.attn.k_proj': layer_module.self_attn.k_proj,
            f'{layer_name}.attn.v_proj': layer_module.self_attn.v_proj,
            f'{layer_name}.attn.q_norm': layer_module.self_attn.q_norm,
            f'{layer_name}.attn.k_norm': layer_module.self_attn.k_norm,
            # CRITICAL: Position embeddings & RoPE debug hooks (need custom handling)
            f'{layer_name}.attn.position_embeddings_global': 'CUSTOM_HOOK',
            f'{layer_name}.attn.position_embeddings_local': 'CUSTOM_HOOK',
            f'{layer_name}.attn.cos_sin_global': 'CUSTOM_HOOK',
            f'{layer_name}.attn.cos_sin_local': 'CUSTOM_HOOK',
            f'{layer_name}.attn.q_before_rope': 'CUSTOM_HOOK',
            f'{layer_name}.attn.k_before_rope': 'CUSTOM_HOOK',
            f'{layer_name}.attn.q_after_rope': 'CUSTOM_HOOK',
            f'{layer_name}.attn.k_after_rope': 'CUSTOM_HOOK',
            # Attention computation steps (need custom handling)
            f'{layer_name}.attn.attention_mask': 'CUSTOM_HOOK',
            f'{layer_name}.attn.causal_mask': 'CUSTOM_HOOK',
            f'{layer_name}.attn.value_states_before_norm': 'CUSTOM_HOOK',
            f'{layer_name}.attn.value_states_transposed': 'CUSTOM_HOOK',
            f'{layer_name}.attn.raw_attn_weights': 'CUSTOM_HOOK',
            f'{layer_name}.attn.scaled_attn_weights': 'CUSTOM_HOOK',
            f'{layer_name}.attn.attn_weights_raw': 'CUSTOM_HOOK',
            f'{layer_name}.attn.attn_weights_pre_softmax': 'CUSTOM_HOOK', 
            f'{layer_name}.attn.attn_weights_post_softmax': 'CUSTOM_HOOK',
            f'{layer_name}.attn.attn_output_pre_reshape': 'CUSTOM_HOOK',
            f'{layer_name}.attn.o_proj_input': (layer_module.self_attn.o_proj, 'input'),
            f'{layer_name}.attn.attn_output': (layer_module.self_attn, 0),
            f'{layer_name}.post_attention_norm': layer_module.post_attention_layernorm,
            f'{layer_name}.pre_ffw_norm': layer_module.pre_feedforward_layernorm,
            f'{layer_name}.mlp.gate_proj': layer_module.mlp.gate_proj,
            f'{layer_name}.mlp.gate_proj_after_sparsity': 'CUSTOM_HOOK',  # Will need custom handling
            f'{layer_name}.mlp.gate_proj_activated': 'CUSTOM_HOOK',      # Will need custom handling 
            f'{layer_name}.mlp.up_proj': layer_module.mlp.up_proj,
            f'{layer_name}.mlp.activations': (layer_module.mlp.down_proj, 'input'),
            f'{layer_name}.mlp_output': layer_module.mlp,
            f'{layer_name}.post_ffw_norm': layer_module.post_feedforward_layernorm,
            f'{layer_name}.processed_active_stream': 'CUSTOM_HOOK',  # Need custom handling
            f'{layer_name}.altup_corrected': 'CUSTOM_HOOK',          # Need custom handling
            f'{layer_name}.corrected_streams': 'CUSTOM_HOOK',        # Need custom handling  
            f'{layer_name}.after_ple': 'CUSTOM_HOOK',                # Need custom handling
            f'{layer_name}.final_output': (layer_module, 0),  # Layer output
        })

    # Special handling for custom hooks that need to be monkey-patched
    custom_hook_names = [name for name in tensor_keys if module_map.get(name) == 'CUSTOM_HOOK']
    original_mlp_forward = None
    original_layer_forward = None
    original_attention_forward = None
    
    if custom_hook_names:
        # Set up layer indices for all layers we want to capture
        original_mlp_forwards = {}
        
        # Add layer_idx attribute to each layer's MLP for identification
        for layer_idx in layers_to_capture:
            model.model.language_model.layers[layer_idx].mlp._layer_idx = layer_idx
        
        # Monkey-patch ALL layers' MLP forward methods to capture intermediate values
        original_mlp_forward = model.model.language_model.layers[0].mlp.__class__.forward
        
        # Monkey-patch the HF attention forward method to capture attention computation steps
        # Use first layer as reference since all layers share the same class
        original_attention_forward = model.model.language_model.layers[0].self_attn.__class__.forward
        
        # Also monkey-patch the layer forward to capture layer-level tensors
        original_layer_forward = model.model.language_model.layers[0].__class__.forward
        
        # Also monkey-patch the language_model forward to capture initial streams
        original_language_model_forward = model.model.language_model.__class__.forward
        
        
        def patched_mlp_forward(self, hidden_states):
            # Follow the exact HF implementation with debug captures
            gate_proj = self.gate_proj(hidden_states)
            
            # Get layer_idx for this MLP
            layer_idx = getattr(self, '_layer_idx', None)
            
            # Add detailed sparsity debugging
            if self.activation_sparsity > 0.0:
                print(f"🔍 HF SPARSITY DEBUG Layer {layer_idx}:")
                print(f"  activation_sparsity: {self.activation_sparsity}")
                print(f"  gate_proj before sparsity: min={gate_proj.min():.6f}, max={gate_proj.max():.6f}, mean={gate_proj.mean():.6f}, std={gate_proj.std():.6f}")
                gate_proj_after_sparsity = self._gaussian_topk(gate_proj)
                sparsity_level = (gate_proj_after_sparsity == 0).float().mean().item()
                print(f"  gate_proj after sparsity: min={gate_proj_after_sparsity.min():.6f}, max={gate_proj_after_sparsity.max():.6f}, mean={gate_proj_after_sparsity.mean():.6f}, std={gate_proj_after_sparsity.std():.6f}")
                print(f"  actual sparsity level: {sparsity_level:.4f}")
            else:
                gate_proj_after_sparsity = gate_proj
                print(f"🔍 HF SPARSITY DEBUG Layer {layer_idx}: NO SPARSITY (activation_sparsity={self.activation_sparsity})")
            
            # Capture from the correct layer based on layer_idx
            if layer_idx is not None and layer_idx in layers_to_capture:
                current_layer_name = f"layer_{layer_idx}"
                
                # Capture sparsity tensor for this specific layer
                if f'{current_layer_name}.mlp.gate_proj_after_sparsity' in custom_hook_names:
                    if f'{current_layer_name}.mlp.gate_proj_after_sparsity' not in captured_tensors:
                        captured_tensors[f'{current_layer_name}.mlp.gate_proj_after_sparsity'] = gate_proj_after_sparsity.detach().cpu()
                        print(f"[HF DEBUG] Captured {current_layer_name}.mlp.gate_proj_after_sparsity: min={gate_proj_after_sparsity.min():.6f}, max={gate_proj_after_sparsity.max():.6f}")
                
                # Capture activated tensor for this specific layer
                activations = self.act_fn(gate_proj_after_sparsity)
                if f'{current_layer_name}.mlp.gate_proj_activated' in custom_hook_names:
                    if f'{current_layer_name}.mlp.gate_proj_activated' not in captured_tensors:
                        captured_tensors[f'{current_layer_name}.mlp.gate_proj_activated'] = activations.detach().cpu()
                        print(f"[HF DEBUG] Captured {current_layer_name}.mlp.gate_proj_activated: min={activations.min():.6f}, max={activations.max():.6f}")
            else:
                activations = self.act_fn(gate_proj_after_sparsity)
                
            up_proj = self.up_proj(hidden_states)
            down_proj = self.down_proj(activations * up_proj)
            return down_proj
            
        def patched_attention_forward(self, hidden_states, position_embeddings, **kwargs):
            # Enhanced approach: capture attention computation intermediates including RoPE and position embeddings
            # Only capture from target layers (check if this is a target layer)
            layer_idx = getattr(self, '_layer_idx', None)
            if layer_idx is None or layer_idx not in layers_to_capture:
                return original_attention_forward(self, hidden_states, position_embeddings, **kwargs)
            
            current_layer_name = f"layer_{layer_idx}"
            
            # Store original tensor operations for restoration
            original_matmul = torch.matmul
            original_softmax = torch.nn.functional.softmax
            original_transpose = torch.transpose
            
            # Counter to track which operations we're in
            matmul_counter = [0]
            
            # CRITICAL: Capture position embeddings first
            if isinstance(position_embeddings, tuple) and len(position_embeddings) == 2:
                cos_global, sin_global = position_embeddings
                if f'{current_layer_name}.attn.cos_sin_global' not in captured_tensors:
                    captured_tensors[f'{current_layer_name}.attn.cos_sin_global'] = torch.stack([cos_global, sin_global]).detach().cpu()
                    print(f"[HF DEBUG] Captured {current_layer_name}.attn.cos_sin_global: shape={captured_tensors[f'{current_layer_name}.attn.cos_sin_global'].shape}")
                
                # 🔍 PRINT DEBUG: Position embeddings
                if layer_idx == 0:
                    print(f"\n🔍 HF ATTENTION DEBUG Layer {layer_idx}:")
                    print(f"  Using GLOBAL position embeddings")
                    print(f"  cos shape={cos_global.shape}, std={cos_global.std():.6f}, mean={cos_global.mean():.6f}")
                    print(f"  sin shape={sin_global.shape}, std={sin_global.std():.6f}, mean={sin_global.mean():.6f}")
                    # Check position_ids from kwargs
                    position_ids = kwargs.get('position_ids', None)
                    print(f"  Position IDs: {position_ids}")
                    if position_ids is not None:
                        print(f"  Position IDs shape: {position_ids.shape}")
                    print(f"  🔍 CRITICAL: Layer should be sliding_attention but using GLOBAL embeddings!")
                    
            elif position_embeddings is not None:
                # Single position embedding (could be local)
                if f'{current_layer_name}.attn.position_embeddings_local' not in captured_tensors:
                    captured_tensors[f'{current_layer_name}.attn.position_embeddings_local'] = position_embeddings.detach().cpu()
                    print(f"[HF DEBUG] Captured {current_layer_name}.attn.position_embeddings_local: shape={position_embeddings.shape}")
                
                # 🔍 PRINT DEBUG: Position embeddings  
                if layer_idx == 0:
                    print(f"\n🔍 HF ATTENTION DEBUG Layer {layer_idx}:")
                    print(f"  Using LOCAL position embeddings (sliding attention)")
                    if hasattr(position_embeddings, 'shape'):
                        print(f"  position_embeddings shape={position_embeddings.shape}, std={position_embeddings.std():.6f}, mean={position_embeddings.mean():.6f}")
                    # Check position_ids from kwargs
                    position_ids = kwargs.get('position_ids', None)
                    print(f"  Position IDs: {position_ids}")
                    if position_ids is not None:
                        print(f"  Position IDs shape: {position_ids.shape}")
            
            def debug_matmul(input_tensor, other_tensor):
                result = original_matmul(input_tensor, other_tensor)
                matmul_counter[0] += 1
                
                # First matmul should be Q @ K^T (attention weights)
                if matmul_counter[0] == 1 and result.dim() == 4:
                    if f'{current_layer_name}.attn.raw_attn_weights' not in captured_tensors:
                        captured_tensors[f'{current_layer_name}.attn.raw_attn_weights'] = result.detach().cpu()
                        print(f"[HF DEBUG] Captured {current_layer_name}.attn.raw_attn_weights: shape={result.shape}")
                    
                    # 🔍 ENHANCED DEBUG: Q@K^T computation details
                    if layer_idx == 0:
                        print(f"\n🔍 HF ATTENTION COMPUTATION:")
                        print(f"  Q@K^T raw result: std={result.std():.6f}, mean={result.mean():.6f}")
                        print(f"  Q input shape: {input_tensor.shape}, std={input_tensor.std():.6f}, mean={input_tensor.mean():.6f}")
                        print(f"  K^T input shape: {other_tensor.shape}, std={other_tensor.std():.6f}, mean={other_tensor.mean():.6f}")
                
                # Second matmul should be attention @ values (pre-reshape output)
                elif matmul_counter[0] == 2 and result.dim() == 4:
                    if f'{current_layer_name}.attn.attn_output_pre_reshape' not in captured_tensors:
                        captured_tensors[f'{current_layer_name}.attn.attn_output_pre_reshape'] = result.detach().cpu()
                        print(f"[HF DEBUG] Captured {current_layer_name}.attn.attn_output_pre_reshape: shape={result.shape}")
                    
                    # 🔍 ENHANCED DEBUG: Attention@Values computation
                    if layer_idx == 0:
                        print(f"\n🔍 HF ATTENTION@VALUES:")
                        print(f"  Attn@V result: std={result.std():.6f}, mean={result.mean():.6f}")
                        print(f"  Attention weights shape: {input_tensor.shape}, std={input_tensor.std():.6f}, mean={input_tensor.mean():.6f}")
                        print(f"  Values shape: {other_tensor.shape}, std={other_tensor.std():.6f}, mean={other_tensor.mean():.6f}")
                        print(f"  🎯 HF Value amplification: {result.std().item() / other_tensor.std().item():.3f}")
                
                return result
            
            def debug_softmax(input_tensor, dim=-1, **kwargs):
                # Capture pre-softmax weights
                if input_tensor.dim() == 4 and f'{current_layer_name}.attn.attn_weights_pre_softmax' not in captured_tensors:
                    captured_tensors[f'{current_layer_name}.attn.attn_weights_pre_softmax'] = input_tensor.detach().cpu()
                    print(f"[HF DEBUG] Captured {current_layer_name}.attn.attn_weights_pre_softmax: shape={input_tensor.shape}")
                    
                    # 🔍 ENHANCED DEBUG: Pre-softmax analysis
                    if layer_idx == 0:
                        print(f"\n🔍 HF PRE-SOFTMAX:")
                        print(f"  Pre-softmax weights: std={input_tensor.std():.6f}, mean={input_tensor.mean():.6f}")
                        print(f"  First 10 pre-softmax values: {input_tensor.flatten()[:10]}")
                        # Check for inf values
                        has_inf = torch.isinf(input_tensor).any()
                        print(f"  Contains -inf values: {has_inf}")
                
                result = original_softmax(input_tensor, dim=dim, **kwargs)
                
                # Capture post-softmax weights
                if result.dim() == 4 and f'{current_layer_name}.attn.attn_weights_post_softmax' not in captured_tensors:
                    captured_tensors[f'{current_layer_name}.attn.attn_weights_post_softmax'] = result.detach().cpu()
                    print(f"[HF DEBUG] Captured {current_layer_name}.attn.attn_weights_post_softmax: shape={result.shape}")
                    
                    # 🔍 ENHANCED DEBUG: Post-softmax analysis
                    if layer_idx == 0:
                        print(f"\n🔍 HF POST-SOFTMAX:")
                        print(f"  Post-softmax weights: std={result.std():.6f}, mean={result.mean():.6f}")
                        print(f"  First 10 post-softmax values: {result.flatten()[:10]}")
                        print(f"  Attention sum (should be ~1.0): {result.sum(dim=-1).mean():.6f}")
                
                return result
            
            def debug_transpose(input_tensor, dim0, dim1):
                result = original_transpose(input_tensor, dim0, dim1)
                
                # Capture transposed values if they look like value states
                if (result.dim() == 4 and result.shape[-1] == 256 and 
                    f'{current_layer_name}.attn.value_states_transposed' not in captured_tensors):
                    captured_tensors[f'{current_layer_name}.attn.value_states_transposed'] = result.detach().cpu()
                    print(f"[HF DEBUG] Captured {current_layer_name}.attn.value_states_transposed: shape={result.shape}")
                
                return result
            
            # Apply patches
            torch.matmul = debug_matmul
            torch.nn.functional.softmax = debug_softmax
            torch.transpose = debug_transpose
            
            # 🔍 PRINT DEBUG: Input to attention 
            if layer_idx == 0:
                print(f"  Hidden states input: shape={hidden_states.shape}, std={hidden_states.std():.6f}, mean={hidden_states.mean():.6f}")
            
            # Store original linear forward for Q/K/V projection debugging
            original_linear_forward = torch.nn.Linear.forward
            projection_counter = [0]
            
            def debug_linear_forward(self_linear, input_tensor):
                result = original_linear_forward(self_linear, input_tensor)
                projection_counter[0] += 1
                
                # Based on Gemma3n architecture: q_proj, k_proj, v_proj, o_proj in that order
                if layer_idx == 0:
                    if projection_counter[0] == 1:  # q_proj
                        print(f"  Q projection: input_shape={input_tensor.shape}, output_shape={result.shape}")
                        print(f"    Q before RoPE: std={result.std():.6f}, mean={result.mean():.6f}")
                    elif projection_counter[0] == 2:  # k_proj  
                        print(f"  K projection: input_shape={input_tensor.shape}, output_shape={result.shape}")
                        print(f"    K before RoPE: std={result.std():.6f}, mean={result.mean():.6f}")
                    elif projection_counter[0] == 3:  # v_proj
                        print(f"  V projection: input_shape={input_tensor.shape}, output_shape={result.shape}")
                        print(f"    V states: std={result.std():.6f}, mean={result.mean():.6f}")
                
                return result
            
            torch.nn.Linear.forward = debug_linear_forward
            
            try:
                # Call original attention forward - this should trigger the debug captures
                result = original_attention_forward(self, hidden_states, position_embeddings, **kwargs)
                
                # 🔍 PRINT DEBUG: Final attention output
                if layer_idx == 0:
                    # result might be a tuple (attention_output, attention_weights)
                    attn_output = result[0] if isinstance(result, tuple) else result
                    print(f"  Final attention output: shape={attn_output.shape}, std={attn_output.std():.6f}, mean={attn_output.mean():.6f}")
                
                return result
            finally:
                # Restore original functions
                torch.matmul = original_matmul
                torch.nn.functional.softmax = original_softmax
                torch.transpose = original_transpose
                torch.nn.Linear.forward = original_linear_forward
        
        # Also monkey-patch AltUp methods to capture router modalities for all layers
        original_methods = {}
        for layer_idx in layers_to_capture:
            layer = model.model.language_model.layers[layer_idx]
            original_methods[layer_idx] = {
                'compute_router_modalities': layer.altup.compute_router_modalities,
                'predict': layer.altup.predict,
                'correct': layer.altup.correct
            }
        
        def create_patched_compute_router_modalities(layer_idx):
            layer_name = f"layer_{layer_idx}"
            original_fn = original_methods[layer_idx]['compute_router_modalities']
            
            def patched_compute_router_modalities(x):
                # Capture router inputs
                router_inputs = model.model.language_model.layers[layer_idx].altup.router_norm(x) * model.model.language_model.layers[layer_idx].altup.router_input_scale
                if f'{layer_name}.altup.router_inputs' not in captured_tensors:
                    captured_tensors[f'{layer_name}.altup.router_inputs'] = router_inputs.detach().cpu()
                    print(f"[HF DEBUG] Captured {layer_name}.altup.router_inputs: shape={router_inputs.shape}")
                
                # Capture routed output
                routed = model.model.language_model.layers[layer_idx].altup.modality_router(router_inputs)
                if f'{layer_name}.altup.routed' not in captured_tensors:
                    captured_tensors[f'{layer_name}.altup.routed'] = routed.detach().cpu()
                    print(f"[HF DEBUG] Captured {layer_name}.altup.routed: shape={routed.shape}")
                
                # Use original function to compute modalities
                modalities = original_fn(x)
                return modalities
            
            return patched_compute_router_modalities
        
        def create_patched_altup_predict(layer_idx):
            layer_name = f"layer_{layer_idx}"
            original_fn = original_methods[layer_idx]['predict']
            
            def patched_altup_predict(hidden_states):
                # Get modalities for predict
                patched_compute_router_modalities = model.model.language_model.layers[layer_idx].altup.compute_router_modalities
                modalities = patched_compute_router_modalities(hidden_states[model.model.language_model.layers[layer_idx].altup.config.altup_active_idx])
                if f'{layer_name}.altup.modalities_predict' not in captured_tensors:
                    captured_tensors[f'{layer_name}.altup.modalities_predict'] = modalities.detach().cpu()
                    print(f"[HF DEBUG] Captured {layer_name}.altup.modalities_predict: shape={modalities.shape}")
                
                # Continue with original prediction logic
                result = original_fn(hidden_states)
                return result
            
            return patched_altup_predict
        
        def create_patched_altup_correct(layer_idx):
            layer_name = f"layer_{layer_idx}"
            original_fn = original_methods[layer_idx]['correct']
            
            def patched_altup_correct(predictions, activated):
                # Capture per-stream predictions and activated tensor
                for i in range(4):
                    stream_key = f'{layer_name}.altup.predictions_stream_{i}'
                    if stream_key not in captured_tensors:
                        captured_tensors[stream_key] = predictions[i].detach().cpu()
                        print(f"[HF DEBUG] Captured {stream_key}: shape={predictions[i].shape}")
                
                if f'{layer_name}.altup.activated' not in captured_tensors:
                    captured_tensors[f'{layer_name}.altup.activated'] = activated.detach().cpu()
                    print(f"[HF DEBUG] Captured {layer_name}.altup.activated: shape={activated.shape}")
                
                # Get modalities for correct
                patched_compute_router_modalities = model.model.language_model.layers[layer_idx].altup.compute_router_modalities
                modalities = patched_compute_router_modalities(activated)
                if f'{layer_name}.altup.modalities_correct' not in captured_tensors:
                    captured_tensors[f'{layer_name}.altup.modalities_correct'] = modalities.detach().cpu()
                    print(f"[HF DEBUG] Captured {layer_name}.altup.modalities_correct: shape={modalities.shape}")
                
                # Continue with original correct logic but capture intermediate values
                # First get the correction coefficients
                correction_coefs_output = model.model.language_model.layers[layer_idx].altup.correction_coefs(modalities)
                if f'{layer_name}.altup.correction_coefs_output' not in captured_tensors:
                    captured_tensors[f'{layer_name}.altup.correction_coefs_output'] = correction_coefs_output.detach().cpu()
                    print(f"[HF DEBUG] Captured {layer_name}.altup.correction_coefs_output: shape={correction_coefs_output.shape}")
                
                all_coefs = correction_coefs_output + 1.0
                if f'{layer_name}.altup.all_coefs' not in captured_tensors:
                    captured_tensors[f'{layer_name}.altup.all_coefs'] = all_coefs.detach().cpu()
                    print(f"[HF DEBUG] Captured {layer_name}.altup.all_coefs: shape={all_coefs.shape}")
                
                # 🔍 DEBUG: Add detailed debugging for AltUp correction logic
                if layer_idx == 1:  # Only debug layer 1 to avoid spam
                    print(f"🔍 HF AltUp.correct() DEBUG for {layer_name}:")
                    print(f"  Modalities: {modalities.flatten()[:4].tolist()}")
                    
                    # Debug innovation computation
                    innovation = activated - predictions[model.model.language_model.layers[layer_idx].altup.config.altup_active_idx]
                    print(f"  Innovation shape before repeat: {innovation.shape}")
                    print(f"  Innovation stats: min={innovation.min():.6f}, max={innovation.max():.6f}, mean={innovation.mean():.6f}, std={innovation.std():.6f}")
                    
                    innovation_repeated = innovation.repeat(model.model.language_model.layers[layer_idx].altup.config.altup_num_inputs, 1, 1, 1)
                    print(f"  Innovation shape after repeat: {innovation_repeated.shape}")
                    print(f"  Innovation after repeat stats: min={innovation_repeated.min():.6f}, max={innovation_repeated.max():.6f}, mean={innovation_repeated.mean():.6f}, std={innovation_repeated.std():.6f}")
                    
                    # Debug coefficient computation
                    print(f"  Correction coefs output shape: {correction_coefs_output.shape}")
                    print(f"  Correction coefs output: {correction_coefs_output.flatten()[:4].tolist()}")
                    print(f"  All coefs before permute: {all_coefs.flatten()[:4].tolist()}")
                    
                    all_coefs_permuted = all_coefs.permute(2, 0, 1).unsqueeze(-1)
                    print(f"  All coefs after permute+unsqueeze shape: {all_coefs_permuted.shape}")
                    print(f"  All coefs after permute+unsqueeze: min={all_coefs_permuted.min():.6f}, max={all_coefs_permuted.max():.6f}, mean={all_coefs_permuted.mean():.6f}, std={all_coefs_permuted.std():.6f}")
                    
                    print(f"  Before mul: innovation shape={innovation_repeated.shape}, all_coefs shape={all_coefs_permuted.shape}")
                    corrected_intermediate = torch.mul(innovation_repeated, all_coefs_permuted)
                    print(f"  After mul: corrected shape={corrected_intermediate.shape}, min={corrected_intermediate.min():.6f}, max={corrected_intermediate.max():.6f}, mean={corrected_intermediate.mean():.6f}, std={corrected_intermediate.std():.6f}")
                    
                    corrected_final = corrected_intermediate + predictions
                    print(f"  After add predictions: corrected shape={corrected_final.shape}, min={corrected_final.min():.6f}, max={corrected_final.max():.6f}, mean={corrected_final.mean():.6f}, std={corrected_final.std():.6f}")
                
                result = original_fn(predictions, activated)
                
                # 🔍 DEBUG: Print actual HF altup.correct() return values
                if layer_idx == 1:
                    print(f"🔍 HF ACTUAL altup.correct() return values (Layer {layer_idx}):")
                    print(f"  result shape: {result.shape}")
                    print(f"  result[1]: min={result[1].min():.6f}, max={result[1].max():.6f}, mean={result[1].mean():.6f}, std={result[1].std():.6f}")
                    print(f"  result[2]: min={result[2].min():.6f}, max={result[2].max():.6f}, mean={result[2].mean():.6f}, std={result[2].std():.6f}")
                    print(f"  result[3]: min={result[3].min():.6f}, max={result[3].max():.6f}, mean={result[3].mean():.6f}, std={result[3].std():.6f}")
                
                # Capture per-stream corrected predictions
                for i in range(4):
                    corrected_key = f'{layer_name}.altup.corrected_stream_{i}'
                    if corrected_key not in captured_tensors:
                        captured_tensors[corrected_key] = result[i].detach().cpu()
                        print(f"[HF DEBUG] Captured {corrected_key}: shape={result[i].shape}")
                
                return result
            
            return patched_altup_correct
        
        # Apply AltUp patches for all layers
        for layer_idx in layers_to_capture:
            layer = model.model.language_model.layers[layer_idx]
            layer.altup.compute_router_modalities = create_patched_compute_router_modalities(layer_idx)
            layer.altup.predict = create_patched_altup_predict(layer_idx)
            layer.altup.correct = create_patched_altup_correct(layer_idx)
        
        def patched_layer_forward(
            self,
            hidden_states,
            position_embeddings_global,
            position_embeddings_local,
            per_layer_input,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            **kwargs,
        ):
            # Call original forward method and capture intermediate tensors
            # Dynamically determine current layer index
            current_layer_idx = None
            for layer_idx in layers_to_capture:
                if model.model.language_model.layers[layer_idx] is self:
                    current_layer_idx = layer_idx
                    break
            
            if current_layer_idx is None:
                # If we can't find the layer, fall back to original behavior
                return original_layer_forward(self, hidden_states, position_embeddings_global, position_embeddings_local, per_layer_input, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, **kwargs)
            
            current_layer_name = f"layer_{current_layer_idx}"
            
            # Capture active_stream_initial (the initial stream selected for processing)
            active_stream_initial = hidden_states[self.config.altup_active_idx]
            if f'{current_layer_name}.active_stream_initial' not in captured_tensors:
                captured_tensors[f'{current_layer_name}.active_stream_initial'] = active_stream_initial.detach().cpu()
                print(f"[HF DEBUG] Captured {current_layer_name}.active_stream_initial: shape={active_stream_initial.shape}")
            
            predictions = self.altup.predict(hidden_states)
            
            # Capture predictions (input to AltUp.correct)
            if f'{current_layer_name}.predictions' not in captured_tensors:
                captured_tensors[f'{current_layer_name}.predictions'] = predictions.detach().cpu()
                print(f"[HF DEBUG] Captured {current_layer_name}.predictions: shape={predictions.shape}")
                
            active_prediction = predictions[self.config.altup_active_idx]
            
            # Capture active_prediction
            if f'{current_layer_name}.active_prediction' not in captured_tensors:
                captured_tensors[f'{current_layer_name}.active_prediction'] = active_prediction.detach().cpu()
                print(f"[HF DEBUG] Captured {current_layer_name}.active_prediction: shape={active_prediction.shape}")
            
            active_prediction_normed = self.input_layernorm(active_prediction)
            
            # Capture active_prediction_normed
            if f'{current_layer_name}.active_prediction_normed' not in captured_tensors:
                captured_tensors[f'{current_layer_name}.active_prediction_normed'] = active_prediction_normed.detach().cpu()
                print(f"[HF DEBUG] Captured {current_layer_name}.active_prediction_normed: shape={active_prediction_normed.shape}")
            
            laurel_output = self.laurel(active_prediction_normed)
            
            # Capture Laurel output
            if f'{current_layer_name}.laurel_out' not in captured_tensors:
                captured_tensors[f'{current_layer_name}.laurel_out'] = laurel_output.detach().cpu()
                print(f"[HF DEBUG] Captured {current_layer_name}.laurel_out: shape={laurel_output.shape}")
            
            if self.self_attn.is_sliding:
                position_embeddings = position_embeddings_local
            else:
                position_embeddings = position_embeddings_global
                
            attn, self_attn_weights = self.self_attn(
                hidden_states=active_prediction_normed,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            attn = self.post_attention_layernorm(attn)
            attn_gated = active_prediction + attn
            attn_laurel = (attn_gated + laurel_output) / math.sqrt(2)
            attn_norm = self.pre_feedforward_layernorm(attn_laurel)
            attn_ffw = self.mlp(attn_norm)
            attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
            attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
            
            # Capture equivalent to processed_active_stream
            if f'{current_layer_name}.processed_active_stream' in custom_hook_names:
                if f'{current_layer_name}.processed_active_stream' not in captured_tensors:
                    captured_tensors[f'{current_layer_name}.processed_active_stream'] = attn_ffw_laurel_gated.detach().cpu()
                    print(f"[HF DEBUG] Captured {current_layer_name}.processed_active_stream: shape={attn_ffw_laurel_gated.shape}")
            
            corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)
            
            # 🔍 DEBUG: Print HF corrected_predictions after altup.correct()
            if layer_idx is not None and layer_idx == 1:
                print(f"🔍 HF CORRECTED_PREDICTIONS AFTER altup.correct() (Layer {layer_idx}):")
                print(f"  corrected_predictions shape: {corrected_predictions.shape}")
                for i in range(4):
                    stream = corrected_predictions[i]
                    print(f"  Stream {i}: min={stream.min():.6f}, max={stream.max():.6f}, mean={stream.mean():.6f}, std={stream.std():.6f}")
            
            # Capture AltUp corrected (before PLE)
            if f'{current_layer_name}.altup_corrected' in custom_hook_names:
                if f'{current_layer_name}.altup_corrected' not in captured_tensors:
                    captured_tensors[f'{current_layer_name}.altup_corrected'] = corrected_predictions.detach().cpu()
                    print(f"[HF DEBUG] Captured {current_layer_name}.altup_corrected: shape={corrected_predictions.shape}")
            
            # Capture equivalent to corrected_streams (legacy)
            if f'{current_layer_name}.corrected_streams' in custom_hook_names:
                if f'{current_layer_name}.corrected_streams' not in captured_tensors:
                    captured_tensors[f'{current_layer_name}.corrected_streams'] = corrected_predictions.detach().cpu()
                    print(f"[HF DEBUG] Captured {current_layer_name}.corrected_streams: shape={corrected_predictions.shape}")
            
            # Continue with PLE processing
            first_prediction = corrected_predictions[self.config.altup_active_idx].clone()
            
            # 🔍 DEBUG: Print HF PLE processing steps
            if layer_idx is not None and layer_idx == 1:
                print(f"🔍 HF PLE PROCESSING (Layer {layer_idx}):")
                print(f"  corrected_predictions[0] before PLE: min={corrected_predictions[0].min():.6f}, max={corrected_predictions[0].max():.6f}, mean={corrected_predictions[0].mean():.6f}, std={corrected_predictions[0].std():.6f}")
                print(f"  corrected_predictions[1] before PLE: min={corrected_predictions[1].min():.6f}, max={corrected_predictions[1].max():.6f}, mean={corrected_predictions[1].mean():.6f}, std={corrected_predictions[1].std():.6f}")
                print(f"  first_prediction initial: min={first_prediction.min():.6f}, max={first_prediction.max():.6f}, mean={first_prediction.mean():.6f}, std={first_prediction.std():.6f}")
            
            if self.config.altup_correct_scale:
                first_prediction = self.altup.scale_corrected_output(first_prediction)
                if layer_idx is not None and layer_idx == 1:
                    print(f"  first_prediction after scale: min={first_prediction.min():.6f}, max={first_prediction.max():.6f}, mean={first_prediction.mean():.6f}, std={first_prediction.std():.6f}")
            
            first_prediction = self.per_layer_input_gate(first_prediction)
            if layer_idx is not None and layer_idx == 1:
                print(f"  first_prediction after gate: min={first_prediction.min():.6f}, max={first_prediction.max():.6f}, mean={first_prediction.mean():.6f}, std={first_prediction.std():.6f}")
            
            first_prediction = self.act_fn(first_prediction)
            if layer_idx is not None and layer_idx == 1:
                print(f"  first_prediction after act_fn: min={first_prediction.min():.6f}, max={first_prediction.max():.6f}, mean={first_prediction.mean():.6f}, std={first_prediction.std():.6f}")
            
            first_prediction = torch.multiply(first_prediction, per_layer_input)
            if layer_idx is not None and layer_idx == 1:
                print(f"  first_prediction after multiply: min={first_prediction.min():.6f}, max={first_prediction.max():.6f}, mean={first_prediction.mean():.6f}, std={first_prediction.std():.6f}")
            
            first_prediction = self.per_layer_projection(first_prediction)
            if layer_idx is not None and layer_idx == 1:
                print(f"  first_prediction after projection: min={first_prediction.min():.6f}, max={first_prediction.max():.6f}, mean={first_prediction.mean():.6f}, std={first_prediction.std():.6f}")
            
            first_prediction = self.post_per_layer_input_norm(first_prediction)
            if layer_idx is not None and layer_idx == 1:
                print(f"  first_prediction after norm: min={first_prediction.min():.6f}, max={first_prediction.max():.6f}, mean={first_prediction.mean():.6f}, std={first_prediction.std():.6f}")
            
            corrected_predictions[1:] += first_prediction
            
            if layer_idx is not None and layer_idx == 1:
                print(f"  corrected_predictions[1] after PLE: min={corrected_predictions[1].min():.6f}, max={corrected_predictions[1].max():.6f}, mean={corrected_predictions[1].mean():.6f}, std={corrected_predictions[1].std():.6f}")
                print(f"  corrected_predictions[2] after PLE: min={corrected_predictions[2].min():.6f}, max={corrected_predictions[2].max():.6f}, mean={corrected_predictions[2].mean():.6f}, std={corrected_predictions[2].std():.6f}")
                print(f"  corrected_predictions[3] after PLE: min={corrected_predictions[3].min():.6f}, max={corrected_predictions[3].max():.6f}, mean={corrected_predictions[3].std():.6f}, std={corrected_predictions[3].std():.6f}")
            
            # Capture after PLE processing
            if f'{current_layer_name}.after_ple' in custom_hook_names:
                if f'{current_layer_name}.after_ple' not in captured_tensors:
                    captured_tensors[f'{current_layer_name}.after_ple'] = corrected_predictions.detach().cpu()
                    print(f"[HF DEBUG] Captured {current_layer_name}.after_ple: shape={corrected_predictions.shape}")
            
            # Capture final output (after PLE)
            if f'{current_layer_name}.final_output' in custom_hook_names:
                if f'{current_layer_name}.final_output' not in captured_tensors:
                    captured_tensors[f'{current_layer_name}.final_output'] = corrected_predictions.detach().cpu()
                    print(f"[HF DEBUG] Captured {current_layer_name}.final_output: shape={corrected_predictions.shape}")
            
            outputs = (corrected_predictions,)
            if output_attentions:
                outputs += (self_attn_weights,)
            return outputs
            
        def patched_language_model_forward(self, input_ids=None, inputs_embeds=None, **kwargs):
            # Capture initial streams - handle both input_ids and inputs_embeds cases
            if input_ids is not None:
                inputs_embeds = self.embed_tokens(input_ids)  # This has embed_scale already applied
                print(f"[HF DEBUG] Got inputs_embeds from input_ids: shape={inputs_embeds.shape}")
            elif inputs_embeds is not None:
                print(f"[HF DEBUG] Got inputs_embeds directly: shape={inputs_embeds.shape}")
            
            # Capture initial_stream regardless of source
            if inputs_embeds is not None:
                if 'initial_stream' not in captured_tensors:
                    captured_tensors['initial_stream'] = inputs_embeds.detach().cpu()
                    print(f"[HF DEBUG] Captured initial_stream: shape={inputs_embeds.shape}")
            
            # SAFE MONKEY PATCH: Capture real hidden_states after torch.stack
            # Store original torch.stack to restore later
            original_torch_stack = torch.stack
            
            def capture_stack_result(*args, **kwargs):
                result = original_torch_stack(*args, **kwargs)
                
                # Capture TWO stacked streams: initial (after embeddings) and final (after decoder layers)
                if len(args) > 0 and isinstance(args[0], list) and len(args[0]) == 4 and result.dim() == 4 and result.shape[0] == 4:
                    
                    # Case 1: Initial stream stack (after embeddings)
                    if ('initial_stream_stack' not in captured_tensors and 
                        inputs_embeds is not None and 
                        torch.allclose(result[0], inputs_embeds, rtol=1e-5)):
                        captured_tensors['initial_stream_stack'] = result.detach().cpu()
                        print(f"[HF DEBUG] Captured REAL initial_stream_stack: shape={result.shape}")
                        print(f"[HF DEBUG] Location: After embeddings")
                    
                    # Case 2: Final stacked streams - SAFEGUARD: Only capture from layer 0 processing
                    elif ('stacked_streams' not in captured_tensors and 
                          'initial_stream_stack' in captured_tensors):  # Only after initial is captured
                        
                        # CRITICAL SAFEGUARD: Since we only debug layer 0, ensure we only capture layer 0 processing
                        # HF runs all 30 layers, but we need the tensor that corresponds to layer 0 output
                        print(f"[HF DEBUG] Found potential stacked_streams: shape={result.shape}")
                        
                        # NEW SAFEGUARD: Check if this is from the final processing pipeline
                        # According to HF code lines 1717-1729, Stream 0 should be unmodified
                        # while streams 1-3 should be processed through altup_unembed_projections
                        
                        # First, let's identify if this is the final processing vs layer-only processing
                        # The final processing happens after ALL layers, so we need to distinguish
                        
                        # TEMPORARY: Capture the tensor to examine the pattern
                        captured_tensors['stacked_streams'] = result.detach().cpu()
                        print(f"[HF DEBUG] ✅ Captured stacked_streams candidate: shape={result.shape}")
                        print(f"[HF DEBUG] Location: After torch.stack() call")
                        
                        # Debug: Show per-stream statistics
                        for i in range(4):
                            stream = result[i]
                            print(f"[HF DEBUG]   Stream {i}: mean={stream.mean():.6f}, std={stream.std():.6f}")
                        
                        # Check if Stream 0 matches layer_0.final_output for validation
                        if 'layer_0.final_output' in captured_tensors:
                            layer_output = captured_tensors['layer_0.final_output']
                            stream_0_matches = torch.allclose(result[0], layer_output[0], rtol=1e-4)
                            print(f"[HF DEBUG] Stream 0 matches layer_0.final_output: {stream_0_matches}")
                            if not stream_0_matches:
                                print(f"[HF DEBUG] Layer 0 output: mean={layer_output[0].mean():.6f}, std={layer_output[0].std():.6f}")
                                print(f"[HF DEBUG] Current result: mean={result[0].mean():.6f}, std={result[0].std():.6f}")
                        else:
                            print(f"[HF DEBUG] layer_0.final_output not yet captured for comparison")
                
                return result
            
            # Temporarily replace torch.stack
            torch.stack = capture_stack_result
            
            try:
                # Call original forward
                result = original_language_model_forward(self, input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
                return result
            finally:
                # Always restore original torch.stack
                torch.stack = original_torch_stack

        # Apply the monkey patches to all layers
        for layer_idx in layers_to_capture:
            model.model.language_model.layers[layer_idx].mlp.__class__.forward = patched_mlp_forward
            model.model.language_model.layers[layer_idx].self_attn.__class__.forward = patched_attention_forward
            model.model.language_model.layers[layer_idx].__class__.forward = patched_layer_forward
            # Set layer index for attention layer
            model.model.language_model.layers[layer_idx].self_attn._layer_idx = layer_idx
        
        model.model.language_model.__class__.forward = patched_language_model_forward

    for name in tensor_keys:
        if name in module_map and module_map[name] != 'CUSTOM_HOOK':
            target = module_map[name]
            if isinstance(target, tuple):
                module, hook_type = target
                if hook_type == 'input':
                    hooks.append(module.register_forward_hook(get_input_activation(name)))
                else: # It's an output index
                    hooks.append(module.register_forward_hook(get_activation(name, output_index=hook_type)))
            else:
                module = target
                hooks.append(module.register_forward_hook(get_activation(name)))
            
    regular_hooks = len(hooks)
    custom_hooks = len(custom_hook_names)
    print(f"✅ Registered {regular_hooks} hooks + {custom_hooks} custom hooks.")

    # --- Inference ---
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    print(f"\n🧪 Running inference with prompt: '{prompt}' (Input IDs: {input_ids})")
    with torch.no_grad():
        model_output = model(input_ids)
    print("✅ Inference complete.")
    
    # Store logits for comparison
    logits = model_output.logits
    captured_tensors['logits'] = logits.detach().cpu()
    
    # --- CRITICAL CHECK: Verify initial_stream and initial_stream_stack were captured ---
    print("\n🔍 VERIFYING INITIAL STREAM CAPTURE:")
    if 'initial_stream' in captured_tensors:
        print(f"✅ initial_stream captured: shape={captured_tensors['initial_stream'].shape}")
    else:
        print("❌ initial_stream NOT captured - monkey patch failed!")
    
    if 'initial_stream_stack' in captured_tensors:
        print(f"✅ initial_stream_stack captured: shape={captured_tensors['initial_stream_stack'].shape}")
    else:
        print("❌ initial_stream_stack NOT captured - monkey patch failed!")
    
    # Stop here if critical tensors weren't captured
    if 'initial_stream' not in captured_tensors or 'initial_stream_stack' not in captured_tensors:
        print("\n🚨 CRITICAL ERROR: Initial stream tensors not captured!")
        print("🚨 Cannot proceed with comparison until monkey patch is fixed!")
        return

    # --- Cleanup and Save ---
    for hook in hooks:
        hook.remove()
    
    # Restore original methods if they were patched (for all layers)
    if original_mlp_forward is not None:
        for layer_idx in layers_to_capture:
            model.model.language_model.layers[layer_idx].mlp.__class__.forward = original_mlp_forward
    if original_attention_forward is not None:
        for layer_idx in layers_to_capture:
            model.model.language_model.layers[layer_idx].self_attn.__class__.forward = original_attention_forward
    if original_layer_forward is not None:
        for layer_idx in layers_to_capture:
            model.model.language_model.layers[layer_idx].__class__.forward = original_layer_forward
    if 'original_language_model_forward' in locals():
        model.model.language_model.__class__.forward = original_language_model_forward
    # Restore AltUp methods for all layers
    if 'original_methods' in locals():
        for layer_idx in layers_to_capture:
            layer = model.model.language_model.layers[layer_idx]
            layer.altup.compute_router_modalities = original_methods[layer_idx]['compute_router_modalities']
            layer.altup.predict = original_methods[layer_idx]['predict']
            layer.altup.correct = original_methods[layer_idx]['correct']
    
    print("✅ Hooks removed.")

    # Define last layer variables for weight capture
    last_layer_idx = max(layers_to_capture)
    last_layer_name = f'layer_{last_layer_idx}'

    # --- Capture Weights ---
    print("...Capturing specified weights...")
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        weight_name = f'{last_layer_name}.attn.{proj_name}_weight'
        weight_tensor = getattr(model.model.language_model.layers[last_layer_idx].self_attn, proj_name).weight.detach().cpu()
        captured_tensors[weight_name] = weight_tensor
        print(f"  [WEIGHT] Captured '{weight_name}': shape={weight_tensor.shape}")
    print("...Weight capture complete.")
    
    # Add debug tensors for comparison with ANEMLL
    print("...Adding debug tensors for comparison...")
    if 'final_norm_output' in captured_tensors:
        final_norm_output = captured_tensors['final_norm_output']
        print(f"🔍 HF ACTUAL final_norm_output: min={final_norm_output.min():.6f}, max={final_norm_output.max():.6f}, mean={final_norm_output.mean():.6f}, std={final_norm_output.std():.6f}")
        
        # Add proper debug tensors that represent HF processing
        # averaged_before_norm should be the mean of stacked_streams
        if 'stacked_streams' in captured_tensors:
            # HF averages the 4 streams before final normalization
            stacked_streams = captured_tensors['stacked_streams']
            print(f"🔍 HF ACTUAL stacked_streams: min={stacked_streams.min():.6f}, max={stacked_streams.max():.6f}, mean={stacked_streams.mean():.6f}, std={stacked_streams.std():.6f}")
            
            averaged_before_norm = stacked_streams.mean(dim=0)  # Average over stream dimension
            print(f"🔍 HF ACTUAL averaged_before_norm: min={averaged_before_norm.min():.6f}, max={averaged_before_norm.max():.6f}, mean={averaged_before_norm.mean():.6f}, std={averaged_before_norm.std():.6f}")
            
            captured_tensors['averaged_before_norm'] = averaged_before_norm.clone()
            captured_tensors['final_norm_input'] = averaged_before_norm.clone()  # Same as averaged in HF
            print(f"🔍 HF ACTUAL final_norm_input: min={averaged_before_norm.min():.6f}, max={averaged_before_norm.max():.6f}, mean={averaged_before_norm.mean():.6f}, std={averaged_before_norm.std():.6f}")
        else:
            # Fallback to final_norm_output if stacked_streams not available
            captured_tensors['final_norm_input'] = final_norm_output.clone()
            captured_tensors['averaged_before_norm'] = final_norm_output.clone()
            print(f"  [DEBUG] ⚠️  Using final_norm_output as fallback for averaged_before_norm")
        
        # NOTE: stacked_streams is now captured as REAL tensor from line 1729 (after DECODER LAYERS/ALTUP/PLE)
        # No need to create fake stacked_streams here
        
        print(f"  [DEBUG] Added final_norm_input: {captured_tensors['final_norm_input'].shape}")
        print(f"  [DEBUG] Added averaged_before_norm: {captured_tensors['averaged_before_norm'].shape}")
        
        # CRITICAL FIX: If stacked_streams wasn't captured, use final layer output as equivalent
        if 'stacked_streams' not in captured_tensors:
            if f'{last_layer_name}.final_output' in captured_tensors:
                captured_tensors['stacked_streams'] = captured_tensors[f'{last_layer_name}.final_output'].clone()
                print(f"  [DEBUG] ✅ Added stacked_streams from {last_layer_name}.final_output: {captured_tensors[f'{last_layer_name}.final_output'].shape}")
                print(f"  [DEBUG] Context: {last_layer_name} stacked streams (4-stream equivalent)")
            else:
                print(f"  [DEBUG] ❌ Neither stacked_streams nor {last_layer_name}.final_output captured")
        else:
            print(f"  [DEBUG] ✅ stacked_streams already captured: {captured_tensors['stacked_streams'].shape}")
        
        # 🔍 COMPREHENSIVE DEBUG ANALYSIS 🔍
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE HF DEBUG ANALYSIS 🔍")
        print("="*80)
        
        # Layer output analysis
        if f'{last_layer_name}.final_output' in captured_tensors:
            layer_final = captured_tensors[f'{last_layer_name}.final_output']
            print(f"\n📋 {last_layer_name.upper()} FINAL OUTPUT:")
            print(f"  Shape: {layer_final.shape}")
            if layer_final.dim() >= 4 and layer_final.shape[0] == 4:
                print(f"  📊 Per-stream magnitude analysis:")
                for i in range(4):
                    stream = layer_final[i]
                    print(f"    HF Stream {i}: mean={stream.mean():.6f}, std={stream.std():.6f}, min={stream.min():.6f}, max={stream.max():.6f}")
            else:
                print(f"  Single tensor: mean={layer_final.mean():.6f}, std={layer_final.std():.6f}")
        
        # AltUp processing analysis
        altup_tensors = [f'{last_layer_name}.predictions', f'{last_layer_name}.altup_corrected', f'{last_layer_name}.after_ple']
        for tensor_name in altup_tensors:
            if tensor_name in captured_tensors:
                tensor = captured_tensors[tensor_name]
                print(f"\n📋 {tensor_name.upper()}:")
                print(f"  Shape: {tensor.shape}")
                if tensor.dim() >= 4 and tensor.shape[0] == 4:
                    print(f"  📊 Per-stream analysis:")
                    for i in range(4):
                        stream = tensor[i]
                        print(f"    HF Stream {i}: mean={stream.mean():.6f}, std={stream.std():.6f}")
        
        # AltUp per-stream predictions analysis
        print(f"\n📋 ALTUP PER-STREAM PREDICTIONS:")
        for i in range(4):
            pred_key = f'{last_layer_name}.altup.predictions_stream_{i}'
            corr_key = f'{last_layer_name}.altup.corrected_stream_{i}'
            if pred_key in captured_tensors:
                pred_stream = captured_tensors[pred_key]
                print(f"  HF Prediction Stream {i}: mean={pred_stream.mean():.6f}, std={pred_stream.std():.6f}")
            if corr_key in captured_tensors:
                corr_stream = captured_tensors[corr_key]
                print(f"  HF Corrected Stream {i}: mean={corr_stream.mean():.6f}, std={corr_stream.std():.6f}")
        
        # Router and modalities analysis
        router_tensors = [f'{last_layer_name}.altup.router_inputs', f'{last_layer_name}.altup.routed', 
                         f'{last_layer_name}.altup.modalities_predict', f'{last_layer_name}.altup.modalities_correct']
        for tensor_name in router_tensors:
            if tensor_name in captured_tensors:
                tensor = captured_tensors[tensor_name]
                print(f"\n📋 {tensor_name.upper()}:")
                print(f"  Shape: {tensor.shape}")
                print(f"  Stats: mean={tensor.mean():.6f}, std={tensor.std():.6f}, min={tensor.min():.6f}, max={tensor.max():.6f}")
        
        # Correction coefficients analysis
        coef_tensors = [f'{last_layer_name}.altup.correction_coefs_output', f'{last_layer_name}.altup.all_coefs']
        for tensor_name in coef_tensors:
            if tensor_name in captured_tensors:
                tensor = captured_tensors[tensor_name]
                print(f"\n📋 {tensor_name.upper()}:")
                print(f"  Shape: {tensor.shape}")
                print(f"  Stats: mean={tensor.mean():.6f}, std={tensor.std():.6f}, min={tensor.min():.6f}, max={tensor.max():.6f}")
                if tensor.numel() <= 20:  # Show values for small tensors
                    print(f"  Values: {tensor.flatten().tolist()}")
        
        # Final norm and stacked streams analysis  
        print(f"\n📋 FINAL PROCESSING CHAIN:")
        print(f"  Final norm output: mean={final_norm_output.mean():.6f}, std={final_norm_output.std():.6f}")
        
        # Show real stacked_streams if captured
        if 'stacked_streams' in captured_tensors:
            print(f"  🔥 REAL STACKED STREAMS (AFTER DECODER LAYERS):")
            real_stacked = captured_tensors['stacked_streams']
            for i in range(4):
                real_stream = real_stacked[i]
                print(f"    Real Stream {i}: mean={real_stream.mean():.6f}, std={real_stream.std():.6f}")
        else:
            print(f"  ⚠️  Real stacked_streams not captured - run inference again")
        print("="*80)

    torch.save(captured_tensors, OUTPUT_FILE)
    print(f"\n💾 Saved {len(captured_tensors)} tensors to {OUTPUT_FILE}")
    
    # Load and compare with ANEMLL if available
    try:
        anemll_path = 'tests/dev/anemll_tensors.pt'
        if os.path.exists(anemll_path):
            print(f"\n🔍 QUICK COMPARISON WITH ANEMLL:")
            anemll_tensors = torch.load(anemll_path, weights_only=False)
            
            comparison_keys = [f'{last_layer_name}.final_output', 'final_norm_output', 'stacked_streams']
            for key in comparison_keys:
                if key in captured_tensors and key in anemll_tensors:
                    hf_tensor = captured_tensors[key]
                    anemll_tensor = anemll_tensors[key]
                    
                    if hf_tensor.shape == anemll_tensor.shape:
                        cos_sim = F.cosine_similarity(hf_tensor.flatten(), anemll_tensor.flatten(), dim=0).item()
                        hf_mag = hf_tensor.std().item()
                        anemll_mag = anemll_tensor.std().item()
                        mag_ratio = anemll_mag / hf_mag if hf_mag > 1e-8 else float('inf')
                        
                        status = "✅ EXCELLENT" if cos_sim > 0.99 else "🟡 GOOD" if cos_sim > 0.9 else "❌ POOR"
                        print(f"  {key}: cos_sim={cos_sim:.6f}, mag_ratio={mag_ratio:.3f} {status}")
                        
                        # Per-stream analysis for multi-stream tensors
                        if key == 'stacked_streams' and hf_tensor.dim() >= 4:
                            for i in range(min(4, hf_tensor.shape[0])):
                                hf_stream = hf_tensor[i]
                                anemll_stream = anemll_tensor[i]
                                stream_cos = F.cosine_similarity(hf_stream.flatten(), anemll_stream.flatten(), dim=0).item()
                                stream_mag_ratio = anemll_stream.std().item() / hf_stream.std().item()
                                print(f"    Stream {i}: cos_sim={stream_cos:.6f}, mag_ratio={stream_mag_ratio:.3f}")
                    else:
                        print(f"  {key}: shape mismatch HF={hf_tensor.shape} vs ANEMLL={anemll_tensor.shape}")
        else:
            print(f"\n⚠️  ANEMLL tensors not found at {anemll_path}")
    except Exception as e:
        print(f"\n⚠️  Error comparing with ANEMLL: {e}")

    print("\n--- Summary of Captured Tensors ---")
    for key in tensor_keys:
        if key in captured_tensors:
            tensor = captured_tensors[key]
            print(f"  - {key:<30} shape: {tensor.shape}, mean: {tensor.mean():.6f}, std: {tensor.std():.6f}")
    
    # Additional captured tensors not in main list
    additional_tensors = [k for k in captured_tensors.keys() if k not in tensor_keys]
    if additional_tensors:
        print("\n--- Additional Debug Tensors ---")
        for key in sorted(additional_tensors):
            tensor = captured_tensors[key]
            print(f"  + {key:<30} shape: {tensor.shape}, mean: {tensor.mean():.6f}, std: {tensor.std():.6f}")
    
    print("-----------------------------------")
    
    # Final summary with magnitude analysis
    print(f"\n📊 MAGNITUDE SUMMARY:")
    key_tensors = ['initial_embedding', f'{last_layer_name}.final_output', 'final_norm_output', 'stacked_streams']
    for key in key_tensors:
        if key in captured_tensors:
            tensor = captured_tensors[key]
            if key == 'stacked_streams' and tensor.dim() >= 4:
                print(f"  {key}:")
                for i in range(min(4, tensor.shape[0])):
                    stream = tensor[i]
                    print(f"    Stream {i}: magnitude={stream.std():.6f}")
            else:
                print(f"  {key}: magnitude={tensor.std():.6f}")

    # --- FINAL LOGITS ANALYSIS ---
    print("\n" + "="*60)
    print("📊 FINAL HF LOGITS ANALYSIS")
    print("="*60)
    
    if 'logits' in captured_tensors:
        logits = captured_tensors['logits']
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits dtype: {logits.dtype}")
        print(f"  Logits stats: mean={logits.mean():.6f}, std={logits.std():.6f}, min={logits.min():.6f}, max={logits.max():.6f}")
        
        # Get top 5 logits and decode them (use last token for multi-token sequences)
        last_token_logits = logits[0, -1]  # Use last token position
        top_logits, top_indices = torch.topk(last_token_logits, 5)
        print(f"  🎯 Top 5 HF logits (last token position):")
        for i, (logit_val, token_idx) in enumerate(zip(top_logits, top_indices)):
            try:
                decoded_token = tokenizer.decode([token_idx.item()])
                print(f"    {i+1}. Token {token_idx.item():5d}: {logit_val.item():8.4f} -> '{decoded_token}'")
            except Exception:
                print(f"    {i+1}. Token {token_idx.item():5d}: {logit_val.item():8.4f} -> [decode error]")
    else:
        print("  ❌ Logits not captured!")
    
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # Smart argument parsing: prioritize numeric arguments as num_layers
    if len(sys.argv) >= 3:
        # Case: python script.py "prompt" 2 (prompt and num_layers)
        prompt = sys.argv[1]
        num_layers = int(sys.argv[2]) if sys.argv[2].isdigit() else 1
    elif len(sys.argv) == 2 and sys.argv[1].isdigit():
        # Case: python script.py 10 (num_layers only, use default prompt)
        prompt = TEST_PROMPT
        num_layers = int(sys.argv[1])
    elif len(sys.argv) == 2:
        # Case: python script.py "custom prompt" (prompt only, single layer)
        prompt = sys.argv[1]
        num_layers = 1
    else:
        # Case: python script.py (no args)
        prompt = TEST_PROMPT
        num_layers = 1
    
    capture_hf_tensors(prompt, num_layers) 
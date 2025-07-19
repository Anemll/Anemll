#!/usr/bin/env python3
# Copyright (c) 2025 ANEMLL
# Licensed under the MIT License

"""
Debug script for the local SimpleGemma3nModel.

This script loads our local PyTorch implementation of the Gemma3n model,
runs a single-token input through it, and uses the built-in debugger hooks
to capture intermediate activation tensors. The keys used for capturing
are intentionally matched with the Hugging Face debug script to allow for
direct comparison.
"""

import torch
import os
import sys
import argparse

# --- Path Setup ---
# Add project root to path to allow importing from tests.dev
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.dev.test_simple_gemma_arch import (
    create_gemma3n_config,
    SimpleGemma3nModel,
    load_gemma3n_weights_from_safetensors
)
from transformers import AutoTokenizer

# --- Configuration ---
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots/0330734afc91972a1aa1ba1bc4495e2723666854/"
)
OUTPUT_FILE = "tests/dev/anemll_tensors.pt"
#TEST_PROMPT = "What"
TEST_PROMPT = "What is the capital of France? Answer:"

# --- Main Execution ---
def run_anemll_model(prompt, debug_mode=True, num_layers=1):
    """Create, load, and run the ANEMLL model, capturing specified tensors."""
    print(f"🔬 Capturing ANEMLL SimpleGemma3nModel tensors...")
    print(f"   Prompt: '{prompt}'")
    print(f"   Processing layers: {num_layers}")
    
    config = create_gemma3n_config()
    # Don't change config.num_hidden_layers as it affects layer initialization
    # Instead, we'll limit processing in the forward pass
    if num_layers < config.num_hidden_layers:
        print(f"   🔍 DEBUG: Will limit processing to {num_layers} layer(s) (original: {config.num_hidden_layers})")
    
    model = SimpleGemma3nModel(config)
    print("✅ Created SimpleGemma3nModel instance.")

    gemma3n_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots/0330734afc91972a1aa1ba1bc4495e2723666854/"
    )
    if not load_gemma3n_weights_from_safetensors(model, gemma3n_path, debug_mode=num_layers):
        print("❌ Weight loading failed.")
        return
    print(f"✅ Loaded weights for {num_layers} layer(s).")
    
    # CRITICAL: Set to eval mode to match HF model behavior
    model.eval()
    print(f"✅ Model set to eval mode (training={model.training})")

    captured_tensors = {}
    
    def get_hook(name):
        def hook(tensor):
            captured_tensors[name] = tensor.detach().cpu()
        return hook

    # Capture tensors from all layers being processed
    layers_to_capture = list(range(num_layers))
    
    print(f"   🔍 DEBUG: Will capture tensors from layers {layers_to_capture}")
    
    debugger_hooks = {
        'initial_embedding': get_hook('initial_embedding'),
        'initial_stream': get_hook('initial_stream'),
        'initial_stream_stack': get_hook('initial_stream_stack'),
        'stacked_streams': get_hook('stacked_streams'),
        'averaged_before_norm': get_hook('averaged_before_norm'),
        'final_norm_input': get_hook('final_norm_input'),
        'final_norm_output': get_hook('final_norm_output'),
        # Stream processing hooks (unembed stage)
        'stream_0_raw': get_hook('stream_0_raw'),
        'stream_1_raw': get_hook('stream_1_raw'),
        'stream_2_raw': get_hook('stream_2_raw'), 
        'stream_3_raw': get_hook('stream_3_raw'),
        'stream_1_projected': get_hook('stream_1_projected'),
        'stream_2_projected': get_hook('stream_2_projected'),
        'stream_3_projected': get_hook('stream_3_projected'),
        'stream_1_normalized': get_hook('stream_1_normalized'),
        'stream_2_normalized': get_hook('stream_2_normalized'),
        'stream_3_normalized': get_hook('stream_3_normalized'),
        'target_magnitude': get_hook('target_magnitude'),
        'hidden_streams_before_unembed': get_hook('hidden_streams_before_unembed'),
        # Logits comparison
        'logits': get_hook('logits'),
    }
    
    # Add hooks for all layers being processed
    for layer_idx in layers_to_capture:
        layer_name = f"layer_{layer_idx}"
        
        # Layer normalization components
        debugger_hooks[f'{layer_name}.pre_attention_norm'] = get_hook(f'{layer_name}.input_layernorm')
        debugger_hooks[f'{layer_name}.post_attention_norm'] = get_hook(f'{layer_name}.post_attention_layernorm')
        debugger_hooks[f'{layer_name}.pre_ffw_norm'] = get_hook(f'{layer_name}.pre_feedforward_layernorm')
        debugger_hooks[f'{layer_name}.post_ffw_norm'] = get_hook(f'{layer_name}.post_feedforward_layernorm')
        
        # Layer outputs
        debugger_hooks[f'{layer_name}.laurel_out'] = get_hook(f'{layer_name}.laurel_out')
        debugger_hooks[f'{layer_name}.mlp_output'] = get_hook(f'{layer_name}.mlp_output')
        debugger_hooks[f'{layer_name}.final_output'] = get_hook(f'{layer_name}.final_output')
        
        # Attention components
        debugger_hooks[f'{layer_name}.attn.q_proj'] = get_hook(f'{layer_name}.attn.q_proj')
        debugger_hooks[f'{layer_name}.attn.k_proj'] = get_hook(f'{layer_name}.attn.k_proj')
        debugger_hooks[f'{layer_name}.attn.v_proj'] = get_hook(f'{layer_name}.attn.v_proj')
        debugger_hooks[f'{layer_name}.attn.q_norm'] = get_hook(f'{layer_name}.attn.q_norm')
        debugger_hooks[f'{layer_name}.attn.k_norm'] = get_hook(f'{layer_name}.attn.k_norm')
        debugger_hooks[f'{layer_name}.attn.attn_output'] = get_hook(f'{layer_name}.attn.attn_output')
        
        # CRITICAL: Position embeddings & RoPE debug hooks
        debugger_hooks[f'{layer_name}.attn.position_embeddings_global'] = get_hook(f'{layer_name}.attn.position_embeddings_global')
        debugger_hooks[f'{layer_name}.attn.position_embeddings_local'] = get_hook(f'{layer_name}.attn.position_embeddings_local')
        debugger_hooks[f'{layer_name}.attn.cos_sin_global'] = get_hook(f'{layer_name}.attn.cos_sin_global')
        debugger_hooks[f'{layer_name}.attn.cos_sin_local'] = get_hook(f'{layer_name}.attn.cos_sin_local')
        debugger_hooks[f'{layer_name}.attn.q_before_rope'] = get_hook(f'{layer_name}.attn.q_before_rope')
        debugger_hooks[f'{layer_name}.attn.k_before_rope'] = get_hook(f'{layer_name}.attn.k_before_rope')
        debugger_hooks[f'{layer_name}.attn.q_after_rope'] = get_hook(f'{layer_name}.attn.q_after_rope')
        debugger_hooks[f'{layer_name}.attn.k_after_rope'] = get_hook(f'{layer_name}.attn.k_after_rope')
        
        # Attention computation intermediates
        debugger_hooks[f'{layer_name}.attn.attention_mask'] = get_hook(f'{layer_name}.attn.attention_mask')
        debugger_hooks[f'{layer_name}.attn.causal_mask'] = get_hook(f'{layer_name}.attn.causal_mask')
        debugger_hooks[f'{layer_name}.attn.value_states_before_norm'] = get_hook(f'{layer_name}.attn.value_states_before_norm')
        debugger_hooks[f'{layer_name}.attn.value_states_transposed'] = get_hook(f'{layer_name}.attn.value_states_transposed')
        debugger_hooks[f'{layer_name}.attn.raw_attn_weights'] = get_hook(f'{layer_name}.attn.raw_attn_weights')
        debugger_hooks[f'{layer_name}.attn.scaled_attn_weights'] = get_hook(f'{layer_name}.attn.scaled_attn_weights')
        debugger_hooks[f'{layer_name}.attn.attn_weights_pre_softmax'] = get_hook(f'{layer_name}.attn.attn_weights_pre_softmax')
        debugger_hooks[f'{layer_name}.attn.attn_weights_post_softmax'] = get_hook(f'{layer_name}.attn.attn_weights_post_softmax')
        debugger_hooks[f'{layer_name}.attn.attn_output_pre_reshape'] = get_hook(f'{layer_name}.attn.attn_output_pre_reshape')
        
        # MLP components
        debugger_hooks[f'{layer_name}.mlp.gate_proj'] = get_hook(f'{layer_name}.mlp.gate_proj')
        debugger_hooks[f'{layer_name}.mlp.gate_proj_after_sparsity'] = get_hook(f'{layer_name}.mlp.gate_proj_after_sparsity')
        debugger_hooks[f'{layer_name}.mlp.gate_proj_activated'] = get_hook(f'{layer_name}.mlp.gate_proj_activated')
        debugger_hooks[f'{layer_name}.mlp.up_proj'] = get_hook(f'{layer_name}.mlp.up_proj')
        debugger_hooks[f'{layer_name}.mlp.activations'] = get_hook(f'{layer_name}.mlp.activations')
        
        # AltUp components
        debugger_hooks[f'{layer_name}.processed_active_stream'] = get_hook(f'{layer_name}.processed_active_stream')
        debugger_hooks[f'{layer_name}.predictions'] = get_hook(f'{layer_name}.predictions')
        debugger_hooks[f'{layer_name}.altup_corrected'] = get_hook(f'{layer_name}.altup_corrected')
        debugger_hooks[f'{layer_name}.corrected_streams'] = get_hook(f'{layer_name}.corrected_streams')
        debugger_hooks[f'{layer_name}.after_ple'] = get_hook(f'{layer_name}.after_ple')
        
        # Intermediate processing tensors
        debugger_hooks[f'{layer_name}.active_stream_initial'] = get_hook(f'{layer_name}.active_stream_initial')
        debugger_hooks[f'{layer_name}.active_prediction'] = get_hook(f'{layer_name}.active_prediction')
        debugger_hooks[f'{layer_name}.active_prediction_normed'] = get_hook(f'{layer_name}.active_prediction_normed')
        
        # AltUp router components
        debugger_hooks[f'{layer_name}.altup.modalities_predict'] = get_hook(f'{layer_name}.altup.modalities_predict')
        debugger_hooks[f'{layer_name}.altup.modalities_correct'] = get_hook(f'{layer_name}.altup.modalities_correct')
        debugger_hooks[f'{layer_name}.altup.router_inputs'] = get_hook(f'{layer_name}.altup.router_inputs')
        debugger_hooks[f'{layer_name}.altup.routed'] = get_hook(f'{layer_name}.altup.routed')
        
        # Per-stream predictions
        debugger_hooks[f'{layer_name}.altup.predictions_stream_0'] = get_hook(f'{layer_name}.altup.predictions_stream_0')
        debugger_hooks[f'{layer_name}.altup.predictions_stream_1'] = get_hook(f'{layer_name}.altup.predictions_stream_1')
        debugger_hooks[f'{layer_name}.altup.predictions_stream_2'] = get_hook(f'{layer_name}.altup.predictions_stream_2')
        debugger_hooks[f'{layer_name}.altup.predictions_stream_3'] = get_hook(f'{layer_name}.altup.predictions_stream_3')
        debugger_hooks[f'{layer_name}.altup.activated'] = get_hook(f'{layer_name}.altup.activated')
        
        # Per-stream corrected predictions
        debugger_hooks[f'{layer_name}.altup.corrected_stream_0'] = get_hook(f'{layer_name}.altup.corrected_stream_0')
        debugger_hooks[f'{layer_name}.altup.corrected_stream_1'] = get_hook(f'{layer_name}.altup.corrected_stream_1')
        debugger_hooks[f'{layer_name}.altup.corrected_stream_2'] = get_hook(f'{layer_name}.altup.corrected_stream_2')
        debugger_hooks[f'{layer_name}.altup.corrected_stream_3'] = get_hook(f'{layer_name}.altup.corrected_stream_3')
        
        # Correction coefficients
        debugger_hooks[f'{layer_name}.altup.correction_coefs_output'] = get_hook(f'{layer_name}.altup.correction_coefs_output')
        debugger_hooks[f'{layer_name}.altup.all_coefs'] = get_hook(f'{layer_name}.altup.all_coefs')
    
    print("✅ Defined debugger hooks.")
    
    tokenizer = AutoTokenizer.from_pretrained(gemma3n_path)
    print("✅ Tokenizer loaded.")
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    
    print(f"\n🧪 Running inference with prompt: '{prompt}' (Input IDs: {input_ids})")
    
    # Add KV cache debugging
    print(f"🔍 KV CACHE DEBUG: Processing {num_layers} layers")
    kv_sharing_start = config.num_hidden_layers - config.num_kv_shared_layers
    print(f"  KV sharing configuration:")
    print(f"    num_hidden_layers: {config.num_hidden_layers}")
    print(f"    num_kv_shared_layers: {config.num_kv_shared_layers}")  
    print(f"    kv_sharing_start: {kv_sharing_start}")
    
    if num_layers >= kv_sharing_start:
        print(f"  🔍 Layer {num_layers-1} should use KV sharing!")
        print(f"    Source layer: {num_layers-1 - config.num_kv_shared_layers}")
    else:
        print(f"  🔍 Layer {num_layers-1} should compute own K/V")
    
    with torch.no_grad():
        model(input_ids, debugger_hooks=debugger_hooks, debug_layer_limit=num_layers)
    print("✅ Inference complete.")
    
    # Logits will be analyzed at the very end

    tensor_keys = list(captured_tensors.keys())
    print("\n--- Summary of Captured Tensors ---")
    for key in sorted(tensor_keys):
        if key in captured_tensors:
            print(f"  - {key:<30} shape: {captured_tensors[key].shape}")
    print("-----------------------------------")

    save_path = "tests/dev/anemll_tensors.pt"
    torch.save(captured_tensors, save_path)
    print(f"\n💾 Saved {len(captured_tensors)} tensors to {save_path}")

    # --- FINAL LOGITS ANALYSIS ---
    print("\n" + "="*60)
    print("📊 FINAL ANEMLL LOGITS ANALYSIS")
    print("="*60)
    
    if 'logits' in captured_tensors:
        logits = captured_tensors['logits']
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits dtype: {logits.dtype}")
        print(f"  Logits stats: mean={logits.mean():.6f}, std={logits.std():.6f}, min={logits.min():.6f}, max={logits.max():.6f}")
        
        # Get top 5 logits and decode them (use last token for multi-token sequences)
        last_token_logits = logits[0, -1]  # Use last token position
        top_logits, top_indices = torch.topk(last_token_logits, 5)
        print(f"  🎯 Top 5 ANEMLL logits (last token position):")
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
    
    run_anemll_model(prompt, debug_mode=num_layers, num_layers=num_layers) 
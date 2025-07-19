#!/usr/bin/env python3
# Copyright (c) 2025 ANEMLL
# Licensed under the MIT License

"""Detailed layer-by-layer comparison to find where HF vs ANEMLL diverges."""

import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from test_simple_gemma_arch import SimpleGemma3nModel, create_gemma3n_config, load_gemma3n_weights_from_safetensors, create_rotary_cache

def detailed_layer_comparison():
    """Compare HF vs ANEMLL layer by layer to find divergence."""
    print("🔍 DETAILED LAYER-BY-LAYER COMPARISON")
    print("=" * 60)
    
    local_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-3n-E2B-it/snapshots/0330734afc91972a1aa1ba1bc4495e2723666854/"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    
    # Load HF model
    print("🤗 Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    hf_model.eval()
    
    # Load our model
    print("🏗️ Loading our SimpleGemma3n model...")
    config = create_gemma3n_config()
    our_model = SimpleGemma3nModel(config)
    load_gemma3n_weights_from_safetensors(our_model, local_path)
    our_model.eval()
    
    # Test input - the problematic case
    test_text = "What is the capital of France? Answer:"
    #test_text = "What"
    test_tokens = tokenizer.encode(test_text, add_special_tokens=True)
    input_ids = torch.tensor([test_tokens], dtype=torch.long)
    
    print(f"\n🧪 Testing: '{test_text}'")
    print(f"Tokens: {test_tokens}")
    print(f"Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        # === HF MODEL FORWARD PASS ===
        print(f"\n🤗 HF MODEL LAYER-BY-LAYER:")
        
        # HF full forward pass to get all hidden states
        hf_full_output = hf_model(input_ids, output_hidden_states=True)
        hf_hidden_states_all = hf_full_output.hidden_states
        
        # HF embeddings (first hidden state)
        hf_hidden_states = hf_hidden_states_all[0]
        print(f"HF Embeddings: {hf_hidden_states.shape}, range [{hf_hidden_states.min():.3f}, {hf_hidden_states.max():.3f}]")
        
        # HF per-layer embeddings (if they exist) - not directly from hidden_states
        try:
            hf_ple = hf_model.model.language_model.embed_tokens_per_layer(input_ids)
            print(f"HF PLE: {hf_ple.shape}, range [{hf_ple.min():.3f}, {hf_ple.max():.3f}]")
        except:
            print("HF PLE: Not accessible or different structure")
        
        # Store HF intermediate states (all hidden states from the full forward pass)
        # hidden_states_all includes embeddings + outputs of all layers
        hf_layer_outputs = [h.clone() for h in hf_hidden_states_all]
        
        # === OUR MODEL FORWARD PASS ===
        print(f"\n🏗️ OUR MODEL LAYER-BY-LAYER:")
        
        # Our embeddings (create 4-stream bundle like in model.forward)
        hidden0 = our_model.embed_tokens(input_ids) * our_model.embed_scale
        print(f"Our Embeddings (single): {hidden0.shape}, range [{hidden0.min():.3f}, {hidden0.max():.3f}]")
        
        # Build 4-stream bundle exactly like model.forward
        target_magnitude = torch.mean(hidden0**2, dim=-1, keepdim=True) ** 0.5
        epsilon_tensor = torch.tensor(1e-5, device=hidden0.device, dtype=hidden0.dtype)
        
        streams = [hidden0]
        for i in range(1, 4):
            altup_proj = our_model.altup_projections[i - 1](hidden0)
            current_stream = altup_proj.to(dtype=hidden0.dtype, device=hidden0.device)
            new_magnitude = torch.mean(current_stream**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
            current_stream = current_stream * target_magnitude / new_magnitude
            streams.append(current_stream)
        
        our_hidden_states = torch.stack(streams, dim=0)  # [4, B, T, D]
        print(f"Our 4-stream bundle: {our_hidden_states.shape}, range [{our_hidden_states.min():.3f}, {our_hidden_states.max():.3f}]")
        
        # Our per-layer embeddings - process them like the main model
        per_layer_scale = (our_model.config.hidden_size_per_layer_input**0.5)
        our_ple_raw = our_model.embed_tokens_per_layer(input_ids) * per_layer_scale
        print(f"Our PLE raw: {our_ple_raw.shape}, range [{our_ple_raw.min():.3f}, {our_ple_raw.max():.3f}]")
        
        # Reshape and process PLE like main model does
        batch_size, seq_len = input_ids.shape
        our_ple_reshaped = our_ple_raw.view(
            batch_size, seq_len, our_model.config.num_hidden_layers, our_model.config.hidden_size_per_layer_input
        )
        # Project per-layer inputs using the first stream (like main model)
        our_ple = our_model.project_per_layer_inputs(our_hidden_states[0], our_ple_reshaped)
        print(f"Our PLE processed: {our_ple.shape}, range [{our_ple.min():.3f}, {our_ple.max():.3f}]")
        
        # Generate position embeddings like the main model does
        bsz, seq_len = input_ids.shape
        
        # Generate global position embeddings (for non-sliding layers)
        position_embeddings_global = create_rotary_cache(
            our_model.head_dim, seq_len, our_model.rope_theta_global,
            device=input_ids.device, dtype=torch.float32, batch_size=bsz
        )
        
        # Generate local position embeddings (for sliding layers)
        position_embeddings_local = create_rotary_cache(
            our_model.head_dim, seq_len, our_model.rope_theta_local,
            device=input_ids.device, dtype=torch.float32, batch_size=bsz
        )
        
        print(f"Position embeddings generated - Global: {position_embeddings_global[0].shape}, Local: {position_embeddings_local[0].shape}")
        
        # CRITICAL FIX: Generate position_ids (missing from original script!)
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.int64).unsqueeze(0).expand(bsz, -1)
        print(f"Position IDs generated: {position_ids.shape} - {position_ids}")
        
        # Store our intermediate states (4-stream tensors)
        our_layer_outputs = [our_hidden_states.clone()]
        
        # === LAYER-BY-LAYER COMPARISON ===
        print(f"\n📊 LAYER-BY-LAYER COMPARISON:")
        
        # Compare embeddings first with dimension check
        print(f"HF embeddings shape: {hf_layer_outputs[0].shape}")
        print(f"Our embeddings shape: {our_layer_outputs[0].shape}")
        
        # Check if dimensions match for cosine similarity (both should now be 4-stream)
        hf_emb = hf_layer_outputs[0]  # [4, B, T, D]
        our_emb = our_layer_outputs[0]  # [4, B, T, D]
        
        # Compare each stream individually for embeddings
        if len(hf_emb.shape) == 4 and len(our_emb.shape) == 4 and hf_emb.shape[0] == 4 and our_emb.shape[0] == 4:
            print(f"  📊 Embedding 4-stream comparison:")
            for stream_idx in range(4):
                hf_stream = hf_emb[stream_idx]  # [B, T, D]
                our_stream = our_emb[stream_idx]  # [B, T, D]
                
                hf_stream_flat = hf_stream.flatten()
                our_stream_flat = our_stream.flatten()
                
                if hf_stream_flat.shape == our_stream_flat.shape:
                    stream_similarity = F.cosine_similarity(
                        hf_stream_flat.unsqueeze(0),
                        our_stream_flat.unsqueeze(0)
                    ).item()
                    
                    if stream_similarity > 0.99:
                        status = "✅ Excellent"
                    elif stream_similarity > 0.95:
                        status = "⚠️  Good"
                    elif stream_similarity > 0.8:
                        status = "⚠️  Moderate"
                    else:
                        status = "❌ Poor"
                    
                    print(f"    Stream {stream_idx}: {stream_similarity:.6f} {status}")
                else:
                    print(f"    Stream {stream_idx}: Dimension mismatch!")
        else:
            print(f"  ⚠️  Unexpected embedding shapes - not both 4-stream!")
        
        # Correct K/V sharing implementation, mirroring the main test script.
        kv_shared_caches = {}
        # Process through layers - compare first few layers in detail
        for i in range(len(our_model.layers)):
            print(f"--- Layer {i} ---")
            
            # Our layer processing  
            try:
                our_layer = our_model.layers[i]
                
                print(f"Our Layer {i}: Input 4-stream range [{our_hidden_states.min():.3f}, {our_hidden_states.max():.3f}]")
                
                # Determine kv_cache for the current layer using reference-accurate mapping
                # Following the same logic as the main model
                layer_types = our_model.config.layer_types
                kv_sharing_start = 20
                
                if i < kv_sharing_start:
                    shared_cache = None
                else:
                    # Use attention-type-aware mapping like the main model
                    concrete_layers = layer_types[:kv_sharing_start]  # layers 0-19
                    
                    # Find last occurrence of each attention type in unshared region
                    shared_sliding_idx = len(concrete_layers) - 1 - concrete_layers[::-1].index("sliding_attention")  # 18
                    shared_global_idx = len(concrete_layers) - 1 - concrete_layers[::-1].index("full_attention")      # 19
                    
                    # Determine source layer based on current layer's attention type
                    layer_type = layer_types[i]
                    if layer_type == "full_attention":
                        source_layer_idx = shared_global_idx    # → 19
                    elif layer_type == "sliding_attention":
                        source_layer_idx = shared_sliding_idx   # → 18
                    
                    shared_cache = kv_shared_caches.get(source_layer_idx)
                    
                    if shared_cache is not None:
                        print(f"🔍 KV CACHE: Layer {i} ({layer_type}) using shared K/V from Layer {source_layer_idx}")
                    else:
                        print(f"⚠️  KV CACHE: Layer {i} ({layer_type}) expected to use Layer {source_layer_idx} but cache not found!")
                
                # Process through our layer (4-stream input -> 4-stream output)
                # Extract per-layer slice for this specific layer (like main model does)
                # our_ple is already processed and has shape [B, T, 30, 256]
                per_layer_slice = our_ple[:, :, i, :]  # [B, T, 256] - slice for layer i
                
                our_hidden_states = our_layer(
                    our_hidden_states,  # [4, B, T, D] - 4-stream tensor
                    per_layer_embeddings=per_layer_slice,   # [B, T, 256] - per-layer slice for layer i
                    position_embeddings_global=position_embeddings_global,  # Add global position embeddings
                    position_embeddings_local=position_embeddings_local,    # Add local position embeddings
                    attention_mask=None,
                    position_ids=position_ids,  # CRITICAL FIX: Add missing position_ids!
                    shared_cache=shared_cache
                )
                
                # Store K/V for layers that will be used as sources (reference implementation)
                # Only store for layers that are actually referenced by the mapping
                kv_sharing_start = 20
                if i < kv_sharing_start:
                    # Check if this layer will be used as a source for KV sharing
                    layer_types = our_model.config.layer_types
                    concrete_layers = layer_types[:kv_sharing_start]  # layers 0-19
                    shared_sliding_idx = len(concrete_layers) - 1 - concrete_layers[::-1].index("sliding_attention")  # 18
                    shared_global_idx = len(concrete_layers) - 1 - concrete_layers[::-1].index("full_attention")      # 19
                    
                    if i in [shared_sliding_idx, shared_global_idx]:  # Only layers 18, 19
                        kv_shared_caches[i] = our_layer.self_attn.kv_for_sharing
                        print(f"🔍 KV CACHE: Stored Layer {i} K/V for sharing (source layer)")
                
                our_layer_outputs.append(our_hidden_states.clone())
                print(f"Our Layer {i}: Output 4-stream range [{our_hidden_states.min():.3f}, {our_hidden_states.max():.3f}]")
                
                # Check for NaN or explosion
                if torch.isnan(our_hidden_states).any():
                    print(f"  🚨 NaN detected in layer {i}!")
                    break
                    
                if our_hidden_states.abs().max() > 1000:
                    print(f"  ⚠️  Large values detected in layer {i}!")
                    
            except Exception as e:
                print(f"Our Layer {i}: Error - {e}")
                import traceback
                traceback.print_exc()
                break
            
            # Cosine similarity check for EVERY layer - COMPARE ALL 4 STREAMS
            if True:  # Check every layer instead of every 5
                # hf_layer_outputs has num_layers + 1 elements (embeddings + layer outputs)
                # So, hf_layer_outputs[i+1] corresponds to the output of layer i
                if len(hf_layer_outputs) > (i + 1) and len(our_layer_outputs) > (i + 1):
                    hf_layer = hf_layer_outputs[i+1]
                    our_layer = our_layer_outputs[i+1]
                    
                    # Check if HF has 4-stream output [4, B, T, H]
                    if len(hf_layer.shape) == 4 and hf_layer.shape[0] == 4:
                        print(f"  📊 Layer {i} - 4-stream comparison:")
                        
                        # Compare each stream individually
                        for stream_idx in range(4):
                            hf_stream = hf_layer[stream_idx]  # [B, T, H]
                            our_stream = our_layer[stream_idx]  # Our model now outputs 4-stream [B, T, H]
                            
                            hf_stream_flat = hf_stream.flatten()
                            our_stream_flat = our_stream.flatten()
                            
                            if hf_stream_flat.shape == our_stream_flat.shape:
                                stream_similarity = F.cosine_similarity(
                                    hf_stream_flat.unsqueeze(0),
                                    our_stream_flat.unsqueeze(0)
                                ).item()
                                
                                # Color code the results
                                if stream_similarity > 0.99:
                                    status = "✅ Excellent"
                                elif stream_similarity > 0.95:
                                    status = "⚠️  Good"
                                elif stream_similarity > 0.8:
                                    status = "⚠️  Moderate"
                                else:
                                    status = "❌ Poor"
                                
                                print(f"    Stream {stream_idx}: {stream_similarity:.6f} {status}")
                            else:
                                print(f"    Stream {stream_idx}: Dimension mismatch! HF: {hf_stream_flat.shape[0]}, Our: {our_stream_flat.shape[0]}")
                        
                        # Also compute overall best match
                        best_similarity = -1
                        best_stream = -1
                        for stream_idx in range(4):
                            hf_stream = hf_layer[stream_idx]
                            our_stream = our_layer[stream_idx]  # Fixed: use our_layer[stream_idx]
                            hf_stream_flat = hf_stream.flatten()
                            our_stream_flat = our_stream.flatten()
                            
                            if hf_stream_flat.shape == our_stream_flat.shape:
                                sim = F.cosine_similarity(
                                    hf_stream_flat.unsqueeze(0),
                                    our_stream_flat.unsqueeze(0)
                                ).item()
                                if sim > best_similarity:
                                    best_similarity = sim
                                    best_stream = stream_idx
                        
                        print(f"    💡 Best match: Stream {best_stream} ({best_similarity:.6f})")
                        
                    else:
                        # Handle single stream case (fallback)
                        hf_layer_flat = hf_layer.flatten()
                        our_layer_flat = our_layer.flatten()
                        
                        if hf_layer_flat.shape == our_layer_flat.shape:
                            layer_similarity = F.cosine_similarity(
                                hf_layer_flat.unsqueeze(0),
                                our_layer_flat.unsqueeze(0)
                            ).item()
                            print(f"  📊 Layer {i} cosine similarity: {layer_similarity:.6f}")
                            if layer_similarity < 0.95:
                                print(f"     ❌ Below 0.95 - significant divergence!")
                            elif layer_similarity > 0.99:
                                print(f"     ✅ Excellent match!")
                            else:
                                print(f"     ⚠️  Moderate divergence")
                        else:
                            print(f"  ⚠️  Layer {i+1} dimension mismatch! HF: {hf_layer_flat.shape[0]}, Our: {our_layer_flat.shape[0]}")
        
        # === FINAL COMPARISON ===
        print(f"\n🎯 FINAL OUTPUTS:")
        
        # HF final output
        try:
            hf_full_output = hf_model(input_ids)
            hf_final_logits = hf_full_output.logits[0, -1, :]
            hf_top_tokens = torch.topk(hf_final_logits, 5)[1]
            
            print(f"HF Final: range [{hf_final_logits.min():.3f}, {hf_final_logits.max():.3f}]")
            print("HF Top 5:")
            for i, idx in enumerate(hf_top_tokens):
                token = tokenizer.decode([int(idx)])
                print(f"  {i+1}. '{token}' ({hf_final_logits[idx]:.3f})")
        except Exception as e:
            print(f"HF Final: Error - {e}")
        
        # Our final output 
        try:
            # UN-EMBED AND MERGE EXACTLY LIKE HF (matching model.forward)
            print(f"Before final un-embedding: 4-stream shape {our_hidden_states.shape}")
            
            scaled = []
            for s_idx in range(4):
                hs = our_hidden_states[s_idx]  # [B, T, D]
                
                # CRITICAL FIX: Match main test script logic exactly
                if s_idx > 0:  # streams 1, 2, 3
                    # Apply unembed projection
                    hs = our_model.altup_unembed_projections[s_idx - 1](hs)
                    
                    # Apply magnitude normalization (like main test script)
                    new_magnitude = torch.mean(hs**2, dim=-1, keepdim=True)
                    epsilon_tensor = torch.tensor(1e-5, device=hs.device, dtype=hs.dtype)
                    new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
                    target_magnitude = torch.mean(our_hidden_states[0]**2, dim=-1, keepdim=True) ** 0.5
                    hs = hs * target_magnitude / new_magnitude
                
                # NO extra scaling - this was the bug!
                
                scaled.append(hs)
            
            # Average the properly scaled and projected streams
            our_final_single_stream = torch.mean(torch.stack(scaled, 0), dim=0)  # [B, T, D]
            print(f"After final HF-style un-embedding and merge: {our_final_single_stream.shape}")
            
            # Apply final norm and LM head
            our_final_states = our_model.norm(our_final_single_stream)
            our_final_logits = our_model.lm_head(our_final_states)
            
            # CRITICAL FIX: Apply softcapping correctly (match main test script)
            if our_model.final_logit_softcapping is not None:
                softcap = our_model.final_logit_softcapping
                our_final_logits = softcap * torch.tanh(our_final_logits / softcap)
            
            our_final_logits = our_final_logits[0, -1, :]
            our_top_tokens = torch.topk(our_final_logits, 5)[1]
            
            print(f"Our Final: range [{our_final_logits.min():.3f}, {our_final_logits.max():.3f}]")
            print("Our Top 5:")
            for i, idx in enumerate(our_top_tokens):
                token = tokenizer.decode([int(idx)])
                print(f"  {i+1}. '{token}' ({our_final_logits[idx]:.3f})")
                
            # Compare final outputs
            print(f"HF final logits shape: {hf_final_logits.shape}")
            print(f"Our final logits shape: {our_final_logits.shape}")
            
            if hf_final_logits.shape == our_final_logits.shape:
                final_similarity = F.cosine_similarity(
                    hf_final_logits.unsqueeze(0),
                    our_final_logits.unsqueeze(0)
                ).item()
                print(f"\nFinal similarity: {final_similarity:.6f} {'❌' if final_similarity < 0.9 else '✅'}")
            else:
                print(f"\n⚠️  Final output dimension mismatch! Cannot compute cosine similarity")
                print(f"HF size: {hf_final_logits.shape[0]}, Our size: {our_final_logits.shape[0]}")
            
        except Exception as e:
            print(f"Our Final: Error - {e}")
            import traceback
            traceback.print_exc()
        
        # === SPECIFIC CHECKS ===
        print(f"\n🔍 SPECIFIC DIAGNOSTIC CHECKS:")
        
        # Check if "Paris" token is getting suppressed (only if prompt contains "France")
        if "france" in test_text.lower():
            print(f"\n🎯 PARIS PREDICTION TEST:")
            try:
                paris_token = tokenizer.encode(" Paris", add_special_tokens=False)[0]
                print(f"  Paris token: ' Paris' (ID: {paris_token})")
                
                if 'hf_final_logits' in locals():
                    hf_paris_logit = hf_final_logits[paris_token].item()
                    hf_paris_rank = (hf_final_logits.argsort(descending=True) == paris_token).nonzero().item() + 1
                    print(f"  HF Paris:  logit {hf_paris_logit:.3f}, rank {hf_paris_rank}")
                
                if 'our_final_logits' in locals():
                    our_paris_logit = our_final_logits[paris_token].item()
                    our_paris_rank = (our_final_logits.argsort(descending=True) == paris_token).nonzero().item() + 1
                    print(f"  Our Paris: logit {our_paris_logit:.3f}, rank {our_paris_rank}")
                    
                    logit_diff = abs(hf_paris_logit - our_paris_logit)
                    rank_diff = abs(hf_paris_rank - our_paris_rank)
                    print(f"  Logit difference: {logit_diff:.3f}")
                    print(f"  Rank difference: {rank_diff}")
                    
                    # Status assessment
                    if our_paris_rank <= 100:
                        status = "✅ EXCELLENT"
                    elif our_paris_rank <= 1000:
                        status = "⚠️  GOOD"
                    elif our_paris_rank <= 10000:
                        status = "⚠️  MODERATE"
                    else:
                        status = "❌ POOR"
                    print(f"  {status}: Paris at rank {our_paris_rank}")
                    
            except Exception as e:
                print(f"  Paris check failed: {e}")
        else:
            print(f"\n💡 Paris prediction test skipped (prompt doesn't contain 'France')")
        
        # Check model configurations
        print(f"\n🔍 MODEL CONFIGURATION COMPARISON:")
        print(f"HF final_logit_softcapping: {hf_model.config.text_config.final_logit_softcapping}")
        print(f"Our final_logit_softcapping: {our_model.config.final_logit_softcapping}")
        print(f"Our embed_scale: {our_model.embed_scale}")


def analyze_activation_patterns():
    """Analyze activation patterns in detail."""
    print(f"\n" + "="*60)
    print("🔬 ACTIVATION PATTERN ANALYSIS")
    print("="*60)
    
    # This would involve more detailed analysis of specific components
    # For now, just run the main comparison
    pass

if __name__ == "__main__":
    detailed_layer_comparison()
    analyze_activation_patterns()
#!/usr/bin/env python3
"""
Layer-by-layer comparison between ANEMLL and transformers to localize issues.
Tests with the GPTQ quantized model to find where outputs diverge.
"""

import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anemll.models.qwen2_5_model import Qwen25ForCausalLM, Qwen25Config


class ModelComparator:
    """Compare ANEMLL and transformers models layer by layer."""
    
    def __init__(self, model_name="smpanaro/Qwen2.5-0.5B-4bit-PerTensor"):
        self.model_name = model_name
        self.model_path = snapshot_download(model_name)
        
        # Disable quantization for fair comparison
        os.environ['SKIP_SP_FORWARD'] = '1'
        
        # Load both models
        print("Loading models...")
        self.load_models()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
    def load_models(self):
        """Load both ANEMLL and transformers models."""
        # Load transformers model
        print("Loading transformers model...")
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.hf_model.eval()
        
        # Load ANEMLL model
        print("Loading ANEMLL model...")
        config = Qwen25Config.from_json(f'{self.model_path}/config.json')
        self.anemll_model = Qwen25ForCausalLM(config, disable_kv_cache=False)
        self.anemll_model.load_pretrained_weights(self.model_path)
        self.anemll_model.eval()
    
    def compare_tensors(self, name, tensor1, tensor2, rtol=1e-4, atol=1e-4):
        """Compare two tensors and report differences."""
        # Convert to same dtype for comparison
        t1 = tensor1.float().detach()
        t2 = tensor2.float().detach()
        
        # Basic stats
        print(f"\n{name}:")
        print(f"  Shape: {t1.shape} vs {t2.shape}")
        print(f"  Dtype: {tensor1.dtype} vs {tensor2.dtype}")
        print(f"  Mean: {t1.mean().item():.6f} vs {t2.mean().item():.6f}")
        print(f"  Std: {t1.std().item():.6f} vs {t2.std().item():.6f}")
        print(f"  Min: {t1.min().item():.6f} vs {t2.min().item():.6f}")
        print(f"  Max: {t1.max().item():.6f} vs {t2.max().item():.6f}")
        
        if t1.shape != t2.shape:
            print(f"  ❌ SHAPE MISMATCH!")
            return False
        
        # Check if close
        close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
        if close:
            print(f"  ✓ Tensors match (rtol={rtol}, atol={atol})")
        else:
            # Find maximum difference
            diff = torch.abs(t1 - t2)
            max_diff = diff.max().item()
            rel_diff = (diff / (torch.abs(t1) + 1e-8)).max().item()
            
            # Find location of max difference
            max_idx = torch.argmax(diff)
            max_pos = np.unravel_index(max_idx.item(), diff.shape)
            
            print(f"  ❌ MISMATCH!")
            print(f"  Max absolute diff: {max_diff:.6f}")
            print(f"  Max relative diff: {rel_diff:.6f}")
            print(f"  At position {max_pos}: {t1[max_pos].item():.6f} vs {t2[max_pos].item():.6f}")
            
            # Show first few values
            print(f"  First 5 values:")
            print(f"    HF: {t1.flatten()[:5].tolist()}")
            print(f"    ANEMLL: {t2.flatten()[:5].tolist()}")
        
        return close
    
    def test_embeddings(self, input_ids):
        """Compare embedding layer outputs."""
        print("\n" + "="*60)
        print("1. EMBEDDING LAYER COMPARISON")
        print("="*60)
        
        with torch.no_grad():
            # Transformers embeddings
            hf_embeds = self.hf_model.model.embed_tokens(input_ids)
            
            # ANEMLL embeddings
            anemll_embeds = self.anemll_model.model.embed_tokens(input_ids)
        
        match = self.compare_tensors("Embeddings", hf_embeds, anemll_embeds)
        return hf_embeds, anemll_embeds, match
    
    def test_first_layer(self, input_ids):
        """Compare first transformer layer outputs."""
        print("\n" + "="*60)
        print("2. FIRST TRANSFORMER LAYER COMPARISON")
        print("="*60)
        
        # Get embeddings first
        hf_embeds, anemll_embeds, _ = self.test_embeddings(input_ids)
        
        # Prepare inputs
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            # Manual computation for HF layer (more reliable)
            print("Computing HF layer manually...")
            hf_layer = self.hf_model.model.layers[0]
            
            # Apply input layernorm
            hidden_states = hf_layer.input_layernorm(hf_embeds)
            
            # Get position embeddings from rotary_emb
            position_embeddings = self.hf_model.model.rotary_emb(hidden_states, position_ids)
            
            # Self attention with proper position embeddings
            attn_result = hf_layer.self_attn(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                position_embeddings=position_embeddings
            )
            
            # Handle different return formats
            if isinstance(attn_result, tuple):
                attn_output = attn_result[0]
            else:
                attn_output = attn_result
            
            # Residual connection
            hidden_states = hf_embeds + attn_output
            
            # Post attention layernorm
            residual = hidden_states
            hidden_states = hf_layer.post_attention_layernorm(hidden_states)
            
            # MLP
            mlp_output = hf_layer.mlp(hidden_states)
            
            # Final residual
            hf_hidden_states = residual + mlp_output
            
            print(f"HF manual computation shape: {hf_hidden_states.shape}")
            
            try:
                # ANEMLL first layer
                anemll_layer = self.anemll_model.model.layers[0]
                
                # Create causal mask for ANEMLL
                causal_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float16)
                for i in range(seq_len):
                    causal_mask[:, :, i, i+1:] = float('-inf')
                
                # Create rotary embeddings for ANEMLL - use the layer's rotary_emb
                cos, sin = anemll_layer.self_attn.rotary_emb(anemll_embeds, position_ids[0])
                
                # ANEMLL layer call
                anemll_output = anemll_layer(
                    anemll_embeds,
                    causal_mask,
                    position_ids[0],
                    (cos, sin),
                    None  # kv_cache
                )
                
                # Extract hidden states
                if isinstance(anemll_output, tuple):
                    anemll_hidden_states = anemll_output[0]
                else:
                    anemll_hidden_states = anemll_output
                
                print(f"ANEMLL layer output shape: {anemll_hidden_states.shape}")
                
            except Exception as e:
                print(f"❌ ANEMLL layer failed: {e}")
                import traceback
                traceback.print_exc()
                return None, None, False
        
        match = self.compare_tensors("Layer 0 output", hf_hidden_states, anemll_hidden_states)
        return hf_hidden_states, anemll_hidden_states, match
    
    def test_attention_components(self, input_ids):
        """Compare attention mechanism components in detail."""
        print("\n" + "="*60)
        print("3. ATTENTION MECHANISM DETAILED COMPARISON")
        print("="*60)
        
        # Get embeddings
        hf_embeds, anemll_embeds, _ = self.test_embeddings(input_ids)
        
        # Access first layer attention
        hf_attn = self.hf_model.model.layers[0].self_attn
        anemll_attn = self.anemll_model.model.layers[0].self_attn
        
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            # Compare projection weights first
            print("\n--- Comparing Projection Weights ---")
            
            # Q projection weights
            hf_q_weight = hf_attn.q_proj.weight
            anemll_q_weight = anemll_attn.q_proj.weight.squeeze(-1).squeeze(-1).t()  # Conv2d to Linear
            self.compare_tensors("Q projection weight", hf_q_weight, anemll_q_weight)
            
            # K projection weights
            hf_k_weight = hf_attn.k_proj.weight
            anemll_k_weight = anemll_attn.k_proj.weight.squeeze(-1).squeeze(-1).t()
            self.compare_tensors("K projection weight", hf_k_weight, anemll_k_weight)
            
            # V projection weights
            hf_v_weight = hf_attn.v_proj.weight
            anemll_v_weight = anemll_attn.v_proj.weight.squeeze(-1).squeeze(-1).t()
            self.compare_tensors("V projection weight", hf_v_weight, anemll_v_weight)
            
            # Compare Q, K, V projections
            print("\n--- Comparing Q, K, V Projections ---")
            
            # HF projections
            hf_q = hf_attn.q_proj(hf_embeds)
            hf_k = hf_attn.k_proj(hf_embeds)
            hf_v = hf_attn.v_proj(hf_embeds)
            
            # ANEMLL projections (need to reshape for Conv2d)
            anemll_embeds_conv = anemll_embeds.permute(0, 2, 1).unsqueeze(2)  # [B, hidden, 1, seq]
            anemll_q = anemll_attn.q_proj(anemll_embeds_conv)
            anemll_k = anemll_attn.k_proj(anemll_embeds_conv)
            anemll_v = anemll_attn.v_proj(anemll_embeds_conv)
            
            # Reshape ANEMLL outputs back
            anemll_q = anemll_q.squeeze(2).permute(0, 2, 1)  # [B, seq, hidden]
            anemll_k = anemll_k.squeeze(2).permute(0, 2, 1)
            anemll_v = anemll_v.squeeze(2).permute(0, 2, 1)
            
            self.compare_tensors("Q projection", hf_q, anemll_q)
            self.compare_tensors("K projection", hf_k, anemll_k)
            self.compare_tensors("V projection", hf_v, anemll_v)
            
            # Compare shapes after reshaping
            print("\n--- Comparing Reshaped Q, K, V ---")
            
            # HF reshaping
            bsz, seq_len, _ = hf_embeds.shape
            hf_q_reshaped = hf_q.view(bsz, seq_len, hf_attn.num_heads, hf_attn.head_dim).transpose(1, 2)
            hf_k_reshaped = hf_k.view(bsz, seq_len, hf_attn.num_key_value_heads, hf_attn.head_dim).transpose(1, 2)
            hf_v_reshaped = hf_v.view(bsz, seq_len, hf_attn.num_key_value_heads, hf_attn.head_dim).transpose(1, 2)
            
            print(f"HF Q reshaped: {hf_q_reshaped.shape}")
            print(f"HF K reshaped: {hf_k_reshaped.shape}")
            print(f"HF V reshaped: {hf_v_reshaped.shape}")
            
            # ANEMLL reshaping (from their forward method)
            anemll_q_reshaped = anemll_q.view(1, anemll_attn.num_heads, seq_len, anemll_attn.head_dim)
            anemll_k_reshaped = anemll_k.view(1, anemll_attn.num_kv_heads, seq_len, anemll_attn.head_dim)
            anemll_v_reshaped = anemll_v.view(1, anemll_attn.num_kv_heads, seq_len, anemll_attn.head_dim)
            
            print(f"ANEMLL Q reshaped: {anemll_q_reshaped.shape}")
            print(f"ANEMLL K reshaped: {anemll_k_reshaped.shape}")
            print(f"ANEMLL V reshaped: {anemll_v_reshaped.shape}")
    
    def test_mlp_components(self, input_ids):
        """Compare MLP components."""
        print("\n" + "="*60)
        print("4. MLP COMPONENTS COMPARISON")
        print("="*60)
        
        # Get first layer input (after attention + residual)
        hf_embeds, anemll_embeds, _ = self.test_embeddings(input_ids)
        
        # For simplicity, just test MLP on embeddings
        hf_mlp = self.hf_model.model.layers[0].mlp
        anemll_mlp = self.anemll_model.model.layers[0].mlp
        
        with torch.no_grad():
            print("\n--- Comparing MLP Weights ---")
            
            # Gate projection
            hf_gate_weight = hf_mlp.gate_proj.weight
            anemll_gate_weight = anemll_mlp.gate_proj.weight.squeeze(-1).squeeze(-1).t()
            self.compare_tensors("Gate projection weight", hf_gate_weight, anemll_gate_weight)
            
            # Up projection
            hf_up_weight = hf_mlp.up_proj.weight
            anemll_up_weight = anemll_mlp.up_proj.weight.squeeze(-1).squeeze(-1).t()
            self.compare_tensors("Up projection weight", hf_up_weight, anemll_up_weight)
            
            # Down projection
            hf_down_weight = hf_mlp.down_proj.weight
            anemll_down_weight = anemll_mlp.down_proj.weight.squeeze(-1).squeeze(-1).t()
            self.compare_tensors("Down projection weight", hf_down_weight, anemll_down_weight)
            
            print("\n--- Comparing MLP Forward Pass ---")
            
            # HF forward
            hf_gate = hf_mlp.act_fn(hf_mlp.gate_proj(hf_embeds))
            hf_up = hf_mlp.up_proj(hf_embeds)
            hf_intermediate = hf_gate * hf_up
            hf_output = hf_mlp.down_proj(hf_intermediate)
            
            # ANEMLL forward (need Conv2d format)
            anemll_embeds_conv = anemll_embeds.permute(0, 2, 1).unsqueeze(2)
            anemll_gate = anemll_mlp.act(anemll_mlp.gate_proj(anemll_embeds_conv))
            anemll_up = anemll_mlp.up_proj(anemll_embeds_conv)
            anemll_intermediate = anemll_gate * anemll_up
            anemll_output = anemll_mlp.down_proj(anemll_intermediate)
            anemll_output = anemll_output.squeeze(2).permute(0, 2, 1)
            
            self.compare_tensors("MLP gate output", hf_gate, anemll_gate.squeeze(2).permute(0, 2, 1))
            self.compare_tensors("MLP up output", hf_up, anemll_up.squeeze(2).permute(0, 2, 1))
            self.compare_tensors("MLP intermediate", hf_intermediate, anemll_intermediate.squeeze(2).permute(0, 2, 1))
            self.compare_tensors("MLP final output", hf_output, anemll_output)
    
    def run_all_tests(self, text="Who are you?"):
        """Run all comparison tests."""
        print(f"\n{'='*80}")
        print(f"LAYER-BY-LAYER COMPARISON: ANEMLL vs Transformers")
        print(f"Model: {self.model_name}")
        print(f"Input text: '{text}'")
        print(f"{'='*80}")
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        print(f"Token IDs: {input_ids.tolist()}")
        
        # Run tests
        embed_match = self.test_embeddings(input_ids)[2]
        layer_match = self.test_first_layer(input_ids)[2]
        self.test_attention_components(input_ids)
        self.test_mlp_components(input_ids)
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Embeddings match: {'✓' if embed_match else '✗'}")
        print(f"First layer match: {'✓' if layer_match else '✗'}")


def main():
    # Disable quantization for fair comparison
    os.environ['SKIP_SP_FORWARD'] = '1'
    print("Note: Running with SKIP_SP_FORWARD=1 (no quantization scaling)")
    
    # Create comparator and run tests
    comparator = ModelComparator()
    comparator.run_all_tests()


if __name__ == "__main__":
    main()
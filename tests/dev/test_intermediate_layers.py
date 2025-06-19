#!/usr/bin/env python3
"""Compare intermediate layer outputs between PyTorch and CoreML models."""

import torch
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
import sys
import glob
import os
from pathlib import Path

# Add our models to path
sys.path.append('.')
from anemll.models.qwen_model import QwenForCausalLM, QwenConfig

class PyTorchModelWithHooks:
    """Wrapper to capture intermediate outputs from PyTorch model."""
    
    def __init__(self, model):
        self.model = model
        self.intermediate_outputs = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate outputs."""
        
        def make_hook(name):
            def hook(module, input, output):
                # Store the output
                if isinstance(output, torch.Tensor):
                    self.intermediate_outputs[name] = output.detach().cpu().numpy()
                elif isinstance(output, (tuple, list)):
                    self.intermediate_outputs[name] = [o.detach().cpu().numpy() if isinstance(o, torch.Tensor) else o for o in output]
            return hook
        
        # Hook key layers
        self.hooks.append(self.model.model.embed_tokens.register_forward_hook(make_hook("embeddings")))
        
        # Hook first few transformer layers
        for i in range(min(3, len(self.model.model.layers))):
            layer = self.model.model.layers[i]
            self.hooks.append(layer.register_forward_hook(make_hook(f"layer_{i}")))
            self.hooks.append(layer.input_layernorm.register_forward_hook(make_hook(f"layer_{i}_input_norm")))
            self.hooks.append(layer.self_attn.register_forward_hook(make_hook(f"layer_{i}_attention")))
            self.hooks.append(layer.post_attention_layernorm.register_forward_hook(make_hook(f"layer_{i}_post_attn_norm")))
            self.hooks.append(layer.mlp.register_forward_hook(make_hook(f"layer_{i}_mlp")))
        
        # Hook final norm and lm_head
        self.hooks.append(self.model.model.norm.register_forward_hook(make_hook("final_norm")))
        
        # Hook individual lm_head parts
        for i in range(1, 17):
            lm_head = getattr(self.model, f'lm_head16_{i}')
            self.hooks.append(lm_head.register_forward_hook(make_hook(f"lm_head16_{i}")))
    
    def __call__(self, *args, **kwargs):
        self.intermediate_outputs.clear()
        return self.model(*args, **kwargs)
    
    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()

def test_intermediate_layers():
    """Compare intermediate layer outputs to identify where divergence occurs."""
    
    print("🔍 Intermediate Layer Analysis: PyTorch vs CoreML")
    print("="*70)
    
    # Load tokenizer
    model_path = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    model_dirs = glob.glob(os.path.expanduser(model_path + "*"))
    if not model_dirs:
        print("❌ Error: Qwen model not found in cache")
        return False
    
    model_dir = model_dirs[0]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    
    # Load PyTorch model with hooks
    print("Loading PyTorch model...")
    config = QwenConfig.from_json(f"{model_dir}/config.json")
    pytorch_model_raw = QwenForCausalLM(config)
    pytorch_model_raw.load_pretrained_weights(model_dir)
    pytorch_model_raw.eval()
    pytorch_model = PyTorchModelWithHooks(pytorch_model_raw)
    
    # Load CoreML model  
    coreml_path = "../qwen-test/qwen_true_rmsnorm.mlpackage"  # Updated to use true RMSNorm model
    if not Path(coreml_path).exists():
        print(f"❌ Error: CoreML model not found at {coreml_path}")
        return False
    
    print("Loading CoreML model...")
    coreml_model = ct.models.MLModel(coreml_path)
    
    # Test inputs
    test_prompt = "What is Apple Neural Engine?"
    print(f"Test prompt: '{test_prompt}'")
    
    inputs = tokenizer(test_prompt, return_tensors="pt", add_special_tokens=True)
    seq_len = inputs.input_ids.shape[1]
    
    # Use single token for single-token model
    test_tokens = [inputs.input_ids[0, -1].item()]  # Take last token only
    
    print(f"Test tokens: {test_tokens}")
    
    # Prepare inputs for single-token model
    batch_input_ids = torch.tensor([test_tokens], dtype=torch.long)  # [1, 1]
    position_ids = torch.tensor([seq_len - 1], dtype=torch.long)  # [1] - position of the token
    current_pos = torch.tensor([seq_len - 1], dtype=torch.long)  # [1] 
    update_mask = torch.ones_like(batch_input_ids, dtype=torch.bool)
    
    # Single token causal mask
    causal_mask = torch.full((1, 1, 1, 256), -torch.inf, dtype=torch.float16)
    causal_mask[:, :, 0, :seq_len] = 0  # Allow attention to all previous tokens
    
    # Run PyTorch model
    print(f"\n🔥 Running PyTorch model with hooks...")
    with torch.no_grad():
        pytorch_logits = pytorch_model(
            input_ids=batch_input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=False
        )
    
    # Run CoreML model
    print(f"🍎 Running CoreML model...")
    coreml_input_ids = np.array([test_tokens], dtype=np.int32)  # [1, 1]
    coreml_position_ids = np.array([seq_len - 1], dtype=np.int32)  # [1]
    coreml_current_pos = np.array([seq_len - 1], dtype=np.int32)  # [1]
    coreml_causal_mask = np.full((1, 1, 1, 256), -np.inf, dtype=np.float16)  # [1, 1, 1, 256]
    coreml_causal_mask[:, :, 0, :seq_len] = 0  # Allow attention to all previous tokens
    
    coreml_inputs = {
        'input_ids': coreml_input_ids,
        'position_ids': coreml_position_ids,
        'causal_mask': coreml_causal_mask,
        'current_pos': coreml_current_pos
    }
    
    try:
        coreml_outputs = coreml_model.predict(coreml_inputs)
    except Exception as e:
        print(f"❌ CoreML inference failed: {e}")
        return False
    
    # Analyze intermediate outputs
    print(f"\n📊 INTERMEDIATE OUTPUT ANALYSIS:")
    print("-" * 50)
    
    # Check embeddings first
    if "embeddings" in pytorch_model.intermediate_outputs:
        pytorch_embeddings = pytorch_model.intermediate_outputs["embeddings"]
        print(f"\n🔤 Embeddings:")
        print(f"  PyTorch shape: {pytorch_embeddings.shape}")
        print(f"  PyTorch range: [{pytorch_embeddings.min():.3f}, {pytorch_embeddings.max():.3f}]")
        print(f"  PyTorch mean: {pytorch_embeddings.mean():.6f}")
        print(f"  PyTorch std: {pytorch_embeddings.std():.6f}")
    
    # Check layer normalization outputs
    for layer_idx in range(3):
        norm_key = f"layer_{layer_idx}_input_norm"
        if norm_key in pytorch_model.intermediate_outputs:
            pytorch_norm = pytorch_model.intermediate_outputs[norm_key]
            print(f"\n🔧 Layer {layer_idx} Input Norm:")
            print(f"  PyTorch shape: {pytorch_norm.shape}")
            print(f"  PyTorch range: [{pytorch_norm.min():.3f}, {pytorch_norm.max():.3f}]")
            print(f"  PyTorch mean: {pytorch_norm.mean():.6f}")
            print(f"  PyTorch std: {pytorch_norm.std():.6f}")
    
    # Check attention outputs
    for layer_idx in range(3):
        attn_key = f"layer_{layer_idx}_attention"
        if attn_key in pytorch_model.intermediate_outputs:
            pytorch_attn = pytorch_model.intermediate_outputs[attn_key]
            if isinstance(pytorch_attn, list) and len(pytorch_attn) > 0:
                pytorch_attn = pytorch_attn[0]  # Get hidden states
            print(f"\n👁️  Layer {layer_idx} Attention:")
            print(f"  PyTorch shape: {pytorch_attn.shape}")
            print(f"  PyTorch range: [{pytorch_attn.min():.3f}, {pytorch_attn.max():.3f}]")
            print(f"  PyTorch mean: {pytorch_attn.mean():.6f}")
            print(f"  PyTorch std: {pytorch_attn.std():.6f}")
    
    # Check MLP outputs
    for layer_idx in range(3):
        mlp_key = f"layer_{layer_idx}_mlp"
        if mlp_key in pytorch_model.intermediate_outputs:
            pytorch_mlp = pytorch_model.intermediate_outputs[mlp_key]
            print(f"\n🧠 Layer {layer_idx} MLP:")
            print(f"  PyTorch shape: {pytorch_mlp.shape}")
            print(f"  PyTorch range: [{pytorch_mlp.min():.3f}, {pytorch_mlp.max():.3f}]")
            print(f"  PyTorch mean: {pytorch_mlp.mean():.6f}")
            print(f"  PyTorch std: {pytorch_mlp.std():.6f}")
    
    # Check final norm
    if "final_norm" in pytorch_model.intermediate_outputs:
        pytorch_final_norm = pytorch_model.intermediate_outputs["final_norm"]
        print(f"\n🏁 Final Norm:")
        print(f"  PyTorch shape: {pytorch_final_norm.shape}")
        print(f"  PyTorch range: [{pytorch_final_norm.min():.3f}, {pytorch_final_norm.max():.3f}]")
        print(f"  PyTorch mean: {pytorch_final_norm.mean():.6f}")
        print(f"  PyTorch std: {pytorch_final_norm.std():.6f}")
    
    # Check individual lm_head outputs
    print(f"\n🎯 LM Head Outputs (16-way split):")
    pytorch_lm_head_parts = []
    for i in range(1, 17):
        lm_head_key = f"lm_head16_{i}"
        if lm_head_key in pytorch_model.intermediate_outputs:
            pytorch_lm_part = pytorch_model.intermediate_outputs[lm_head_key]
            pytorch_lm_head_parts.append(pytorch_lm_part)
            print(f"  PyTorch lm_head16_{i} shape: {pytorch_lm_part.shape}")
            print(f"  PyTorch lm_head16_{i} range: [{pytorch_lm_part.min():.3f}, {pytorch_lm_part.max():.3f}]")
    
    # Compare with CoreML lm_head outputs
    print(f"\n🍎 CoreML LM Head Outputs:")
    coreml_lm_head_parts = []
    for i in range(1, 17):
        coreml_key = f"logits{i}"
        if coreml_key in coreml_outputs:
            coreml_lm_part = coreml_outputs[coreml_key]
            coreml_lm_head_parts.append(coreml_lm_part)
            print(f"  CoreML logits{i} shape: {coreml_lm_part.shape}")
            print(f"  CoreML logits{i} range: [{coreml_lm_part.min():.3f}, {coreml_lm_part.max():.3f}]")
    
    # Compare lm_head parts
    if len(pytorch_lm_head_parts) == len(coreml_lm_head_parts) == 16:
        print(f"\n🔍 LM Head Parts Comparison:")
        for i in range(16):
            pytorch_part = pytorch_lm_head_parts[i]
            coreml_part = coreml_lm_head_parts[i]
            
            # Reshape if needed for comparison
            if pytorch_part.shape != coreml_part.shape:
                print(f"  Part {i+1}: Shape mismatch - PyTorch {pytorch_part.shape} vs CoreML {coreml_part.shape}")
                continue
            
            diff = np.abs(pytorch_part - coreml_part)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            print(f"  Part {i+1}: max_diff={max_diff:.3f}, mean_diff={mean_diff:.6f}")
            
            if max_diff > 1.0:
                print(f"    ⚠️  Large difference in part {i+1}!")
    
    # Final logits comparison
    pytorch_final_logits = pytorch_logits[0, -1, :].cpu().numpy()  # Last token's logits
    
    if coreml_lm_head_parts:
        # For single token: each part should be [1, 1, vocab_split]
        coreml_final_logits = np.concatenate(coreml_lm_head_parts, axis=-1)[0, 0, :]  # [vocab_size]
        
        logits_diff = np.abs(pytorch_final_logits - coreml_final_logits)
        max_diff = logits_diff.max()
        mean_diff = logits_diff.mean()
        
        print(f"\n🎯 FINAL LOGITS COMPARISON:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        
        # Top tokens
        pytorch_top_idx = np.argmax(pytorch_final_logits)
        coreml_top_idx = np.argmax(coreml_final_logits)
        
        print(f"  PyTorch top token: {pytorch_top_idx} ('{tokenizer.decode([pytorch_top_idx])}')")
        print(f"  CoreML top token: {coreml_top_idx} ('{tokenizer.decode([coreml_top_idx])}')")
        print(f"  Tokens match: {pytorch_top_idx == coreml_top_idx}")
    
    # Cleanup
    pytorch_model.cleanup()
    
    return True

if __name__ == "__main__":
    success = test_intermediate_layers()
    if success:
        print("\n✅ Analysis completed!")
    else:
        print("\n❌ Analysis failed!")
        sys.exit(1) 
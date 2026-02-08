#!/usr/bin/env python3
"""Compare layer-by-layer outputs between HF and our implementation for Qwen3-1.7B."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

MODEL_ID = "Qwen/Qwen3-1.7B"
TOKENIZER_PATH = "/Volumes/Models/ANE/qwen3_1.7b_ctx2048_FP16"

def compare_embeddings():
    """Compare embeddings between HF and our model."""
    print("\n" + "=" * 70)
    print("Comparing Embeddings")
    print("=" * 70)

    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, local_files_only=True)
    hf_model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

    # Test input
    prompt = "Hello"
    tokens = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    print(f"Tokens: {tokens.tolist()}")

    # HF embeddings
    with torch.no_grad():
        hf_embeds = hf_model.model.embed_tokens(tokens)
    print(f"HF embeddings shape: {hf_embeds.shape}")
    print(f"HF embeddings[0,0,:5]: {hf_embeds[0,0,:5].tolist()}")

    # Our model
    from anemll.models.qwen_model import QwenConfig, QwenForCausalLM, MODEL_DTYPE

    # Find model path
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots")
    snapshot_dirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    model_path = os.path.join(cache_dir, snapshot_dirs[0])

    hf_config = hf_model.config
    qwen_config = QwenConfig(
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        head_dim=getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads),
        num_hidden_layers=hf_config.num_hidden_layers,
        intermediate_size=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=hf_config.rope_theta,
        context_length=256,
        state_length=256,
    )

    our_model = QwenForCausalLM(qwen_config, enable_coreml=False, disable_kv_cache=True)
    our_model.load_pretrained_weights(model_path)
    our_model.eval()

    # Our embeddings
    with torch.no_grad():
        our_embeds = our_model.model.embed_tokens(tokens)
    print(f"Our embeddings shape: {our_embeds.shape}")
    print(f"Our embeddings[0,0,:5]: {our_embeds[0,0,:5].tolist()}")

    # Compare
    embed_diff = (hf_embeds.float() - our_embeds.float()).abs()
    print(f"Embedding diff max: {embed_diff.max().item():.6f}")
    print(f"Embedding diff mean: {embed_diff.mean().item():.6f}")

    if embed_diff.max() > 0.01:
        print("WARNING: Embeddings differ significantly!")
    else:
        print("OK: Embeddings match")

    return hf_model, our_model, tokens

def compare_first_layer(hf_model, our_model, tokens):
    """Compare first layer output."""
    print("\n" + "=" * 70)
    print("Comparing First Layer (RMSNorm only)")
    print("=" * 70)

    with torch.no_grad():
        # HF: Get embeddings
        hf_embeds = hf_model.model.embed_tokens(tokens)
        hf_layer = hf_model.model.layers[0]

        # Our model
        our_embeds = our_model.model.embed_tokens(tokens)

        # RMSNorm comparison
        our_normed = our_model.model.layers[0].input_layernorm(our_embeds)
        hf_normed = hf_layer.input_layernorm(hf_embeds)
        norm_diff = (hf_normed.float() - our_normed.float()).abs()
        print(f"Input LayerNorm diff max: {norm_diff.max().item():.6f}")
        print(f"Input LayerNorm diff mean: {norm_diff.mean().item():.6f}")
        print(f"HF normed[0,0,:5]: {hf_normed[0,0,:5].tolist()}")
        print(f"Our normed[0,0,:5]: {our_normed[0,0,:5].tolist()}")

        if norm_diff.max() > 0.01:
            print("WARNING: RMSNorm outputs differ significantly!")
        else:
            print("OK: RMSNorm outputs match")

def compare_qkv_projections(hf_model, our_model, tokens):
    """Compare Q/K/V projections in layer 0."""
    print("\n" + "=" * 70)
    print("Comparing Q/K/V Projections (Layer 0)")
    print("=" * 70)

    with torch.no_grad():
        # Get embeddings
        hf_embeds = hf_model.model.embed_tokens(tokens)
        our_embeds = our_model.model.embed_tokens(tokens)

        # Get normalized embeddings
        hf_hidden = hf_model.model.layers[0].input_layernorm(hf_embeds)
        our_hidden = our_model.model.layers[0].input_layernorm(our_embeds)

        hf_attn = hf_model.model.layers[0].self_attn
        our_attn = our_model.model.layers[0].self_attn

        # HF Q projection
        hf_q = hf_attn.q_proj(hf_hidden)
        print(f"HF Q shape: {hf_q.shape}")
        print(f"HF Q[0,0,:5]: {hf_q[0,0,:5].tolist()}")

        # Our Q projection (need to reshape for Conv2d)
        our_hidden_reshaped = our_hidden.permute(0, 2, 1).unsqueeze(2).to(torch.float16)
        our_q = our_attn.q_proj(our_hidden_reshaped)
        our_q_flat = our_q.squeeze(2).transpose(1, 2)  # Back to [B, S, D]
        print(f"Our Q shape after projection: {our_q_flat.shape}")
        print(f"Our Q[0,0,:5]: {our_q_flat[0,0,:5].tolist()}")

        q_diff = (hf_q.float() - our_q_flat.float()).abs()
        print(f"Q diff max: {q_diff.max().item():.6f}")
        print(f"Q diff mean: {q_diff.mean().item():.6f}")

        if q_diff.max() > 0.1:
            print("WARNING: Q projections differ!")
            # Check weights
            print(f"\nHF q_proj weight shape: {hf_attn.q_proj.weight.shape}")
            print(f"Our q_proj weight shape: {our_attn.q_proj.weight.shape}")
            print(f"HF q_proj weight[:5,:5]:\n{hf_attn.q_proj.weight[:5,:5].tolist()}")
            print(f"Our q_proj weight[0:5,0:5,0,0]:\n{our_attn.q_proj.weight[:5,:5,0,0].tolist()}")

            weight_diff = (hf_attn.q_proj.weight.float() - our_attn.q_proj.weight[:,:,0,0].float()).abs()
            print(f"Q weight diff max: {weight_diff.max().item():.6f}")
        else:
            print("OK: Q projections match")

def compare_full_forward(hf_model, our_model, tokens):
    """Compare full forward pass outputs."""
    print("\n" + "=" * 70)
    print("Comparing Full Forward Pass")
    print("=" * 70)

    from anemll.models.qwen_model import MODEL_DTYPE

    seq_len = tokens.shape[1]

    with torch.no_grad():
        # HF forward
        hf_outputs = hf_model(tokens, use_cache=False)
        hf_logits = hf_outputs.logits
        print(f"HF logits shape: {hf_logits.shape}")
        print(f"HF logits[0,-1,:10]: {hf_logits[0,-1,:10].tolist()}")

        # Our forward
        position_ids = torch.arange(seq_len, dtype=torch.int32)
        context_length = 256
        causal_mask = torch.zeros((1, 1, seq_len, context_length), dtype=MODEL_DTYPE)
        for i in range(seq_len):
            for j in range(i + 1, context_length):
                causal_mask[0, 0, i, j] = float('-inf')
        update_mask = torch.zeros((1, 1, context_length, 1), dtype=MODEL_DTYPE)
        current_pos = torch.tensor([seq_len - 1], dtype=torch.int32)

        tokens_int32 = tokens.to(torch.int32)
        our_outputs = our_model(
            input_ids=tokens_int32,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
            IN_PREFILL=True,
        )

        if isinstance(our_outputs, tuple):
            our_logits = torch.cat(our_outputs, dim=-1)
        else:
            our_logits = our_outputs

        print(f"Our logits shape: {our_logits.shape}")
        print(f"Our logits[0,-1,:10]: {our_logits[0,-1,:10].tolist()}")

        # Compare last position logits
        logits_diff = (hf_logits[0, -1, :].float() - our_logits[0, -1, :].float()).abs()
        print(f"\nLogits diff at last position:")
        print(f"  max: {logits_diff.max().item():.4f}")
        print(f"  mean: {logits_diff.mean().item():.4f}")

        # Top-5 predictions
        hf_probs = torch.softmax(hf_logits[0, -1, :].float(), dim=-1)
        our_probs = torch.softmax(our_logits[0, -1, :].float(), dim=-1)

        _, hf_top5 = torch.topk(hf_probs, 5)
        _, our_top5 = torch.topk(our_probs, 5)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/Volumes/Models/ANE/qwen3_1.7b_ctx2048_FP16", local_files_only=True)

        print(f"\nHF top-5: {[tokenizer.decode([t.item()]) for t in hf_top5]}")
        print(f"Our top-5: {[tokenizer.decode([t.item()]) for t in our_top5]}")

def main():
    hf_model, our_model, tokens = compare_embeddings()
    compare_first_layer(hf_model, our_model, tokens)
    compare_qkv_projections(hf_model, our_model, tokens)
    compare_full_forward(hf_model, our_model, tokens)

if __name__ == "__main__":
    main()

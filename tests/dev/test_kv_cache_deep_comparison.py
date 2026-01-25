#!/usr/bin/env python3
"""Deep comparison of KV cache between PyTorch and CoreML.

This test compares the actual KV cache values after each token to find
exactly where divergence occurs.
"""

import sys
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

import os
import torch
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

from anemll.models.gemma3_model import (
    Gemma3ForCausalLM,
    Gemma3Config,
    MODEL_DTYPE,
    TEST_DEVICE,
)
from anemll.ane_converter.gemma3_converter import Gemma3Converter


def test_kv_cache_comparison():
    """Compare KV cache values token by token."""

    hf_model_path = '/Users/anemll/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3'
    coreml_dir = '/tmp/gemma3_test3'
    context_length = 64

    print("Loading PyTorch model...")
    config = Gemma3Config.from_json(f'{hf_model_path}/config.json')
    config.context_length = context_length
    config.state_length = context_length

    pt_model = Gemma3ForCausalLM(config)
    converter = Gemma3Converter(pt_model, context_length=context_length, batch_size=context_length)
    converter.load_weights_from_hf(hf_model_path)
    pt_model.eval()

    print("Loading CoreML models...")
    embed_model = ct.models.MLModel(f'{coreml_dir}/gemma3_270m_embeddings.mlpackage')
    ffn_model = ct.models.MLModel(f'{coreml_dir}/gemma3_270m_FFN_chunk_01of01.mlpackage')
    lmhead_model = ct.models.MLModel(f'{coreml_dir}/gemma3_270m_lm_head.mlpackage')

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)

    # Simple test sequence
    tokens = [2, 105, 3]  # BOS, some token, some token
    print(f"\nTest tokens: {tokens}")

    # Initialize
    pt_model.model.kv_cache_0.zero_()
    cm_state = ffn_model.make_state()

    # Clear CoreML state
    initial_kv = cm_state.read_state("model_model_kv_cache_0")
    cm_state.write_state("model_model_kv_cache_0", np.zeros_like(initial_kv))

    print("\n" + "="*80)
    print("TOKEN BY TOKEN COMPARISON")
    print("="*80)

    for pos, token_id in enumerate(tokens):
        print(f"\n--- Token {pos}: ID={token_id} ---")

        # Create input
        input_ids = torch.tensor([[token_id]], dtype=torch.long)

        # Get embeddings
        with torch.no_grad():
            pt_embed = pt_model.model.embed_tokens(input_ids)
            pt_hidden = (pt_embed * pt_model.model.embedding_scale).to(MODEL_DTYPE)

        cm_embed = embed_model.predict({
            'input_ids': input_ids.numpy().astype(np.int32)
        })['hidden_states']

        # Compare embeddings
        embed_diff = np.abs(pt_hidden.numpy() - cm_embed).max()
        print(f"  Embedding diff: {embed_diff:.6f}")

        # Create inputs for FFN
        position_ids = torch.tensor([pos], dtype=torch.int32)
        causal_mask = torch.zeros((1, 1, 1, context_length), dtype=MODEL_DTYPE)
        causal_mask[0, 0, 0, pos+1:] = float('-inf')
        current_pos = torch.tensor([pos], dtype=torch.int32)

        # Run PyTorch FFN
        with torch.no_grad():
            pt_out = pt_model.model.process_layers(
                pt_hidden,
                position_ids,
                causal_mask,
                current_pos,
                start_layer=0,
                end_layer=None,
                IN_PREFILL=False,
            )
            pt_out = pt_model.model.norm(pt_out)

        # Run CoreML FFN
        cm_out = ffn_model.predict({
            'hidden_states': cm_embed.astype(np.float16),
            'position_ids': position_ids.numpy().astype(np.int32),
            'causal_mask': causal_mask.numpy().astype(np.float16),
            'current_pos': current_pos.numpy().astype(np.int32),
        }, cm_state)['output_hidden_states']

        # Compare outputs
        out_diff = np.abs(pt_out.numpy() - cm_out).max()
        out_mean_pt = float(pt_out.float().mean())
        out_mean_cm = float(cm_out.mean())
        print(f"  Output diff: {out_diff:.4f}")
        print(f"    PT mean: {out_mean_pt:.4f}, CM mean: {out_mean_cm:.4f}")

        # Compare KV cache values at this position
        pt_kv = pt_model.model.kv_cache_0.numpy()
        cm_kv = cm_state.read_state("model_model_kv_cache_0")

        # Compare layer by layer at this position
        num_layers = config.num_hidden_layers
        print(f"\n  KV cache at position {pos} by layer:")

        for layer_idx in range(min(3, num_layers)):  # First 3 layers
            k_idx = layer_idx
            v_idx = layer_idx + num_layers

            pt_k = pt_kv[k_idx, :, pos, :]
            cm_k = cm_kv[k_idx, :, pos, :]
            pt_v = pt_kv[v_idx, :, pos, :]
            cm_v = cm_kv[v_idx, :, pos, :]

            k_diff = np.abs(pt_k - cm_k).max()
            v_diff = np.abs(pt_v - cm_v).max()

            print(f"    Layer {layer_idx}: K diff={k_diff:.4f}, V diff={v_diff:.4f}")

            if k_diff > 0.1 or v_diff > 0.1:
                print(f"      PT K: mean={pt_k.mean():.4f}, std={pt_k.std():.4f}")
                print(f"      CM K: mean={cm_k.mean():.4f}, std={cm_k.std():.4f}")
                print(f"      PT V: mean={pt_v.mean():.4f}, std={pt_v.std():.4f}")
                print(f"      CM V: mean={cm_v.mean():.4f}, std={cm_v.std():.4f}")

        # Get predicted tokens
        pt_logits = get_logits(pt_model, pt_out)
        pt_token = pt_logits.argmax(dim=-1).item()

        cm_lmhead_out = lmhead_model.predict({
            'hidden_states': cm_out.astype(np.float16)
        })
        cm_logits = combine_logits(cm_lmhead_out)
        cm_token = cm_logits.argmax()

        print(f"\n  Predictions:")
        print(f"    PT: {pt_token} -> '{tokenizer.decode([pt_token])}'")
        print(f"    CM: {cm_token} -> '{tokenizer.decode([cm_token])}'")
        print(f"    Match: {pt_token == cm_token}")

    # Final full KV cache comparison
    print("\n" + "="*80)
    print("FULL KV CACHE COMPARISON")
    print("="*80)

    pt_kv = pt_model.model.kv_cache_0.numpy()
    cm_kv = cm_state.read_state("model_model_kv_cache_0")

    full_diff = np.abs(pt_kv - cm_kv)
    print(f"\nOverall KV cache difference:")
    print(f"  Max: {full_diff.max():.4f}")
    print(f"  Mean: {full_diff.mean():.6f}")

    # Check filled positions only
    filled_mask = (np.abs(pt_kv) > 0.001) | (np.abs(cm_kv) > 0.001)
    if filled_mask.sum() > 0:
        filled_diff = full_diff[filled_mask]
        print(f"  Filled positions max diff: {filled_diff.max():.4f}")
        print(f"  Filled positions mean diff: {filled_diff.mean():.6f}")


def get_logits(model, hidden_states):
    """Get logits from hidden states using split LM head."""
    hidden_for_lm = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

    all_logits = []
    for j in range(1, 17):
        lm_head = getattr(model, f'lm_head16_{j}')
        part_logits = lm_head(hidden_for_lm).squeeze(2).transpose(1, 2)
        all_logits.append(part_logits)

    logits = torch.cat(all_logits, dim=-1).squeeze(1)
    return logits


def combine_logits(lmhead_out):
    """Combine logits from split LM head output."""
    parts = []
    for i in range(1, 17):
        key = f'logits{i}'  # Note: no underscore
        if key in lmhead_out:
            parts.append(lmhead_out[key])
    return np.concatenate(parts, axis=-1).squeeze()


if __name__ == "__main__":
    test_kv_cache_comparison()

#!/usr/bin/env python3
"""Token-by-token comparison of CoreML vs PyTorch Gemma3 inference.

This test compares the outputs at each step to identify where divergence occurs.
"""

import sys
sys.path.insert(0, '/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll')

import os
import torch
import numpy as np
from transformers import AutoTokenizer
import coremltools as ct

from anemll.models.gemma3_model import (
    Gemma3ForCausalLM,
    Gemma3Config,
    MODEL_DTYPE,
    TEST_DEVICE,
)
from anemll.ane_converter.gemma3_converter import Gemma3Converter


def load_pytorch_model(model_path, context_length=64):
    """Load ANEMLL PyTorch model."""
    config = Gemma3Config.from_json(f'{model_path}/config.json')
    config.context_length = context_length
    config.state_length = context_length

    model = Gemma3ForCausalLM(config)
    converter = Gemma3Converter(model, context_length=context_length, batch_size=context_length)
    converter.load_weights_from_hf(model_path)
    model.eval()

    return model


def load_coreml_models(model_dir):
    """Load CoreML models."""
    models = {}

    # Load embeddings
    embed_path = f'{model_dir}/gemma3_270m_embeddings.mlpackage'
    if os.path.exists(embed_path):
        models['embeddings'] = ct.models.MLModel(embed_path)
        print(f"Loaded embeddings from {embed_path}")

    # Load LM head
    lmhead_path = f'{model_dir}/gemma3_270m_lm_head.mlpackage'
    if os.path.exists(lmhead_path):
        models['lmhead'] = ct.models.MLModel(lmhead_path)
        print(f"Loaded LM head from {lmhead_path}")

    # Load FFN (inference)
    ffn_path = f'{model_dir}/gemma3_270m_FFN_chunk_01of01.mlpackage'
    if os.path.exists(ffn_path):
        models['ffn'] = ct.models.MLModel(ffn_path)
        print(f"Loaded FFN from {ffn_path}")

    # Load prefill
    prefill_path = f'{model_dir}/gemma3_270m_prefill_chunk_01of01.mlpackage'
    if os.path.exists(prefill_path):
        models['prefill'] = ct.models.MLModel(prefill_path)
        print(f"Loaded prefill from {prefill_path}")

    return models


def get_pytorch_logits(model, hidden_states):
    """Get logits from hidden states using split LM head."""
    hidden_for_lm = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

    all_logits = []
    for j in range(1, 17):
        lm_head = getattr(model, f'lm_head16_{j}')
        part_logits = lm_head(hidden_for_lm).squeeze(2).transpose(1, 2)
        all_logits.append(part_logits)

    logits = torch.cat(all_logits, dim=-1).squeeze(1)
    return logits


def compare_arrays(name, pytorch_arr, coreml_arr, threshold=1.0):
    """Compare PyTorch and CoreML arrays."""
    pt = pytorch_arr.float().detach().numpy() if torch.is_tensor(pytorch_arr) else pytorch_arr
    cm = coreml_arr if isinstance(coreml_arr, np.ndarray) else np.array(coreml_arr)

    # Flatten for comparison
    pt_flat = pt.flatten()
    cm_flat = cm.flatten()

    if pt_flat.shape != cm_flat.shape:
        print(f"  {name}: SHAPE MISMATCH PT={pt.shape} vs CM={cm.shape}")
        return False

    diff = np.abs(pt_flat - cm_flat)
    max_diff = diff.max()
    mean_diff = diff.mean()

    status = "OK" if max_diff < threshold else "DIVERGED"
    print(f"  {name}: max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f} [{status}]")
    print(f"    PT:  mean={pt_flat.mean():.4f}, std={pt_flat.std():.4f}, max={np.abs(pt_flat).max():.4f}")
    print(f"    CM:  mean={cm_flat.mean():.4f}, std={cm_flat.std():.4f}, max={np.abs(cm_flat).max():.4f}")

    return max_diff < threshold


def test_single_token_no_prefill():
    """Test single token inference without prefill - just embedding + FFN."""
    hf_model_path = '/Users/anemll/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3'
    coreml_dir = '/tmp/gemma3_test3'  # Use existing converted models
    context_length = 64

    print("Loading PyTorch model...")
    pt_model = load_pytorch_model(hf_model_path, context_length)

    print("\nLoading CoreML models...")
    cm_models = load_coreml_models(coreml_dir)

    if 'embeddings' not in cm_models or 'ffn' not in cm_models:
        print("ERROR: Required CoreML models not found!")
        return

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)

    # Test with single BOS token
    input_ids = torch.tensor([[2]], dtype=torch.long)  # BOS token

    print(f"\n{'='*60}")
    print("TEST: Single BOS Token Embedding")
    print(f"{'='*60}")

    # PyTorch embeddings
    with torch.no_grad():
        pt_embed = pt_model.model.embed_tokens(input_ids)
        pt_embed_scaled = pt_embed * pt_model.model.embedding_scale
        pt_embed_scaled = pt_embed_scaled.to(MODEL_DTYPE)

    # CoreML embeddings
    cm_embed_out = cm_models['embeddings'].predict({
        'input_ids': input_ids.numpy().astype(np.int32)
    })
    cm_embed = cm_embed_out['hidden_states']

    print("\n--- Embedding Comparison ---")
    compare_arrays("embeddings", pt_embed_scaled, cm_embed, threshold=0.1)

    # Now test FFN with this single token
    print(f"\n{'='*60}")
    print("TEST: Single Token FFN (no KV cache)")
    print(f"{'='*60}")

    # Reset KV cache in PyTorch
    pt_model.model.kv_cache_0.zero_()

    # Create inputs for FFN
    position_ids = torch.tensor([0], dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, 1, context_length), dtype=MODEL_DTYPE)
    # Mask future positions
    for j in range(1, context_length):
        causal_mask[0, 0, 0, j] = float('-inf')
    current_pos = torch.tensor([0], dtype=torch.int32)

    # Create FFNWrapper-like forward in PyTorch
    with torch.no_grad():
        pt_hidden = pt_embed_scaled
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

    # CoreML FFN - need to get state first
    print("\nCreating CoreML state...")

    # Get state shape from model
    state = cm_models['ffn'].make_state()

    # Run CoreML FFN
    cm_ffn_out = cm_models['ffn'].predict({
        'hidden_states': cm_embed.astype(np.float16),
        'position_ids': position_ids.numpy().astype(np.int32),
        'causal_mask': causal_mask.numpy().astype(np.float16),
        'current_pos': current_pos.numpy().astype(np.int32),
    }, state)
    cm_out = cm_ffn_out['output_hidden_states']

    print("\n--- FFN Output Comparison ---")
    compare_arrays("ffn_output", pt_out, cm_out, threshold=10.0)

    # Get logits and compare predicted token
    print("\n--- Token Prediction ---")
    pt_logits = get_pytorch_logits(pt_model, pt_out)
    pt_token = pt_logits.argmax(dim=-1).item()

    # CoreML logits
    cm_lmhead_out = cm_models['lmhead'].predict({
        'hidden_states': cm_out.astype(np.float16)
    })

    # Combine logits from split LM head
    cm_logits_parts = []
    for i in range(1, 17):
        key = f'logits_{i}'
        if key in cm_lmhead_out:
            cm_logits_parts.append(cm_lmhead_out[key])

    if cm_logits_parts:
        cm_logits = np.concatenate(cm_logits_parts, axis=-1).squeeze()
        cm_token = cm_logits.argmax()

        print(f"  PT predicted: {pt_token} -> '{tokenizer.decode([pt_token])}'")
        print(f"  CM predicted: {cm_token} -> '{tokenizer.decode([cm_token])}'")
        print(f"  Match: {pt_token == cm_token}")

        compare_arrays("logits", pt_logits.squeeze(), cm_logits, threshold=100.0)


def test_prefill_vs_ffn():
    """Compare prefill output with sequential FFN calls."""
    hf_model_path = '/Users/anemll/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3'
    coreml_dir = '/tmp/gemma3_test3'
    context_length = 64
    batch_size = 64

    print("Loading PyTorch model...")
    pt_model = load_pytorch_model(hf_model_path, context_length)

    print("\nLoading CoreML models...")
    cm_models = load_coreml_models(coreml_dir)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)

    # Use chat template
    messages = [{"role": "user", "content": "Hello"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    seq_len = input_ids.shape[1]

    print(f"\nInput: {seq_len} tokens")
    print(f"Tokens: {input_ids[0].tolist()}")

    # Reset KV caches
    pt_model.model.kv_cache_0.zero_()

    # Get embeddings
    with torch.no_grad():
        pt_embed = pt_model.model.embed_tokens(input_ids)
        pt_hidden = pt_embed * pt_model.model.embedding_scale
        pt_hidden = pt_hidden.to(MODEL_DTYPE)

    # Pad to batch_size for prefill
    pt_hidden_padded = torch.nn.functional.pad(
        pt_hidden, (0, 0, 0, batch_size - seq_len), value=0
    )

    # Create prefill inputs
    position_ids = torch.arange(batch_size, dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, batch_size, context_length), dtype=MODEL_DTYPE)
    for i in range(batch_size):
        for j in range(i + 1, context_length):
            causal_mask[0, 0, i, j] = float('-inf')
    current_pos = torch.tensor([0], dtype=torch.int32)

    print(f"\n{'='*60}")
    print("TEST: PyTorch Prefill")
    print(f"{'='*60}")

    with torch.no_grad():
        pt_prefill_out = pt_model.model.process_layers(
            pt_hidden_padded,
            position_ids,
            causal_mask,
            current_pos,
            start_layer=0,
            end_layer=None,
            IN_PREFILL=True,
        )

    print(f"PyTorch prefill output shape: {pt_prefill_out.shape}")
    print(f"  mean={pt_prefill_out.float().mean():.4f}, std={pt_prefill_out.float().std():.4f}")

    # Check KV cache
    print(f"\nKV cache after prefill:")
    kv = pt_model.model.kv_cache_0
    print(f"  shape={kv.shape}")
    print(f"  non-zero: {(kv != 0).sum().item()}")

    # Now run single token inference at the last position
    print(f"\n{'='*60}")
    print("TEST: Single Token After Prefill")
    print(f"{'='*60}")

    # Get embedding for last token
    last_embed = pt_model.model.embed_tokens(input_ids[:, -1:])
    last_hidden = last_embed * pt_model.model.embedding_scale
    last_hidden = last_hidden.to(MODEL_DTYPE)

    pos = torch.tensor([seq_len - 1], dtype=torch.int32)
    single_mask = torch.zeros((1, 1, 1, context_length), dtype=MODEL_DTYPE)
    for j in range(seq_len, context_length):
        single_mask[0, 0, 0, j] = float('-inf')
    current = torch.tensor([seq_len - 1], dtype=torch.int32)

    with torch.no_grad():
        pt_ffn_out = pt_model.model.process_layers(
            last_hidden,
            pos,
            single_mask,
            current,
            start_layer=0,
            end_layer=None,
            IN_PREFILL=False,
        )
        pt_ffn_out_normed = pt_model.model.norm(pt_ffn_out)

    print(f"PyTorch FFN output: shape={pt_ffn_out_normed.shape}")
    print(f"  mean={pt_ffn_out_normed.float().mean():.4f}, std={pt_ffn_out_normed.float().std():.4f}")

    # Get logits and predict
    pt_logits = get_pytorch_logits(pt_model, pt_ffn_out_normed)
    pt_token = pt_logits.argmax(dim=-1).item()
    print(f"\nPyTorch predicted: {pt_token} -> '{tokenizer.decode([pt_token])}'")

    # Now do the same with CoreML
    if 'prefill' in cm_models and 'ffn' in cm_models:
        print(f"\n{'='*60}")
        print("TEST: CoreML Prefill + FFN")
        print(f"{'='*60}")

        # Get CoreML embeddings
        cm_embed_out = cm_models['embeddings'].predict({
            'input_ids': input_ids.numpy().astype(np.int32)
        })
        cm_hidden = cm_embed_out['hidden_states']

        # Pad for prefill
        cm_hidden_padded = np.pad(
            cm_hidden, ((0, 0), (0, batch_size - seq_len), (0, 0)),
            mode='constant', constant_values=0
        )

        # Create state
        state = cm_models['prefill'].make_state()

        # Run prefill
        cm_prefill_out = cm_models['prefill'].predict({
            'hidden_states': cm_hidden_padded.astype(np.float16),
            'position_ids': position_ids.numpy().astype(np.int32),
            'causal_mask': causal_mask.numpy().astype(np.float16),
            'current_pos': current_pos.numpy().astype(np.int32),
        }, state)

        print(f"CoreML prefill output shape: {cm_prefill_out['output_hidden_states'].shape}")

        # Run FFN for next token (reusing state from prefill)
        cm_last_embed = cm_models['embeddings'].predict({
            'input_ids': input_ids[:, -1:].numpy().astype(np.int32)
        })['hidden_states']

        cm_ffn_out = cm_models['ffn'].predict({
            'hidden_states': cm_last_embed.astype(np.float16),
            'position_ids': pos.numpy().astype(np.int32),
            'causal_mask': single_mask.numpy().astype(np.float16),
            'current_pos': current.numpy().astype(np.int32),
        }, state)

        cm_out = cm_ffn_out['output_hidden_states']
        print(f"CoreML FFN output: shape={cm_out.shape}")
        print(f"  mean={cm_out.mean():.4f}, std={cm_out.std():.4f}")

        compare_arrays("ffn_output", pt_ffn_out_normed, cm_out, threshold=10.0)

        # Get CoreML logits
        cm_lmhead_out = cm_models['lmhead'].predict({
            'hidden_states': cm_out.astype(np.float16)
        })

        cm_logits_parts = []
        for i in range(1, 17):
            key = f'logits_{i}'
            if key in cm_lmhead_out:
                cm_logits_parts.append(cm_lmhead_out[key])

        if cm_logits_parts:
            cm_logits = np.concatenate(cm_logits_parts, axis=-1).squeeze()
            cm_token = cm_logits.argmax()
            print(f"\nCoreML predicted: {cm_token} -> '{tokenizer.decode([cm_token])}'")
            print(f"Match: {pt_token == cm_token}")


if __name__ == "__main__":
    print("="*70)
    print("SINGLE TOKEN TEST (NO PREFILL)")
    print("="*70)
    test_single_token_no_prefill()

    print("\n\n")
    print("="*70)
    print("PREFILL + FFN TEST")
    print("="*70)
    test_prefill_vs_ffn()

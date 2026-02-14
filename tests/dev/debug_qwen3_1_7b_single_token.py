#!/usr/bin/env python3
"""Debug single-token CoreML inference for Qwen3-1.7B - Embeddings and LM Head only."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_DIR = "/Volumes/Models/ANE/qwen3_1.7b_ctx2048_FP16"
TOKENIZER_PATH = MODEL_DIR

def load_model(path, function_name=None, compute_unit=ct.ComputeUnit.CPU_AND_NE):
    """Load CoreML model - handles .mlmodelc and .mlpackage."""
    path = Path(path)
    if path.suffix == '.mlmodelc':
        if function_name:
            return ct.models.CompiledMLModel(str(path), compute_unit, function_name=function_name)
        return ct.models.CompiledMLModel(str(path), compute_unit)
    else:
        if function_name:
            return ct.models.MLModel(str(path), compute_units=compute_unit, function_name=function_name)
        return ct.models.MLModel(str(path), compute_units=compute_unit)

def main():
    print("=" * 70)
    print("Qwen3-1.7B Single Token Comparison (Embeddings + LM Head Only)")
    print("=" * 70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    print(f"Tokenizer loaded")

    # Load HF model for reference
    print("\nLoading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", torch_dtype=torch.float16, local_files_only=True
    )
    hf_model.eval()

    # Load CoreML embeddings and LM head
    print("\nLoading CoreML models...")
    embed_path = Path(MODEL_DIR) / "qwen_embeddings.mlmodelc"
    lmhead_path = Path(MODEL_DIR) / "qwen_lm_head.mlmodelc"

    embed_model = load_model(embed_path)
    lmhead_model = load_model(lmhead_path)
    print(f"  Embeddings loaded: {embed_path.name}")
    print(f"  LM head loaded: {lmhead_path.name}")

    # Test single token
    test_token = 9707  # "Hello"
    print(f"\nTest token: {test_token} = '{tokenizer.decode([test_token])}'")

    # Compare embeddings
    print("\n" + "=" * 70)
    print("Step 1: Compare Embeddings")
    print("=" * 70)

    tokens = torch.tensor([[test_token]], dtype=torch.long)

    # HF embeddings
    with torch.no_grad():
        hf_embeds = hf_model.model.embed_tokens(tokens)
    print(f"HF embeddings shape: {hf_embeds.shape}")
    print(f"HF embeds[0,0,:10]: {hf_embeds[0,0,:10].tolist()}")

    # CoreML embeddings
    coreml_input = {'input_ids': tokens.numpy().astype(np.int32)}
    coreml_embeds = embed_model.predict(coreml_input)['hidden_states']
    print(f"CoreML embeddings shape: {coreml_embeds.shape}")
    print(f"CoreML embeds[0,0,:10]: {coreml_embeds[0,0,:10].tolist()}")

    embed_diff = np.abs(hf_embeds.numpy() - coreml_embeds)
    print(f"\nEmbedding diff max: {embed_diff.max():.6f}")
    print(f"Embedding diff mean: {embed_diff.mean():.6f}")

    if embed_diff.max() < 0.001:
        print("OK: Embeddings match perfectly!")
    else:
        print("WARNING: Embeddings differ!")

    # Now run HF hidden states through CoreML LM head
    print("\n" + "=" * 70)
    print("Step 2: Test LM Head with HF Hidden States")
    print("=" * 70)

    # Get HF hidden states after full transformer pass (no cache, single forward)
    with torch.no_grad():
        hf_outputs = hf_model(tokens, use_cache=False, output_hidden_states=True)
        hf_hidden = hf_outputs.hidden_states[-1]  # Last layer hidden states
        hf_logits = hf_outputs.logits

    print(f"HF final hidden states shape: {hf_hidden.shape}")
    print(f"HF hidden[0,0,:10]: {hf_hidden[0,0,:10].tolist()}")
    print(f"HF logits shape: {hf_logits.shape}")

    # Run HF hidden states through CoreML LM head
    lm_input = {'hidden_states': hf_hidden.numpy().astype(np.float16)}
    coreml_lm_output = lmhead_model.predict(lm_input)

    # Combine logits
    logits_list = []
    for i in range(1, 17):
        key = f'logits{i}'
        if key in coreml_lm_output:
            logits_list.append(coreml_lm_output[key])

    if logits_list:
        coreml_logits = np.concatenate(logits_list, axis=-1)
    else:
        print(f"ERROR: No logits found. Keys: {list(coreml_lm_output.keys())}")
        return

    print(f"CoreML logits shape: {coreml_logits.shape}")
    print(f"CoreML logits[0,0,:10]: {coreml_logits[0,0,:10].tolist()}")
    print(f"HF logits[0,0,:10]: {hf_logits[0,0,:10].tolist()}")

    logits_diff = np.abs(hf_logits.numpy() - coreml_logits)
    print(f"\nLogits diff max: {logits_diff.max():.4f}")
    print(f"Logits diff mean: {logits_diff.mean():.4f}")

    # Compare top predictions
    hf_probs = torch.softmax(hf_logits[0, 0, :].float(), dim=-1)
    coreml_probs = torch.softmax(torch.from_numpy(coreml_logits[0, 0, :]).float(), dim=-1)

    _, hf_top5 = torch.topk(hf_probs, 5)
    _, coreml_top5 = torch.topk(coreml_probs, 5)

    print(f"\nHF top-5: {[tokenizer.decode([t.item()]) for t in hf_top5]}")
    print(f"CoreML LM head top-5: {[tokenizer.decode([t.item()]) for t in coreml_top5]}")

    overlap = set(hf_top5.tolist()) & set(coreml_top5.tolist())
    print(f"Overlap: {len(overlap)}/5")

    if len(overlap) >= 4:
        print("\nOK: LM head works correctly with HF hidden states!")
        print("This means the issue is in the FFN conversion, not LM head.")
    else:
        print("\nWARNING: LM head produces different results even with HF hidden states!")

    # Step 3: Test end-to-end single token with embeddings only (no FFN)
    print("\n" + "=" * 70)
    print("Step 3: CoreML Embeddings -> HF Model Final Norm -> CoreML LM Head")
    print("=" * 70)

    # Get CoreML embeddings as torch tensor
    coreml_embeds_torch = torch.from_numpy(coreml_embeds).to(torch.float16)

    # Run through HF model's layers (using embeddings from CoreML)
    with torch.no_grad():
        # Get position embeddings and run through layers
        hidden_states = coreml_embeds_torch
        position_ids = torch.arange(1, dtype=torch.long).unsqueeze(0)

        # Process through HF decoder layers
        for layer in hf_model.model.layers:
            layer_output = layer(hidden_states, position_ids=position_ids)
            hidden_states = layer_output[0]

        # Apply final norm
        hidden_states = hf_model.model.norm(hidden_states)

    print(f"HF-processed hidden states from CoreML embeds: {hidden_states[0,0,:10].tolist()}")

    # Now run through CoreML LM head
    lm_input = {'hidden_states': hidden_states.numpy().astype(np.float16)}
    coreml_lm_output = lmhead_model.predict(lm_input)

    logits_list = []
    for i in range(1, 17):
        key = f'logits{i}'
        if key in coreml_lm_output:
            logits_list.append(coreml_lm_output[key])

    if logits_list:
        coreml_logits = np.concatenate(logits_list, axis=-1)
        coreml_probs = torch.softmax(torch.from_numpy(coreml_logits[0, 0, :]).float(), dim=-1)
        _, coreml_top5 = torch.topk(coreml_probs, 5)
        print(f"Top-5 (CoreML embed -> HF layers -> CoreML LM head): {[tokenizer.decode([t.item()]) for t in coreml_top5]}")

        overlap2 = set(hf_top5.tolist()) & set(coreml_top5.tolist())
        print(f"Overlap with HF: {len(overlap2)}/5")

if __name__ == "__main__":
    main()

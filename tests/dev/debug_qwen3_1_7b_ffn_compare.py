#!/usr/bin/env python3
"""Compare FFN outputs between PyTorch and CoreML for single token."""

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
CONTEXT_LENGTH = 2048

def load_model(path, function_name=None, compute_unit=ct.ComputeUnit.CPU_AND_NE):
    """Load CoreML model - handles .mlmodelc and .mlpackage."""
    path = Path(path)
    if path.suffix == '.mlmodelc':
        if function_name:
            try:
                return ct.models.CompiledMLModel(str(path), compute_unit, function_name=function_name)
            except RuntimeError:
                # Multi-function compiled models don't support function_name
                return ct.models.CompiledMLModel(str(path), compute_unit)
        return ct.models.CompiledMLModel(str(path), compute_unit)
    else:
        if function_name:
            return ct.models.MLModel(str(path), function_name=function_name)
        return ct.models.MLModel(str(path))

def main():
    print("=" * 70)
    print("Qwen3-1.7B FFN Single Token Comparison")
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

    # Load CoreML models
    print("\nLoading CoreML models...")
    embed_path = Path(MODEL_DIR) / "qwen_embeddings.mlmodelc"

    # Use .mlpackage for multi-function models
    ffn_paths = [
        Path(MODEL_DIR) / f"qwen_FFN_PF_chunk_0{i}of04.mlpackage"
        for i in range(1, 5)
    ]

    embed_model = load_model(embed_path)
    print(f"  Embeddings loaded")

    # Load FFN chunks with 'infer' function
    ffn_models = []
    for path in ffn_paths:
        model = load_model(path, function_name='infer')
        ffn_models.append(model)
        print(f"  Loaded FFN chunk: {path.name}")

    # Test single token
    test_token = 9707  # "Hello"
    print(f"\nTest token: {test_token} = '{tokenizer.decode([test_token])}'")

    tokens = torch.tensor([[test_token]], dtype=torch.long)

    # Get embeddings
    coreml_input = {'input_ids': tokens.numpy().astype(np.int32)}
    coreml_embeds = embed_model.predict(coreml_input)['hidden_states']
    print(f"CoreML embeddings shape: {coreml_embeds.shape}")

    # Also get HF embeddings for comparison
    with torch.no_grad():
        hf_embeds = hf_model.model.embed_tokens(tokens)
    print(f"HF embeddings[0,0,:5]: {hf_embeds[0,0,:5].tolist()}")
    print(f"CoreML embeddings[0,0,:5]: {coreml_embeds[0,0,:5].tolist()}")

    embed_diff = np.abs(hf_embeds.numpy() - coreml_embeds)
    print(f"Embedding diff max: {embed_diff.max():.6f}")

    # Create inputs for FFN single-token (infer function)
    position_ids = np.array([0], dtype=np.int32)  # Position 0 for first token
    current_pos = np.array([0], dtype=np.int32)

    # Create causal mask for single token at position 0
    causal_mask = np.zeros((1, 1, 1, CONTEXT_LENGTH), dtype=np.float16)
    # For position 0, we can attend to position 0 only
    causal_mask[0, 0, 0, 1:] = -np.inf  # Mask out all positions except 0

    print(f"\nFFN Inputs:")
    print(f"  hidden_states shape: {coreml_embeds.shape}")
    print(f"  position_ids: {position_ids}")
    print(f"  current_pos: {current_pos}")
    print(f"  causal_mask shape: {causal_mask.shape}")

    # Run through each FFN chunk (CoreML) WITHOUT state (stateless)
    print("\n" + "=" * 70)
    print("Running CoreML FFN (stateless predict)...")
    print("=" * 70)
    hidden_states = coreml_embeds.astype(np.float16)

    # Need to create state for each chunk
    states = []
    for ffn_model in ffn_models:
        try:
            state = ffn_model.make_state()
            states.append(state)
        except Exception as e:
            print(f"  Warning: Could not create state: {e}")
            states.append(None)

    for chunk_idx, (ffn_model, state) in enumerate(zip(ffn_models, states)):
        inputs = {
            'hidden_states': hidden_states,
            'position_ids': position_ids,
            'causal_mask': causal_mask,
            'current_pos': current_pos
        }

        if state is not None:
            output = ffn_model.predict(inputs, state)
        else:
            output = ffn_model.predict(inputs)

        hidden_states = output['output_hidden_states']
        print(f"  Chunk {chunk_idx+1} output shape: {hidden_states.shape}")
        print(f"  Chunk {chunk_idx+1} output[0,0,:5]: {hidden_states[0,0,:5].tolist()}")

    print(f"\nFinal CoreML FFN output[0,0,:10]: {hidden_states[0,0,:10].tolist()}")

    # Compare with HF model full pass
    print("\n" + "=" * 70)
    print("Running HF model...")
    print("=" * 70)
    with torch.no_grad():
        hf_outputs = hf_model(tokens, use_cache=False, output_hidden_states=True)
        hf_hidden = hf_outputs.hidden_states[-1]  # Final hidden states (after norm)
        hf_logits = hf_outputs.logits

    print(f"HF final hidden states[0,0,:10]: {hf_hidden[0,0,:10].tolist()}")

    # Compare hidden states
    hidden_diff = np.abs(hf_hidden.numpy() - hidden_states)
    print(f"\nHidden states diff:")
    print(f"  max: {hidden_diff.max():.4f}")
    print(f"  mean: {hidden_diff.mean():.4f}")

    # Check top-5 overlap via LM head
    from anemll.models.qwen_model import QwenConfig, QwenForCausalLM

    # Load LM head for comparison
    lmhead_path = Path(MODEL_DIR) / "qwen_lm_head.mlmodelc"
    lmhead_model = load_model(lmhead_path)

    lm_input = {'hidden_states': hidden_states.astype(np.float16)}
    coreml_lm_output = lmhead_model.predict(lm_input)

    logits_list = []
    for i in range(1, 17):
        key = f'logits{i}'
        if key in coreml_lm_output:
            logits_list.append(coreml_lm_output[key])

    if logits_list:
        coreml_logits = np.concatenate(logits_list, axis=-1)
    else:
        print(f"ERROR: No logits found")
        return

    print(f"\nCoreML logits shape: {coreml_logits.shape}")
    print(f"HF logits shape: {hf_logits.shape}")

    # Compare top-5
    hf_probs = torch.softmax(hf_logits[0, 0, :].float(), dim=-1)
    coreml_probs = torch.softmax(torch.from_numpy(coreml_logits[0, 0, :]).float(), dim=-1)

    _, hf_top5 = torch.topk(hf_probs, 5)
    _, coreml_top5 = torch.topk(coreml_probs, 5)

    print(f"\nHF top-5: {[tokenizer.decode([t.item()]) for t in hf_top5]}")
    print(f"CoreML top-5: {[tokenizer.decode([t.item()]) for t in coreml_top5]}")

    overlap = set(hf_top5.tolist()) & set(coreml_top5.tolist())
    print(f"Overlap: {len(overlap)}/5")

    if len(overlap) >= 4:
        print("\nOK: CoreML FFN outputs match HF!")
    else:
        print("\n*** DIVERGENCE DETECTED ***")
        print("The FFN layers are producing different hidden states.")

if __name__ == "__main__":
    main()

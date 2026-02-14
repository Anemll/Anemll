#!/usr/bin/env python3
"""Debug single-token FFN CoreML inference for Qwen3-1.7B."""

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
            return ct.models.CompiledMLModel(str(path), compute_unit, function_name=function_name)
        return ct.models.CompiledMLModel(str(path), compute_unit)
    else:
        if function_name:
            return ct.models.MLModel(str(path), compute_units=compute_unit, function_name=function_name)
        return ct.models.MLModel(str(path), compute_units=compute_unit)

def main():
    print("=" * 70)
    print("Qwen3-1.7B FFN Single Token Test")
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
    lmhead_path = Path(MODEL_DIR) / "qwen_lm_head.mlmodelc"
    # Use .mlpackage for multi-function models
    ffn_paths = [
        Path(MODEL_DIR) / f"qwen_FFN_PF_chunk_0{i}of04.mlpackage"
        for i in range(1, 5)
    ]

    embed_model = load_model(embed_path)
    lmhead_model = load_model(lmhead_path)
    print(f"  Embeddings loaded")
    print(f"  LM head loaded")

    # Load FFN chunks with 'infer' function
    ffn_models = []
    for path in ffn_paths:
        model = load_model(path, function_name='infer')
        ffn_models.append(model)
        print(f"  Loaded FFN chunk: {path.name}")

    # Create state from first FFN model
    state = ffn_models[0].make_state()
    print(f"  State created")

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

    # Run through each FFN chunk
    hidden_states = coreml_embeds.astype(np.float16)
    for chunk_idx, ffn_model in enumerate(ffn_models):
        inputs = {
            'hidden_states': hidden_states,
            'position_ids': position_ids,
            'causal_mask': causal_mask,
            'current_pos': current_pos
        }
        output = ffn_model.predict(inputs, state)
        hidden_states = output['output_hidden_states']
        print(f"  Chunk {chunk_idx+1} output shape: {hidden_states.shape}")
        print(f"  Chunk {chunk_idx+1} output[0,0,:5]: {hidden_states[0,0,:5].tolist()}")

    print(f"\nFinal CoreML FFN output[0,0,:10]: {hidden_states[0,0,:10].tolist()}")

    # Compare with HF model full pass
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

    # Run through LM head
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

    print(f"\nCoreML logits[0,0,:10]: {coreml_logits[0,0,:10].tolist()}")
    print(f"HF logits[0,0,:10]: {hf_logits[0,0,:10].tolist()}")

    logits_diff = np.abs(hf_logits.numpy() - coreml_logits)
    print(f"\nLogits diff:")
    print(f"  max: {logits_diff.max():.4f}")
    print(f"  mean: {logits_diff.mean():.4f}")

    # Compare top-5
    hf_probs = torch.softmax(hf_logits[0, 0, :].float(), dim=-1)
    coreml_probs = torch.softmax(torch.from_numpy(coreml_logits[0, 0, :]).float(), dim=-1)

    _, hf_top5 = torch.topk(hf_probs, 5)
    _, coreml_top5 = torch.topk(coreml_probs, 5)

    print(f"\nHF top-5: {[tokenizer.decode([t.item()]) for t in hf_top5]}")
    print(f"CoreML top-5: {[tokenizer.decode([t.item()]) for t in coreml_top5]}")

    overlap = set(hf_top5.tolist()) & set(coreml_top5.tolist())
    print(f"Overlap: {len(overlap)}/5")

    if len(overlap) < 3:
        print("\n*** DIVERGENCE DETECTED ***")
        print("Investigating layer by layer...")

        # Compare our PyTorch implementation
        from anemll.models.qwen_model import QwenConfig, QwenForCausalLM, MODEL_DTYPE

        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots")
        snapshot_dirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
        model_path = os.path.join(cache_dir, snapshot_dirs[0])

        qwen_config = QwenConfig(
            hidden_size=hf_model.config.hidden_size,
            num_attention_heads=hf_model.config.num_attention_heads,
            num_key_value_heads=hf_model.config.num_key_value_heads,
            head_dim=getattr(hf_model.config, 'head_dim', 128),
            num_hidden_layers=hf_model.config.num_hidden_layers,
            intermediate_size=hf_model.config.intermediate_size,
            vocab_size=hf_model.config.vocab_size,
            rms_norm_eps=hf_model.config.rms_norm_eps,
            rope_theta=hf_model.config.rope_theta,
            context_length=CONTEXT_LENGTH,
            state_length=CONTEXT_LENGTH,
        )

        our_model = QwenForCausalLM(qwen_config, enable_coreml=False, disable_kv_cache=True)
        our_model.load_pretrained_weights(model_path)
        our_model.eval()

        # Run our PyTorch model
        position_ids_torch = torch.tensor([0], dtype=torch.int32)
        causal_mask_torch = torch.zeros((1, 1, 1, CONTEXT_LENGTH), dtype=MODEL_DTYPE)
        causal_mask_torch[0, 0, 0, 1:] = float('-inf')
        update_mask = torch.zeros((1, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE)
        current_pos_torch = torch.tensor([0], dtype=torch.int32)

        with torch.no_grad():
            our_outputs = our_model(
                input_ids=tokens.to(torch.int32),
                update_mask=update_mask,
                position_ids=position_ids_torch,
                causal_mask=causal_mask_torch,
                current_pos=current_pos_torch,
                IN_PREFILL=False,
            )

        if isinstance(our_outputs, tuple):
            our_logits = torch.cat(our_outputs, dim=-1)
        else:
            our_logits = our_outputs

        our_probs = torch.softmax(our_logits[0, 0, :].float(), dim=-1)
        _, our_top5 = torch.topk(our_probs, 5)
        print(f"\nOur PyTorch top-5: {[tokenizer.decode([t.item()]) for t in our_top5]}")

        overlap2 = set(hf_top5.tolist()) & set(our_top5.tolist())
        print(f"Our PyTorch vs HF overlap: {len(overlap2)}/5")

if __name__ == "__main__":
    main()

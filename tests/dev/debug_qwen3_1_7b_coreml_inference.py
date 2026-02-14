#!/usr/bin/env python3
"""Debug CoreML inference for Qwen3-1.7B converted model."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer

MODEL_DIR = "/Volumes/Models/ANE/qwen3_1.7b_ctx2048_FP16"
TOKENIZER_PATH = MODEL_DIR
CONTEXT_LENGTH = 2048

def load_coreml_model(path, function_name=None, compute_unit=ct.ComputeUnit.CPU_AND_NE):
    """Load a CoreML model with proper compilation for state support."""
    import subprocess
    import tempfile
    import shutil

    # For mlpackage, we need to compile it first to get a proper mlmodelc with state support
    if path.endswith('.mlpackage'):
        # Check if corresponding .mlmodelc exists
        mlmodelc_path = path.replace('.mlpackage', '.mlmodelc')
        if os.path.exists(mlmodelc_path):
            path = mlmodelc_path
        else:
            # Compile using coremlcompiler
            temp_dir = tempfile.mkdtemp()
            result = subprocess.run(
                ['xcrun', 'coremlcompiler', 'compile', path, temp_dir],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"Compilation failed: {result.stderr}")
                raise RuntimeError(f"Failed to compile {path}")
            # Find the compiled model
            for item in os.listdir(temp_dir):
                if item.endswith('.mlmodelc'):
                    path = os.path.join(temp_dir, item)
                    break
            print(f"Compiled {os.path.basename(path)} to {path}")

    if function_name:
        model = ct.models.MLModel(path, compute_units=compute_unit, function_name=function_name)
    else:
        model = ct.models.MLModel(path, compute_units=compute_unit)
    return model

def main():
    print("=" * 70)
    print("Qwen3-1.7B CoreML Inference Debug")
    print("=" * 70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    print(f"Tokenizer loaded: vocab_size={len(tokenizer)}")

    # Load models - use .mlpackage files
    embed_path = os.path.join(MODEL_DIR, "qwen_embeddings.mlpackage")
    lmhead_path = os.path.join(MODEL_DIR, "qwen_lm_head.mlpackage")
    ffn_paths = [
        os.path.join(MODEL_DIR, f"qwen_FFN_PF_chunk_0{i}of04.mlpackage")
        for i in range(1, 5)
    ]

    print("\nLoading embeddings model...")
    embed_model = load_coreml_model(embed_path)
    print(f"  Inputs: {[i.name for i in embed_model.get_spec().description.input]}")
    print(f"  Outputs: {[o.name for o in embed_model.get_spec().description.output]}")

    print("\nLoading LM head model...")
    lmhead_model = load_coreml_model(lmhead_path)
    print(f"  Inputs: {[i.name for i in lmhead_model.get_spec().description.input]}")
    print(f"  Outputs: {[o.name for o in lmhead_model.get_spec().description.output]}")

    print("\nLoading FFN models...")
    ffn_models = []
    for path in ffn_paths:
        # Load infer and prefill functions
        model_infer = load_coreml_model(path, function_name='infer')
        model_prefill = load_coreml_model(path, function_name='prefill')
        ffn_models.append({'infer': model_infer, 'prefill': model_prefill})
        print(f"  Loaded: {os.path.basename(path)}")
        print(f"    Infer inputs: {[i.name for i in model_infer.get_spec().description.input]}")
        print(f"    State: {[s.name for s in model_infer.get_spec().description.state]}")

    # Create unified state
    print("\nCreating unified state...")
    state = ffn_models[0]['infer'].make_state()
    print(f"State created successfully")

    # Test input
    prompt = "Who are you?"
    tokens = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {tokens.tolist()}")

    seq_len = tokens.shape[1]

    # Create causal mask for full context
    def make_causal_mask(length, start):
        mask = np.full((1, 1, length, CONTEXT_LENGTH), -np.inf, dtype=np.float16)
        row_indices = np.arange(length).reshape(length, 1)
        col_indices = np.arange(CONTEXT_LENGTH).reshape(1, CONTEXT_LENGTH)
        mask[:, :, col_indices <= (row_indices + start)] = 0
        return mask

    # Run prefill (process all input tokens at once with 64-batch prefill)
    print("\n" + "=" * 70)
    print("Running Prefill (batch=64)")
    print("=" * 70)

    batch_size = 64
    batch_pos = 0

    # Pad tokens to batch_size
    padded_tokens = torch.zeros((1, batch_size), dtype=torch.int32)
    padded_tokens[0, :seq_len] = tokens[0]

    # Run embeddings
    embed_output = embed_model.predict({
        'input_ids': padded_tokens.numpy().astype(np.int32)
    })
    hidden_states = embed_output['hidden_states']
    print(f"Embeddings output shape: {hidden_states.shape}")
    print(f"Embeddings[0,0,:5]: {hidden_states[0,0,:5]}")

    # Create position IDs for batch
    position_ids = np.arange(batch_pos, batch_pos + batch_size, dtype=np.int32)
    causal_mask = make_causal_mask(batch_size, batch_pos)
    current_pos = np.array([batch_pos], dtype=np.int32)

    print(f"\nInputs to FFN prefill:")
    print(f"  hidden_states shape: {hidden_states.shape}")
    print(f"  position_ids: {position_ids[:10]}...")
    print(f"  causal_mask shape: {causal_mask.shape}")
    print(f"  current_pos: {current_pos}")

    # Run through FFN chunks
    for chunk_idx, ffn_model in enumerate(ffn_models):
        inputs = {
            'hidden_states': hidden_states.astype(np.float16),
            'position_ids': position_ids,
            'causal_mask': causal_mask.astype(np.float16),
            'current_pos': current_pos
        }
        output = ffn_model['prefill'].predict(inputs, state)
        hidden_states = output['output_hidden_states']
        print(f"  Chunk {chunk_idx+1}: output shape {hidden_states.shape}")

    print(f"\nFFN output[0,seq_len-1,:5]: {hidden_states[0,seq_len-1,:5]}")

    # Run LM head on the last actual token position
    last_token_hidden = hidden_states[:, seq_len-1:seq_len, :]
    print(f"Last token hidden shape: {last_token_hidden.shape}")

    lm_output = lmhead_model.predict({'hidden_states': last_token_hidden.astype(np.float16)})

    # Combine logits
    logits_list = []
    for i in range(1, 17):
        key = f'logits{i}'
        if key in lm_output:
            logits_list.append(lm_output[key])

    if logits_list:
        logits = np.concatenate(logits_list, axis=-1)
    else:
        logits = lm_output.get('logits', None)

    if logits is None:
        print(f"ERROR: No logits found in LM head output. Keys: {list(lm_output.keys())}")
        return

    print(f"Logits shape: {logits.shape}")
    print(f"Logits[0,0,:10]: {logits[0,0,:10]}")

    # Get top tokens
    probs = torch.softmax(torch.from_numpy(logits[0, 0, :]).float(), dim=-1)
    top_probs, top_indices = torch.topk(probs, 10)

    print(f"\nCoreML Top 10 predictions:")
    for i, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
        token_str = tokenizer.decode([idx])
        print(f"  {i+1}. '{token_str}' (id={idx}, prob={prob:.4f})")

    # Compare with HF model
    print("\n" + "=" * 70)
    print("Comparing with HuggingFace Model")
    print("=" * 70)

    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.float16, local_files_only=True)
    hf_model.eval()

    with torch.no_grad():
        hf_outputs = hf_model(tokens, use_cache=False)
        hf_logits = hf_outputs.logits

    hf_probs = torch.softmax(hf_logits[0, -1, :].float(), dim=-1)
    hf_top_probs, hf_top_indices = torch.topk(hf_probs, 10)

    print(f"HF Top 10 predictions:")
    for i, (prob, idx) in enumerate(zip(hf_top_probs.tolist(), hf_top_indices.tolist())):
        token_str = tokenizer.decode([idx])
        print(f"  {i+1}. '{token_str}' (id={idx}, prob={prob:.4f})")

    # Check overlap
    coreml_top5 = set(top_indices[:5].tolist())
    hf_top5 = set(hf_top_indices[:5].tolist())
    overlap = coreml_top5 & hf_top5
    print(f"\nOverlap in top-5: {len(overlap)}/5")

    if len(overlap) < 3:
        print("\nWARNING: Poor CoreML vs HF overlap!")
        print("Investigating embeddings...")

        # Compare embeddings
        with torch.no_grad():
            hf_embeds = hf_model.model.embed_tokens(tokens)
        coreml_embeds = embed_model.predict({'input_ids': tokens.numpy().astype(np.int32)})['hidden_states']

        print(f"HF embeddings[0,0,:5]: {hf_embeds[0,0,:5].tolist()}")
        print(f"CoreML embeddings[0,0,:5]: {coreml_embeds[0,0,:5].tolist()}")

        embed_diff = np.abs(hf_embeds.numpy() - coreml_embeds[:,:seq_len,:])
        print(f"Embedding diff max: {embed_diff.max():.6f}")

if __name__ == "__main__":
    main()

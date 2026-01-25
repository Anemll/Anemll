#!/usr/bin/env python3
"""Gemma3n ANE chat with fixed KV cache state sharing between chunks.

The key fix: Each chunk model has its own independent state. We must manually
copy the KV cache state between chunks after each prediction.
"""

import argparse
from pathlib import Path
from typing import List

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer

from gemma3n_coreml_inputs import (
    create_position_mask,
    create_position_one_hot,
    create_rotary_embeddings,
)


def load_mlpackage(path: Path) -> ct.models.MLModel:
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    return ct.models.MLModel(str(path))


def find_infer_chunks(bundle_dir: Path) -> List[Path]:
    chunks = sorted(bundle_dir.glob("gemma3n_infer_chunk_*of*.mlpackage"))
    return chunks


def concat_logits(outputs: dict) -> np.ndarray:
    split_keys = sorted(
        [k for k in outputs.keys() if k.startswith("logits_split_")],
        key=lambda k: int(k.split("_")[-1]),
    )
    if split_keys:
        parts = [outputs[k] for k in split_keys]
        return np.concatenate(parts, axis=-1)
    if "output_logits" in outputs:
        return outputs["output_logits"]
    raise KeyError(f"Unexpected LM head outputs: {list(outputs.keys())}")


def summarize_tensor(name: str, arr: np.ndarray, max_vals: int = 5) -> None:
    flat = arr.reshape(-1)
    sample = flat[:max_vals]
    print(
        f"{name}: shape={arr.shape} dtype={arr.dtype} "
        f"min={arr.min():.6f} max={arr.max():.6f} "
        f"mean={arr.mean():.6f} std={arr.std():.6f} "
        f"sample={sample}"
    )


def sample_next_token(logits: np.ndarray, temperature: float, top_k: int) -> int:
    logits = logits.astype(np.float32)
    if temperature <= 0.0:
        return int(np.argmax(logits))
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        top_idx = np.argpartition(logits, -top_k)[-top_k:]
        top_logits = logits[top_idx]
        top_logits = top_logits - np.max(top_logits)
        probs = np.exp(top_logits)
        probs = probs / np.sum(probs)
        return int(np.random.choice(top_idx, p=probs))
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)
    return int(np.random.choice(np.arange(logits.shape[-1]), p=probs))


def apply_logit_softcap(logits: np.ndarray, softcap: float = 30.0) -> np.ndarray:
    """Match Gemma3n final logit softcapping (tanh(logits / softcap) * softcap)."""
    return np.tanh(logits / softcap) * softcap


class ChunkedInferenceManager:
    """Manages stateful inference across multiple chunks with proper state sharing."""

    def __init__(self, chunk_models: List[ct.models.MLModel], state_name: str = "model_kv_cache_0"):
        self.chunk_models = chunk_models
        self.state_name = state_name
        self.num_chunks = len(chunk_models)

        # Create a state for each chunk
        self.chunk_states = [model.make_state() for model in chunk_models]

        # Get KV cache shape from first chunk
        kv_cache = self.chunk_states[0].read_state(state_name)
        self.kv_shape = np.array(kv_cache).shape
        self.head_dim = self.kv_shape[-1]
        self.context_length = self.kv_shape[2]
        print(f"KV cache shape: {self.kv_shape}")
        print(f"  = {self.kv_shape[0]//2} layers × 2 (K/V) × {self.kv_shape[1]} heads × {self.kv_shape[2]} seq × {self.kv_shape[3]} dim")

    def reset_states(self):
        """Reset all chunk states to zeros."""
        zero_kv = np.zeros(self.kv_shape, dtype=np.float16)
        for state in self.chunk_states:
            state.write_state(self.state_name, zero_kv)

    def predict(self, inputs: dict) -> dict:
        """Run prediction through all chunks with proper state propagation."""
        hidden_states = inputs["hidden_states"]
        # Read current shared KV cache (use chunk 0's state as the "master")
        kv_cache = self.chunk_states[0].read_state(self.state_name)

        # Run through each chunk, propagating KV updates between chunks
        for i, (model, state) in enumerate(zip(self.chunk_models, self.chunk_states)):
            if i > 0:
                # Ensure this chunk sees updates from previous chunks
                state.write_state(self.state_name, kv_cache)
            chunk_inputs = {
                "hidden_states": hidden_states,
                "per_layer_inputs": inputs["per_layer_inputs"],
                "causal_mask": inputs["causal_mask"],
                "current_pos": inputs["current_pos"],
                "position_one_hot": inputs["position_one_hot"],
                "rotary_cos_local": inputs["rotary_cos_local"],
                "rotary_sin_local": inputs["rotary_sin_local"],
                "rotary_cos_global": inputs["rotary_cos_global"],
                "rotary_sin_global": inputs["rotary_sin_global"],
            }
            out = model.predict(chunk_inputs, state)
            hidden_states = out["output_hidden_states"]

            # After each chunk, read its updated KV cache for propagation
            kv_cache = state.read_state(self.state_name)

        # Update master state (chunk 0) with the accumulated updates
        self.chunk_states[0].write_state(self.state_name, kv_cache)

        return {"output_hidden_states": hidden_states}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma3n ANE chat with fixed state sharing")
    parser.add_argument("--bundle", default="/tmp/gemma3n-fixed/infer", help="Directory with .mlpackage files")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--context-length", type=int, default=512, help="Context length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--debug-tensors", action="store_true", help="Print tensor stats")
    parser.add_argument("--debug-steps", type=int, default=2, help="Steps to debug")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    print(f"Bundle: {bundle}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(bundle), use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # Load models
    print("Loading CoreML models...")
    infer_init = load_mlpackage(bundle / "gemma3n_infer_init.mlpackage")
    combine_model = load_mlpackage(bundle / "gemma3n_combine_streams.mlpackage")
    lm_head = load_mlpackage(bundle / "gemma3n_lm_head.mlpackage")

    infer_chunk_paths = find_infer_chunks(bundle)
    if not infer_chunk_paths:
        raise FileNotFoundError("No infer chunks found")

    infer_chunk_models = [load_mlpackage(p) for p in infer_chunk_paths]
    print(f"Loaded {len(infer_chunk_models)} infer chunks")

    # Create inference manager with proper state sharing
    print("\nInitializing chunked inference manager with state sharing...")
    infer_manager = ChunkedInferenceManager(infer_chunk_models)

    # Infer actual head dimension/context length from KV cache state
    head_dim = infer_manager.head_dim
    ctx_len = infer_manager.context_length

    # Note: Position-specific masks are created per-step via create_position_mask()
    # This avoids gather() issues in CoreML tracing

    # Encode prompt
    input_ids = tokenizer(args.prompt, return_tensors="np")["input_ids"].astype(np.int32)
    token_ids = input_ids[0].tolist()

    print(f"\nPrompt: {args.prompt}")
    print(f"Prompt tokens: {len(token_ids)}")
    print("Generating...\n")

    # Prefill phase
    last_hidden = None
    for pos, tok in enumerate(token_ids):
        init_out = infer_init.predict(
            {"input_ids": np.array([[int(tok)]], dtype=np.int32)}
        )
        hidden_states = init_out["hidden_states"]
        per_layer_inputs = init_out["per_layer_inputs"]

        if args.debug_tensors and pos < args.debug_steps:
            summarize_tensor(f"prefill[{pos}].init.hidden_states", hidden_states)

        # Create position-specific mask, one-hot, and rotary embeddings
        pos_mask = create_position_mask(pos, ctx_len)
        pos_one_hot = create_position_one_hot(pos, ctx_len)
        cos_local, sin_local, cos_global, sin_global = create_rotary_embeddings(pos, head_dim)

        out = infer_manager.predict({
            "hidden_states": hidden_states,
            "per_layer_inputs": per_layer_inputs,
            "causal_mask": pos_mask,
            "current_pos": np.array([pos], dtype=np.int32),
            "position_one_hot": pos_one_hot,
            "rotary_cos_local": cos_local,
            "rotary_sin_local": sin_local,
            "rotary_cos_global": cos_global,
            "rotary_sin_global": sin_global,
        })
        hidden_states = out["output_hidden_states"]

        if args.debug_tensors and pos < args.debug_steps:
            summarize_tensor(f"prefill[{pos}].infer.hidden_states", hidden_states)

        last_hidden = combine_model.predict({"hidden_states": hidden_states})["output_hidden_states"]

    if last_hidden is None:
        raise RuntimeError("Prefill failed")

    # Generate first token
    lm_out = lm_head.predict({"hidden_states": last_hidden.astype(np.float16)})
    logits = concat_logits(lm_out)[0, 0]
    logits = apply_logit_softcap(logits)
    next_id = sample_next_token(logits, args.temperature, args.top_k)
    token_ids.append(next_id)
    decoded = tokenizer.decode([next_id])
    print(decoded, end="", flush=True)

    # Generation phase
    current_pos = len(token_ids) - 1
    for gen_step in range(args.max_new_tokens - 1):
        init_out = infer_init.predict(
            {"input_ids": np.array([[int(token_ids[-1])]], dtype=np.int32)}
        )
        hidden_states = init_out["hidden_states"]
        per_layer_inputs = init_out["per_layer_inputs"]

        # Create position-specific mask and one-hot for current position
        pos_mask = create_position_mask(current_pos, ctx_len)
        pos_one_hot = create_position_one_hot(current_pos, ctx_len)
        cos_local, sin_local, cos_global, sin_global = create_rotary_embeddings(current_pos, head_dim)

        out = infer_manager.predict({
            "hidden_states": hidden_states,
            "per_layer_inputs": per_layer_inputs,
            "causal_mask": pos_mask,
            "current_pos": np.array([current_pos], dtype=np.int32),
            "position_one_hot": pos_one_hot,
            "rotary_cos_local": cos_local,
            "rotary_sin_local": sin_local,
            "rotary_cos_global": cos_global,
            "rotary_sin_global": sin_global,
        })
        hidden_states = out["output_hidden_states"]

        if args.debug_tensors and gen_step < args.debug_steps:
            summarize_tensor(f"gen[{gen_step}].infer.hidden_states", hidden_states)

        combined = combine_model.predict({"hidden_states": hidden_states})["output_hidden_states"]
        lm_out = lm_head.predict({"hidden_states": combined.astype(np.float16)})
        logits = concat_logits(lm_out)[0, 0]
        logits = apply_logit_softcap(logits)

        if args.debug_tensors and gen_step < args.debug_steps:
            summarize_tensor(f"gen[{gen_step}].logits", logits)

        next_id = sample_next_token(logits, args.temperature, args.top_k)
        token_ids.append(next_id)
        current_pos += 1

        decoded = tokenizer.decode([next_id])
        print(decoded, end="", flush=True)

    print("\n\nDone.")


if __name__ == "__main__":
    main()

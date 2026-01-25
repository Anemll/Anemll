#!/usr/bin/env python3
"""Gemma3n ANE chat with fixed KV cache state sharing between chunks.

The key fix: Each chunk model has its own independent state. We must manually
copy the KV cache state between chunks after each prediction.
"""

import argparse
import time
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
    # Prefer ANE with CPU fallback; avoid GPU unless explicitly requested.
    return ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)


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
    """Manages stateful inference across multiple chunks with a single shared state."""

    def __init__(self, chunk_models: List[ct.models.MLModel], state_name: str = "model_kv_cache_0"):
        self.chunk_models = chunk_models
        self.state_name = state_name
        self.num_chunks = len(chunk_models)

        # Single unified state shared across all chunks (like tests/chat.py).
        self.shared_state = self.chunk_models[0].make_state()

        kv_cache = self.shared_state.read_state(state_name)
        self.kv_shape = np.array(kv_cache).shape
        self.head_dim = self.kv_shape[-1]
        self.context_length = self.kv_shape[2]
        print(f"KV cache shape: {self.kv_shape}")
        print(f"  = {self.kv_shape[0]//2} layers × 2 (K/V) × {self.kv_shape[1]} heads × {self.kv_shape[2]} seq × {self.kv_shape[3]} dim")
        self.last_chunk_times = [0.0 for _ in range(self.num_chunks)]

    def reset_states(self):
        """Reset shared state to zeros."""
        zero_kv = np.zeros(self.kv_shape, dtype=np.float16)
        self.shared_state.write_state(self.state_name, zero_kv)

    def predict(self, inputs: dict) -> dict:
        """Run prediction through all chunks with a unified KV cache state."""
        hidden_states = inputs["hidden_states"]
        chunk_inputs = {
            "per_layer_inputs": inputs["per_layer_inputs"],
            "causal_mask": inputs["causal_mask"],
            "current_pos": inputs["current_pos"],
            "position_one_hot": inputs["position_one_hot"],
            "rotary_cos_local": inputs["rotary_cos_local"],
            "rotary_sin_local": inputs["rotary_sin_local"],
            "rotary_cos_global": inputs["rotary_cos_global"],
            "rotary_sin_global": inputs["rotary_sin_global"],
        }

        for idx, model in enumerate(self.chunk_models):
            chunk_inputs["hidden_states"] = hidden_states
            t0 = time.perf_counter()
            out = model.predict(chunk_inputs, self.shared_state)
            self.last_chunk_times[idx] = time.perf_counter() - t0
            hidden_states = out["output_hidden_states"]

        return {"output_hidden_states": hidden_states}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma3n ANE chat with fixed state sharing")
    parser.add_argument("--bundle", default="/tmp/gemma3n-fixed/infer", help="Directory with .mlpackage files")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--context-length", type=int, default=512, help="Context length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (no-op)")
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

    # Timing accumulators (seconds)
    timing = {
        "infer_init": 0.0,
        "infer_chunks": [0.0 for _ in range(len(infer_chunk_models))],
        "combine": 0.0,
        "lm_head": 0.0,
    }
    timing_counts = {
        "infer_init": 0,
        "infer_chunks": [0 for _ in range(len(infer_chunk_models))],
        "combine": 0,
        "lm_head": 0,
    }

    # Prefill phase
    last_hidden = None
    prefill_start = time.time()
    for pos, tok in enumerate(token_ids):
        t0 = time.perf_counter()
        init_out = infer_init.predict({"input_ids": np.array([[int(tok)]], dtype=np.int32)})
        timing["infer_init"] += time.perf_counter() - t0
        timing_counts["infer_init"] += 1
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
        for i, t in enumerate(infer_manager.last_chunk_times):
            timing["infer_chunks"][i] += t
            timing_counts["infer_chunks"][i] += 1
        hidden_states = out["output_hidden_states"]

        if args.debug_tensors and pos < args.debug_steps:
            summarize_tensor(f"prefill[{pos}].infer.hidden_states", hidden_states)

        t0 = time.perf_counter()
        last_hidden = combine_model.predict({"hidden_states": hidden_states})["output_hidden_states"]
        timing["combine"] += time.perf_counter() - t0
        timing_counts["combine"] += 1

    if last_hidden is None:
        raise RuntimeError("Prefill failed")
    prefill_time = time.time() - prefill_start
    if token_ids:
        prefill_tps = len(token_ids) / max(prefill_time, 1e-6)
        print(f"Prefill: {len(token_ids)} tokens in {prefill_time:.2f}s ({prefill_tps:.2f} t/s)")

    eos_id = tokenizer.eos_token_id

    # Generate first token
    decode_tokens = 0
    decode_start = time.time()
    t0 = time.perf_counter()
    lm_out = lm_head.predict({"hidden_states": last_hidden.astype(np.float16)})
    timing["lm_head"] += time.perf_counter() - t0
    timing_counts["lm_head"] += 1
    logits = concat_logits(lm_out)[0, 0]
    logits = apply_logit_softcap(logits)
    next_id = sample_next_token(logits, args.temperature, args.top_k)
    token_ids.append(next_id)
    decoded = tokenizer.decode([next_id])
    print(decoded, end="", flush=True)
    decode_tokens += 1
    if eos_id is not None and next_id == eos_id:
        decode_time = time.time() - decode_start
        decode_tps = decode_tokens / max(decode_time, 1e-6)
        print(f"\nDecode: {decode_tokens} tokens in {decode_time:.2f}s ({decode_tps:.2f} t/s)")
        print("\n\nDone.")
        return

    # Generation phase
    current_pos = len(token_ids) - 1
    for gen_step in range(args.max_new_tokens - 1):
        t0 = time.perf_counter()
        init_out = infer_init.predict({"input_ids": np.array([[int(token_ids[-1])]], dtype=np.int32)})
        timing["infer_init"] += time.perf_counter() - t0
        timing_counts["infer_init"] += 1
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
        for i, t in enumerate(infer_manager.last_chunk_times):
            timing["infer_chunks"][i] += t
            timing_counts["infer_chunks"][i] += 1
        hidden_states = out["output_hidden_states"]

        if args.debug_tensors and gen_step < args.debug_steps:
            summarize_tensor(f"gen[{gen_step}].infer.hidden_states", hidden_states)

        t0 = time.perf_counter()
        combined = combine_model.predict({"hidden_states": hidden_states})["output_hidden_states"]
        timing["combine"] += time.perf_counter() - t0
        timing_counts["combine"] += 1
        t0 = time.perf_counter()
        lm_out = lm_head.predict({"hidden_states": combined.astype(np.float16)})
        timing["lm_head"] += time.perf_counter() - t0
        timing_counts["lm_head"] += 1
        logits = concat_logits(lm_out)[0, 0]
        logits = apply_logit_softcap(logits)

        if args.debug_tensors and gen_step < args.debug_steps:
            summarize_tensor(f"gen[{gen_step}].logits", logits)

        next_id = sample_next_token(logits, args.temperature, args.top_k)
        token_ids.append(next_id)
        current_pos += 1

        decoded = tokenizer.decode([next_id])
        print(decoded, end="", flush=True)
        decode_tokens += 1
        if eos_id is not None and next_id == eos_id:
            break

    decode_time = time.time() - decode_start
    if decode_tokens > 0:
        decode_tps = decode_tokens / max(decode_time, 1e-6)
        print(f"\nDecode: {decode_tokens} tokens in {decode_time:.2f}s ({decode_tps:.2f} t/s)")

    if args.verbose:
        total_time = timing["infer_init"] + sum(timing["infer_chunks"]) + timing["combine"] + timing["lm_head"]
        if total_time <= 0:
            total_time = 1e-9
        print("\nPer-model time breakdown:")
        print(f"  infer_init: {timing['infer_init']:.3f}s ({timing['infer_init']/total_time*100:.1f}%) "
              f"[{timing_counts['infer_init']} calls]")
        chunk_total = sum(timing["infer_chunks"])
        print(f"  infer_chunks (all): {chunk_total:.3f}s ({chunk_total/total_time*100:.1f}%)")
        for i, t in enumerate(timing["infer_chunks"]):
            calls = timing_counts["infer_chunks"][i]
            pct = (t / total_time * 100.0) if total_time > 0 else 0.0
            print(f"    chunk {i:02d}: {t:.3f}s ({pct:.1f}%) [{calls} calls]")
        print(f"  combine: {timing['combine']:.3f}s ({timing['combine']/total_time*100:.1f}%) "
              f"[{timing_counts['combine']} calls]")
        print(f"  lm_head: {timing['lm_head']:.3f}s ({timing['lm_head']/total_time*100:.1f}%) "
              f"[{timing_counts['lm_head']} calls]")

    print("\n\nDone.")


if __name__ == "__main__":
    main()

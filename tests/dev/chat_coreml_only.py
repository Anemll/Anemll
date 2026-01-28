#!/usr/bin/env python3
"""CoreML-only chat runner using infer chunks or a single infer model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import coremltools as ct
import numpy as np
import yaml
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


def load_meta(meta_path: Optional[Path]) -> Dict:
    if meta_path is None:
        return {}
    data = yaml.safe_load(meta_path.read_text())
    return data or {}


def concat_logits(outputs: Dict[str, np.ndarray]) -> np.ndarray:
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


def apply_logit_softcap(logits: np.ndarray, softcap: float) -> np.ndarray:
    return np.tanh(logits / softcap) * softcap


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


def get_state_name(state: ct.models.MLModelState) -> Optional[str]:
    for candidate in ("model_kv_cache_0", "kv_cache_0"):
        try:
            state.read_state(candidate)
            return candidate
        except Exception:
            continue
    return None


class ChunkedInferRunner:
    def __init__(self, models: List[ct.models.MLModel]):
        self.models = models
        self.states = []
        self.state_name = None
        self.kv_shape = None
        self.needs_position_one_hot = False
        self.needs_rotary = False

        for model in models:
            try:
                state = model.make_state()
            except Exception:
                state = None
            self.states.append(state)

        if self.states and self.states[0] is not None:
            self.state_name = get_state_name(self.states[0])
            if self.state_name:
                kv_cache = self.states[0].read_state(self.state_name)
                self.kv_shape = np.array(kv_cache).shape

        inputs = {inp.name for inp in models[0].get_spec().description.input}
        self.needs_position_one_hot = "position_one_hot" in inputs
        self.needs_rotary = "rotary_cos_local" in inputs

    def get_kv_cache(self) -> Optional[np.ndarray]:
        if self.state_name and self.states and self.states[0] is not None:
            return np.array(self.states[0].read_state(self.state_name))
        return None

    def set_kv_cache(self, kv_cache: np.ndarray) -> None:
        if self.state_name and self.states and self.states[0] is not None:
            self.states[0].write_state(self.state_name, kv_cache)

    def _build_inputs(
        self,
        hidden_states: np.ndarray,
        per_layer_inputs: Optional[np.ndarray],
        pos: int,
        ctx_len: int,
        head_dim: int,
        current_pos: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        inputs = {
            "hidden_states": hidden_states,
            "causal_mask": create_position_mask(pos, ctx_len),
            "current_pos": current_pos,
        }
        if per_layer_inputs is not None:
            inputs["per_layer_inputs"] = per_layer_inputs
        if self.needs_position_one_hot:
            inputs["position_one_hot"] = create_position_one_hot(pos, ctx_len)
        if self.needs_rotary:
            cos_l, sin_l, cos_g, sin_g = create_rotary_embeddings(pos, head_dim)
            inputs.update(
                {
                    "rotary_cos_local": cos_l,
                    "rotary_sin_local": sin_l,
                    "rotary_cos_global": cos_g,
                    "rotary_sin_global": sin_g,
                }
            )
        return inputs

    def predict(
        self,
        hidden_states: np.ndarray,
        per_layer_inputs: Optional[np.ndarray],
        pos: int,
        ctx_len: int,
        head_dim: int,
    ) -> np.ndarray:
        current_pos = np.array([pos], dtype=np.int32)
        kv_cache = None
        if self.state_name and self.states[0] is not None:
            kv_cache = self.states[0].read_state(self.state_name)

        for i, (model, state) in enumerate(zip(self.models, self.states)):
            if state is not None and kv_cache is not None and i > 0:
                state.write_state(self.state_name, kv_cache)

            inputs = self._build_inputs(hidden_states, per_layer_inputs, pos, ctx_len, head_dim, current_pos)
            if state is not None:
                out = model.predict(inputs, state)
                kv_cache = state.read_state(self.state_name) if self.state_name else kv_cache
            else:
                out = model.predict(inputs)
            hidden_states = out["output_hidden_states"]

        if self.state_name and self.states[0] is not None and kv_cache is not None:
            self.states[0].write_state(self.state_name, kv_cache)

        return hidden_states


def resolve_path(base: Path, rel: Optional[str], fallback: Optional[str]) -> Optional[Path]:
    if rel:
        candidate = base / rel
        if candidate.exists():
            return candidate
    if fallback:
        candidate = base / fallback
        if candidate.exists():
            return candidate
    return None


def find_infer_rotate_chunks(bundle_dir: Path) -> List[Path]:
    return sorted(bundle_dir.glob("*infer_rotate_chunk_*of*.mlpackage"))


def main() -> None:
    parser = argparse.ArgumentParser(description="CoreML-only chat runner")
    parser.add_argument("--bundle", help="Directory containing CoreML artifacts")
    parser.add_argument("--meta", help="Path to meta.yaml (optional)")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--debug-rotation", action="store_true")
    args = parser.parse_args()

    if not args.bundle and not args.meta:
        raise ValueError("Provide --bundle or --meta.")

    meta_path = Path(args.meta) if args.meta else None
    meta = load_meta(meta_path)
    base_dir = meta_path.parent if meta_path else Path(args.bundle)
    bundle_dir = Path(args.bundle) if args.bundle else base_dir

    context_length = args.context_length or meta.get("context_length", 512)
    softcap = float(meta.get("final_logit_softcapping", 30.0))

    infer_init_path = resolve_path(
        bundle_dir, meta.get("infer_init_path"), "gemma3n_infer_init.mlpackage"
    )
    if not infer_init_path:
        raise FileNotFoundError("Missing infer init model")

    infer_chunks = meta.get("infer_chunks")
    if infer_chunks:
        infer_chunk_paths = [bundle_dir / p for p in infer_chunks]
    else:
        infer_chunk_paths = sorted(bundle_dir.glob("*infer_chunk_*of*.mlpackage"))

    infer_path = resolve_path(bundle_dir, meta.get("infer_path"), "gemma3n_infer.mlpackage")

    # Optional rotate models
    rotate_chunks = meta.get("infer_rotate_chunks")
    if rotate_chunks:
        infer_rotate_paths = [bundle_dir / p for p in rotate_chunks]
    else:
        infer_rotate_paths = find_infer_rotate_chunks(bundle_dir)
    infer_rotate_path = resolve_path(bundle_dir, meta.get("infer_rotate_path"), "gemma3n_infer_rotate.mlpackage")

    if infer_chunk_paths:
        infer_models = [load_mlpackage(p) for p in infer_chunk_paths]
        runner = ChunkedInferRunner(infer_models)
    elif infer_path:
        infer_models = [load_mlpackage(infer_path)]
        runner = ChunkedInferRunner(infer_models)
    else:
        raise FileNotFoundError("No infer model(s) found.")

    rotate_runner = None
    if infer_rotate_paths:
        rotate_models = [load_mlpackage(p) for p in infer_rotate_paths]
        rotate_runner = ChunkedInferRunner(rotate_models)
    elif infer_rotate_path:
        rotate_runner = ChunkedInferRunner([load_mlpackage(infer_rotate_path)])

    combine_path = resolve_path(
        bundle_dir, meta.get("combine_streams_path"), "gemma3n_combine_streams.mlpackage"
    )
    combine_model = load_mlpackage(combine_path) if combine_path else None

    lm_head_path = resolve_path(bundle_dir, meta.get("lm_head_path"), "gemma3n_lm_head.mlpackage")
    if not lm_head_path:
        raise FileNotFoundError("Missing lm_head model")
    lm_head = load_mlpackage(lm_head_path)

    tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir), use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    infer_init = load_mlpackage(infer_init_path)

    token_ids = tokenizer(args.prompt, return_tensors="np")["input_ids"].astype(np.int32)[0].tolist()
    print(f"Prompt: {args.prompt}")

    if runner.kv_shape is not None:
        ctx_len = runner.kv_shape[2]
        head_dim = runner.kv_shape[3]
    else:
        ctx_len = int(context_length)
        head_dim = int(meta.get("hidden_size", 2048)) // int(meta.get("num_attention_heads", 8))

    sliding_window = int(args.sliding_window or meta.get("sliding_window", ctx_len))

    last_hidden = None
    active_runner = runner
    prefill_start = time.time()
    for pos, tok in enumerate(token_ids):
        init_out = infer_init.predict({"input_ids": np.array([[int(tok)]], dtype=np.int32)})
        hidden_states = init_out["hidden_states"]
        per_layer_inputs = init_out.get("per_layer_inputs")
        if rotate_runner is not None and pos >= sliding_window:
            if active_runner is not rotate_runner:
                kv_cache = active_runner.get_kv_cache()
                if kv_cache is not None:
                    rotate_runner.set_kv_cache(kv_cache)
                active_runner = rotate_runner
                if args.debug_rotation:
                    print(f"\n[rotation] prefill switch at pos={pos} (sliding_window={sliding_window})")
        hidden_states = active_runner.predict(hidden_states, per_layer_inputs, pos, ctx_len, head_dim)
        last_hidden = hidden_states

    if last_hidden is None:
        raise RuntimeError("Prefill failed")
    prefill_time = time.time() - prefill_start
    if token_ids:
        prefill_tps = len(token_ids) / max(prefill_time, 1e-6)
        print(f"Prefill: {len(token_ids)} tokens in {prefill_time:.2f}s ({prefill_tps:.2f} t/s)")

    def logits_from_hidden(hidden_states: np.ndarray) -> np.ndarray:
        if combine_model is not None:
            hidden_states = combine_model.predict({"hidden_states": hidden_states})["output_hidden_states"]
        lm_out = lm_head.predict({"hidden_states": hidden_states.astype(np.float16)})
        logits = concat_logits(lm_out)[0, 0]
        return apply_logit_softcap(logits, softcap)

    logits = logits_from_hidden(last_hidden)
    next_id = sample_next_token(logits, args.temperature, args.top_k)
    decode_tokens = 0
    decode_start = None
    if args.max_new_tokens > 0:
        decode_start = time.time()
        print(tokenizer.decode([next_id]), end="", flush=True)
        decode_tokens += 1

    current_pos = len(token_ids)
    for _ in range(max(args.max_new_tokens - 1, 0)):
        init_out = infer_init.predict({"input_ids": np.array([[int(next_id)]], dtype=np.int32)})
        hidden_states = init_out["hidden_states"]
        per_layer_inputs = init_out.get("per_layer_inputs")
        if rotate_runner is not None and current_pos >= sliding_window:
            if active_runner is not rotate_runner:
                kv_cache = active_runner.get_kv_cache()
                if kv_cache is not None:
                    rotate_runner.set_kv_cache(kv_cache)
                active_runner = rotate_runner
                if args.debug_rotation:
                    print(f"\n[rotation] infer switch at pos={current_pos} (sliding_window={sliding_window})")
        hidden_states = active_runner.predict(hidden_states, per_layer_inputs, current_pos, ctx_len, head_dim)
        logits = logits_from_hidden(hidden_states)
        next_id = sample_next_token(logits, args.temperature, args.top_k)
        current_pos += 1
        print(tokenizer.decode([next_id]), end="", flush=True)
        decode_tokens += 1

    if decode_start is not None and decode_tokens > 0:
        decode_time = time.time() - decode_start
        decode_tps = decode_tokens / max(decode_time, 1e-6)
        print(f"\nDecode: {decode_tokens} tokens in {decode_time:.2f}s ({decode_tps:.2f} t/s)")

    print("\n\nDone.")


if __name__ == "__main__":
    main()

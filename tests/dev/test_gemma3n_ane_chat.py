#!/usr/bin/env python3
"""Standalone Gemma3n ANE chat smoke test.

This script exercises the CoreML artifacts produced by the Gemma3n converter
without relying on tests/chat.py (which expects a different meta.yaml schema).

Note: This is a smoke test. By default it uses embeddings -> FFN -> LM head to
validate CoreML execution and tokenization. If a stateful infer model is
present, you can enable KV-cache single-token inference via --use-infer.
"""

import argparse
import json
from pathlib import Path
from typing import List

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer


def load_mlpackage(path: Path) -> ct.models.MLModel:
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    return ct.models.MLModel(str(path))

def describe_model_io(label: str, model: ct.models.MLModel) -> None:
    spec = model.get_spec()
    inputs = []
    for inp in spec.description.input:
        shape = list(inp.type.multiArrayType.shape)
        inputs.append(f"{inp.name}:{shape}")
    outputs = []
    for out in spec.description.output:
        shape = list(out.type.multiArrayType.shape)
        outputs.append(f"{out.name}:{shape}")
    print(f"  {label} IO: inputs={inputs} outputs={outputs}")

def find_ffn_chunks(bundle_dir: Path) -> List[Path]:
    chunks = sorted(bundle_dir.glob("gemma3n_FFN_chunk_*of*.mlpackage"))
    if not chunks:
        raise FileNotFoundError(f"No FFN chunks found in {bundle_dir}")
    return chunks

def find_infer_chunks(bundle_dir: Path) -> List[Path]:
    chunks = sorted(bundle_dir.glob("gemma3n_infer_chunk_*of*.mlpackage"))
    return chunks

def build_full_input_ids(token_ids: List[int], context_length: int, pad_token_id: int) -> np.ndarray:
    if len(token_ids) > context_length:
        raise ValueError(f"Token length {len(token_ids)} exceeds context length {context_length}")
    input_ids = np.full((1, context_length), pad_token_id, dtype=np.int32)
    input_ids[0, : len(token_ids)] = np.array(token_ids, dtype=np.int32)
    return input_ids


def concat_logits(outputs: dict) -> np.ndarray:
    # Prefer split logits if present.
    split_keys = sorted([k for k in outputs.keys() if k.startswith("logits_split_")])
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma3n ANE chat smoke test")
    parser.add_argument("--bundle", default="/tmp/gemma3n-infer/bundle", help="Directory with .mlpackage files and tokenizer")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Max tokens to generate")
    parser.add_argument("--context-length", type=int, default=512, help="Context length used during conversion")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0=disabled)")
    parser.add_argument("--single-token", action="store_true",
                        help="Run a single-token prediction (no prompt history).")
    parser.add_argument("--token-id", type=int, default=None,
                        help="Token id to use for --single-token (defaults to BOS/EOS).")
    parser.add_argument("--top-k-print", type=int, default=5,
                        help="Top-k to print for --single-token.")
    parser.add_argument("--sliding-window", type=int, default=512,
                        help="Sliding window size for local attention mask.")
    parser.add_argument("--use-infer", action="store_true",
                        help="Use stateful Gemma3n infer model with KV cache if available.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print stage-by-stage progress (loading, masks, inference).")
    parser.add_argument("--debug-tensors", action="store_true",
                        help="Print intermediate tensor stats (init/combined/logits).")
    parser.add_argument("--debug-steps", type=int, default=1,
                        help="How many steps to print tensor stats for (default: 1).")
    args = parser.parse_args()
    if not args.verbose:
        args.verbose = True

    bundle = Path(args.bundle)
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle}")

    if args.verbose:
        print(f"Bundle: {bundle}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(bundle), use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    use_infer = args.use_infer
    infer_chunks = []
    infer_chunk_models = None
    infer_path = bundle / "gemma3n_infer.mlpackage"
    infer_init_path = bundle / "gemma3n_infer_init.mlpackage"
    combine_path = bundle / "gemma3n_combine_streams.mlpackage"
    infer_model = None
    infer_init = None
    combine_model = None
    if use_infer:
        if not infer_init_path.exists():
            raise FileNotFoundError(f"Missing infer init model: {infer_init_path}")
        if not combine_path.exists():
            raise FileNotFoundError(f"Missing combine-streams model: {combine_path}")
        if args.verbose:
            print("Loading CoreML models (infer path)...")
        infer_init = load_mlpackage(infer_init_path)
        if args.verbose:
            print(f"  infer init: {infer_init_path}")
            describe_model_io("infer_init", infer_init)
        combine_model = load_mlpackage(combine_path)
        if args.verbose:
            print(f"  combine:    {combine_path}")
            describe_model_io("combine", combine_model)
        infer_chunks = find_infer_chunks(bundle)
        if infer_chunks:
            infer_chunk_models = []
            for p in infer_chunks:
                infer_chunk_models.append(load_mlpackage(p))
                if args.verbose:
                    print(f"  infer:      {p}")
        else:
            if not infer_path.exists():
                raise FileNotFoundError(f"Missing infer model: {infer_path}")
            infer_model = load_mlpackage(infer_path)
            if args.verbose:
                print(f"  infer:      {infer_path}")

    # Load stateless models if not using infer
    if infer_model is None and not infer_chunk_models:
        if args.verbose:
            print("Loading CoreML models (stateless path)...")
        embeddings = load_mlpackage(bundle / "gemma3n_embeddings.mlpackage")
        lm_head = load_mlpackage(bundle / "gemma3n_lm_head.mlpackage")
        ffn_chunks = [load_mlpackage(p) for p in find_ffn_chunks(bundle)]
        if args.verbose:
            print(f"  embeddings: {bundle / 'gemma3n_embeddings.mlpackage'}")
            print(f"  lm_head:    {bundle / 'gemma3n_lm_head.mlpackage'}")
            for p in find_ffn_chunks(bundle):
                print(f"  ffn:        {p}")
    else:
        lm_head = load_mlpackage(bundle / "gemma3n_lm_head.mlpackage")

    if args.single_token:
        token_id = args.token_id
        if token_id is None:
            token_id = tokenizer.bos_token_id
            if token_id is None:
                token_id = tokenizer.eos_token_id or 0
        token_ids = [int(token_id)]
        print(f"Single-token mode: token_id={token_ids[0]}")

        if infer_model is not None or infer_chunks:
            # KV-cache infer path
            causal = np.zeros((1, 1, args.context_length, args.context_length), dtype=np.float16)
            i_idx, j_idx = np.triu_indices(args.context_length, k=1)
            causal[:, :, i_idx, j_idx] = float("-inf")
            init_out = infer_init.predict(
                {"input_ids": np.array([[token_ids[0]]], dtype=np.int32)}
            )
            hidden_states = init_out["hidden_states"]
            per_layer_inputs = init_out["per_layer_inputs"]
            if args.debug_tensors:
                summarize_tensor("init.hidden_states", hidden_states)
                summarize_tensor("init.per_layer_inputs", per_layer_inputs)

            if infer_model is not None:
                state = infer_model.make_state()
                out = infer_model.predict(
                    {
                        "hidden_states": hidden_states,
                        "per_layer_inputs": per_layer_inputs,
                        "causal_mask": causal,
                        "current_pos": np.array([0], dtype=np.int32),
                    },
                    state,
                )
                hidden_states = out["output_hidden_states"]
            else:
                state = infer_chunk_models[0].make_state()
                for model in infer_chunk_models:
                    out = model.predict(
                        {
                            "hidden_states": hidden_states,
                            "per_layer_inputs": per_layer_inputs,
                            "causal_mask": causal,
                            "current_pos": np.array([0], dtype=np.int32),
                        },
                        state,
                    )
                    hidden_states = out["output_hidden_states"]
            if args.debug_tensors:
                summarize_tensor("infer.hidden_states", hidden_states)
            combined = combine_model.predict({"hidden_states": hidden_states})["output_hidden_states"]
            if args.debug_tensors:
                summarize_tensor("combine.hidden_states", combined)
            lm_input = combined
            lm_out = lm_head.predict({"hidden_states": lm_input.astype(np.float16)})
            logits = concat_logits(lm_out)[0, 0]
            if args.debug_tensors:
                summarize_tensor("lm_head.logits", logits)
            topk = max(1, args.top_k_print)
            top_ids = np.argsort(logits)[-topk:][::-1].tolist()
            decoded = [tokenizer.decode([tid]) for tid in top_ids]
            print("Top predictions:")
            for rank, (tid, text) in enumerate(zip(top_ids, decoded), start=1):
                print(f"  {rank}. {tid} -> {repr(text)}")
            print("\nDone.")
            return

        full_ids = build_full_input_ids(token_ids, args.context_length, tokenizer.pad_token_id)

        # Embeddings (conv2d layout: [B, C, T, 1])
        emb_out = embeddings.predict({"input_ids": full_ids})
        hidden_states = emb_out.get("embeddings")
        if hidden_states is None:
            raise KeyError(f"Embeddings output missing. Keys: {list(emb_out.keys())}")

        if args.verbose:
            print("Building masks...")
        # Precomputed masks (global causal + sliding window)
        causal = np.zeros((1, 1, args.context_length, args.context_length), dtype=np.float32)
        i_idx, j_idx = np.triu_indices(args.context_length, k=1)
        causal[:, :, i_idx, j_idx] = float("-inf")

        if args.verbose:
            print("Running embeddings...")
        # FFN chunks (no attention in this smoke test)
        for ffn in ffn_chunks:
            if args.verbose:
                print(f"Running FFN chunk: {ffn}")
            ffn_out = ffn.predict({
                "hidden_states": hidden_states.astype(np.float16),
                "causal_mask": causal,
            })
            hidden_states = ffn_out.get("output_hidden_states")
            if hidden_states is None:
                raise KeyError(f"FFN output missing. Keys: {list(ffn_out.keys())}")

        # Take token hidden state (position 0 for single-token mode)
        last_hidden = hidden_states[:, :, 0, 0]  # [1, hidden]
        lm_input = last_hidden[:, None, :]  # [1, 1, hidden]

        if args.verbose:
            print("Running LM head...")
        lm_out = lm_head.predict({"hidden_states": lm_input.astype(np.float16)})
        logits = concat_logits(lm_out)[0, 0]

        topk = max(1, args.top_k_print)
        top_ids = np.argsort(logits)[-topk:][::-1].tolist()
        decoded = [tokenizer.decode([tid]) for tid in top_ids]
        print("Top predictions:")
        for rank, (tid, text) in enumerate(zip(top_ids, decoded), start=1):
            print(f"  {rank}. {tid} -> {repr(text)}")
        print("\nDone.")
        return

    if infer_model is not None or infer_chunks:
        if args.verbose:
            print("Using stateful infer model with KV cache...")

        causal = np.zeros((1, 1, args.context_length, args.context_length), dtype=np.float16)
        i_idx, j_idx = np.triu_indices(args.context_length, k=1)
        causal[:, :, i_idx, j_idx] = float("-inf")
        if infer_model is not None:
            state = infer_model.make_state()
        else:
            state = infer_chunk_models[0].make_state()

        # Encode prompt
        input_ids = tokenizer(args.prompt, return_tensors="np")["input_ids"].astype(np.int32)
        token_ids = input_ids[0].tolist()

        print(f"Prompt tokens: {len(token_ids)}")
        print("Generating...\n")

        last_hidden = None
        for pos, tok in enumerate(token_ids):
            init_out = infer_init.predict(
                {"input_ids": np.array([[int(tok)]], dtype=np.int32)}
            )
            hidden_states = init_out["hidden_states"]
            per_layer_inputs = init_out["per_layer_inputs"]
            if args.debug_tensors and pos < args.debug_steps:
                summarize_tensor(f"prefill[{pos}].init.hidden_states", hidden_states)
                summarize_tensor(f"prefill[{pos}].init.per_layer_inputs", per_layer_inputs)

            if infer_model is not None:
                out = infer_model.predict(
                    {
                        "hidden_states": hidden_states,
                        "per_layer_inputs": per_layer_inputs,
                        "causal_mask": causal,
                        "current_pos": np.array([pos], dtype=np.int32),
                    },
                    state,
                )
                hidden_states = out["output_hidden_states"]
            else:
                for model in infer_chunk_models:
                    out = model.predict(
                        {
                            "hidden_states": hidden_states,
                            "per_layer_inputs": per_layer_inputs,
                            "causal_mask": causal,
                            "current_pos": np.array([pos], dtype=np.int32),
                        },
                        state,
                    )
                    hidden_states = out["output_hidden_states"]
            if args.debug_tensors and pos < args.debug_steps:
                summarize_tensor(f"prefill[{pos}].infer.hidden_states", hidden_states)
            last_hidden = combine_model.predict({"hidden_states": hidden_states})["output_hidden_states"]
            if args.debug_tensors and pos < args.debug_steps:
                summarize_tensor(f"prefill[{pos}].combine.hidden_states", last_hidden)

        if last_hidden is None:
            raise RuntimeError("Prefill failed to produce hidden states")
        lm_out = lm_head.predict({"hidden_states": last_hidden.astype(np.float16)})
        logits = concat_logits(lm_out)[0, 0]
        next_id = sample_next_token(logits, args.temperature, args.top_k)
        token_ids.append(next_id)
        decoded = tokenizer.decode([next_id])
        print(decoded, end="", flush=True)

        current_pos = len(token_ids) - 1
        gen_step = 0
        for _ in range(args.max_new_tokens - 1):
            init_out = infer_init.predict(
                {"input_ids": np.array([[int(token_ids[-1])]], dtype=np.int32)}
            )
            hidden_states = init_out["hidden_states"]
            per_layer_inputs = init_out["per_layer_inputs"]
            if args.debug_tensors and gen_step < args.debug_steps:
                summarize_tensor(f"gen[{gen_step}].init.hidden_states", hidden_states)
                summarize_tensor(f"gen[{gen_step}].init.per_layer_inputs", per_layer_inputs)

            if infer_model is not None:
                out = infer_model.predict(
                    {
                        "hidden_states": hidden_states,
                        "per_layer_inputs": per_layer_inputs,
                        "causal_mask": causal,
                        "current_pos": np.array([current_pos], dtype=np.int32),
                    },
                    state,
                )
                hidden_states = out["output_hidden_states"]
            else:
                for model in infer_chunk_models:
                    out = model.predict(
                        {
                            "hidden_states": hidden_states,
                            "per_layer_inputs": per_layer_inputs,
                            "causal_mask": causal,
                            "current_pos": np.array([current_pos], dtype=np.int32),
                        },
                        state,
                    )
                    hidden_states = out["output_hidden_states"]
            if args.debug_tensors and gen_step < args.debug_steps:
                summarize_tensor(f"gen[{gen_step}].infer.hidden_states", hidden_states)
            combined = combine_model.predict({"hidden_states": hidden_states})["output_hidden_states"]
            if args.debug_tensors and gen_step < args.debug_steps:
                summarize_tensor(f"gen[{gen_step}].combine.hidden_states", combined)
            lm_out = lm_head.predict({"hidden_states": combined.astype(np.float16)})
            logits = concat_logits(lm_out)[0, 0]
            if args.debug_tensors and gen_step < args.debug_steps:
                summarize_tensor(f"gen[{gen_step}].lm_head.logits", logits)
            next_id = sample_next_token(logits, args.temperature, args.top_k)
            token_ids.append(next_id)
            current_pos += 1
            decoded = tokenizer.decode([next_id])
            print(decoded, end="", flush=True)
            gen_step += 1

        print("\n\nDone.")
        return

    # Encode prompt
    input_ids = tokenizer(args.prompt, return_tensors="np")["input_ids"].astype(np.int32)
    token_ids = input_ids[0].tolist()

    print(f"Prompt tokens: {len(token_ids)}")
    print("Generating...\n")

    for _ in range(args.max_new_tokens):
        full_ids = build_full_input_ids(token_ids, args.context_length, tokenizer.pad_token_id)

        # Embeddings (conv2d layout: [B, C, T, 1])
        emb_out = embeddings.predict({"input_ids": full_ids})
        hidden_states = emb_out.get("embeddings")
        if hidden_states is None:
            raise KeyError(f"Embeddings output missing. Keys: {list(emb_out.keys())}")

        if args.verbose:
            print("Building mask...")
        # Precomputed causal mask
        causal = np.zeros((1, 1, args.context_length, args.context_length), dtype=np.float32)
        i_idx, j_idx = np.triu_indices(args.context_length, k=1)
        causal[:, :, i_idx, j_idx] = float("-inf")

        if args.verbose:
            print("Running embeddings...")
        # FFN chunks (no attention in this smoke test)
        for ffn in ffn_chunks:
            if args.verbose:
                print(f"Running FFN chunk: {ffn}")
            ffn_out = ffn.predict({
                "hidden_states": hidden_states.astype(np.float16),
                "causal_mask": causal,
            })
            hidden_states = ffn_out.get("output_hidden_states")
            if hidden_states is None:
                raise KeyError(f"FFN output missing. Keys: {list(ffn_out.keys())}")

        # Take last position hidden state
        pos = len(token_ids) - 1
        last_hidden = hidden_states[:, :, pos, 0]  # [1, hidden]
        lm_input = last_hidden[:, None, :]  # [1, 1, hidden]

        if args.verbose:
            print("Running LM head...")
        lm_out = lm_head.predict({"hidden_states": lm_input.astype(np.float16)})
        logits = concat_logits(lm_out)[0, 0]

        next_id = sample_next_token(logits, args.temperature, args.top_k)
        token_ids.append(next_id)

        # Stream decode
        decoded = tokenizer.decode([next_id])
        print(decoded, end="", flush=True)

    print("\n\nDone.")


if __name__ == "__main__":
    main()

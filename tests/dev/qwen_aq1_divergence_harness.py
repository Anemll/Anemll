#!/usr/bin/env python3
"""Divergence harness for generating ANE instability datasets.

This script runs PyTorch vs CoreML comparison over multiple prompts,
collecting per-token divergence metrics for training and validation.

Features:
- Tokenwise prompt feed (teacher forcing) - exercises single-token path
- CoreML-driven decode for realistic ANE behavior
- Outputs per-prompt NPZ arrays and JSON summaries
- Aggregate statistics for identifying unstable prompts

Usage:
    python tests/dev/qwen_aq1_divergence_harness.py \
        ~/Downloads/snapped_step1800.pt \
        /Users/anemll/Models/ANE/q4_r32_lut_ka \
        --dataset prompts.jsonl \
        --out-dir runs/exp1 \
        --context-length 1024 \
        --max-new-tokens 256 \
        --driver coreml

Input JSONL format:
    {"id": "prompt_001", "prompt": "What is the capital of France?"}
    {"id": "prompt_002", "prompt": "Explain quantum computing"}

Output:
    runs/exp1/<prompt_id>.npz  - arrays (driver_tokens, pt_argmax, cm_argmax, kl, entropy, etc.)
    runs/exp1/<prompt_id>.json - summary metadata
    runs/exp1/summary.jsonl    - aggregate results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import helpers from the compare script
from tests.dev.test_qwen_aq1_compare import (
    load_coreml_models,
    load_pytorch_model,
    load_tokenizer,
    tokenwise_prompt_feed_teacher,
    pytorch_forward_single,
    coreml_forward_single,
    create_causal_mask,
    compute_stability_metrics,
    compute_repetition_score,
)


def ngram_repeat_rate(tokens, n=4):
    """Compute n-gram repetition rate (0=unique, 1=all repeated)."""
    if len(tokens) < n + 1:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def rolling_repetition(tokens, n=4, window=128):
    """Compute rolling n-gram repetition rate per position."""
    out = np.zeros(len(tokens), dtype=np.float32)
    for i in range(len(tokens)):
        start = max(0, i - window + 1)
        out[i] = ngram_repeat_rate(tokens[start:i+1], n=n)
    return out


def iter_jsonl(path: str):
    """Iterate over JSONL file, yielding (id, prompt, full_obj) tuples."""
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj.get("id", f"line_{i:06d}")
            prompt = obj["prompt"]
            yield pid, prompt, obj


@torch.no_grad()
def run_one_prompt(
    pid: str,
    prompt: str,
    pytorch_model,
    pytorch_config,
    coreml_models,
    coreml_metadata,
    tokenizer,
    max_new_tokens: int,
    driver: str = "coreml",
    stop_on_instability: bool = True,
    no_think: bool = False,
    verbose: bool = False,
):
    """Run divergence analysis on a single prompt.

    Args:
        pid: Prompt ID
        prompt: The prompt text
        pytorch_model: PyTorch QwenInferenceModel
        pytorch_config: QwenConfig
        coreml_models: Tuple of (embed_model, ffn_infer, ffn_prefill, lmhead_model)
        coreml_metadata: Dict with context_length, batch_size, split_lm_head
        tokenizer: HuggingFace tokenizer
        max_new_tokens: Maximum decode tokens
        driver: 'coreml' (realistic ANE) or 'pt' (parity testing)
        stop_on_instability: Stop decode if repetition > 0.30
        no_think: Use /no_think prefix
        verbose: Print progress

    Returns:
        summary: Dict with summary metadata
        arrays: Dict of numpy arrays for NPZ
    """
    embed_model, ffn_infer, ffn_prefill, lmhead_model = coreml_models
    context_length = coreml_metadata["context_length"]
    split_lm_head = coreml_metadata["split_lm_head"]

    # Tokenize
    if no_think:
        messages = [{"role": "user", "content": f"/no_think {prompt}"}]
    else:
        messages = [{"role": "user", "content": prompt}]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    input_ids_np = input_ids.numpy().astype(np.int32)
    seq_len = input_ids_np.shape[1]

    if verbose:
        print(f"  [{pid}] prompt_len={seq_len}")

    # Reset PyTorch KV cache
    if hasattr(pytorch_model.model, "kv_cache_0"):
        pytorch_model.model.kv_cache_0.zero_()

    # Create CoreML state (prefer infer state)
    try:
        coreml_state = ffn_infer.make_state()
    except Exception:
        coreml_state = ffn_prefill.make_state()

    # --- Prompt feed (teacher-forced, tokenwise) ---
    pt_logits_last, cm_logits_last, prompt_metrics, prompt_first_mismatch = tokenwise_prompt_feed_teacher(
        pytorch_model=pytorch_model,
        pytorch_config=pytorch_config,
        embed_model=embed_model,
        ffn_infer=ffn_infer,
        lmhead_model=lmhead_model,
        input_ids_np=input_ids_np,
        context_length=context_length,
        split_lm_head=split_lm_head,
        coreml_state=coreml_state,
        probe_metrics=True,
    )

    # Post-prompt next-token selection
    pt_vec = pt_logits_last[0, -1, :].detach().cpu().numpy()
    cm_vec = cm_logits_last[0, 0, :]
    pt_next = int(np.argmax(pt_vec))
    cm_next = int(np.argmax(cm_vec))
    next_token = cm_next if driver == "coreml" else pt_next

    # --- Decode phase ---
    decode_metrics = []
    driver_tokens = []
    pt_argmax_tokens = []
    cm_argmax_tokens = []

    current_pos = seq_len
    stop_reason = None

    for t in range(max_new_tokens):
        if current_pos >= context_length - 1:
            stop_reason = "context_limit"
            break

        # Check EOS
        if next_token in [151643, 151644, 151645]:
            stop_reason = "eos"
            break

        driver_tokens.append(next_token)

        # PyTorch single-token step
        pt_in = torch.tensor([[next_token]], dtype=torch.long)
        pt_mask = create_causal_mask(1, pytorch_config.state_length, current_pos=current_pos)
        pt_logits = pytorch_forward_single(
            pytorch_model,
            pt_in,
            torch.tensor([current_pos]),
            pt_mask,
            current_pos=current_pos,
            prefill=False,
        )

        # CoreML single-token step
        cm_in = np.array([[next_token]], dtype=np.int32)
        cm_logits = coreml_forward_single(
            embed_model, ffn_infer, lmhead_model,
            cm_in, current_pos, context_length, coreml_state, split_lm_head
        )

        pt_vec = pt_logits[0, -1, :].detach().cpu().numpy()
        cm_vec = cm_logits[0, 0, :]

        m = compute_stability_metrics(pt_vec, cm_vec)
        decode_metrics.append(m)

        pt_next = int(np.argmax(pt_vec))
        cm_next = int(np.argmax(cm_vec))
        pt_argmax_tokens.append(pt_next)
        cm_argmax_tokens.append(cm_next)

        next_token = cm_next if driver == "coreml" else pt_next
        current_pos += 1

        # Instability check: high repetition
        if stop_on_instability:
            rep = compute_repetition_score(driver_tokens, n=4, window=128)
            if rep > 0.30:
                stop_reason = "repetition"
                break

    if stop_reason is None:
        stop_reason = "max_tokens"

    # --- Build outputs ---
    summary = {
        "id": pid,
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "prompt_len": int(seq_len),
        "prompt_first_mismatch_pos": prompt_first_mismatch,
        "prompt_kl_max": float(np.max([x["kl_divergence"] for x in prompt_metrics])) if prompt_metrics else 0.0,
        "prompt_kl_avg": float(np.mean([x["kl_divergence"] for x in prompt_metrics])) if prompt_metrics else 0.0,
        "decode_len": len(decode_metrics),
        "decode_kl_max": float(np.max([x["kl_divergence"] for x in decode_metrics])) if decode_metrics else 0.0,
        "decode_kl_avg": float(np.mean([x["kl_divergence"] for x in decode_metrics])) if decode_metrics else 0.0,
        "driver": driver,
        "stop_reason": stop_reason,
    }

    # Count decode mismatches
    if pt_argmax_tokens and cm_argmax_tokens:
        mismatches = sum(1 for p, c in zip(pt_argmax_tokens, cm_argmax_tokens) if p != c)
        summary["decode_mismatches"] = mismatches
        summary["decode_match_rate"] = 1.0 - (mismatches / len(pt_argmax_tokens)) if pt_argmax_tokens else 1.0

    # Compute rolling 4-gram repetition for decode phase
    decode_rep4 = rolling_repetition(driver_tokens, n=4, window=128) if driver_tokens else np.array([], dtype=np.float32)

    arrays = {
        # Prompt tokens (for replay without re-tokenization)
        "prompt_tokens": input_ids_np.flatten().astype(np.int32),
        # Decode tokens
        "driver_tokens": np.array(driver_tokens, dtype=np.int32),
        "pt_argmax": np.array(pt_argmax_tokens, dtype=np.int32),
        "cm_argmax": np.array(cm_argmax_tokens, dtype=np.int32),
        # Prompt metrics
        "prompt_kl": np.array([x["kl_divergence"] for x in prompt_metrics], dtype=np.float32),
        "prompt_entropy_cm": np.array([x["cm_entropy"] for x in prompt_metrics], dtype=np.float32),
        # Decode metrics
        "decode_kl": np.array([x["kl_divergence"] for x in decode_metrics], dtype=np.float32),
        "decode_entropy_cm": np.array([x["cm_entropy"] for x in decode_metrics], dtype=np.float32),
        "decode_margin_cm": np.array([x["cm_top1_margin"] for x in decode_metrics], dtype=np.float32),
        "decode_maxlogit_cm": np.array([x["cm_max_logit"] for x in decode_metrics], dtype=np.float32),
        "decode_correlation": np.array([x["correlation"] for x in decode_metrics], dtype=np.float32),
        # Rolling repetition (precomputed for convenience)
        "decode_rep4": decode_rep4,
    }

    return summary, arrays


def main():
    parser = argparse.ArgumentParser(
        description="Divergence harness for ANE instability dataset generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("checkpoint", type=str, help="Path to AQ1 checkpoint for PyTorch")
    parser.add_argument("coreml_dir", type=str, help="Path to CoreML model directory")
    parser.add_argument("--dataset", required=True, help="JSONL file with {id, prompt}")
    parser.add_argument("--out-dir", required=True, help="Output directory for NPZ and JSON files")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Context length (default: from CoreML meta.yaml)")
    parser.add_argument("--state-length", type=int, default=None,
                        help="State length for KV cache (default: same as context-length)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum tokens to generate per prompt")
    parser.add_argument("--driver", choices=["coreml", "pt"], default="coreml",
                        help="Driver mode: coreml (realistic ANE) or pt (parity testing)")
    parser.add_argument("--no-think", action="store_true", help="Use /no_think prefix")
    parser.add_argument("--no-stop-on-instability", action="store_true",
                        help="Don't stop on high repetition")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    checkpoint_path = os.path.expanduser(args.checkpoint)
    coreml_dir = os.path.expanduser(args.coreml_dir)
    dataset_path = os.path.expanduser(args.dataset)
    out_dir = Path(args.out_dir)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not os.path.exists(coreml_dir):
        print(f"Error: CoreML directory not found: {coreml_dir}")
        sys.exit(1)

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ANE Divergence Harness")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"CoreML dir: {coreml_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {out_dir}")
    print(f"Driver: {args.driver}")
    print(f"Max new tokens: {args.max_new_tokens}")
    if args.context_length:
        print(f"Context length: {args.context_length} (override)")
    if args.state_length:
        print(f"State length: {args.state_length} (override)")

    # Load models
    print("\n--- Loading CoreML models ---")
    embed_model, ffn_infer, ffn_prefill, lmhead_model, coreml_metadata = load_coreml_models(
        coreml_dir, verbose=True
    )

    if args.context_length is not None:
        coreml_metadata["context_length"] = args.context_length
        print(f"  Overriding context_length: {args.context_length}")

    context_length = coreml_metadata["context_length"]
    state_length = args.state_length if args.state_length is not None else context_length

    print("\n--- Loading PyTorch model ---")
    pytorch_model, pytorch_config = load_pytorch_model(
        checkpoint_path, context_length, state_length=state_length, verbose=True
    )

    print("\n--- Loading tokenizer ---")
    tokenizer = load_tokenizer(coreml_dir)
    print(f"  Vocab size: {len(tokenizer)}")

    # Count total prompts first
    total_prompts = sum(1 for _ in iter_jsonl(dataset_path))

    # Process prompts
    print("\n" + "=" * 60)
    print(f"Processing {total_prompts} prompts...")
    print("=" * 60)

    summaries = []
    t_start = time.time()
    prompt_times = []

    for i, (pid, prompt, _) in enumerate(iter_jsonl(dataset_path), 1):
        t0 = time.time()

        summary, arrays = run_one_prompt(
            pid=pid,
            prompt=prompt,
            pytorch_model=pytorch_model,
            pytorch_config=pytorch_config,
            coreml_models=(embed_model, ffn_infer, ffn_prefill, lmhead_model),
            coreml_metadata=coreml_metadata,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            driver=args.driver,
            stop_on_instability=not args.no_stop_on_instability,
            no_think=args.no_think,
            verbose=args.verbose,
        )

        # Save outputs
        json_path = out_dir / f"{pid}.json"
        npz_path = out_dir / f"{pid}.npz"

        json_path.write_text(json.dumps(summary, indent=2))
        np.savez_compressed(npz_path, **arrays)

        summaries.append(summary)

        elapsed = time.time() - t0
        prompt_times.append(elapsed)

        # Calculate ETA
        avg_time = sum(prompt_times) / len(prompt_times)
        remaining = total_prompts - i
        eta_seconds = avg_time * remaining
        if eta_seconds >= 3600:
            eta_str = f"{eta_seconds/3600:.1f}h"
        elif eta_seconds >= 60:
            eta_str = f"{eta_seconds/60:.1f}m"
        else:
            eta_str = f"{eta_seconds:.0f}s"

        status = "OK" if summary.get("decode_match_rate", 1.0) > 0.9 else "DIVERGED"
        incorrect = summary.get("decode_mismatches", 0)
        print(f"  [{i}/{total_prompts}] {pid}: {status} | prompt={summary['prompt_len']} decode={summary['decode_len']} "
              f"kl_max={summary['decode_kl_max']:.4f} | incorrect={incorrect} | {elapsed:.1f}s | ETA: {eta_str}")

    # Write aggregate summary
    summary_path = out_dir / "summary.jsonl"
    summary_path.write_text("\n".join(json.dumps(s) for s in summaries) + "\n")

    # Print aggregate stats
    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print("AGGREGATE SUMMARY")
    print("=" * 60)
    print(f"Total prompts: {len(summaries)}")
    print(f"Total time: {total_time:.1f}s")

    if summaries:
        prompt_mismatches = sum(1 for s in summaries if s.get("prompt_first_mismatch_pos") is not None)
        decode_diverged = sum(1 for s in summaries if s.get("decode_match_rate", 1.0) < 0.9)
        stopped_repetition = sum(1 for s in summaries if s.get("stop_reason") == "repetition")

        kl_maxes = [s["decode_kl_max"] for s in summaries]
        match_rates = [s.get("decode_match_rate", 1.0) for s in summaries]

        print(f"\nPrompt-phase mismatches: {prompt_mismatches}/{len(summaries)}")
        print(f"Decode-phase diverged (match_rate < 0.9): {decode_diverged}/{len(summaries)}")
        print(f"Stopped due to repetition: {stopped_repetition}/{len(summaries)}")
        print(f"\nDecode KL max: avg={np.mean(kl_maxes):.4f} max={np.max(kl_maxes):.4f}")
        print(f"Decode match rate: avg={np.mean(match_rates):.3f} min={np.min(match_rates):.3f}")

    print(f"\nOutputs saved to: {out_dir}")
    print(f"  - {len(summaries)} NPZ files")
    print(f"  - {len(summaries)} JSON files")
    print(f"  - summary.jsonl")


if __name__ == "__main__":
    main()

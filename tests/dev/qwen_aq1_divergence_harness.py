#!/usr/bin/env python3
"""Divergence harness for generating ANE instability datasets.

This script runs PyTorch vs CoreML comparison over multiple prompts,
collecting per-token divergence metrics for training and validation.

Features:
- Tokenwise prompt feed (teacher forcing) - exercises single-token path
- CoreML-driven decode for realistic ANE behavior
- Multi-signal repetition detection (stuck token, phrase repeat, rep4 spike)
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
    {"id": "prompt_002", "prompt": "Explain quantum computing", "risk": "low"}

Output:
    runs/exp1/<prompt_id>.npz  - arrays (prompt_tokens, driver_tokens, pt_argmax, cm_argmax, kl, entropy, etc.)
    runs/exp1/<prompt_id>.json - summary metadata (with stop_reason, stop_step, stop_details)
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


# =============================================================================
# MULTI-SIGNAL REPETITION DETECTION
# =============================================================================

def _suffix_run_length(tokens: list[int]) -> int:
    """Length of the trailing run of identical tokens."""
    if not tokens:
        return 0
    last = tokens[-1]
    run = 1
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i] != last:
            break
        run += 1
    return run


def _check_consecutive_phrase_repeat(
    tokens: list[int],
    *,
    min_len: int = 8,
    repeats: int = 3,
    max_len: int = 64,
) -> tuple[bool, int | None, list[int] | None]:
    """Detect exact consecutive repetition of a phrase.

    Returns True if there exists some phrase length L >= min_len such that
    the last L tokens are repeated `repeats` times consecutively.

    Example for repeats=3:
        tokens[-3L:-2L] == tokens[-2L:-L] == tokens[-L:]

    Notes:
    - This works on token IDs (no decoding), so it's fast and deterministic.
    - Scans phrase lengths from max_len down to min_len to catch longer loops first.
    """
    if repeats < 2:
        raise ValueError("repeats must be >= 2")

    n = len(tokens)
    if n < min_len * repeats:
        return False, None, None

    max_L = min(max_len, n // repeats)
    for L in range(max_L, min_len - 1, -1):
        block = tokens[-L:]
        ok = True
        for r in range(1, repeats):
            a = -(r + 1) * L
            b = -r * L
            if tokens[a:b] != block:
                ok = False
                break
        if ok:
            return True, L, block

    return False, None, None


def _is_expected_repetition(prompt_meta: dict | None) -> bool:
    """Heuristic: prompts that *request* lots of repetition.

    If your JSONL carries a 'risk' field (recommended), use it to avoid
    stopping early just because the output is repetitive by design.
    """
    if not prompt_meta:
        return False
    risk = str(prompt_meta.get("risk", "")).lower()
    return risk in {"high_repetition", "extreme_repetition"}


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
    prompt_meta: dict | None = None,
    stop_cfg: dict | None = None,
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
        stop_on_instability: Enable early-stop repetition checks
        no_think: Use /no_think prefix
        verbose: Print progress
        prompt_meta: Optional per-prompt metadata from JSONL (category, risk, etc.)
        stop_cfg: Early-stop config dictionary

    Returns:
        summary: Dict with summary metadata
        arrays: Dict of numpy arrays for NPZ
    """
    embed_model, ffn_infer, _ffn_prefill, lmhead_model = coreml_models
    context_length = coreml_metadata["context_length"]
    split_lm_head = coreml_metadata["split_lm_head"]

    # Default early-stop config
    if stop_cfg is None:
        stop_cfg = {
            # Exact phrase repeat: detect loops of length 8+ repeated >=3 times
            "phrase_min_len": 8,
            "phrase_max_len": 64,
            "phrase_repeats": 3,
            # Stuck token: same token repeated 10+ times
            "stuck_token_run": 10,
            # rep4 spike: jumps from <0.15 to >0.40 within 32 tokens
            "rep4_window": 128,
            "rep4_spike_lookback": 32,
            "rep4_spike_low": 0.15,
            "rep4_spike_high": 0.40,
            "enable_rep4_spike": True,
        }

    # Adjust thresholds for high/extreme repetition prompts (avoids false positives)
    # These prompts naturally produce repetitive output - need stricter thresholds
    if _is_expected_repetition(prompt_meta):
        # Disable rep4 spike (these prompts naturally have high rep4)
        stop_cfg["enable_rep4_spike"] = False
        # Stricter phrase repeat: require longer phrases and more repeats
        # to avoid flagging "format boilerplate repeated 3x"
        stop_cfg["phrase_min_len"] = max(stop_cfg.get("phrase_min_len", 8), 16)
        stop_cfg["phrase_repeats"] = max(stop_cfg.get("phrase_repeats", 3), 4)

    # Tokenize - use enable_thinking=False to match chat.py behavior (pre-fills <think></think>)
    messages = [{"role": "user", "content": prompt}]
    template_kwargs = {
        "tokenize": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    if no_think:
        template_kwargs["enable_thinking"] = False

    input_ids = tokenizer.apply_chat_template(messages, **template_kwargs)
    input_ids_np = input_ids.numpy().astype(np.int32)
    seq_len = input_ids_np.shape[1]

    if verbose:
        print(f"  [{pid}] prompt_len={seq_len}")

    # Reset PyTorch KV cache
    if hasattr(pytorch_model.model, "kv_cache_0"):
        pytorch_model.model.kv_cache_0.zero_()

    # Create CoreML state - MUST use ffn_infer state since we always use infer path
    # Do NOT silently fall back to prefill state (incompatible, causes subtle drift)
    try:
        coreml_state = ffn_infer.make_state()
    except Exception as e:
        raise RuntimeError(
            f"[{pid}] Failed to create ffn_infer state: {e}. "
            f"Model: ffn_infer, prompt_len={seq_len}. "
            "The harness requires ffn_infer.make_state() since it uses single-token stepping."
        )

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
    driver_tokens: list[int] = []
    pt_argmax_tokens: list[int] = []
    cm_argmax_tokens: list[int] = []
    rep4_hist: list[float] = []

    current_pos = seq_len
    stop_reason: str | None = None
    stop_step: int | None = None
    stop_details: dict | None = None

    # Track first instability onset (even if we don't stop)
    first_stuck_step: int | None = None
    first_phrase_step: int | None = None
    first_rep4_spike_step: int | None = None
    min_entropy_step: int | None = None
    min_entropy_val: float = float('inf')
    max_margin_step: int | None = None
    max_margin_val: float = 0.0

    for t in range(max_new_tokens):
        if current_pos >= context_length - 1:
            stop_reason = "context_limit"
            stop_step = t
            break

        # Check EOS
        if next_token in [151643, 151644, 151645]:
            stop_reason = "eos"
            stop_step = t
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
            embed_model,
            ffn_infer,
            lmhead_model,
            cm_in,
            current_pos,
            context_length,
            coreml_state,
            split_lm_head,
        )

        pt_vec = pt_logits[0, -1, :].detach().cpu().numpy()
        cm_vec = cm_logits[0, 0, :]

        m = compute_stability_metrics(pt_vec, cm_vec)
        decode_metrics.append(m)

        # Track min entropy / max margin onset (even if we don't stop)
        if m["cm_entropy"] < min_entropy_val:
            min_entropy_val = m["cm_entropy"]
            min_entropy_step = t
        if m["cm_top1_margin"] > max_margin_val:
            max_margin_val = m["cm_top1_margin"]
            max_margin_step = t

        pt_next = int(np.argmax(pt_vec))
        cm_next = int(np.argmax(cm_vec))
        pt_argmax_tokens.append(pt_next)
        cm_argmax_tokens.append(cm_next)

        next_token = cm_next if driver == "coreml" else pt_next
        current_pos += 1

        # Track repetition score every step (useful even if stop_on_instability=False)
        rep4_prev_min = None
        lookback = int(stop_cfg["rep4_spike_lookback"])
        if len(rep4_hist) >= lookback:
            rep4_prev_min = float(np.min(rep4_hist[-lookback:]))

        rep4 = compute_repetition_score(driver_tokens, n=4, window=int(stop_cfg["rep4_window"]))
        rep4_hist.append(float(rep4))

        # Track instability onset (even if we don't stop)
        # 1) Stuck token onset
        run_len = _suffix_run_length(driver_tokens)
        if first_stuck_step is None and run_len >= int(stop_cfg["stuck_token_run"]):
            first_stuck_step = t

        # 2) Phrase repeat onset
        ok_phrase, _, _ = _check_consecutive_phrase_repeat(
            driver_tokens,
            min_len=int(stop_cfg["phrase_min_len"]),
            repeats=int(stop_cfg["phrase_repeats"]),
            max_len=int(stop_cfg["phrase_max_len"]),
        )
        if first_phrase_step is None and ok_phrase:
            first_phrase_step = t

        # 3) Rep4 spike onset
        if first_rep4_spike_step is None and rep4_prev_min is not None:
            low = float(stop_cfg["rep4_spike_low"])
            high = float(stop_cfg["rep4_spike_high"])
            if rep4_prev_min < low and rep4 > high:
                first_rep4_spike_step = t

        # Multi-signal early-stop
        if stop_on_instability:
            # 1) Stuck token (very reliable)
            run_len = _suffix_run_length(driver_tokens)
            if run_len >= int(stop_cfg["stuck_token_run"]):
                stop_reason = "repetition_stuck_token"
                stop_step = t
                stop_details = {
                    "token": int(driver_tokens[-1]),
                    "run_len": int(run_len),
                }
                if verbose:
                    print(f"    [STOP] Stuck token: {driver_tokens[-1]} repeated {run_len}x")
                break

            # 2) Exact phrase repeated consecutively (reliable)
            ok, phrase_len, phrase_tokens = _check_consecutive_phrase_repeat(
                driver_tokens,
                min_len=int(stop_cfg["phrase_min_len"]),
                repeats=int(stop_cfg["phrase_repeats"]),
                max_len=int(stop_cfg["phrase_max_len"]),
            )
            if ok:
                stop_reason = "repetition_phrase"
                stop_step = t
                stop_details = {
                    "phrase_len": int(phrase_len or 0),
                    "repeats": int(stop_cfg["phrase_repeats"]),
                    "phrase_tokens": [int(x) for x in (phrase_tokens or [])],
                }
                if verbose:
                    print(f"    [STOP] Phrase repeat: {phrase_len}-token phrase repeated {stop_cfg['phrase_repeats']}x")
                break

            # 3) rep4 spike (early warning; disable for prompts that explicitly request repetition)
            enable_spike = bool(stop_cfg.get("enable_rep4_spike", True))
            if enable_spike and _is_expected_repetition(prompt_meta):
                enable_spike = False

            if enable_spike and rep4_prev_min is not None:
                low = float(stop_cfg["rep4_spike_low"])
                high = float(stop_cfg["rep4_spike_high"])
                if rep4_prev_min < low and rep4 > high:
                    stop_reason = "repetition_rep4_spike"
                    stop_step = t
                    stop_details = {
                        "rep4_prev_min": float(rep4_prev_min),
                        "rep4_current": float(rep4),
                        "lookback": int(lookback),
                        "low": float(low),
                        "high": float(high),
                    }
                    if verbose:
                        print(f"    [STOP] Rep4 spike: {rep4_prev_min:.2f} -> {rep4:.2f} in {lookback} steps")
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
        "stop_step": int(stop_step) if stop_step is not None else None,
        "stop_details": stop_details,
        # Instability onset tracking (even if we didn't stop)
        "first_stuck_step": first_stuck_step,
        "first_phrase_step": first_phrase_step,
        "first_rep4_spike_step": first_rep4_spike_step,
        "min_entropy_step": min_entropy_step,
        "min_entropy_val": float(min_entropy_val) if min_entropy_val != float('inf') else None,
        "max_margin_step": max_margin_step,
        "max_margin_val": float(max_margin_val) if max_margin_val > 0 else None,
    }

    # Compute first_instability_decode_step and global_pos for window extraction
    onset_steps = [s for s in [first_stuck_step, first_phrase_step, first_rep4_spike_step] if s is not None]
    if onset_steps:
        first_instability_decode_step = min(onset_steps)
        summary["first_instability_decode_step"] = first_instability_decode_step
        # Global position = prompt_len + decode_step (for token window extraction)
        summary["first_instability_global_pos"] = int(seq_len) + first_instability_decode_step
    else:
        summary["first_instability_decode_step"] = None
        summary["first_instability_global_pos"] = None

    # Include optional metadata (small + stable fields only)
    if prompt_meta:
        if "category" in prompt_meta:
            summary["category"] = prompt_meta["category"]
        if "risk" in prompt_meta:
            summary["risk"] = prompt_meta["risk"]

    # Count decode mismatches
    if pt_argmax_tokens and cm_argmax_tokens:
        mismatches = sum(1 for p, c in zip(pt_argmax_tokens, cm_argmax_tokens) if p != c)
        summary["decode_mismatches"] = mismatches
        summary["decode_match_rate"] = 1.0 - (mismatches / len(pt_argmax_tokens)) if pt_argmax_tokens else 1.0

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
        "decode_rep4": np.array(rep4_hist, dtype=np.float32),
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
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Context length (default: from CoreML meta.yaml)",
    )
    parser.add_argument(
        "--state-length",
        type=int,
        default=None,
        help="State length for KV cache (default: same as context-length)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--driver",
        choices=["coreml", "pt"],
        default="coreml",
        help="Driver mode: coreml (realistic ANE) or pt (parity testing)",
    )
    parser.add_argument("--no-think", action="store_true", help="Disable thinking mode (uses enable_thinking=False)")
    parser.add_argument(
        "--no-stop-on-instability",
        action="store_true",
        help="Don't stop early on repetition/instability signals",
    )

    # Early-stop repetition config
    parser.add_argument("--stop-phrase-min-len", type=int, default=8,
                        help="Min phrase length (tokens) for exact-repeat stop")
    parser.add_argument("--stop-phrase-max-len", type=int, default=64,
                        help="Max phrase length (tokens) to scan for exact-repeat stop")
    parser.add_argument("--stop-phrase-repeats", type=int, default=3,
                        help="How many consecutive repeats to trigger exact-phrase stop")
    parser.add_argument("--stop-stuck-token-run", type=int, default=10,
                        help="Stop if the same token repeats this many times in a row")

    parser.add_argument("--stop-rep4-window", type=int, default=128,
                        help="Window size for rolling rep4 computation")
    parser.add_argument("--stop-rep4-spike-lookback", type=int, default=32,
                        help="Lookback steps for rep4 spike detection")
    parser.add_argument("--stop-rep4-spike-low", type=float, default=0.15,
                        help="rep4 must have been below this within lookback")
    parser.add_argument("--stop-rep4-spike-high", type=float, default=0.40,
                        help="rep4 must exceed this to count as a spike")
    parser.add_argument("--no-stop-rep4-spike", action="store_true",
                        help="Disable rep4-spike early-stop (keep stuck-token + exact-phrase)")

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

    stop_cfg = {
        "phrase_min_len": args.stop_phrase_min_len,
        "phrase_max_len": args.stop_phrase_max_len,
        "phrase_repeats": args.stop_phrase_repeats,
        "stuck_token_run": args.stop_stuck_token_run,
        "rep4_window": args.stop_rep4_window,
        "rep4_spike_lookback": args.stop_rep4_spike_lookback,
        "rep4_spike_low": args.stop_rep4_spike_low,
        "rep4_spike_high": args.stop_rep4_spike_high,
        "enable_rep4_spike": not args.no_stop_rep4_spike,
    }

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

    print("\nEarly-stop config:")
    print(f"  exact phrase: min_len={stop_cfg['phrase_min_len']} max_len={stop_cfg['phrase_max_len']} repeats={stop_cfg['phrase_repeats']}")
    print(f"  stuck token:  run>={stop_cfg['stuck_token_run']}")
    print(f"  rep4 spike:   window={stop_cfg['rep4_window']} lookback={stop_cfg['rep4_spike_lookback']} low={stop_cfg['rep4_spike_low']} high={stop_cfg['rep4_spike_high']} enabled={stop_cfg['enable_rep4_spike']}")

    # Load models
    print("\n--- Loading CoreML models ---")
    embed_model, ffn_infer, ffn_prefill, lmhead_model, coreml_metadata = load_coreml_models(coreml_dir, verbose=True)

    if args.context_length is not None:
        coreml_metadata["context_length"] = args.context_length
        print(f"  Overriding context_length: {args.context_length}")

    context_length = coreml_metadata["context_length"]
    state_length = args.state_length if args.state_length is not None else context_length

    print("\n--- Loading PyTorch model ---")
    pytorch_model, pytorch_config = load_pytorch_model(checkpoint_path, context_length, state_length=state_length, verbose=True)

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

    for i, (pid, prompt, meta) in enumerate(iter_jsonl(dataset_path), 1):
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
            prompt_meta=meta,
            stop_cfg=stop_cfg,
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

        # Granular status thresholds with short decode guard
        match_rate = summary.get("decode_match_rate", 1.0)
        decode_len = summary.get("decode_len", 0)
        mismatches = summary.get("decode_mismatches", 0)

        # Short decode guard: <= 1 mismatch on short sequences is effectively OK
        if decode_len < 64 and mismatches <= 1:
            status = "OK"
        elif match_rate >= 0.98:
            status = "OK"
        elif match_rate >= 0.95:
            status = "NEAR"  # Near boundary - worth watching
        else:
            status = "DIVERGED"

        incorrect = summary.get("decode_mismatches", 0)
        stop_reason = summary.get("stop_reason")
        print(
            f"  [{i}/{total_prompts}] {pid}: {status} | prompt={summary['prompt_len']} decode={summary['decode_len']} "
            f"kl_max={summary['decode_kl_max']:.4f} | incorrect={incorrect} | stop={stop_reason} | {elapsed:.1f}s | ETA: {eta_str}"
        )

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

        # Granular decode status counts (with short decode guard)
        def get_status(s):
            mr = s.get("decode_match_rate", 1.0)
            dl = s.get("decode_len", 0)
            mm = s.get("decode_mismatches", 0)
            if dl < 64 and mm <= 1:
                return "OK"
            elif mr >= 0.98:
                return "OK"
            elif mr >= 0.95:
                return "NEAR"
            else:
                return "DIVERGED"

        decode_ok = sum(1 for s in summaries if get_status(s) == "OK")
        decode_near = sum(1 for s in summaries if get_status(s) == "NEAR")
        decode_diverged = sum(1 for s in summaries if get_status(s) == "DIVERGED")

        # Stop reasons
        stop_reason_counts: dict[str, int] = {}
        for s in summaries:
            r = str(s.get("stop_reason", ""))
            stop_reason_counts[r] = stop_reason_counts.get(r, 0) + 1

        stopped_repetition = sum(
            1 for s in summaries if str(s.get("stop_reason", "")).startswith("repetition")
        )

        # Instability onset counts (even if didn't stop)
        had_stuck = sum(1 for s in summaries if s.get("first_stuck_step") is not None)
        had_phrase = sum(1 for s in summaries if s.get("first_phrase_step") is not None)
        had_spike = sum(1 for s in summaries if s.get("first_rep4_spike_step") is not None)

        kl_maxes = [s["decode_kl_max"] for s in summaries]
        match_rates = [s.get("decode_match_rate", 1.0) for s in summaries]

        print(f"\nPrompt-phase mismatches: {prompt_mismatches}/{len(summaries)}")
        print(f"\nDecode-phase status:")
        print(f"  OK (>=98%):        {decode_ok}/{len(summaries)}")
        print(f"  NEAR (95-98%):     {decode_near}/{len(summaries)}")
        print(f"  DIVERGED (<95%):   {decode_diverged}/{len(summaries)}")
        print(f"\nStopped due to repetition signals: {stopped_repetition}/{len(summaries)}")
        print(f"\nInstability onset detected (even if didn't stop):")
        print(f"  Stuck token:  {had_stuck}/{len(summaries)}")
        print(f"  Phrase repeat: {had_phrase}/{len(summaries)}")
        print(f"  Rep4 spike:   {had_spike}/{len(summaries)}")

        # Breakdown
        print("\nStop reason breakdown:")
        for k in sorted(stop_reason_counts.keys()):
            print(f"  {k:24s} {stop_reason_counts[k]}")

        print(f"\nDecode KL max: avg={np.mean(kl_maxes):.4f} max={np.max(kl_maxes):.4f}")
        print(f"Decode match rate: avg={np.mean(match_rates):.3f} min={np.min(match_rates):.3f}")

    print(f"\nOutputs saved to: {out_dir}")
    print(f"  - {len(summaries)} NPZ files")
    print(f"  - {len(summaries)} JSON files")
    print(f"  - summary.jsonl")


if __name__ == "__main__":
    main()

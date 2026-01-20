#!/usr/bin/env python3
"""Analyze divergence harness runs and generate instability datasets.

Reads NPZ + JSON outputs from qwen_aq1_divergence_harness.py and produces:
- train_instability.jsonl  - prompts that triggered instability
- val_instability.jsonl    - held-out unstable prompts (deterministic split)
- stable_control.jsonl     - prompts that remained stable
- near_boundary.jsonl      - prompts at match_rate 95-98% (worth watching)
- metrics.csv              - sortable summary of all runs

Instability labeling rules:
- Uses stop_reason from harness (repetition_stuck_token, repetition_phrase, repetition_rep4_spike)
- Also detects streak-based patterns in NPZ data:
  - Repetition loop: decode_rep4 > 0.30 for ≥8 consecutive steps
  - Entropy collapse: decode_entropy_cm < 0.5 for ≥8 consecutive steps
  - Margin explosion: decode_margin_cm > 20 for ≥4 consecutive steps
  - Logit explosion: decode_maxlogit_cm > threshold

Usage:
    python tests/dev/analyze_runs.py runs/exp1 --output datasets/exp1

"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def deterministic_split(prompt_id: str, val_fraction: float = 0.1) -> str:
    """Deterministically assign prompt to train/val based on hash of ID.

    Returns 'val' if hash(id) % 100 < val_fraction * 100, else 'train'.
    This ensures reproducible splits regardless of processing order.
    """
    h = hashlib.md5(prompt_id.encode()).hexdigest()
    bucket = int(h[:8], 16) % 100
    return "val" if bucket < (val_fraction * 100) else "train"


def find_streak(arr, threshold, min_streak, comparator='gt'):
    """Find first position where condition holds for min_streak consecutive steps.

    Args:
        arr: 1D array
        threshold: value to compare against
        min_streak: minimum consecutive steps
        comparator: 'gt' (>), 'lt' (<), 'gte' (>=), 'lte' (<=)

    Returns:
        First index where streak starts, or None
    """
    if len(arr) < min_streak:
        return None

    if comparator == 'gt':
        mask = arr > threshold
    elif comparator == 'lt':
        mask = arr < threshold
    elif comparator == 'gte':
        mask = arr >= threshold
    elif comparator == 'lte':
        mask = arr <= threshold
    else:
        raise ValueError(f"Unknown comparator: {comparator}")

    streak_count = 0
    for i, val in enumerate(mask):
        if val:
            streak_count += 1
            if streak_count >= min_streak:
                return i - min_streak + 1  # Start of streak
        else:
            streak_count = 0
    return None


def _is_expected_repetition(summary: dict) -> bool:
    """Returns True if this prompt has risk=high_repetition or extreme_repetition.

    These prompts intentionally request repetitive output, so low entropy
    and high rep4 are expected behavior, not instability.
    """
    risk = str(summary.get("risk", "")).lower()
    return risk in {"high_repetition", "extreme_repetition"}


def analyze_npz(npz_path: Path, json_path: Path, config: dict) -> dict:
    """Analyze a single NPZ file and return metrics + instability labels."""
    data = np.load(npz_path)

    with open(json_path) as f:
        summary = json.load(f)

    # Check if this prompt expects repetitive output
    expected_rep = _is_expected_repetition(summary)

    result = {
        "id": summary["id"],
        "prompt": summary.get("prompt", ""),
        "prompt_len": summary["prompt_len"],
        "decode_len": summary["decode_len"],
        "stop_reason": summary.get("stop_reason", "unknown"),
        "stop_step": summary.get("stop_step"),
        "driver": summary.get("driver", "unknown"),
        "category": summary.get("category"),
        "risk": summary.get("risk"),
        "expected_repetition": expected_rep,
    }

    # Copy onset tracking fields from summary
    result["first_stuck_step"] = summary.get("first_stuck_step")
    result["first_phrase_step"] = summary.get("first_phrase_step")
    result["first_rep4_spike_step"] = summary.get("first_rep4_spike_step")
    result["min_entropy_step"] = summary.get("min_entropy_step")
    result["min_entropy_val"] = summary.get("min_entropy_val")
    result["max_margin_step"] = summary.get("max_margin_step")
    result["max_margin_val"] = summary.get("max_margin_val")

    # Basic metrics
    driver_tokens = data["driver_tokens"]
    pt_argmax = data["pt_argmax"]
    cm_argmax = data["cm_argmax"]

    mismatch_mask = pt_argmax != cm_argmax
    mismatch_indices = np.flatnonzero(mismatch_mask)

    result["mismatch_count"] = int(len(mismatch_indices))
    result["first_mismatch_step"] = int(mismatch_indices[0]) if len(mismatch_indices) else None
    result["match_rate"] = 1.0 - (len(mismatch_indices) / len(pt_argmax)) if len(pt_argmax) else 1.0

    # Decode metrics
    decode_kl = data["decode_kl"]
    decode_entropy_cm = data["decode_entropy_cm"]
    decode_margin_cm = data["decode_margin_cm"]
    decode_maxlogit_cm = data["decode_maxlogit_cm"]
    decode_correlation = data["decode_correlation"]
    decode_rep4 = data.get("decode_rep4", np.array([]))

    result["kl_max"] = float(decode_kl.max()) if len(decode_kl) else 0.0
    result["kl_avg"] = float(decode_kl.mean()) if len(decode_kl) else 0.0
    result["entropy_min"] = float(decode_entropy_cm.min()) if len(decode_entropy_cm) else 0.0
    result["entropy_avg"] = float(decode_entropy_cm.mean()) if len(decode_entropy_cm) else 0.0
    result["margin_max"] = float(decode_margin_cm.max()) if len(decode_margin_cm) else 0.0
    result["maxlogit_max"] = float(decode_maxlogit_cm.max()) if len(decode_maxlogit_cm) else 0.0
    result["corr_min"] = float(decode_correlation.min()) if len(decode_correlation) else 0.0
    result["rep4_max"] = float(decode_rep4.max()) if len(decode_rep4) else 0.0

    # Prompt metrics
    prompt_kl = data.get("prompt_kl", np.array([]))
    result["prompt_kl_max"] = float(prompt_kl.max()) if len(prompt_kl) else 0.0
    result["prompt_first_mismatch"] = summary.get("prompt_first_mismatch_pos")

    # --- Instability detection ---
    instability_flags = []
    instability_step = None

    # Use harness stop_reason as primary signal
    stop_reason = summary.get("stop_reason", "")
    if stop_reason.startswith("repetition_"):
        instability_flags.append(stop_reason)
        if summary.get("stop_step") is not None:
            instability_step = summary["stop_step"]

    # Also detect streak-based patterns in NPZ data (catches cases harness didn't stop on)

    # 1) Repetition loop: decode_rep4 > 0.30 for ≥8 steps
    #    Skip for prompts with expected_repetition (high rep4 is expected)
    if len(decode_rep4) and not expected_rep:
        rep_streak = find_streak(decode_rep4, config["rep_threshold"], config["rep_streak"], 'gt')
        if rep_streak is not None and "repetition_loop" not in instability_flags:
            instability_flags.append("repetition_loop_streak")
            if instability_step is None or rep_streak < instability_step:
                instability_step = rep_streak

    # 2) Entropy collapse: decode_entropy_cm < 0.5 for ≥8 steps
    #    Skip for prompts with expected_repetition (low entropy is expected)
    if len(decode_entropy_cm) and not expected_rep:
        entropy_streak = find_streak(decode_entropy_cm, config["entropy_threshold"], config["entropy_streak"], 'lt')
        if entropy_streak is not None:
            instability_flags.append("entropy_collapse")
            if instability_step is None or entropy_streak < instability_step:
                instability_step = entropy_streak

    # 3) Margin explosion: decode_margin_cm > 20 for ≥4 steps
    if len(decode_margin_cm):
        margin_streak = find_streak(decode_margin_cm, config["margin_threshold"], config["margin_streak"], 'gt')
        if margin_streak is not None:
            instability_flags.append("margin_explosion")
            if instability_step is None or margin_streak < instability_step:
                instability_step = margin_streak

    # 4) Logit explosion: decode_maxlogit_cm > threshold
    if len(decode_maxlogit_cm):
        logit_max = decode_maxlogit_cm.max()
        if logit_max > config["maxlogit_threshold"]:
            instability_flags.append("logit_explosion")
            first_explosion = int(np.argmax(decode_maxlogit_cm > config["maxlogit_threshold"]))
            if instability_step is None or first_explosion < instability_step:
                instability_step = first_explosion

    result["instability_flags"] = instability_flags
    result["is_unstable"] = len(instability_flags) > 0
    result["instability_step"] = instability_step

    # Compute first_instability_step from any onset (harness-tracked or streak-detected)
    onset_steps = [
        s for s in [
            result.get("first_stuck_step"),
            result.get("first_phrase_step"),
            result.get("first_rep4_spike_step"),
            instability_step,
        ] if s is not None
    ]
    result["first_instability_step"] = min(onset_steps) if onset_steps else None

    # Copy global position from summary if available, else compute it
    if summary.get("first_instability_global_pos") is not None:
        result["first_instability_global_pos"] = summary["first_instability_global_pos"]
    elif result["first_instability_step"] is not None:
        result["first_instability_global_pos"] = result["prompt_len"] + result["first_instability_step"]
    else:
        result["first_instability_global_pos"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze divergence runs and generate instability datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("runs_dir", type=str, help="Directory containing NPZ + JSON outputs")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: <runs_dir>/analysis)")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of prompts for validation (default: 0.1, deterministic)")

    # Instability thresholds
    parser.add_argument("--rep-threshold", type=float, default=0.30,
                        help="4-gram repetition threshold (default: 0.30)")
    parser.add_argument("--rep-streak", type=int, default=8,
                        help="Minimum consecutive steps for repetition (default: 8)")
    parser.add_argument("--entropy-threshold", type=float, default=0.1,
                        help="Entropy collapse threshold (default: 0.1 - detects near-zero entropy only)")
    parser.add_argument("--entropy-streak", type=int, default=16,
                        help="Minimum consecutive steps for entropy collapse (default: 16)")
    parser.add_argument("--margin-threshold", type=float, default=20.0,
                        help="Margin explosion threshold (default: 20.0)")
    parser.add_argument("--margin-streak", type=int, default=4,
                        help="Minimum consecutive steps for margin explosion (default: 4)")
    parser.add_argument("--maxlogit-threshold", type=float, default=50.0,
                        help="Max logit explosion threshold (default: 50.0)")

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output) if args.output else runs_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "rep_threshold": args.rep_threshold,
        "rep_streak": args.rep_streak,
        "entropy_threshold": args.entropy_threshold,
        "entropy_streak": args.entropy_streak,
        "margin_threshold": args.margin_threshold,
        "margin_streak": args.margin_streak,
        "maxlogit_threshold": args.maxlogit_threshold,
    }

    print("=" * 60)
    print("Analyzing Divergence Runs")
    print("=" * 60)
    print(f"Runs directory: {runs_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Val fraction: {args.val_fraction} (deterministic hash-based)")
    print(f"\nInstability thresholds:")
    print(f"  Repetition: rep4 > {config['rep_threshold']} for >= {config['rep_streak']} steps")
    print(f"  Entropy:    entropy < {config['entropy_threshold']} for >= {config['entropy_streak']} steps")
    print(f"  Margin:     margin > {config['margin_threshold']} for >= {config['margin_streak']} steps")
    print(f"  Max logit:  maxlogit > {config['maxlogit_threshold']}")

    # Find all NPZ files
    npz_files = sorted(runs_dir.glob("*.npz"))
    print(f"\nFound {len(npz_files)} NPZ files")

    if not npz_files:
        print("No NPZ files found!")
        return

    # Analyze each run
    results = []
    for npz_path in npz_files:
        json_path = npz_path.with_suffix(".json")
        if not json_path.exists():
            print(f"  Warning: Missing JSON for {npz_path.name}, skipping")
            continue

        try:
            result = analyze_npz(npz_path, json_path, config)
            results.append(result)
        except Exception as e:
            print(f"  Error analyzing {npz_path.name}: {e}")

    print(f"Successfully analyzed {len(results)} runs")

    # Categorize results
    unstable = [r for r in results if r["is_unstable"]]
    stable = [r for r in results if not r["is_unstable"]]

    # Status classification with short decode guard
    def get_status(r):
        mr = r.get("match_rate", 1.0)
        dl = r.get("decode_len", 0)
        mm = r.get("mismatch_count", 0)
        # Short decode guard: <= 1 mismatch on short sequences is effectively OK
        if dl < 64 and mm <= 1:
            return "OK"
        elif mr >= 0.98:
            return "OK"
        elif mr >= 0.95:
            return "NEAR"
        else:
            return "DIVERGED"

    # Further split stable into OK vs NEAR boundary
    stable_ok = [r for r in stable if get_status(r) == "OK"]
    near_boundary = [r for r in stable if get_status(r) == "NEAR"]
    diverged_not_flagged = [r for r in stable if get_status(r) == "DIVERGED"]

    print(f"\nResults:")
    print(f"  Unstable (flagged):     {len(unstable)} ({100*len(unstable)/len(results):.1f}%)")
    print(f"  Stable OK (>=98%):      {len(stable_ok)} ({100*len(stable_ok)/len(results):.1f}%)")
    print(f"  Near boundary (95-98%): {len(near_boundary)} ({100*len(near_boundary)/len(results):.1f}%)")
    if diverged_not_flagged:
        print(f"  Diverged but not flagged (<95%): {len(diverged_not_flagged)}")

    # Count instability types
    flag_counts = {}
    for r in unstable:
        for flag in r["instability_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    if flag_counts:
        print(f"\nInstability breakdown:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"  {flag}: {count}")

    # Deterministic train/val split using hash
    train_unstable = [r for r in unstable if deterministic_split(r["id"], args.val_fraction) == "train"]
    val_unstable = [r for r in unstable if deterministic_split(r["id"], args.val_fraction) == "val"]

    print(f"\nTrain/val split (unstable, deterministic hash):")
    print(f"  Train: {len(train_unstable)}")
    print(f"  Val:   {len(val_unstable)}")

    # Write outputs
    def write_jsonl(path, items):
        with open(path, "w") as f:
            for item in items:
                # Clean up for output (remove numpy types)
                clean = {}
                for k, v in item.items():
                    if isinstance(v, (np.floating, np.integer)):
                        clean[k] = float(v) if isinstance(v, np.floating) else int(v)
                    else:
                        clean[k] = v
                f.write(json.dumps(clean) + "\n")

    write_jsonl(output_dir / "train_instability.jsonl", train_unstable)
    write_jsonl(output_dir / "val_instability.jsonl", val_unstable)
    write_jsonl(output_dir / "stable_control.jsonl", stable_ok)
    write_jsonl(output_dir / "near_boundary.jsonl", near_boundary)
    write_jsonl(output_dir / "diverged_unflagged.jsonl", diverged_not_flagged)

    # Count expected_repetition prompts
    expected_rep_prompts = [r for r in results if r.get("expected_repetition")]
    if expected_rep_prompts:
        print(f"\nExpected repetition prompts (entropy/rep4 flags skipped): {len(expected_rep_prompts)}")
        for r in expected_rep_prompts:
            print(f"  - {r['id']}: risk={r.get('risk')}")

    # Write CSV with all metrics (sortable)
    csv_path = output_dir / "metrics.csv"
    fieldnames = [
        "id", "prompt_len", "decode_len", "stop_reason", "stop_step",
        "is_unstable", "instability_flags", "instability_step",
        "first_instability_step", "first_instability_global_pos",
        "match_rate", "mismatch_count", "first_mismatch_step",
        "kl_max", "kl_avg", "entropy_min", "entropy_avg",
        "margin_max", "maxlogit_max", "corr_min", "rep4_max",
        "prompt_kl_max", "prompt_first_mismatch",
        "first_stuck_step", "first_phrase_step", "first_rep4_spike_step",
        "min_entropy_step", "min_entropy_val", "max_margin_step", "max_margin_val",
        "category", "risk", "expected_repetition",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda x: (-x["is_unstable"], x.get("first_instability_step") or 9999, -x["rep4_max"])):
            # Convert instability_flags list to string
            row = dict(r)
            row["instability_flags"] = "|".join(r["instability_flags"]) if r["instability_flags"] else ""
            writer.writerow(row)

    print(f"\nOutputs written to: {output_dir}")
    print(f"  - train_instability.jsonl ({len(train_unstable)} prompts)")
    print(f"  - val_instability.jsonl ({len(val_unstable)} prompts)")
    print(f"  - stable_control.jsonl ({len(stable_ok)} prompts)")
    print(f"  - near_boundary.jsonl ({len(near_boundary)} prompts)")
    if diverged_not_flagged:
        print(f"  - diverged_unflagged.jsonl ({len(diverged_not_flagged)} prompts)")
    print(f"  - metrics.csv (all {len(results)} runs)")

    # Print top unstable prompts by earliest instability onset
    if unstable:
        print(f"\nTop 5 earliest instability onset:")
        for i, r in enumerate(sorted(unstable, key=lambda x: x.get("first_instability_step") or 9999)[:5], 1):
            print(f"  {i}. {r['id']}: onset_step={r.get('first_instability_step')} flags={r['instability_flags']}")


if __name__ == "__main__":
    main()

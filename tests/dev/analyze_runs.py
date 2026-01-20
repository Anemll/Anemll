#!/usr/bin/env python3
"""Analyze divergence harness runs and generate instability datasets.

Reads NPZ + JSON outputs from qwen_aq1_divergence_harness.py and produces:
- train_instability.jsonl  - prompts that triggered instability
- val_instability.jsonl    - held-out unstable prompts
- stable_control.jsonl     - prompts that remained stable
- metrics.csv              - sortable summary of all runs

Instability labeling rules (streak-based):
- Repetition loop: decode_rep4 > 0.30 for ≥8 consecutive steps
- Entropy collapse: decode_entropy_cm < 0.5 for ≥8 consecutive steps
- Margin explosion: decode_margin_cm > 20 for ≥4 consecutive steps
- Logit explosion: decode_maxlogit_cm > threshold (99.9th percentile)

Usage:
    python tests/dev/analyze_runs.py runs/exp1 --output datasets/exp1

"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np


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


def analyze_npz(npz_path: Path, json_path: Path, config: dict) -> dict:
    """Analyze a single NPZ file and return metrics + instability labels."""
    data = np.load(npz_path)

    with open(json_path) as f:
        summary = json.load(f)

    result = {
        "id": summary["id"],
        "prompt": summary.get("prompt", ""),
        "prompt_len": summary["prompt_len"],
        "decode_len": summary["decode_len"],
        "stop_reason": summary.get("stop_reason", "unknown"),
        "driver": summary.get("driver", "unknown"),
    }

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

    # 1) Repetition loop: decode_rep4 > 0.30 for ≥8 steps
    if len(decode_rep4):
        rep_streak = find_streak(decode_rep4, config["rep_threshold"], config["rep_streak"], 'gt')
        if rep_streak is not None:
            instability_flags.append("repetition_loop")
            if instability_step is None or rep_streak < instability_step:
                instability_step = rep_streak

    # 2) Entropy collapse: decode_entropy_cm < 0.5 for ≥8 steps
    if len(decode_entropy_cm):
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

    # 5) Stop reason: if harness stopped due to repetition
    if summary.get("stop_reason") == "repetition":
        if "repetition_loop" not in instability_flags:
            instability_flags.append("early_stop_repetition")

    result["instability_flags"] = instability_flags
    result["is_unstable"] = len(instability_flags) > 0
    result["instability_step"] = instability_step

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
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of unstable prompts for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")

    # Instability thresholds
    parser.add_argument("--rep-threshold", type=float, default=0.30,
                        help="4-gram repetition threshold (default: 0.30)")
    parser.add_argument("--rep-streak", type=int, default=8,
                        help="Minimum consecutive steps for repetition (default: 8)")
    parser.add_argument("--entropy-threshold", type=float, default=0.5,
                        help="Entropy collapse threshold (default: 0.5)")
    parser.add_argument("--entropy-streak", type=int, default=8,
                        help="Minimum consecutive steps for entropy collapse (default: 8)")
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

    # Separate stable vs unstable
    unstable = [r for r in results if r["is_unstable"]]
    stable = [r for r in results if not r["is_unstable"]]

    print(f"\nResults:")
    print(f"  Unstable: {len(unstable)} ({100*len(unstable)/len(results):.1f}%)")
    print(f"  Stable:   {len(stable)} ({100*len(stable)/len(results):.1f}%)")

    # Count instability types
    flag_counts = {}
    for r in unstable:
        for flag in r["instability_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    if flag_counts:
        print(f"\nInstability breakdown:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"  {flag}: {count}")

    # Split unstable into train/val
    random.seed(args.seed)
    random.shuffle(unstable)
    val_size = int(len(unstable) * args.val_split)
    val_unstable = unstable[:val_size]
    train_unstable = unstable[val_size:]

    print(f"\nTrain/val split (unstable only):")
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
    write_jsonl(output_dir / "stable_control.jsonl", stable)

    # Write CSV with all metrics (sortable)
    csv_path = output_dir / "metrics.csv"
    fieldnames = [
        "id", "prompt_len", "decode_len", "stop_reason", "is_unstable",
        "instability_flags", "instability_step",
        "match_rate", "mismatch_count", "first_mismatch_step",
        "kl_max", "kl_avg", "entropy_min", "entropy_avg",
        "margin_max", "maxlogit_max", "corr_min", "rep4_max",
        "prompt_kl_max", "prompt_first_mismatch",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda x: (-x["is_unstable"], -x["rep4_max"], -x["kl_max"])):
            # Convert instability_flags list to string
            row = dict(r)
            row["instability_flags"] = "|".join(r["instability_flags"]) if r["instability_flags"] else ""
            writer.writerow(row)

    print(f"\nOutputs written to: {output_dir}")
    print(f"  - train_instability.jsonl ({len(train_unstable)} prompts)")
    print(f"  - val_instability.jsonl ({len(val_unstable)} prompts)")
    print(f"  - stable_control.jsonl ({len(stable)} prompts)")
    print(f"  - metrics.csv (all {len(results)} runs, sorted by instability)")

    # Print top unstable prompts
    if unstable:
        print(f"\nTop 5 most unstable prompts (by instability_step):")
        for i, r in enumerate(sorted(unstable, key=lambda x: x["instability_step"] or 999)[:5], 1):
            print(f"  {i}. {r['id']}: step={r['instability_step']} flags={r['instability_flags']}")


if __name__ == "__main__":
    main()

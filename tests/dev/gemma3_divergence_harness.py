#!/usr/bin/env python3
"""Batch divergence harness for CoreML vs HF comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from test_gemma3_compare import compare_prompt, _resolve_meta_path


def _read_prompts(path: Path) -> Iterable[str]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                obj = json.loads(line)
                prompt = obj.get("prompt") or obj.get("text") or obj.get("input")
                if prompt:
                    yield prompt
            else:
                yield line


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch divergence harness.")
    parser.add_argument("coreml", type=str, help="Path to CoreML model dir or meta.yaml")
    parser.add_argument("--hf-reference", type=str, required=True, help="HF model ID or path")
    parser.add_argument("--dataset", type=str, required=True, help="JSONL prompts file")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens")
    parser.add_argument("--driver", choices=["coreml", "pt"], default="coreml")
    parser.add_argument("--no-think", action="store_true", help="Disable reasoning via system prompt")
    parser.add_argument("--no-template", action="store_true", help="Disable chat template")
    args = parser.parse_args(argv)

    meta_path = _resolve_meta_path(args.coreml)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    summaries = []
    with results_path.open("w") as out_f:
        for prompt in _read_prompts(Path(args.dataset)):
            summary = compare_prompt(
                meta_path=meta_path,
                hf_reference=args.hf_reference,
                prompt=prompt,
                max_tokens=args.max_new_tokens,
                driver=args.driver,
                no_think=args.no_think,
                use_template=not args.no_template,
            )
            summaries.append(summary)
            out_f.write(json.dumps(summary) + "\n")

    # Aggregate
    def _mean(values):
        return sum(values) / len(values) if values else float("nan")

    agg = {
        "count": len(summaries),
        "kl_mean": _mean([s["metrics"]["kl_mean"] for s in summaries]),
        "corr_mean": _mean([s["metrics"]["corr_mean"] for s in summaries]),
        "entropy_mean": _mean([s["metrics"]["entropy_mean"] for s in summaries]),
        "match_rate": _mean([s["metrics"]["match_rate"] for s in summaries]),
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(agg, f, indent=2)

    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Variable Context Size Demo — Long-form generation on Apple Neural Engine.

Demonstrates ANEMLL's dynamic context growth and overflow handling for reasoning
models that produce long outputs (code generation, multi-step reasoning, essays).

The model starts with a small KV cache (e.g. 512 tokens) and automatically
transitions to larger contexts (1024 → 2048 → 3072 → 4096) as the output grows.
When the largest context fills up, shift-refill compacts the KV cache by keeping
recent tokens and the original prompt, then continues generating.

This enables generating outputs far longer than any single context window —
24,000+ tokens from a 4096-context model — while running entirely on-device
via Apple Neural Engine.

Requirements:
    - macOS with Apple Silicon (M1/M2/M3/M4+)
    - A pre-converted ANEMLL model with multi-context state exports
    - Python environment with: coremltools, transformers, numpy, torch, pyyaml

Usage:
    python examples/variable_context_demo.py --meta <path-to-model>/meta.yaml

    # With custom prompt:
    python examples/variable_context_demo.py \\
        --meta ~/Models/ANE/vibethinker_xstates/meta.yaml \\
        --prompt "Write a game of Tic Tac Toe in Python"

    # With time limit instead of token limit:
    python examples/variable_context_demo.py \\
        --meta ~/Models/ANE/vibethinker_xstates/meta.yaml \\
        --max-time 300

See examples/VARIABLE_CONTEXT.md for full documentation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Resolve repo root so the script works from any directory
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))


DEFAULT_PROMPT = (
    "Write a complete game of Tic Tac Toe in Python. "
    "The program should let a human play against the computer. "
    "Include input validation, a minimax AI opponent, "
    "and display the board after each move."
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Variable context size demo for long-form generation on ANE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --meta ~/Models/ANE/vibethinker_xstates/meta.yaml\n"
            "  %(prog)s --meta ~/Models/ANE/vibethinker_xstates/meta.yaml "
            '--prompt "Explain quantum computing"\n'
            "  %(prog)s --meta ~/Models/ANE/vibethinker_xstates/meta.yaml "
            "--max-time 300\n"
        ),
    )

    # Required
    ap.add_argument(
        "--meta",
        required=True,
        help="Path to meta.yaml of a multi-context state export.",
    )

    # Generation control
    ap.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Input prompt for generation.",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=24000,
        help="Maximum tokens to generate (default: 24000).",
    )
    ap.add_argument(
        "--max-time",
        type=float,
        default=None,
        help=(
            "Maximum wall-clock time in seconds. "
            "When set, --max-tokens is ignored and generation runs until time runs out."
        ),
    )
    ap.add_argument(
        "--max-context-size",
        type=int,
        default=4096,
        help="Cap for active context growth (default: 4096).",
    )

    # Sampling
    ap.add_argument(
        "--sampling-mode",
        choices=["auto", "greedy"],
        default="auto",
        help=(
            "Sampling strategy: 'auto' reads recommended_sampling from meta.yaml, "
            "'greedy' uses argmax (default: auto)."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible sampling (default: 123).",
    )

    # Overflow
    ap.add_argument(
        "--overflow-reserve-batches",
        type=int,
        default=9,
        help=(
            "Number of batch-sized slots to reserve when compacting. "
            "Higher values keep more recent context (default: 9)."
        ),
    )

    # Model behavior
    ap.add_argument(
        "--no-think",
        action="store_true",
        default=True,
        help="Disable thinking mode for Qwen3/VibeThinker (default: on).",
    )
    ap.add_argument(
        "--think",
        action="store_true",
        help="Enable thinking mode (overrides --no-think).",
    )

    # Display
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress/event logs (tokens still stream to stdout).",
    )

    args = ap.parse_args()

    # Resolve meta path
    meta_path = Path(args.meta).expanduser().resolve()
    if not meta_path.exists():
        print(f"Error: meta.yaml not found: {meta_path}", file=sys.stderr)
        return 1

    # Build the command for the state transition runner
    runner = REPO_ROOT / "tests" / "dev" / "state_transition_growing_inference.py"
    if not runner.exists():
        print(
            f"Error: state_transition_growing_inference.py not found at: {runner}",
            file=sys.stderr,
        )
        return 1

    cmd_args = [
        sys.executable,
        str(runner),
        "--meta", str(meta_path),
        "--prompt", args.prompt,
        "--prefill-mode", "batch-prefill",
        "--sampling-mode", args.sampling_mode,
        "--max-context-size", str(args.max_context_size),
        "--overflow-preserve-prompt",
        "--overflow-policy", "shift-refill",
        "--overflow-reserve-batches", str(args.overflow_reserve_batches),
        "--live-events",
        "--seed", str(args.seed),
    ]

    if args.max_time is not None:
        cmd_args.extend(["--max-time", str(args.max_time)])
    else:
        cmd_args.extend(["--max-tokens", str(args.max_tokens)])

    no_think = args.no_think and not args.think
    if no_think:
        cmd_args.append("--no-think")

    if args.quiet:
        cmd_args.extend(["--progress-stream", "none"])

    os.execv(sys.executable, cmd_args)


if __name__ == "__main__":
    raise SystemExit(main())

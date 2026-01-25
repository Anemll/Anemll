# BoolQ (BooQ) Experiment Guide

This document explains the BoolQ/BooQ investigation that lives under `tests/dev`.  
The goal of the work is to validate Apple Neural Engine (ANE) CoreML exports
against MLX and PyTorch references, understand dataset preprocessing differences,
and provide a repeatable regression-style test for BoolQ accuracy.

## Why This Experiment Exists

- **Regression target** – Detect the large ANE‑vs‑MLX accuracy gap that showed up
  on BoolQ/`arc_easy` when running CoreML models.
- **Reproduce lm-eval behavior** – Copy the harness preprocessing (tokenization,
  target delimiters, segmentation) so ANE runs can be compared fairly.
- **Triangulate bugs** – Use PyTorch + Transformers baselines to determine if a
  discrepancy comes from quantization, CoreML runtime, or preprocessing.
- **Automate slicing** – Generate per-segment accuracy tables so the “worst”
  sections of the dataset can be examined in isolation.

## Core Scripts and Their Roles

| Path | Purpose |
| ---- | ------- |
| `evaluate/ane/batch_boolq_segments_ane_vs_mlx.sh` | Canonical end-to-end regression. Drives ANE harness + MLX baseline in matching windows, emits TSV summaries, and records every segment into JSON. |
| `evaluate/ane/_evaluate_with_harness.py` | LM Eval harness adapter that runs the CoreML models strictly serially on ANE/CPU. |
| `tests/dev/mlx_batch_segments.py` | Standalone MLX evaluation that mirrors the official MLX boolq scorer so we can check segments without the ANE harness. |
| `tests/dev/mlx_simple_eval.py` | Lightweight BoolQ scorer for MLX (single window) used by helper scripts. |
| `tests/dev/test_pytorch_qwen25_*` & `tests/dev/test_boolq_transformers_baseline.py` | PyTorch sanity checks using ANEMLL’s custom implementation or vanilla HF Transformers. |
| `tests/dev/full_pytorch_qwen3_evaluate.py` | Multi-purpose PyTorch evaluator for BoolQ + arc_challenge with batching + KV cache support. |
| `tests/dev/compare_*` / `tests/dev/extract_*` / `tests/dev/debug_*` | Focused utilities to diff ANE vs MLX logits, inspect prompts, dump contexts, or reproduce harness steps (see `tests/dev/dev-test-log.MD` for a chronological list). |
| `auto_log_updater.py` / `generate_log.py` | Scripts that snapshot which test files were touched and update `tests/dev/dev-test-log.MD` so the investigation is auditable. |

## Running the Independent BoolQ Test

1. **Prepare models**
   - ANE/CoreML export directory with `meta.yaml`.
   - MLX model id/path (e.g. `Qwen/Qwen2.5-0.5B`).
2. **Execute the batch comparison**

   ```bash
   ./evaluate/ane/batch_boolq_segments_ane_vs_mlx.sh \
     --model /path/to/ane_model \
     --mlx-model Qwen/Qwen2.5-0.5B \
     --step 100 \
     --worst 5 \
     --output-dir results/boolq_regression
   ```

   This will:
   - Call `evaluate_with_harness.py` for each segment on ANE (strictly serial).
   - Call `tests/dev/mlx_simple_eval.py` on the same slice.
   - Emit TSV summaries (`ane_boolq_segments_summary.tsv`,
     `mlx_boolq_segments_summary.tsv`, `ane_vs_mlx_comparison.tsv`).
   - Produce JSON blobs per window and combined outputs for later analysis.

3. **Inspect the output**
   - `ane_vs_mlx_comparison.tsv` lists the worst divergences (ANE–MLX).
   - `results/eval_ane_*` and `results/eval_mlx_*` store the raw harness-style
     JSON for reproducibility.

This script is the recommended “independent test” to gate BoolQ accuracy before
shipping CoreML exports.

## Focused Baselines & Utilities

- **MLX-only regression**  
  `python tests/dev/mlx_batch_segments.py --mlx-model Qwen/Qwen2.5-0.5B --step 200`
  – runs the official MLX scoring loop without touching ANE.

- **PyTorch reference checks**  
  - `python tests/dev/test_boolq_pytorch_baseline.py` – ANEMLL PyTorch model,
    sample #31, prints token probs for `" no"`/`" yes"`.
  - `python tests/dev/test_boolq_transformers_baseline.py` – vanilla HF
    Transformers sanity check on the same prompt.
  - `python tests/dev/full_pytorch_qwen3_evaluate.py --task boolq --limit 20`
    – multi-question evaluator that mirrors the harness interface.

- **Prompt / preprocessing validation**  
  `tests/dev/debug_boolq_prompt.py`, `tests/dev/test_target_delimiter_hypothesis.py`,
  and `tests/dev/replicate_lm_eval_boolq.py` expose the exact dataset formatting,
  making it easy to chase delimiter or tokenization bugs.

- **Result comparison helpers**  
  `tests/dev/compare_mlx_ane_segments.py`, `tests/dev/extract_{ane,mlx}_scores_for_questions.py`,
  and `tests/dev/direct_ane_vs_mlx_comparison.py` operate on the JSON/TSV
  artifacts to highlight mismatched questions.

## Logging Progress

After running experiments, capture what changed inside `tests/dev` so future
contributors know which utilities were touched:

```bash
python auto_log_updater.py --model "ane-qwen25" --segment-size "100"
```

This rewrites `tests/dev/dev-test-log.MD` with timestamps, descriptions, and
argument summaries for the files modified in the last day.

## When to Run Which Script

- **Need a single go/no-go BoolQ regression?** Run
  `batch_boolq_segments_ane_vs_mlx.sh`.
- **Suspect MLX-only regressions?** Use `mlx_batch_segments.py`.
- **Verifying a CoreML export vs PyTorch?** Use the `test_boolq_pytorch_*`
  scripts or `full_pytorch_qwen3_evaluate.py`.
- **Trying to understand preprocessing bugs** (target delimiter, prompt shape,
  etc.)? Run the `debug_*` helpers and refer to `README_segmentation.md`.

This structure keeps the BooQ/BoolQ experiment self-contained, auditable, and
repeatable as we iterate on CoreML exports.

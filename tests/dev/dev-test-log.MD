# Development Test Log

This log tracks daily testing activities. New entries should be added to the top.

---
## 2025-06-28

**07:00:04** `mlx_batch_segments.py`
Model: `dummy`, Segment Size: `0`
Description: Standalone MLX batch segment evaluation for BoolQ.
Runs all segments and exports results for comparison with ANE.

**06:49:44** `test_pytorch_qwen25_optimized.py`
Model: `dummy`, Segment Size: `0`
Description: OPTIMIZED PyTorch Qwen2.5 evaluation for BoolQ - extracts both token probabilities in one pass

files updated:
- `tests/dev//`: `mlx_batch_segments.py`, `test_pytorch_qwen25_optimized.py`

---
## 2025-06-27

**22:36:53** `test_pytorch_qwen25_fixed.py`
Model: `dummy`, Segment Size: `0`
Description: FIXED PyTorch Qwen2.5 segmented evaluation using proper ANEMLL inference pattern
Based on test_final_inference.py as the official/reference implementation

**16:19:43** `test_final_inference.py`
Model: `dummy`, Segment Size: `0`
Description: Final test with properly enabled SP quantization to see if output is now coherent.

**16:14:33** `test_official_hf_model.py`
Model: `dummy`, Segment Size: `0`
Description: Test the official HuggingFace model to see if it also has this bias

**16:14:21** `test_ane_segmented_simple_evaluate.py`
Model: `dummy`, Segment Size: `0`
Description: Test script for ANE segmented evaluation using the full lm-eval preprocessing pipeline.
This uses the existing ANE evaluation script with segmentation support.

**16:14:00** `debug_direct_comparison.py`
Model: `dummy`, Segment Size: `0`
Description: Direct comparison with ANE methodology to find the systematic bias

**16:11:41** `debug_score_patterns.py`
Model: `dummy`, Segment Size: `0`
Description: Debug score patterns to see if there's a systematic bias

**15:56:03** `test_pytorch_qwen25_segmented.py`
Model: `dummy`, Segment Size: `0`
Description: Test script for PyTorch Qwen2.5 segmented evaluation using anemll/models/qwen2_5_model.py
This mirrors the ANE evaluation approach but uses your custom PyTorch model.

**15:54:07** `debug_lm_eval_direct.py`
Model: `dummy`, Segment Size: `0`
Description: Direct test with lm-eval to bypass simple_evaluate_segmented

**15:52:23** `debug_ground_truth.py`
Model: `dummy`, Segment Size: `0`
Description: Debug ground truth values for the first few BoolQ samples

**15:28:46** `quick_pytorch_test.py`
Model: `dummy`, Segment Size: `0`
Description: Quick test of PyTorch implementation on single sample

**15:27:04** `debug_single_pytorch_sample.py`
Model: `dummy`, Segment Size: `0`
Description: Debug single sample with restructured PyTorch implementation

**15:17:26** `debug_prompt_comparison.py`
Model: `dummy`, Segment Size: `0`
Description: Debug by comparing prompts received by PyTorch vs ANE implementations

**15:12:27** `debug_boolq_prompt.py`
Model: `dummy`, Segment Size: `0`
Description: Debug the actual BoolQ prompt format being generated

**15:09:12** `debug_loglikelihood_detailed.py`
Model: `dummy`, Segment Size: `0`
Description: Debug loglikelihood function in detail to see why accuracy is 0%

**15:08:08** `debug_tokenization_issue.py`
Model: `dummy`, Segment Size: `0`
Description: Debug tokenization boundary issues that might cause 0% accuracy

**15:06:38** `debug_boolq_format.py`
Model: `dummy`, Segment Size: `0`
Description: Debug BoolQ format exactly as lm-eval processes it

**15:01:26** `debug_pytorch_single_sample.py`
Model: `dummy`, Segment Size: `0`
Description: Debug single sample PyTorch Qwen2.5 evaluation to see what's going wrong

**12:12:27** `test_pytorch_hf_segmented.py`
Model: `dummy`, Segment Size: `0`
Description: Test script for PyTorch segmented evaluation using lm-eval's native HuggingFace support.
This uses the built-in HFLM wrapper from lm-eval.

**11:14:50** `extract_pytorch_scores_for_questions.py`
Model: `dummy`, Segment Size: `0`
Description: Extract PyTorch Qwen2.5 scores for specific questions to compare with MLX and ANE.
This evaluates the same questions using the ./anemll/models/qwen2_5_model.py implementation.

**10:35:02** `update_candidates_with_ane_results.py`
Model: `dummy`, Segment Size: `0`
Description: Update the candidates file with ANE vs MLX comparison results

**10:28:00** `extract_ane_scores_for_questions.py`
Model: `dummy`, Segment Size: `0`
Description: Extract ANE scores for specific questions to compare directly with MLX scores.
This evaluates the same questions that showed low MLX confidence in segment 1800-1899.

**10:24:07** `extract_segment_differences.py`
Model: `dummy`, Segment Size: `0`
Description: Extract specific questions where MLX and ANE differ in a given segment.
Focus on questions where MLX is correct but ANE is wrong.

**09:52:18** `compare_mlx_ane_segments.py`
Model: `dummy`, Segment Size: `0`
Description: Compare MLX vs ANE segment results using the stored JSON files.
This script analyzes the segment-by-segment differences.

**09:18:33** `test_segmented_simple_evaluate.py`
Model: `dummy`, Segment Size: `0`
Description: Test script to verify our copied simple_evaluate function works correctly
and then test segmentation.

**09:01:40** `README_segmentation.md`
Model: `dummy`, Segment Size: `0`
Description: This document describes the standalone copy of `lm_eval.simple_evaluate` with dataset segmentation support. The `simple_evaluate_with_segmentation.py` file contains a complete copy of the `simple_evaluate` function from lm-evaluation-harness v0.4.9, with additional parameters to support dataset segmentation for parallel evaluation across multiple processes or machines. - **Identical tokenization**: Uses the same tokenizer and preprocessing as the original harness

**08:54:14** `simple_evaluate_with_segmentation.py`
Model: `dummy`, Segment Size: `0`
Description: Standalone copy of lm_eval.simple_evaluate with segmentation support.

This is a complete copy of the simple_evaluate function from lm-evaluation-harness v0.4.9
with additional parameters to support dataset segmentation for parallel evaluation.

Usage:
    from simple_evaluate_with_segmentation import simple_evaluate_segmented
    
    # Evaluate segment 0-99 out of 1000 total examples
    results = simple_evaluate_segmented(
        model=my_model,
        tasks=["boolq"],
        segment_start=0,
        segment_size=100,
        total_dataset_size=1000,
        # ... other parameters
    )

**08:41:40** `replicate_lm_eval_boolq.py`
Model: `dummy`, Segment Size: `0`
Description: Demonstrate how to replicate lm_eval's BoolQ results by using the same preprocessing.
This shows the exact steps needed to match simple_evaluate's accuracy.

**08:40:49** `analyze_lm_eval_preprocessing.py`
Model: `dummy`, Segment Size: `0`
Description: Comprehensive analysis of lm_eval preprocessing for BoolQ task.
This script examines all the preprocessing steps that lm_eval applies.

**08:39:56** `test_target_delimiter_hypothesis.py`
Model: `dummy`, Segment Size: `0`
Description: Test to verify that the target_delimiter (space prefix) is the cause of accuracy difference
between direct loglikelihood calls and simple_evaluate.

This tests the hypothesis that lm_eval adds a space prefix to choices via target_delimiter.

files updated:
- `tests/dev//`: `test_pytorch_qwen25_fixed.py`, `test_final_inference.py`, `test_official_hf_model.py`, `test_ane_segmented_simple_evaluate.py`, `debug_direct_comparison.py`, `debug_score_patterns.py`, `test_pytorch_qwen25_segmented.py`, `debug_lm_eval_direct.py`, `debug_ground_truth.py`, `quick_pytorch_test.py`, `debug_single_pytorch_sample.py`, `debug_prompt_comparison.py`, `debug_boolq_prompt.py`, `debug_loglikelihood_detailed.py`, `debug_tokenization_issue.py`, `debug_boolq_format.py`, `debug_pytorch_single_sample.py`, `test_pytorch_hf_segmented.py`, `extract_pytorch_scores_for_questions.py`, `update_candidates_with_ane_results.py`, `extract_ane_scores_for_questions.py`, `extract_segment_differences.py`, `compare_mlx_ane_segments.py`, `test_segmented_simple_evaluate.py`, `README_segmentation.md`, `simple_evaluate_with_segmentation.py`, `replicate_lm_eval_boolq.py`, `analyze_lm_eval_preprocessing.py`, `test_target_delimiter_hypothesis.py`

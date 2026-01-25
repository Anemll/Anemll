# TODO: Arc Challenge Qwen3 Investigation

## Problem Statement
Arc_challenge evaluation produces incorrect results with Qwen3 model on ANEMLL (Apple Neural Engine). HuggingFace evaluation (MPS) produces correct results. Other models like LLaMA 3 work correctly with arc_challenge.

## Issue Details
- **Model**: Qwen3 0.6B
- **Task**: arc_challenge
- **Platform**: Apple Neural Engine (ANE) on-device
- **Symptom**: Different scores compared to HuggingFace MPS baseline
- **Suspected Causes**: 
  1. Precision issues (FP16 on ANE vs BF16/FP32 on MPS)
  2. Model implementation differences (Qwen3 has biases)
  3. Possible bug in evaluate_with_harness.py specific to Qwen3

## Evidence
```
ANEMLL scores: ['-28.2740', '-30.0654', '-34.6862', '-26.9266']
HF scores:     [["-27.25", "False"], ["-25.0", "False"], ["-36.5", "False"], ["-27.875", "False"]]
```

## Test Commands
```bash
# ANEMLL test (single sample)
python ./evaluate/ane/evaluate_with_harness.py --model /Volumes/Models/ANE/test-qwen25-0.5b-base3 --tasks arc_challenge --skip 9 --limit 1 --verbose-output

# HF test 
lm_eval --model hf --model_args pretrained="Qwen/Qwen2.5-0.5B" --tasks arc_challenge --device mps --batch_size 1 --limit 10 --verbosity DEBUG -s --output_path ./results/
```

## Investigation Plan

### ✅ Phase 1: Precision Isolation
- [x] **Add backend option to chat.py and evaluate_with_harness.py for CPU execution**
  - Added `--backend` option with choices: ANE, CPU, GPU
  - Modified load_model functions to accept backend parameter
  - Updated both chat.py and evaluate_with_harness.py

### 🔄 Phase 2: Model Export and Evaluation
- [ ] **Export Qwen3 model in FP32 precision**
  - Currently limited to np.Float16 and Model_dtype pytorch fp16
  - Need to modify export process to support FP32
  
- [ ] **Create full_pytorch_qwen3_evaluate.py for Qwen3 model evaluation**
  - Based on existing tests/dev/full_pytorch_qwen25_evaluate.py
  - Use qwen_model.py instead of qwen2_5_model.py
  - Support multiple questions with long sequences like arc_challenge

### 🔄 Phase 3: Testing and Comparison
- [ ] **Test single problematic sample (skip 9) with different precisions**
  - Test with CPU backend (no ANE precision issues)
  - Test with FP32 export if available
  - Compare scores across different backends
  
- [ ] **Compare logits between ANE, CPU, and PyTorch implementations**
  - Direct logit comparison for same input
  - Identify where precision differences occur
  - Check if differences are in embeddings, FFN, or LM head

### 🔄 Phase 4: Analysis
- [ ] **Analyze bias handling differences between models**
  - Qwen3 has biases, LLaMA 3 does not
  - Check if bias handling causes precision issues on ANE
  - Compare with PyTorch reference implementation

## Implementation Notes

### Backend Support Added
- Both chat.py and evaluate_with_harness.py now support `--backend` option
- Usage: `python chat.py --backend CPU` or `python evaluate_with_harness.py --backend CPU`
- This allows testing without ANE precision limitations

### Files Modified
- `/tests/chat.py`: Added backend parameter to load_model functions
- `/evaluate/ane/evaluate_with_harness.py`: Added backend support to ANELM class

### Next Steps
1. Create PyTorch evaluation script for Qwen3
2. Test problematic sample with CPU backend
3. Export model in FP32 if possible
4. Compare logits systematically

## Expected Outcomes
- Identify if issue is precision-related (ANE FP16 vs MPS BF16/FP32)
- Determine if bias handling in Qwen3 causes numerical instability
- Provide solution or workaround for correct arc_challenge evaluation

## Files to Create/Modify
- `tests/dev/full_pytorch_qwen3_evaluate.py` (new)
- Modify model export scripts for FP32 support
- Add precision comparison utilities
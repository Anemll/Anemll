# Absolute Question Numbering Implementation

## Summary

Modified both `evaluate_with_harness.py` (ANE) and `full_mlx_evaluate.py` (MLX) to use absolute question numbers instead of relative ones in the `_log_incorrect_answers` function. The absolute question number is calculated as `skip + relative_index + 1`.

## Changes Made

### 1. ANE Implementation (`evaluate/ane/evaluate_with_harness.py`)

#### Constructor Changes:
- Added `skip: int = 0` parameter to `ANELM.__init__()`
- Store skip value as `self.skip`
- Pass skip parameter when initializing model in main()

#### Logging Changes:
- Modified `_log_incorrect_answers()` to calculate absolute question number:
  ```python
  absolute_question_num = self.skip + question_idx + 1
  ```
- Updated console output and log file to use `absolute_question_num`

### 2. MLX Implementation (`tests/dev/full_mlx_evaluate.py`)

#### Constructor Changes:
- Added `skip: int = 0` parameter to `MLXLM.__init__()`
- Store skip value as `self.skip`
- Pass skip parameter when initializing model in main()

#### Logging Changes:
- Modified `_log_incorrect_answers()` to calculate absolute question number:
  ```python
  absolute_question_num = self.skip + question_idx + 1
  ```
- Updated console output and log file to use `absolute_question_num`

## Behavior

### Before Changes:
- Question numbers were relative to the processed batch
- With `--skip 45 --limit 50`, questions were numbered 1-50
- Made it difficult to correlate with original dataset indices

### After Changes:
- Question numbers are absolute (original dataset indices)
- With `--skip 45 --limit 50`, questions are numbered 46-95
- Easy correlation with original dataset indices

## Example Usage

```bash
# ANE evaluation with skip
python evaluate/ane/evaluate_with_harness.py \
  --model /path/to/model \
  --tasks boolq \
  --skip 45 \
  --limit 50 \
  --log-incorrect-answers

# MLX evaluation with skip  
python tests/dev/full_mlx_evaluate.py \
  --model Qwen/Qwen2.5-0.5B \
  --tasks boolq \
  --skip 45 \
  --limit 50 \
  --log-incorrect-answers
```

Both will now log incorrect answers with question numbers 46-95 instead of 1-50.

## Testing

Created `tests/dev/test_absolute_question_numbering.py` to verify the implementation:
- ✓ ANE implementation correctly calculates absolute numbers
- ✓ MLX implementation correctly calculates absolute numbers
- ✓ Both implementations handle skip=0 and skip>0 cases correctly
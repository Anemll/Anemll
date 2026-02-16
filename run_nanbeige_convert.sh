#!/bin/bash
# Convert Nanbeige4.1-3B for ANE (model at moved path)
cd "$(dirname "$0")"
time ./anemll/utils/convert_model.sh \
  --model /Users/anemll/Models/ANE/nanbeige41-3b-ane-quick-fp16 \
  --output /Volumes/Models/ANE/nanbeige41_3b_lut64_4096 \
  --prefix nanbeige41 \
  --context 4096 \
  --batch 16 \
  --chunk auto \
  --lut1 8,4 \
  --lut2 6,4 \
  --lut3 6,4

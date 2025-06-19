# Qwen 3 Conversion Guide

This document explains how to convert Qwen 3 checkpoints for use with the ANEMLL
pipeline. The process mirrors the existing Llama flow and works for models up to
8&nbsp;B parameters.

## Usage

```bash
./anemll/utils/convert_model.sh --model <path_to_qwen3> --output <out> --prefix qwen
```

After conversion you can run inference with the Python chat script:

```bash
python tests/chat.py --meta <out>/meta.yaml
```

## Notes

* Only 4-D tensors are produced to comply with ANE requirements.
* LM head layers are automatically sliced when the width exceeds 16&nbsp;384.
* Both the 0.6&nbsp;B development checkpoint and the 8&nbsp;B model have been
validated.

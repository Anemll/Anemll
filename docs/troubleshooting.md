# Troubleshooting Guide

This document provides solutions to common issues encountered while using the Anemll project.

Model conversion script provides dependecies checks during start up
It checks for Python, pip, coremltools, coremlcompiler, and macOS version.
We also check for supported model architectures and quantization.

Script is located in `./anemll/utils/check_dependencies.sh`

Parent script is `./anemll/utils/convert_model.sh`

If you confident it's incorrectly reporting an issue, you can skip the check by adding `--skip-check` flag to convert_model.sh

## Common Issues

### 1. Python Not Installed
**Problem:**
- Error message: "Python is required but it's not installed. Aborting. (Issue #1)"

**Solution:**
- Ensure Python is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
- Verify the installation by running `python --version` in your terminal.

### 2. Pip Not Installed
**Problem:**
- Error message: "pip is required but it's not installed. Aborting. (Issue #2)"

**Solution:**

- **If you're using pip:** Install pip by following the instructions at [pip.pypa.io](https://pip.pypa.io/en/stable/installation/). Verify with `pip --version`.

- **If you're using UV:** Create and activate a UV environment with seeded pip:
  ```bash
  uv venv --python 3.9 --seed env-anemll
  source env-anemll/bin/activate
  ```
  Then install dependencies (for example: `uv pip install -r requirements.txt`). Verify with `pip --version` or `uv pip --version`.

### 3. CoreML Tools Not Installed
**Problem:**
- Error message: "coremltools is required but not installed via pip. Aborting. (Issue #3)"

**Solution:**
- Install coremltools using pip: `pip install "coremltools>=9.0"`.
- Install coremltools using uv: `uv pip install "coremltools>=9.0"`.
- Verify the installation by running `pip show coremltools`.

### 4. CoreML Compiler Not Found
**Problem:**
- Error message: "coremlcompiler is required but not found. Aborting. (Issue #4)"

**Solution:**
- Ensure that Xcode Command Line Tools are installed, as they include `coremlcompiler`.
- You can install them by running `xcode-select --install`.
- Verify that the `xcrun` command is available and correctly configured in your PATH.
- Download Xcode from the App Store.
- Run `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer/` to set the path.
- Use `xcrun --find coremlcompiler` to verify the installation.
- Run `sudo xcodebuild -license` and agree to the license.

### 5. macOS Version Below 15
**Problem:**
- Error message: "macOS version 15 or higher is required. Aborting. (Issue #5)"

**Solution:**
- Upgrade your macOS to version 15 or higher to ensure compatibility with the tools.

### 6. Required Model Files Missing
**Problem:**
- Error message: "config.json/tokenizer.json/tokenizer_config.json is required but not found in the model directory. Aborting. (Issue #5)"

**Solution:**
- Ensure all required model files are present in the model directory:
  - config.json
  - tokenizer.json
  - tokenizer_config.json
- Download the complete model from HuggingFace or your model provider.

### 7. Quantized Models Not Supported
**Problem:**
- Error message: "Quantized models are not supported. Aborting. (Issue #6)"

**Solution:**
- Use a version of the model without quantization, as ANEMLL cannot properly convert quantized weights to FP16.

### 8. Unsupported Architecture
**Problem:**
- Error message: "Unsupported architecture or model type in config.json. Supported types: llama. Aborting. (Issue #7)"

**Solution:**
- Ensure that the model architecture specified in `config.json` is supported. Currently, only the "llama" architecture is supported.

### 9. CoreML Tools Version Too Low
**Problem:**
- Error message: "coremltools version 9.x or higher is required. Aborting. (Issue #9)"

**Solution:**
- Upgrade coremltools to version 9.0+ using pip: `pip install --upgrade "coremltools>=9.0"`.
- Or with UV: `uv pip install --upgrade "coremltools>=9.0"`.
- Verify the version by running `pip show coremltools`.

## Additional Help
For further assistance, please refer to the official documentation or contact via github issues. 

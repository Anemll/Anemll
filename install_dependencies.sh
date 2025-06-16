#!/bin/bash

# Detect Python command
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Detect pip command
if command -v pip &> /dev/null; then
    PIP_CMD=pip
elif command -v pip3 &> /dev/null; then
    PIP_CMD=pip3
else
    echo "Error: pip is not installed or not in PATH"
    echo "Please install pip first. You can try: $PYTHON_CMD -m ensurepip"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo "Using pip: $PIP_CMD"

# Upgrade pip
$PIP_CMD install --upgrade pip

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')
echo "Detected Python version: $PYTHON_VERSION"

# Check if Python version is compatible with ANEMLL
if [[ $(echo "$PYTHON_VERSION != 3.9" | bc) -eq 1 ]]; then
    echo "⚠️ WARNING: ANEMLL is designed to work with Python 3.9.x"
    echo "Current Python version is $PYTHON_VERSION"
    echo "For best compatibility, consider creating a virtual environment with Python 3.9"
    echo "Example: python3.9 -m venv env-anemll-bench"
    echo "Continuing with installation, but you may encounter issues..."
    echo ""
fi

# Install PyTorch based on Python version
if [[ $(echo "$PYTHON_VERSION >= 3.13" | bc) -eq 1 ]]; then
    echo "Python 3.13+ detected. Installing PyTorch 2.6.0 which is compatible with newer Python versions."
    $PIP_CMD install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
elif [[ $(echo "$PYTHON_VERSION >= 3.10" | bc) -eq 1 ]]; then
    echo "Python 3.10-3.12 detected. Installing PyTorch 2.5.0."
    $PIP_CMD install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
else
    echo "Python 3.9 or earlier detected. Installing PyTorch 2.5.0 with specific flags for older Python versions."
    $PIP_CMD install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
fi

# Install coremltools after PyTorch
$PIP_CMD install coremltools>=8.2

# Install the rest of the dependencies
$PIP_CMD install -r requirements.txt

# Verify PyTorch installation and MPS availability
$PYTHON_CMD -c "import torch; print('PyTorch version: {}'.format(torch.__version__)); print('MPS available: {}'.format(torch.backends.mps.is_available()))"

# Verify coremltools installation
$PYTHON_CMD -c "import coremltools; print('CoreMLTools version: {}'.format(coremltools.__version__))"

echo "Installation complete!" 
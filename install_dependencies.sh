#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
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
    pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
elif [[ $(echo "$PYTHON_VERSION >= 3.10" | bc) -eq 1 ]]; then
    echo "Python 3.10-3.12 detected. Installing PyTorch 2.5.0."
    pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
else
    echo "Python 3.9 or earlier detected. Installing PyTorch 2.5.0 with specific flags for older Python versions."
    pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
fi

# Install coremltools after PyTorch
pip install coremltools>=8.2

# Install the rest of the dependencies
pip install -r requirements.txt

# Verify PyTorch installation and MPS availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"

# Verify coremltools installation
python -c "import coremltools; print(f'CoreMLTools version: {coremltools.__version__}')"

echo "Installation complete!" 
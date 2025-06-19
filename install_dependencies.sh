#!/bin/bash

# ANEMLL Dependencies Installation Script
# Installs all required dependencies for ANEMLL development and usage

set -e

echo "🚀 Installing ANEMLL Dependencies..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
    # In virtual environment, prefer python3 if python is not available
    if command -v python &> /dev/null; then
        PYTHON_CMD=python
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    else
        echo "❌ Error: Neither python nor python3 found in virtual environment"
        exit 1
    fi
    
    if command -v pip &> /dev/null; then
        PIP_CMD=pip
    elif command -v pip3 &> /dev/null; then
        PIP_CMD=pip3
    else
        echo "❌ Error: Neither pip nor pip3 found in virtual environment"
        exit 1
    fi
elif [[ -f "./env-anemll/bin/activate" ]]; then
    echo "🔄 Found env-anemll virtual environment, activating it..."
    source ./env-anemll/bin/activate
    PYTHON_CMD=python
    PIP_CMD=pip
else
    echo "⚠️  No virtual environment detected"
    # Detect Python command
    if command -v python3.9 &> /dev/null; then
        PYTHON_CMD=python3.9
        echo "✓ Found Python 3.9 (recommended)"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    elif command -v python &> /dev/null; then
        PYTHON_CMD=python
    else
        echo "❌ Error: Python is not installed or not in PATH"
        echo "Please install Python 3.9+ first"
        exit 1
    fi
    
    # Detect pip command
    if command -v pip3 &> /dev/null; then
        PIP_CMD=pip3
    elif command -v pip &> /dev/null; then
        PIP_CMD=pip
    else
        echo "❌ Error: pip is not installed or not in PATH"
        echo "Please install pip first. You can try: $PYTHON_CMD -m ensurepip"
        exit 1
    fi
fi

echo "Using Python: $PYTHON_CMD"
echo "Using pip: $PIP_CMD"

# Check Python version
PYTHON_VERSION_FULL=$($PYTHON_CMD -c 'import sys; print("{}.{}.{}".format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro))')
PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')
echo "Detected Python version: $PYTHON_VERSION_FULL"

# Check if Python version is compatible with ANEMLL
if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 9 ]]; then
    echo "❌ ERROR: ANEMLL requires Python 3.9 or higher"
    echo "Current Python version is $PYTHON_VERSION_FULL"
    echo "Please upgrade Python or create a virtual environment with Python 3.9+"
    exit 1
elif [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -gt 11 ]]; then
    echo "⚠️  WARNING: ANEMLL is tested with Python 3.9-3.11"
    echo "Current Python version is $PYTHON_VERSION_FULL"
    echo "Python 3.12+ may work but is not officially supported"
    echo "Continuing with installation..."
    echo ""
elif [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ne 9 ]]; then
    echo "ℹ️  Using Python $PYTHON_VERSION_FULL (ANEMLL is optimized for Python 3.9)"
    echo "Continuing with installation..."
    echo ""
else
    echo "✅ Using Python $PYTHON_VERSION_FULL (recommended version)"
fi

# Check for macOS and Xcode Command Line Tools
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected - checking for Xcode Command Line Tools..."
    if ! xcode-select -p &> /dev/null; then
        echo "❌ Xcode Command Line Tools not found"
        echo "Installing Xcode Command Line Tools..."
        xcode-select --install
        echo "Please complete the Xcode Command Line Tools installation and run this script again"
        exit 1
    else
        echo "✅ Xcode Command Line Tools found"
    fi
    
    # Check for coremlcompiler
    if command -v xcrun coremlcompiler &> /dev/null; then
        echo "✅ CoreML compiler found"
    else
        echo "⚠️  CoreML compiler not found - some features may not work"
    fi
else
    echo "⚠️  Non-macOS system detected - some Apple-specific features will not be available"
fi

# Ensure we're not installing to user directory in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Using virtual environment: $VIRTUAL_ENV"
    # Unset any user installation flags
    unset PIP_USER
fi

# Upgrade pip first
echo "📦 Upgrading pip..."
$PIP_CMD install --upgrade pip

# Install PyTorch based on Python version and platform
echo "🔥 Installing PyTorch..."
if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 13 ]]; then
    echo "Python 3.13+ detected. Installing latest PyTorch nightly for compatibility."
    $PIP_CMD install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
elif [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 10 ]]; then
    echo "Python 3.10+ detected. Installing PyTorch 2.5.0."
    $PIP_CMD install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "Python 3.9 detected. Installing PyTorch 2.5.0 (recommended)."
    $PIP_CMD install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install CoreML Tools (must be after PyTorch)
echo "🧠 Installing CoreML Tools..."
$PIP_CMD install coremltools>=8.2

# Install core ANEMLL dependencies
echo "📚 Installing core dependencies..."
$PIP_CMD install transformers>=4.36.0
$PIP_CMD install numpy>=1.24.0
$PIP_CMD install "scikit-learn<=1.5.1"
$PIP_CMD install datasets
$PIP_CMD install accelerate
$PIP_CMD install safetensors
$PIP_CMD install tokenizers
$PIP_CMD install sentencepiece
$PIP_CMD install pyyaml

# Install development dependencies
echo "🛠️  Installing development dependencies..."
$PIP_CMD install black
$PIP_CMD install flake8
$PIP_CMD install pytest
$PIP_CMD install jupyter
$PIP_CMD install ipykernel

# Install optional but recommended dependencies
echo "⚡ Installing optional dependencies..."
$PIP_CMD install huggingface_hub
$PIP_CMD install tqdm
$PIP_CMD install matplotlib
$PIP_CMD install seaborn

# Install ANEMLL package in development mode if setup.py or pyproject.toml exists
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "📦 Installing ANEMLL package in development mode..."
    $PIP_CMD install -e .
fi

# Verify installations
echo ""
echo "🔍 Verifying installations..."

# Verify PyTorch
echo -n "PyTorch: "
$PYTHON_CMD -c "import torch; print(f'✅ {torch.__version__}')" 2>/dev/null || echo "❌ Failed"

# Verify CoreML Tools
echo -n "CoreML Tools: "
$PYTHON_CMD -c "import coremltools; print(f'✅ {coremltools.__version__}')" 2>/dev/null || echo "❌ Failed"

# Verify Transformers
echo -n "Transformers: "
$PYTHON_CMD -c "import transformers; print(f'✅ {transformers.__version__}')" 2>/dev/null || echo "❌ Failed"

# Check MPS availability on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -n "MPS (Metal Performance Shaders): "
    $PYTHON_CMD -c "import torch; print('✅ Available' if torch.backends.mps.is_available() else '❌ Not Available')" 2>/dev/null || echo "❌ Failed"
fi

# Check ANE availability
echo -n "Apple Neural Engine: "
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Check if we're on Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        echo "✅ Available (Apple Silicon detected)"
    else
        echo "❌ Not Available (Intel Mac - ANE requires Apple Silicon)"
    fi
else
    echo "❌ Not Available (ANE only available on Apple devices)"
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Verify your virtual environment is activated"
echo "  2. Test conversion with: ./anemll/utils/convert_model.sh --help"
echo "  3. Run tests with: python tests/test_qwen_model.py"
echo ""
echo "📖 For more information, see:"
echo "  - README.md for usage instructions"
echo "  - CLAUDE.md for development guidelines"
echo "  - docs/ directory for detailed documentation"
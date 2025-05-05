#!/bin/bash

# Set the path to Python 3.9 from Homebrew (primary supported version)
PYTHON_PATH="/opt/homebrew/opt/python@3.9/bin/python3.9"
PYTHON_VERSION="3.9"

# Check if Python 3.9 is installed
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Python 3.9 is not found at $PYTHON_PATH"
    
    # Try Python 3.10 as an alternative
    ALT_PATH_310="/opt/homebrew/opt/python@3.10/bin/python3.10"
    if [ -f "$ALT_PATH_310" ]; then
        echo "Found Python 3.10 at $ALT_PATH_310"
        echo "Using Python 3.10 instead."
        PYTHON_PATH="$ALT_PATH_310"
        PYTHON_VERSION="3.10"
    # Try Python 3.11 as an alternative
    elif [ -f "/opt/homebrew/opt/python@3.11/bin/python3.11" ]; then
        ALT_PATH_311="/opt/homebrew/opt/python@3.11/bin/python3.11"
        echo "Found Python 3.11 at $ALT_PATH_311"
        echo "Using Python 3.11 instead."
        PYTHON_PATH="$ALT_PATH_311"
        PYTHON_VERSION="3.11"
    # Check if Python 3.9 might be installed in an alternative location
    else
        ALTERNATE_PATH=$(which python3.9 2>/dev/null)
        if [ -n "$ALTERNATE_PATH" ]; then
            echo "Found Python 3.9 at alternate location: $ALTERNATE_PATH"
            echo "Using this instead."
            PYTHON_PATH="$ALTERNATE_PATH"
        else
            echo "Please install Python 3.9 using:"
            echo "  brew install python@3.9"
            echo ""
            echo "Alternatively, you can install Python 3.10 or 3.11:"
            echo "  brew install python@3.10"
            echo "  brew install python@3.11"
            echo ""
            echo "NOTE: Python 3.12 and 3.13 are NOT supported by this project."
            exit 1
        fi
    fi
fi

echo "Found Python $PYTHON_VERSION at $PYTHON_PATH"

# Check if the environment already exists and remove it if it does
ENV_NAME="env-anemll"
if [ -d "$ENV_NAME" ]; then
    echo "Found existing $ENV_NAME environment. Removing it..."
    rm -rf "$ENV_NAME"
    echo "Existing environment removed."
fi

# Create a virtual environment with Python
echo "Creating a fresh virtual environment with Python $PYTHON_VERSION..."
"$PYTHON_PATH" -m venv "$ENV_NAME"

# Activate the virtual environment
echo "Activating the virtual environment..."
source "$ENV_NAME/bin/activate"

# Verify Python version
python_version=$(python --version)
echo "Using $python_version"

# Copy the installation files to the new environment
echo "Copying installation files to the new environment..."
cp install_dependencies.sh "$ENV_NAME/"
cp requirements.txt "$ENV_NAME/"

echo ""
echo "Python $PYTHON_VERSION virtual environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source $ENV_NAME/bin/activate"
echo ""
echo "Then run the installation script:"
echo "  cd $ENV_NAME"
echo "  ./install_dependencies.sh"
echo ""
echo "After installation, return to the main directory and run your scripts with this Python environment"
echo ""
echo "IMPORTANT: This project works with Python 3.9, 3.10, or 3.11."
echo "Python 3.12 and 3.13 are NOT supported and will likely cause errors." 
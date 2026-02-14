#!/bin/bash

set -euo pipefail

ENV_NAME="env-anemll"
PYTHON_VERSION="3.9"

usage() {
    cat <<'EOF'
Usage: ./create_uv_env.sh [options]

Create ANEMLL virtual environment using uv.

Options:
  --env <name>      Environment directory name (default: env-anemll)
  --python <ver>    Python version for uv venv (default: 3.9)
  -h, --help        Show this help

Examples:
  ./create_uv_env.sh
  ./create_uv_env.sh --python 3.11
  ./create_uv_env.sh --env .venv --python 3.10
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            ENV_NAME="${2:-}"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed."
    echo "Install with: brew install uv"
    exit 1
fi

if [ -d "$ENV_NAME" ]; then
    echo "Found existing $ENV_NAME environment."
    echo "  Python: $("$ENV_NAME/bin/python" --version 2>/dev/null || echo 'unknown')"
    read -r -p "Recreate from scratch? [y/N] " response
    case "$response" in
        [yY]|[yY][eE][sS])
            echo "Removing $ENV_NAME..."
            rm -rf "$ENV_NAME"
            ;;
        *)
            echo "Resuming existing environment."
            echo ""
            # shellcheck disable=SC1090
            source "$ENV_NAME/bin/activate"
            echo "Using $(python --version)"
            echo "Using $(python -m pip --version)"
            echo ""
            echo "Environment is ready. To activate in a new terminal:"
            echo ""
            echo "  source $ENV_NAME/bin/activate"
            echo ""
            echo "Next steps:"
            echo "  1. ./install_dependencies.sh"
            echo "  2. python tests/test_gemma3_model.py   # test conversion pipeline"
            echo ""
            echo "Alternative direct UV install:"
            echo "  uv pip install -r requirements.txt"
            echo "  uv pip install -e ."
            exit 0
            ;;
    esac
fi

echo "Creating uv environment: $ENV_NAME (Python $PYTHON_VERSION)"
uv venv --python "$PYTHON_VERSION" --seed "$ENV_NAME"

echo ""
echo "Activating environment..."
# shellcheck disable=SC1090
source "$ENV_NAME/bin/activate"

echo "Using $(python --version)"
echo "Using $(python -m pip --version)"
echo ""
echo "UV environment created successfully."
echo ""
echo "To activate in a new terminal:"
echo ""
echo "  source $ENV_NAME/bin/activate"
echo ""
echo "Next steps:"
echo "  1. ./install_dependencies.sh"
echo "  2. python tests/test_gemma3_model.py   # test conversion pipeline"
echo ""
echo "Alternative direct UV install:"
echo "  uv pip install -r requirements.txt"
echo "  uv pip install -e ."

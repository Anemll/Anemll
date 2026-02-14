#!/bin/bash

set -euo pipefail

# Resolve repository paths from this script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

usage() {
    cat <<'EOF'
Usage: ./anemll/utils/fp16_preflight.sh --model <model_id_or_path> [options]

Runs FP16 ANE compatibility checks before conversion and saves a JSON report.

Options:
  --model, -m <id|path>   HuggingFace model id or local model path (required)
  --report <path>         Output JSON report path (default: tests/dev/logs/fp16_preflight_<timestamp>.json)
  --no-sweep              Disable clamp sweep (faster)
  -h, --help              Show help

All other arguments are passed through to fp16_compatibility_check.py.

Examples:
  ./anemll/utils/fp16_preflight.sh --model Qwen/Qwen3-4B
  ./anemll/utils/fp16_preflight.sh --model google/gemma-3-4b-it-qat-int4-unquantized --quick --max-tokens 16
  ./anemll/utils/fp16_preflight.sh --model ./local_model --no-sweep --device cpu
EOF
}

# Auto-activate a local virtual environment if none is active.
if [ -z "${VIRTUAL_ENV:-}" ] && [ "${ANEMLL_AUTO_VENV:-1}" != "0" ]; then
    ACTIVATE_CANDIDATES=()
    if [ -n "${ANEMLL_VENV_ACTIVATE:-}" ]; then
        ACTIVATE_CANDIDATES+=("${ANEMLL_VENV_ACTIVATE}")
    fi
    if [ -n "${ANEMLL_VENV:-}" ]; then
        ACTIVATE_CANDIDATES+=("${ANEMLL_VENV}/bin/activate")
    fi
    ACTIVATE_CANDIDATES+=(
        "${PROJECT_ROOT}/env-anemll/bin/activate"
        "${PROJECT_ROOT}/anemll-env/bin/activate"
        "${PROJECT_ROOT}/.venv/bin/activate"
        "${PROJECT_ROOT}/venv/bin/activate"
    )

    for ACTIVATE in "${ACTIVATE_CANDIDATES[@]}"; do
        if [ -f "${ACTIVATE}" ]; then
            # shellcheck disable=SC1090
            source "${ACTIVATE}"
            echo "Activated Python environment: ${ACTIVATE}"
            break
        fi
    done
fi

MODEL_ID=""
REPORT_PATH=""
RUN_SWEEP=1
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|-m)
            MODEL_ID="${2:-}"
            shift 2
            ;;
        --report|--save-report)
            REPORT_PATH="${2:-}"
            shift 2
            ;;
        --no-sweep)
            RUN_SWEEP=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$MODEL_ID" ]; then
    echo "Error: --model is required."
    usage
    exit 1
fi

if [ -z "$REPORT_PATH" ]; then
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    REPORT_PATH="${PROJECT_ROOT}/tests/dev/logs/fp16_preflight_${TIMESTAMP}.json"
fi

mkdir -p "$(dirname "$REPORT_PATH")"

HAS_SWEEP=0
for ARG in "${PASSTHROUGH_ARGS[@]}"; do
    if [ "$ARG" = "--sweep" ]; then
        HAS_SWEEP=1
        break
    fi
done

PYTHON_BIN="${ANEMLL_PYTHON:-}"
if [ -z "$PYTHON_BIN" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "Error: python/python3 not found in PATH."
        exit 1
    fi
fi

CHECK_SCRIPT="${PROJECT_ROOT}/anemll/utils/fp16_compatibility_check.py"

CMD=(
    "$PYTHON_BIN"
    "$CHECK_SCRIPT"
    --model "$MODEL_ID"
)

if [ "$RUN_SWEEP" -eq 1 ] && [ "$HAS_SWEEP" -eq 0 ]; then
    CMD+=(--sweep)
fi

CMD+=("${PASSTHROUGH_ARGS[@]}")
CMD+=(--save-report "$REPORT_PATH")

echo "Running FP16 preflight for: $MODEL_ID"
echo "Report path: $REPORT_PATH"
echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}"

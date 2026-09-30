#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "======================================"
echo "Vision Transformer Study Environment"
echo "======================================"
echo "Study directory: $SCRIPT_DIR"
echo "Virtual env    : $VENV_DIR"
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 was not found."
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating virtual environment..."
    if ! python3 -m venv "$VENV_DIR"; then
        echo
        echo "Failed to create .venv."
        echo "On Ubuntu/Debian, install the venv package first:"
        echo "  sudo apt install python3-venv"
        exit 1
    fi
else
    echo "[1/4] Existing .venv found. Reusing it."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/4] Upgrading pip..."
python -m pip install --upgrade pip

echo "[3/4] Installing CPU PyTorch and torchvision..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "[4/4] Installing study tools..."
python -m pip install pillow matplotlib jupyter

echo
echo "======================================"
echo "Setup complete"
echo "======================================"
echo
echo "Activate later with:"
echo "  source study/.venv/bin/activate"
echo
echo "Run the first exercise with:"
echo "  python study/01_patchify.py"

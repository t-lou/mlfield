#!/bin/bash
set -e

VENV_PATH="${HOME}/onnx2tf-env"

# 1. Abort if NVIDIA GPU is visible
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "❌ NVIDIA GPU detected. onnx2tf must run in a CPU-only environment."
    echo "   Please run this script in a CPU-only container or disable GPU visibility (runtime: nvidia)."
    exit 1
fi

# 2. Create venv only if missing
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH..."
    python3.10 -m venv "$VENV_PATH"
fi

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# 3. Install dependencies only if onnx2tf is missing
if ! python3 -c "import onnx2tf" >/dev/null 2>&1; then
    echo "Installing required packages..."
    pip install --upgrade pip
    pip install tensorflow-cpu==2.15
    pip install onnx==1.13.1 onnxruntime onnxsim
    pip install onnx2tf
else
    echo "onnx2tf already installed — skipping pip install."
fi

# 4. Convert ONNX → TF
# Example usage: bash convert_onnx2tf.sh ./yolo26n.onnx ./ -iqd int8
echo "Converting ONNX model to TensorFlow format..."
onnx2tf -i "$1" -o "$2" "${@:3}"

echo "Conversion completed. Deactivating virtual environment..."
deactivate

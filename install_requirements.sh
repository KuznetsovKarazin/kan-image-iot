#!/bin/bash
# Installation script to resolve dependency conflicts
# Guided installation to avoid conflicts between tf2onnx and protobuf

echo "============================================================"
echo "Requirements Installation with Conflict Resolution"
echo "============================================================"
echo

# Check if venv exists
if [ ! -f "venv_test/bin/python" ]; then
    echo "[ERROR] venv_test not found!"
    echo "Run first: python -m venv venv_test"
    exit 1
fi

echo "[1/5] Upgrading pip..."
venv_test/bin/python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "[ERROR] Pip upgrade failed"
    exit 1
fi
echo

echo "[2/5] Installing base packages and PyTorch..."
venv_test/bin/python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
if [ $? -ne 0 ]; then
    echo "[ERROR] PyTorch installation failed"
    exit 1
fi
echo

echo "[3/5] Installing TensorFlow and core packages (without tf2onnx)..."
venv_test/bin/python -m pip install \
  "tensorflow==2.20.0" \
  "tf_keras==2.20.1" \
  "keras==3.13.2" \
  "protobuf==5.28.3" \
  "onnx==1.17.0" \
  "ml_dtypes==0.5.4" \
  "onnx-simplifier==0.4.36" \
  "onnx-tf==1.10.0" \
  "sng4onnx==1.0.5" \
  "tensorflow-probability==0.25.0" \
  "onnx2tf==1.28.8" \
  "onnx_graphsurgeon==0.5.8" \
  "psutil==7.2.2" \
  "tensorboard==2.20.0"

if [ $? -ne 0 ]; then
    echo "[ERROR] TensorFlow/ONNX installation failed"
    exit 1
fi
echo

echo "[4/5] Installing tf2onnx (without dependencies to avoid conflicts)..."
venv_test/bin/python -m pip install "tf2onnx==1.16.1" --no-deps
if [ $? -ne 0 ]; then
    echo "[ERROR] tf2onnx installation failed"
    exit 1
fi
echo

echo "[5/5] Installing remaining packages..."
venv_test/bin/python -m pip install \
  "numpy==1.26.4" \
  "pandas==2.3.2" \
  "matplotlib==3.10.5" \
  "seaborn==0.13.2" \
  "scikit-learn==1.7.1" \
  "scipy==1.16.1" \
  "pykan==0.2.8" \
  "h5py==3.15.1" \
  "PyYAML==6.0.2" \
  "tqdm==4.67.1"

if [ $? -ne 0 ]; then
    echo "[WARNING] Remaining packages installation partially failed"
    echo "Continuing anyway..."
fi
echo

echo "============================================================"
echo "Installation completed!"
echo "============================================================"
echo
echo "Verifying key package versions:"
venv_test/bin/python -m pip list | grep -i "tensorflow\|keras\|onnx\|protobuf\|tf2onnx"
echo
echo "To test TFLite conversion, after generating the torch model, run:"
echo "venv_test/bin/python scripts/convert_to_tflite.py --force_config --quantize int8"
echo

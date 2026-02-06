#!/bin/bash
# Script per installare le dipendenze necessarie per la conversione TFLite
# Sistema operativo: Linux/macOS

set -e  # Exit on error

echo "============================================================"
echo "Installazione dipendenze per conversione TFLite"
echo "============================================================"
echo

echo "[1/4] Installazione pykan (richiesto per caricare il modello KAN)..."
pip install pykan
echo "OK - pykan installato"
echo

echo "[2/4] Installazione ONNX (formato intermedio per conversione)..."
pip install "onnx>=1.14.0"
echo "OK - ONNX installato"
echo

echo "[3/4] Installazione onnx-tf (converter ONNX to TensorFlow)..."
pip install "onnx-tf>=1.10.0"
echo "OK - onnx-tf installato"
echo

echo "[4/4] Installazione TensorFlow (per TFLite converter)..."
pip install "tensorflow>=2.13.0"
echo "OK - TensorFlow installato"
echo

echo "============================================================"
echo "INSTALLAZIONE COMPLETATA CON SUCCESSO!"
echo "============================================================"
echo
echo "Puoi ora eseguire la conversione con:"
echo "  python scripts/convert_to_tflite.py <modello.pt> --quantize float16"
echo
echo "============================================================"

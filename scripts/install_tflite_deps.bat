@echo off
REM Script per installare le dipendenze necessarie per la conversione TFLite
REM Sistema operativo: Windows

echo ============================================================
echo Installazione dipendenze per conversione TFLite
echo ============================================================
echo.

echo [1/4] Installazione pykan (richiesto per caricare il modello KAN)...
pip install pykan
if %errorlevel% neq 0 (
    echo ERRORE: Installazione pykan fallita
    pause
    exit /b 1
)
echo OK - pykan installato
echo.

echo [2/4] Installazione ONNX (formato intermedio per conversione)...
pip install onnx>=1.14.0
if %errorlevel% neq 0 (
    echo ERRORE: Installazione ONNX fallita
    pause
    exit /b 1
)
echo OK - ONNX installato
echo.

echo [3/4] Installazione onnx-tf (converter ONNX to TensorFlow)...
pip install onnx-tf>=1.10.0
if %errorlevel% neq 0 (
    echo ERRORE: Installazione onnx-tf fallita
    pause
    exit /b 1
)
echo OK - onnx-tf installato
echo.

echo [4/4] Installazione TensorFlow (per TFLite converter)...
pip install tensorflow>=2.13.0
if %errorlevel% neq 0 (
    echo ERRORE: Installazione TensorFlow fallita
    pause
    exit /b 1
)
echo OK - TensorFlow installato
echo.

echo ============================================================
echo INSTALLAZIONE COMPLETATA CON SUCCESSO!
echo ============================================================
echo.
echo Puoi ora eseguire la conversione con:
echo   python scripts/convert_to_tflite.py ^<modello.pt^> --quantize float16
echo.
echo ============================================================
pause

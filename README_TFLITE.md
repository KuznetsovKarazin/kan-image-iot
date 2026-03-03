# Conversione TFLite per KAN Image Classifier

Questo script converte il modello KAN-based Image Classifier da PyTorch a TensorFlow Lite per deployment su dispositivi IoT.

## Requisiti

Installa le dipendenze necessarie:

```bash
pip install onnx>=1.14.0 onnx-tf>=1.10.0 tensorflow>=2.13.0
```

O installa tutte le dipendenze del progetto:

```bash
pip install -r requirements.txt
```

## Pipeline di Conversione

Lo script esegue la seguente catena di conversione:

```
PyTorch (.pt) → ONNX (.onnx) → TensorFlow (SavedModel) → TFLite (.tflite)
```

### Perché questa pipeline?

- **PyTorch → ONNX**: Formato intermedio standard per esportare modelli PyTorch
- **ONNX → TensorFlow**: Usa `onnx-tf` per convertire in TensorFlow SavedModel
- **TensorFlow → TFLite**: TFLiteConverter ottimizza per deployment mobile/IoT

## Utilizzo

### Conversione Base (Float32)

```bash
python scripts/convert_to_tflite.py model/best_model.pt --output_name model.tflite
```

Questo genera un modello TFLite in formato float32 nella cartella `tflite_models/`.

### Conversione con Quantizzazione Float16

```bash
python scripts/convert_to_tflite.py model/best_model.pt --output_name model_f16.tflite --quantize float16
```

La quantizzazione float16:
- Riduce le dimensioni del modello del ~50%
- Mantiene buona accuratezza (perdita tipica < 1%)
- Compatibile con molti dispositivi IoT

### Conversione con Quantizzazione INT8

```bash
python scripts/convert_to_tflite.py model/best_model.pt --output_name model_int8.tflite --quantize int8
```

La quantizzazione int8:
- Riduce le dimensioni del modello del ~75%
- Richiede calibrazione con dati di validazione
- Massima efficienza su dispositivi embedded
- Può avere perdita di accuratezza maggiore (tipicamente 1-3%)

> **Nota**: Per int8, lo script usa automaticamente il validation dataset in `data/processed/vww_subset/val` per calibrazione.

### Disabilitare Verifica

```bash
python scripts/convert_to_tflite.py model/best_model.pt --no_verify
```

Salta il confronto delle predizioni PyTorch vs TFLite (utile per conversioni veloci).

## Parametri dello Script

| Parametro | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `model_path` | str | - | Path al checkpoint PyTorch (.pt) |
| `--output_name` | str | `model.tflite` | Nome del file TFLite di output |
| `--quantize` | str | `none` | Tipo di quantizzazione: `none`, `float16`, `int8` |
| `--verify` | bool | `True` | Verifica il modello convertito |
| `--no_verify` | flag | - | Disabilita la verifica |
| `--img_size` | int | `224` | Dimensione immagini di input |

## Output

Lo script genera i seguenti file nella cartella `tflite_models/`:

- `model.onnx`: Modello intermedio in formato ONNX
- `tf_model/`: TensorFlow SavedModel (cartella)
- `<output_name>.tflite`: Modello finale TFLite

## Verifica del Modello

Durante la verifica, lo script:

1. Carica il modello TFLite
2. Esegue inference su 10 immagini di test
3. Confronta le predizioni con il modello PyTorch originale
4. Riporta la percentuale di accordo

**Accordo atteso**:
- Float32: ~100% (predizioni identiche o molto simili)  
- Float16: ≥98% (minime differenze numeriche)
- INT8: ≥90% (accettabile con quantizzazione aggressiva)

## Esempio Completo

```bash
# 1. Converti il modello con quantizzazione float16
python scripts/convert_to_tflite.py experiment_data/kan_64_32_24_16_grid5_deg3_img224_bs256_lr0.003_wd1e-05_do0.05_mobilenetv3_small_wm0.25/models/best_model.pt --output_name kan_model_f16.tflite --quantize float16

# Output previsto:
# ✓ Successfully exported to ONNX: tflite_models/model.onnx
# ✓ Successfully converted to TensorFlow: tflite_models/tf_model
# ✓ Successfully converted to TFLite: tflite_models/kan_model_f16.tflite
#   Model size: 0.89 MB
# Prediction agreement (first 10 samples): 100.0%
# ✓ TFLite model verification PASSED
```

## Risoluzione Problemi

### Errore: "Module 'onnx_tf' not found"

Installa `onnx-tf`:
```bash
pip install onnx-tf
```

### Errore: "Module 'tensorflow' not found"

Installa TensorFlow:
```bash
pip install tensorflow
```

Per GPU (opzionale):
```bash
pip install tensorflow-gpu
```

### Accordo basso tra PyTorch e TFLite (<90%)

Possibili cause:
1. **Quantizzazione int8 troppo aggressiva**: Prova con `float16`
2. **Operazioni KAN complesse**: Alcune operazioni potrebbero non convertirsi perfettamente
3. **Normalizzazione diversa**: Verifica che i preprocessing siano identici

Soluzioni:
- Usa quantizzazione meno aggressiva (`float16` invece di `int8`)
- Aumenta il numero di sample per calibrazione int8
- Confronta output layer per layer per diagnosticare differenze

### Il modello è troppo grande (>2MB)

Strategie per ridurre le dimensioni:
1. Usa quantizzazione `int8` invece di `float16`
2. Riduci i parametri del modello (es. width_mult più basso)
3. Applica pruning prima della conversione
4. Considera model distillation per un modello più piccolo

## Deploy su Dispositivi IoT

Dopo la conversione, puoi usare il modello TFLite su:

- **Raspberry Pi**: Usa TFLite runtime Python
  ```bash
  pip install tflite-runtime
  ```

- **Android/iOS**: Integra TFLite SDK nelle app mobile

- **Microcontrollori** (ESP32, Arduino): Usa TFLite Micro per C++
  ```cpp
  #include "tensorflow/lite/micro/micro_interpreter.h"
  ```

- **Edge TPU** (Coral): Compila il modello per accelerazione hardware
  ```bash
  edgetpu_compiler model.tflite
  ```

## Performance Attese

Su un modello tipico KAN con MobileNetV3 Small (width_mult=0.25):

| Quantizzazione | Dimensione | Latenza (CPU) | Accuratezza | 
|----------------|------------|---------------|-------------|
| Float32 | ~1.8 MB | 45ms | 87.2% |
| Float16 | ~0.9 MB | 40ms | 87.1% |
| INT8 | ~0.5 MB | 25ms | 85.8% |

*Latenza misurata su Raspberry Pi 4 (single core, batch_size=1)*

## Riferimenti

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [ONNX Documentation](https://onnx.ai/)
- [Post-Training Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)

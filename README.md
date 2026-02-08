# KAN-based Person Detection for IoT Devices (v2)

A reproducible pipeline for **person detection on IoT devices** using a hybrid architecture:
a lightweight pretrained CNN (MobileNetV3) as a feature extractor and a
Kolmogorov–Arnold Network (KAN) as the classifier.

This branch (**v2**) extends the original project with:

- a **clean training & analysis pipeline**,
- **hybrid INT8 quantization** (CNN in int8, KAN in fp32),
- optional **pruning experiments**,
- **Pareto analysis** for size–accuracy trade-offs,
- reproducible comparison between **supervisor** and **student** models.

---

## Overview

The goal of this project is to design and evaluate **compact person detection models**
suitable for memory-constrained IoT devices (e.g., embedded boards, gateways).

Key ideas:

- Use a **pretrained MobileNetV3** as a lightweight feature extractor.
- Use a **KAN classifier** on top of CNN features to improve expressiveness.
- Apply **hybrid quantization** to compress the CNN part to INT8 while keeping KAN
  in FP32, ensuring stable inference.
- Systematically compare **baseline FP32**, **hybrid INT8**, and **pruning**-based
  compression in terms of:
  - model size (MB),
  - accuracy on the Visual Wake Words (VWW) subset,
  - practical suitability for IoT deployment.

---

## Features (v2)

- **Reproducible dataset preparation** for a VWW subset
- **Training pipeline** with configurable architecture and device
- **Detailed analysis** of trained models (`analyze.py`)
- **Hybrid INT8 quantization**:
  - MobileNetV3 weights compressed to INT8 on disk
  - KAN kept in FP32 for numerical stability
- **Negative-result pruning experiments** (hybrid + 30% pruning)
- **Pareto frontier plotting** for size vs. accuracy
- GPU support (if available), otherwise CPU-only works fine

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ and torchvision
- CUDA-capable GPU (optional but recommended for training)
- See `requirements.txt` for the full list of dependencies

Install dependencies (after creating and activating a virtual environment):

```bash
pip install -r requirements.txt
```

---

## Project Structure

```bash
kan-image-iot/
│
├── src/
│   ├── models/
│   │   └── kan_model.py          # MobileNetV3 + KAN architecture
│   ├── train.py                  # Training script
│   ├── analyze.py                # Model analysis & evaluation
│   └── ...
│
├── scripts/
│   ├── download_dataset.py       # Download COCO + generate VWW annotations
│   ├── prepare_dataset.py        # Build VWW subset (train/val/test folders)
│   ├── quantize_hybrid_manual.py # Hybrid INT8 quantization (CNN int8 + KAN fp32)
│   ├── hybrid_plus_pruning.py    # Pruning + hybrid quantization (experimental)
│   ├── pareto_summary.py         # Size–accuracy summary and Pareto plot
│   └── ...
│
├── data/
│   └── processed/
│       └── vww_subset/           # Prepared dataset (created by scripts)
│
├── experiment_data/              # Checkpoints, logs (git-ignored)
├── quantized_models/             # Quantized models & Pareto plots
│
├── requirements.txt
└── README.md                     # This file (v2 branch)

```

---

## Dataset Preparation

We use a subset of the Visual Wake Words (VWW) dataset (person / no_person), constructed from MS COCO data and VWW annotations.

1. Download COCO and create VWW annotations:

```bash
python scripts/download_dataset.py
```

This script will:

- download COCO images and annotations,
- create COCO train/minival split,
- generate Visual Wake Words annotations.

2. Prepare a smaller VWW subset for experiments:

```bash
python scripts/prepare_dataset.py
```

This will create:

```bash
data/processed/vww_subset/
    train/person, train/no_person
    val/person,   val/no_person
    test/person,  test/no_person
```

---

## Training

Train a baseline FP32 model (MobileNetV3 + KAN) on the VWW subset:

```bash
# CPU training
python src/train.py

# GPU training (if available)
python src/train.py --device cuda

# Example with custom hyperparameters
python src/train.py --device cuda --batch_size 128 --lr 0.003
```

The training script will save checkpoints and logs under:

```bash
experiment_data/<experiment_name>/
    models/kan_person_detector_best.pt
    ...
```

---

## Model Analysis & Evaluation

Use analyze.py to inspect a trained model:

```bash
# Analyze the latest or default experiment
python src/analyze.py

# Analyze a specific checkpoint
python src/analyze.py \
  --model_path experiment_data/kan_64_32-24-16_grid5_deg3_img224_bs64_lr0.002_wd1e-05_do0.05/models/kan_person_detector_best.pt
```

The script reports:

- model size (MB),
- total / trainable parameters,
- distribution between CNN preprocessor and KAN,
- accuracy, precision, recall, F1-score on the test set,
- basic inference-time measurements.

Analysis reports and summaries are stored in:

```bash
experiment_data/<experiment_name>/analysis/
```
---

## Hybrid INT8 Quantization

Why hybrid?

- MobileNetV3 is a standard CNN → INT8 quantization is well-supported.
- KAN uses spline-based, non-standard operations → pure INT8 execution is not stable or supported in standard PyTorch.
- A natve "everything INT8" approach can break inference or produce unreliable outputs.

To address this, we use **hybrid quantization**:

- On disk: MobileNetV3 weights are stored in INT8 + scale (per parameter tensor).
- At runtime: INT8 weights are de-quantized back to FP32 before inference, and KAN always runs in FP32.

This achieves a ~65% reduction in model size on disk with only a small drop in accuracy, while keeping inference numerically stable.

---

## Running hybrid quantization

```bash
# Quantize supervisor baseline
python scripts/quantize_hybrid_manual.py \
  experiment_data/kan_64_32-24-16_grid5_deg3_img224_bs64_lr0.002_wd1e-05_do0.05/models/kan_person_detector_best.pt \
  --output_name kan_hybrid_manual_supervisor.pt

# Quantize student baseline
python scripts/quantize_hybrid_manual.py \
  experiment_data/student_baseline/models/kan_person_detector_student_best.pt \
  --output_name kan_hybrid_manual_student.pt
```

The script will:

- load the FP32 checkpoint,
- quantize MobileNet weights to INT8 (storage),
- keep KAN in FP32,
- restore weights to FP32 for inference,
- evaluate accuracy on the test set,
- save a compact checkpoint in quantized_models/.

In our experiments (example):

- FP32 baseline (supervisor): ~4.02 MB, ~88-89% accuracy
- Hybrid INT8 (supervisor): ~1.40 MB, ~86% accuracy
- FP32 baseline (student): ~4.02 MB, ~87% accuracy
- Hybrid INT8 (student): ~1.40 MB, ~84% accuracy

---

## Static INT8 "Storage-only" Variant

There is also a static INT8 configuration that quantizes more aggressively.

However:

- due to limitations of KAN and operator support,
- this variant is not guaranteed to run stable inference.

We therefore treat it as a storage-only / theoretical lower bound on model size,
not as a production-ready model.

---

## Pruning Experiments (Negative Result)

The script `hybrid_plus_pruning.py` explores unstructured pruning + hybrid quantization:

```bash
python scripts/hybrid_plus_pruning.py \
  --model_path experiment_data/kan_64_32-24-16_grid5_deg3_img224_bs64_lr0.002_wd1e-05_do0.05/models/kan_person_detector_best.pt \
  --prune_amount 0.3 \
  --output_name kan_hybrid_pruned_30_supervisor.pt
```

Findings:

- Model size on disk remains ~1.40 MB (weights are still stored densely).
- Accuracy drops dramatically (e.g., ~56% with 30% pruning) without fine-tuning.
- The model becomes unusable for person detection.

This is an important negative result: naive unstructured pruning without fine-tuning
and without a sparse-aware runtime does not provide a good size–accuracy trade-off.

---

## Pareto Frontier: Size vs Accuracy

The script `pareto_summary.py` collects and visualizes the main configurations:

- FP32 baseline (supervisor & student),
- Hybrid INT8 (supervisor & student),
- Static INT8 (storage-only),
- Hybrid + pruning (negative example).

Run:

```bash
python scripts/pareto_summary.py
```

It will:

- print a summary table (size, accuracy, category),
- generate a plot `quantized_models/pareto_frontier_supervisor_student.png`
showing the size–accuracy trade-off for all configurations.

This plot is suitable for inclusion in a thesis or paper.

---


---

## TFLite Conversion & Testing

To deploy models on edge devices, we support conversion to TensorFlow Lite.

### Convert to TFLite

Use `scripts/convert_to_tflite.py` to convert a PyTorch checkpoint to TFLite format. This script handles the conversion pipeline: PyTorch -> ONNX -> TensorFlow -> TFLite.


```bash
# Basic conversion (uses model from config.py by default)
python scripts/convert_to_tflite.py

# Convert specific checkpoint
python scripts/convert_to_tflite.py <path_to_checkpoint.pt>

# Force configuration from config.py (useful if checkpoint config mismatch)
python scripts/convert_to_tflite.py <path_to_checkpoint.pt> --force_config
```

The script will save the `.tflite` model in the same experiment directory.

### Test TFLite Model

Use `scripts/test_tflite.py` to evaluate the converted model on the validation set.

```bash
# Evaluate accuracy and latency (limit to 100 batches for quick check)
python scripts/test_tflite.py <path_to_model.tflite> --limit 100
```

This script reports:
- Accuracy on the validation set
- Average inference latency (ms/sample)
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

---

## How to Use this Branch (v2) in Practice

For a fresh clone:

```bash
git clone https://github.com/KuznetsovKarazin/kan-image-iot.git
cd kan-image-iot

# Switch to the v2 branch
git checkout v2

# Setup environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

Then follow the steps:

- `python scripts/download_dataset.py`
- `python scripts/prepare_dataset.py`
- `python src/train.py` (or use your own config)
- `python src/analyze.py`
- `python scripts/quantize_hybrid_manual.py` ...
- `python scripts/pareto_summary.py`

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

Oleksandr Kuznetsov - oleksandr.o.kuznetsov@gmail.com


---

## Acknowledgments

Original KAN-based person detection idea:
https://github.com/KuznetsovKarazin/kan-image-iot

Visual Wake Words utilities:
https://github.com/Mxbonn/visualwakewords

KAN implementation based on pykan:
https://github.com/KindXiaoming/pykan

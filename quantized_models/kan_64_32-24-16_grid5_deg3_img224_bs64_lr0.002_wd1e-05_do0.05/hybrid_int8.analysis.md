# Hybrid INT8 Model Analysis

**Experiment:** `kan_64_32-24-16_grid5_deg3_img224_bs64_lr0.002_wd1e-05_do0.05`  
**Source FP32 checkpoint:** `experiment_data/kan_64_32-24-16_grid5_deg3_img224_bs64_lr0.002_wd1e-05_do0.05/models/kan_person_detector_best.pt`  

## Quantization
- Method: hybrid_manual_int8 (MobileNet INT8 storage, KAN FP32 inference)
- FP32 size: 4.03 MB
- Hybrid size: 1.41 MB
- Reduction: 65.1%  
- Compression ratio: 2.86x

## KAN Configuration
- img_size: 224
- feature_dim: 64
- hidden_dims: [32, 24, 16]
- grid: 5
- degree: 3

## Classification Metrics (test set)
- Accuracy: 86.35%
- Macro F1: 0.8630
- Person precision: 0.9145
- Person recall: 0.8020
- ROC AUC (person vs no_person): 0.9390

## Inference Performance (96×96, CPU)
- Single image (bs=1): 295.86 ms
- Batch inference (16 images): 23.52 ms per image
- Batch inference (32 images): 11.59 ms per image

## Confusion Matrix

rows = true class, columns = predicted class

| true\pred | no_person | person |
|-----------|-----------|-----------|
| no_person | 1850 | 150 |
| person | 396 | 1604 |


_Classification report saved in `hybrid_int8.report.txt`._

# Hybrid INT8 Model Analysis

**Experiment:** `kan_32_24-16_grid5_deg3_img224_bs64_lr0.002_wd1e-05_do0.05_mobilenetv3_small`  
**Source FP32 checkpoint:** `experiment_data/kan_32_24-16_grid5_deg3_img224_bs64_lr0.002_wd1e-05_do0.05_mobilenetv3_small/models/kan_person_detector_best.pt`  

## Quantization
- Method: hybrid_manual_int8 (MobileNet INT8 storage, KAN FP32 inference)
- FP32 size: 3.83 MB
- Hybrid size: 1.21 MB
- Reduction: 68.5%  
- Compression ratio: 3.18x

## KAN Configuration
- img_size: 224
- feature_dim: 32
- hidden_dims: [24, 16]
- grid: 5
- degree: 3

## Classification Metrics (test set)
- Accuracy: 83.65%
- Macro F1: 0.8360
- Person precision: 0.8772
- Person recall: 0.7825
- ROC AUC (person vs no_person): 0.9207

## Inference Performance (96×96, CPU)
- Single image (bs=1): 137.40 ms
- Batch inference (16 images): 10.81 ms per image
- Batch inference (32 images): 5.24 ms per image

## Confusion Matrix

rows = true class, columns = predicted class

| true\pred | no_person | person |
|-----------|-----------|-----------|
| no_person | 1781 | 219 |
| person | 435 | 1565 |


_Classification report saved in `hybrid_int8.report.txt`._

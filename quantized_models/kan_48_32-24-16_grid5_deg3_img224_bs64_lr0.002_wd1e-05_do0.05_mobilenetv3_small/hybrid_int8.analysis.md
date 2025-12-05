# Hybrid INT8 Model Analysis

**Experiment:** `kan_48_32-24-16_grid5_deg3_img224_bs64_lr0.002_wd1e-05_do0.05_mobilenetv3_small`  
**Source FP32 checkpoint:** `experiment_data/kan_48_32-24-16_grid5_deg3_img224_bs64_lr0.002_wd1e-05_do0.05_mobilenetv3_small/models/kan_person_detector_best.pt`  

## Quantization
- Method: hybrid_manual_int8 (MobileNet INT8 storage, KAN FP32 inference)
- FP32 size: 3.97 MB
- Hybrid size: 1.34 MB
- Reduction: 66.2%  
- Compression ratio: 2.96x

## KAN Configuration
- img_size: 224
- feature_dim: 48
- hidden_dims: [32, 24, 16]
- grid: 5
- degree: 3

## Classification Metrics (test set)
- Accuracy: 83.83%
- Macro F1: 0.8378
- Person precision: 0.8798
- Person recall: 0.7835
- ROC AUC (person vs no_person): 0.9189

## Inference Performance (96×96, CPU)
- Single image (bs=1): 317.21 ms
- Batch inference (16 images): 20.67 ms per image
- Batch inference (32 images): 10.34 ms per image

## Confusion Matrix

rows = true class, columns = predicted class

| true\pred | no_person | person |
|-----------|-----------|-----------|
| no_person | 1786 | 214 |
| person | 433 | 1567 |


_Classification report saved in `hybrid_int8.report.txt`._

# KAN-based Person Detection for IoT Devices

A lightweight implementation of Kolmogorov-Arnold Network (KAN) using pretrained CNN's for efficient person detection on resource-constrained IoT devices. 
This project is forked from https://github.com/KuznetsovKarazin/kan-image-iot

## Overview

This project implements an efficient person detection model using the KAN architecture. The model is designed to run on resource-constrained IoT devices while maintaining good accuracy.
It can use some pretrained CNN's to allow choosing among them for the most suitabile model due the resource constraints and precision wanted.

## Features

- **Lightweight Architecture**: Optimized for IoT deployment with minimal memory footprint
- **GPU-accelerated Training**: Supports CPU and GPU training with mixed precision
- **Comprehensive Analysis**: Detailed metrics and visualizations for model performance

## Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (for GPU acceleration)
- See `requirements.txt` for complete dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cnn-kan-image.git
cd cnn-kan-image

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The model is trained on a subset of the Visual Wake Words (VWW) dataset, which consists of images with and without people. The dataset preparation script will automatically download and process the dataset.

```bash
# Prepare the dataset
python scripts/download_dataset.py
python scripts/prepare_dataset.py
```

## Training

```bash
# Train on CPU
python src/train.py

# Train on GPU with mixed precision
python src/train.py --device cuda --precision mixed

# Train with custom parameters
python src/train.py --batch_size 128 --lr 0.003
```

## Evaluation

```bash
# Evaluate the trained model
python src/analyze.py

# Evaluate a specific model checkpoint
python src/analyze.py --model_path experiment_data/your_experiment/models/kan_person_detector_best.pt
```


## Model Architecture

The model consists of two main components:
1. **CNN Preprocessor**: Lightweight pretrained convolutional layers for feature extraction
2. **KAN Network**: Kolmogorov-Arnold Network for efficient classification

## Results

- **Model Size**: ~9.0 MB
- **Accuracy**: ~85% on test set
- **Inference Time**: ~200ms per image (batch mode) on CPU

## Citation

```
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ricky T. Q. and Noy, Jiří and Saracchi, Riccardo and Gupta, Samar and Lavaei, Javad and Anandkumar, Anima and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Daniele Faggi - daniele.faggi@gmail.com
- Oleksandr Kuznetsov - oleksandr.o.kuznetsov@gmail.com
- Project Link: https://github.com/KuznetsovKarazin/cnn-kan-image

## Acknowledgments

- KAN-based Person Detection for IoT Devices (https://github.com/KuznetsovKarazin/kan-image-iot)
- Python library to work with the Visual Wake Words Dataset (https://github.com/Mxbonn/visualwakewords)
- KAN implementation based on pykan (https://github.com/KindXiaoming/pykan)

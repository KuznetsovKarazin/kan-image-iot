"""
Configuration file for KAN-based Person Detection Model.
This module contains all configurable parameters for dataset preparation,
model architecture, training, and evaluation.

Author: Oleksandr Kuznetsov
Date: March 2025
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'experiment_data'

# Dataset configuration
DATASET_CONFIG = {
    'dataset_name': 'vww',  # Options: 'vww', 'custom'
    'raw_data_dir': DATA_DIR / 'raw' / 'coco',
    'processed_data_dir': DATA_DIR / 'processed',
    'subset_dir': DATA_DIR / 'processed' / 'vww_subset',
    'img_size': 128,  # Image size for model input
    'train_samples_per_class': 50000,  # Number of training samples per class
    'val_samples_per_class': 5000,     # Number of validation samples per class
    'test_samples_per_class': 2000,    # Number of test samples per class
    'balanced': True,                 # Whether to ensure balanced classes
    'random_seed': 42,                # Random seed for reproducibility
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    'horizontal_flip_prob': 0.5,  # Good value for person detection
    'rotation_degrees': 10,       # Moderate rotation for better generalization
    'translate': (0.1, 0.1),      # Translation range (horizontal, vertical)
    'scale': (0.95, 1.05),        # Scale range for diversity
    'brightness': 0.1,            # Brightness adjustment range
    'contrast': 0.1,              # Contrast adjustment range
    'saturation': 0.1,            # Saturation adjustment (lower to preserve skin tones)
    'hue': 0.01,                  # Minimal hue adjustment to preserve natural colors
    'random_erase_prob': 0.05,    # Probability of random erasing for occlusion simulation
    'random_erase_scale': (0.02, 0.1),  # Size range of erased areas
    'enable_color_jitter': False,
    'enable_random_affine': False,
    'enable_random_perspective': False,  # Enable for viewpoint diversity
    'enable_random_grayscale': False,    # Enable for lighting condition robustness
    'mixup_alpha': 0.2,                 # Mixup regularization parameter
    'mixup_prob': 0.2,                 # Probability of applying mixup
    'cutmix_alpha': 0.1,                # CutMix regularization parameter
    'cutmix_prob': 0.1,                # Probability of applying cutmix
}
"""
AUGMENTATION_CONFIG = {
    'horizontal_flip_prob': 0.5,  # Good value for person detection
    'rotation_degrees': 10,       # Moderate rotation for better generalization
    'translate': (0.1, 0.1),      # Translation range (horizontal, vertical)
    'scale': (0.95, 1.05),        # Scale range for diversity
    'brightness': 0.2,            # Brightness adjustment range
    'contrast': 0.2,              # Contrast adjustment range
    'saturation': 0.1,            # Saturation adjustment (lower to preserve skin tones)
    'hue': 0.05,                  # Minimal hue adjustment to preserve natural colors
    'random_erase_prob': 0.1,    # Probability of random erasing for occlusion simulation
    'random_erase_scale': (0.02, 0.1),  # Size range of erased areas
    'enable_color_jitter': True,
    'enable_random_affine': True,
    'enable_random_perspective': True,  # Enable for viewpoint diversity
    'enable_random_grayscale': True,    # Enable for lighting condition robustness
    'mixup_alpha': 0.2,                 # Mixup regularization parameter
    'mixup_prob': 0.15,                 # Probability of applying mixup
    'cutmix_alpha': 0.2,                # CutMix regularization parameter
    'cutmix_prob': 0.15,                # Probability of applying cutmix
}
"""
# CNN Preprocessor configuration
PREPROCESSOR_CONFIG = {
    'input_channels': 3,
    'output_features': 64,            # Feature dimension after preprocessing
    'conv_channels': [16, 24, 32],    # Channels in each conv layer
    'kernel_size': [3, 3, 3],                 # Kernel size for convolutions
    'pool_kernel_size': 2,
    'final_pool_size': 4,             # Final spatial dimension after pooling
    'use_batch_norm': True,
    'dropout_rate': 0.05,              # Dropout rate for regularization
    'l2_regularization': 1e-5,        # L2 regularization strength
    'stochastic_depth_rate': 0.1,     # Stochastic depth regularization parameter
}

# KAN configuration
KAN_CONFIG = {
    'feature_dim': 64,          # Input dimension to KAN (same as preprocessor output)
    'hidden_dims': [32, 24, 16],    # Hidden layer dimensions
    'grid': 5,                  # Number of grid points
    'degree': 3,                # Spline degree
    'seed': 42,                 # Random seed for initialization
    'dropout_rate': 0.05,        # Dropout rate for KAN layers
    'weight_decay': 1e-5,       # Weight decay (L2 regularization) for KAN parameters
    'activation_l1': 1e-5,      # L1 regularization for activations (sparsity)
    'use_batchnorm': True,      # Use batch normalization between KAN layers
}

# Training configuration CPU
TRAINING_CONFIG = {
    'batch_size': 64,
    'val_batch_size': 64,
    'epochs': 50,
    'learning_rate': 0.002,
    'weight_decay': 1e-5,
    'lr_scheduler': 'cosine',    # Options: 'reducelr', 'cosine', 'step', 'onecycle'
    'lr_patience': 3,            # For ReduceLROnPlateau
    'lr_factor': 0.5,            # For ReduceLROnPlateau
    'early_stopping_patience': 15,
    'optimizer': 'adamw',        # Options: 'adam', 'adamw', 'sgd'
    'num_workers': min(os.cpu_count(), 12),  # Workers for data loading
    'save_checkpoints': True,
    'checkpoint_interval': 50,   # Save checkpoint every N epochs
    'gradient_clip_val': 1.0,    # Gradient clipping value
    'ema_decay': 0.99,           # Exponential moving average decay for model weights
    'label_smoothing': 0.1,      # Label smoothing factor for regularization
    'enable_mixup': False,        # Enable mixup augmentation
    'enable_cutmix': False,       # Enable cutmix augmentation
    'cross_validation_folds': 0, # Number of cross-validation folds (0 = disabled)
    'precision': 'float32',        # Options: 'float32', 'mixed', 'float16'
}
"""
# Training configuration GPU
TRAINING_CONFIG = {
    'batch_size': 256,  # Increased for GPU - adjust based on available VRAM
    'val_batch_size': 256,  # Can use larger batches for validation
    'epochs': 50,
    'learning_rate': 0.003,  # Slightly increased for use with larger batches
    'weight_decay': 1e-5,
    'lr_scheduler': 'cosine',    # Options: 'reducelr', 'cosine', 'step', 'onecycle'
    'lr_patience': 3,            # For ReduceLROnPlateau
    'lr_factor': 0.5,            # For ReduceLROnPlateau
    'early_stopping_patience': 15,
    'optimizer': 'adamw',        # Options: 'adam', 'adamw', 'sgd'
    'num_workers': min(os.cpu_count(), 4),  # Not too many workers for Colab - adjust based on testing
    'save_checkpoints': True,
    'checkpoint_interval': 10,   # Save checkpoint every N epochs
    'gradient_clip_val': 1.0,    # Gradient clipping value
    'ema_decay': 0.99,           # Exponential moving average decay for model weights
    'label_smoothing': 0.1,      # Label smoothing factor for regularization
    'enable_mixup': False,        # Enable mixup augmentation
    'enable_cutmix': False,       # Enable cutmix augmentation
    'cross_validation_folds': 0, # Number of cross-validation folds (0 = disabled)
    'precision': 'mixed',        # Default to mixed precision for GPU training
}
"""
# Experiment names will be generated based on key parameters
def get_experiment_name():
    """Generate experiment name based on current configuration"""
    exp_name = f"kan_{KAN_CONFIG['feature_dim']}_{'-'.join(map(str, KAN_CONFIG['hidden_dims']))}"
    exp_name += f"_grid{KAN_CONFIG['grid']}_deg{KAN_CONFIG['degree']}"
    exp_name += f"_img{DATASET_CONFIG['img_size']}"
    exp_name += f"_bs{TRAINING_CONFIG['batch_size']}"
    exp_name += f"_lr{TRAINING_CONFIG['learning_rate']}"
    
    # Add regularization info
    exp_name += f"_wd{TRAINING_CONFIG['weight_decay']}"
    
    # Add dropout info if enabled
    if KAN_CONFIG['dropout_rate'] > 0:
        exp_name += f"_do{KAN_CONFIG['dropout_rate']}"
    
    return exp_name

# Paths for current experiment
def get_experiment_paths():
    """Get paths for the current experiment"""
    exp_name = get_experiment_name()
    
    paths = {
        'experiment_dir': OUTPUT_DIR / exp_name,
        'model_dir': OUTPUT_DIR / exp_name / 'models',
        'figure_dir': OUTPUT_DIR / exp_name / 'figures',
        'analysis_dir': OUTPUT_DIR / exp_name / 'analysis',
    }
    
    # Create directories if they don't exist
    for dir_path in paths.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return paths
"""
Training Pipeline for KAN-based Person Detection Model.
This module implements the complete training workflow for a KAN model
optimized for efficient person detection on resource-constrained devices.
Includes advanced regularization techniques to prevent overfitting.

Author: Oleksandr Kuznetsov
Date: March 2025
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, CosineAnnealingLR, StepLR, OneCycleLR
)
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import argparse
import json
import random
from sklearn.model_selection import KFold

# Add the parent directory to Python's path to find the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up multi-threading optimizations
torch.set_num_threads(os.cpu_count())

# Import project modules
from models.kan_model import KANImageClassifier
from utils.metrics import AverageMeter, accuracy
from utils.visualization import plot_training_curves, plot_confusion_matrix, show_misclassified

# Import configuration from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATASET_CONFIG, AUGMENTATION_CONFIG, PREPROCESSOR_CONFIG, 
    KAN_CONFIG, TRAINING_CONFIG, get_experiment_paths
)

def seed_everything(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_amp(precision, device_type='cpu'):
    """Initialize Automatic Mixed Precision (AMP) based on precision and device type"""
    use_amp = precision in ['mixed', 'float16']
    scaler = None
    
    if use_amp:
        if device_type == 'cuda' and torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            print("Using Automatic Mixed Precision (AMP) training")
            if precision == 'float16':
                # Use full FP16 training (more aggressive, higher performance, less stable)
                print("Warning: Using full FP16 precision may be unstable. Consider 'mixed' for better stability.")
            
            # For newer PyTorch versions on supported GPUs, enable cudnn benchmark and TF32
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
                print("CUDNN benchmark enabled for better performance")
            
            if hasattr(torch.backends.cudnn, 'allow_tf32') and torch.cuda.get_device_capability()[0] >= 8:
                # Enable TF32 format on Ampere (SM80) or later GPUs (A100, A6000, A10, RTX 30 series, etc.)
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                print("TF32 enabled for Tensor Cores on supported GPUs")
        else:
            # AMP not available on CPU, disable it
            use_amp = False
            print("Warning: AMP requested but not available on CPU. Using float32 precision.")
    
    return use_amp, scaler

class MixUpCutMixTransform:
    """
    Implements MixUp and CutMix data augmentation techniques.
    These are effective regularization methods that improve model generalization.
    
    MixUp: Blends two images and their labels with a weight sampled from a beta distribution.
    CutMix: Replaces a random region of one image with a patch from another, adjusting labels accordingly.
    """
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=0.2, mixup_prob=0.5, cutmix_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.mode = None
        
    def mixup(self, inputs, targets):
        """MixUp augmentation"""
        batch_size = inputs.size(0)
        indices = torch.randperm(batch_size).to(inputs.device)
        
        # Sample lambda from beta distribution
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0
            
        # Create mixed inputs and targets
        mixed_inputs = lam * inputs + (1 - lam) * inputs[indices]
        
        return mixed_inputs, targets, targets[indices], lam
    
    def cutmix(self, inputs, targets):
        """CutMix augmentation"""
        batch_size = inputs.size(0)
        indices = torch.randperm(batch_size).to(inputs.device)
        
        # Sample lambda from beta distribution
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1.0
            
        # Get dimensions
        h, w = inputs.size()[2:]
        
        # Random cut region
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        # Random center point
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Boundary
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        inputs_copy = inputs.clone()
        inputs_copy[:, :, bby1:bby2, bbx1:bbx2] = inputs[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda proportionally to area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        return inputs_copy, targets, targets[indices], lam
    
    def __call__(self, inputs, targets):
        """Apply either MixUp, CutMix, or no augmentation based on probability"""
        r = np.random.rand()
        
        if r < self.mixup_prob and self.mixup_alpha > 0:
            self.mode = 'mixup'
            return self.mixup(inputs, targets)
        elif r < self.mixup_prob + self.cutmix_prob and self.cutmix_alpha > 0:
            self.mode = 'cutmix'
            return self.cutmix(inputs, targets)
        else:
            self.mode = 'none'
            return inputs, targets, targets, 1.0

class EMA:
    """
    Exponential Moving Average for model weights.
    Maintains a shadow copy of model weights updated as an exponential moving average
    of the trained weights. This often leads to better generalization.
    """
    def __init__(self, model, decay=0.999, device=None):
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        """Register model parameters for EMA tracking"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone().detach()
    
    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone().detach()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def get_transforms(config, is_training=True):
    """
    Create data transformations based on configuration
    
    Args:
        config: Augmentation configuration
        is_training: Whether this is for training set
        
    Returns:
        torchvision transforms composition
    """
    img_size = DATASET_CONFIG['img_size']
    
    if is_training:
        transform_list = [
            transforms.Resize((img_size, img_size))
        ]
        
        # Add training-specific transforms
        if config['enable_random_affine']:
            transform_list.append(
                transforms.RandomAffine(
                    degrees=config['rotation_degrees'],
                    translate=config['translate'],
                    scale=config['scale']
                )
            )
        else:
            # Add individual transforms
            transform_list.append(transforms.RandomHorizontalFlip(p=config['horizontal_flip_prob']))
            transform_list.append(transforms.RandomRotation(degrees=config['rotation_degrees']))
            
        if config['enable_random_perspective']:
            transform_list.append(transforms.RandomPerspective(distortion_scale=0.2, p=0.5))
            
        if config['enable_color_jitter']:
            transform_list.append(transforms.ColorJitter(
                brightness=config['brightness'],
                contrast=config['contrast'],
                saturation=config['saturation'],
                hue=config['hue']
            ))
            
        if config['enable_random_grayscale']:
            transform_list.append(transforms.RandomGrayscale(p=0.1))
            
        if config['random_erase_prob'] > 0:
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ))
            transform_list.append(transforms.RandomErasing(
                p=config['random_erase_prob'],
                scale=config['random_erase_scale']
            ))
        else:
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ))
    else:
        # Validation/test transforms - just resize, convert to tensor, and normalize
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    return transforms.Compose(transform_list)

def compute_class_weights(dataset):
    """Calculate class weights for handling class imbalance"""
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
        class_counts = np.bincount(labels)
    else:
        # For ImageFolder, extract targets
        labels = [y for _, y in dataset.samples]
        class_counts = np.bincount(labels)
    
    total = len(labels)
    weights = [total / (count * len(class_counts)) for count in class_counts]
    return torch.FloatTensor(weights)

def train_one_epoch(model, dataloader, criterion, optimizer, device, 
                    mixup_cutmix=None, gradient_clip_val=0.0, ema=None):
    """Train model for one epoch with various regularization techniques"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use non-blocking transfers with GPU
    non_blocking = device.type == 'cuda'
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device, non_blocking=non_blocking), targets.to(device, non_blocking=non_blocking)
        
        # Apply mixup or cutmix if provided
        if mixup_cutmix is not None:
            inputs, targets_a, targets_b, lam = mixup_cutmix(inputs, targets)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss with mixed targets
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            
            # For accuracy calculation (using dominant label)
            _, predicted = outputs.max(1)
            if lam > 0.5:
                correct += predicted.eq(targets_a).sum().item()
            else:
                correct += predicted.eq(targets_b).sum().item()
        else:
            # Standard forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        
        # Add activation regularization if available
        if hasattr(model, 'get_activation_regularization'):
            activation_reg = model.get_activation_regularization()
            loss = loss + activation_reg
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping if enabled
        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            
        optimizer.step()
        
        # Update EMA if provided
        if ema is not None:
            ema.update()
        
        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        total += targets.size(0)
            
    return running_loss / total, 100. * correct / total

def validate(model, dataloader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use non-blocking transfers with GPU
    non_blocking = device.type == 'cuda'
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating"):
            inputs, targets = inputs.to(device, non_blocking=non_blocking), targets.to(device, non_blocking=non_blocking)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return running_loss / total, 100. * correct / total

def get_optimizer(optimizer_name, model_parameters, lr, weight_decay):
    """Create optimizer based on configuration"""
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model_parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        print(f"Warning: Unknown optimizer {optimizer_name}, using AdamW")
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)

def get_scheduler(scheduler_name, optimizer, epochs, steps_per_epoch=None, **kwargs):
    """Create learning rate scheduler based on configuration"""
    if scheduler_name.lower() == 'reducelr':
        return ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 3),
            verbose=True,
            min_lr=1e-6
        )
    elif scheduler_name.lower() == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=kwargs.get('min_lr', 1e-6)
        )
    elif scheduler_name.lower() == 'step':
        return StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.5)
        )
    elif scheduler_name.lower() == 'onecycle' and steps_per_epoch is not None:
        return OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=kwargs.get('pct_start', 0.3),
            div_factor=kwargs.get('div_factor', 25.0),
            final_div_factor=kwargs.get('final_div_factor', 10000.0)
        )
    else:
        print(f"Warning: Unknown scheduler {scheduler_name}, using CosineAnnealingLR")
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

def save_config(config_dict, save_path):
    """Save configuration to JSON file"""
    # Convert Path objects to strings for JSON serialization
    for key, value in config_dict.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, Path):
                    config_dict[key][subkey] = str(subvalue)
        elif isinstance(value, Path):
            config_dict[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def train_single_run(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                    model_dir, figure_dir, epochs, patience, mixup_cutmix=None,
                    gradient_clip_val=0, ema=None, use_amp=False, scaler=None, fold=None):
    """Run a single training process with given configuration"""
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    
    # Training setup
    best_val_acc = 0.0
    no_improve_epochs = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epochs': [], 'lr': []
    }
    
    fold_str = f" (Fold {fold})" if fold is not None else ""
    print(f'\nStarting training{fold_str} for {epochs} epochs...')
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_start = time.time()
        
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply mixup or cutmix if provided
            targets_a = targets_b = targets
            lam = 1.0
            if mixup_cutmix is not None:
                inputs, targets_a, targets_b, lam = mixup_cutmix(inputs, targets)
            
            # Use AMP if enabled
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Compute loss with mixed targets if mixup/cutmix was applied
                    if mixup_cutmix is not None and mixup_cutmix.mode != 'none':
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    else:
                        loss = criterion(outputs, targets)
                    
                    # Add activation regularization if available
                    if hasattr(model, 'get_activation_regularization'):
                        activation_reg = model.get_activation_regularization()
                        loss = loss + activation_reg
                
                # Backward and optimize with scaler
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping if enabled
                if gradient_clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass without AMP
                outputs = model(inputs)
                
                # Compute loss with mixed targets if mixup/cutmix was applied
                if mixup_cutmix is not None and mixup_cutmix.mode != 'none':
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets)
                
                # Add activation regularization if available
                if hasattr(model, 'get_activation_regularization'):
                    activation_reg = model.get_activation_regularization()
                    loss = loss + activation_reg
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping if enabled
                if gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                
                optimizer.step()
            
            # Update EMA if provided
            if ema is not None:
                ema.update()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            
            # For accuracy calculation
            total += targets.size(0)
            if mixup_cutmix is not None and mixup_cutmix.mode != 'none':
                # Use dominant label for accuracy calculation
                if lam > 0.5:
                    correct += predicted.eq(targets_a).sum().item()
                else:
                    correct += predicted.eq(targets_b).sum().item()
            else:
                correct += predicted.eq(targets).sum().item()
        
        train_loss = running_loss / total
        train_acc = 100. * correct / total
        
        # Validation
        # Apply EMA weights for validation if available
        if ema is not None:
            ema.apply_shadow()
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Restore original weights after validation if using EMA
        if ema is not None:
            ema.restore()
        
        # Update scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Save statistics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch + 1)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - epoch_start
        print(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            
            # Save model with EMA weights if available
            if ema is not None:
                ema.apply_shadow()
            """
            best_model_path = model_dir / 'kan_person_detector_best.pt'
            torch.save({     
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history,
                'used_ema': ema is not None
            }, best_model_path)
            """
            
            best_model_path = model_dir / 'kan_person_detector_best.pt'
            torch.save({     
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history,
                'used_ema': ema is not None,
                'model_info': {
                    'config': {
                        'img_size': DATASET_CONFIG['img_size'],
                        'preprocessor': {
                            'input_channels': PREPROCESSOR_CONFIG['input_channels'],
                            'output_features': PREPROCESSOR_CONFIG['output_features'],
                            'conv_channels': PREPROCESSOR_CONFIG['conv_channels'],
                            'kernel_size': PREPROCESSOR_CONFIG['kernel_size'],
                            'pool_kernel_size': PREPROCESSOR_CONFIG['pool_kernel_size'],
                            'final_pool_size': PREPROCESSOR_CONFIG['final_pool_size'],
                            'use_batch_norm': PREPROCESSOR_CONFIG['use_batch_norm'],
                            'dropout_rate': PREPROCESSOR_CONFIG.get('dropout_rate', 0.0),
                            'l2_regularization': PREPROCESSOR_CONFIG.get('l2_regularization', 0.0),
                            'stochastic_depth_rate': PREPROCESSOR_CONFIG.get('stochastic_depth_rate', 0.0)
                        },
                        'kan': {
                            'feature_dim': KAN_CONFIG['feature_dim'],
                            'hidden_dims': KAN_CONFIG['hidden_dims'],
                            'grid': KAN_CONFIG['grid'],
                            'degree': KAN_CONFIG['degree'],
                            'dropout_rate': KAN_CONFIG.get('dropout_rate', 0.0),
                            'weight_decay': KAN_CONFIG.get('weight_decay', 0.0),
                            'activation_l1': KAN_CONFIG.get('activation_l1', 0.0),
                            'use_batchnorm': KAN_CONFIG.get('use_batchnorm', False),
                            'seed': KAN_CONFIG.get('seed', 42)
                        },
                        'augmentation': {
                            'horizontal_flip_prob': AUGMENTATION_CONFIG.get('horizontal_flip_prob', 0.0),
                            'rotation_degrees': AUGMENTATION_CONFIG.get('rotation_degrees', 0),
                            'translate': AUGMENTATION_CONFIG.get('translate', (0.0, 0.0)),
                            'scale': AUGMENTATION_CONFIG.get('scale', (1.0, 1.0)),
                            'brightness': AUGMENTATION_CONFIG.get('brightness', 0.0),
                            'contrast': AUGMENTATION_CONFIG.get('contrast', 0.0),
                            'saturation': AUGMENTATION_CONFIG.get('saturation', 0.0),
                            'hue': AUGMENTATION_CONFIG.get('hue', 0.0),
                            'enable_color_jitter': AUGMENTATION_CONFIG.get('enable_color_jitter', False),
                            'enable_random_affine': AUGMENTATION_CONFIG.get('enable_random_affine', False),
                            'enable_random_perspective': AUGMENTATION_CONFIG.get('enable_random_perspective', False),
                            'enable_random_grayscale': AUGMENTATION_CONFIG.get('enable_random_grayscale', False),
                            'random_erase_prob': AUGMENTATION_CONFIG.get('random_erase_prob', 0.0),
                            'random_erase_scale': AUGMENTATION_CONFIG.get('random_erase_scale', (0.02, 0.33)),
                            'mixup_alpha': AUGMENTATION_CONFIG.get('mixup_alpha', 0.0),
                            'mixup_prob': AUGMENTATION_CONFIG.get('mixup_prob', 0.0),
                            'cutmix_alpha': AUGMENTATION_CONFIG.get('cutmix_alpha', 0.0),
                            'cutmix_prob': AUGMENTATION_CONFIG.get('cutmix_prob', 0.0)
                        },
                        'training': {
                            'batch_size': TRAINING_CONFIG['batch_size'],
                            'learning_rate': TRAINING_CONFIG['learning_rate'],
                            'weight_decay': TRAINING_CONFIG['weight_decay'],
                            'lr_scheduler': TRAINING_CONFIG['lr_scheduler'],
                            'optimizer': TRAINING_CONFIG['optimizer'],
                            'label_smoothing': TRAINING_CONFIG.get('label_smoothing', 0.0),
                            'gradient_clip_val': TRAINING_CONFIG.get('gradient_clip_val', 0.0),
                            'ema_decay': TRAINING_CONFIG.get('ema_decay', 0.0),
                            'enable_mixup': TRAINING_CONFIG.get('enable_mixup', False),
                            'enable_cutmix': TRAINING_CONFIG.get('enable_cutmix', False)
                        }
                    },
                    'performance': {
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                        'train_acc': train_acc,
                        'train_loss': train_loss
                    },
                    'model_architecture': {
                        'total_params': param_counts['total'] if 'param_counts' in locals() else None,
                        'trainable_params': param_counts['trainable'] if 'param_counts' in locals() else None,
                        'cnn_params': param_counts['cnn'] if 'param_counts' in locals() else None,
                        'kan_params': param_counts['kan'] if 'param_counts' in locals() else None,
                        'model_size_mb': model_size_mb if 'model_size_mb' in locals() else None
                    }
                }
            }, best_model_path)
            
            # Restore original weights if using EMA
            if ema is not None:
                ema.restore()
                
            print(f"New best model saved! Accuracy: {val_acc:.2f}%")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")
        
        # Early stopping
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
        
        # Save checkpoint every N epochs
        if TRAINING_CONFIG['save_checkpoints'] and (epoch + 1) % TRAINING_CONFIG['checkpoint_interval'] == 0:
            checkpoint_path = model_dir / f'kan_person_detector_epoch_{epoch+1}.pt'
            
            if ema is not None:
                ema.apply_shadow()
                
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'used_ema': ema is not None
            }, checkpoint_path)
            
            if ema is not None:
                ema.restore()
                
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Total training time
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training curves
    plot_training_curves(history, save_dir=figure_dir)
    
    # Save final model with EMA if available
    if ema is not None:
        ema.apply_shadow()
    
    """
    final_model_path = model_dir / 'kan_person_detector_final.pt'
    torch.save({
        'epoch': len(history['epochs']),
        'model_state_dict': model.state_dict(),
        'history': history,
        'accuracy': best_val_acc,
        'used_ema': ema is not None
    }, final_model_path)
    """
    final_model_path = model_dir / 'kan_person_detector_final.pt'
    torch.save({
        'epoch': len(history['epochs']),
        'model_state_dict': model.state_dict(),
        'history': history,
        'accuracy': best_val_acc,
        'used_ema': ema is not None,
        'model_info': {
            'config': {
                'img_size': DATASET_CONFIG['img_size'],
                'preprocessor': {
                    'input_channels': PREPROCESSOR_CONFIG['input_channels'],
                    'output_features': PREPROCESSOR_CONFIG['output_features'],
                    'conv_channels': PREPROCESSOR_CONFIG['conv_channels'],
                    'kernel_size': PREPROCESSOR_CONFIG['kernel_size'],
                    'pool_kernel_size': PREPROCESSOR_CONFIG['pool_kernel_size'],
                    'final_pool_size': PREPROCESSOR_CONFIG['final_pool_size'],
                    'use_batch_norm': PREPROCESSOR_CONFIG['use_batch_norm'],
                    'dropout_rate': PREPROCESSOR_CONFIG.get('dropout_rate', 0.0),
                    'l2_regularization': PREPROCESSOR_CONFIG.get('l2_regularization', 0.0),
                    'stochastic_depth_rate': PREPROCESSOR_CONFIG.get('stochastic_depth_rate', 0.0)
                },
                'kan': {
                    'feature_dim': KAN_CONFIG['feature_dim'],
                    'hidden_dims': KAN_CONFIG['hidden_dims'],
                    'grid': KAN_CONFIG['grid'],
                    'degree': KAN_CONFIG['degree'],
                    'dropout_rate': KAN_CONFIG.get('dropout_rate', 0.0),
                    'weight_decay': KAN_CONFIG.get('weight_decay', 0.0),
                    'activation_l1': KAN_CONFIG.get('activation_l1', 0.0),
                    'use_batchnorm': KAN_CONFIG.get('use_batchnorm', False),
                    'seed': KAN_CONFIG.get('seed', 42)
                },
                'augmentation': {
                    'horizontal_flip_prob': AUGMENTATION_CONFIG.get('horizontal_flip_prob', 0.0),
                    'rotation_degrees': AUGMENTATION_CONFIG.get('rotation_degrees', 0),
                    'translate': AUGMENTATION_CONFIG.get('translate', (0.0, 0.0)),
                    'scale': AUGMENTATION_CONFIG.get('scale', (1.0, 1.0)),
                    'brightness': AUGMENTATION_CONFIG.get('brightness', 0.0),
                    'contrast': AUGMENTATION_CONFIG.get('contrast', 0.0),
                    'saturation': AUGMENTATION_CONFIG.get('saturation', 0.0),
                    'hue': AUGMENTATION_CONFIG.get('hue', 0.0),
                    'enable_color_jitter': AUGMENTATION_CONFIG.get('enable_color_jitter', False),
                    'enable_random_affine': AUGMENTATION_CONFIG.get('enable_random_affine', False),
                    'enable_random_perspective': AUGMENTATION_CONFIG.get('enable_random_perspective', False),
                    'enable_random_grayscale': AUGMENTATION_CONFIG.get('enable_random_grayscale', False),
                    'random_erase_prob': AUGMENTATION_CONFIG.get('random_erase_prob', 0.0),
                    'random_erase_scale': AUGMENTATION_CONFIG.get('random_erase_scale', (0.02, 0.33)),
                    'mixup_alpha': AUGMENTATION_CONFIG.get('mixup_alpha', 0.0),
                    'mixup_prob': AUGMENTATION_CONFIG.get('mixup_prob', 0.0),
                    'cutmix_alpha': AUGMENTATION_CONFIG.get('cutmix_alpha', 0.0),
                    'cutmix_prob': AUGMENTATION_CONFIG.get('cutmix_prob', 0.0)
                },
                'training': {
                    'batch_size': TRAINING_CONFIG['batch_size'],
                    'learning_rate': TRAINING_CONFIG['learning_rate'],
                    'weight_decay': TRAINING_CONFIG['weight_decay'],
                    'lr_scheduler': TRAINING_CONFIG['lr_scheduler'],
                    'optimizer': TRAINING_CONFIG['optimizer'],
                    'label_smoothing': TRAINING_CONFIG.get('label_smoothing', 0.0),
                    'gradient_clip_val': TRAINING_CONFIG.get('gradient_clip_val', 0.0),
                    'ema_decay': TRAINING_CONFIG.get('ema_decay', 0.0),
                    'enable_mixup': TRAINING_CONFIG.get('enable_mixup', False),
                    'enable_cutmix': TRAINING_CONFIG.get('enable_cutmix', False)
                }
            },
            'performance': {
                'val_acc': best_val_acc,
                'val_loss': val_loss,  # Final validation loss
                'train_acc': train_acc,  # Final training accuracy 
                'train_loss': train_loss  # Final training loss
            },
            'model_architecture': {
                'total_params': param_counts['total'] if 'param_counts' in locals() else None,
                'trainable_params': param_counts['trainable'] if 'param_counts' in locals() else None,
                'cnn_params': param_counts['cnn'] if 'param_counts' in locals() else None,
                'kan_params': param_counts['kan'] if 'param_counts' in locals() else None,
                'model_size_mb': model_size_mb if 'model_size_mb' in locals() else None
            }
        }
    }, final_model_path)
    
    if ema is not None:
        ema.restore()
    
    # Save training summary
    summary = {
        'training_time': training_time,
        'best_accuracy': best_val_acc,
        'final_learning_rate': optimizer.param_groups[0]['lr'],
        'total_epochs': len(history['epochs']),
        'early_stopped': len(history['epochs']) < epochs,
        'used_mixup': mixup_cutmix is not None,
        'used_ema': ema is not None,
        'used_amp': use_amp
    }
    
    summary_path = Path(model_dir).parent / 'analysis' / f'training_summary{"_fold_"+str(fold) if fold else ""}.json'
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Final model saved to: {final_model_path}")
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Train KAN Image Classifier')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom configuration JSON file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--cross_val', action='store_true',
                        help='Enable cross-validation')
    parser.add_argument('--disable_mixup', action='store_true',
                        help='Disable mixup and cutmix augmentation')
    parser.add_argument('--disable_ema', action='store_true',
                        help='Disable exponential moving average')
    parser.add_argument('--precision', type=str, choices=['float32', 'mixed', 'float16'], 
                        default=None, help='Training precision')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    # Add GPU-specific arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--gpu_index', type=int, default=0,
                        help='GPU index to use if multiple GPUs available')
    args = parser.parse_args()
    
    # Use custom configuration if provided
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            # Update configurations
            for key, value in custom_config.items():
                if key.upper() in globals():
                    globals()[key.upper()].update(value)
    
    # Set random seed
    seed = args.seed if args.seed is not None else DATASET_CONFIG['random_seed']
    seed_everything(seed)
    
    # Get paths for current experiment
    paths = get_experiment_paths()
    model_dir = paths['model_dir']
    figure_dir = paths['figure_dir']
    analysis_dir = paths['analysis_dir']
    
    # Save configuration
    config_dict = {
        'dataset_config': DATASET_CONFIG,
        'augmentation_config': AUGMENTATION_CONFIG,
        'preprocessor_config': PREPROCESSOR_CONFIG,
        'kan_config': KAN_CONFIG,
        'training_config': TRAINING_CONFIG,
    }
    save_config(config_dict, paths['experiment_dir'] / 'config.json')
    
    # Set device - Enhanced detection for Colab
    if args.device:
        device_name = args.device
    else:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # For multi-GPU setups, allow selecting a specific GPU
    if device_name == 'cuda' and torch.cuda.device_count() > 1:
        gpu_idx = args.gpu_index if args.gpu_index < torch.cuda.device_count() else 0
        device = torch.device(f'cuda:{gpu_idx}')
        print(f'Multiple GPUs detected. Using GPU {gpu_idx}: {torch.cuda.get_device_name(gpu_idx)}')
    else:
        device = torch.device(device_name)
    
    print(f'Using device: {device}')
    
    # Show GPU info if using CUDA
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(device)}')
        print(f'CUDA Version: {torch.version.cuda}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB')
        # Set optimal CUDA performance
        torch.backends.cudnn.benchmark = True
    
    # Set precision - Encourage mixed precision for GPU
    precision = args.precision if args.precision else TRAINING_CONFIG['precision']
    if device.type == 'cuda' and precision == 'float32':
        print("Consider using mixed precision for better performance on GPU. Use --precision mixed")
    print(f'Using precision: {precision}')
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp, scaler = initialize_amp(precision, device_type)

    # Data transformations
    train_transform = get_transforms(AUGMENTATION_CONFIG, is_training=True)
    val_transform = get_transforms(AUGMENTATION_CONFIG, is_training=False)
    
    # Load data
    train_dir = Path(DATASET_CONFIG['subset_dir']) / 'train'
    val_dir = Path(DATASET_CONFIG['subset_dir']) / 'val'
    
    if train_dir.exists() and val_dir.exists():
        # Load datasets with transformations
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        
        # Calculate class weights for handling class imbalance
        class_weights = compute_class_weights(train_dataset)
        print(f"Class weights: {class_weights}")
        
        # Create data loaders with GPU-optimized settings
        num_workers = TRAINING_CONFIG['num_workers']
        print(f"Using {num_workers} workers for data loading")

        # When using GPU, we want to pin memory for faster data transfer
        pin_memory = device.type == 'cuda'
        # Use non-blocking transfers when possible for better performance with GPUs
        non_blocking = device.type == 'cuda'

        # Adjust prefetch factor for better utilization of workers with GPUs
        prefetch_factor = 2 if device.type == 'cuda' else None

        train_loader = DataLoader(
            train_dataset, 
            batch_size=TRAINING_CONFIG['batch_size'],
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAINING_CONFIG['val_batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
                       
        print(f"Training dataset: {len(train_dataset)} images")
        print(f"Validation dataset: {len(val_dataset)} images")
        print(f"Classes: {train_dataset.classes}")
    else:
        print(f"Directories not found: {train_dir} or {val_dir}")
        print("Please run prepare_dataset.py first.")
        sys.exit(1)
    
    # Create model with all regularization parameters
    model = KANImageClassifier(
        input_channels=PREPROCESSOR_CONFIG['input_channels'],
        conv_channels=PREPROCESSOR_CONFIG['conv_channels'],
        img_size=DATASET_CONFIG['img_size'],
        num_classes=2,
        feature_dim=KAN_CONFIG['feature_dim'],
        kan_hidden_dims=KAN_CONFIG['hidden_dims'],
        kan_grid=KAN_CONFIG['grid'],
        kan_degree=KAN_CONFIG['degree'],
        use_batch_norm=PREPROCESSOR_CONFIG['use_batch_norm'],
        dropout_rate=KAN_CONFIG['dropout_rate'],
        activation_l1=KAN_CONFIG['activation_l1'],
        stochastic_depth_rate=PREPROCESSOR_CONFIG.get('stochastic_depth_rate', 0.0),
        seed=KAN_CONFIG['seed']
    )
    model = model.to(device)
    
    # Print model info
    param_counts = model.get_parameter_count()
    model_size_mb = model.get_model_size()
    
    print(f'Model information:')
    print(f'  Total parameters: {param_counts["total"]:,}')
    print(f'  Trainable parameters: {param_counts["trainable"]:,}')
    print(f'  Model size: {model_size_mb:.2f} MB')
    print(f'  CNN parameters: {param_counts["cnn"]:,} ({param_counts["cnn"]/param_counts["total"]*100:.1f}% of total)')
    print(f'  KAN parameters: {param_counts["kan"]:,} ({param_counts["kan"]/param_counts["total"]*100:.1f}% of total)')
    
    # Save model architecture summary
    with open(analysis_dir / 'model_architecture.txt', 'w') as f:
        f.write(f"KAN Image Classifier Architecture\n")
        f.write("="*50 + "\n\n")
        f.write(f"Image size: {DATASET_CONFIG['img_size']}x{DATASET_CONFIG['img_size']}\n")
        f.write(f"Feature dimension: {KAN_CONFIG['feature_dim']}\n")
        f.write(f"KAN hidden dimensions: {KAN_CONFIG['hidden_dims']}\n")
        f.write(f"KAN grid points: {KAN_CONFIG['grid']}\n")
        f.write(f"KAN spline degree: {KAN_CONFIG['degree']}\n")
        f.write(f"Dropout rate: {KAN_CONFIG['dropout_rate']}\n")
        f.write(f"Activation L1: {KAN_CONFIG['activation_l1']}\n\n")
        f.write(f"Total parameters: {param_counts['total']:,}\n")
        f.write(f"Trainable parameters: {param_counts['trainable']:,}\n")
        f.write(f"Model size: {model_size_mb:.2f} MB\n")
        f.write(f"CNN parameters: {param_counts['cnn']:,} ({param_counts['cnn']/param_counts['total']*100:.1f}% of total)\n")
        f.write(f"KAN parameters: {param_counts['kan']:,} ({param_counts['kan']/param_counts['total']*100:.1f}% of total)\n")
    
    # Setup for MixUp and CutMix
    enable_mixup = TRAINING_CONFIG['enable_mixup'] and not args.disable_mixup
    enable_cutmix = TRAINING_CONFIG['enable_cutmix'] and not args.disable_mixup
    
    # Create MixUp/CutMix transform if enabled
    mixup_cutmix = None
    if enable_mixup or enable_cutmix:
        mixup_alpha = AUGMENTATION_CONFIG.get('mixup_alpha', 0.2) if enable_mixup else 0
        cutmix_alpha = AUGMENTATION_CONFIG.get('cutmix_alpha', 0.2) if enable_cutmix else 0
        mixup_prob = AUGMENTATION_CONFIG.get('mixup_prob', 0.5) if enable_mixup else 0
        cutmix_prob = AUGMENTATION_CONFIG.get('cutmix_prob', 0.5) if enable_cutmix else 0
        
        mixup_cutmix = MixUpCutMixTransform(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mixup_prob=mixup_prob,
            cutmix_prob=cutmix_prob
        )
        
        print(f"MixUp enabled: {enable_mixup}, alpha={mixup_alpha}")
        print(f"CutMix enabled: {enable_cutmix}, alpha={cutmix_alpha}")
    
    # Setup for EMA
    enable_ema = TRAINING_CONFIG.get('ema_decay', 0) > 0 and not args.disable_ema
    ema = None
    if enable_ema:
        ema = EMA(model, decay=TRAINING_CONFIG['ema_decay'], device=device)
        print(f"EMA enabled with decay {TRAINING_CONFIG['ema_decay']}")
    
    # Loss function with label smoothing if configured
    label_smoothing = TRAINING_CONFIG.get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing
    )
    
    # Optimizer
    optimizer = get_optimizer(
        TRAINING_CONFIG['optimizer'],
        model.parameters()
    # Загрузить модель из чекпоинта, если указан параметр resume_from
    if args.resume_from and os.path.exists(args.resume_from):
        print(f'Resuming from checkpoint: {args.resume_from}')
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_acc = checkpoint.get('val_acc', 0)
        print(f'Loaded model with validation accuracy: {best_val_acc:.2f}%')
    ,
        TRAINING_CONFIG['learning_rate'],
        TRAINING_CONFIG['weight_decay']
    )
    
    # Cross-validation setup
    n_folds = TRAINING_CONFIG.get('cross_validation_folds', 0)
    if args.cross_val:
        n_folds = max(5, n_folds)  # Use at least 5 folds for cross-validation
    
    if n_folds > 1:
        print(f"Starting {n_folds}-fold cross-validation...")
        # Initialize fold results
        fold_results = []
        
        # Create K-fold cross-validator
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        # Convert dataset to numpy arrays for splitting
        X = np.array(train_dataset.samples)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\n{'='*40}\nFold {fold+1}/{n_folds}\n{'='*40}")
            
            # Reset model for each fold
            model = KANImageClassifier(
                input_channels=PREPROCESSOR_CONFIG['input_channels'],
                conv_channels=PREPROCESSOR_CONFIG['conv_channels'],
                img_size=DATASET_CONFIG['img_size'],
                num_classes=2,
                feature_dim=KAN_CONFIG['feature_dim'],
                kan_hidden_dims=KAN_CONFIG['hidden_dims'],
                kan_grid=KAN_CONFIG['grid'],
                kan_degree=KAN_CONFIG['degree'],
                use_batch_norm=PREPROCESSOR_CONFIG['use_batch_norm'],
                dropout_rate=KAN_CONFIG['dropout_rate'],
                activation_l1=KAN_CONFIG['activation_l1'],
                stochastic_depth_rate=PREPROCESSOR_CONFIG.get('stochastic_depth_rate', 0.0),
                seed=KAN_CONFIG['seed']
            ).to(device)
            
            # Reset optimizer and scheduler
            optimizer = get_optimizer(
                TRAINING_CONFIG['optimizer'],
                model.parameters(),
                TRAINING_CONFIG['learning_rate'],
                TRAINING_CONFIG['weight_decay']
            )
            
            # Create fold-specific dataloaders
            fold_train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            fold_val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            fold_train_loader = DataLoader(
                train_dataset,
                batch_size=TRAINING_CONFIG['batch_size'],
                sampler=fold_train_sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            fold_val_loader = DataLoader(
                train_dataset,
                batch_size=TRAINING_CONFIG['val_batch_size'],
                sampler=fold_val_sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            # Learning rate scheduler
            scheduler = get_scheduler(
                TRAINING_CONFIG['lr_scheduler'],
                optimizer,
                TRAINING_CONFIG['epochs'],
                steps_per_epoch=len(fold_train_loader),
                factor=TRAINING_CONFIG['lr_factor'],
                patience=TRAINING_CONFIG['lr_patience']
            )
            
            # Reset EMA if used
            if enable_ema:
                ema = EMA(model, decay=TRAINING_CONFIG['ema_decay'], device=device)
            
            # Train for this fold
            fold_best_val_acc = train_single_run(
                model=model,
                train_loader=fold_train_loader,
                val_loader=fold_val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                model_dir=model_dir / f"fold_{fold+1}",
                figure_dir=figure_dir / f"fold_{fold+1}",
                epochs=TRAINING_CONFIG['epochs'],
                patience=TRAINING_CONFIG['early_stopping_patience'],
                mixup_cutmix=mixup_cutmix,
                gradient_clip_val=TRAINING_CONFIG.get('gradient_clip_val', 0),
                ema=ema,
                use_amp=use_amp,
                scaler=scaler,
                fold=fold+1
            )
            
            fold_results.append(fold_best_val_acc)
            
        # Report cross-validation results
        print("\nCross-validation results:")
        for fold, acc in enumerate(fold_results):
            print(f"Fold {fold+1}: {acc:.2f}%")
        print(f"Mean accuracy: {np.mean(fold_results):.2f}%")
        print(f"Standard deviation: {np.std(fold_results):.2f}%")
        
        # Save cross-validation results
        cv_results = {
            'fold_accuracies': fold_results,
            'mean_accuracy': float(np.mean(fold_results)),
            'std_accuracy': float(np.std(fold_results))
        }
        
        with open(analysis_dir / 'cross_validation_results.json', 'w') as f:
            json.dump(cv_results, f, indent=2)
    else:
        # Standard single training run
        # Learning rate scheduler
        scheduler = get_scheduler(
            TRAINING_CONFIG['lr_scheduler'],
            optimizer,
            TRAINING_CONFIG['epochs'],
            steps_per_epoch=len(train_loader),
            factor=TRAINING_CONFIG['lr_factor'],
            patience=TRAINING_CONFIG['lr_patience']
        )
        
        best_val_acc = train_single_run(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            model_dir=model_dir,
            figure_dir=figure_dir,
            epochs=TRAINING_CONFIG['epochs'],
            patience=TRAINING_CONFIG['early_stopping_patience'],
            mixup_cutmix=mixup_cutmix,
            gradient_clip_val=TRAINING_CONFIG.get('gradient_clip_val', 0),
            ema=ema,
            use_amp=use_amp,
            scaler=scaler
        )
        
        print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
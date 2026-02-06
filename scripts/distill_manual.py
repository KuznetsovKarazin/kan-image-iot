import os
import torch

from pathlib import Path

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import argparse
import json
import sys

# Add the parent directory to Python's path to find the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc,
    precision_recall_curve, 
    average_precision_score,
    precision_score,
    recall_score
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Import project modules
from models.kan_model import KANImageClassifier
from utils.metrics import measure_inference_time
from utils.visualization import plot_confusion_matrix, show_misclassified

# Import configuration from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_experiment_paths, get_experiment_name, DATASET_CONFIG, KAN_CONFIG, PREPROCESSOR_CONFIG, TRAINING_CONFIG, AUGMENTATION_CONFIG, OUTPUT_DIR

import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    # Hard loss: cross-entropy con etichette vere
    hard_loss = F.cross_entropy(student_logits, labels)

    # Soft loss: KL divergence tra distribuzioni teacher e student
    soft_teacher = F.log_softmax(teacher_logits / T, dim=1)
    soft_student = F.softmax(student_logits / T, dim=1)
    soft_loss = F.kl_div(soft_teacher, soft_student, reduction="batchmean") * (T * T)

    return (1 - alpha) * hard_loss + alpha * soft_loss

def load_model_and_config(model_path, device):
    """Load trained model and its configuration from checkpoint"""
    model_path = Path(model_path)

    # ВАЖНО: всегда грузим на CPU, чтобы не падать на чужих CUDA-чекпоинтах
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    
    # Extract model configuration from checkpoint
    model_info = checkpoint.get('model_info', {})
    config = model_info.get('config', {})
    
    # Default architecture from current config if not specified in checkpoint
    img_size = config.get('img_size', DATASET_CONFIG['img_size'])
    preprocessor_config = config.get('preprocessor', {})
    kan_config = config.get('kan', {})
    
    # Extract parameters with defaults from current config
    feature_dim = kan_config.get('feature_dim', KAN_CONFIG['feature_dim'])
    hidden_dims = kan_config.get('hidden_dims', KAN_CONFIG['hidden_dims'])
    grid = kan_config.get('grid', KAN_CONFIG['grid'])
    degree = kan_config.get('degree', KAN_CONFIG['degree'])
    conv_channels = preprocessor_config.get('conv_channels', PREPROCESSOR_CONFIG['conv_channels'])
    use_batch_norm = preprocessor_config.get('use_batch_norm', PREPROCESSOR_CONFIG['use_batch_norm'])
    preprocessor_type = preprocessor_config.get('preprocessor_type', PREPROCESSOR_CONFIG['preprocessor_type'])

    # Create model with the same architecture
    model = KANImageClassifier(
        input_channels=3,
        img_size=img_size,
        num_classes=2,
        feature_dim=feature_dim,
        kan_hidden_dims=hidden_dims,
        kan_grid=grid,
        kan_degree=degree,
        conv_channels=conv_channels,
        use_batch_norm=use_batch_norm,
        preprocessor_type=preprocessor_type,
    )
    
    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)

    # Переносим модель на нужное устройство (cuda / cpu / mps)
    model = model.to(device)
    
    # Extract additional info
    try:
        best_epoch = checkpoint.get('epoch', 0)
        best_accuracy = checkpoint.get('val_acc', checkpoint.get('accuracy', 0))
        history = checkpoint.get('history', {})
    except Exception:
        best_epoch = 0
        best_accuracy = 0
        history = {}
    
    return model, {
        'best_epoch': best_epoch,
        'best_accuracy': best_accuracy,
        'history': history,
        'img_size': img_size,
        'feature_dim': feature_dim,
        'hidden_dims': hidden_dims,
        'grid': grid,
        'degree': degree
    }

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

# Funzione per salvare il checkpoint
def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint salvato all'epoca {epoch} in {path}")

# Funzione per caricare il checkpoint
def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location="cpu")  # usa "cuda" se vuoi forzare GPU
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        loss = checkpoint["loss"]
        print(f"Checkpoint caricato da {path}, riparto dall'epoca {start_epoch}")
        return start_epoch, loss
    else:
        print("Nessun checkpoint trovato, riparto da zero.")
        return 0, None

def save_model(model, epoch=0, val_acc=0.0, path="model.pth"):

     torch.save({     
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'model_info': {
                    'config': {
                        'img_size': DATASET_CONFIG['img_size'],
                        'preprocessor': {
                            'preprocessor_type': PREPROCESSOR_CONFIG['preprocessor_type'],
                            'width_mult': PREPROCESSOR_CONFIG.get('width_mult', 0.75),
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
                        'val_loss': None,
                        'train_acc': None,
                        'train_loss': None
                    },
                    'model_architecture': {
                        'total_params': param_counts['total'] if 'param_counts' in locals() else None,
                        'trainable_params': param_counts['trainable'] if 'param_counts' in locals() else None,
                        'cnn_params': param_counts['cnn'] if 'param_counts' in locals() else None,
                        'kan_params': param_counts['kan'] if 'param_counts' in locals() else None,
                        'model_size_mb': model_size_mb if 'model_size_mb' in locals() else None
                    }
                }
            }, path)

# Training loop con monitoraggio accuracy
def train_distill(student, teacher, train_loader, val_loader, optimizer, device, epochs=30):

    paths = get_experiment_paths()

    teacher.eval()  # teacher congelato
    student.train()

    prev_val_acc = 0.0

    for epoch in range(epochs):
        total, correct = 0, 0
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"): 
            images, labels = images.to(device), labels.to(device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher(images)

            # Student forward
            student_logits = student(images)

            # Loss
            loss = distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy su batch
            _, predicted = torch.max(student_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        # Accuracy epoca
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # Valutazione su validation set
        student.eval()
        val_total, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total
        student.train()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {avg_loss:.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}%")
        
        checkpoint_path = paths['model_dir'] / f"distilled_checkpoint_epoch{epoch+1}.pth"
        save_checkpoint(student, optimizer, epoch, avg_loss, path=checkpoint_path)

        if val_acc > prev_val_acc:
            print("Best model up to now, model saved.")
            best_model_path = paths['model_dir'] / "distilled_best_model.pth"
            save_model(student, epoch, val_acc, path=best_model_path)
            prev_val_acc = val_acc

        
def main():
    parser = argparse.ArgumentParser(description='Analyze KAN Person Detection Model')
    parser.add_argument('--model_path', type=str, 
                        default=None,
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, 
                        default=None,
                        help='Path to validation data')
    parser.add_argument('--test_dir', type=str, 
                        default=None,
                        help='Path to test data (if different from validation)')
    parser.add_argument('--experiment_name', type=str, 
                        default=None,
                        help='Experiment name (if not using current config)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for evaluation')
    # Add GPU-specific arguments
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                        default='auto', help='Device to use for evaluation')
    parser.add_argument('--gpu_index', type=int, default=0,
                        help='GPU index to use if multiple GPUs available')
    args = parser.parse_args()
    
    # Set device with enhanced detection for Colab
    if args.device == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_name = args.device
    
    # For multi-GPU setups, allow selecting a specific GPU
    if device_name == 'cuda' and torch.cuda.device_count() > 1:
        gpu_idx = args.gpu_index if args.gpu_index < torch.cuda.device_count() else 0
        device = torch.device(f'cuda:{gpu_idx}')
        print(f'Multiple GPUs detected. Using GPU {gpu_idx}: {torch.cuda.get_device_name(gpu_idx)}')
    else:
        device = torch.device(device_name)
    
    print(f"Using device: {device}")
    
    # Show GPU info if using CUDA
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(device)}')
        print(f'CUDA Version: {torch.version.cuda}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB')
        
        # Set optimal CUDA performance settings for inference
        torch.backends.cudnn.benchmark = True
   
    # Get experiment paths
    if args.experiment_name:
        # Use provided experiment name
        exp_name = args.experiment_name
        paths = {
            'experiment_dir': Path('experiment_data') / exp_name,
            'model_dir': Path('experiment_data') / exp_name / 'models',
            'figure_dir': Path('experiment_data') / exp_name / 'figures',
            'analysis_dir': Path('experiment_data') / exp_name / 'analysis',
        }
    else:
        # Use current configuration
        paths = get_experiment_paths()
    
    # Create directories if they don't exist
    for dir_path in paths.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        # Try to find the best model in the experiment directory
        best_model_path = paths['model_dir'] / 'kan_person_detector_best.pt'
        final_model_path = paths['model_dir'] / 'kan_person_detector_final.pt'
        
        if best_model_path.exists():
            model_path = best_model_path
        elif final_model_path.exists():
            model_path = final_model_path
        else:
            print(f"No model found in {paths['model_dir']}")
            print("Please specify --model_path or train a model first")
            return
    
    # Load model and configuration 
    teacher, model_config = load_model_and_config(model_path, device=device)
    teacher.eval()

    # Specular to teacher
    student = KANImageClassifier(
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
                seed=KAN_CONFIG['seed'],
                preprocessor_type=PREPROCESSOR_CONFIG['preprocessor_type'],
                width_mult=0.75,
                preprocessor_pretrained=False,
                preprocessor_freeze=False,
            ).to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=TRAINING_CONFIG['learning_rate'])

    train_dir = Path(DATASET_CONFIG['subset_dir']) / 'train'
    val_dir = Path(DATASET_CONFIG['subset_dir']) / 'val'

    # Definisci le trasformazioni
    train_transform = get_transforms(AUGMENTATION_CONFIG, is_training=True)
    val_transform = get_transforms(AUGMENTATION_CONFIG, is_training=False)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=TRAINING_CONFIG['batch_size'],
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=TRAINING_CONFIG['val_batch_size'],
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    train_distill(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=30
    )





if __name__ == "__main__":
    main()


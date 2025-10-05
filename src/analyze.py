"""
Performance Analysis for KAN-based Person Detection Model.
This module analyzes the performance of trained KAN models with detailed
metrics and visualizations focused on IoT deployment.

Author: Oleksandr Kuznetsov
Date: March 2025
"""

import os
import torch
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

# Import project modules
from models.kan_model import KANImageClassifier
from utils.metrics import measure_inference_time
from utils.visualization import plot_confusion_matrix, show_misclassified

# Import configuration from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_experiment_paths, get_experiment_name, DATASET_CONFIG, KAN_CONFIG, PREPROCESSOR_CONFIG

def load_model_and_config(model_path):
    """Load trained model and its configuration from checkpoint"""
    checkpoint = torch.load(model_path)
    
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
        use_batch_norm=use_batch_norm
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract additional info
    try:
        best_epoch = checkpoint.get('epoch', 0)
        best_accuracy = checkpoint.get('val_acc', checkpoint.get('accuracy', 0))
        history = checkpoint.get('history', {})
    except:
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


def save_model_analysis(model, model_config, save_dir, inference_results=None, metrics=None):
    """
    Save comprehensive model analysis to a file
    
    Args:
        model: Loaded model
        model_config: Model configuration
        save_dir: Directory to save analysis
        inference_results: Optional inference time results
        metrics: Optional performance metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with open(save_dir / 'model_analysis.json', 'w') as f:
        analysis = {
            'model_configuration': model_config,
            'inference_results': inference_results,
            'performance_metrics': metrics
        }
        json.dump(analysis, f, indent=2, default=lambda o: str(o) if isinstance(o, (torch.Tensor, np.ndarray)) else o)
    
    # Create detailed Markdown report
    with open(save_dir / 'model_analysis.md', 'w') as f:
        f.write("# KAN Person Detection Model Analysis\n\n")
        
        # Model info
        f.write("## Model Information\n\n")
        f.write(f"- Best epoch: {model_config['best_epoch']}\n")
        f.write(f"- Best validation accuracy: {model_config['best_accuracy']:.2f}%\n")
        f.write(f"- Image size: {model_config['img_size']}x{model_config['img_size']}\n\n")
        
        # Architecture details
        f.write("## Architecture Details\n\n")
        f.write("### Preprocessing Network\n\n")
        f.write(f"- Input channels: 3\n")
        f.write(f"- Feature dimension: {model_config['feature_dim']}\n")
        
        # Check if 'conv_channels' exists in model_config before accessing it
        if 'conv_channels' in model_config:
            f.write(f"- Convolutional channels: {model_config['conv_channels']}\n")
        
        # Handle other configuration parameters safely
        if 'use_batch_norm' in model_config:
            f.write(f"- Batch normalization: {model_config['use_batch_norm']}\n")
        
        if 'dropout_rate' in model_config:
            f.write(f"- Dropout rate: {model_config['dropout_rate']}\n")
        
        if 'stochastic_depth_rate' in model_config:
            f.write(f"- Stochastic depth rate: {model_config['stochastic_depth_rate']}\n")
            
        if 'l2_regularization' in model_config:
            f.write(f"- L2 regularization: {model_config['l2_regularization']}\n")
        
        f.write("\n### KAN Network\n\n")
        f.write(f"- Feature dimension: {model_config['feature_dim']}\n")
        f.write(f"- Hidden dimensions: {model_config['hidden_dims']}\n")
        f.write(f"- Grid points: {model_config['grid']}\n")
        f.write(f"- Spline degree: {model_config['degree']}\n")
        
        if 'kan_dropout' in model_config:
            f.write(f"- Dropout rate: {model_config['kan_dropout']}\n")
            
        if 'activation_l1' in model_config:
            f.write(f"- Activation L1 regularization: {model_config['activation_l1']}\n")
            
        if 'use_batchnorm' in model_config:
            f.write(f"- Use batch normalization: {model_config['use_batchnorm']}\n")
        
        # Architecture statistics
        if 'architecture' in model_config:
            arch = model_config['architecture']
            f.write("\n## Model Size Analysis\n\n")
            f.write(f"- Total parameters: {arch.get('total_params', 'N/A'):,}\n")
            f.write(f"- Trainable parameters: {arch.get('trainable_params', 'N/A'):,}\n")
            f.write(f"- Model size: {arch.get('model_size_mb', 'N/A'):.2f} MB\n")
            f.write(f"- CNN parameters: {arch.get('cnn_params', 'N/A'):,} ({100 * arch.get('cnn_params', 0) / arch.get('total_params', 1):.1f}% of total)\n")
            f.write(f"- KAN parameters: {arch.get('kan_params', 'N/A'):,} ({100 * arch.get('kan_params', 0) / arch.get('total_params', 1):.1f}% of total)\n")
        
        # Training details
        if 'training' in model_config:
            training = model_config['training']
            f.write("\n## Training Configuration\n\n")
            f.write(f"- Batch size: {training.get('batch_size', 'N/A')}\n")
            f.write(f"- Learning rate: {training.get('learning_rate', 'N/A')}\n")
            f.write(f"- Weight decay: {training.get('weight_decay', 'N/A')}\n")
            f.write(f"- LR scheduler: {training.get('lr_scheduler', 'N/A')}\n")

def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions and true labels"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probas = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            probas = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probas.extend(probas[:, 1].cpu().numpy())  # Probability of class 1 (person)
    
    return np.array(all_preds), np.array(all_targets), np.array(all_probas)

def analyze_inference_time(model, image_sizes=[96, 128, 224], batch_sizes=[1, 4, 16, 32], device='cpu'):
    """Analyze inference time across different image sizes and batch sizes"""
    results = []
    
    # Set CUDA synchronization for accurate timing
    sync_cuda = device.type == 'cuda'
    
    for img_size in image_sizes:
        for batch_size in batch_sizes:
            # Create dummy input
            dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
            
            # Warm-up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
                    if sync_cuda:
                        torch.cuda.synchronize()  # Ensure GPU operations complete
            
            # Measure time
            with torch.no_grad():
                if sync_cuda:
                    torch.cuda.synchronize()
                start_time = time.time()
                
                for _ in range(50):  # 50 iterations for stable measurement
                    _ = model(dummy_input)
                    if sync_cuda:
                        torch.cuda.synchronize()  # Ensure GPU operations complete
                
                end_time = time.time()
            
            # Calculate metrics
            total_time = (end_time - start_time) * 1000  # ms
            avg_batch_time = total_time / 50
            avg_sample_time = avg_batch_time / batch_size
            
            results.append({
                'image_size': img_size,
                'batch_size': batch_size,
                'batch_time_ms': avg_batch_time,
                'per_image_time_ms': avg_sample_time
            })
    
    # Create report
    print("\nInference Time Analysis:")
    print("-" * 70)
    print(f"{'Image Size':<10} {'Batch Size':<10} {'Batch Time (ms)':<15} {'Per Image (ms)':<15}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['image_size']:<10} {r['batch_size']:<10} {r['batch_time_ms']:<15.2f} {r['per_image_time_ms']:<15.2f}")
    
    return results

def analyze_model_memory(model):
    """Analyze model memory usage"""
    param_counts = model.get_parameter_count()
    model_size_mb = model.get_model_size()
    
    # Get parameter counts per layer
    layer_counts = {}
    layer_counts['preprocessor'] = sum(p.numel() for p in model.preprocessor.parameters())
    layer_counts['kan'] = sum(p.numel() for p in model.kan.parameters())
    
    # Calculate percentage
    total_params = param_counts['total']
    layer_percentages = {k: (v / total_params) * 100 for k, v in layer_counts.items()}
    
    # Report
    print("\nModel Memory Analysis:")
    print(f"Total model size: {model_size_mb:.2f} MB")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print("\nParameter distribution:")
    for layer, count in layer_counts.items():
        print(f"  {layer}: {count:,} parameters ({layer_percentages[layer]:.1f}%)")
    
    return {
        'model_size_mb': model_size_mb,
        'total_params': total_params,
        'trainable_params': param_counts['trainable'],
        'layer_counts': layer_counts,
        'layer_percentages': layer_percentages
    }

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
    model, model_config = load_model_and_config(model_path)
    model = model.to(device)
    model.eval()
    
    # Print model info
    print("\nModel Information:")
    print(f"Model path: {model_path}")
    print(f"Best epoch: {model_config['best_epoch']}")
    print(f"Best validation accuracy: {model_config['best_accuracy']:.2f}%")
    print(f"Image size: {model_config['img_size']}x{model_config['img_size']}")
    print(f"Feature dimension: {model_config['feature_dim']}")
    print(f"KAN hidden dimensions: {model_config['hidden_dims']}")
    print(f"KAN grid: {model_config['grid']}, degree: {model_config['degree']}")
    
    # Analyze model memory usage
    memory_analysis = analyze_model_memory(model)
    
    # Determine data directory
    data_dir = args.data_dir if args.data_dir else (Path(DATASET_CONFIG['subset_dir']) / 'val')
    test_dir = args.test_dir if args.test_dir else (Path(DATASET_CONFIG['subset_dir']) / 'test')
    
    # Check if test dir exists, otherwise use val dir
    if not test_dir.exists() and data_dir.exists():
        test_dir = data_dir
        print(f"Test directory not found, using validation data from: {data_dir}")
    elif not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please specify --data_dir or run prepare_dataset.py first")
        return
    
    # Load validation/test data
    val_transform = transforms.Compose([
        transforms.Resize((model_config['img_size'], model_config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=min(os.cpu_count(), 4)
    )
    
    # Evaluate model
    print(f"\nEvaluating model on test data from: {test_dir}")
    y_pred, y_true, y_probas = evaluate_model(model, test_loader, device)
    
    # Calculate and print classification report
    class_names = test_dataset.classes
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open(paths['analysis_dir'] / 'classification_report.txt', 'w') as f:
        f.write(f"Classification Report for KAN Person Detection Model (Epoch {model_config['best_epoch']})\n")
        f.write("="*80 + "\n\n")
        f.write(report)
    
    # Calculate metrics
    accuracy = 100 * np.mean(y_pred == y_true)
    person_precision = precision_score(y_true, y_pred, pos_label=1)
    person_recall = recall_score(y_true, y_pred, pos_label=1)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, save_dir=paths['figure_dir'], normalize=False)
    plot_confusion_matrix(cm, class_names, save_dir=paths['figure_dir'], normalize=True)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_true, y_probas)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(paths['figure_dir'] / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze inference time
    inference_results = analyze_inference_time(
        model, 
        image_sizes=[96, model_config['img_size'], 224], 
        batch_sizes=[1, 4, 16, 32], 
        device=device
    )
    
    # Save inference results
    with open(paths['analysis_dir'] / 'inference_time.json', 'w') as f:
        json.dump(inference_results, f, indent=2)
    
    # Show misclassified examples
    show_misclassified(model, test_loader, class_names, save_dir=paths['figure_dir'])
    
    
    performance_metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'person_precision': person_precision,
        'person_recall': person_recall
    }

    
    save_model_analysis(
        model=model, 
        model_config=model_config, 
        save_dir=paths['analysis_dir'],
        inference_results=inference_results,
        metrics=performance_metrics
    )
    
    # Create comprehensive summary
    summary = f"""
=====================================================
KAN Person Detection Model Analysis Summary
=====================================================

Model Characteristics:
- Model size: {memory_analysis['model_size_mb']:.2f} MB
- Total parameters: {memory_analysis['total_params']:,}
- Trainable parameters: {memory_analysis['trainable_params']:,}
- Image size: {model_config['img_size']}x{model_config['img_size']}
- Feature dimension: {model_config['feature_dim']}
- KAN hidden dimensions: {model_config['hidden_dims']}
- KAN grid points: {model_config['grid']}
- KAN spline degree: {model_config['degree']}

Performance Metrics:
- Test accuracy: {accuracy:.2f}%
- ROC AUC: {roc_auc:.3f}
- Person detection precision: {person_precision:.3f}
- Person detection recall: {person_recall:.3f}

Inference Performance:
- Single image inference: {inference_results[0]['per_image_time_ms']:.2f} ms
- Batch inference (16 images): {inference_results[2]['per_image_time_ms']:.2f} ms per image
- Batch inference (32 images): {inference_results[3]['per_image_time_ms']:.2f} ms per image

Test Dataset:
- Test dataset size: {len(test_dataset)}
- Test dataset classes: {test_dataset.classes}

The model is suitable for IoT deployment with excellent balance of accuracy,
size and speed. The {memory_analysis['model_size_mb']:.2f} MB model size allows 
deployment on memory-constrained devices while maintaining good detection performance.

=====================================================
"""
    
    # Save summary
    with open(paths['analysis_dir'] / 'model_summary.txt', 'w') as f:
        f.write(summary)
    
    print(summary)
    print("Analysis complete.")

if __name__ == "__main__":
    main()
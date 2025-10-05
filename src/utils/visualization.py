"""
Visualization Utilities for Person Detection Model Analysis.
This module implements functions for visualization of model training,
performance metrics, and prediction results.

Author: Oleksandr Kuznetsov
Date: March 2025
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pathlib import Path

def plot_training_curves(history, save_dir='experiment_data/figures'):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary with training history
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['epochs'], history['train_loss'], label='Train Loss')
    plt.plot(history['epochs'], history['val_loss'], label='Val Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['epochs'], history['train_acc'], label='Train Accuracy')
    plt.plot(history['epochs'], history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate if available
    if 'lr' in history:
        plt.subplot(2, 2, 3)
        plt.plot(history['epochs'], history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, classes, save_dir='experiment_data/figures', normalize=False):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        classes: List of class names
        save_dir: Directory to save plot
        normalize: Whether to normalize the confusion matrix
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        filename = 'confusion_matrix_normalized.png'
    else:
        filename = 'confusion_matrix.png'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
               cmap='Blues', cbar=False,
               xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

def show_misclassified(model, dataloader, classes, save_dir='experiment_data/figures', num_images=8):
    """
    Display and save misclassified examples
    
    Args:
        model: Trained model
        dataloader: Validation dataloader
        classes: List of class names
        save_dir: Directory to save images
        num_images: Number of misclassified images to show
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            # Find misclassified examples
            incorrect_idx = torch.nonzero(preds != targets).squeeze()
            
            # Handle single misclassification case
            if incorrect_idx.dim() == 0 and incorrect_idx.nelement() > 0:
                incorrect_idx = incorrect_idx.unsqueeze(0)
            
            # Process all found misclassifications
            if incorrect_idx.nelement() > 0:
                for idx in incorrect_idx:
                    idx = idx.item()
                    if len(misclassified) < num_images:
                        misclassified.append({
                            'image': inputs[idx].cpu(),
                            'true': targets[idx].item(),
                            'pred': preds[idx].item()
                        })
                    
            if len(misclassified) >= num_images:
                break
    
    if not misclassified:
        print("No misclassified examples found!")
        return
    
    # Display misclassified images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, item in enumerate(misclassified):
        if i >= len(axes):
            break
            
        img = item['image'].permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"True: {classes[item['true']]}\nPred: {classes[item['pred']]}")
        axes[i].axis('off')
    
    # If fewer than 8 images, hide unused axes
    for i in range(len(misclassified), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'misclassified_examples.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_throughput_analysis(throughput_data, save_dir='experiment_data/figures'):
    """
    Plot throughput analysis results
    
    Args:
        throughput_data: List of dictionaries with throughput data
        save_dir: Directory to save plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    batch_sizes = [item['batch_size'] for item in throughput_data]
    images_per_second = [item['images_per_second'] for item in throughput_data]
    ms_per_image = [item['ms_per_image'] for item in throughput_data]
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, images_per_second, 'o-')
    plt.title('Throughput vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Images per Second')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(batch_sizes, ms_per_image, 'o-')
    plt.title('Latency vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Milliseconds per Image')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'throughput_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_kan_activations(model, inputs, layer_idx=0, neuron_idx=0, save_dir='experiment_data/figures'):
    """
    Visualize KAN neuron activations for specific inputs
    
    Args:
        model: KAN model
        inputs: Input tensor or batch
        layer_idx: Index of KAN layer to visualize
        neuron_idx: Index of neuron in the layer to visualize
        save_dir: Directory to save visualization
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Ensure inputs has batch dimension
    if len(inputs.shape) == 3:
        inputs = inputs.unsqueeze(0)
    
    # Get preprocessed features
    with torch.no_grad():
        features = model.preprocessor(inputs)
        
        # Get activations from KAN layers
        # Note: This requires modifications to the KAN class to expose activations
        try:
            activations = model.kan.visualize_activations(features, layer_idx, neuron_idx)
            
            plt.figure(figsize=(10, 6))
            
            # Plot input features
            plt.subplot(2, 1, 1)
            plt.plot(features[0].cpu().numpy())
            plt.title(f"Input Features (First Sample)")
            plt.grid(True)
            
            # Plot activations
            plt.subplot(2, 1, 2)
            plt.plot(activations[0].cpu().numpy())
            plt.title(f"Layer {layer_idx}, Neuron {neuron_idx} Activations")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(save_dir / f'kan_activations_l{layer_idx}_n{neuron_idx}.png', dpi=300)
            plt.close()
            
        except AttributeError:
            print("KAN activations visualization requires modified KAN implementation")
            model.kan.plot()
            plt.savefig(save_dir / 'kan_structure.png', dpi=300)
            plt.close()
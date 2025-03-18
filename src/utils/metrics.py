"""
Metrics Utilities for Person Detection Model Evaluation.
This module implements functions for evaluating model performance,
especially focused on metrics relevant for IoT deployments.

Author: Oleksandr Kuznetsov
Date: March 2025
"""

import numpy as np
import torch
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target):
    """Computes the accuracy"""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target).sum().item()
        return correct / target.size(0)

def measure_inference_time(model, input_size, num_iterations=100, batch_size=1):
    """
    Measure model inference time
    
    Args:
        model: PyTorch model
        input_size: Input image size
        num_iterations: Number of iterations to measure
        batch_size: Batch size for inference
        
    Returns:
        Dictionary with inference time statistics
    """
    model.eval()
    
    # Create random input tensor
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_time = (end_time - start_time) * 1000 / num_iterations  # in milliseconds
    per_sample_time = avg_time / batch_size
    
    return avg_time, per_sample_time

def calculate_metrics(y_true, y_pred, classes):
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: List of class names
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1': f1_score(y_true, y_pred, average='binary'),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape[0] == 2:  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics['confusion_matrix'] = cm
    
    # Class-specific metrics
    class_report = {}
    for i, class_name in enumerate(classes):
        class_report[class_name] = {
            'precision': precision_score(y_true, y_pred, average=None)[i] if i < len(precision_score(y_true, y_pred, average=None)) else 0,
            'recall': recall_score(y_true, y_pred, average=None)[i] if i < len(recall_score(y_true, y_pred, average=None)) else 0,
            'f1': f1_score(y_true, y_pred, average=None)[i] if i < len(f1_score(y_true, y_pred, average=None)) else 0
        }
    
    metrics['class_report'] = class_report
    
    return metrics

def measure_memory_usage(model):
    """Measure model's memory usage in MB"""
    # Calculate parameter memory
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # Calculate buffer memory (e.g., for BatchNorm)
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    size_mb = total_size / (1024 * 1024)
    
    return {
        'param_size_mb': param_size / (1024 * 1024),
        'buffer_size_mb': buffer_size / (1024 * 1024),
        'total_size_mb': size_mb
    }

def measure_batch_throughput(model, input_size, batch_sizes=[1, 2, 4, 8, 16, 32, 64], device='cpu'):
    """
    Measure model throughput for different batch sizes
    
    Args:
        model: PyTorch model
        input_size: Input image size
        batch_sizes: List of batch sizes to test
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Dictionary with throughput metrics
    """
    model.eval()
    results = []
    
    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
        
        # Measure time
        with torch.no_grad():
            start_time = time.time()
            for _ in range(20):  # 20 iterations for stable measurement
                _ = model(dummy_input)
            end_time = time.time()
        
        elapsed_time = end_time - start_time
        images_per_second = (20 * batch_size) / elapsed_time
        ms_per_image = 1000 * elapsed_time / (20 * batch_size)
        
        results.append({
            'batch_size': batch_size,
            'images_per_second': images_per_second,
            'ms_per_image': ms_per_image
        })
    
    return results
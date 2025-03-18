"""
Visual Wake Words Dataset Loader for KAN Image Processing Project.
This module handles loading and preprocessing of the Visual Wake Words dataset
for person detection tasks on resource-constrained devices.

Author: Oleksandr Kuznetsov
Date: March 2025
"""

import os
import torch
from torchvision import transforms
import pyvww
from torch.utils.data import DataLoader, random_split
from pathlib import Path

def get_data_transforms(img_size=128):
    """
    Define data transformations for training and validation sets.
    
    Args:
        img_size: Target image size for model input
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_visual_wakewords(data_dir, ann_dir, batch_size=64, img_size=128, num_workers=4):
    """
    Load Visual Wake Words dataset for person detection.
    
    Args:
        data_dir: Path to MSCOCO images directory
        ann_dir: Path to Visual Wake Words annotations
        batch_size: Batch size for dataloaders
        img_size: Target image size (default: 128x128)
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
        dataset_info: Dict with dataset information
    """
    train_transform, val_transform = get_data_transforms(img_size)
    
    print("Loading Visual Wake Words dataset...")
    print(f"Data directory: {data_dir}")
    print(f"Annotation directory: {ann_dir}")
    
    try:
        # Load training dataset
        train_ann_file = os.path.join(ann_dir, 'instances_train.json')
        train_dataset = pyvww.pytorch.VisualWakeWordsClassification(
            root=data_dir,
            annFile=train_ann_file,
            transform=train_transform
        )
        
        # Load validation dataset
        val_ann_file = os.path.join(ann_dir, 'instances_val.json')
        val_dataset = pyvww.pytorch.VisualWakeWordsClassification(
            root=data_dir,
            annFile=val_ann_file,
            transform=val_transform
        )
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Calculate class distribution
        train_labels = [sample[1] for sample in train_dataset]
        class_dist = {
            'person': sum(train_labels),
            'no_person': len(train_labels) - sum(train_labels)
        }
        
        # Dataset information
        dataset_info = {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'class_distribution': class_dist,
            'img_size': img_size,
            'classes': ['no_person', 'person']
        }
        
        return train_loader, val_loader, dataset_info
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise
"""
KAN-based Image Classification Model for IoT Devices.
This module implements a lightweight Kolmogorov-Arnold Network architecture 
optimized for person detection on resource-constrained devices.

Author: Oleksandr Kuznetsov
Date: March 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN
import numpy as np

class StochasticDepth(nn.Module):
    """
    Implements Stochastic Depth regularization technique.
    During training, randomly drops entire layers with probability p.
    """
    def __init__(self, drop_prob=0.0):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * random_tensor / keep_prob

class ImagePreprocessor(nn.Module):
    """
    Preprocessing module to convert image features to format suitable for KAN.
    Uses lightweight convolutional layers to extract features while keeping parameters low.
    """
    def __init__(self, input_channels=3, output_features=48, img_size=128, 
                 conv_channels=[16, 32, 64], kernel_size=3, pool_kernel_size=2,
                 final_pool_size=4, use_batch_norm=True, dropout_rate=0.1,
                 stochastic_depth_rate=0.0):
        super(ImagePreprocessor, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        num_blocks = len(conv_channels)
        
        # Create convolutional layers dynamically based on config
        for i, out_channels in enumerate(conv_channels):
            # Calculate current stochastic depth probability
            # Linearly increase probability from 0 -> stochastic_depth_rate
            curr_stochastic_depth_prob = stochastic_depth_rate * i / (num_blocks - 1) if num_blocks > 1 else 0
            
            conv_block = []
            # Convolution
            conv_block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                        stride=1, padding=kernel_size//2))
            
            # Normalization
            if use_batch_norm:
                conv_block.append(nn.BatchNorm2d(out_channels))
            
            # Activation
            conv_block.append(nn.ReLU(inplace=True))
            
            # Dropout for regularization
            if dropout_rate > 0:
                conv_block.append(nn.Dropout2d(dropout_rate))
            
            # Pooling
            conv_block.append(nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2))
            
            # Stochastic depth
            if stochastic_depth_rate > 0 and i < num_blocks - 1:  # No stochastic depth for last layer
                conv_block.append(StochasticDepth(curr_stochastic_depth_prob))
            
            self.conv_layers.append(nn.Sequential(*conv_block))
            in_channels = out_channels
        
        # Final pooling
        self.final_pool = nn.AdaptiveAvgPool2d((final_pool_size, final_pool_size))
        
        # Calculate feature size after convolutions and pooling
        self.feature_size = conv_channels[-1] * final_pool_size * final_pool_size
        
        # Linear projection to desired output feature size
        self.projection = nn.Sequential(
            nn.Linear(self.feature_size, output_features),
            nn.BatchNorm1d(output_features) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        
    def forward(self, x):
        # Apply each convolutional block
        for conv_block in self.conv_layers:
            x = conv_block(x)
        
        # Apply final pooling
        x = self.final_pool(x)
        
        # Flatten and project
        x = x.view(-1, self.feature_size)
        x = self.projection(x)
        
        return x


class RegularizedKAN(nn.Module):
    """
    A wrapper around the KAN class that adds regularization techniques
    like dropout and activation regularization.
    """
    def __init__(self, width, grid=4, degree=3, 
                 dropout_rate=0.0, use_batchnorm=False, activation_l1=0.0, seed=42):
        super(RegularizedKAN, self).__init__()
        
        self.kan = KAN(width=width, grid=grid, k=degree, seed=seed)
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.activation_l1 = activation_l1
        self.n_layers = len(width) - 1  # Number of KAN layers
        
        # Create batch norm layers if requested
        if use_batchnorm:
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(width[i+1]) for i in range(len(width)-2)
            ])
        
        # Store activations for regularization if needed
        self.activations = []
        
    def forward(self, x):
        # Reset activations
        self.activations = []
        
        # Forward pass through KAN with regularization
        # Store the input for activation regularization if needed
        if self.activation_l1 > 0 and self.training:
            self.activations.append(x)
        
        # Process through KAN
        x = self.kan(x)
        
        # Add output to activations
        if self.activation_l1 > 0 and self.training:
            self.activations.append(x)
        
        return x
    
    def get_activation_regularization(self):
        """Calculate L1 regularization on activations (sparsity)"""
        if self.activation_l1 <= 0 or not self.activations:
            return 0.0
        
        # Apply L1 regularization to all activations except output
        reg_loss = 0.0
        for act in self.activations[:-1]:  # Exclude output layer
            reg_loss += torch.mean(torch.abs(act)) * self.activation_l1
        
        return reg_loss
    
    def plot_groups(self, save_dir='figures'):
        """Visualize KAN splines and save to directory"""
        try:
            self.kan.plot_groups(save_dir=save_dir)
        except AttributeError:
            print("KAN implementation doesn't support plot_groups. Using available visualization...")
            try:
                self.kan.plot(save_dir=save_dir)
            except AttributeError:
                print("KAN visualization methods not available in this implementation.")


class KANImageClassifier(nn.Module):
    """
    KAN-based image classifier optimized for resource-constrained IoT devices.
    Combines convolutional feature extraction with KAN for efficient processing.
    Includes regularization techniques to prevent overfitting.
    """
    def __init__(self, input_channels=3, img_size=128, num_classes=2, feature_dim=48, 
                 kan_hidden_dims=[24, 12], kan_grid=4, kan_degree=3, conv_channels=[16, 32, 64],
                 use_batch_norm=True, dropout_rate=0.1, activation_l1=0.0, 
                 stochastic_depth_rate=0.0, seed=42):
        super(KANImageClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.img_size = img_size
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Image preprocessing network
        self.preprocessor = ImagePreprocessor(
            input_channels=input_channels,
            output_features=feature_dim,
            img_size=img_size,
            conv_channels=conv_channels,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            stochastic_depth_rate=stochastic_depth_rate
        )
        
        # KAN network for classification with regularization
        kan_width = [feature_dim] + kan_hidden_dims + [num_classes]
        self.kan = RegularizedKAN(
            width=kan_width,
            grid=kan_grid,
            degree=kan_degree,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batch_norm,
            activation_l1=activation_l1,
            seed=seed
        )
        
    def forward(self, x):
        # Preprocess image to extract features
        features = self.preprocessor(x)
        
        # Process features through KAN
        output = self.kan(features)
        
        return output
    
    def get_model_size(self):
        """Calculate model size in MB"""
        # Total model size
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        # CNN preprocessor size
        cnn_size = 0
        for param in self.preprocessor.parameters():
            cnn_size += param.nelement() * param.element_size()
        
        # KAN size
        kan_size = 0
        for param in self.kan.parameters():
            kan_size += param.nelement() * param.element_size()
        
        total_mb = param_size / (1024 * 1024)
        cnn_mb = cnn_size / (1024 * 1024)
        kan_mb = kan_size / (1024 * 1024)
        
        return total_mb
    
    def get_parameter_count(self):
        """Return total and trainable parameter counts"""
        # Total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # CNN preprocessor parameters
        cnn_params = sum(p.numel() for p in self.preprocessor.parameters())
        cnn_trainable = sum(p.numel() for p in self.preprocessor.parameters() if p.requires_grad)
        
        # KAN parameters
        kan_params = sum(p.numel() for p in self.kan.parameters())
        kan_trainable = sum(p.numel() for p in self.kan.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'cnn': cnn_params,
            'kan': kan_params
        }
    
    def get_activation_regularization(self):
        """Get activation regularization from KAN"""
        return self.kan.get_activation_regularization()
    
    def visualize_splines(self, save_dir='figures'):
        """Visualize KAN splines and save to directory"""
        self.kan.plot_groups(save_dir=save_dir)
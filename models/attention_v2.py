#!/usr/bin/env python3
"""
FeatherFace V2 - Coordinate Attention Module

Coordinate Attention implementation for FeatherFace V2.
Replaces CBAM with mobile-optimized spatial-aware attention.

Scientific Foundation: Hou et al. CVPR 2021
"""

import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module for FeatherFace V2
    
    Replaces CBAM with coordinate attention that preserves spatial information
    through 1D factorization and provides mobile-optimized performance.
    
    Args:
        in_channels (int): Number of input channels
        reduction_ratio (int): Channel reduction ratio (default: 32 for mobile optimization)
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 32):
        super(CoordinateAttention, self).__init__()
        
        # Reduced channels for mobile efficiency
        self.reduced_channels = max(in_channels // reduction_ratio, 8)
        
        # Coordinate attention components
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Horizontal pooling [B, C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Vertical pooling [B, C, 1, W]
        
        # Shared transformation for efficiency
        self.conv_transform = nn.Conv2d(in_channels, self.reduced_channels, 
                                       kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_transform = nn.BatchNorm2d(self.reduced_channels)
        self.activation = nn.ReLU(inplace=True)
        
        # Separate directional attention generators
        self.conv_h = nn.Conv2d(self.reduced_channels, in_channels, 
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(self.reduced_channels, in_channels, 
                               kernel_size=1, stride=1, padding=0, bias=False)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for mobile optimization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Coordinate Attention
        
        Process:
        1. Factorize spatial information into H and W directions
        2. Apply shared transformation for efficiency
        3. Generate directional attention maps
        4. Apply coordinate attention to input features
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Attention-enhanced features [B, C, H, W]
        """
        # Step 1: Spatial factorization (key advantage vs CBAM)
        x_h = self.pool_h(x)  # [B, C, H, 1] - Horizontal spatial info
        x_w = self.pool_w(x)  # [B, C, 1, W] - Vertical spatial info
        
        # Step 2: Process each direction separately
        x_h_trans = self.conv_transform(x_h)  # [B, reduced_channels, H, 1]
        x_h_trans = self.bn_transform(x_h_trans)
        x_h_trans = self.activation(x_h_trans)
        
        x_w_trans = self.conv_transform(x_w)  # [B, reduced_channels, 1, W]
        x_w_trans = self.bn_transform(x_w_trans)
        x_w_trans = self.activation(x_w_trans)
        
        # Step 3: Generate directional attention maps
        attention_h = self.conv_h(x_h_trans)  # [B, C, H, 1]
        attention_w = self.conv_w(x_w_trans)  # [B, C, 1, W]
        
        # Apply sigmoid activation for attention weights
        attention_h = torch.sigmoid(attention_h)
        attention_w = torch.sigmoid(attention_w)
        
        # Step 4: Apply coordinate attention
        out = x * attention_h * attention_w
        
        return out
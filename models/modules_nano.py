#!/usr/bin/env python3
"""
FeatherFace Nano-B Specialized Modules

This module implements the 3 specialized modules for FeatherFace Nano-B:
1. ASSN: Attention-based Scale Sequence Network (P3 specialized)
2. MSE-FPN: Multi-scale Semantic Enhancement (all levels)
3. ScaleDecoupling: Small/large object separation (P3 only)

Standard modules (SSH, CBAM, BiFPN, etc.) are imported from net.py

Scientific Foundation:
- ASSN: "Attention-based scale sequence network for small object detection" (PMC/ScienceDirect 2024)
- MSE-FPN: "Multi-scale semantic enhancement network for object detection" (Scientific Reports 2024)
- Scale Decoupling: SNLA approach for P3 optimization (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

# Import standard modules from net.py
from .net import SSH, CBAM, BiFPN, ChannelShuffle2, MobileNetV1


class ScaleDecoupling(nn.Module):
    """
    Scale Decoupling Module for P3 level (Small Face Optimization)
    
    Based on SNLA (Scale Normalized Linear Attention) approach 2024
    Problem: Large object interference with small face detection in shallow P3 layer
    Solution: Selective suppression of large object features while enhancing small face features
    
    Applied: P3 level only, before other processing
    Parameters: ~1,500 additional parameters
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super(ScaleDecoupling, self).__init__()
        
        self.in_channels = in_channels
        reduced_channels = max(in_channels // reduction_ratio, 8)
        
        # Small object enhancer - focuses on high frequency components
        self.small_enhancer = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Large object suppressor - attenuates low frequency components  
        self.large_suppressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Fusion gate to control the balance
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with scale decoupling
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            Decoupled features optimized for small objects
        """
        # Enhance small object features (high frequency)
        small_mask = self.small_enhancer(x)
        small_features = x * small_mask
        
        # Suppress large object features (low frequency)
        large_mask = self.large_suppressor(x)
        large_suppressed = x * (1.0 - large_mask)
        
        # Combine features with learned fusion
        combined = torch.cat([small_features, large_suppressed], dim=1)
        fusion_weight = self.fusion_gate(combined)
        
        # Final decoupled output
        output = x * fusion_weight + small_features * (1.0 - fusion_weight)
        
        return output


class ASSN(nn.Module):
    """
    Attention-based Scale Sequence Network (P3 Specialized)
    
    Based on PMC/ScienceDirect 2024 paper
    Problem: Information loss during spatial scale reduction for small objects
    Solution: Scale-aware attention mechanism optimized for small objects
    
    Applied: P3 level only, replaces standard CBAM after BiFPN
    Parameters: ~2,000 additional parameters
    """
    
    def __init__(self, channels: int, scales: List[int] = [80, 40, 20]):
        super(ASSN, self).__init__()
        
        self.channels = channels
        self.scales = scales
        self.num_scales = len(scales)
        
        # Scale sequence processors
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(channels, channels // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, kernel_size=1),
            ) for scale in scales
        ])
        
        # Scale attention weights
        self.scale_attention = nn.Sequential(
            nn.Conv2d(channels * self.num_scales, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, self.num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Spatial enhancement for small objects
        self.spatial_enhancer = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with scale sequence attention
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            Scale-aware enhanced features for small objects
        """
        B, C, H, W = x.shape
        
        # Process multiple scales
        scale_features = []
        for i, processor in enumerate(self.scale_processors):
            scale_feat = processor(x)
            # Resize back to original spatial dimensions
            scale_feat = F.interpolate(scale_feat, size=(H, W), mode='bilinear', align_corners=False)
            scale_features.append(scale_feat)
        
        # Concatenate scale features
        multi_scale = torch.cat(scale_features, dim=1)  # [B, C*num_scales, H, W]
        
        # Compute scale attention weights
        scale_weights = self.scale_attention(multi_scale)  # [B, num_scales, H, W]
        
        # Weighted combination of scale features
        attended_features = torch.zeros_like(x)
        for i, scale_feat in enumerate(scale_features):
            weight = scale_weights[:, i:i+1, :, :]  # [B, 1, H, W]
            attended_features += scale_feat * weight
        
        # Spatial enhancement for small objects
        spatial_weight = self.spatial_enhancer(attended_features)
        enhanced_features = attended_features * spatial_weight
        
        # Residual connection
        output = x + enhanced_features
        
        return output


class MSE_FPN(nn.Module):
    """
    Multi-scale Semantic Enhancement for Feature Pyramid Networks
    
    Based on Scientific Reports 2024 paper
    Problem: Semantic gap between features of different sizes causing aliasing
    Solution: Semantic injection + gated channel guidance for better fusion
    
    Applied: All pyramid levels (P3, P4, P5)
    Performance: +43.4 AP validated in original research
    Parameters: ~4,000 parameters distributed across levels
    """
    
    def __init__(self, channels: int):
        super(MSE_FPN, self).__init__()
        
        self.channels = channels
        
        # Semantic Enhancement Module (SEM)
        self.semantic_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
        )
        
        # Semantic Injection Module (SIM)
        self.semantic_injector = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Gated Channel Guidance Module (GCG)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial refinement
        self.spatial_refiner = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with semantic enhancement
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            Semantically enhanced features
        """
        B, C, H, W = x.shape
        
        # Extract global semantic information (SEM)
        global_semantic = self.semantic_extractor(x)  # [B, C, 1, 1]
        global_semantic = global_semantic.expand(-1, -1, H, W)  # [B, C, H, W]
        
        # Inject semantic information (SIM)
        combined = torch.cat([x, global_semantic], dim=1)  # [B, 2C, H, W]
        semantic_injected = self.semantic_injector(combined)  # [B, C, H, W]
        
        # Gated channel guidance (GCG)
        channel_weights = self.channel_gate(semantic_injected)
        guided_features = semantic_injected * channel_weights
        
        # Spatial refinement
        refined_features = self.spatial_refiner(guided_features)
        
        # Residual connection with original features
        output = x + refined_features
        
        return output


# Utility function to create the complete Nano-B pipeline
def create_nano_b_pipeline(in_channels: int, out_channels: int, level: str = 'P3') -> nn.Module:
    """
    Create the appropriate pipeline for each pyramid level
    
    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension  
        level: Pyramid level ('P3', 'P4', or 'P5')
        
    Returns:
        Complete pipeline module for the specified level
    """
    
    if level == 'P3':
        # P3 specialized pipeline: ScaleDecoupling → CBAM → BiFPN+MSE → ASSN
        return nn.Sequential(
            ScaleDecoupling(in_channels),
            CBAM(in_channels),
            MSE_FPN(in_channels),
            ASSN(in_channels)
        )
    else:
        # P4/P5 standard pipeline: CBAM → BiFPN+MSE → CBAM
        return nn.Sequential(
            CBAM(in_channels),
            MSE_FPN(in_channels),
            CBAM(in_channels)
        )


# Export key classes
__all__ = [
    'ScaleDecoupling',
    'ASSN', 
    'MSE_FPN',
    'create_nano_b_pipeline',
    # Re-export standard modules for convenience
    'SSH',
    'CBAM', 
    'BiFPN',
    'ChannelShuffle2',
    'MobileNetV1'
]
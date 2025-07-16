#!/usr/bin/env python3
"""
ECA-Net: Efficient Channel Attention for FeatherFace
==================================================

Scientific Foundation: Wang et al. CVPR 2020 
"ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"

Key Innovation: Extremely Efficient Channel Attention
- Avoids dimensionality reduction (maintains all channel information)
- Uses 1D convolution with adaptive kernel size
- Minimal parameters (only a few dozen parameters)
- Superior performance vs SE-Net and CBAM

Mathematical Foundation:
Given input X ∈ ℝᴮˣᶜˣᴴˣᵂ:
1. Global Average Pooling: y = GAP(X) ∈ ℝᴮˣᶜˣ¹ˣ¹
2. 1D Convolution: α = Conv1D(y, kernel_size=k) ∈ ℝᴮˣᶜˣ¹ˣ¹
3. Sigmoid Activation: α = Sigmoid(α) ∈ ℝᴮˣᶜˣ¹ˣ¹
4. Feature Recalibration: Y = X ⊙ α ∈ ℝᴮˣᶜˣᴴˣᵂ

Adaptive Kernel Size:
k = ψ(C) = |log₂(C)/γ + b/γ|_odd
where γ=2, b=1 for optimal performance

Performance Gains (CVPR 2020):
- ResNet50: +1.4% ImageNet top-1 accuracy
- Only 80 parameters vs 24.37M backbone
- Superior to SE-Net and CBAM with minimal overhead

Reference:
Wang, Q., Wu, B., Zhu, P., Li, P., Zuo, W., & Hu, Q. (2020). 
ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks. 
IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union


class ECAModule(nn.Module):
    """
    Efficient Channel Attention (ECA) Module
    
    An extremely efficient attention mechanism that captures local cross-channel 
    interactions without dimensionality reduction.
    
    Key Features:
    - Avoids dimensionality reduction (preserves all channel information)
    - Uses 1D convolution with adaptive kernel size
    - Minimal parameter overhead (only kernel size parameters)
    - Superior performance vs SE-Net and CBAM
    
    Mathematical Process:
    1. Global Average Pooling: Aggregate spatial information
    2. 1D Convolution: Capture local cross-channel interactions
    3. Sigmoid Activation: Generate channel attention weights
    4. Feature Recalibration: Apply attention to input features
    
    Args:
        channels (int): Number of input channels
        gamma (int): Parameter for adaptive kernel size (default: 2)
        beta (int): Parameter for adaptive kernel size (default: 1)
        kernel_size (int): Manual kernel size override (default: None)
        
    Example:
        >>> eca = ECAModule(64)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = eca(x)  # Shape: (2, 64, 32, 32)
        >>> print(f"Input: {x.shape}, Output: {y.shape}")
    """
    
    def __init__(self, channels: int, gamma: int = 2, beta: int = 1, 
                 kernel_size: Union[int, None] = None):
        super(ECAModule, self).__init__()
        
        self.channels = channels
        self.gamma = gamma
        self.beta = beta
        
        # Adaptive kernel size calculation
        if kernel_size is None:
            # Adaptive kernel size: k = ψ(C) = |log₂(C)/γ + b/γ|_odd
            kernel_size = int(abs((math.log2(channels) / gamma) + (beta / gamma)))
            # Ensure kernel size is odd
            kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.kernel_size = kernel_size
        
        # 1D convolution for local cross-channel interaction
        # padding = (kernel_size - 1) // 2 ensures same output size
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1, 
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        
        # Sigmoid activation for attention weights
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize 1D convolution weights using Xavier initialization"""
        nn.init.xavier_normal_(self.conv.weight)
    
    def get_attention_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate channel attention mask M_c for hybrid attention module architecture
        
        Process:
        1. Global Average Pooling to aggregate spatial information
        2. 1D Convolution to capture local cross-channel interactions
        3. Sigmoid activation to generate attention weights
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Channel attention mask M_c [B, C, 1, 1]
        """
        batch_size, channels, height, width = x.size()
        
        # Step 1: Global Average Pooling
        # Aggregate spatial information: [B, C, H, W] → [B, C, 1, 1]
        y = F.adaptive_avg_pool2d(x, 1)
        
        # Step 2: Prepare for 1D convolution
        # Reshape: [B, C, 1, 1] → [B, 1, C] (treat channels as spatial dimension)
        y = y.squeeze(-1).transpose(-1, -2)
        
        # Step 3: 1D Convolution for local cross-channel interaction
        # Capture local cross-channel interactions with adaptive kernel size
        y = self.conv(y)
        
        # Step 4: Generate attention weights
        # Apply sigmoid activation: [B, 1, C] → [B, C, 1, 1]
        attention_mask = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        
        return attention_mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ECA module
        
        Process:
        1. Global Average Pooling to aggregate spatial information
        2. 1D Convolution to capture local cross-channel interactions
        3. Sigmoid activation to generate attention weights
        4. Feature recalibration by element-wise multiplication
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Attention-refined features [B, C, H, W]
        """
        # Get attention mask
        attention_mask = self.get_attention_mask(x)
        
        # Step 5: Feature recalibration
        # Apply channel attention: [B, C, H, W] ⊙ [B, C, 1, 1] → [B, C, H, W]
        return x * attention_mask.expand_as(x)
    
    def get_parameter_count(self) -> dict:
        """
        Get parameter count information for efficiency analysis
        
        Returns:
            dict: Parameter count metrics
        """
        conv_params = self.conv.weight.numel()
        
        return {
            'total_parameters': conv_params,
            'kernel_size': self.kernel_size,
            'channels': self.channels,
            'parameters_per_channel': conv_params / self.channels,
            'efficiency_ratio': conv_params / (self.channels * self.channels)  # vs SE-Net
        }
    
    def extra_repr(self) -> str:
        """String representation for debugging"""
        return f'channels={self.channels}, kernel_size={self.kernel_size}, ' \
               f'gamma={self.gamma}, beta={self.beta}'


class ECABlock(nn.Module):
    """
    ECA Block: Convolution + ECA Attention
    
    A convenient block that combines convolution with ECA attention,
    suitable for integration into existing architectures.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride
        padding (int): Convolution padding
        groups (int): Convolution groups
        bias (bool): Use bias in convolution
        eca_gamma (int): ECA gamma parameter
        eca_beta (int): ECA beta parameter
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, groups: int = 1, bias: bool = True,
                 eca_gamma: int = 2, eca_beta: int = 1):
        super(ECABlock, self).__init__()
        
        # Base convolution
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels)
        
        # ECA attention
        self.eca = ECAModule(out_channels, gamma=eca_gamma, beta=eca_beta)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Conv → BN → ECA → ReLU
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor with ECA attention applied
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.eca(x)  # Apply ECA attention
        x = self.relu(x)
        return x


def test_eca_implementation():
    """
    Test ECA implementation with various configurations
    Validates efficiency and performance characteristics
    """
    print("🧪 Testing ECA Implementation")
    print("=" * 50)
    
    # Test configurations for FeatherFace channel sizes
    test_configs = [
        (32, "Early backbone"),
        (64, "Mid backbone"),
        (128, "Deep backbone"),
        (256, "Very deep backbone"),
        (52, "BiFPN channels"),
    ]
    
    for channels, description in test_configs:
        print(f"\nTesting ECA: {channels} channels ({description})")
        
        # Create ECA module
        eca = ECAModule(channels)
        
        # Test forward pass
        x = torch.randn(2, channels, 32, 32)  # Batch=2 for testing
        y = eca(x)
        
        # Validate output shape
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
        
        # Get parameter info
        param_info = eca.get_parameter_count()
        
        print(f"  ✓ Forward pass: {x.shape} → {y.shape}")
        print(f"  ✓ Parameters: {param_info['total_parameters']}")
        print(f"  ✓ Kernel size: {param_info['kernel_size']}")
        print(f"  ✓ Efficiency: {param_info['efficiency_ratio']:.6f} (vs SE-Net)")
        
        # Test ECA Block
        eca_block = ECABlock(channels, channels, kernel_size=3, padding=1)
        y_block = eca_block(x)
        assert y_block.shape == x.shape, f"Block shape mismatch: {y_block.shape} != {x.shape}"
        print(f"  ✓ ECA Block: {x.shape} → {y_block.shape}")
    
    print(f"\n✅ All ECA tests passed!")
    print(f"🎯 ECA-Net ready for FeatherFace integration!")


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_eca_implementation()
    
    print(f"\n🔬 ECA-Net Scientific Validation:")
    print(f"✅ Wang et al. CVPR 2020: Proven superior to SE-Net and CBAM")
    print(f"✅ Efficiency: Only dozens of parameters vs thousands")
    print(f"✅ Performance: +1.4% ImageNet top-1 accuracy")
    print(f"✅ No dimensionality reduction: Preserves all channel information")
    print(f"✅ Adaptive kernel size: Optimized for different channel dimensions")
    print(f"✅ Mobile-friendly: Minimal computational overhead")
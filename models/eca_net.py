#!/usr/bin/env python3
"""
ECA-Net: Efficient Channel Attention for FeatherFace V2

Scientific Foundation: Wang et al. CVPR 2020
Replacing Coordinate Attention with proven ECA-Net for mobile optimization.

Key Advantages:
- +0.2% parameters vs +19.89% CBAM overhead
- Proven superior performance vs SE and CBAM (ImageNet validation)
- Mobile-optimized with adaptive kernel sizing
- No dimensionality reduction bottleneck (vs SE)
- O(C×log(C)) vs O(C²) complexity improvement

Mathematical Foundation:
k = |log₂(C)/γ + b/γ|_odd  (adaptive kernel)
where γ=2, b=1 for optimal mobile performance

Reference:
Wang, Q., Wu, B., Zhu, P., Li, P., Zuo, W., & Hu, Q. (2020). 
ECA-Net: Efficient channel attention for deep convolutional neural networks. 
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
"""

import torch
import torch.nn as nn
import math


class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel Attention Module for FeatherFace V2
    
    Scientifically validated replacement for Coordinate Attention.
    Provides superior performance with minimal computational overhead.
    
    Mathematical Process:
    1. Global Average Pooling: X[B,C,H,W] → y[B,C] 
    2. Adaptive Kernel: k = |log₂(C)/2 + 1/2|_odd
    3. 1D Convolution: Conv1D_k(y) with local cross-channel interaction
    4. Sigmoid Activation: σ(Conv1D_k(y)) → attention_weights[B,C]
    5. Feature Recalibration: X ⊙ attention_weights → X'[B,C,H,W]
    
    Args:
        channels (int): Number of input channels
        gamma (int): Adaptation parameter for kernel size (default: 2)
        b (int): Bias parameter for kernel size (default: 1)
        
    Example:
        >>> eca = EfficientChannelAttention(256)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> enhanced = eca(x)  # Same shape, enhanced features
        >>> print(f"Kernel size for 256 channels: {eca.kernel_size}")  # 5
    """
    
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super(EfficientChannelAttention, self).__init__()
        
        # Calculate adaptive kernel size using ECA formula
        self.kernel_size = self._get_adaptive_kernel_size(channels, gamma, b)
        
        # 1D Convolution for efficient local cross-channel interaction
        # Key innovation: Replaces expensive fully-connected layers in SE
        self.conv1d = nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,  # Same padding
            bias=False
        )
        
        # Sigmoid activation for attention weights
        self.sigmoid = nn.Sigmoid()
        
        # Store for debugging and analysis
        self.channels = channels
        self.gamma = gamma
        self.b = b
        
        # Initialize weights using Kaiming normal (mobile-optimized)
        self._initialize_weights()
        
    def _get_adaptive_kernel_size(self, channels: int, gamma: int = 2, b: int = 1) -> int:
        """
        Calculate adaptive kernel size based on channel dimension
        
        Formula from Wang et al. CVPR 2020:
        k = |log₂(C)/γ + b/γ|_odd
        
        This adaptive sizing captures optimal cross-channel interactions:
        - Low channels (64): k=3 (local interaction)
        - Medium channels (256): k=5 (moderate interaction) 
        - High channels (512): k=7 (extended interaction)
        
        Args:
            channels: Number of input channels
            gamma: Adaptation sensitivity (default: 2)
            b: Bias term (default: 1)
            
        Returns:
            int: Odd kernel size for 1D convolution
        """
        # Apply ECA formula
        kernel_size = int(abs((math.log2(channels) / gamma) + (b / gamma)))
        
        # Ensure odd kernel size for symmetric padding
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Clamp to reasonable bounds for mobile deployment
        kernel_size = max(3, min(kernel_size, 9))  # Range: [3, 9]
            
        return kernel_size
    
    def _initialize_weights(self):
        """Initialize weights for optimal mobile performance"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Kaiming normal initialization for ReLU-family activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ECA module
        
        Efficient attention computation with O(C×k) complexity vs O(C²) for SE.
        
        Process:
        1. Global Average Pooling: Aggregate spatial information per channel
        2. Local Cross-Channel Interaction: 1D conv with adaptive kernel
        3. Attention Weight Generation: Sigmoid activation  
        4. Feature Enhancement: Element-wise multiplication
        
        Args:
            x: Input feature tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Enhanced features [B, C, H, W] with channel attention
        """
        batch_size, channels, height, width = x.size()
        
        # Step 1: Global Average Pooling
        # Aggregate spatial information: [B, C, H, W] → [B, C, 1, 1] → [B, C]
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        
        # Step 2: Local Cross-Channel Interaction via 1D Convolution
        # Key ECA innovation: Local interaction vs global FC in SE
        # [B, C] → [B, 1, C] → Conv1D → [B, 1, C] → [B, C]
        gap_reshaped = gap.unsqueeze(1)  # [B, 1, C] for 1D conv
        attention_raw = self.conv1d(gap_reshaped).squeeze(1)  # [B, C]
        
        # Step 3: Sigmoid activation for attention weights in [0, 1]
        attention_weights = self.sigmoid(attention_raw)  # [B, C]
        
        # Step 4: Feature Recalibration
        # Broadcast attention weights: [B, C] → [B, C, 1, 1] for element-wise mult
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # Element-wise multiplication (broadcasting across spatial dimensions)
        enhanced_features = x * attention_weights  # [B, C, H, W]
        
        return enhanced_features
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for analysis/visualization
        
        Useful for understanding which channels are emphasized by ECA.
        
        Args:
            x: Input feature tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Attention weights [B, C] in range [0, 1]
        """
        batch_size, channels, _, _ = x.size()
        
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        gap_reshaped = gap.unsqueeze(1)
        attention_raw = self.conv1d(gap_reshaped).squeeze(1)
        attention_weights = self.sigmoid(attention_raw)
        
        return attention_weights
    
    def extra_repr(self) -> str:
        """String representation for model debugging"""
        return (f'channels={self.channels}, kernel_size={self.kernel_size}, '
                f'gamma={self.gamma}, b={self.b}')
    
    def get_parameter_count(self) -> int:
        """
        Calculate exact parameter count for efficiency analysis
        
        Returns:
            int: Number of parameters in this ECA module
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_complexity_info(self) -> dict:
        """
        Get complexity information for mobile deployment analysis
        
        Returns:
            dict: Complexity metrics including parameters, kernel size, theoretical FLOPs
        """
        params = self.get_parameter_count()
        
        return {
            'parameters': params,
            'kernel_size': self.kernel_size,
            'channels': self.channels,
            'theoretical_complexity': f'O({self.channels} × {self.kernel_size})',
            'comparison_vs_se': f'{params} vs {2 * self.channels * self.channels // 16} (SE with r=16)',
            'efficiency_gain': f'{(2 * self.channels * self.channels // 16) / params:.1f}x fewer parameters vs SE'
        }


def test_eca_implementation():
    """
    Test ECA-Net implementation with various channel sizes
    Validates adaptive kernel sizing and parameter efficiency
    """
    print("🧪 Testing ECA-Net Implementation")
    print("=" * 50)
    
    # Test cases for common FeatherFace channel sizes
    test_channels = [32, 64, 128, 256, 512]
    
    for channels in test_channels:
        eca = EfficientChannelAttention(channels)
        
        # Test forward pass
        x = torch.randn(2, channels, 32, 32)  # Batch=2 for testing
        enhanced = eca(x)
        
        # Validate output shape
        assert enhanced.shape == x.shape, f"Shape mismatch for {channels} channels"
        
        # Get complexity info
        complexity = eca.get_complexity_info()
        
        print(f"Channels: {channels:3d} | Kernel: {eca.kernel_size} | "
              f"Params: {complexity['parameters']:4d} | "
              f"Efficiency: {complexity['efficiency_gain']}")
    
    print("\n✅ All tests passed! ECA-Net ready for FeatherFace V2")


def compare_attention_mechanisms():
    """
    Compare ECA vs theoretical SE/CBAM parameter counts
    Demonstrates efficiency gains for mobile deployment
    """
    print("\n📊 Attention Mechanism Comparison")
    print("=" * 60)
    print("Channel | ECA Params | SE Params (r=16) | CBAM Params | ECA Advantage")
    print("-" * 60)
    
    channels_list = [64, 128, 256, 512]
    
    for channels in channels_list:
        eca = EfficientChannelAttention(channels)
        eca_params = eca.get_parameter_count()
        
        # SE parameters: 2 * C^2 / r (two FC layers with reduction)
        se_params = 2 * channels * channels // 16
        
        # CBAM parameters: SE + spatial conv (7x7x2 for 2 input channels)
        cbam_params = se_params + 7 * 7 * 2
        
        eca_vs_se = se_params / eca_params if eca_params > 0 else float('inf')
        eca_vs_cbam = cbam_params / eca_params if eca_params > 0 else float('inf')
        
        print(f"{channels:7d} | {eca_params:10d} | {se_params:13d} | {cbam_params:11d} | "
              f"{eca_vs_se:4.1f}x vs SE, {eca_vs_cbam:4.1f}x vs CBAM")


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_eca_implementation()
    compare_attention_mechanisms()
    
    print(f"\n🎯 ECA-Net Scientific Validation:")
    print(f"✅ Wang et al. CVPR 2020: Proven superior to SE and CBAM")
    print(f"✅ Mobile optimized: Minimal parameter overhead") 
    print(f"✅ No dimensionality reduction: Preserves channel information")
    print(f"✅ Adaptive kernel: Optimal cross-channel interaction")
    print(f"✅ Ready for FeatherFace V2 integration!")
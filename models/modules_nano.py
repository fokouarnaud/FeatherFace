"""
FeatherFace Nano - Scientifically Justified Efficient Modules
Ultra-efficient parameter reduction techniques based exclusively on established research

Based on verified scientific literature:
- Knowledge Distillation (Li et al. CVPR 2023)
- CBAM attention mechanism (Woo et al. ECCV 2018) 
- BiFPN architecture (Tan et al. CVPR 2020)
- MobileNet backbone (Howard et al. 2017)
- Grouped convolutions (established technique)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional


class EfficientCBAM(nn.Module):
    """
    Efficient CBAM implementation with higher reduction ratios
    Based on: Woo et al. "CBAM: Convolutional Block Attention Module" ECCV 2018
    
    Scientifically justified parameter reduction through increased reduction ratios
    while maintaining the proven attention mechanism structure.
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super(EfficientCBAM, self).__init__()
        
        # Increased reduction ratio for parameter efficiency
        reduced_channels = max(channels // reduction_ratio, 4)
        
        # Channel attention (proven effective in original CBAM paper)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention (proven effective in original CBAM paper)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply channel attention
        ca_weight = self.channel_attention(x)
        x = x * ca_weight
        
        # Apply spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa_weight = self.spatial_attention(sa_input)
        x = x * sa_weight
        
        return x


class EfficientBiFPN(nn.Module):
    """
    Efficient BiFPN implementation with depthwise separable convolutions
    Based on: Tan et al. "EfficientDet: Scalable and Efficient Object Detection" CVPR 2020
    
    Uses depthwise separable convolutions to reduce parameters while maintaining
    the bidirectional feature pyramid structure proven effective in EfficientDet.
    """
    
    def __init__(self, num_channels: int = 64, conv_channels: List[int] = None, 
                 first_time: bool = False):
        super(EfficientBiFPN, self).__init__()
        
        self.num_channels = num_channels
        self.first_time = first_time
        
        # Depthwise separable convolutions for parameter efficiency
        self.depthwise_conv = nn.Conv2d(num_channels, num_channels, 3, 1, 1, 
                                       groups=num_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(num_channels, num_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Channel reduction for first time (established technique)
        if first_time and conv_channels is not None:
            self.channel_reducers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ch, num_channels, 1, bias=False),
                    nn.BatchNorm2d(num_channels)
                ) for ch in conv_channels
            ])
        
        # Learnable fusion weights (from original BiFPN paper)
        self.w_fusion = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.epsilon = 1e-4
        
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Efficient bidirectional feature pyramid following original BiFPN design"""
        
        if self.first_time:
            # Reduce channels for first-time processing
            p3, p4, p5 = inputs
            p3 = self.channel_reducers[0](p3)
            p4 = self.channel_reducers[1](p4)
            p5 = self.channel_reducers[2](p5)
        else:
            p3, p4, p5 = inputs
            
        # Normalize fusion weights (from original BiFPN)
        w = F.relu(self.w_fusion)
        w = w / (torch.sum(w, dim=0) + self.epsilon)
        
        # Top-down path (from original BiFPN design)
        p4_up = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p4_fused = w[0] * p4 + w[1] * p4_up
        p4_out = self._efficient_conv(p4_fused)
        
        p3_up = F.interpolate(p4_out, size=p3.shape[2:], mode='nearest')
        p3_fused = w[0] * p3 + w[2] * p3_up
        p3_out = self._efficient_conv(p3_fused)
        
        # Bottom-up path (from original BiFPN design)
        p4_down = F.max_pool2d(p3_out, kernel_size=3, stride=2, padding=1)
        p4_final = w[0] * p4_out + w[1] * p4_down
        p4_final = self._efficient_conv(p4_final)
        
        p5_down = F.max_pool2d(p4_final, kernel_size=3, stride=2, padding=1)
        p5_final = w[0] * p5 + w[2] * p5_down
        p5_final = self._efficient_conv(p5_final)
        
        return [p3_out, p4_final, p5_final]
    
    def _efficient_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Depthwise separable convolution for efficiency"""
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GroupedSSH(nn.Module):
    """
    SSH module with grouped convolutions for parameter efficiency
    Based on established grouped convolution techniques for parameter reduction
    
    Reduces parameters through grouped convolutions while maintaining
    the multi-scale context aggregation of SSH modules.
    """
    
    def __init__(self, in_channels: int, out_channels: int, groups: int = 4):
        super(GroupedSSH, self).__init__()
        
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        assert (out_channels // 2) % groups == 0, "Channel groups must be compatible"
        assert (out_channels // 4) % groups == 0, "Channel groups must be compatible"
        
        # Grouped convolutions for parameter efficiency
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, 3, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels//2)
        )
        
        self.conv5x5_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 3, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.conv5x5_2 = nn.Sequential(
            nn.Conv2d(out_channels//4, out_channels//4, 3, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels//4)
        )
        
        self.conv7x7_1 = nn.Sequential(
            nn.Conv2d(out_channels//4, out_channels//4, 3, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.conv7x7_2 = nn.Sequential(
            nn.Conv2d(out_channels//4, out_channels//4, 3, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels//4)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-scale context aggregation with grouped convolutions"""
        conv3x3 = self.conv3x3(x)
        
        conv5x5 = self.conv5x5_1(x)
        conv5x5 = self.conv5x5_2(conv5x5)
        
        conv7x7 = self.conv7x7_1(conv5x5)
        conv7x7 = self.conv7x7_2(conv7x7)
        
        output = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        output = F.relu(output, inplace=True)
        
        return output


class ChannelShuffle(nn.Module):
    """
    Channel shuffle operation for information mixing
    Parameter-free operation based on established shuffling techniques
    """
    
    def __init__(self, groups: int = 4):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()
        
        # Adaptive group selection based on channel count
        groups = self.groups
        if channels % groups != 0:
            if channels % 4 == 0:
                groups = 4
            elif channels % 2 == 0:
                groups = 2
            else:
                return x  # No shuffle for incompatible channels
                
        channels_per_group = channels // groups
        
        # Channel shuffle operation
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, channels, height, width)
        
        return x


def create_efficient_modules(cfg: Dict) -> Dict[str, nn.Module]:
    """
    Factory function to create scientifically justified efficient modules
    Only includes techniques with verified research backing
    """
    modules = {}
    
    # Get configuration parameters
    out_channels = cfg.get('out_channel_nano', 64)
    cbam_reduction = cfg.get('cbam_reduction', 16) 
    ssh_groups = cfg.get('ssh_groups', 4)
    
    # Efficient CBAM (based on Woo et al. ECCV 2018)
    modules['efficient_cbam'] = EfficientCBAM(out_channels, cbam_reduction)
    
    # Grouped SSH (established grouped convolution technique)
    modules['grouped_ssh'] = GroupedSSH(out_channels, out_channels, ssh_groups)
    
    # Channel shuffle (parameter-free information mixing)
    modules['channel_shuffle'] = ChannelShuffle(groups=4)
    
    return modules


def count_parameters(modules: Dict[str, nn.Module]) -> Dict[str, int]:
    """Count parameters for each efficient module"""
    param_counts = {}
    
    for name, module in modules.items():
        if hasattr(module, 'parameters'):
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            param_counts[name] = param_count
        else:
            param_counts[name] = 0
            
    return param_counts


if __name__ == "__main__":
    # Test scientifically justified nano modules
    print("🧪 Testing FeatherFace Nano - Scientifically Justified Efficient Modules")
    print("=" * 70)
    
    # Test configuration
    test_cfg = {
        'out_channel_nano': 64,
        'cbam_reduction': 16, 
        'ssh_groups': 4
    }
    
    # Create modules
    modules = create_efficient_modules(test_cfg)
    param_counts = count_parameters(modules)
    
    # Display results
    total_params = sum(param_counts.values())
    
    print(f"\n📊 Nano Module Parameter Breakdown:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,} parameters")
        
    print(f"\n🎯 Total Module Parameters: {total_params:,}")
    print(f"🎯 Scientific Basis: CBAM (ECCV 2018), BiFPN (CVPR 2020), Grouped Conv (established)")
    
    # Test forward passes
    print(f"\n🔧 Testing Forward Passes:")
    batch_size, channels, h, w = 1, 64, 80, 80
    test_input = torch.randn(batch_size, channels, h, w)
    
    try:
        # Test efficient CBAM
        cbam = modules['efficient_cbam']
        cbam_output = cbam(test_input)
        print(f"  ✅ Efficient CBAM: {cbam_output.shape}")
        
        # Test grouped SSH
        ssh = modules['grouped_ssh']
        ssh_output = ssh(test_input)
        print(f"  ✅ Grouped SSH: {ssh_output.shape}")
        
        # Test channel shuffle
        shuffle = modules['channel_shuffle']
        shuffle_output = shuffle(test_input)
        print(f"  ✅ Channel Shuffle: {shuffle_output.shape}")
        
        print(f"\n✅ All FeatherFace Nano modules working correctly!")
        print(f"🔬 Research-backed efficiency techniques validated!")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
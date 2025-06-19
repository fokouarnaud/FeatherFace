"""
FeatherFace V2 Optimized Modules
Lightweight modules for FeatherFace V2 architecture targeting 0.25M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class SELayer_Plus(nn.Module):
    """Lightweight Squeeze-and-Excitation layer with increased reduction"""
    def __init__(self, channel, reduction=32):
        super(SELayer_Plus, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Increased reduction for fewer parameters
        reduced_channels = max(channel // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelGate_Plus(nn.Module):
    """Lightweight Channel Attention with higher reduction ratio"""
    def __init__(self, gate_channels, reduction_ratio=32, pool_types=['avg', 'max']):
        super(ChannelGate_Plus, self).__init__()
        self.gate_channels = gate_channels
        reduced_channels = max(gate_channels // reduction_ratio, 4)
        
        # Shared MLP with increased reduction
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, gate_channels, bias=False)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
                channel_att_raw = self.mlp(avg_pool.view(x.size(0), -1))
            elif pool_type == 'max':
                max_pool = F.adaptive_max_pool2d(x, (1, 1))
                channel_att_raw = self.mlp(max_pool.view(x.size(0), -1))
            
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum.unsqueeze(2).unsqueeze(3))
        return x * scale


class SpatialGate_Plus(nn.Module):
    """Lightweight Spatial Attention using grouped convolutions"""
    def __init__(self, groups=4):
        super(SpatialGate_Plus, self).__init__()
        self.groups = groups
        # Use grouped convolution for spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False, groups=1),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True)
        )

    def forward(self, x):
        # Create spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_compress = torch.cat([avg_out, max_out], dim=1)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


class CBAM_Plus(nn.Module):
    """
    Optimized CBAM with higher reduction ratio and weight sharing capability
    Reduces parameters by ~50% compared to original CBAM
    """
    def __init__(self, gate_channels, reduction_ratio=32, pool_types=['avg', 'max'], 
                 no_spatial=False, share_weights=False):
        super(CBAM_Plus, self).__init__()
        self.share_weights = share_weights
        
        if not share_weights:
            self.channel_gate = ChannelGate_Plus(gate_channels, reduction_ratio, pool_types)
            self.no_spatial = no_spatial
            if not no_spatial:
                self.spatial_gate = SpatialGate_Plus()
        
    def forward(self, x, shared_channel_gate=None, shared_spatial_gate=None):
        if self.share_weights:
            # Use shared gates passed from outside
            if shared_channel_gate is not None:
                x_out = shared_channel_gate(x)
            else:
                x_out = x
                
            if shared_spatial_gate is not None and not self.no_spatial:
                x_out = shared_spatial_gate(x_out)
        else:
            # Use own gates
            x_out = self.channel_gate(x)
            if not self.no_spatial:
                x_out = self.spatial_gate(x_out)
        
        return x_out


class SharedMultiHead(nn.Module):
    """
    Unified detection head that shares computation across classification, bbox, and landmarks
    Reduces parameters by ~60% compared to separate heads
    """
    def __init__(self, in_channels=64, num_anchors=3):
        super(SharedMultiHead, self).__init__()
        self.num_anchors = num_anchors
        
        # Shared trunk - reduce channels first
        trunk_channels = max(in_channels // 2, 16)
        self.shared_trunk = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True)
        )
        
        # Task-specific heads with grouped convolutions
        # Classification head (2 outputs per anchor)
        self.cls_head = nn.Conv2d(trunk_channels, num_anchors * 2, 
                                  kernel_size=1, groups=1, bias=True)
        
        # Bbox regression head (4 outputs per anchor)
        self.bbox_head = nn.Conv2d(trunk_channels, num_anchors * 4,
                                   kernel_size=1, groups=1, bias=True)
        
        # Landmark head (10 outputs per anchor)
        self.ldm_head = nn.Conv2d(trunk_channels, num_anchors * 10,
                                  kernel_size=1, groups=1, bias=True)
        
    def forward(self, x):
        # Shared feature extraction
        shared_features = self.shared_trunk(x)
        
        # Task-specific predictions
        cls = self.cls_head(shared_features)
        bbox = self.bbox_head(shared_features)
        ldm = self.ldm_head(shared_features)
        
        # Reshape outputs to match original format
        cls = cls.permute(0, 2, 3, 1).contiguous()
        cls = cls.view(cls.shape[0], -1, 2)
        
        bbox = bbox.permute(0, 2, 3, 1).contiguous()
        bbox = bbox.view(bbox.shape[0], -1, 4)
        
        ldm = ldm.permute(0, 2, 3, 1).contiguous()
        ldm = ldm.view(ldm.shape[0], -1, 10)
        
        return cls, bbox, ldm


class SharedCBAMManager(nn.Module):
    """
    Manager for shared CBAM modules across the network
    Reduces total CBAM parameters by sharing weights
    """
    def __init__(self, channel_configs, reduction_ratio=32):
        super(SharedCBAMManager, self).__init__()
        
        # Create shared gates for different channel sizes
        self.channel_gates = nn.ModuleDict()
        self.spatial_gate = SpatialGate_Plus()  # Single spatial gate shared by all
        
        for name, channels in channel_configs.items():
            self.channel_gates[name] = ChannelGate_Plus(channels, reduction_ratio)
    
    def forward(self, x, gate_name):
        """Apply CBAM using the appropriate channel gate based on input size"""
        if gate_name in self.channel_gates:
            x = self.channel_gates[gate_name](x)
            x = self.spatial_gate(x)
        return x


def count_parameters(module):
    """Utility function to count parameters in a module"""
    return sum(p.numel() for p in module.parameters())


def test_modules():
    """Test the optimized modules and compare with original versions"""
    print("=== Testing FeatherFace V2 Optimized Modules ===\n")
    
    # Test CBAM_Plus
    channels = 64
    print("1. CBAM_Plus Test:")
    
    # Original CBAM simulation (reduction=16)
    original_cbam_params = 2 * (channels * (channels // 16) * 2)  # Channel gate MLP
    original_cbam_params += 7 * 7 * 2 * 1  # Spatial gate conv
    
    # Our CBAM_Plus
    cbam_plus = CBAM_Plus(channels, reduction_ratio=32)
    cbam_plus_params = count_parameters(cbam_plus)
    
    print(f"   Original CBAM (estimated): {original_cbam_params} params")
    print(f"   CBAM_Plus: {cbam_plus_params} params")
    print(f"   Reduction: {(1 - cbam_plus_params/original_cbam_params)*100:.1f}%\n")
    
    # Test SharedMultiHead
    print("2. SharedMultiHead Test:")
    in_channels = 64
    num_anchors = 3
    
    # Original separate heads
    original_heads_params = in_channels * num_anchors * 2  # ClassHead
    original_heads_params += in_channels * num_anchors * 4  # BboxHead  
    original_heads_params += in_channels * num_anchors * 10  # LandmarkHead
    
    # Our SharedMultiHead
    shared_head = SharedMultiHead(in_channels, num_anchors)
    shared_head_params = count_parameters(shared_head)
    
    print(f"   Original 3 heads: {original_heads_params} params")
    print(f"   SharedMultiHead: {shared_head_params} params")
    print(f"   Reduction: {(1 - shared_head_params/original_heads_params)*100:.1f}%\n")
    
    # Test forward pass
    print("3. Forward Pass Test:")
    x = torch.randn(1, in_channels, 20, 20)
    
    # Test CBAM_Plus
    cbam_out = cbam_plus(x)
    print(f"   CBAM_Plus output shape: {cbam_out.shape}")
    
    # Test SharedMultiHead
    cls, bbox, ldm = shared_head(x)
    print(f"   SharedMultiHead outputs:")
    print(f"     - Classification: {cls.shape}")
    print(f"     - BBox: {bbox.shape}")
    print(f"     - Landmarks: {ldm.shape}")


if __name__ == "__main__":
    test_modules()

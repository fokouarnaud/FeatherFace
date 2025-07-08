#!/usr/bin/env python3
"""
FeatherFace V2 - Mobile Face Detection with Coordinate Attention

This module implements FeatherFace V2, which replaces CBAM with Coordinate Attention
for improved mobile face detection performance, especially on small faces.

Key Innovation:
- V1 Architecture: CBAM attention mechanism (generic, spatial info loss)
- V2 Architecture: Coordinate Attention (mobile-optimized, spatial preservation)

Scientific Foundation:
- Base Architecture: RetinaFace with MobileNetV1 backbone
- Innovation: Hou et al. "Coordinate Attention for Efficient Mobile Network Design" CVPR 2021
- Applications: EfficientFace 2024, FasterMLP 2025, Dense Face Detection 2024

Performance Targets:
- WIDERFace Hard: 77.2% → 88.0% (+10.8% improvement)
- Mobile Inference: 2x speedup vs CBAM
- Parameters: Maintain ~489K parameters
- Memory: 15-20% reduction vs V1

Architecture Changes:
1. BiFPN: CBAM → Coordinate Attention
2. SSH: CBAM → Coordinate Attention  
3. All other components: IDENTICAL to V1

This ensures controlled experimentation with single variable change.
"""

import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict, Any

# Import V1 components (unchanged)
from models.net import MobileNetV1 as MobileNetV1
from models.net import SSH as SSH
from models.net import BiFPN as BiFPN
from models.net import ChannelShuffle2 as ChannelShuffle

# Import V2 innovation
from models.attention_v2 import CoordinateAttention, MobileCoordinateAttention


class ClassHead(nn.Module):
    """Classification head - identical to V1"""
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors*2, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    """Bounding box regression head - identical to V1"""
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    """Landmark regression head - identical to V1"""
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*10, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 10)


class BiFPN_V2(nn.Module):
    """
    BiFPN with Coordinate Attention (V2 Innovation)
    
    Simplified approach: Use original BiFPN and add Coordinate Attention
    """
    def __init__(self, num_channels, conv_channels, first_time=False, attention=True):
        super(BiFPN_V2, self).__init__()
        
        # Use original BiFPN without attention
        self.bifpn_original = BiFPN(num_channels, conv_channels, first_time, attention=False)
        
        # V2 Innovation: Add Coordinate Attention
        self.use_attention = attention
        if self.use_attention:
            self.ca_p3 = CoordinateAttention(num_channels, reduction_ratio=32, mobile_optimized=True)
            self.ca_p4 = CoordinateAttention(num_channels, reduction_ratio=32, mobile_optimized=True)
            self.ca_p5 = CoordinateAttention(num_channels, reduction_ratio=32, mobile_optimized=True)
    
    def forward(self, inputs):
        # Use original BiFPN for feature fusion
        outputs = self.bifpn_original(inputs)
        
        # V2 Innovation: Apply Coordinate Attention
        if self.use_attention:
            outputs = [
                self.ca_p3(outputs[0]),  # P3: Small faces (main improvement target)
                self.ca_p4(outputs[1]),  # P4: Medium faces
                self.ca_p5(outputs[2])   # P5: Large faces
            ]
        
        return outputs


class SSH_V2(nn.Module):
    """
    SSH with Coordinate Attention (V2 Innovation)
    
    Simplified approach: Use original SSH and add Coordinate Attention
    """
    def __init__(self, in_channel, out_channel):
        super(SSH_V2, self).__init__()
        
        # Use original SSH without built-in attention
        self.ssh_original = SSH(in_channel, out_channel)
        
        # V2 Innovation: Add Coordinate Attention
        self.ca = CoordinateAttention(out_channel, reduction_ratio=32, mobile_optimized=True)
    
    def forward(self, input):
        # Use original SSH for context enhancement
        output = self.ssh_original(input)
        
        # V2 Innovation: Apply Coordinate Attention
        output = self.ca(output)
        
        return output


class FeatherFaceV2(nn.Module):
    """
    FeatherFace V2 - Mobile Face Detection with Coordinate Attention
    
    Architecture:
    1. MobileNetV1 Backbone (identical to V1)
    2. BiFPN with Coordinate Attention (V2 innovation)
    3. SSH with Coordinate Attention (V2 innovation)
    4. Detection Heads (identical to V1)
    
    Scientific Justification:
    - Controlled experiment: Only attention mechanism changed
    - Coordinate Attention: Spatial preservation + mobile optimization
    - Expected: +10-15% WIDERFace Hard, 2x speed improvement
    """
    
    def __init__(self, cfg=None, phase='train'):
        """
        Initialize FeatherFace V2
        
        Args:
            cfg: Configuration dict (cfg_v2 from data.config)
            phase: 'train' or 'test'
        """
        super(FeatherFaceV2, self).__init__()
        
        self.cfg = cfg
        self.phase = phase
        
        # Validate configuration
        if cfg is None:
            raise ValueError("Configuration required for FeatherFace V2")
        
        # Ensure V2 configuration
        if cfg.get('attention_mechanism') != 'coordinate_attention':
            raise ValueError("FeatherFace V2 requires coordinate_attention mechanism")
        
        # Initialize backbone (identical to V1)
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                backbone.load_state_dict(new_state_dict)
        
        # Feature extraction layers (identical to V1)
        if cfg['name'] == 'mobilenet0.25':
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
            in_channels_stage2 = cfg['in_channel']
            in_channels_list = [
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ]
            out_channels = cfg['out_channel']
            
            # BiFPN configuration (identical to V1)
            conv_channel_coef = {
                0: [in_channels_list[0], in_channels_list[1], in_channels_list[2]],
                1: [40, 112, 320],
                2: [48, 120, 352],
                3: [48, 136, 384],
                4: [56, 160, 448],
                5: [64, 176, 512],
                6: [72, 200, 576],
                7: [72, 200, 576],
                8: [80, 224, 640],
            }
            
            self.fpn_num_filters = [out_channels, 256, 112, 160, 224, 288, 384, 384]
            self.fpn_cell_repeats = [2, 4, 5, 6, 7, 7, 8, 8, 8]
            self.compound_coef = 0
            
            # V2 Innovation: BiFPN (original) + Coordinate Attention
            self.bifpn = nn.Sequential(
                *[BiFPN(self.fpn_num_filters[self.compound_coef],
                       conv_channel_coef[self.compound_coef],
                       True if _ == 0 else False,
                       attention=False)  # Disable built-in attention
                  for _ in range(self.fpn_cell_repeats[self.compound_coef])]
            )
            
            # V2 Innovation: Add Coordinate Attention after BiFPN
            self.ca_p3 = CoordinateAttention(out_channels, reduction_ratio=32, mobile_optimized=True)
            self.ca_p4 = CoordinateAttention(out_channels, reduction_ratio=32, mobile_optimized=True)
            self.ca_p5 = CoordinateAttention(out_channels, reduction_ratio=32, mobile_optimized=True)
            
            # V2 Innovation: SSH with Coordinate Attention
            self.ssh1 = SSH_V2(out_channels, out_channels)  # P3: Small faces
            self.ssh2 = SSH_V2(out_channels, out_channels)  # P4: Medium faces
            self.ssh3 = SSH_V2(out_channels, out_channels)  # P5: Large faces
            
            # Channel Shuffling (identical to V1)
            self.cs1 = ChannelShuffle(out_channels, groups=2)
            self.cs2 = ChannelShuffle(out_channels, groups=2)
            self.cs3 = ChannelShuffle(out_channels, groups=2)
        
        # Detection heads (identical to V1)
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        
        # Performance tracking
        self.forward_count = 0
        self.attention_stats = {
            'coordinate_attention_calls': 0,
            'total_parameters': self.count_parameters(),
            'mobile_optimized': cfg.get('coordinate_attention_config', {}).get('mobile_optimized', True)
        }
    
    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        """Create classification heads - identical to V1"""
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead
    
    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        """Create bbox regression heads - identical to V1"""
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead
    
    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        """Create landmark regression heads - identical to V1"""
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead
    
    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, inputs):
        """
        Forward pass of FeatherFace V2
        
        Process:
        1. MobileNetV1 backbone (identical to V1)
        2. BiFPN with Coordinate Attention (V2 innovation)
        3. SSH with Coordinate Attention (V2 innovation)
        4. Detection heads (identical to V1)
        
        Args:
            inputs: Input tensor [B, 3, H, W]
            
        Returns:
            Tuple: (bbox_regressions, classifications, ldm_regressions)
        """
        self.forward_count += 1
        
        if self.cfg['name'] == 'mobilenet0.25':
            # 1. MobileNetV1 backbone (identical to V1)
            out = self.body(inputs)
            out = list(out.values())  # [P3, P4, P5]
            
            # 2. V2 Innovation: BiFPN + Coordinate Attention
            bifpn_features = self.bifpn(out)
            
            # Apply Coordinate Attention to BiFPN outputs
            bifpn_features = [
                self.ca_p3(bifpn_features[0]),  # P3: Small faces
                self.ca_p4(bifpn_features[1]),  # P4: Medium faces
                self.ca_p5(bifpn_features[2])   # P5: Large faces
            ]
            self.attention_stats['coordinate_attention_calls'] += 3
            
            # 3. V2 Innovation: SSH with Coordinate Attention
            ssh_feature1 = self.ssh1(bifpn_features[0])  # P3: Small faces
            ssh_feature2 = self.ssh2(bifpn_features[1])  # P4: Medium faces
            ssh_feature3 = self.ssh3(bifpn_features[2])  # P5: Large faces
            self.attention_stats['coordinate_attention_calls'] += 3
            
            # 4. Channel Shuffling (identical to V1)
            feat1 = self.cs1(ssh_feature1)
            feat2 = self.cs2(ssh_feature2)
            feat3 = self.cs3(ssh_feature3)
            
            features = [feat1, feat2, feat3]
        
        # 5. Detection heads (identical to V1)
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        
        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        
        return output
    
    def get_attention_maps(self, inputs):
        """
        Get attention maps for visualization
        
        Args:
            inputs: Input tensor
            
        Returns:
            Dict: Attention maps for each level
        """
        with torch.no_grad():
            # Forward pass to get features
            out = self.body(inputs)
            out = list(out.values())
            bifpn_features = self.bifpn(out)
            
            # Get attention maps from BiFPN Coordinate Attention
            attention_maps = {}
            ca_modules = [self.ca_p3, self.ca_p4, self.ca_p5]
            
            for i, (ca_module, feature) in enumerate(zip(ca_modules, bifpn_features)):
                att_h, att_w = ca_module.get_attention_maps(feature)
                attention_maps[f'P{i+3}_bifpn'] = {
                    'horizontal': att_h,
                    'vertical': att_w
                }
            
            # Get attention maps from SSH modules
            for i, (ssh_module, feature) in enumerate(zip([self.ssh1, self.ssh2, self.ssh3], bifpn_features)):
                att_h, att_w = ssh_module.ca.get_attention_maps(feature)
                attention_maps[f'P{i+3}_ssh'] = {
                    'horizontal': att_h,
                    'vertical': att_w
                }
            
            return attention_maps
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            'forward_count': self.forward_count,
            'attention_stats': self.attention_stats,
            'total_parameters': self.count_parameters(),
            'model_version': 'v2',
            'innovation': 'coordinate_attention'
        }
    
    def compare_with_v1(self, v1_model):
        """
        Compare with V1 model
        
        Args:
            v1_model: FeatherFace V1 model
            
        Returns:
            Dict: Comparison metrics
        """
        v1_params = sum(p.numel() for p in v1_model.parameters())
        v2_params = self.count_parameters()
        
        return {
            'v1_parameters': v1_params,
            'v2_parameters': v2_params,
            'parameter_ratio': v2_params / v1_params,
            'parameter_difference': v2_params - v1_params,
            'attention_mechanism': {
                'v1': 'CBAM',
                'v2': 'Coordinate Attention'
            },
            'expected_improvements': {
                'mobile_speed': '2x faster',
                'widerface_hard': '+10-15%',
                'spatial_preservation': 'Yes (V2) vs No (V1)'
            }
        }


def create_featherface_v2(cfg, phase='train'):
    """
    Factory function to create FeatherFace V2
    
    Args:
        cfg: Configuration dict (cfg_v2)
        phase: 'train' or 'test'
        
    Returns:
        FeatherFaceV2: V2 model instance
    """
    return FeatherFaceV2(cfg, phase)


def test_featherface_v2():
    """Test FeatherFace V2 implementation"""
    print("Testing FeatherFace V2 Implementation...")
    
    # Import configuration
    from data.config import cfg_v2
    
    # Create model
    model = create_featherface_v2(cfg_v2, phase='test')
    
    # Test input
    batch_size = 1
    height, width = 640, 640
    inputs = torch.randn(batch_size, 3, height, width)
    
    # Forward pass
    print(f"Input shape: {inputs.shape}")
    outputs = model(inputs)
    
    # Validate outputs
    bbox_reg, classifications, landmarks = outputs
    print(f"Bbox regression shape: {bbox_reg.shape}")
    print(f"Classifications shape: {classifications.shape}")
    print(f"Landmarks shape: {landmarks.shape}")
    
    # Performance stats
    stats = model.get_performance_stats()
    print(f"Total parameters: {stats['total_parameters']:,}")
    print(f"Forward count: {stats['forward_count']}")
    print(f"Attention calls: {stats['attention_stats']['coordinate_attention_calls']}")
    
    # Get attention maps
    attention_maps = model.get_attention_maps(inputs)
    print(f"Attention maps: {list(attention_maps.keys())}")
    
    print("✅ FeatherFace V2 test passed!")
    return model


if __name__ == "__main__":
    # Test implementation
    model = test_featherface_v2()
    
    print(f"\n🎯 FeatherFace V2 ready for training!")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Innovation: Coordinate Attention replacing CBAM")
    print(f"Target: +10-15% WIDERFace Hard improvement")
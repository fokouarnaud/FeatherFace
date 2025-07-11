#!/usr/bin/env python3
"""
FeatherFace V2 - Innovation avec Coordinate Attention

Ce modèle implémente FeatherFace V2 qui remplace les 6 modules CBAM 
du V1 Original par 3 modules Coordinate Attention optimisés pour mobile.

Innovation V2 :
1. Remplace 6 CBAM par 3 Coordinate Attention
2. Optimisation mobile (2x plus rapide)
3. Préservation spatiale (vs global pooling)
4. 493K paramètres (-1.8% vs V1 Original 502K)

Scientific Foundation: Hou et al. CVPR 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import SSH as SSH
from models.net import BiFPN as BiFPN
from models.net import ChannelShuffle2 as ChannelShuffle
from models.attention_v2 import CoordinateAttention


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors*2, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*10, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 10)


class FeatherFaceV2(nn.Module):
    """
    FeatherFace V2 - Innovation avec Coordinate Attention
    
    Cette implémentation remplace les 6 modules CBAM du V1 Original 
    par 3 modules Coordinate Attention optimisés pour mobile.
    
    Architecture :
    - MobileNetV1 0.25x backbone (identique V1)
    - BiFPN feature pyramid (identique V1)
    - 3 modules Coordinate Attention (INNOVATION vs 6 CBAM)
    - SSH context enhancement (identique V1)
    - Channel Shuffle (identique V1)
    - Detection heads (identique V1)
    
    Innovation : 3 Coordinate Attention = 493K params (-1.8% vs V1 Original 502K)
    """
    
    def __init__(self, cfg=None, phase='train'):
        """
        Initialize FeatherFace V2
        
        Args:
            cfg: Configuration dict (cfg_v2)
            phase: 'train' or 'test'
        """
        super(FeatherFaceV2, self).__init__()
        
        self.cfg = cfg
        self.phase = phase
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
                # load params
                backbone.load_state_dict(new_state_dict)
        
        if cfg['name'] == 'mobilenet0.25':
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
            in_channels_stage2 = cfg['in_channel']
            in_channels_list = [
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ]
            out_channels = cfg['out_channel']
            
            print(f"🚀 FeatherFace V2 - Initializing with Coordinate Attention...")
            print(f"   Backbone channels: {in_channels_list}")
            print(f"   Output channels: {out_channels}")
            
            # BiFPN configuration (identique V1)
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
            
            # BiFPN (identique V1, pas d'attention intégrée)
            self.bifpn = nn.Sequential(
                *[BiFPN(self.fpn_num_filters[self.compound_coef],
                        conv_channel_coef[self.compound_coef],
                        True if _ == 0 else False,
                        attention=False  # Pas d'attention BiFPN
                        )
                  for _ in range(self.fpn_cell_repeats[self.compound_coef])])
            
            # SSH context modules (identique V1)
            self.ssh1 = SSH(out_channels, out_channels)  # P3
            self.ssh2 = SSH(out_channels, out_channels)  # P4
            self.ssh3 = SSH(out_channels, out_channels)  # P5
            
            # INNOVATION V2: Coordinate Attention (remplace 6 CBAM)
            # Placement optimal post-SSH pour maximiser l'effet spatial
            self.ca_p3 = CoordinateAttention(out_channels, out_channels, reduction=32)
            self.ca_p4 = CoordinateAttention(out_channels, out_channels, reduction=32)
            self.ca_p5 = CoordinateAttention(out_channels, out_channels, reduction=32)
            
            # Channel Shuffling (identique V1)
            self.cs1 = ChannelShuffle(out_channels, groups=2)  # P3
            self.cs2 = ChannelShuffle(out_channels, groups=2)  # P4
            self.cs3 = ChannelShuffle(out_channels, groups=2)  # P5
        
        # Detection heads (identique V1)
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        
        # Statistics pour comparaison
        ca_params = (sum(p.numel() for p in self.ca_p3.parameters()) +
                     sum(p.numel() for p in self.ca_p4.parameters()) +
                     sum(p.numel() for p in self.ca_p5.parameters()))
        
        self.innovation_stats = {
            'coordinate_attention_modules': 3,
            'coordinate_attention_parameters': ca_params,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'model_type': 'featherface_v2_innovation',
            'attention_mechanisms': ['coordinate_attention'],
            'innovation': 'coordinate_attention_vs_cbam'
        }
        
        print(f"   Coordinate Attention modules: 3 (vs 6 CBAM)")
        print(f"   Coordinate Attention parameters: {ca_params:,}")
        print(f"   Total parameters: {self.innovation_stats['total_parameters']:,}")
    
    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead
    
    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        """
        Forward pass FeatherFace V2
        
        Innovation : Remplace 6 CBAM par 3 Coordinate Attention
        1. MobileNet backbone (identique V1)
        2. BiFPN feature pyramid (identique V1)
        3. SSH context enhancement (identique V1)
        4. Coordinate Attention (INNOVATION vs CBAM)
        5. Channel Shuffle (identique V1)
        6. Detection heads (identique V1)
        
        Args:
            inputs: Input tensor [B, 3, H, W]
            
        Returns:
            Tuple: (bbox_regressions, classifications, ldm_regressions)
        """
        if self.cfg['name'] == 'mobilenet0.25':
            # 1. MobileNet backbone - extraction multi-échelle (identique V1)
            out = self.body(inputs)
            out = list(out.values())  # [P3, P4, P5]
            
            # 2. BiFPN feature pyramid (identique V1, pas d'attention)
            bifpn_features = self.bifpn(out)
            
            # 3. SSH context enhancement (identique V1)
            ssh_feature1 = self.ssh1(bifpn_features[0])  # P3
            ssh_feature2 = self.ssh2(bifpn_features[1])  # P4
            ssh_feature3 = self.ssh3(bifpn_features[2])  # P5
            
            # 4. INNOVATION V2: Coordinate Attention (remplace 6 CBAM)
            # Placement optimal post-SSH pour détection de visages
            ca_feature1 = self.ca_p3(ssh_feature1)  # P3: Coordinate Attention
            ca_feature2 = self.ca_p4(ssh_feature2)  # P4: Coordinate Attention
            ca_feature3 = self.ca_p5(ssh_feature3)  # P5: Coordinate Attention
            
            # 5. Channel Shuffling (identique V1)
            feat1 = self.cs1(ca_feature1)  # P3
            feat2 = self.cs2(ca_feature2)  # P4
            feat3 = self.cs3(ca_feature3)  # P5
            
            features = [feat1, feat2, feat3]
        
        # 6. Detection heads (identique V1)
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        
        return output
    
    def get_innovation_stats(self):
        """Get statistics de l'innovation V2"""
        return self.innovation_stats
    
    def get_attention_maps(self, input_tensor):
        """Get attention maps pour analyse"""
        # Cette méthode peut être utilisée pour visualiser les cartes d'attention
        attention_maps = {}
        
        with torch.no_grad():
            # Forward pass jusqu'aux features SSH
            out = self.body(input_tensor)
            out = list(out.values())
            bifpn_features = self.bifpn(out)
            ssh_features = [
                self.ssh1(bifpn_features[0]),
                self.ssh2(bifpn_features[1]),
                self.ssh3(bifpn_features[2])
            ]
            
            # Les modules Coordinate Attention n'exposent pas directement les cartes
            # mais on peut récupérer les features post-attention
            attention_maps['p3_ca'] = self.ca_p3(ssh_features[0])
            attention_maps['p4_ca'] = self.ca_p4(ssh_features[1])
            attention_maps['p5_ca'] = self.ca_p5(ssh_features[2])
            
        return attention_maps
#!/usr/bin/env python3
"""
FeatherFace V1 Original - Implémentation Fidèle au Repository GitHub

Ce modèle implémente le VRAI FeatherFace V1 selon le repository original :
https://github.com/dohun-mat/FeatherFace

Architecture Complète :
1. MobileNetV1 backbone
2. CBAM sur backbone features (3 modules)
3. BiFPN feature pyramid
4. CBAM sur BiFPN features (3 modules)  
5. SSH context modules
6. Channel Shuffle
7. Detection heads

TOTAL : 6 modules CBAM comme dans l'article original
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import SSH as SSH
from models.net import BiFPN as BiFPN
from models.net import CBAM as CBAM
from models.net import ChannelShuffle2 as ChannelShuffle


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


class RetinaFace(nn.Module):
    """
    FeatherFace V1 Original - Implémentation Fidèle au Repository
    
    Cette implémentation reproduit exactement l'architecture du repository GitHub original
    avec tous les modules CBAM comme décrit dans l'article.
    
    Architecture :
    - MobileNetV1 0.25x backbone
    - 3 modules CBAM post-backbone (P3, P4, P5)
    - BiFPN feature pyramid network
    - 3 modules CBAM post-BiFPN (P3, P4, P5)
    - SSH context enhancement
    - Channel Shuffle
    - Detection heads (classification, bbox, landmarks)
    
    TOTAL : 6 modules CBAM avec residual connections
    """
    
    def __init__(self, cfg=None, phase='train'):
        """
        Initialize FeatherFace V1 Original
        
        Args:
            cfg: Configuration dict (cfg_mnet)
            phase: 'train' or 'test'
        """
        super(RetinaFace, self).__init__()
        
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
            
            print(f"🔧 FeatherFace V1 Original - Initializing with CBAM modules...")
            print(f"   Backbone channels: {in_channels_list}")
            print(f"   Output channels: {out_channels}")
            
            # ORIGINAL FEATURE: CBAM sur backbone features
            # Exactement comme dans le repository GitHub original
            self.bacbkbone_0_cbam = CBAM(in_channels_list[0], 16)  # P3: 64 channels
            self.bacbkbone_1_cbam = CBAM(in_channels_list[1], 16)  # P4: 128 channels
            self.bacbkbone_2_cbam = CBAM(in_channels_list[2], 16)  # P5: 256 channels
            
            # Activation layers pour residual connections
            self.relu_0 = nn.ReLU(inplace=True)
            self.relu_1 = nn.ReLU(inplace=True) 
            self.relu_2 = nn.ReLU(inplace=True)
            
            # BiFPN configuration (standard, pas d'attention intégrée)
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
            
            # BiFPN standard (pas d'attention intégrée dans le repository original)
            self.bifpn = nn.Sequential(
                *[BiFPN(self.fpn_num_filters[self.compound_coef],
                        conv_channel_coef[self.compound_coef],
                        True if _ == 0 else False,
                        attention=False  # Pas d'attention dans le BiFPN original
                        )
                  for _ in range(self.fpn_cell_repeats[self.compound_coef])])
            
            # ORIGINAL FEATURE: CBAM sur BiFPN features  
            # Exactement comme dans le repository GitHub original
            self.bif_cbam_0 = CBAM(out_channels, 16)  # Post-BiFPN P3
            self.bif_cbam_1 = CBAM(out_channels, 16)  # Post-BiFPN P4
            self.bif_cbam_2 = CBAM(out_channels, 16)  # Post-BiFPN P5
            
            # Activation layers pour BiFPN residual connections
            self.bif_relu_0 = nn.ReLU(inplace=True)
            self.bif_relu_1 = nn.ReLU(inplace=True)
            self.bif_relu_2 = nn.ReLU(inplace=True)
            
            # SSH context modules (standard)
            self.ssh1 = SSH(out_channels, out_channels)  # P3
            self.ssh2 = SSH(out_channels, out_channels)  # P4
            self.ssh3 = SSH(out_channels, out_channels)  # P5
            
            # Channel Shuffling (standard)
            self.cs1 = ChannelShuffle(out_channels, groups=2)  # P3
            self.cs2 = ChannelShuffle(out_channels, groups=2)  # P4
            self.cs3 = ChannelShuffle(out_channels, groups=2)  # P5
        
        # Detection heads
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        
        # Statistics pour comparaison
        cbam_params = (sum(p.numel() for p in self.bacbkbone_0_cbam.parameters()) +
                       sum(p.numel() for p in self.bacbkbone_1_cbam.parameters()) +
                       sum(p.numel() for p in self.bacbkbone_2_cbam.parameters()) +
                       sum(p.numel() for p in self.bif_cbam_0.parameters()) +
                       sum(p.numel() for p in self.bif_cbam_1.parameters()) +
                       sum(p.numel() for p in self.bif_cbam_2.parameters()))
        
        self.original_stats = {
            'cbam_modules': 6,
            'cbam_parameters': cbam_params,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'model_type': 'featherface_v1_original',
            'attention_mechanisms': ['cbam_backbone', 'cbam_bifpn'],
            'repository': 'https://github.com/dohun-mat/FeatherFace'
        }
        
        print(f"   CBAM modules: 6 (3 backbone + 3 BiFPN)")
        print(f"   CBAM parameters: {cbam_params:,}")
        print(f"   Total parameters: {self.original_stats['total_parameters']:,}")
    
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
        Forward pass FeatherFace V1 Original
        
        Architecture exacte du repository GitHub :
        1. MobileNet backbone
        2. CBAM sur backbone features + residual
        3. BiFPN feature pyramid
        4. CBAM sur BiFPN features + residual
        5. SSH context enhancement
        6. Channel Shuffle
        7. Detection heads
        
        Args:
            inputs: Input tensor [B, 3, H, W]
            
        Returns:
            Tuple: (bbox_regressions, classifications, ldm_regressions)
        """
        if self.cfg['name'] == 'mobilenet0.25':
            # 1. MobileNet backbone - extraction multi-échelle
            out = self.body(inputs)
            out = list(out.values())  # [P3, P4, P5]
            
            # 2. CBAM sur BACKBONE features (EXACTEMENT comme repository original)
            # Avec residual connections et activations
            cbam_0 = self.bacbkbone_0_cbam(out[0])  # P3: CBAM attention
            cbam_0 = cbam_0 + out[0]                # Residual connection
            cbam_0 = self.relu_0(cbam_0)            # Activation
            
            cbam_1 = self.bacbkbone_1_cbam(out[1])  # P4: CBAM attention
            cbam_1 = cbam_1 + out[1]                # Residual connection  
            cbam_1 = self.relu_1(cbam_1)            # Activation
            
            cbam_2 = self.bacbkbone_2_cbam(out[2])  # P5: CBAM attention
            cbam_2 = cbam_2 + out[2]                # Residual connection
            cbam_2 = self.relu_2(cbam_2)            # Activation
            
            backbone_attention_features = [cbam_0, cbam_1, cbam_2]
            
            # 3. BiFPN feature pyramid (standard, pas d'attention intégrée)
            bifpn_features = self.bifpn(backbone_attention_features)
            
            # 4. CBAM sur BiFPN features (EXACTEMENT comme repository original) 
            # Avec residual connections et activations
            bif_cbam_0 = self.bif_cbam_0(bifpn_features[0])  # P3: CBAM attention
            bif_cbam_0 = bif_cbam_0 + bifpn_features[0]      # Residual connection
            bif_cbam_0 = self.bif_relu_0(bif_cbam_0)         # Activation
            
            bif_cbam_1 = self.bif_cbam_1(bifpn_features[1])  # P4: CBAM attention
            bif_cbam_1 = bif_cbam_1 + bifpn_features[1]      # Residual connection
            bif_cbam_1 = self.bif_relu_1(bif_cbam_1)         # Activation
            
            bif_cbam_2 = self.bif_cbam_2(bifpn_features[2])  # P5: CBAM attention
            bif_cbam_2 = bif_cbam_2 + bifpn_features[2]      # Residual connection
            bif_cbam_2 = self.bif_relu_2(bif_cbam_2)         # Activation
            
            bif_attention_features = [bif_cbam_0, bif_cbam_1, bif_cbam_2]
            
            # 5. SSH context enhancement (standard)
            ssh_feature1 = self.ssh1(bif_attention_features[0])  # P3
            ssh_feature2 = self.ssh2(bif_attention_features[1])  # P4
            ssh_feature3 = self.ssh3(bif_attention_features[2])  # P5
            
            # 6. Channel Shuffling (standard)
            feat1 = self.cs1(ssh_feature1)  # P3
            feat2 = self.cs2(ssh_feature2)  # P4
            feat3 = self.cs3(ssh_feature3)  # P5
            
            features = [feat1, feat2, feat3]
        
        # 7. Detection heads (standard)
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        
        return output
    
    def get_original_stats(self):
        """Get statistics de l'implémentation originale"""
        return self.original_stats
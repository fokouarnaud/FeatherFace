"""
FeatherFace ECA-CBAM Parallel Implementation
===========================================

Ce module implémente FeatherFace avec un mécanisme d'attention hybride parallèle
ECA-CBAM, utilisant une fusion multiplicative simple selon Wang et al. (2024).

Architecture Parallèle vs Séquentielle:
- Séquentiel: X → ECA(X) → F_eca → SAM(F_eca) → Y
- Parallèle: M_c = ECA(X), M_s = SAM(X), M_hybrid = M_c ⊙ M_s, Y = X ⊙ M_hybrid

Fondation Scientifique:
- ECA-Net: Wang et al. CVPR 2020 (Efficient Channel Attention)
- CBAM: Woo et al. ECCV 2018 (Convolutional Block Attention Module)
- Hybrid Parallel: Wang et al. 2024 (Multiplicative Fusion)

Innovation Architecturale:
- Génération parallèle des masques canal et spatial
- Fusion multiplicative pure (sans poids apprenables)
- Meilleure complémentarité canal/spatial
- Réduction des interférences entre modules

Caractéristiques Clés:
- MobileNet-0.25 backbone pour efficience mobile
- ECA-CBAM hybrid attention parallèle (6 modules total)
- BiFPN pour agrégation multiscale
- SSH avec convolutions déformables
- Channel shuffle optimization
- Objectif: ~476K paramètres (identique au séquentiel)

Cibles de Performance (WIDERFace):
- Easy: ≥94.0% AP (+1.5% à +2.5% vs séquentiel attendu)
- Medium: ≥92.0% AP
- Hard: ≥80.0% AP
- Overall: ≥88.7% AP

Avantages Attendus (Wang et al. 2024):
- Meilleure densité de recalibrage sur régions pertinentes
- Réduction lissage excessif de l'attention spatiale
- Amélioration robustesse sur sous-ensembles difficiles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from collections import OrderedDict

from models.net import MobileNetV1, SSH, BiFPN, ChannelShuffle2
from models.eca_cbam_hybrid import ECAcbaM_Parallel_Simple


class ClassHead(nn.Module):
    """Classification head for face detection"""
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors*2, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    """Bounding box regression head"""
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    """Facial landmark detection head"""
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*10, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 10)


class FeatherFaceECAcbaMParallel(nn.Module):
    """
    FeatherFace avec Attention Hybride Parallèle ECA-CBAM

    Implémente le schéma parallèle innovant qui combine:
    - ECA-Net pour attention canal efficiente (22 paramètres par module)
    - CBAM SAM pour attention spatiale (98 paramètres par module)
    - Fusion multiplicative simple (0 paramètres supplémentaires)

    Architecture Parallèle (Wang et al. 2024):
    Pour chaque feature map X:
    1. M_c = σ(Conv1D(GAP(X)))           [Masque canal en parallèle]
    2. M_s = σ(Conv2D([AvgPool; MaxPool]))  [Masque spatial en parallèle]
    3. M_hybrid = M_c ⊙ M_s              [Fusion multiplicative]
    4. Y = X ⊙ M_hybrid                  [Application finale]

    Vue d'Ensemble Architecture:
    1. MobileNet-0.25 backbone (~213K paramètres)
    2. Modules attention parallèle ECA-CBAM (6× ~100 params chacun)
    3. BiFPN agrégation features (~93K paramètres)
    4. SSH detection heads avec DCN (~150K paramètres)
    5. Channel shuffle optimization (~10K paramètres)
    6. Detection heads (5.5K paramètres)

    Total: ~476K paramètres (identique au séquentiel, 2.5% réduction vs CBAM baseline)

    Innovation Clé:
    - Génération parallèle des masques vs traitement séquentiel
    - Fusion multiplicative pure pour complémentarité optimale
    - Réduction interférences entre branches canal/spatial
    - Meilleure préservation des features originales
    """

    def __init__(self, cfg=None, phase='train'):
        super(FeatherFaceECAcbaMParallel, self).__init__()
        self.phase = phase
        self.cfg = cfg

        # Configuration validation
        if cfg is None:
            raise ValueError("Configuration required for FeatherFace ECA-CBAM Parallel")

        # 1. MobileNet-0.25 Backbone (~213K parameters)
        backbone = MobileNetV1()
        if cfg['name'] == 'mobilenet0.25':
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
            in_channels_stage2 = cfg['in_channel']
            in_channels_list = [
                in_channels_stage2 * 2,  # 64
                in_channels_stage2 * 4,  # 128
                in_channels_stage2 * 8,  # 256
            ]
            out_channels = cfg['out_channel']  # 52 for optimal parameter count

        # 2. ECA-CBAM Parallel Attention Modules (Innovation)
        # Backbone ECA-CBAM parallel modules (3×) - fusion multiplicative simple
        self.backbone_attention_0 = ECAcbaM_Parallel_Simple(
            channels=in_channels_list[0],  # 64 channels
            gamma=cfg.get('eca_gamma', 2),
            beta=cfg.get('eca_beta', 1),
            spatial_kernel_size=cfg.get('sam_kernel_size', 7)
        )
        self.backbone_attention_1 = ECAcbaM_Parallel_Simple(
            channels=in_channels_list[1],  # 128 channels
            gamma=cfg.get('eca_gamma', 2),
            beta=cfg.get('eca_beta', 1),
            spatial_kernel_size=cfg.get('sam_kernel_size', 7)
        )
        self.backbone_attention_2 = ECAcbaM_Parallel_Simple(
            channels=in_channels_list[2],  # 256 channels
            gamma=cfg.get('eca_gamma', 2),
            beta=cfg.get('eca_beta', 1),
            spatial_kernel_size=cfg.get('sam_kernel_size', 7)
        )

        # 3. BiFPN Feature Aggregation (~93K parameters)
        conv_channel_coef = {
            0: [in_channels_list[0], in_channels_list[1], in_channels_list[2]],
        }
        self.fpn_num_filters = [out_channels, 256, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.compound_coef = 0

        # Create BiFPN layers
        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[self.compound_coef],
                    True if _ == 0 else False,
                    attention=True)
              for _ in range(self.fpn_cell_repeats[self.compound_coef])])

        # BiFPN ECA-CBAM parallel modules (3×)
        self.bifpn_attention_0 = ECAcbaM_Parallel_Simple(
            channels=out_channels,  # P3 - 52 channels
            gamma=cfg.get('eca_gamma', 2),
            beta=cfg.get('eca_beta', 1),
            spatial_kernel_size=cfg.get('sam_kernel_size', 7)
        )
        self.bifpn_attention_1 = ECAcbaM_Parallel_Simple(
            channels=out_channels,  # P4 - 52 channels
            gamma=cfg.get('eca_gamma', 2),
            beta=cfg.get('eca_beta', 1),
            spatial_kernel_size=cfg.get('sam_kernel_size', 7)
        )
        self.bifpn_attention_2 = ECAcbaM_Parallel_Simple(
            channels=out_channels,  # P5 - 52 channels
            gamma=cfg.get('eca_gamma', 2),
            beta=cfg.get('eca_beta', 1),
            spatial_kernel_size=cfg.get('sam_kernel_size', 7)
        )

        # 4. SSH Detection Heads with DCN (~150K parameters)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        # 5. Channel Shuffle Optimization (~10K parameters)
        self.ssh1_cs = ChannelShuffle2(out_channels, 2)
        self.ssh2_cs = ChannelShuffle2(out_channels, 2)
        self.ssh3_cs = ChannelShuffle2(out_channels, 2)

        # 6. Detection Heads (5.5K parameters)
        self.ClassHead = nn.ModuleList([
            ClassHead(out_channels, 2) for _ in range(3)
        ])
        self.BboxHead = nn.ModuleList([
            BboxHead(out_channels, 2) for _ in range(3)
        ])
        self.LandmarkHead = nn.ModuleList([
            LandmarkHead(out_channels, 2) for _ in range(3)
        ])

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """
        Forward pass avec attention hybride parallèle ECA-CBAM

        Processus:
        1. Extraction features backbone
        2. Attention hybride parallèle sur features backbone (M_c ∥ M_s → fusion)
        3. Agrégation features BiFPN
        4. Attention hybride parallèle sur features BiFPN (M_c ∥ M_s → fusion)
        5. SSH detection avec DCN
        6. Channel shuffle optimization
        7. Detection heads

        Args:
            inputs: Input tensor [B, 3, H, W]

        Returns:
            tuple: (bbox_regressions, classifications, ldm_regressions)
        """

        # 1. Backbone feature extraction
        out = self.body(inputs)

        # Extract multiscale features
        feat1 = out[1]  # stage1 → 64 channels
        feat2 = out[2]  # stage2 → 128 channels
        feat3 = out[3]  # stage3 → 256 channels

        # 2. Apply parallel ECA-CBAM attention to backbone features
        # Innovation: Parallel mask generation + multiplicative fusion
        feat1 = self.backbone_attention_0(feat1)  # M_c ∥ M_s → M_hybrid
        feat2 = self.backbone_attention_1(feat2)  # M_c ∥ M_s → M_hybrid
        feat3 = self.backbone_attention_2(feat3)  # M_c ∥ M_s → M_hybrid

        # 3. BiFPN feature aggregation
        features = [feat1, feat2, feat3]
        features = self.bifpn(features)
        p3, p4, p5 = features

        # 4. Apply parallel ECA-CBAM attention to BiFPN features
        # Innovation: Parallel mask generation + multiplicative fusion
        p3 = self.bifpn_attention_0(p3)  # M_c ∥ M_s → M_hybrid
        p4 = self.bifpn_attention_1(p4)  # M_c ∥ M_s → M_hybrid
        p5 = self.bifpn_attention_2(p5)  # M_c ∥ M_s → M_hybrid

        # 5. SSH detection with DCN
        f1 = self.ssh1(p3)
        f2 = self.ssh2(p4)
        f3 = self.ssh3(p5)

        # 6. Channel shuffle optimization
        f1 = self.ssh1_cs(f1)
        f2 = self.ssh2_cs(f2)
        f3 = self.ssh3_cs(f3)

        # 7. Detection heads
        features = [f1, f2, f3]

        bbox_regressions = []
        classifications = []
        ldm_regressions = []

        for i, feature in enumerate(features):
            bbox_regressions.append(self.BboxHead[i](feature))
            classifications.append(self.ClassHead[i](feature))
            ldm_regressions.append(self.LandmarkHead[i](feature))

        if self.phase == 'train':
            output = (
                torch.cat(bbox_regressions, dim=1),
                torch.cat(classifications, dim=1),
                torch.cat(ldm_regressions, dim=1)
            )
        else:
            output = (
                torch.cat(bbox_regressions, dim=1),
                F.softmax(torch.cat(classifications, dim=1), dim=-1),
                torch.cat(ldm_regressions, dim=1)
            )

        return output

    def get_parameter_count(self):
        """Get detailed parameter count breakdown pour ECA-CBAM parallel"""

        # Backbone parameters
        backbone_params = sum(p.numel() for p in self.body.parameters())

        # ECA-CBAM parallel backbone attention parameters
        ecacbam_backbone_params = (
            sum(p.numel() for p in self.backbone_attention_0.parameters()) +
            sum(p.numel() for p in self.backbone_attention_1.parameters()) +
            sum(p.numel() for p in self.backbone_attention_2.parameters())
        )

        # BiFPN parameters
        bifpn_params = sum(p.numel() for p in self.bifpn.parameters())

        # ECA-CBAM parallel BiFPN attention parameters
        ecacbam_bifpn_params = (
            sum(p.numel() for p in self.bifpn_attention_0.parameters()) +
            sum(p.numel() for p in self.bifpn_attention_1.parameters()) +
            sum(p.numel() for p in self.bifpn_attention_2.parameters())
        )

        # SSH parameters
        ssh_params = (
            sum(p.numel() for p in self.ssh1.parameters()) +
            sum(p.numel() for p in self.ssh2.parameters()) +
            sum(p.numel() for p in self.ssh3.parameters())
        )

        # Channel shuffle parameters
        cs_params = (
            sum(p.numel() for p in self.ssh1_cs.parameters()) +
            sum(p.numel() for p in self.ssh2_cs.parameters()) +
            sum(p.numel() for p in self.ssh3_cs.parameters())
        )

        # Detection heads parameters
        head_params = (
            sum(p.numel() for p in self.ClassHead.parameters()) +
            sum(p.numel() for p in self.BboxHead.parameters()) +
            sum(p.numel() for p in self.LandmarkHead.parameters())
        )

        # Total parameters
        total = (backbone_params + ecacbam_backbone_params + bifpn_params +
                ecacbam_bifpn_params + ssh_params + cs_params + head_params)

        # Attention efficiency analysis
        total_attention_params = ecacbam_backbone_params + ecacbam_bifpn_params

        return {
            'backbone': backbone_params,
            'ecacbam_parallel_backbone': ecacbam_backbone_params,
            'bifpn': bifpn_params,
            'ecacbam_parallel_bifpn': ecacbam_bifpn_params,
            'ssh': ssh_params,
            'channel_shuffle': cs_params,
            'detection_heads': head_params,
            'total': total,
            'total_attention': total_attention_params,
            'cbam_baseline': 488664,
            'eca_cbam_sequential': 476345,
            'parameter_reduction_vs_cbam': 488664 - total,
            'parameter_diff_vs_sequential': total - 476345,
            'efficiency_gain_vs_cbam': ((488664 - total) / 488664) * 100,
            'attention_efficiency': total_attention_params / 6,  # Per module
            'fusion_type': 'multiplicative_simple (0 learnable params)',
            'validation': {
                'similar_to_sequential': abs(total - 476345) < 1000,  # ~identique
                'efficiency_vs_cbam': total < 488664,
                'attention_efficient': total_attention_params < 5000
            }
        }

    def get_attention_heatmaps(self, x):
        """
        Extrait heatmaps d'attention pour analyse qualitative parallèle vs séquentiel

        Args:
            x: Input tensor pour analyse

        Returns:
            dict: Heatmaps complets (canal, spatial, hybrid) pour chaque module
        """
        heatmaps = {}

        with torch.no_grad():
            # Backbone features
            backbone_features = self.body(x)
            feat1, feat2, feat3 = backbone_features[1], backbone_features[2], backbone_features[3]

            # Backbone attention heatmaps
            heatmaps['backbone'] = {
                'stage1': self.backbone_attention_0.get_attention_heatmaps(feat1),
                'stage2': self.backbone_attention_1.get_attention_heatmaps(feat2),
                'stage3': self.backbone_attention_2.get_attention_heatmaps(feat3)
            }

            # Apply backbone attention
            feat1_att = self.backbone_attention_0(feat1)
            feat2_att = self.backbone_attention_1(feat2)
            feat3_att = self.backbone_attention_2(feat3)

            # BiFPN features
            features = [feat1_att, feat2_att, feat3_att]
            features = self.bifpn(features)
            p3, p4, p5 = features

            # BiFPN attention heatmaps
            heatmaps['bifpn'] = {
                'P3': self.bifpn_attention_0.get_attention_heatmaps(p3),
                'P4': self.bifpn_attention_1.get_attention_heatmaps(p4),
                'P5': self.bifpn_attention_2.get_attention_heatmaps(p5)
            }

        return heatmaps

    def get_attention_analysis(self, x):
        """
        Analyse complète des patterns d'attention parallèle

        Args:
            x: Input tensor pour analyse

        Returns:
            dict: Analyse complète attention parallèle
        """
        analysis = {}

        with torch.no_grad():
            # Backbone features
            backbone_features = self.body(x)
            feat1, feat2, feat3 = backbone_features[1], backbone_features[2], backbone_features[3]

            # Analyze backbone parallel attention
            backbone_analysis = {}
            backbone_analysis['stage1'] = self.backbone_attention_0.get_attention_analysis(feat1)
            backbone_analysis['stage2'] = self.backbone_attention_1.get_attention_analysis(feat2)
            backbone_analysis['stage3'] = self.backbone_attention_2.get_attention_analysis(feat3)

            # Apply backbone attention
            feat1_att = self.backbone_attention_0(feat1)
            feat2_att = self.backbone_attention_1(feat2)
            feat3_att = self.backbone_attention_2(feat3)

            # BiFPN features
            features = [feat1_att, feat2_att, feat3_att]
            features = self.bifpn(features)
            p3, p4, p5 = features

            # Analyze BiFPN parallel attention
            bifpn_analysis = {}
            bifpn_analysis['P3'] = self.bifpn_attention_0.get_attention_analysis(p3)
            bifpn_analysis['P4'] = self.bifpn_attention_1.get_attention_analysis(p4)
            bifpn_analysis['P5'] = self.bifpn_attention_2.get_attention_analysis(p5)

            analysis = {
                'backbone_attention': backbone_analysis,
                'bifpn_attention': bifpn_analysis,
                'parameter_count': self.get_parameter_count(),
                'attention_summary': {
                    'mechanism': 'ECA-CBAM Hybrid Parallel (Simple Multiplicative)',
                    'modules_count': 6,
                    'channel_attention': 'ECA-Net (efficient, parallel generation)',
                    'spatial_attention': 'CBAM SAM (localization, parallel generation)',
                    'fusion_type': 'Multiplicative (M_c ⊙ M_s)',
                    'architecture': 'Parallel processing (Wang et al. 2024)',
                    'innovation': 'Parallel mask generation for better complementarity'
                }
            }

        return analysis

    def compare_with_sequential(self):
        """Compare parallel avec séquentiel ECA-CBAM"""
        param_info = self.get_parameter_count()

        comparison = {
            'parameter_comparison': {
                'cbam_baseline': 488664,
                'eca_cbam_sequential': 476345,
                'eca_cbam_parallel': param_info['total'],
                'diff_vs_sequential': param_info['parameter_diff_vs_sequential'],
                'reduction_vs_cbam': param_info['parameter_reduction_vs_cbam']
            },
            'architecture_comparison': {
                'sequential': {
                    'flow': 'X → ECA(X) → F_eca → SAM(F_eca) → Y',
                    'processing': 'Channel then spatial (cascaded)',
                    'fusion': 'Direct application (no explicit fusion)',
                    'masques': 'M_c appliqué d\'abord, puis M_s sur résultat'
                },
                'parallel': {
                    'flow': 'X → [M_c = ECA(X) ∥ M_s = SAM(X)] → M_hybrid = M_c ⊙ M_s → Y',
                    'processing': 'Channel and spatial simultaneously',
                    'fusion': 'Multiplicative (M_c ⊙ M_s)',
                    'masques': 'M_c et M_s générés simultanément depuis X'
                }
            },
            'expected_advantages_parallel': {
                'complementarity': 'Meilleure complémentarité canal/spatial',
                'interference': 'Réduction interférences entre modules',
                'density': 'Amélioration densité recalibrage régions pertinentes',
                'smoothing': 'Moins de lissage excessif attention spatiale',
                'performance': '+1.5% à +2.5% mAP vs séquentiel (Wang et al. 2024)'
            },
            'scientific_validation': {
                'wang_2024': 'Parallel hybrid attention mechanisms',
                'eca_net': 'Wang et al. CVPR 2020',
                'cbam': 'Woo et al. ECCV 2018',
                'fusion': 'Multiplicative simple (0 params supplémentaires)'
            }
        }

        return comparison


def create_eca_cbam_parallel_model(cfg_eca_cbam_parallel, phase='train'):
    """
    Factory function pour créer modèle FeatherFace ECA-CBAM Parallel

    Args:
        cfg_eca_cbam_parallel: Configuration pour ECA-CBAM parallel
        phase: 'train' ou 'test'

    Returns:
        FeatherFaceECAcbaMParallel model avec ~476K params (identique séquentiel)
    """
    model = FeatherFaceECAcbaMParallel(cfg=cfg_eca_cbam_parallel, phase=phase)

    # Parameter analysis
    param_info = model.get_parameter_count()

    print(f"🔬 FeatherFace ECA-CBAM Parallel Model Created")
    print(f"📊 Total parameters: {param_info['total']:,}")
    print(f"📈 vs CBAM baseline: {param_info['parameter_reduction_vs_cbam']:,} ({param_info['efficiency_gain_vs_cbam']:.1f}%)")
    print(f"📈 vs Sequential: {param_info['parameter_diff_vs_sequential']:,} params")
    print(f"🎯 Attention efficiency: {param_info['attention_efficiency']:.0f} params/module")
    print(f"🔀 Fusion type: {param_info['fusion_type']}")
    print(f"✅ Validation: {param_info['validation']}")

    if param_info['validation']['similar_to_sequential']:
        print(f"🚀 Innovation: Parallel attention avec même nombre de paramètres que séquentiel!")

    return model


def test_eca_cbam_parallel_featherface():
    """Test ECA-CBAM Parallel FeatherFace implementation"""
    print("🧪 Testing FeatherFace ECA-CBAM Parallel")
    print("=" * 60)

    # Mock configuration for testing
    cfg_test = {
        'name': 'mobilenet0.25',
        'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
        'in_channel': 32,
        'out_channel': 52,
        'eca_gamma': 2,
        'eca_beta': 1,
        'sam_kernel_size': 7
    }

    # Create model
    model = FeatherFaceECAcbaMParallel(cfg=cfg_test, phase='test')

    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        bbox_reg, cls, ldm = model(x)

    print(f"✅ Forward pass successful:")
    print(f"  📦 Input shape: {x.shape}")
    print(f"  📦 Bbox regression: {bbox_reg.shape}")
    print(f"  📦 Classification: {cls.shape}")
    print(f"  📦 Landmarks: {ldm.shape}")

    # Parameter analysis
    param_info = model.get_parameter_count()
    print(f"\n📊 Parameter Analysis:")
    print(f"  🔢 Total parameters: {param_info['total']:,}")
    print(f"  🔢 vs CBAM baseline: {param_info['parameter_reduction_vs_cbam']:,}")
    print(f"  🔢 vs Sequential: {param_info['parameter_diff_vs_sequential']:,}")
    print(f"  🔢 Attention efficiency: {param_info['attention_efficiency']:.0f} params/module")

    # Comparison with sequential
    comparison = model.compare_with_sequential()
    print(f"\n🔬 Parallel vs Sequential:")
    print(f"  📊 Architecture: {comparison['architecture_comparison']['parallel']['flow']}")
    print(f"  📊 Fusion: {comparison['architecture_comparison']['parallel']['fusion']}")
    print(f"  📊 Expected advantages: {comparison['expected_advantages_parallel']['performance']}")

    print(f"\n🎯 ECA-CBAM Parallel FeatherFace Ready!")
    print(f"🚀 Innovation: Parallel attention avec fusion multiplicative simple")
    print(f"📈 Attendu: +1.5% à +2.5% mAP vs séquentiel (Wang et al. 2024)")


if __name__ == "__main__":
    test_eca_cbam_parallel_featherface()

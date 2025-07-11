# FeatherFace - Structure Simplifiée

## 🎯 Architecture Claire et Épurée

Cette structure simplifiée permet une comparaison directe et claire entre V1 Original et V2 Innovation.

## 📁 Structure Finale

```
FeatherFace/
├── 🔧 models/                    # Modèles principaux
│   ├── net.py                    # Composants de base (MobileNet, BiFPN, CBAM, SSH)
│   ├── retinaface.py            # V1 Original (502K params, 6 CBAM modules)
│   ├── featherface_v2.py        # V2 Innovation (493K params, 3 Coordinate Attention)
│   └── attention_v2.py          # Module Coordinate Attention
├── 🚀 Scripts d'entraînement:
│   ├── train_v1.py              # Entraînement V1 Original
│   └── train_v2.py              # Entraînement V2 Innovation
├── 📊 data/                     # Dataset et configurations
│   ├── config.py                # cfg_mnet (V1) et cfg_v2 (V2)
│   └── widerface/               # Dataset WIDERFace
└── 📚 Documentation:
    ├── README.md                # Documentation principale
    ├── CLAUDE.md                # Instructions Claude
    └── docs/                    # Documentation détaillée
```

## ⚡ Utilisation Simplifiée

### Entraîner V1 Original (Baseline)
```bash
python train_v1.py --training_dataset ./data/widerface/train/label.txt --network mobile0.25
```

### Entraîner V2 Innovation
```bash
python train_v2.py --training_dataset ./data/widerface/train/label.txt --network v2
```

## 🔬 Comparaison Scientifique

| Aspect | V1 Original | V2 Innovation |
|--------|-------------|---------------|
| **Fichier** | `models/retinaface.py` | `models/featherface_v2.py` |
| **Script** | `train_v1.py` | `train_v2.py` |
| **Paramètres** | 502K | 493K (-1.8%) |
| **Attention** | 6 CBAM modules | 3 Coordinate Attention |
| **Config** | `cfg_mnet` | `cfg_v2` |
| **Innovation** | Baseline GitHub | Coordinate Attention |

## ✅ Avantages de cette Structure

1. **Clarté Maximale** : 2 modèles, 2 scripts, comparaison directe
2. **Méthodologie Claire** : V1 Original → V2 Innovation
3. **Reproductibilité** : Scripts indépendants et configurations claires
4. **Maintenance Simple** : Moins de fichiers, moins de confusion
5. **Comparaison Scientifique** : Changement contrôlé (CBAM → Coordinate Attention)

## 🎯 Points Clés

- **V1 Original** : Implémentation fidèle du repository GitHub avec 6 CBAM
- **V2 Innovation** : Remplacement des 6 CBAM par 3 Coordinate Attention
- **Gain** : -9K paramètres (-1.8%) avec amélioration performance attendue
- **Foundation** : Hou et al. CVPR 2021 (Coordinate Attention)

Cette structure épurée permet de se concentrer sur l'essentiel : la comparaison scientifique entre les deux approches d'attention.
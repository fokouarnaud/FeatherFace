# Notebook 03: Comparaison Séquentiel vs Parallèle

## Vue d'ensemble

Ce notebook compare les architectures séquentielle et parallèle pour l'attention hybride ECA-CBAM.

## Structure du Notebook

### 1. Setup Environnement
- Import librairies (torch, matplotlib, pandas, seaborn)
- Configuration device (GPU/CPU)
- Import modèles et configurations

### 2. Validation Modèles
```python
from models.featherface_cbam_exact import FeatherFaceCBAMExact
from models.featherface_eca_cbam import FeatherFaceECAcbaM
from models.featherface_eca_cbam_parallel import FeatherFaceECAcbaMParallel

# Créer les 3 modèles
models = {
    'CBAM Baseline': FeatherFaceCBAMExact(cfg=cfg_cbam_paper_exact),
    'ECA-CBAM Séquentiel': FeatherFaceECAcbaM(cfg=cfg_eca_cbam),
    'ECA-CBAM Parallèle': FeatherFaceECAcbaMParallel(cfg=cfg_eca_cbam_parallel)
}

# Comparer paramètres
for name, model in models.items():
    params = model.get_parameter_count()
    print(f"{name}: {params['total']:,} parameters")
```

### 3. Test Forward Pass et Latence
- Mesure latence CPU/GPU (50 runs, moyenne)
- Calcul FPS
- Tableau comparatif

### 4. Extraction Heatmaps
```python
# Pour modèle parallèle
heatmaps = model_parallel.get_attention_heatmaps(test_image)

# Visualiser:
# - heatmaps['backbone']['stage1-3']: channel_mask, spatial_mask, hybrid_mask
# - heatmaps['bifpn']['P3-5']: channel_mask, spatial_mask, hybrid_mask

# Afficher avec matplotlib (2x4 grid)
```

### 5. Entraînement
```bash
# Séquentiel
python train_eca_cbam.py --max_epoch 350

# Parallèle
python train_eca_cbam_parallel.py --max_epoch 350
```

### 6. Évaluation WIDERFace
```bash
# Générer prédictions
python test_widerface.py --network eca_cbam_parallel --trained_model weights/eca_cbam_parallel/Final.pth

# Calculer mAP
cd widerface_evaluate && python evaluation.py
```

### 7. Tableau Comparatif Final
| Schéma | Params | AP Easy | AP Medium | AP Hard | mAP | Latence |
|--------|--------|---------|-----------|---------|-----|---------|
| CBAM   | 488K   | 92.7%   | 90.7%     | 78.3%   | 87.2% | 4.5ms |
| Séq    | 476K   | 85.8%   | 83.9%     | 78.3%   | 82.7% | 4.1ms |
| Par    | 476K   | **XX%** | **XX%**   | **XX%** | **XX%** | 4.1ms |

**Cibles**: Easy 94.5%, Medium 92.5%, Hard 80.5%, mAP 89.2%

### 8. Analyse Convergence
- Visualisation TensorBoard logs
- Courbes loss train/val
- Stabilité entraînement

### 9. Conclusion
- Résumé résultats clés
- Recommandations architecture
- Prochaines étapes

## Usage

```bash
# Lancer notebook
cd notebooks
jupyter notebook 03_comparaison_sequentiel_parallele.ipynb
```

## Résultats Attendus

D'après Wang et al. 2024, l'architecture parallèle devrait montrer:
- **+6.5% mAP** vs séquentiel
- **+2.0% mAP** vs CBAM baseline
- Meilleure robustesse occlusion/petits visages
- Convergence plus rapide (~10 epochs)
- Heatmaps plus denses et précis

## Fichiers Générés

- `figures/attention_heatmaps_parallel.png`: Visualisations heatmaps
- `results/comparison_table.csv`: Tableau résultats
- `results/latency_benchmark.csv`: Mesures latence

## Références

- Notebook 01: CBAM baseline training
- Notebook 02: ECA-CBAM sequential training
- `docs/scientific/comparaison_sequentiel_parallele.md`: Documentation complète

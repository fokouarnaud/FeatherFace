# FeatherFace Workflow Scripts - V1 → V2_Ultra Architecture

Ce document présente tous les scripts de workflow pour l'architecture V1 (baseline) → V2_Ultra (revolutionary innovation).

## 🏃‍♂️ Scripts d'Entraînement

### V1 (Baseline)
```bash
# Entraînement complet V1 (489K paramètres)
python train_v1.py --training_dataset ./data/widerface/train/label.txt --network mobile0.25 --epochs 350

# Avec options avancées
python train_v1.py \
    --training_dataset ./data/widerface/train/label.txt \
    --network mobile0.25 \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 350 \
    --save_folder weights/
```


### V2 Ultra (Revolutionary Innovation - 248K parameters)
```bash
# Entraînement V2 Ultra avec innovations révolutionnaires
python train_v2_ultra.py --epochs 400 --teacher_model weights/mobilenet0.25_Final.pth

# Avec options complètes (revolutionary techniques)
python train_v2_ultra.py \
    --training_dataset ./data/widerface/train/label.txt \
    --teacher_model weights/mobilenet0.25_Final.pth \
    --save_folder weights/v2_ultra/ \
    --epochs 400 \
    --temperature 4.0 \
    --alpha 0.7 \
    --ultra_innovations \
    --zero_param_boost 0.05
```

## 🧪 Scripts de Test et Évaluation

### Test WIDERFace
```bash
# Test V1
python test_widerface.py -m weights/mobilenet0.25_Final.pth --network mobile0.25

# Test V2  
python test_widerface.py -m weights/v2/FeatherFaceV2_final.pth --network mobile0.25

# Évaluation simplifiée (recommandé)
python evaluate_widerface.py --model weights/mobilenet0.25_Final.pth --network mobile0.25 --show_results
python evaluate_widerface.py --model weights/v2/FeatherFaceV2_final.pth --version v2 --show_results
```

### Comparaison V1 vs V2
```bash
# Comparaison directe des performances
python test_v1_v2_comparison.py

# Analyse détaillée des paramètres
python validate_model.py --quick-check
```

## ✅ Scripts de Validation

### Validation des Modèles
```bash
# Validation V1
python validate_model.py --version v1

# Validation V2
python validate_model.py --version v2

# Validation V2 Ultra
python validate_model.py --version v2_ultra

# Vérification rapide de tous les modèles
python validate_model.py --quick-check
```

### Validation des Claims Révolutionnaires
```bash
# Validation complète des claims V2 Ultra
python validate_claims.py

# Avec benchmarks de performance
python validate_claims.py --benchmark

# Rapport détaillé
python validate_claims.py --detailed --save-report
```

### Validation V2 Ultra Spécifique
```bash
# Validation des innovations révolutionnaires
python validate_v2_ultra.py
```

## 📊 Scripts d'Analyse

### Analyse Comparative
```bash
# Comparaison détaillée V1/V2
python test_v1_v2_comparison.py

# Analyse d'architecture
python analysis/analyze_architecture.py
```

## 🎯 Workflows Recommandés

### 1. Premier Entraînement (V1)
```bash
# 1. Valider la configuration
python validate_model.py --version v1

# 2. Entraîner V1 (teacher model)
python train_v1.py --training_dataset ./data/widerface/train/label.txt --network mobile0.25

# 3. Tester V1
python evaluate_widerface.py --model weights/mobilenet0.25_Final.pth --show_results
```

### 2. Entraînement V2 avec Distillation
```bash
# 1. Vérifier le teacher model
python validate_model.py --version v1

# 2. Entraîner V2
python start_v2_training.py  # OU python train_v2.py --teacher_model weights/mobilenet0.25_Final.pth

# 3. Valider V2
python validate_model.py --version v2

# 4. Comparer V1 vs V2
python test_v1_v2_comparison.py
```

### 3. V2 Ultra (Révolutionnaire)
```bash
# 1. Entraîner V2 Ultra
python train_v2_ultra.py --epochs 400 --teacher_model weights/mobilenet0.25_Final.pth

# 2. Valider les claims révolutionnaires
python validate_claims.py --detailed

# 3. Validation spécifique V2 Ultra
python validate_v2_ultra.py
```

## 📁 Structure des Scripts

### Scripts Principaux (Root)
- `train_v1.py` - Entraînement FeatherFace V1 (489K params)
- `train_v2.py` - Entraînement FeatherFace V2 avec distillation (256K params)
- `train_v2_ultra.py` - Entraînement V2 Ultra révolutionnaire (248K params)
- `start_v2_training.py` - Démarrage rapide V2
- `test_widerface.py` - Test WIDERFace standard
- `test_v1_v2_comparison.py` - Comparaison V1/V2
- `evaluate_widerface.py` - Évaluation simplifiée WIDERFace
- `validate_model.py` - Validation uniforme des modèles
- `validate_claims.py` - Validation des claims révolutionnaires
- `validate_v2_ultra.py` - Validation spécifique V2 Ultra

### Scripts Utilitaires (scripts/)
- `scripts/training/` - Scripts d'entraînement originaux
- `scripts/validation/` - Outils de validation détaillés
- `scripts/detection/` - Scripts de détection
- `scripts/utils/` - Utilitaires divers

## 🔧 Configuration

### Variables d'Environnement (Optionnel)
```bash
export CUDA_VISIBLE_DEVICES=0  # GPU à utiliser
export FEATHERFACE_DATA_PATH=./data/widerface  # Chemin du dataset
export FEATHERFACE_WEIGHTS_PATH=./weights  # Chemin des poids
```

### Dépendances
```bash
# Installation du projet
pip install -e .

# Vérification des dépendances
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print('torchvision: OK')"
```

## 📝 Notes d'Utilisation

### Ordre d'Exécution Recommandé
1. **V1 d'abord** : Entraîner le modèle teacher (V1)
2. **V2 ensuite** : Utiliser V1 comme teacher pour V2
3. **V2 Ultra** : Evolution révolutionnaire de V2

### Performance Attendue
- **V1** : 87% mAP, 489K paramètres
- **V2** : 92%+ mAP, 256K paramètres (-47%)
- **V2 Ultra** : 90.5%+ mAP, 248K paramètres (-49%, 2.0x efficacité)

### Monitoring
- Les logs sont sauvegardés dans `weights/` ou `weights/v2/`
- Utiliser `validate_model.py --quick-check` pour un aperçu rapide
- Les métriques détaillées sont dans `validate_claims.py --detailed`

## 🚀 Exemples Rapides

```bash
# Workflow complet en une ligne (après avoir téléchargé les données)
python train_v1.py --training_dataset ./data/widerface/train/label.txt --network mobile0.25 && \
python train_v2.py --teacher_model weights/mobilenet0.25_Final.pth --epochs 400 && \
python validate_claims.py --detailed

# Validation rapide de tous les modèles
python validate_model.py --quick-check

# Comparaison de performance
python test_v1_v2_comparison.py
```

Cette organisation permet une exécution simple et directe de tous les workflows FeatherFace depuis la racine du projet ! 🎯
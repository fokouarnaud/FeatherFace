#!/usr/bin/env python3
"""
FeatherFace Setup Verification Script
Vérifie que tous les scripts et dépendances sont correctement installés

Usage: python check_setup.py
"""

import os
import sys
import importlib
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f"   {title}")
    print(f"{'='*60}")

def check_scripts():
    print_header("SCRIPTS DE WORKFLOW (ROOT)")
    
    scripts = [
        ("train_v1.py", "✓ Entraînement V1 (489K params)"),
        ("train_v2_ultra.py", "✓ Entraînement V2 Ultra (248K params)"),
        ("test_widerface.py", "✓ Test WIDERFace"),
        ("test_v1_v2_ultra_comparison.py", "✓ Comparaison V1/V2 Ultra"),
        ("evaluate_widerface.py", "✓ Évaluation simplifiée"),
        ("validate_model.py", "✓ Validation modèles"),
        ("validate_claims.py", "✓ Validation claims révolutionnaires"),
        ("validate_v2_ultra.py", "✓ Validation V2 Ultra"),
        ("help.py", "✓ Script d'aide")
    ]
    
    all_present = True
    for script, description in scripts:
        exists = Path(script).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {script:<25} {description}")
        if not exists:
            all_present = False
    
    return all_present

def check_dependencies():
    print_header("DÉPENDANCES PYTHON")
    
    deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("PIL", "Pillow"),
        ("tqdm", "TQDM"),
        ("pandas", "Pandas"),
        ("onnx", "ONNX (optionnel)"),
        ("onnxruntime", "ONNX Runtime (optionnel)")
    ]
    
    all_deps = True
    for module, name in deps:
        try:
            importlib.import_module(module)
            print(f"  ✅ {name}")
        except ImportError:
            if module in ["onnx", "onnxruntime"]:
                print(f"  ⚠️  {name} (optionnel)")
            else:
                print(f"  ❌ {name}")
                all_deps = False
    
    return all_deps

def check_project_structure():
    print_header("STRUCTURE DU PROJET")
    
    dirs = [
        ("data/", "Dossier données"),
        ("models/", "Modèles PyTorch"),
        ("layers/", "Modules et fonctions"),
        ("utils/", "Utilitaires"),
        ("weights/", "Poids des modèles"),
        ("notebooks/", "Jupyter notebooks"),
        ("docs/", "Documentation"),
        ("scripts/", "Scripts utilitaires"),
        ("widerface_evaluate/", "Outils d'évaluation")
    ]
    
    all_dirs = True
    for dir_path, description in dirs:
        exists = Path(dir_path).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {dir_path:<20} {description}")
        if not exists and dir_path not in ["weights/", "data/"]:
            all_dirs = False
    
    return all_dirs

def check_key_files():
    print_header("FICHIERS CLÉS")
    
    files = [
        ("data/config.py", "Configuration des modèles"),
        ("models/retinaface.py", "Modèle V1"),
        ("models/retinaface_v2.py", "Modèle V2"),
        ("models/net.py", "Backbone MobileNet"),
        ("CLAUDE.md", "Guide Claude"),
        ("WORKFLOW_SCRIPTS.md", "Guide des scripts"),
        ("pyproject.toml", "Configuration projet"),
        ("README.md", "Documentation principale")
    ]
    
    all_files = True
    for file_path, description in files:
        exists = Path(file_path).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path:<25} {description}")
        if not exists:
            all_files = False
    
    return all_files

def check_models():
    print_header("MODÈLES DISPONIBLES")
    
    try:
        # Test imports
        from models.retinaface import RetinaFace
        from data.config import cfg_mnet
        print("  ✅ V1 Model import successful")
        
        try:
            from models.retinaface_v2 import get_retinaface_v2
            from data.config import cfg_mnet_v2
            print("  ✅ V2 Model import successful")
        except ImportError:
            try:
                from models.retinaface_v2 import get_retinaface as get_retinaface_v2
                print("  ✅ V2 Model import successful (alternative)")
            except ImportError:
                print("  ❌ V2 Model import failed")
                return False
        
        try:
            from models.retinaface_v2_ultra import RetinaFaceV2Ultra
            print("  ✅ V2 Ultra Model import successful")
        except ImportError:
            print("  ⚠️  V2 Ultra Model import failed (optionnel)")
        
        # Test model creation
        v1_model = RetinaFace(cfg=cfg_mnet, phase='test')
        v1_params = sum(p.numel() for p in v1_model.parameters())
        print(f"  ✅ V1 Model: {v1_params:,} parameters")
        
        v2_model = get_retinaface_v2(cfg_mnet_v2, phase='test')
        v2_params = sum(p.numel() for p in v2_model.parameters())
        print(f"  ✅ V2 Model: {v2_params:,} parameters")
        
        reduction = (1 - v2_params / v1_params) * 100
        efficiency = v1_params / v2_params
        print(f"  📊 Compression: {reduction:.1f}% reduction, {efficiency:.2f}x efficiency")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        return False

def check_data():
    print_header("DONNÉES ET POIDS")
    
    # Check dataset
    dataset_files = [
        ("data/widerface/train/label.txt", "Training labels"),
        ("data/widerface/val/wider_val.txt", "Validation labels"),
        ("data/widerface/train/images/", "Training images"),
        ("data/widerface/val/images/", "Validation images")
    ]
    
    dataset_ok = True
    for file_path, description in dataset_files:
        exists = Path(file_path).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path:<35} {description}")
        if not exists:
            dataset_ok = False
    
    if not dataset_ok:
        print("  💡 Download WIDERFace: https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS")
    
    # Check weights
    weight_files = [
        ("weights/mobilenetV1X0.25_pretrain.tar", "Pretrained backbone"),
        ("weights/mobilenet0.25_Final.pth", "V1 trained model"),
        ("weights/v2/FeatherFaceV2_final.pth", "V2 trained model")
    ]
    
    weights_ok = 0
    for file_path, description in weight_files:
        exists = Path(file_path).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path:<35} {description}")
        if exists:
            weights_ok += 1
    
    return dataset_ok, weights_ok

def print_recommendations():
    print_header("RECOMMANDATIONS")
    
    print("1. 📋 WORKFLOW RECOMMANDÉ:")
    print("   python help.py                    # Afficher l'aide")
    print("   python validate_model.py --quick-check  # Vérifier les modèles")
    print("   python train_v1.py --training_dataset ./data/widerface/train/label.txt")
    print("   python train_v2.py --teacher_model weights/mobilenet0.25_Final.pth")
    
    print("\n2. 🔧 EN CAS DE PROBLÈME:")
    print("   pip install -e .                  # Réinstaller le projet")
    print("   python help.py issues             # Voir les solutions communes")
    print("   python scripts/setup/install_dependencies.py  # Installer dépendances")
    
    print("\n3. 📖 DOCUMENTATION:")
    print("   WORKFLOW_SCRIPTS.md               # Guide des scripts")
    print("   docs/ARCHITECTURE_V2_OPTIMIZED.md # Architecture V2")
    print("   notebooks/                        # Jupyter notebooks")

def main():
    print("🚀 " + "="*58)
    print("   FEATHERFACE SETUP VERIFICATION")
    print("   Vérification de l'installation et configuration")
    print("="*60)
    
    # Run all checks
    scripts_ok = check_scripts()
    deps_ok = check_dependencies() 
    structure_ok = check_project_structure()
    files_ok = check_key_files()
    models_ok = check_models()
    dataset_ok, weights_count = check_data()
    
    # Summary
    print_header("RÉSUMÉ")
    
    total_score = 0
    max_score = 6
    
    checks = [
        ("Scripts workflow", scripts_ok, 1),
        ("Dépendances Python", deps_ok, 1),
        ("Structure projet", structure_ok, 1),
        ("Fichiers clés", files_ok, 1),
        ("Modèles PyTorch", models_ok, 1),
        ("Dataset WIDERFace", dataset_ok, 1)
    ]
    
    for name, status, points in checks:
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")
        if status:
            total_score += points
    
    print(f"\n  📊 Poids disponibles: {weights_count}/3")
    print(f"  🏆 Score global: {total_score}/{max_score}")
    
    if total_score == max_score and weights_count >= 1:
        print(f"\n  🎉 INSTALLATION PARFAITE ! Prêt pour l'entraînement !")
    elif total_score >= 4:
        print(f"\n  ✅ Installation OK. Quelques éléments optionnels manquants.")
    else:
        print(f"\n  ⚠️  Installation incomplète. Consulter les recommandations.")
    
    print_recommendations()

if __name__ == "__main__":
    main()
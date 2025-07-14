#!/usr/bin/env python3
"""
Script de validation complète FeatherFace ODConv
Orchestrate tous les tests et validations pour migration ECA -> ODConv

Usage:
    python validate_odconv_complete.py [--quick] [--verbose] [--gpu-only]

Auteur: FeatherFace ODConv Team
Date: Juillet 2025
"""

import argparse
import sys
import os
import time
import torch
import subprocess
from pathlib import Path
from datetime import datetime

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from models.odconv import ODConv2d
from models.featherface_odconv import FeatherFaceODConv
from models.retinaface import RetinaFace
from data.config import cfg_odconv, cfg_mnet

def print_header():
    """Affiche header de validation"""
    print("🚀 FEATHERFACE ODCONV COMPLETE VALIDATION")
    print("="*60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*60)

def validate_file_structure():
    """Validation structure fichiers ODConv"""
    print("\n📁 VALIDATING FILE STRUCTURE")
    print("-" * 40)
    
    required_files = {
        'Implementation': [
            'models/odconv.py',
            'models/featherface_odconv.py',
            'train_odconv.py'
        ],
        'Configuration': [
            'data/config.py'
        ],
        'Documentation': [
            'docs/scientific/systematic_literature_review.md',
            'docs/scientific/odconv_mathematical_foundations.md',
            'docs/scientific/performance_analysis.md'
        ],
        'Diagrams': [
            'diagrams/odconv_architecture.png',
            'diagrams/attention_comparison.png'
        ],
        'Notebooks': [
            'notebooks/02_train_odconv_innovation.ipynb'
        ],
        'Tests': [
            'test_odconv_validation.py',
            'tests/test_odconv_unit.py'
        ]
    }
    
    all_files_exist = True
    
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file_path in files:
            exists = Path(file_path).exists()
            status = "✅" if exists else "❌"
            print(f"  {status} {file_path}")
            if not exists:
                all_files_exist = False
    
    # Vérifier suppression fichiers ECA
    eca_files_check = [
        'models/eca_net.py',
        'models/featherface_v2.py',
        'train_v2.py',
        'train_v2_simple.py',
        'notebooks/02_train_evaluate_featherface_v2.ipynb'
    ]
    
    print(f"\nECA Files Removal Check:")
    for file_path in eca_files_check:
        exists = Path(file_path).exists()
        status = "❌" if exists else "✅"  # Inversé: on veut qu'ils n'existent PAS
        action = "STILL EXISTS" if exists else "REMOVED"
        print(f"  {status} {file_path} ({action})")
        if exists:
            all_files_exist = False
    
    return all_files_exist

def validate_imports():
    """Validation imports ODConv"""
    print("\n🔌 VALIDATING IMPORTS")
    print("-" * 30)
    
    import_tests = [
        ('ODConv2d', 'from models.odconv import ODConv2d'),
        ('FeatherFaceODConv', 'from models.featherface_odconv import FeatherFaceODConv'),
        ('cfg_odconv', 'from data.config import cfg_odconv'),
    ]
    
    all_imports_ok = True
    
    for name, import_cmd in import_tests:
        try:
            exec(import_cmd)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            all_imports_ok = False
        except Exception as e:
            print(f"  ⚠️ {name}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def validate_model_architectures():
    """Validation architectures modèles"""
    print("\n🏗️ VALIDATING MODEL ARCHITECTURES")
    print("-" * 40)
    
    try:
        # Test ODConv module basique
        print("Testing ODConv2d module...")
        odconv = ODConv2d(in_channels=64, out_channels=128, kernel_size=3)
        x = torch.randn(1, 64, 32, 32)
        with torch.no_grad():
            output = odconv(x)
        print(f"  ✅ ODConv2d: {x.shape} -> {output.shape}")
        
        # Test FeatherFace ODConv complet
        print("Testing FeatherFace ODConv...")
        model_odconv = FeatherFaceODConv(cfg=cfg_odconv, phase='train')
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model_odconv(x)
        print(f"  ✅ FeatherFace ODConv: {len(outputs)} outputs")
        
        # Comparaison paramètres
        print("Comparing with CBAM baseline...")
        model_cbam = RetinaFace(cfg=cfg_mnet, phase='train')
        
        params_odconv = sum(p.numel() for p in model_odconv.parameters())
        params_cbam = sum(p.numel() for p in model_cbam.parameters())
        
        print(f"  📊 ODConv parameters: {params_odconv:,}")
        print(f"  📊 CBAM parameters: {params_cbam:,}")
        print(f"  📊 Difference: {params_odconv - params_cbam:+,} ({(params_odconv-params_cbam)/params_cbam*100:+.2f}%)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Architecture validation failed: {e}")
        return False

def run_unit_tests(verbose=False):
    """Exécute tests unitaires"""
    print("\n🧪 RUNNING UNIT TESTS")
    print("-" * 25)
    
    try:
        # Import et exécution tests unitaires
        from tests.test_odconv_unit import run_unit_tests
        success = run_unit_tests()
        
        if success:
            print("  ✅ All unit tests passed")
        else:
            print("  ❌ Some unit tests failed")
        
        return success
        
    except Exception as e:
        print(f"  ❌ Unit tests execution failed: {e}")
        return False

def run_integration_tests():
    """Exécute tests d'intégration"""
    print("\n🔗 RUNNING INTEGRATION TESTS")
    print("-" * 35)
    
    try:
        # Lancer script validation principal
        result = subprocess.run([
            sys.executable, 'test_odconv_validation.py'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("  ✅ Integration tests passed")
            if "VALIDATION SUCCESSFUL" in result.stdout:
                print("  🎉 ODConv ready for deployment")
            return True
        else:
            print("  ❌ Integration tests failed")
            print(f"  Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ⏰ Integration tests timed out")
        return False
    except Exception as e:
        print(f"  ❌ Integration tests execution failed: {e}")
        return False

def validate_documentation():
    """Validation documentation"""
    print("\n📚 VALIDATING DOCUMENTATION")
    print("-" * 35)
    
    doc_files = [
        'docs/scientific/systematic_literature_review.md',
        'docs/scientific/odconv_mathematical_foundations.md',
        'docs/scientific/performance_analysis.md'
    ]
    
    all_docs_valid = True
    
    for doc_file in doc_files:
        if Path(doc_file).exists():
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Vérifications basiques
                if len(content) < 100:
                    print(f"  ⚠️ {doc_file}: Too short")
                    all_docs_valid = False
                elif 'ODConv' not in content:
                    print(f"  ⚠️ {doc_file}: Missing ODConv references")
                    all_docs_valid = False
                else:
                    print(f"  ✅ {doc_file}")
                    
            except Exception as e:
                print(f"  ❌ {doc_file}: {e}")
                all_docs_valid = False
        else:
            print(f"  ❌ {doc_file}: Not found")
            all_docs_valid = False
    
    return all_docs_valid

def generate_validation_report(results):
    """Génère rapport de validation"""
    print("\n📊 VALIDATION REPORT")
    print("="*40)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"Total validations: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {total_tests - passed_tests} ❌")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    print(f"\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    # Statut final
    if passed_tests == total_tests:
        print(f"\n🎉 ALL VALIDATIONS PASSED!")
        print(f"🚀 FeatherFace ODConv migration is COMPLETE and READY!")
        return True
    else:
        print(f"\n⚠️ VALIDATION ISSUES DETECTED")
        print(f"❗ Review failed tests before proceeding")
        return False

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='FeatherFace ODConv Complete Validation')
    parser.add_argument('--quick', action='store_true', help='Quick validation (skip heavy tests)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--gpu-only', action='store_true', help='Only run on GPU')
    
    args = parser.parse_args()
    
    # Vérification GPU si requis
    if args.gpu_only and not torch.cuda.is_available():
        print("❌ GPU required but not available")
        sys.exit(1)
    
    # Header
    print_header()
    
    # Exécuter validations
    results = {}
    
    print(f"\n🔍 Starting validation suite...")
    if args.quick:
        print(f"⚡ Running in quick mode")
    
    # 1. Structure fichiers
    results['File Structure'] = validate_file_structure()
    
    # 2. Imports
    results['Imports'] = validate_imports()
    
    # 3. Architectures
    results['Model Architectures'] = validate_model_architectures()
    
    # 4. Documentation
    results['Documentation'] = validate_documentation()
    
    # 5. Tests unitaires (si pas quick)
    if not args.quick:
        results['Unit Tests'] = run_unit_tests(args.verbose)
        
        # 6. Tests intégration (si pas quick)
        results['Integration Tests'] = run_integration_tests()
    else:
        print(f"\n⚡ Skipping heavy tests in quick mode")
    
    # Rapport final
    success = generate_validation_report(results)
    
    # Exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
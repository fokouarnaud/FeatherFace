#!/usr/bin/env python3
"""
FeatherFace Nano Quick Validation Script
Validates that the Nano architecture is properly implemented and all V2 Ultra references are removed.

Usage: python validate_nano.py
"""

import os
import sys
from pathlib import Path
import glob

def print_header(title):
    print(f"\n{'='*60}")
    print(f"   {title}")
    print(f"{'='*60}")

def check_v2_ultra_cleanup():
    """Check that all V2 Ultra references have been removed"""
    print_header("V2 ULTRA CLEANUP VALIDATION")
    
    # Check for remaining V2 Ultra files
    v2_ultra_patterns = [
        "*v2_ultra*",
        "*V2_Ultra*",
        "*V2Ultra*",
        "*revolutionary*"
    ]
    
    remaining_files = []
    for pattern in v2_ultra_patterns:
        files = glob.glob(f"**/{pattern}", recursive=True)
        remaining_files.extend(files)
    
    if remaining_files:
        print("❌ Found remaining V2 Ultra files:")
        for file in remaining_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ No V2 Ultra files found - cleanup successful")
        return True

def check_nano_files():
    """Check that all Nano files are present"""
    print_header("NANO FILES VALIDATION")
    
    required_files = [
        "models/featherface_nano.py",
        "models/modules_nano.py", 
        "train_nano.py",
        "test_v1_nano_comparison.py",
        "notebooks/03_train_evaluate_featherface_nano.ipynb",
        "scripts/generate_nano_architecture.py",
        "docs/NANO_ARCHITECTURE.md",
        "docs/featherface_nano_architecture.png"
    ]
    
    all_present = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_present = False
    
    return all_present

def check_configuration():
    """Check that configuration is properly updated"""
    print_header("CONFIGURATION VALIDATION")
    
    try:
        from data.config import cfg_mnet, cfg_nano
        
        # Check cfg_nano exists
        if hasattr(sys.modules['data.config'], 'cfg_nano'):
            print("✅ cfg_nano configuration found")
        else:
            print("❌ cfg_nano configuration missing")
            return False
            
        # Check key Nano parameters
        nano_keys = ['cbam_reduction', 'ssh_groups', 'knowledge_distillation']
        missing_keys = [key for key in nano_keys if key not in cfg_nano]
        
        if missing_keys:
            print(f"❌ Missing Nano config keys: {missing_keys}")
            return False
        else:
            print("✅ Nano configuration parameters complete")
            
        return True
        
    except ImportError as e:
        print(f"❌ Configuration import error: {e}")
        return False

def check_model_imports():
    """Check that Nano models can be imported"""
    print_header("MODEL IMPORTS VALIDATION")
    
    models_to_check = [
        ("models.featherface_nano", "FeatherFaceNano"),
        ("models.modules_nano", "EfficientCBAM"),
        ("models.modules_nano", "EfficientBiFPN"),
        ("models.modules_nano", "GroupedSSH"),
        ("models.modules_nano", "ChannelShuffle")
    ]
    
    all_imports_ok = True
    for module_name, class_name in models_to_check:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except ImportError as e:
            print(f"❌ {module_name}.{class_name} - Import Error: {e}")
            all_imports_ok = False
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name} - Attribute Error: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def check_parameter_counts():
    """Check that parameter counts are correct"""
    print_header("PARAMETER COUNTS VALIDATION")
    
    try:
        from models.retinaface import RetinaFace
        from models.featherface_nano import FeatherFaceNano
        from data.config import cfg_mnet, cfg_nano
        
        # Count V1 parameters
        v1_model = RetinaFace(cfg=cfg_mnet, phase='test')
        v1_params = sum(p.numel() for p in v1_model.parameters())
        
        # Count Nano parameters  
        nano_model = FeatherFaceNano(cfg=cfg_nano, phase='test')
        nano_params = sum(p.numel() for p in nano_model.parameters())
        
        # Calculate reduction
        reduction_percent = ((v1_params - nano_params) / v1_params) * 100
        
        print(f"📊 V1 Parameters: {v1_params:,}")
        print(f"📊 Nano Parameters: {nano_params:,}")
        print(f"📉 Reduction: {reduction_percent:.1f}%")
        
        # Validate targets
        target_v1 = 487000  # ±5K tolerance
        target_nano = 344000  # ±5K tolerance
        target_reduction = 29.3  # ±2% tolerance
        
        v1_ok = abs(v1_params - target_v1) < 5000
        nano_ok = abs(nano_params - target_nano) < 5000
        reduction_ok = abs(reduction_percent - target_reduction) < 2.0
        
        print(f"✅ V1 target (~487K): {'✅' if v1_ok else '❌'}")
        print(f"✅ Nano target (~344K): {'✅' if nano_ok else '❌'}")
        print(f"✅ Reduction target (~29.3%): {'✅' if reduction_ok else '❌'}")
        
        return v1_ok and nano_ok and reduction_ok
        
    except Exception as e:
        print(f"❌ Parameter counting error: {e}")
        return False

def check_scientific_references():
    """Check that scientific references are properly documented"""
    print_header("SCIENTIFIC REFERENCES VALIDATION")
    
    required_references = [
        "Li et al. CVPR 2023",
        "Woo et al. ECCV 2018", 
        "Tan et al. CVPR 2020",
        "Howard et al. 2017"
    ]
    
    files_to_check = [
        "README.md",
        "docs/NANO_ARCHITECTURE.md",
        "CLAUDE.md"
    ]
    
    all_refs_found = True
    
    for ref in required_references:
        found_in_files = []
        for file_path in files_to_check:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if ref in content:
                        found_in_files.append(file_path)
        
        if found_in_files:
            print(f"✅ {ref}: Found in {', '.join(found_in_files)}")
        else:
            print(f"❌ {ref}: Not found in documentation")
            all_refs_found = False
    
    return all_refs_found

def run_comprehensive_validation():
    """Run all validation checks"""
    print("🔬 FeatherFace Nano Comprehensive Validation")
    print("=" * 60)
    
    checks = [
        ("V2 Ultra Cleanup", check_v2_ultra_cleanup),
        ("Nano Files Present", check_nano_files),
        ("Configuration", check_configuration),
        ("Model Imports", check_model_imports),
        ("Parameter Counts", check_parameter_counts),
        ("Scientific References", check_scientific_references)
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name}: Exception - {e}")
            results[check_name] = False
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    passed_checks = sum(results.values())
    total_checks = len(results)
    success_rate = (passed_checks / total_checks) * 100
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check_name}")
    
    print(f"\n📊 Overall Success Rate: {success_rate:.1f}% ({passed_checks}/{total_checks})")
    
    if success_rate >= 85:
        print("🎉 FeatherFace Nano validation SUCCESSFUL!")
        print("🔬 Project is ready for scientific deployment")
        return True
    else:
        print("⚠️  Some validation checks failed")
        print("🔧 Please address the issues above")
        return False

def main():
    """Main validation entry point"""
    
    # Change to project directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Add to Python path
    sys.path.insert(0, str(script_dir))
    
    success = run_comprehensive_validation()
    
    if success:
        print(f"\n✨ FeatherFace Nano is scientifically validated and ready!")
        sys.exit(0)
    else:
        print(f"\n🔧 Please fix validation issues before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main()
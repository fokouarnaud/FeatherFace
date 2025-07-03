#!/usr/bin/env python3
"""
FeatherFace Helper Script
Affiche l'aide et les commandes disponibles pour FeatherFace V1/V2/V2Ultra

Usage: python help.py [command]
"""

import sys
import os
from pathlib import Path

def print_banner():
    print("🚀 " + "="*60)
    print("   FEATHERFACE - EFFICIENT FACE DETECTION TOOLKIT")
    print("   V1 (489K) → V2 Ultra (248K) - Revolutionary 2.0x Efficiency")
    print("="*63)

def print_quick_start():
    print("\n🎯 QUICK START")
    print("-" * 30)
    print("1. Train V1 (Teacher):    python train_v1.py --training_dataset ./data/widerface/train/label.txt")
    print("2. Train V2 Ultra:       python train_v2_ultra.py --teacher_model weights/mobilenet0.25_Final.pth")
    print("3. Validate Results:      python validate_model.py --quick-check")
    print("4. Compare Performance:   python test_v1_v2_ultra_comparison.py")

def print_training_commands():
    print("\n🏃 TRAINING COMMANDS")
    print("-" * 30)
    print("V1 (Baseline - 489K params):")
    print("  python train_v1.py --training_dataset ./data/widerface/train/label.txt --network mobile0.25")
    print("")
    print("V2 Ultra (Revolutionary - 248K params):")
    print("  python train_v2_ultra.py --epochs 400 --teacher_model weights/mobilenet0.25_Final.pth")
    print("  python validate_v2_ultra.py  # Validate revolutionary innovations")

def print_testing_commands():
    print("\n🧪 TESTING & EVALUATION COMMANDS")
    print("-" * 30)
    print("Test on WIDERFace:")
    print("  python test_widerface.py -m weights/mobilenet0.25_Final.pth --network mobile0.25")
    print("  python evaluate_widerface.py --model weights/mobilenet0.25_Final.pth --show_results")
    print("")
    print("Compare V1 vs V2 Ultra:")
    print("  python test_v1_v2_ultra_comparison.py")
    print("")
    print("Validate Models:")
    print("  python validate_model.py --version v1")
    print("  python validate_model.py --version v2_ultra") 
    print("  python validate_model.py --quick-check  # All models")

def print_validation_commands():
    print("\n✅ VALIDATION COMMANDS")
    print("-" * 30)
    print("Model Validation:")
    print("  python validate_model.py --version v1|v2_ultra")
    print("  python validate_model.py --quick-check")
    print("")
    print("Revolutionary Claims (V2 Ultra):")
    print("  python validate_claims.py")
    print("  python validate_claims.py --detailed --benchmark")
    print("  python validate_v2_ultra.py")

def print_notebooks():
    print("\n📓 JUPYTER NOTEBOOKS")
    print("-" * 30)
    print("notebooks/01_train_evaluate_featherface.ipynb     - V1 Complete workflow")
    print("notebooks/03_train_evaluate_featherface_v2_ultra.ipynb  - V2 Ultra Complete workflow")

def print_architecture_info():
    print("\n🏗️ ARCHITECTURE SUMMARY")
    print("-" * 30)
    print("V1 Foundation (487K params):")
    print("  • MobileNetV1-0.25 + BiFPN + CBAM + DCN")
    print("  • 87% mAP on WIDERFace")
    print("  • Teacher model for knowledge distillation")
    print("")
    print("V2 Ultra Revolution (248K params, -49%):")
    print("  • 5 zero-parameter innovations")
    print("  • 90.5%+ mAP with 2.0x parameter efficiency")
    print("  • Revolutionary 'Intelligence > Capacity' paradigm")
    print("  • Direct evolution from V1 baseline")

def print_files_status():
    print("\n📁 PROJECT STATUS")
    print("-" * 30)
    
    # Check key files
    files_to_check = [
        ("train_v1.py", "V1 Training Script"),
        ("train_v2_ultra.py", "V2 Ultra Training Script"),
        ("test_widerface.py", "WIDERFace Test Script"),
        ("validate_model.py", "Model Validation Script"),
        ("validate_claims.py", "Claims Validation Script"),
        ("data/widerface/train/label.txt", "Training Dataset"),
        ("weights/mobilenetV1X0.25_pretrain.tar", "Pretrained Weights"),
        ("weights/mobilenet0.25_Final.pth", "V1 Model (Teacher)")
    ]
    
    for file_path, description in files_to_check:
        exists = Path(file_path).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {description}: {file_path}")

def print_common_issues():
    print("\n🔧 COMMON ISSUES & SOLUTIONS")
    print("-" * 30)
    print("1. SSH Constraint Error (V1):")
    print("   This is expected for V1 models (uses DCN, not SSH_Grouped)")
    print("   V1 models don't need divisibility by 4 constraint")
    print("")
    print("2. Teacher Model Not Found:")
    print("   Train V1 first: python train_v1.py --training_dataset ./data/widerface/train/label.txt")
    print("")
    print("3. Dataset Missing:")
    print("   Download WIDERFace: https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS")
    print("   Extract to: data/widerface/")
    print("")
    print("4. CUDA Out of Memory:")
    print("   Reduce batch_size: --batch_size 16 or --batch_size 8")
    print("")
    print("5. Import Errors:")
    print("   Install project: pip install -e .")

def print_performance_targets():
    print("\n📊 PERFORMANCE TARGETS")
    print("-" * 30)
    print("Model      | Params  | Reduction | WIDERFace Easy | Efficiency")
    print("-----------|---------|-----------|----------------|----------")
    print("V1         | 489K    | 0%        | 87.0%          | Baseline")
    print("V2 Ultra   | 248K    | -49.1%    | 90.5%+         | 2.0x revolutionary")

def show_help(command=None):
    print_banner()
    
    if command is None:
        print_quick_start()
        print_training_commands()
        print_testing_commands()
        print_validation_commands()
        print_notebooks()
        print_architecture_info()
        print_files_status()
        print_performance_targets()
        print_common_issues()
        
        print(f"\n💡 TIP: Run 'python help.py <command>' for specific help")
        print(f"Available commands: train, test, validate, notebook, files, issues")
        
    elif command == "train":
        print_training_commands()
    elif command == "test":
        print_testing_commands()
    elif command == "validate":
        print_validation_commands()
    elif command == "notebook":
        print_notebooks()
    elif command == "files":
        print_files_status()
    elif command == "issues":
        print_common_issues()
    elif command == "arch":
        print_architecture_info()
    elif command == "perf":
        print_performance_targets()
    else:
        print(f"\n❌ Unknown command: {command}")
        print("Available commands: train, test, validate, notebook, files, issues, arch, perf")

def main():
    command = sys.argv[1] if len(sys.argv) > 1 else None
    show_help(command)

if __name__ == "__main__":
    main()
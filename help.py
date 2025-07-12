#!/usr/bin/env python3
"""
FeatherFace Helper Script
Affiche l'aide et les commandes disponibles pour FeatherFace V1/Nano

Usage: python help.py [command]
"""

import sys
import os
from pathlib import Path

def print_banner():
    print("🚀 " + "="*60)
    print("   FEATHERFACE - EFFICIENT FACE DETECTION TOOLKIT")
    print("   V1 (489K) → V2 (493K) → Nano (344K) - Scientific Innovation")
    print("="*63)

def print_quick_start():
    print("\n🎯 QUICK START")
    print("-" * 30)
    print("1. Train V1 (Baseline):   python train_v1.py --training_dataset ./data/widerface/train/label.txt")
    print("2. Train V2:              python train_v2.py --training_dataset ./data/widerface/train/label.txt")
    print("3. Train Nano:            python train_nano.py --teacher_model weights/mobilenet0.25_Final.pth")
    print("4. Validate Results:      python validate_model.py --quick-check")
    print("5. Compare Performance:   python test_v1_v2_comparison.py")

def print_training_commands():
    print("\n🏃 TRAINING COMMANDS")
    print("-" * 30)
    print("V1 (Baseline - 489K params):")
    print("  python train_v1.py --training_dataset ./data/widerface/train/label.txt --network mobile0.25")
    print("")
    print("V2 (Coordinate Attention - 493K params):")
    print("  python train_v2.py --training_dataset ./data/widerface/train/label.txt")
    print("  python train_v2.py --resume_net weights/v2/featherface_v2_epoch_100.pth")
    print("")
    print("Nano (Ultra-Efficient - 344K params):")
    print("  python train_nano.py --epochs 400 --teacher_model weights/mobilenet0.25_Final.pth")
    print("  python validate_nano.py  # Validate scientific optimizations")

def print_testing_commands():
    print("\n🧪 TESTING & EVALUATION COMMANDS")
    print("-" * 30)
    print("Test on WIDERFace:")
    print("  # V1 Baseline (487K parameters)")
    print("  python test_widerface.py -m weights/mobilenet0.25_Final.pth --network mobile0.25")
    print("  # Nano Ultra-Efficient (344K parameters, 29.3% reduction)")
    print("  python test_widerface.py -m weights/nano/nano_final.pth --network nano")
    print("  python evaluate_widerface.py --model weights/mobilenet0.25_Final.pth --show_results")
    print("")
    print("Compare V1 vs Nano:")
    print("  python test_v1_nano_comparison.py")
    print("")
    print("Validate Models:")
    print("  python validate_model.py --version v1")
    print("  python validate_model.py --version nano") 
    print("  python validate_model.py --quick-check  # All models")

def print_validation_commands():
    print("\n✅ VALIDATION COMMANDS")
    print("-" * 30)
    print("Model Validation:")
    print("  python validate_model.py --version v1|nano")
    print("  python validate_model.py --quick-check")
    print("")
    print("Scientific Claims (Nano):")
    print("  python validate_claims.py")
    print("  python validate_claims.py --detailed --benchmark")
    print("  python validate_nano.py")

def print_notebooks():
    print("\n📓 JUPYTER NOTEBOOKS")
    print("-" * 30)
    print("notebooks/01_train_evaluate_featherface.ipynb        - V1 Complete workflow")
    print("notebooks/03_train_evaluate_featherface_nano.ipynb   - Nano Complete workflow")

def print_architecture_info():
    print("\n🏗️ ARCHITECTURE SUMMARY")
    print("-" * 30)
    print("V1 Foundation (487K params):")
    print("  • MobileNetV1-0.25 + BiFPN + CBAM + DCN")
    print("  • 87% mAP on WIDERFace")
    print("  • Teacher model for knowledge distillation")
    print("")
    print("Nano Ultra-Efficient (344K params, -29.3%):")
    print("  • Scientifically justified optimizations")
    print("  • Competitive mAP with 29.3% parameter reduction")
    print("  • Research-backed efficiency techniques")
    print("  • Knowledge distillation from V1 baseline")

def print_scientific_foundation():
    print("\n🔬 SCIENTIFIC FOUNDATION")
    print("-" * 30)
    print("Nano is based on 4 verified research publications:")
    print("  1. ✅ Knowledge Distillation (Li et al. CVPR 2023)")
    print("  2. ✅ CBAM Attention (Woo et al. ECCV 2018)")
    print("  3. ✅ BiFPN Architecture (Tan et al. CVPR 2020)")
    print("  4. ✅ MobileNet Backbone (Howard et al. 2017)")
    print("")
    print("Key Techniques:")
    print("  • CBAM: Convolutional Block Attention (Woo et al. ECCV 2018)")
    print("  • BiFPN: Bidirectional Feature Pyramid (Tan et al. CVPR 2020)")
    print("  • SSH: Single Stage Headless Face Detector (Najibi et al. ICCV 2017)")
    print("  • Channel Shuffle: Parameter-free information mixing")

def print_files_status():
    print("\n📁 PROJECT STATUS")
    print("-" * 30)
    
    # Check key files
    files_to_check = [
        ("train_v1.py", "V1 Training Script"),
        ("train_nano.py", "Nano Training Script"),
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
    print("1. 🚨 TEACHER MODEL STATE DICT ERROR (CRITICAL for V2/Nano):")
    print("   Error: 'Unexpected key(s) in state_dict: total_ops, total_params'")
    print("   Cause: V1 teacher model saved with thop profiling metadata")
    print("   Solution: Already fixed in current version - git pull origin main")
    print("   Manual fix: Filter profiling keys when loading state dict")
    print("")
    print("2. SSH Constraint Error (V1):")
    print("   This is expected for V1 models (uses DCN, not SSH_Grouped)")
    print("   V1 models don't need divisibility by 4 constraint")
    print("")
    print("3. Teacher Model Not Found:")
    print("   Train V1 first: python train_v1.py --training_dataset ./data/widerface/train/label.txt")
    print("")
    print("4. Dataset Missing:")
    print("   Download WIDERFace: https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS")
    print("   Extract to: data/widerface/")
    print("")
    print("5. CUDA Out of Memory:")
    print("   Reduce batch_size: --batch_size 16 or --batch_size 8")
    print("")
    print("6. Import Errors:")
    print("   Install project: pip install -e .")
    print("")
    print("7. Nano Model Loading:")
    print("   Ensure FeatherFaceNano model is properly imported")
    print("   Check models/featherface_nano.py exists")
    print("")
    print("8. V2 Knowledge Distillation Issues:")
    print("   Ensure teacher model loads correctly (see issue #1)")
    print("   Check: python test_v2_training.py")

def print_performance_targets():
    print("\n📊 PERFORMANCE TARGETS")
    print("-" * 30)
    print("Model      | Params  | Reduction | WIDERFace Easy | Foundation")
    print("-----------|---------|-----------|----------------|----------")
    print("V1         | 487K    | 0%        | 87.0%          | Baseline")
    print("Nano       | 344K    | -29.3%    | Competitive    | Scientific")

def show_help(command=None):
    print_banner()
    
    if command is None:
        print_quick_start()
        print_training_commands()
        print_testing_commands()
        print_validation_commands()
        print_notebooks()
        print_architecture_info()
        print_scientific_foundation()
        print_files_status()
        print_performance_targets()
        print_common_issues()
        
        print(f"\n💡 TIP: Run 'python help.py <command>' for specific help")
        print(f"Available commands: train, test, validate, notebook, files, issues, arch, science")
        
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
    elif command == "science":
        print_scientific_foundation()
    elif command == "perf":
        print_performance_targets()
    else:
        print(f"\n❌ Unknown command: {command}")
        print("Available commands: train, test, validate, notebook, files, issues, arch, science, perf")

def main():
    command = sys.argv[1] if len(sys.argv) > 1 else None
    show_help(command)

if __name__ == "__main__":
    main()
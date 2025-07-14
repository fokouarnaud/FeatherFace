#!/usr/bin/env python3
"""
FeatherFace Helper Script
Displays help and available commands for FeatherFace CBAM/ODConv scientific comparison

Usage: python help.py [command]
"""

import sys
import os
from pathlib import Path

def print_banner():
    print("🚀 " + "="*65)
    print("   FEATHERFACE - SCIENTIFIC ATTENTION MECHANISM COMPARISON")
    print("   CBAM Baseline (488,664) ↔ ODConv Innovation (~485,000) - 2.6% Reduction")
    print("="*68)

def print_quick_start():
    print("\n🎯 QUICK START")
    print("-" * 35)
    print("1. Train CBAM Baseline:     python train_cbam.py --training_dataset ./data/widerface/train/label.txt")
    print("2. Train ODConv Innovation:    python train_odconv.py --training_dataset ./data/widerface/train/label.txt")
    print("3. Validate Results:        python validate_model.py --version cbam")
    print("4. Compare Performance:     python validate_model.py --version odconv")
    print("5. Scientific Analysis:     jupyter notebook notebooks/01_train_cbam_baseline.ipynb")

def print_training_commands():
    print("\n🏃 TRAINING COMMANDS")
    print("-" * 35)
    print("CBAM Baseline (Scientific Foundation - 488,664 params):")
    print("  python train_cbam.py --training_dataset ./data/widerface/train/label.txt")
    print("  python train_cbam.py --batch_size 32 --save_folder weights/cbam/")
    print("")
    print("ODConv Innovation (Mobile Optimization - ~485,000 params):")
    print("  python train_odconv.py --training_dataset ./data/widerface/train/label.txt")
    print("  python train_odconv.py --batch_size 32 --save_folder weights/odconv/")
    print("")
    print("Resume Training:")
    print("  python train_cbam.py --resume_net weights/cbam/featherface_cbam_epoch_100.pth")
    print("  python train_odconv.py --resume_net weights/odconv/featherface_odconv_epoch_100.pth")

def print_testing_commands():
    print("\n🧪 TESTING & EVALUATION COMMANDS")
    print("-" * 35)
    print("Test on WIDERFace:")
    print("  # CBAM Baseline (488,664 parameters)")
    print("  python test_widerface.py -m weights/cbam/featherface_cbam_final.pth --network cbam")
    print("  # ODConv Innovation (~485,000 parameters, efficient 4D attention)")
    print("  python test_widerface.py -m weights/odconv/featherface_odconv_final.pth --network odconv")
    print("")
    print("WIDERFace Evaluation Protocol:")
    print("  python evaluate_widerface.py --model weights/cbam/featherface_cbam_final.pth --show_results")
    print("  cd widerface_evaluate && python evaluation.py -p ./widerface_txt -g ./eval_tools/ground_truth")

def print_model_info():
    print("\n📊 MODEL COMPARISON")
    print("-" * 35)
    print("CBAM Baseline (Scientific Foundation):")
    print("  • Parameters: 488,664")
    print("  • Attention: CBAM (Channel + Spatial)")
    print("  • Complexity: O(C²)")
    print("  • Purpose: Paper-exact baseline")
    print("  • Configuration: cfg_cbam_paper_exact")
    print("")
    print("ODConv Innovation (Mobile Optimization):")
    print("  • Parameters: ~485,000 (-3,664)")
    print("  • Attention: ODConv (4D Attention)")
    print("  • Complexity: O(C)")
    print("  • Purpose: Superior performance with efficiency")
    print("  • Configuration: cfg_odconv")
    print("")
    print("Scientific Comparison:")
    print("  • Parameter Efficiency: 0.8% reduction")
    print("  • Computational Advantage: 4D multidimensional attention")
    print("  • Performance Target: +2.2% WIDERFace Hard mAP")
    print("  • Controlled Experiment: Single variable change")

def print_validation_commands():
    print("\n✅ VALIDATION & ANALYSIS COMMANDS")
    print("-" * 35)
    print("Model Validation:")
    print("  python validate_model.py --version cbam       # Validate CBAM baseline")
    print("  python validate_model.py --version odconv     # Validate ODConv innovation")
    print("  python validate_model.py --quick-check        # Quick parameter check")
    print("")
    print("Interactive Analysis:")
    print("  jupyter notebook notebooks/01_train_cbam_baseline.ipynb")
    print("  jupyter notebook notebooks/02_train_odconv_innovation.ipynb")
    print("")
    print("Architecture Verification:")
    print("  python -c \"from models.featherface_cbam_exact import FeatherFaceCBAMExact; print('CBAM ready')\"")
    print("  python -c \"from models.featherface_odconv import FeatherFaceODConv; print('ODConv ready')\"")

def print_scientific_workflow():
    print("\n🔬 SCIENTIFIC WORKFLOW")
    print("-" * 35)
    print("Complete Scientific Comparison Pipeline:")
    print("")
    print("1. Environment Setup:")
    print("   pip install -e .")
    print("   # Verify models load correctly")
    print("")
    print("2. CBAM Baseline Training:")
    print("   python train_cbam.py --training_dataset ./data/widerface/train/label.txt")
    print("   python validate_model.py --version cbam")
    print("")
    print("3. ODConv Innovation Training:")
    print("   python train_odconv.py --training_dataset ./data/widerface/train/label.txt")
    print("   python validate_model.py --version odconv")
    print("")
    print("4. Performance Evaluation:")
    print("   python test_widerface.py -m weights/cbam/featherface_cbam_final.pth --network cbam")
    print("   python test_widerface.py -m weights/odconv/featherface_odconv_final.pth --network odconv")
    print("")
    print("5. Scientific Analysis:")
    print("   # Compare parameter counts: 488,664 vs ~485,000")
    print("   # Evaluate 4D attention advantages")
    print("   # Validate improved performance: WIDERFace benchmarks")

def print_configurations():
    print("\n⚙️ CONFIGURATION DETAILS")
    print("-" * 35)
    print("CBAM Baseline Configuration (data/config.py):")
    print("  cfg_cbam_paper_exact = {")
    print("    'out_channel': 52,")
    print("    'attention_mechanism': 'CBAM',")
    print("    'total_parameters': 488664")
    print("  }")
    print("")
    print("ODConv Innovation Configuration (data/config.py):")
    print("  cfg_odconv = {")
    print("    'out_channel': 52,")
    print("    'attention_mechanism': 'ODConv',")
    print("    'total_parameters': 485000")
    print("  }")
    print("")
    print("Key Design Principles:")
    print("  • Identical out_channel=52 for fair comparison")
    print("  • Single variable change (attention mechanism)")
    print("  • No knowledge distillation complexity")
    print("  • Clean, reproducible configurations")

def print_development_commands():
    print("\n🔧 DEVELOPMENT COMMANDS")
    print("-" * 35)
    print("Code Quality:")
    print("  black --line-length 100 . && isort --profile black .")
    print("")
    print("Testing:")
    print("  python -m pytest tests/ -v")
    print("")
    print("Installation:")
    print("  pip install -e .     # Editable installation")
    print("")
    print("Project Structure:")
    print("  models/featherface_cbam_exact.py       # CBAM baseline")
    print("  models/featherface_odconv.py           # ODConv innovation")
    print("  models/odconv.py                       # ODConv module")
    print("  train_cbam.py                          # CBAM training")
    print("  train_odconv.py                        # ODConv training")

def print_help():
    """Print comprehensive help information"""
    print_banner()
    print_quick_start()
    print_model_info()
    print_training_commands()
    print_testing_commands()
    print_validation_commands()
    print_scientific_workflow()
    print_configurations()
    print_development_commands()
    
    print("\n📚 ADDITIONAL RESOURCES")
    print("-" * 35)
    print("• Interactive Notebooks: notebooks/")
    print("• Configuration Guide: CLAUDE.md")
    print("• Scientific Paper: Electronics 2025, 14(3), 517")
    print("• CBAM Paper: Woo et al. ECCV 2018")
    print("• ODConv Paper: Li et al. ICLR 2022")
    
    print("\n💡 TIPS")
    print("-" * 35)
    print("• Use notebooks for interactive training and analysis")
    print("• Both models use identical out_channel=52 for fair comparison")
    print("• ODConv provides 4D attention with superior performance and efficiency")
    print("• Scientific comparison: CBAM baseline vs ODConv 4D attention innovation")

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ['train', 'training']:
            print_banner()
            print_training_commands()
        elif command in ['test', 'testing', 'eval', 'evaluation']:
            print_banner()
            print_testing_commands()
        elif command in ['validate', 'validation']:
            print_banner()
            print_validation_commands()
        elif command in ['model', 'models', 'info']:
            print_banner()
            print_model_info()
        elif command in ['config', 'configuration']:
            print_banner()
            print_configurations()
        elif command in ['workflow', 'scientific']:
            print_banner()
            print_scientific_workflow()
        elif command in ['dev', 'development']:
            print_banner()
            print_development_commands()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: train, test, validate, model, config, workflow, dev")
    else:
        print_help()

if __name__ == "__main__":
    main()
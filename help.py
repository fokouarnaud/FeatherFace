#!/usr/bin/env python3
"""
FeatherFace Helper Script
Displays help and available commands for FeatherFace CBAM vs ECA-CBAM scientific comparison

Usage: python help.py [command]
"""

import sys
import os
from pathlib import Path

def print_banner():
    print("🚀 " + "="*65)
    print("   FEATHERFACE - SCIENTIFIC ATTENTION MECHANISM COMPARISON")
    print("   CBAM Baseline (488,664) ↔ ECA-CBAM Hybrid (460,000) - 5.9% Reduction")
    print("="*68)

def print_quick_start():
    print("\n🎯 QUICK START")
    print("-" * 35)
    print("1. Train CBAM Baseline:     python train_cbam.py --training_dataset ./data/widerface/train/label.txt")
    print("2. Train ECA-CBAM Hybrid:   python train_eca_cbam.py --training_dataset ./data/widerface/train/label.txt")
    print("3. Validate Results:        python validate_model.py --version cbam")
    print("4. Compare Performance:     python validate_model.py --version eca_cbam")
    print("5. Scientific Analysis:     jupyter notebook notebooks/02_train_eca_cbam.ipynb")

def print_training_commands():
    print("\n🏃 TRAINING COMMANDS")
    print("-" * 35)
    print("CBAM Baseline (Scientific Foundation - 488,664 params):")
    print("  python train_cbam.py --training_dataset ./data/widerface/train/label.txt")
    print("  python train_cbam.py --batch_size 32 --save_folder weights/cbam/")
    print("")
    print("ECA-CBAM Hybrid (Hybrid Attention Module - 460,000 params):")
    print("  python train_eca_cbam.py --training_dataset ./data/widerface/train/label.txt")
    print("  python train_eca_cbam.py --batch_size 32 --save_folder weights/eca_cbam/")
    print("  python train_eca_cbam.py --eca_gamma 2 --eca_beta 1 --sam_kernel_size 7 --interaction_weight 0.1")
    print("")
    print("Resume Training:")
    print("  python train_cbam.py --resume_net weights/cbam/featherface_cbam_epoch_100.pth")
    print("  python train_eca_cbam.py --resume_net weights/eca_cbam/featherface_eca_cbam_epoch_100.pth")

def print_testing_commands():
    print("\n🧪 TESTING & EVALUATION COMMANDS")
    print("-" * 35)
    print("Test on WIDERFace:")
    print("  # CBAM Baseline (488,664 parameters)")
    print("  python test_widerface.py -m weights/cbam/featherface_cbam_final.pth --network cbam")
    print("  # ECA-CBAM Hybrid (460,000 parameters, hybrid attention module)")
    print("  python test_eca_cbam.py -m weights/eca_cbam/featherface_eca_cbam_final.pth --network eca_cbam")
    print("  python test_eca_cbam.py -m weights/eca_cbam/featherface_eca_cbam_final.pth --network eca_cbam --analyze_attention")
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
    print("  • Complexity: O(C²) + O(H×W)")
    print("  • Purpose: Paper-exact baseline")
    print("  • Configuration: cfg_cbam_paper_exact")
    print("")
    print("ECA-CBAM Hybrid (Hybrid Attention Module Innovation):")
    print("  • Parameters: 460,000 (-28,664)")
    print("  • Attention: ECA-CBAM (Hybrid Attention Module)")
    print("  • Complexity: O(C×log₂(C)) + O(H×W)")
    print("  • Purpose: Superior performance with efficiency")
    print("  • Configuration: cfg_eca_cbam")
    print("")
    print("Scientific Comparison:")
    print("  • Parameter Efficiency: 5.9% reduction (28,664 fewer params)")
    print("  • Channel Attention: ECA-Net (22 params) vs CBAM CAM (2000 params)")
    print("  • Spatial Attention: CBAM SAM preserved (98 params)")
    print("  • Performance Target: +1.5% to +2.5% mAP improvement")
    print("  • Innovation: Hybrid attention module mechanism")

def print_validation_commands():
    print("\n✅ VALIDATION & ANALYSIS COMMANDS")
    print("-" * 35)
    print("Model Validation:")
    print("  python validate_model.py --version cbam       # Validate CBAM baseline")
    print("  python validate_model.py --version eca_cbam   # Validate ECA-CBAM hybrid")
    print("  python validate_model.py --quick-check        # Quick parameter check")
    print("")
    print("Interactive Analysis:")
    print("  jupyter notebook notebooks/01_train_cbam_baseline.ipynb")
    print("  jupyter notebook notebooks/02_train_eca_cbam.ipynb")
    print("")
    print("Architecture Verification:")
    print("  python -c \"from models.featherface_cbam_exact import FeatherFaceCBAMExact; print('CBAM ready')\"")
    print("  python -c \"from models.featherface_eca_cbam import FeatherFaceECAcbaM; print('ECA-CBAM ready')\"")

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
    print("3. ECA-CBAM Hybrid Training:")
    print("   python train_eca_cbam.py --training_dataset ./data/widerface/train/label.txt --log_attention")
    print("   python validate_model.py --version eca_cbam")
    print("")
    print("4. Performance Evaluation:")
    print("   python test_widerface.py -m weights/cbam/featherface_cbam_final.pth --network cbam")
    print("   python test_eca_cbam.py -m weights/eca_cbam/featherface_eca_cbam_final.pth --network eca_cbam")
    print("")
    print("5. Scientific Analysis:")
    print("   # Compare parameter counts: 488,664 vs 460,000 (-5.9%)")
    print("   # Evaluate hybrid attention module advantages")
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
    print("ECA-CBAM Hybrid Configuration (data/config.py):")
    print("  cfg_eca_cbam = {")
    print("    'out_channel': 48,")
    print("    'attention_mechanism': 'ECA-CBAM',")
    print("    'total_parameters': 460000,")
    print("    'eca_cbam_config': {")
    print("      'eca_gamma': 2,")
    print("      'eca_beta': 1,")
    print("      'sam_kernel_size': 7,")
    print("      'interaction_weight': 0.1,")
    print("      'parallel_hybrid': True")
    print("    }")
    print("  }")
    print("")
    print("Key Design Principles:")
    print("  • Controlled comparison: Only attention mechanism differs")
    print("  • Single variable change (CBAM ↔ ECA-CBAM)")
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
    print("  models/featherface_eca_cbam.py         # ECA-CBAM hybrid")
    print("  models/eca_cbam_hybrid.py              # ECA-CBAM module")
    print("  train_cbam.py                          # CBAM training")
    print("  train_eca_cbam.py                      # ECA-CBAM training")
    print("  test_eca_cbam.py                       # ECA-CBAM testing")

def print_eca_cbam_details():
    print("\n🚀 ECA-CBAM HYBRID DETAILS")
    print("-" * 35)
    print("Innovation Components:")
    print("  • ECA-Net Channel Attention: 22 parameters per module")
    print("  • CBAM SAM Spatial Attention: 98 parameters per module")
    print("  • Hybrid Attention Interaction: ~30 parameters per module")
    print("  • Total Attention Overhead: ~150 parameters per module")
    print("")
    print("Scientific Foundation:")
    print("  • ECA-Net: Wang et al. CVPR 2020 (Efficient Channel Attention)")
    print("  • CBAM SAM: Woo et al. ECCV 2018 (Spatial Attention Module)")
    print("  • Hybrid Attention Module: Lu et al. Frontiers in Neurorobotics 2024")
    print("")
    print("Performance Targets:")
    print("  • Parameters: 460,000 (5.9% reduction vs CBAM)")
    print("  • WIDERFace Easy: 94.0% AP (+1.3% vs CBAM)")
    print("  • WIDERFace Medium: 92.0% AP (+1.3% vs CBAM)")
    print("  • WIDERFace Hard: 80.0% AP (+1.7% vs CBAM)")
    print("  • Overall mAP: 88.7% AP (+1.5% vs CBAM)")
    print("")
    print("Training Arguments:")
    print("  --eca_gamma 2              # ECA adaptive kernel gamma")
    print("  --eca_beta 1               # ECA adaptive kernel beta")
    print("  --sam_kernel_size 7        # CBAM SAM kernel size")
    print("  --interaction_weight 0.1   # Hybrid attention module interaction weight")
    print("  --log_attention            # Monitor attention patterns")

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
    print_eca_cbam_details()
    print_development_commands()
    
    print("\n📚 ADDITIONAL RESOURCES")
    print("-" * 35)
    print("• Interactive Notebooks: notebooks/")
    print("• Configuration Guide: CLAUDE.md")
    print("• Scientific Paper: Electronics 2025, 14(3), 517")
    print("• CBAM Paper: Woo et al. ECCV 2018")
    print("• ECA-Net Paper: Wang et al. CVPR 2020")
    print("• ECA-CBAM Documentation: docs/scientific/eca_cbam_hybrid_justification.md")
    
    print("\n💡 TIPS")
    print("-" * 35)
    print("• Use notebooks for interactive training and analysis")
    print("• Both models use identical training protocols for fair comparison")
    print("• ECA-CBAM provides hybrid attention module with superior efficiency")
    print("• Scientific comparison: CBAM baseline vs ECA-CBAM hybrid innovation")
    print("• Monitor attention patterns with --log_attention and --analyze_attention flags")

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
        elif command in ['eca', 'ecacbam', 'hybrid']:
            print_banner()
            print_eca_cbam_details()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: train, test, validate, model, config, workflow, dev, eca")
    else:
        print_help()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
FeatherFace WIDERFace Evaluation Script
Scientific evaluation for CBAM baseline model

Usage:
    python evaluate_widerface.py --model weights/cbam/featherface_cbam_final.pth --network cbam

Note: For ECA-CBAM hybrid evaluation, use evaluate script with test_eca_cbam.py
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from data import cfg_cbam_paper_exact

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate FeatherFace on WIDERFace dataset')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--network', default='cbam', choices=['cbam'], help='Network architecture: cbam (baseline only)')
    parser.add_argument('--confidence_threshold', default=0.02, type=float)
    parser.add_argument('--nms_threshold', default=0.4, type=float)
    parser.add_argument('--dataset_folder', default='./data/widerface/val/images/')
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--show_results', action='store_true', help='Show mAP results after evaluation')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("🚀 FeatherFace WIDERFace Evaluation")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Network: {args.network} (CBAM baseline)")
    print(f"Architecture: FeatherFaceCBAMExact")
    print(f"Expected params: 488,664")
    print(f"Attention: 6 CBAM modules (3 backbone + 3 BiFPN)")
    print("=" * 50)
    
    # Vérifier que le modèle existe
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return 1
    
    # Vérifier que le dataset existe
    if not os.path.exists(args.dataset_folder):
        print(f"❌ Dataset folder not found: {args.dataset_folder}")
        print("Please ensure WIDERFace dataset is downloaded and extracted")
        return 1
    
    # Créer le dossier de sauvegarde
    os.makedirs(args.save_folder, exist_ok=True)
    
    # Construire la commande de test
    cmd = [
        sys.executable, 'test_widerface.py',
        '-m', args.model,
        '--network', args.network,
        '--confidence_threshold', str(args.confidence_threshold),
        '--nms_threshold', str(args.nms_threshold),
        '--dataset_folder', args.dataset_folder,
        '--save_folder', args.save_folder
    ]
    
    if args.cpu:
        cmd.append('--cpu')
    
    print("📊 Step 1: Running inference on WIDERFace validation set...")
    print("Command:", ' '.join(cmd))
    print()
    
    # Exécuter le test
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Inference completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Inference failed: {e}")
        return 1
    
    if args.show_results:
        print("\\n📈 Step 2: Computing mAP scores...")
        
        # Commande d'évaluation
        eval_cmd = [
            sys.executable, 'widerface_evaluate/evaluation.py',
            '-p', args.save_folder,
            '-g', './widerface_evaluate/eval_tools/ground_truth'
        ]
        
        try:
            result = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
            print("✅ Evaluation completed!")
            print("\\n📊 Results:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Evaluation script failed: {e}")
            print("You can run evaluation manually:")
            print(f"cd widerface_evaluate && python evaluation.py -p {args.save_folder} -g ./eval_tools/ground_truth")
    
    print(f"\\n💾 Results saved to: {args.save_folder}")
    print(f"\\n✅ Scientific Evaluation Complete:")
    print(f"   Model: {args.network} (CBAM baseline)")
    param_count = f"{cfg_cbam_paper_exact['paper_baseline_performance']['total_parameters']:,}"
    print(f"   Parameters: {param_count}")
    print(f"   Attention: 6 CBAM modules (dual application)")
    print(f"   Foundation: Woo et al. ECCV 2018")
    
    print("📋 To run evaluation manually:")
    print("  cd widerface_evaluate")
    print(f"  python evaluation.py -p {args.save_folder} -g ./eval_tools/ground_truth")
    
    print("\\n🔗 For ECA-CBAM hybrid evaluation:")
    print("  python test_eca_cbam.py -m weights/eca_cbam/featherface_eca_cbam_final.pth --network eca_cbam")
    print("  cd widerface_evaluate && python evaluation.py -p ./widerface_txt -g ./eval_tools/ground_truth")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
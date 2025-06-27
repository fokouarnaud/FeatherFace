#!/usr/bin/env python
"""
Quick start script for FeatherFace V2 training
This script sets up and launches the training notebook
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Check if environment is ready"""
    print("Checking environment...")
    
    # Check Python
    print(f"✓ Python {sys.version.split()[0]}")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    # Check notebook
    notebook_path = Path("notebooks/03_train_evaluate_featherface_v2.ipynb")
    if notebook_path.exists():
        print(f"✓ Notebook found: {notebook_path}")
    else:
        print(f"✗ Notebook not found: {notebook_path}")
        return False
    
    return True

def check_data_and_weights():
    """Check if data and weights are ready"""
    print("\nChecking data and weights...")
    
    issues = []
    
    # Check dataset
    if not Path("data/widerface/train/label.txt").exists():
        issues.append("WIDERFace dataset not found")
        print("✗ WIDERFace dataset missing")
        print("  Download from: https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS")
    else:
        print("✓ WIDERFace dataset found")
    
    # Check MobileNet weights
    if not Path("weights/mobilenetV1X0.25_pretrain.tar").exists():
        issues.append("MobileNet pretrained weights not found")
        print("✗ MobileNet weights missing")
        print("  Download from: https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1")
    else:
        print("✓ MobileNet weights found")
    
    # Check teacher model
    if not Path("weights/FeatherNetB_se.pth").exists():
        print("⚠ Teacher model not found (optional)")
        print("  Train V1 first or download pretrained weights")
    else:
        print("✓ Teacher model found")
    
    return len(issues) == 0

def launch_notebook():
    """Launch Jupyter notebook"""
    print("\nLaunching notebook...")
    
    notebook_path = "notebooks/03_train_evaluate_featherface_v2.ipynb"
    
    try:
        # Try to open in browser
        if sys.platform == "win32":
            os.startfile(notebook_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", notebook_path])
        else:
            subprocess.run(["xdg-open", notebook_path])
            
        # Also start Jupyter if not running
        print("\nStarting Jupyter notebook server...")
        subprocess.run(["jupyter", "notebook", notebook_path])
        
    except Exception as e:
        print(f"Could not launch automatically: {e}")
        print(f"\nPlease run manually:")
        print(f"  jupyter notebook {notebook_path}")

def main():
    print("="*60)
    print("FeatherFace V2 Training Quick Start")
    print("="*60)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment setup incomplete")
        print("Please install PyTorch first")
        return
    
    # Check data
    data_ready = check_data_and_weights()
    
    if not data_ready:
        print("\n⚠️  Some data/weights are missing")
        print("The notebook will guide you through downloading them")
    
    # Launch
    print("\n" + "="*60)
    response = input("Launch training notebook? (y/n): ")
    
    if response.lower() == 'y':
        launch_notebook()
    else:
        print("\nTo launch manually:")
        print("  jupyter notebook notebooks/03_train_evaluate_featherface_v2.ipynb")
    
    print("\nGood luck with training! 🚀")

if __name__ == "__main__":
    main()
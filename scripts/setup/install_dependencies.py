#!/usr/bin/env python3
"""
Install missing dependencies for FeatherFace
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install required packages."""
    print("Installing FeatherFace dependencies...")
    print("=" * 40)
    
    # Required packages
    packages = [
        "gdown>=4.0.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "opencv-contrib-python>=4.5.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.62.0",
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n{success_count}/{len(packages)} packages installed successfully")
    
    # Install project in editable mode
    print("\nInstalling FeatherFace project...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("✅ FeatherFace project installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install project: {e}")

if __name__ == "__main__":
    main()
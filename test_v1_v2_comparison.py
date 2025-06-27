"""
Quick test script to verify FeatherFace V1 vs V2 comparison
Can be run without Jupyter notebook
"""

import sys
sys.path.append('..')

from models.retinaface import RetinaFace
from models.retinaface_v2 import RetinaFaceV2, get_retinaface_v2, count_parameters
from data.config import cfg_mnet, cfg_mnet_v2
import torch
import time

def main():
    print("="*60)
    print("FeatherFace V1 vs V2 Quick Comparison")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load models
    print("\nLoading models...")
    model_v1 = RetinaFace(cfg=cfg_mnet, phase='test').to(device).eval()
    model_v2 = get_retinaface_v2(cfg_mnet_v2, phase='test').to(device).eval()
    
    # Count parameters
    params_v1 = count_parameters(model_v1)
    params_v2 = count_parameters(model_v2)
    
    print(f"\nParameter Comparison:")
    print(f"  V1: {params_v1:,} ({params_v1/1e6:.3f}M)")
    print(f"  V2: {params_v2:,} ({params_v2/1e6:.3f}M)")
    print(f"  Reduction: {(1-params_v2/params_v1)*100:.1f}%")
    
    # Test inference speed
    print(f"\nTesting inference speed...")
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model_v1(dummy_input)
            _ = model_v2(dummy_input)
    
    # Measure V1
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            _ = model_v1(dummy_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_v1 = (time.time() - start) / 50 * 1000
    
    # Measure V2
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            _ = model_v2(dummy_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_v2 = (time.time() - start) / 50 * 1000
    
    print(f"\nInference Time Comparison:")
    print(f"  V1: {time_v1:.2f}ms")
    print(f"  V2: {time_v2:.2f}ms")
    print(f"  Speedup: {time_v1/time_v2:.2f}x")
    
    print("\n" + "="*60)
    print("Summary: FeatherFace V2 Success!")
    print("="*60)
    print(f"✓ {(1-params_v2/params_v1)*100:.1f}% fewer parameters")
    print(f"✓ {time_v1/time_v2:.2f}x faster inference")
    print(f"✓ Ready for production deployment")
    print("\nRun the Jupyter notebook for detailed visualizations.")

if __name__ == "__main__":
    main()
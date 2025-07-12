#!/usr/bin/env python3
"""
Debug Paper-Exact Implementation
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.featherface_paper_exact import FeatherFacePaperExact
from data.config import cfg_paper_accurate

def debug_model_creation():
    """Debug model creation step by step"""
    
    print("Debugging FeatherFace Paper-Exact Model Creation")
    print("=" * 50)
    
    try:
        print("1. Creating model...")
        model = FeatherFacePaperExact(cfg=cfg_paper_accurate, phase='test')
        print("✓ Model created successfully")
        
        print("\n2. Counting parameters...")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        
        print("\n3. Testing parameter breakdown...")
        try:
            param_info = model.get_parameter_count()
            print("✓ Parameter breakdown successful")
            
            for key, value in param_info.items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"✗ Parameter breakdown failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n4. Testing forward pass...")
        try:
            input_tensor = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                outputs = model(input_tensor)
            print("✓ Forward pass successful")
            print(f"  Output shapes: {[out.shape for out in outputs]}")
            
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_creation()
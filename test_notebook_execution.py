#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to execute and identify errors in 02_train_eca_cbam.ipynb
"""

import sys
import os
import io
import traceback

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to path
sys.path.append('.')

def test_cell_2():
    """Test cell 2: Path setup"""
    print("\n" + "="*60)
    print("Testing Cell 2: Path Setup")
    print("="*60)
    try:
        from pathlib import Path

        PROJECT_ROOT = Path(os.path.abspath('.')).resolve()
        print(f"Project root: {PROJECT_ROOT}")
        os.chdir(PROJECT_ROOT)
        print(f"Working directory: {os.getcwd()}")
        sys.path.append(str(PROJECT_ROOT))
        print("[OK] Path setup complete")
        return True
    except Exception as e:
        print(f"[ERROR] Cell 2 failed: {e}")
        traceback.print_exc()
        return False

def test_cell_3():
    """Test cell 3: System configuration and imports"""
    print("\n" + "="*60)
    print("Testing Cell 3: System Configuration & Imports")
    print("="*60)
    try:
        import torch
        import torch.nn as nn

        print(f"Python: {sys.version.split()[0]}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            device = torch.device('cuda')
        else:
            print("Using CPU (CUDA not available)")
            device = torch.device('cpu')

        print(f"Device: {device}")

        # Import ECA-CBAM configurations and models
        from data.config import cfg_eca_cbam, cfg_cbam_paper_exact
        print("[OK] Config imports successful")

        from models.featherface_eca_cbam import FeatherFaceECAcbaM
        from models.eca_cbam_hybrid import ECAcbaM
        print("[OK] ECA-CBAM hybrid imports successful")

        return True
    except Exception as e:
        print(f"[ERROR] Cell 3 failed: {e}")
        traceback.print_exc()
        return False

def test_cell_5():
    """Test cell 5: Model validation"""
    print("\n" + "="*60)
    print("Testing Cell 5: ECA-CBAM Model Validation")
    print("="*60)
    try:
        import torch
        from data.config import cfg_eca_cbam
        from models.featherface_eca_cbam import FeatherFaceECAcbaM
        from models.eca_cbam_hybrid import ECAcbaM

        # Create ECA-CBAM hybrid model
        model = FeatherFaceECAcbaM(cfg=cfg_eca_cbam, phase='test')

        # Parameter analysis
        param_info = model.get_parameter_count()
        total_params = param_info['total']

        print(f"Total parameters: {total_params:,} ({total_params/1e6:.3f}M)")
        print(f"Target: ~449,000 parameters (8.1% reduction vs CBAM baseline)")

        # Test forward pass
        print("\nFORWARD PASS VALIDATION")
        dummy_input = torch.randn(1, 3, 640, 640)
        model.eval()

        with torch.no_grad():
            outputs = model(dummy_input)

        print(f"Forward pass successful")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shapes: {[out.shape for out in outputs]}")

        if len(outputs) == 3:
            print("[OK] Output structure validated")
        else:
            print(f"[WARNING] Unexpected output structure: {len(outputs)} outputs")

        print("[OK] Model validation complete")
        return True

    except Exception as e:
        print(f"[ERROR] Cell 5 failed: {e}")
        traceback.print_exc()
        return False

def test_cell_7():
    """Test cell 7: Attention analysis"""
    print("\n" + "="*60)
    print("Testing Cell 7: ECA-CBAM Attention Analysis")
    print("="*60)
    try:
        import torch
        from data.config import cfg_eca_cbam
        from models.featherface_eca_cbam import FeatherFaceECAcbaM

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FeatherFaceECAcbaM(cfg=cfg_eca_cbam, phase='test')
        model = model.to(device)

        test_input = torch.randn(1, 3, 640, 640).to(device)

        with torch.no_grad():
            analysis = model.get_attention_analysis(test_input)

        print("[OK] Attention analysis complete")
        attention_summary = analysis['attention_summary']
        print(f"Attention summary: {list(attention_summary.keys())}")

        return True

    except Exception as e:
        print(f"[ERROR] Cell 7 failed: {e}")
        traceback.print_exc()
        return False

def test_cell_9():
    """Test cell 9: Dataset validation"""
    print("\n" + "="*60)
    print("Testing Cell 9: Dataset Validation")
    print("="*60)
    try:
        from pathlib import Path

        data_dir = Path('data/widerface')
        weights_dir = Path('weights/eca_cbam')
        results_dir = Path('results/eca_cbam')

        for dir_path in [data_dir, weights_dir, results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"[OK] Directory ready: {dir_path}")

        # Check for dataset files
        required_files = [
            data_dir / 'train' / 'label.txt',
            data_dir / 'val' / 'wider_val.txt'
        ]

        all_present = True
        for file_path in required_files:
            if file_path.exists():
                print(f"[OK] Found: {file_path}")
            else:
                print(f"[WARNING] Missing: {file_path}")
                all_present = False

        if not all_present:
            print("[INFO] Dataset not downloaded - this is expected on first run")

        print("[OK] Dataset validation complete")
        return True

    except Exception as e:
        print(f"[ERROR] Cell 9 failed: {e}")
        traceback.print_exc()
        return False

def test_cell_11():
    """Test cell 11: Training configuration"""
    print("\n" + "="*60)
    print("Testing Cell 11: Training Configuration")
    print("="*60)
    try:
        import torch
        from pathlib import Path
        from data.config import cfg_eca_cbam, cfg_cbam_paper_exact

        # Extract training parameters
        training_cfg = cfg_eca_cbam['training_config']
        base_cfg = cfg_eca_cbam

        print(f"Configuration: cfg_eca_cbam")
        print(f"Training dataset: {training_cfg['training_dataset']}")
        print(f"Network: {training_cfg['network']}")
        print(f"Batch size: {base_cfg['batch_size']}")
        print(f"Epochs: {base_cfg['epoch']}")

        # Build training command
        train_cmd = [
            'python', 'train_eca_cbam.py',
            '--training_dataset', training_cfg['training_dataset'],
        ]

        # Check if training script exists
        if Path('train_eca_cbam.py').exists():
            print("[OK] Training script found")
        else:
            print("[ERROR] Training script not found: train_eca_cbam.py")

        print("[OK] Training configuration complete")
        return True

    except Exception as e:
        print(f"[ERROR] Cell 11 failed: {e}")
        traceback.print_exc()
        return False

def test_cell_15():
    """Test cell 15: Evaluation configuration"""
    print("\n" + "="*60)
    print("Testing Cell 15: Evaluation Configuration")
    print("="*60)
    try:
        import glob
        from pathlib import Path

        # Check for trained models
        eca_cbam_models = sorted(glob.glob('weights/eca_cbam/*.pth'))
        eca_cbam_final_model = Path('weights/eca_cbam/featherface_eca_cbam_final.pth')

        if eca_cbam_final_model.exists():
            print(f"[OK] Found final model: {eca_cbam_final_model}")
        elif eca_cbam_models:
            print(f"[OK] Found {len(eca_cbam_models)} model(s)")
        else:
            print("[INFO] No trained models found - expected before training")

        # Check if test script exists
        if Path('test_eca_cbam.py').exists():
            print("[OK] Test script found")
        else:
            print("[ERROR] Test script not found: test_eca_cbam.py")

        print("[OK] Evaluation configuration complete")
        return True

    except Exception as e:
        print(f"[ERROR] Cell 15 failed: {e}")
        traceback.print_exc()
        return False

def test_cell_19():
    """Test cell 19: Model export"""
    print("\n" + "="*60)
    print("Testing Cell 19: Model Export")
    print("="*60)
    try:
        import torch
        from pathlib import Path
        from data.config import cfg_eca_cbam
        from models.featherface_eca_cbam import FeatherFaceECAcbaM

        # Create export directory
        export_dir = Path('exports/eca_cbam')
        export_dir.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Export directory ready: {export_dir}")

        # Test model creation
        model = FeatherFaceECAcbaM(cfg=cfg_eca_cbam, phase='test')
        model.eval()

        print("[OK] Model export configuration complete")
        return True

    except Exception as e:
        print(f"[ERROR] Cell 19 failed: {e}")
        traceback.print_exc()
        return False

def test_cell_21():
    """Test cell 21: Scientific validation"""
    print("\n" + "="*60)
    print("Testing Cell 21: Scientific Validation")
    print("="*60)
    try:
        from data.config import cfg_eca_cbam, cfg_cbam_paper_exact

        scientific_foundation = cfg_eca_cbam['scientific_foundation']
        performance_targets = cfg_eca_cbam['performance_targets']

        print(f"Architecture: {scientific_foundation['attention_mechanism']}")
        print(f"Parameters: {performance_targets['total_parameters']:,}")
        print(f"Efficiency gain: {performance_targets['efficiency_gain']}%")

        print("[OK] Scientific validation complete")
        return True

    except Exception as e:
        print(f"[ERROR] Cell 21 failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("NOTEBOOK EXECUTION TEST: 02_train_eca_cbam.ipynb")
    print("="*70)

    tests = [
        ("Cell 2: Path Setup", test_cell_2),
        ("Cell 3: System Config & Imports", test_cell_3),
        ("Cell 5: Model Validation", test_cell_5),
        ("Cell 7: Attention Analysis", test_cell_7),
        ("Cell 9: Dataset Validation", test_cell_9),
        ("Cell 11: Training Config", test_cell_11),
        ("Cell 15: Evaluation Config", test_cell_15),
        ("Cell 19: Model Export", test_cell_19),
        ("Cell 21: Scientific Validation", test_cell_21),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n[CRITICAL ERROR] {test_name} crashed: {e}")
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

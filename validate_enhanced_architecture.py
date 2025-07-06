#!/usr/bin/env python3
"""
FeatherFace Nano-B Enhanced Architecture Validation
Validates that the cleaned up architecture works correctly with Enhanced 2024 components
"""

import torch
import sys
import os

def validate_imports():
    """Validate that all Enhanced 2024 components import correctly"""
    print("🔍 Validating Enhanced 2024 Component Imports...")
    
    try:
        # Core Enhanced components from net.py
        from models.net import MobileNetV1, CBAM, BiFPN, SSH, ChannelShuffle2
        print("✅ Standard components (MobileNetV1, CBAM, BiFPN, SSH, ChannelShuffle2)")
        
        # Enhanced 2024 modules from featherface_nano_b.py
        from models.featherface_nano_b import (
            ScaleSequenceAttention, 
            SemanticEnhancementModule, 
            ScaleDecouplingModule,
            WeightedKnowledgeDistillation,
            FeatherFaceNanoB
        )
        print("✅ Enhanced 2024 modules (ASSN, MSE-FPN, Scale Decoupling, Weighted KD)")
        
        # Centralized configuration
        from data.config import cfg_nano_b
        print("✅ Centralized configuration (cfg_nano_b)")
        
        # Pruning components
        from models.pruning_b_fpgm import FeatherFaceNanoBPruner
        print("✅ Pruning components (B-FPGM)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def validate_architecture():
    """Validate Enhanced 2024 architecture components"""
    print("\n🏗️ Validating Enhanced 2024 Architecture...")
    
    try:
        from models.featherface_nano_b import create_featherface_nano_b
        from data.config import cfg_nano_b
        
        # Create model with Enhanced 2024 architecture
        model = create_featherface_nano_b(
            cfg=cfg_nano_b,
            phase='train',
            use_pruned_conv=False  # For testing
        )
        
        print("✅ Model creation successful")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        model.eval()
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shapes: {[out.shape for out in outputs]}")
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Parameter count: {total_params:,}")
        
        # Verify it's in the Enhanced 2024 range
        if 100_000 <= total_params <= 200_000:
            print(f"✅ Parameter count in Enhanced 2024 range (100K-200K)")
        else:
            print(f"⚠️  Parameter count outside typical range")
        
        return True
        
    except Exception as e:
        print(f"❌ Architecture validation error: {e}")
        return False

def validate_configuration():
    """Validate centralized configuration"""
    print("\n⚙️ Validating Centralized Configuration...")
    
    try:
        from data.config import cfg_nano_b
        
        # Check Enhanced 2024 specific settings
        required_keys = [
            'name', 'out_channel', 'cbam_reduction', 'bifpn_channels',
            'knowledge_distillation', 'distillation_temperature', 'distillation_alpha',
            'pruning_enabled', 'target_reduction', 'bayesian_iterations'
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in cfg_nano_b:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"❌ Missing configuration keys: {missing_keys}")
            return False
        
        print("✅ All required configuration keys present")
        
        # Verify Enhanced 2024 specific values
        if cfg_nano_b.get('out_channel') == 32:
            print("✅ Enhanced 2024 channel configuration (32)")
        else:
            print(f"⚠️  Channel configuration: {cfg_nano_b.get('out_channel')}")
        
        if cfg_nano_b.get('knowledge_distillation'):
            print("✅ Knowledge distillation enabled")
        
        if cfg_nano_b.get('pruning_enabled'):
            print("✅ Bayesian pruning enabled")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation error: {e}")
        return False

def validate_obsolete_removal():
    """Validate that obsolete modules are removed"""
    print("\n🗑️ Validating Obsolete Module Removal...")
    
    # Check that obsolete files don't exist
    obsolete_files = [
        'models/modules_v2.py',
        'models/retinaface_v2.py'
    ]
    
    all_removed = True
    for file_path in obsolete_files:
        if os.path.exists(file_path):
            print(f"❌ Obsolete file still exists: {file_path}")
            all_removed = False
        else:
            print(f"✅ Obsolete file removed: {file_path}")
    
    # Try importing obsolete modules (should fail)
    try:
        from models.modules_v2 import ChannelShuffle_Light
        print("❌ Obsolete modules_v2 still importable")
        all_removed = False
    except ImportError:
        print("✅ modules_v2 properly removed")
    
    try:
        from models.retinaface_v2 import RetinaFaceV2
        print("❌ Obsolete retinaface_v2 still importable")
        all_removed = False
    except ImportError:
        print("✅ retinaface_v2 properly removed")
    
    # Check for obsolete test files
    obsolete_test_files = [
        'tests/test_modules_v2.py'
    ]
    
    for test_file in obsolete_test_files:
        if os.path.exists(test_file):
            print(f"❌ Obsolete test file still exists: {test_file}")
            all_removed = False
        else:
            print(f"✅ Obsolete test file removed: {test_file}")
    
    return all_removed

def validate_documentation():
    """Validate documentation organization"""
    print("\n📚 Validating Documentation Organization...")
    
    expected_structure = {
        'docs/README.md': 'Main documentation index',
        'docs/architecture/nano_b_enhanced_2024.md': 'Main Enhanced architecture',
        'docs/architecture/enhanced_diagram.md': 'Architecture diagrams',
        'docs/architecture/enhanced_for_kids.md': 'Child-friendly explanation',
        'docs/guides/metaphors.md': 'Metaphors and analogies',
        'docs/legacy/': 'Legacy documentation'
    }
    
    all_present = True
    for path, description in expected_structure.items():
        if os.path.exists(path):
            print(f"✅ {description}: {path}")
        else:
            print(f"❌ Missing {description}: {path}")
            all_present = False
    
    return all_present

def main():
    """Run all validation tests"""
    print("🎯 FeatherFace Nano-B Enhanced 2024 Architecture Validation")
    print("=" * 70)
    
    tests = [
        ("Component Imports", validate_imports),
        ("Architecture", validate_architecture),
        ("Configuration", validate_configuration),
        ("Obsolete Removal", validate_obsolete_removal),
        ("Documentation", validate_documentation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        print("-" * 40)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("-" * 70)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ FeatherFace Nano-B Enhanced 2024 architecture is clean and consistent")
        print("✅ Ready for production use with Enhanced components only")
    else:
        print("⚠️  SOME VALIDATIONS FAILED")
        print("❌ Please check the failed tests above")
    
    print("=" * 70)
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
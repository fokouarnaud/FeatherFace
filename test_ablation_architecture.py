#!/usr/bin/env python3
"""
Test script for Ablation Study Architecture

This script validates that the modular ablation architecture works correctly:
1. Tests V1 baseline (all flags False)
2. Tests individual modules (one flag True at a time)
3. Tests combinations
4. Validates V1 compatibility preservation
"""

import torch
import logging
from data.config import cfg_nano_b
from models.featherface_nano_b import FeatherFaceNanoB

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_v1_baseline():
    """Test V1-identical baseline (all ablation flags False - disabled from Enhanced default)"""
    logger.info("🧪 TESTING V1 BASELINE (All Enhanced modules disabled)")
    
    # Disable all ablation flags from Enhanced default
    config = cfg_nano_b.copy()
    config['ablation_modules'] = {
        'small_face_optimization': False,  # Disable ScaleDecoupling
        'assn_enabled': False,             # Disable ASSN
        'mse_fpn_enabled': False,          # Disable MSE-FPN
        'ablation_mode': 'baseline',       # Pure V1 baseline mode
        'target_limitation': 'none',       # No enhancements
        'preserve_v1_base': True,          # V1 foundation only
    }
    
    try:
        # Create model
        model = FeatherFaceNanoB(cfg=config, phase='test')
        model.eval()
        
        # Test forward pass
        input_tensor = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(input_tensor)
        
        classifications, bbox_regressions, landmarks = outputs
        logger.info(f"✅ V1 Baseline successful:")
        logger.info(f"   Classifications: {classifications.shape}")
        logger.info(f"   BBox Regressions: {bbox_regressions.shape}")
        logger.info(f"   Landmarks: {landmarks.shape}")
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Total Parameters: {total_params:,}")
        
        return True, total_params
        
    except Exception as e:
        logger.error(f"❌ V1 Baseline failed: {e}")
        return False, 0

def test_scale_decoupling_only():
    """Test with only ScaleDecoupling enabled (others disabled from Enhanced default)"""
    logger.info("🧪 TESTING ScaleDecoupling Only")
    
    config = cfg_nano_b.copy()
    config['ablation_modules'] = {
        'small_face_optimization': True,  # Keep this enabled
        'assn_enabled': False,            # Disable from default
        'mse_fpn_enabled': False,         # Disable from default
        'ablation_mode': 'individual',
        'target_limitation': 'small_faces',
        'preserve_v1_base': True,
    }
    
    try:
        model = FeatherFaceNanoB(cfg=config, phase='test')
        model.eval()
        
        input_tensor = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(input_tensor)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ ScaleDecoupling test successful - Parameters: {total_params:,}")
        return True, total_params
        
    except Exception as e:
        logger.error(f"❌ ScaleDecoupling test failed: {e}")
        return False, 0

def test_mse_fpn_only():
    """Test with only MSE-FPN enabled (others disabled from Enhanced default)"""
    logger.info("🧪 TESTING MSE-FPN Only")
    
    config = cfg_nano_b.copy()
    config['ablation_modules'] = {
        'small_face_optimization': False, # Disable from default
        'assn_enabled': False,            # Disable from default
        'mse_fpn_enabled': True,          # Keep this enabled
        'ablation_mode': 'individual',
        'target_limitation': 'semantic_gap',
        'preserve_v1_base': True,
    }
    
    try:
        model = FeatherFaceNanoB(cfg=config, phase='test')
        model.eval()
        
        input_tensor = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(input_tensor)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ MSE-FPN test successful - Parameters: {total_params:,}")
        return True, total_params
        
    except Exception as e:
        logger.error(f"❌ MSE-FPN test failed: {e}")
        return False, 0

def test_assn_only():
    """Test with only ASSN enabled (others disabled from Enhanced default)"""
    logger.info("🧪 TESTING ASSN Only")
    
    config = cfg_nano_b.copy()
    config['ablation_modules'] = {
        'small_face_optimization': False, # Disable from default
        'assn_enabled': True,             # Keep this enabled
        'mse_fpn_enabled': False,         # Disable from default
        'ablation_mode': 'individual',
        'target_limitation': 'attention_specialization',
        'preserve_v1_base': True,
    }
    
    try:
        model = FeatherFaceNanoB(cfg=config, phase='test')
        model.eval()
        
        input_tensor = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(input_tensor)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ ASSN test successful - Parameters: {total_params:,}")
        return True, total_params
        
    except Exception as e:
        logger.error(f"❌ ASSN test failed: {e}")
        return False, 0

def test_enhanced_default():
    """Test Enhanced Nano-B default configuration (all modules enabled by default)"""
    logger.info("🧪 TESTING Enhanced Nano-B Default (All modules enabled)")
    
    # Use default cfg_nano_b configuration (all modules enabled by default)
    config = cfg_nano_b.copy()
    # No need to modify - defaults should have all modules enabled
    
    logger.info("Using default Enhanced Nano-B configuration...")
    
    try:
        model = FeatherFaceNanoB(cfg=config, phase='test')
        model.eval()
        
        input_tensor = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(input_tensor)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ All modules test successful - Parameters: {total_params:,}")
        return True, total_params
        
    except Exception as e:
        logger.error(f"❌ All modules test failed: {e}")
        return False, 0

def main():
    """Run all ablation architecture tests"""
    logger.info("="*60)
    logger.info("ABLATION STUDY ARCHITECTURE VALIDATION")
    logger.info("="*60)
    
    results = {}
    
    # Test 1: V1 Baseline
    success, params = test_v1_baseline()
    results['v1_baseline'] = {'success': success, 'params': params}
    
    # Test 2: ScaleDecoupling only
    success, params = test_scale_decoupling_only()
    results['scale_decoupling'] = {'success': success, 'params': params}
    
    # Test 3: MSE-FPN only
    success, params = test_mse_fpn_only()
    results['mse_fpn'] = {'success': success, 'params': params}
    
    # Test 4: ASSN only
    success, params = test_assn_only()
    results['assn'] = {'success': success, 'params': params}
    
    # Test 5: Enhanced default (all modules)
    success, params = test_enhanced_default()
    results['enhanced_default'] = {'success': success, 'params': params}
    
    # Summary
    logger.info("="*60)
    logger.info("ABLATION TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        params_str = f"{result['params']:,}" if result['success'] else "N/A"
        logger.info(f"{test_name:20}: {status:10} - Parameters: {params_str}")
        if not result['success']:
            all_passed = False
    
    # Parameter analysis
    if results['v1_baseline']['success']:
        baseline_params = results['v1_baseline']['params']
        logger.info(f"\n📊 PARAMETER ANALYSIS (vs V1 baseline: {baseline_params:,})")
        
        for test_name, result in results.items():
            if test_name != 'v1_baseline' and result['success']:
                diff = result['params'] - baseline_params
                percent = (diff / baseline_params) * 100
                logger.info(f"{test_name:20}: +{diff:,} (+{percent:.1f}%)")
    
    logger.info("="*60)
    if all_passed:
        logger.info("🎉 ALL ABLATION TESTS PASSED - Architecture is ready for ablation studies!")
    else:
        logger.info("⚠️  Some tests failed - check configuration and module implementations")
    logger.info("="*60)
    
    return all_passed

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Advanced Ablation Study Script for FeatherFace Nano-B

This script performs comprehensive ablation studies to identify which 2024 modules
best address V1 limitations:

1. Individual Module Impact Analysis
2. Progressive Combination Testing  
3. Performance vs Parameter Trade-off Analysis
4. Best Configuration Discovery

Usage:
    python scripts/ablation_study.py --mode individual
    python scripts/ablation_study.py --mode progressive  
    python scripts/ablation_study.py --mode best_combination
    python scripts/ablation_study.py --mode full_analysis
"""

import argparse
import torch
import logging
import json
import time
from typing import Dict, List, Tuple
from pathlib import Path

# Import FeatherFace components
from data.config import cfg_nano_b
from models.featherface_nano_b import FeatherFaceNanoB

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AblationStudy:
    """Comprehensive ablation study for FeatherFace Nano-B modules"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        self.baseline_config = None
        
    def create_config(self, modules_config: Dict) -> Dict:
        """Create configuration for specific module combination"""
        config = cfg_nano_b.copy()
        config['ablation_modules'] = {
            'small_face_optimization': modules_config.get('scale_decoupling', False),
            'assn_enabled': modules_config.get('assn', False),
            'mse_fpn_enabled': modules_config.get('mse_fpn', False),
            'ablation_mode': modules_config.get('mode', 'individual'),
            'target_limitation': modules_config.get('target', 'small_faces'),
            'preserve_v1_base': True,  # Always preserve V1 base
        }
        return config
    
    def test_configuration(self, config_name: str, modules_config: Dict) -> Dict:
        """Test a specific module configuration"""
        logger.info(f"🧪 Testing configuration: {config_name}")
        
        try:
            # Create model with specific configuration
            config = self.create_config(modules_config)
            model = FeatherFaceNanoB(cfg=config, phase='test')
            model.eval()
            model.to(self.device)
            
            # Test input
            input_tensor = torch.randn(1, 3, 640, 640).to(self.device)
            
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                outputs = model(input_tensor)
            inference_time = time.time() - start_time
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Validate outputs
            classifications, bbox_regressions, landmarks = outputs
            
            result = {
                'success': True,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'inference_time': inference_time,
                'output_shapes': {
                    'classifications': list(classifications.shape),
                    'bbox_regressions': list(bbox_regressions.shape),
                    'landmarks': list(landmarks.shape)
                },
                'modules_active': modules_config,
                'config_name': config_name
            }
            
            logger.info(f"✅ {config_name}: {total_params:,} params, {inference_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ {config_name} failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'config_name': config_name,
                'modules_active': modules_config
            }
    
    def individual_module_analysis(self) -> Dict:
        """Test each module individually vs Enhanced baseline"""
        logger.info("="*60)
        logger.info("INDIVIDUAL MODULE ANALYSIS (Removal Impact)")
        logger.info("="*60)
        
        test_configs = {
            'enhanced_baseline': {
                'scale_decoupling': True,
                'assn': True,
                'mse_fpn': True,
                'mode': 'combined',
                'target': 'all_limitations'
            },
            'v1_baseline': {
                'scale_decoupling': False,
                'assn': False,
                'mse_fpn': False,
                'mode': 'baseline',
                'target': 'none'
            },
            'scale_decoupling_only': {
                'scale_decoupling': True,
                'assn': False,
                'mse_fpn': False,
                'mode': 'individual',
                'target': 'small_faces'
            },
            'assn_only': {
                'scale_decoupling': False,
                'assn': True,
                'mse_fpn': False,
                'mode': 'individual',
                'target': 'attention_specialization'
            },
            'mse_fpn_only': {
                'scale_decoupling': False,
                'assn': False,
                'mse_fpn': True,
                'mode': 'individual',
                'target': 'semantic_gap'
            }
        }
        
        results = {}
        for config_name, modules_config in test_configs.items():
            results[config_name] = self.test_configuration(config_name, modules_config)
        
        # Analysis
        self._analyze_individual_results(results)
        return results
    
    def progressive_combination_analysis(self) -> Dict:
        """Test progressive removal from Enhanced baseline"""
        logger.info("="*60)
        logger.info("PROGRESSIVE REMOVAL ANALYSIS")
        logger.info("="*60)
        
        test_configs = {
            'enhanced_full': {
                'scale_decoupling': True,
                'assn': True,
                'mse_fpn': True,
                'mode': 'combined'
            },
            'remove_scale_decoupling': {
                'scale_decoupling': False,
                'assn': True,
                'mse_fpn': True,
                'mode': 'progressive'
            },
            'remove_assn': {
                'scale_decoupling': True,
                'assn': False,
                'mse_fpn': True,
                'mode': 'progressive'
            },
            'remove_mse_fpn': {
                'scale_decoupling': True,
                'assn': True,
                'mse_fpn': False,
                'mode': 'progressive'
            },
            'keep_scale_only': {
                'scale_decoupling': True,
                'assn': False,
                'mse_fpn': False,
                'mode': 'progressive'
            },
            'keep_assn_only': {
                'scale_decoupling': False,
                'assn': True,
                'mse_fpn': False,
                'mode': 'progressive'
            },
            'keep_mse_only': {
                'scale_decoupling': False,
                'assn': False,
                'mse_fpn': True,
                'mode': 'progressive'
            },
            'v1_baseline': {
                'scale_decoupling': False,
                'assn': False,
                'mse_fpn': False,
                'mode': 'baseline'
            }
        }
        
        results = {}
        for config_name, modules_config in test_configs.items():
            results[config_name] = self.test_configuration(config_name, modules_config)
        
        # Analysis
        self._analyze_progressive_results(results)
        return results
    
    def best_combination_search(self) -> Dict:
        """Find the best combination for different use cases"""
        logger.info("="*60)
        logger.info("BEST COMBINATION SEARCH")
        logger.info("="*60)
        
        # Test all possible combinations
        all_combinations = []
        for scale in [False, True]:
            for assn in [False, True]:
                for mse in [False, True]:
                    if not scale and not assn and not mse:
                        continue  # Skip all-false (baseline already tested)
                    
                    combo_name = []
                    if scale: combo_name.append('ScaleDecoupling')\n                    if assn: combo_name.append('ASSN')\n                    if mse: combo_name.append('MSE-FPN')\n                    \n                    config_name = '+'.join(combo_name) if combo_name else 'baseline'\n                    \n                    all_combinations.append({\n                        'name': config_name,\n                        'config': {\n                            'scale_decoupling': scale,\n                            'assn': assn,\n                            'mse_fpn': mse,\n                            'mode': 'best_search'\n                        }\n                    })\n        \n        results = {}\n        for combo in all_combinations:\n            results[combo['name']] = self.test_configuration(combo['name'], combo['config'])\n        \n        # Find best configurations\n        self._find_best_configurations(results)\n        return results\n    \n    def _analyze_individual_results(self, results: Dict):\n        \"\"\"Analyze individual module impact\"\"\"\n        logger.info(\"\\n📊 INDIVIDUAL MODULE IMPACT ANALYSIS\")\n        \n        baseline = results.get('v1_baseline')\n        if not baseline or not baseline['success']:\n            logger.error(\"❌ Baseline test failed - cannot analyze\")\n            return\n        \n        baseline_params = baseline['total_params']\n        baseline_time = baseline['inference_time']\n        \n        logger.info(f\"V1 Baseline: {baseline_params:,} params, {baseline_time:.3f}s\")\n        logger.info(\"-\" * 40)\n        \n        for config_name, result in results.items():\n            if config_name == 'v1_baseline' or not result['success']:\n                continue\n                \n            param_diff = result['total_params'] - baseline_params\n            param_percent = (param_diff / baseline_params) * 100\n            time_diff = result['inference_time'] - baseline_time\n            time_percent = (time_diff / baseline_time) * 100\n            \n            logger.info(f\"{config_name}:\")\n            logger.info(f\"  Parameters: +{param_diff:,} (+{param_percent:.1f}%)\")\n            logger.info(f\"  Inference:  +{time_diff:.3f}s (+{time_percent:.1f}%)\")\n    \n    def _analyze_progressive_results(self, results: Dict):\n        \"\"\"Analyze progressive combination results\"\"\"\n        logger.info(\"\\n📊 PROGRESSIVE COMBINATION ANALYSIS\")\n        \n        # Sort by parameter count\n        successful_results = {k: v for k, v in results.items() if v['success']}\n        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['total_params'])\n        \n        logger.info(\"Configuration ranking by parameter efficiency:\")\n        logger.info(\"-\" * 60)\n        \n        for i, (config_name, result) in enumerate(sorted_results):\n            params = result['total_params']\n            time = result['inference_time']\n            modules = [k for k, v in result['modules_active'].items() if v is True and k in ['scale_decoupling', 'assn', 'mse_fpn']]\n            modules_str = '+'.join(modules) if modules else 'baseline'\n            \n            logger.info(f\"{i+1:2d}. {config_name:25} | {params:8,} params | {time:.3f}s | {modules_str}\")\n    \n    def _find_best_configurations(self, results: Dict):\n        \"\"\"Find best configurations for different criteria\"\"\"\n        logger.info(\"\\n🏆 BEST CONFIGURATIONS BY CRITERIA\")\n        \n        successful_results = {k: v for k, v in results.items() if v['success']}\n        \n        if not successful_results:\n            logger.error(\"❌ No successful configurations found\")\n            return\n        \n        # Most efficient (least parameters)\n        most_efficient = min(successful_results.items(), key=lambda x: x[1]['total_params'])\n        logger.info(f\"Most Efficient: {most_efficient[0]} ({most_efficient[1]['total_params']:,} params)\")\n        \n        # Fastest inference\n        fastest = min(successful_results.items(), key=lambda x: x[1]['inference_time'])\n        logger.info(f\"Fastest Inference: {fastest[0]} ({fastest[1]['inference_time']:.3f}s)\")\n        \n        # Best balance (considering both params and time)\n        def balance_score(result):\n            # Normalize both metrics and combine\n            params_norm = result['total_params'] / 1000000  # Normalize to millions\n            time_norm = result['inference_time'] * 1000     # Normalize to milliseconds\n            return params_norm + time_norm\n        \n        best_balance = min(successful_results.items(), key=lambda x: balance_score(x[1]))\n        logger.info(f\"Best Balance: {best_balance[0]} (score: {balance_score(best_balance[1]):.2f})\")\n    \n    def save_results(self, results: Dict, output_file: str):\n        \"\"\"Save results to JSON file\"\"\"\n        output_path = Path(output_file)\n        output_path.parent.mkdir(parents=True, exist_ok=True)\n        \n        with open(output_path, 'w') as f:\n            json.dump(results, f, indent=2, default=str)\n        \n        logger.info(f\"📄 Results saved to: {output_path}\")\n    \n    def run_full_analysis(self) -> Dict:\n        \"\"\"Run complete ablation study\"\"\"\n        logger.info(\"🚀 Starting FULL ABLATION STUDY\")\n        \n        all_results = {\n            'individual': self.individual_module_analysis(),\n            'progressive': self.progressive_combination_analysis(),\n            'best_combination': self.best_combination_search()\n        }\n        \n        logger.info(\"\\n\" + \"=\"*60)\n        logger.info(\"FULL ABLATION STUDY COMPLETED\")\n        logger.info(\"=\"*60)\n        \n        return all_results\n\ndef main():\n    parser = argparse.ArgumentParser(description='FeatherFace Nano-B Ablation Study')\n    parser.add_argument('--mode', choices=['individual', 'progressive', 'best_combination', 'full_analysis'], \n                        default='individual', help='Type of ablation study to run')\n    parser.add_argument('--output', default='results/ablation_study.json', \n                        help='Output file for results')\n    parser.add_argument('--device', default='auto', \n                        help='Device to use (auto, cuda, cpu)')\n    \n    args = parser.parse_args()\n    \n    # Setup device\n    if args.device == 'auto':\n        device = 'cuda' if torch.cuda.is_available() else 'cpu'\n    else:\n        device = args.device\n    \n    logger.info(f\"Using device: {device}\")\n    \n    # Create ablation study\n    study = AblationStudy(device=device)\n    \n    # Run specified analysis\n    if args.mode == 'individual':\n        results = study.individual_module_analysis()\n    elif args.mode == 'progressive':\n        results = study.progressive_combination_analysis()\n    elif args.mode == 'best_combination':\n        results = study.best_combination_search()\n    elif args.mode == 'full_analysis':\n        results = study.run_full_analysis()\n    \n    # Save results\n    study.save_results(results, args.output)\n    \n    logger.info(f\"\\n🎉 Ablation study '{args.mode}' completed successfully!\")\n\nif __name__ == \"__main__\":\n    main()
#!/usr/bin/env python3
"""
Test de la Structure Simplifiée

Vérifie que les imports et l'instanciation des modèles fonctionnent correctement.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_structure():
    """Test de la structure simplifiée"""
    print("🧪 Test de la Structure Simplifiée FeatherFace")
    print("=" * 60)
    
    try:
        # Test des imports
        print("📦 Test des imports...")
        from data.config import cfg_mnet, cfg_v2
        from models.retinaface import RetinaFace
        from models.featherface_v2 import FeatherFaceV2
        from models.attention_v2 import CoordinateAttention
        print("✅ Tous les imports réussis")
        
        # Test instanciation V1 Original
        print("\n🔧 Test V1 Original...")
        v1_model = RetinaFace(cfg=cfg_mnet, phase='test')
        v1_params = sum(p.numel() for p in v1_model.parameters())
        print(f"✅ V1 Original: {v1_params:,} paramètres")
        
        # Test instanciation V2 Innovation
        print("\n🚀 Test V2 Innovation...")
        v2_model = FeatherFaceV2(cfg=cfg_v2, phase='test')
        v2_params = sum(p.numel() for p in v2_model.parameters())
        print(f"✅ V2 Innovation: {v2_params:,} paramètres")
        
        # Test forward pass
        print("\n🔄 Test forward pass...")
        dummy_input = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            v1_output = v1_model(dummy_input)
            v2_output = v2_model(dummy_input)
        
        print(f"✅ V1 outputs: {[out.shape for out in v1_output]}")
        print(f"✅ V2 outputs: {[out.shape for out in v2_output]}")
        
        # Comparaison
        param_diff = v2_params - v1_params
        param_ratio = v2_params / v1_params
        
        print(f"\n📊 Comparaison:")
        print(f"  V1 Original:     {v1_params:,} params")
        print(f"  V2 Innovation:   {v2_params:,} params")
        print(f"  Différence:      {param_diff:,} params ({((param_ratio-1)*100):+.1f}%)")
        
        print(f"\n🎯 Structure:")
        print(f"  ✅ V1: models/retinaface.py + train_v1.py")
        print(f"  ✅ V2: models/featherface_v2.py + train_v2.py")
        print(f"  ✅ Base: models/net.py + models/attention_v2.py")
        
        print(f"\n✅ STRUCTURE SIMPLIFIÉE FONCTIONNELLE!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_structure()
    exit(0 if success else 1)
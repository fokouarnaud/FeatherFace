#!/usr/bin/env python3
"""
Script pour corriger les références Lu et al. 2024 et les déplacer vers Perspectives
"""
import os

def fix_readme(filepath):
    """Correction complète du README.md"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Retirer la ligne Lu et al. 2024 de la section Research Papers
    content = content.replace(
        '- **Hybrid Attention Module**: Lu W, Yang Y and Yang L. 2024 - Fine-grained image classification method based on hybrid attention module. Frontiers in Neurorobotics (DOI: 10.3389/fnbot.2024.1391791)',
        '- **ECA-CBAM Application**: ECA-CBAM: Classification of Diabetic Retinopathy. ACM AIAI 2022 (DOI: 10.1145/3529466.3529468)'
    )

    # Corriger la section Key Findings
    content = content.replace(
        '- **Hybrid Attention Module**: Synergistic effects validated in verified scientific literature (Lu et al. 2024, Frontiers in Neurorobotics)',
        '- **Sequential Attention Architecture**: ECA-Net efficiency combined with CBAM spatial attention in sequential processing (Wang et al. 2020; Woo et al. 2018)'
    )

    # Ajouter section Future Work avant "Citation"
    future_work_section = '''
## 🔮 Future Work and Alternative Approaches

### Parallel Hybrid Attention Architecture

Recent work by Lu et al. (2024) proposes an alternative **parallel architecture** where channel and spatial attention maps are computed independently and then multiplied together, rather than applied sequentially:

```python
# Lu et al. 2024 Parallel Approach
M_channel = channel_attention(X)  # Parallel branch 1
M_spatial = spatial_attention(X)  # Parallel branch 2
M_hybrid = M_channel * M_spatial   # Attention map multiplication
output = X + (M_hybrid * X)        # Residual connection
```

**Reference:** Lu W, Yang Y and Yang L. (2024). Fine-grained image classification method based on hybrid attention module. *Frontiers in Neurorobotics*. DOI: 10.3389/fnbot.2024.1391791

**Key Differences from Our Sequential Approach:**
- **Parallel computation** vs sequential (ECA → SAM)
- **Multiplication of attention maps** vs direct application
- **Explicit residual connection** to preserve original features
- May reduce information loss from strict sequential processing

**Why We Chose Sequential:**
- ✅ Aligned with standard CBAM architecture (Woo et al. 2018)
- ✅ Proven parameter efficiency (449,017 vs 488,664 params)
- ✅ Stable convergence during training
- ✅ Better mobile deployment compatibility
- ✅ Demonstrated performance gains (+1.7% mAP Hard)

**Future Exploration:**
An empirical comparison between sequential and parallel hybrid attention architectures would be valuable for understanding the trade-offs in face detection applications.

'''

    # Insérer la section Future Work avant "## 📄 Citation"
    content = content.replace(
        '## 📄 Citation',
        future_work_section + '## 📄 Citation'
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Fixed: {filepath}")
    return True

# Exécution
base_path = r'C:/Users/cedric/Desktop/box/01-Projects/Face-Recognition/FeatherFace'
readme_path = os.path.join(base_path, 'README.md')

print("="*70)
print("Fixing Lu et al. 2024 references - Moving to Future Work section")
print("="*70)

if os.path.exists(readme_path):
    fix_readme(readme_path)
    print("="*70)
    print("SUCCESS: README.md corrected")
    print("- Removed Lu et al. 2024 from main justification")
    print("- Added comprehensive Future Work section")
    print("- Added Diabetic Retinopathy paper reference")
    print("="*70)
else:
    print(f"ERROR: {readme_path} not found")

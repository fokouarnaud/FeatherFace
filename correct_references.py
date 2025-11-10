#!/usr/bin/env python3
"""
Script to correct scientific references from Wang et al. to Lu et al.
"""
import os
import re

def correct_file(filepath):
    """Correct Wang et al. 2024 references to Lu et al. 2024"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Pattern 1: Full reference with DOI
        content = re.sub(
            r'Wang et al\. 2024 Frontiers in Neurorobotics \(DOI: 10\.3389/fnbot\.2024\.1391791\)',
            'Lu W, Yang Y and Yang L. 2024 - Fine-grained image classification method based on hybrid attention module. Frontiers in Neurorobotics (DOI: 10.3389/fnbot.2024.1391791)',
            content
        )

        # Pattern 2: Short reference
        content = re.sub(
            r'\(Wang et al\. 2024, Frontiers in Neurorobotics\)',
            '(Lu et al. 2024, Frontiers in Neurorobotics)',
            content
        )

        # Pattern 3: Inline mentions
        content = re.sub(
            r'Wang et al\. Frontiers in Neurorobotics 2024',
            'Lu et al. Frontiers in Neurorobotics 2024',
            content
        )

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Corrected: {filepath}")
            return True
        else:
            print(f"No changes: {filepath}")
            return False
    except Exception as e:
        print(f"Error with {filepath}: {e}")
        return False

# Files to correct
files_to_correct = [
    'README.md',
    'help.py',
    'docs/scientific/systematic_literature_review.md',
    'docs/scientific/eca_cbam_hybrid_justification.md',
    'data/config.py',
    'notebooks/02_train_eca_cbam.ipynb',
    'models/eca_cbam_hybrid.py',
    'train/README.md',
]

base_path = r'C:/Users/cedric/Desktop/box/01-Projects/Face-Recognition/FeatherFace'

print("Correcting Wang et al. to Lu et al. in scientific references...")
print("="* 70)

corrected_count = 0
for file_rel in files_to_correct:
    filepath = os.path.join(base_path, file_rel)
    if os.path.exists(filepath):
        if correct_file(filepath):
            corrected_count += 1
    else:
        print(f"File not found: {filepath}")

print("=" * 70)
print(f"Correction complete: {corrected_count} files modified")

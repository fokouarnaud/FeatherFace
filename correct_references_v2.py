#!/usr/bin/env python3
"""
Script to correct scientific references from Wang et al. to Lu et al. (comprehensive version)
"""
import os
import re

def correct_file_comprehensive(filepath):
    """Correct Wang et al. 2024 references to Lu et al. 2024 - comprehensive patterns"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes = []

        # Pattern 1: Full reference with DOI (already done)
        if 'Wang et al. 2024 Frontiers in Neurorobotics (DOI: 10.3389/fnbot.2024.1391791)' in content:
            content = content.replace(
                'Wang et al. 2024 Frontiers in Neurorobotics (DOI: 10.3389/fnbot.2024.1391791)',
                'Lu W, Yang Y and Yang L. 2024 - Fine-grained image classification method based on hybrid attention module. Frontiers in Neurorobotics (DOI: 10.3389/fnbot.2024.1391791)'
            )
            changes.append("Full reference with DOI")

        # Pattern 2: Short reference (already done)
        if '(Wang et al. 2024, Frontiers in Neurorobotics)' in content:
            content = content.replace(
                '(Wang et al. 2024, Frontiers in Neurorobotics)',
                '(Lu et al. 2024, Frontiers in Neurorobotics)'
            )
            changes.append("Short parenthetical reference")

        # Pattern 3: Inline mentions (already done)
        if 'Wang et al. Frontiers in Neurorobotics 2024' in content:
            content = content.replace(
                'Wang et al. Frontiers in Neurorobotics 2024',
                'Lu et al. Frontiers in Neurorobotics 2024'
            )
            changes.append("Inline mention")

        # Pattern 4: French "Selon Wang et al."
        if 'Selon Wang et al. dans *Frontiers in Neurorobotics* (2024)' in content:
            content = content.replace(
                'Selon Wang et al. dans *Frontiers in Neurorobotics* (2024)',
                'Selon Lu et al. dans *Frontiers in Neurorobotics* (2024)'
            )
            changes.append("French 'Selon Wang et al.'")

        # Pattern 5: Hybrid Attention Module references
        if 'Hybrid attention module validé (Wang et al. 2024)' in content:
            content = content.replace(
                'Hybrid attention module validé (Wang et al. 2024)',
                'Hybrid attention module validé (Lu et al. 2024)'
            )
            changes.append("Hybrid attention module reference")

        if 'Wang et al. 2024: Multi-phase training' in content:
            content = content.replace(
                'Wang et al. 2024: Multi-phase training',
                'Lu et al. 2024: Multi-phase training'
            )
            changes.append("Multi-phase training reference")

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Corrected: {filepath}")
            for change in changes:
                print(f"  - {change}")
            return True
        else:
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
    'models/eca_cbam_hybrid_parallel_backup.py',
    'train/README.md',
    'MODIFICATIONS_CONFORMITE_MEMOIRE.md',
    'VALIDATION_FINALE_TESTS.md',
]

base_path = r'C:/Users/cedric/Desktop/box/01-Projects/Face-Recognition/FeatherFace'

print("=" * 70)
print("Correcting Wang et al. to Lu et al. (comprehensive pass)")
print("=" * 70)

corrected_count = 0
skipped_count = 0

for file_rel in files_to_correct:
    filepath = os.path.join(base_path, file_rel)
    if os.path.exists(filepath):
        if correct_file_comprehensive(filepath):
            corrected_count += 1
        else:
            print(f"No changes needed: {filepath}")
            skipped_count += 1
    else:
        print(f"File not found: {filepath}")

print("=" * 70)
print(f"Results: {corrected_count} files corrected, {skipped_count} files already OK")
print("=" * 70)

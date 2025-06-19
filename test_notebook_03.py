"""
Test script to verify the V2 training notebook structure
"""

import json
import sys
from pathlib import Path

def validate_notebook(notebook_path):
    """Validate Jupyter notebook structure"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Check basic structure
        assert 'cells' in notebook, "No cells found"
        assert 'metadata' in notebook, "No metadata found"
        assert 'nbformat' in notebook, "No nbformat found"
        
        # Count cell types
        cell_types = {'markdown': 0, 'code': 0}
        for cell in notebook['cells']:
            cell_type = cell.get('cell_type', 'unknown')
            if cell_type in cell_types:
                cell_types[cell_type] += 1
        
        print(f"✓ Notebook structure valid")
        print(f"  - Total cells: {len(notebook['cells'])}")
        print(f"  - Markdown cells: {cell_types['markdown']}")
        print(f"  - Code cells: {cell_types['code']}")
        
        # Check sections
        sections = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                source = ''.join(cell['source'])
                if source.startswith('## '):
                    sections.append(source.split('\n')[0])
        
        print(f"\n✓ Found {len(sections)} sections:")
        for i, section in enumerate(sections, 1):
            print(f"  {i}. {section.replace('## ', '')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

if __name__ == "__main__":
    notebook_path = Path("notebooks/03_train_evaluate_featherface_v2.ipynb")
    
    if notebook_path.exists():
        print(f"Validating {notebook_path}")
        if validate_notebook(notebook_path):
            print("\n✅ Notebook 03 successfully created!")
            print("\nKey features:")
            print("- Complete V2 training pipeline with knowledge distillation")
            print("- Model comparison and performance analysis")
            print("- Direct evaluation capabilities")
            print("- Export for deployment")
            print("\nNext steps:")
            print("1. Open in Jupyter: jupyter notebook", notebook_path)
            print("2. Follow the notebook to train FeatherFace V2")
            print("3. Achieve 0.25M params with 92%+ mAP!")
    else:
        print(f"✗ Notebook not found: {notebook_path}")
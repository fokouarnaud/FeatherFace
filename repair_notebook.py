#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to repair and clean the Jupyter notebook
"""

import json
import sys

def repair_notebook(input_path, output_path):
    """
    Load, clean, and save a Jupyter notebook
    """
    print(f"Reading notebook from: {input_path}")

    # Load notebook
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Notebook format version: {nb.get('nbformat', 'unknown')}")
    print(f"Number of cells: {len(nb.get('cells', []))}")

    # Clean outputs from all cells
    cleaned_cells = 0
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            # Clear outputs
            cell['outputs'] = []
            cell['execution_count'] = None
            cleaned_cells += 1

    print(f"Cleaned outputs from {cleaned_cells} code cells")

    # Save cleaned notebook
    print(f"Writing cleaned notebook to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Notebook repair completed successfully")
    return True

if __name__ == "__main__":
    input_file = 'notebooks/02_train_eca_cbam.ipynb'
    output_file = 'notebooks/02_train_eca_cbam_clean.ipynb'

    try:
        repair_notebook(input_file, output_file)
        print(f"\nRepaired notebook saved as: {output_file}")
        print(f"Original backup available at: {input_file}.backup")
        sys.exit(0)
    except Exception as e:
        print(f"Error repairing notebook: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

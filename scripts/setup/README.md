# Setup Scripts

This directory contains scripts for setting up the FeatherFace V1 baseline and Nano-B Enhanced 2024 environment.

## Scripts

### `install_dependencies.py`
Installs all required dependencies for FeatherFace V1 baseline and Nano-B Enhanced 2024 specialized training and validation.

**Usage:**
```bash
# From project root
python scripts/setup/install_dependencies.py

# Or directly
cd scripts/setup && python install_dependencies.py
```

**Features:**
- Installs core dependencies (torch, torchvision, etc.)
- Installs computer vision packages (opencv, albumentations)
- Installs project in editable mode
- Provides detailed installation feedback

## Related Directories

- `scripts/validation/` - Model validation and testing scripts
- `scripts/training/` - Training scripts
- `scripts/deployment/` - Export and deployment scripts
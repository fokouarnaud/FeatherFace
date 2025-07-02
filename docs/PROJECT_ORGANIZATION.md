# Project Organization Guide

This document explains the organized structure of the FeatherFace project and how to navigate it efficiently.

## 📁 Directory Structure Overview

```
FeatherFace/
├── 📖 README.md                    # Main project documentation
├── 📄 CLAUDE.md                    # Claude AI assistant instructions
├── 📄 LICENSE                     # Project license
├── ⚙️ pyproject.toml              # Python project configuration
│
├── 📊 notebooks/                   # Training notebooks and analysis
│   ├── 01_train_evaluate_featherface.ipynb       # V1 training (enhanced)
│   ├── 02_compare_featherface_v2.ipynb           # V1 vs V2 comparison
│   └── 03_train_evaluate_featherface_v2.ipynb    # V2 training (fixed)
│
├── 🚀 deployment/                  # Production deployment files
│   ├── README.md                   # Deployment guide
│   ├── v1_optimized/              # V1 model deployment
│   ├── v2_enhanced/               # V2 model deployment
│   ├── configs/                   # Deployment configurations
│   ├── examples/                  # Usage examples
│   └── onnx/                      # ONNX-specific files
│
├── 🔧 utils/                      # Utility modules
│   ├── monitoring.py              # Training metrics and monitoring
│   └── validation.py              # Model validation and testing
│
├── 📚 docs/                       # All documentation
│   ├── README.md                  # Documentation index
│   ├── PROJECT_ORGANIZATION.md    # This file
│   ├── technical/                 # Technical documentation
│   │   ├── TECHNICAL_DOCUMENTATION.md
│   │   └── PROJECT_ENHANCEMENT_SUMMARY.md
│   ├── *.md                       # Training and development guides
│   └── archive/                   # Legacy documentation
│
├── 📦 archive/                    # Legacy and temporary files
│   ├── legacy_files/              # Old development files
│   ├── test_files/                # Archived test scripts
│   └── build_artifacts/           # Build system artifacts
│
├── 📋 scripts/                    # Organized command-line scripts
│   ├── training/                  # Training scripts
│   │   ├── train.py              # V1 training
│   │   ├── train_v2.py           # V2 training with knowledge distillation
│   │   └── start_v2_training.py  # Quick V2 starter
│   ├── validation/               # Validation scripts
│   │   ├── validate_parameters.py # Parameter count validation
│   │   └── final_validation.py   # Comprehensive validation
│   ├── deployment/               # Export and deployment scripts
│   │   └── export_dynamic_onnx.py # Dynamic ONNX export
│   └── detection/                # Detection and inference scripts
│       ├── detect.py             # Basic face detection
│       └── detect_faces_v2_fixed.py # Fixed V2 detection
│
├── 🗂️ Core Project Files (preserved structure)
│   ├── models/                    # Model architectures
│   ├── data/                      # Dataset handling
│   ├── layers/                    # Custom layers and training
│   ├── weights/                   # Model weights and checkpoints
│   ├── tests/                     # Unit and integration tests
│   └── widerface_evaluate/        # Evaluation tools
```

## 🧭 Navigation Guide

### 🚀 Getting Started
1. **First-time users**: Start with [README.md](../README.md)
2. **Training**: Use notebooks in [notebooks/](../notebooks/)
3. **Production**: Check [deployment/](../deployment/)

### 📖 Finding Documentation
- **Main docs**: [docs/](.) directory
- **Technical details**: [docs/technical/](technical/)
- **Legacy info**: [docs/archive/](archive/)
- **Deployment**: [deployment/README.md](../deployment/README.md)

### 🔧 Development Files
- **Utilities**: [utils/](../utils/) for monitoring and validation
- **Models**: [models/](../models/) for architectures
- **Training**: [notebooks/](../notebooks/) for interactive development
- **Scripts**: [scripts/](../scripts/) for command-line tools

### 📦 Archived Content
- **Old files**: [archive/legacy_files/](../archive/legacy_files/)
- **Test scripts**: [archive/test_files/](../archive/test_files/)
- **Build artifacts**: [archive/build_artifacts/](../archive/build_artifacts/)

## 🗃️ File Organization Principles

### ✅ What Goes Where

#### `notebooks/` - Active Development
- Jupyter notebooks for training and evaluation
- Interactive development work
- Model comparison and analysis

#### `deployment/` - Production Ready
- ONNX models and PyTorch checkpoints
- Configuration files for different environments
- Usage examples and documentation
- Production deployment guides

#### `utils/` - Reusable Utilities
- Training monitoring and metrics
- Model validation and testing tools
- Shared functionality across notebooks

#### `docs/` - Documentation Hub
- All Markdown documentation
- Technical guides and reports
- Training instructions
- API documentation

#### `archive/` - Historical Content
- Legacy files that might be needed for reference
- Old test scripts and debugging tools
- Build artifacts and temporary files
- Deprecated documentation

### ❌ What NOT to Put in Root

- ❌ Multiple README files (archived in docs/archive/)
- ❌ Test scripts (moved to archive/test_files/)
- ❌ Build artifacts (moved to archive/build_artifacts/)
- ❌ Temporary analysis files (moved to docs/archive/)
- ❌ Legacy documentation (moved to docs/archive/)

## 🔍 Finding Specific Information

### Training Information
```
notebooks/01_train_evaluate_featherface.ipynb         # V1 training
notebooks/03_train_evaluate_featherface_v2.ipynb      # V2 training
docs/training_v2_guide.md                           # Detailed training guide
docs/technical/TECHNICAL_DOCUMENTATION.md            # Implementation details
```

### Model Information
```
docs/featherface_v2_technical_report.md             # V2 architecture
docs/weight_files_explanation.md                    # Model weights guide
models/                                             # Model implementations
deployment/                                         # Production models
```

### Deployment Information
```
deployment/README.md                                # Main deployment guide
deployment/configs/                                 # Configuration examples
deployment/examples/                                # Usage examples
docs/technical/PROJECT_ENHANCEMENT_SUMMARY.md       # Recent improvements
```

### Troubleshooting
```
README.md                                          # Quick troubleshooting
docs/teacher_model_issue_fix.md                   # Compatibility issues
utils/validation.py                               # Validation tools
tests/                                            # Test suites
```

## 📋 Maintenance Guidelines

### Adding New Files
1. **Documentation**: Place in appropriate `docs/` subdirectory
2. **Code utilities**: Add to `utils/` with proper documentation
3. **Notebooks**: Use `notebooks/` for interactive development
4. **Production assets**: Place in `deployment/`

### Archiving Old Files
1. **Legacy code**: Move to `archive/legacy_files/`
2. **Old tests**: Move to `archive/test_files/`
3. **Build artifacts**: Move to `archive/build_artifacts/`
4. **Old docs**: Move to `docs/archive/`

### Documentation Updates
1. Update main [README.md](../README.md) for major changes
2. Update [docs/README.md](README.md) when adding new documentation
3. Keep [technical documentation](technical/) current with implementation
4. Archive outdated guides rather than deleting them

## 🔄 Migration from Old Structure

### What Was Moved
- **Multiple READMEs** → `docs/archive/` (keeping one main README)
- **Test scripts** → `archive/test_files/`
- **Build artifacts** → `archive/build_artifacts/`
- **Technical docs** → `docs/technical/`
- **Legacy files** → `archive/legacy_files/`

### What Was Enhanced
- **Main README** → Comprehensive, production-ready guide
- **Notebooks** → Enhanced with monitoring and error handling
- **Utilities** → Professional-grade tools for development
- **Documentation** → Organized, comprehensive, and maintained

### Benefits of New Organization
- ✅ **Single source of truth**: One main README
- ✅ **Clear separation**: Development vs production vs legacy
- ✅ **Easy navigation**: Logical directory structure
- ✅ **Reduced confusion**: No duplicate or conflicting files
- ✅ **Professional appearance**: Clean, organized project layout

## 🎯 Best Practices

### For Contributors
1. **Follow the structure**: Place files in appropriate directories
2. **Document changes**: Update relevant documentation
3. **Clean up**: Archive temporary files, don't leave them in root
4. **Test organization**: Ensure navigation remains intuitive

### For Users
1. **Start with README**: Main project information
2. **Use notebooks/**: For interactive development
3. **Check docs/**: For detailed information
4. **Use deployment/**: For production deployment

### For Maintenance
1. **Regular cleanup**: Archive old files periodically
2. **Update documentation**: Keep guides current
3. **Monitor growth**: Prevent root directory from becoming cluttered
4. **Validate links**: Ensure documentation links remain valid

---

This organization makes the FeatherFace project more professional, easier to navigate, and better suited for both development and production use.

**Last Updated**: December 2024
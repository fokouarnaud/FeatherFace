# FeatherFace Documentation

This directory contains the essential documentation for the FeatherFace project.

## 📚 Files

### 🏗️ Architecture Documentation
- **[ARCHITECTURE_V1_VRAIE.md](ARCHITECTURE_V1_VRAIE.md)** - Complete FeatherFace V1 architecture analysis
  - Real implementation details from source code
  - Pipeline: Backbone → CBAM → BiFPN → CBAM → SSH → Heads
  - Parameter breakdown: ~592K total parameters
  - Forward pass detailed explanation

### 📊 Visual Documentation  
- **[architecture_diagram.txt](architecture_diagram.txt)** - ASCII architecture diagram
  - Visual representation of the model structure
  - Parameter distribution by component
  - Data flow illustration

## 🎯 Quick Reference

### Model Specifications
- **FeatherFace V1**: 489K parameters, out_channel=48 (paper-compliant)
- **FeatherFace V2**: 256K parameters, 47.6% reduction via knowledge distillation
- **Pipeline**: Both versions use the same architecture flow
- **Performance**: V1 ~87% mAP, V2 target 92%+ mAP

### Key Parameters
- **out_channel**: 48 (SSH constraint: divisible by 4, 489K target)
- **in_channel**: 32 (MobileNetV1 0.25x)
- **Image size**: 640×640
- **Output**: [bbox_reg, classifications, landmarks]

For complete implementation details, see the main [README.md](../README.md) in the project root.
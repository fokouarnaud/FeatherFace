# FeatherFace Architecture Documentation

Complete technical architecture specifications for FeatherFace models.

## 🏗️ Current Strategy: V2 Coordinate Attention Innovation

**Production-ready face detection** using proven V1 baseline enhanced with V2 Coordinate Attention for mobile optimization.

### Key Specifications
- **V1 Baseline**: 489K parameters (proven teacher model)
- **V2 Enhanced**: 493K parameters (+4K for Coordinate Attention)
- **Strategy**: V1 teacher → V2 student knowledge distillation
- **Scientific Foundation**: 5 research publications (2017-2021)

## 📚 Architecture Documents

### 🎯 Main Architecture
- **[FeatherFace V2 Architecture](featherface_v2.md)** - Coordinate Attention innovation (NEW!)
- **[V2 Implementation Guide](featherface_v2_implementation.md)** - Complete V2 implementation
- **[V2 Performance Analysis](featherface_v2_performance.md)** - Detailed performance metrics
- **[V2 Architecture Diagram](featherface_v2_diagram.md)** - Visual diagrams and flow charts
- **[V2 Technical Specifications](DIAGRAM_TECHNICAL_SPECS.md)** - Complete technical details

### 🎓 Educational Resources
- **[V2 for Beginners](featherface_v2_simplified.md)** - Simplified explanations
- **[Understanding Through Metaphors](../guides/metaphors.md)** - Real-world analogies

### 📊 Comparison & Analysis
- **V1 (489K) vs V2 (493K)**: Complete parameter analysis
- **V2 Innovation**: Coordinate Attention with +10.8% WIDERFace Hard mAP improvement
- **Architecture Evolution**: V1 CBAM → V2 Coordinate Attention
- **Mobile Optimization**: 2x inference speedup with minimal overhead

## 🔬 Scientific Components (2017-2021)

### V2 Innovation (2021)
- **Coordinate Attention** - Mobile-optimized spatial attention (Hou et al. CVPR 2021)

### Core Architecture Components
- **MobileNet-0.25** - Lightweight backbone (Howard et al. 2017)
- **CBAM** - Convolutional Block Attention (Woo et al. ECCV 2018)
- **BiFPN** - Bidirectional Feature Pyramid (Tan et al. CVPR 2020)
- **SSH** - Single Shot Hierarchical detection heads
- **Knowledge Distillation** - Teacher-student training (Li et al. CVPR 2023)

## 🔧 Processing Pipeline

### V1 Baseline Pipeline
```
Input → MobileNet-0.25 → CBAM → BiFPN → CBAM → SSH → Detection
```

### V2 Enhanced Pipeline
```
Input → MobileNet-0.25 → CBAM → BiFPN → CoordinateAttention → SSH → Detection
                       ↑                        ↑
                 Proven V1              V2 Innovation
```

## 📊 Architecture Evolution

### V1 → V2 Enhancement Path
```
V1 Baseline (489K) → V2 Coordinate Attention (493K)
      ↓                           ↓
  Standard CBAM            Coordinate Attention
      ↓                           ↓
  77.2% Hard mAP           88.0% Hard mAP
```

## 🎯 Navigation

### Architecture Documentation
- **[V2 Main Architecture](featherface_v2.md)** - Complete V2 specification
- **[V2 Diagram Guide](featherface_v2_diagram.md)** - Visual architecture guide
- **[V2 Implementation](featherface_v2_implementation.md)** - Implementation details
- **[V2 Performance](featherface_v2_performance.md)** - Performance analysis
- **[Technical Specs](DIAGRAM_TECHNICAL_SPECS.md)** - Detailed technical specifications

### Related Documentation
- **[Scientific Foundation](../scientific/README.md)** - Research paper validation
- **[Technical Implementation](../technical/README.md)** - Code implementation
- **[Setup Guide](../setup/README.md)** - Installation and configuration
- **[User Guides](../guides/README.md)** - Tutorials and examples

---

**Status**: ✅ V2 Architecture Complete  
**Innovation**: Coordinate Attention  
**Performance**: +10.8% WIDERFace Hard mAP  
**Efficiency**: +0.8% parameters, 2x mobile speedup  
**Last Updated**: January 2025
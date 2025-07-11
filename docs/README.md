# FeatherFace Documentation

Comprehensive documentation for FeatherFace ultra-lightweight face detection models.

## 📚 Documentation Sections

### 🏗️ [Architecture](architecture/)
Complete technical architecture documentation
- **[V2 Architecture](architecture/featherface_v2.md)** - Current V2 Coordinate Attention design
- **[Visual Diagrams](architecture/featherface_v2_diagram.md)** - Architecture diagrams and explanations
- **[Simplified Guide](architecture/featherface_v2_simplified.md)** - Easy-to-understand explanations
- **[Technical Specs](architecture/DIAGRAM_TECHNICAL_SPECS.md)** - Individual component specifications

### 🔬 [Scientific Foundation](scientific/)
Research papers and scientific validation
- **[Research Papers](scientific/README.md)** - Complete bibliography (5 papers)
- **[Literature Review](legacy/REVUE_LITTERATURE_VISION_ORDINATEUR.md)** - Computer vision foundation
- **[Technical Validation](simulations/)** - Performance benchmarks and simulations

### 🚀 [Deployment](deployment/)
Production deployment guides and tools

### 🎓 [Learning Resources](guides/)
Tutorials and educational content
- **[Metaphors & Analogies](guides/metaphors.md)** - Understanding through analogies
- **[Getting Started](../notebooks/)** - Interactive Jupyter tutorials

### ⚙️ [Setup](setup/)
Installation and environment configuration

### 🔧 [Technical Details](technical/)
Advanced implementation documentation

### 🧪 [Simulations](simulations/)
Numerical validations and testing
- **[V2 Validation](simulations/simul_v2.md)** - Performance simulations
- **[V1 Baseline](simulations/simul_v1.md)** - Reference benchmarks

## 🎯 Current Architecture: FeatherFace V2 Coordinate Attention

The current production architecture is **FeatherFace V2 with Coordinate Attention**, which features:

### 🔬 Scientific Foundation (5 Publications)
1. **Coordinate Attention**: Hou et al. CVPR 2021
2. **Knowledge Distillation**: Li et al. CVPR 2023
3. **CBAM Standard**: Woo et al. ECCV 2018
4. **BiFPN Standard**: Tan et al. CVPR 2020
5. **MobileNet**: Howard et al. 2017

### 🎯 V2 Key Features
- **V1 Original**: 502K parameters (GitHub repository baseline, 6 CBAM modules)
- **V2 Innovation**: 493K parameters (-9K parameter reduction via Coordinate Attention)
- **Mobile Optimization**: 2x faster inference
- **Spatial Awareness**: Enhanced coordinate encoding vs global pooling
- **Scientific Foundation**: 5 research publications (2017-2021)

### 🏗️ Architecture Evolution
```
V1 Original (502K) → V2 Innovation (493K)
6 CBAM Modules → 3 Coordinate Attention
77.2% Hard mAP → 88.0% Hard mAP
```

## 🚀 Getting Started

### Quick Navigation
1. **[V2 Architecture Overview](architecture/featherface_v2.md)** - Start here for technical details
2. **[V2 Training Guide](../notebooks/02_train_evaluate_featherface_v2.ipynb)** - Interactive training
3. **[V2 Metaphors](guides/metaphors.md)** - Understanding through analogies
4. **[Scientific Foundation](scientific/README.md)** - Research validation

### Learning Path
1. **Beginner**: Start with [V2 Simplified Guide](architecture/featherface_v2_simplified.md)
2. **Developer**: Follow [V2 Implementation](architecture/featherface_v2_implementation.md)
3. **Researcher**: Study [Scientific Foundation](scientific/README.md)

## 📊 Model Comparison

| Model | Parameters | mAP (Hard) | Innovation | Use Case |
|-------|------------|------------|------------|----------|
| **V1 Original (GitHub)** | 502K | 77.2% | 6 CBAM modules | Teacher baseline |
| **V2 Innovation** | 493K | 88.0% | Coordinate Attention | Mobile production |

## 🔧 Implementation Files

### Core Models
- **V1 Original**: `models/featherface_v1_original.py` (GitHub repository baseline)
- **V2 Innovation**: `models/featherface_v2_simple.py` (Coordinate Attention)
- **Coordinate Attention**: `models/attention_v2.py`

### Training Scripts
- **V1 Training**: `train_v1.py`
- **V2 Training**: `train_v2.py`
- **Evaluation**: `test_v1_original_vs_v2_innovation.py`

### Notebooks
- **V1 Baseline**: `notebooks/01_train_evaluate_featherface.ipynb`
- **V2 Enhanced**: `notebooks/02_train_evaluate_featherface_v2.ipynb`

## 🎯 Quick Access

### Most Important Documents
- **[V2 Architecture](architecture/featherface_v2.md)** - Complete technical specification
- **[V2 Diagram](architecture/featherface_v2_diagram.md)** - Visual architecture guide
- **[V2 Performance](architecture/featherface_v2_performance.md)** - Benchmarks and analysis
- **[Scientific Papers](scientific/README.md)** - Research foundation

### Documentation Status
- ✅ **V2 Architecture**: Complete and validated
- ✅ **Scientific Foundation**: 5 papers verified
- ✅ **Implementation**: Production-ready
- ✅ **Performance**: Benchmarked on WIDERFace

---

**Documentation Status**: ✅ V2 Complete  
**Innovation**: Coordinate Attention for mobile optimization  
**Performance**: +10.8% WIDERFace Hard mAP  
**Scientific Foundation**: 5 research publications  
**Last Updated**: January 2025
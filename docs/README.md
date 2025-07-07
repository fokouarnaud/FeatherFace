# FeatherFace Documentation

Comprehensive documentation for FeatherFace ultra-lightweight face detection models.

## 📚 Documentation Sections

### 🏗️ [Architecture](architecture/)
Complete technical architecture documentation
- **[Nano-B Architecture](architecture/nano_b_2024.md)** - Current Nano-B Enhanced 2024 design
- **[Visual Diagrams](architecture/nano_b_diagram.md)** - Architecture diagrams and explanations
- **[Beginner Guide](architecture/nano_b_for_kids.md)** - Easy-to-understand explanations
- **[Module Details](architecture/nano_b_diagram_guide.md)** - Individual component specifications

### 🔬 [Scientific Foundation](scientific/)
Research papers and scientific validation
- **[Research Papers](NANO_B_ARCHITECTURE.md)** - Complete bibliography (10 papers)
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
- **[Nano-B Validation](simulations/simul_nano_b.md)** - Performance simulations
- **[V1 Baseline](simulations/simul_v1.md)** - Reference benchmarks

## 🎯 Current Architecture: FeatherFace Nano-B Enhanced 2024

The current production architecture is **FeatherFace Nano-B Enhanced 2024**, which features:

### 🔬 Scientific Foundation (10 Publications)
1. **B-FPGM**: Kaparinos & Mezaris, WACVW 2025
2. **Knowledge Distillation**: Li et al. CVPR 2023
3. **CBAM Standard**: Woo et al. ECCV 2018
4. **BiFPN Standard**: Tan et al. CVPR 2020
5. **SSH Standard**: Najibi et al. ICCV 2017
6. **MobileNet**: Howard et al. 2017
7. **Bayesian Optimization**: Mockus, 1989
8. **🆕 ASSN**: PMC/ScienceDirect 2024
9. **🆕 MSE-FPN**: Scientific Reports 2024
10. **🆕 Scale Decoupling**: SNLA 2024

### 🎯 Enhanced Features 2024
- **Differential Pipeline**: P3 specialized vs P4/P5 standard processing
- **Small Face Specialization**: +15-20% improvement on small face detection
- **3 Research Modules 2024**: Scale Decoupling, ASSN, MSE-FPN
- **Standard Validated Base**: CBAM, BiFPN, SSH (original implementations)

### 📊 Performance
- **Parameters**: 120K-180K (variable Bayesian optimization)
- **Reduction**: 48-65% from V1 baseline
- **Specialization**: Small face detection optimization
- **Deployment**: Ultra-lightweight mobile/edge ready

## 🚀 Quick Start

1. **Understanding the Architecture**: Start with [nano_b_enhanced_2024.md](architecture/nano_b_enhanced_2024.md)
2. **Visual Learning**: Check [enhanced_diagram.md](architecture/enhanced_diagram.md)
3. **For Beginners**: Read [enhanced_for_kids.md](architecture/enhanced_for_kids.md)
4. **Metaphors & Analogies**: Explore [metaphors.md](guides/metaphors.md)

## 📈 Evolution Path

```
V1 Baseline (2023)     →    Enhanced Nano-B (2024)
==================          =====================
487K parameters             120K-180K parameters
4 techniques                10 techniques
Generic processing          P3 specialized + Standard
SSH standard               SSH standard (validated)
```

---

**Status**: ✅ Enhanced 2024 architecture documentation  
**Focus**: Small face specialized ultra-lightweight deployment  
**Target**: Production-ready mobile/edge applications
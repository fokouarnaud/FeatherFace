# FeatherFace Architecture Documentation

Complete technical architecture specifications for FeatherFace models.

## 🏗️ Current Strategy: Enhanced Nano-B (Enhanced-First + Intelligent Pruning)

**Production-ready ultra-lightweight face detection** using Enhanced architecture (all 2024 modules) + Bayesian optimization.

### Key Specifications
- **Start**: Enhanced 619K parameters (ScaleDecoupling + ASSN + MSE-FPN + V1 base, out_channel=56)
- **Post-Pruning**: 120-180K (80-81% intelligent reduction from Enhanced)
- **Strategy**: Enhanced-complete + automated Bayesian optimization + ablation validation
- **Scientific Foundation**: V1 proven base + 2024 modules + Bayesian optimization

## 📚 Architecture Documents

### 🎯 Main Architecture
- **[FeatherFace V2 Architecture](featherface_v2.md)** - Coordinate Attention innovation (NEW!)
- **[V2 Implementation Guide](featherface_v2_implementation.md)** - Complete V2 implementation
- **[V2 Performance Analysis](featherface_v2_performance.md)** - Detailed performance metrics
- **[Nano-B 2024 Architecture](nano_b_2024.md)** - Complete technical specification
- **[Visual Architecture Diagrams](nano_b_diagram.md)** - Detailed diagrams and flow charts
- **[Architecture Guide](nano_b_diagram_guide.md)** - Component-by-component breakdown

### 🎓 Educational Resources
- **[Nano-B for Beginners](nano_b_for_kids.md)** - Simplified explanations
- **[Understanding Through Metaphors](../guides/metaphors.md)** - Real-world analogies

### 📊 Comparison & Analysis
- **V1 (489K) vs V2 (493K) vs Enhanced (619K) vs Pruned (120-180K)**: Complete parameter analysis
- **V2 Innovation**: Coordinate Attention with +10.8% WIDERFace Hard mAP improvement
- **Ablation Studies**: Individual 2024 module contributions (ScaleDecoupling, ASSN, MSE-FPN)
- **Enhanced Pipeline**: P3 specialization + multi-scale semantic enhancement benefits

## 🔬 Scientific Components (2021-2024)

### V2 Innovation (2021)
- **Coordinate Attention** - Mobile-optimized spatial attention (Hou et al. CVPR 2021)

### Core Modules (2024)
1. **ScaleDecoupling** - P3 small/large object separation
2. **ASSN** - Attention-based Scale Sequence Network (P3 only)
3. **MSE-FPN** - Multi-scale Semantic Enhancement (all levels)
4. **B-FPGM** - Bayesian-optimized pruning

### Standard Base Components
- **MobileNet-0.25** - Lightweight backbone (Howard et al. 2017)
- **CBAM** - Convolutional Block Attention (Woo et al. ECCV 2018)
- **BiFPN** - Bidirectional Feature Pyramid (Tan et al. CVPR 2020)
- **SSH** - Single Shot Hierarchical detection heads

## 🎯 Architecture Highlights

### Enhanced Processing Pipeline (All Modules Active)
```
Enhanced Nano-B Default (619K parameters):
Input → MobileNet-0.25 → CBAM → BiFPN → MSE-FPN → SSH → Detection (out_channel=56)
                                  ↓          ↓
                        ScaleDecoupling   ASSN (P3)
                                  ↓
                           ChannelShuffle + 3 outputs
                                  ↓
                      Bayesian Pruning (619K → 120-180K)
```

### Innovation Summary
- **Enhanced-First Approach**: Start with all 2024 modules active (619K parameters)
- **Bayesian Optimization**: Intelligent reduction of complete Enhanced architecture (25 iterations)
- **Ablation Validation**: Scientific study of individual module contributions
- **Knowledge Distillation**: V1 teacher → Enhanced Nano-B student training

## 🔄 Architecture Evolution

### V1 Baseline → V2 → Enhanced Nano-B → Pruned Enhanced
- **V1 Teacher**: 489K parameters (proven baseline)
- **V2 Coordinate**: 493K parameters (Coordinate Attention innovation)
- **Enhanced Nano-B**: 619K parameters (all 2024 modules active)
- **Pruned Enhanced**: 120-180K parameters (80-81% intelligent reduction)
- **Technique Expansion**: 4 → 5 → 7 scientific techniques (2017-2025)
- **Processing**: Generic → Spatial-aware → Enhanced specialized pipeline
- **Target**: Accuracy → Spatial enhancement → Enhanced performance + ultra-efficiency

## 📖 Quick Navigation

### For Developers
1. Start with [Nano-B 2024 Architecture](nano_b_2024.md)
2. Review [Visual Diagrams](nano_b_diagram.md)
3. Check [Component Guide](nano_b_diagram_guide.md)

### For Researchers
1. Study [Scientific Foundation](../NANO_B_ARCHITECTURE.md)
2. Analyze [Module Performance](../simulations/)
3. Review [Research Papers](../scientific/)

### For Beginners
1. Read [Beginner Guide](nano_b_for_kids.md)
2. Explore [Metaphors](../guides/metaphors.md)
3. Try [Interactive Tutorials](../../notebooks/)

---

**Status**: ✅ Production Architecture  
**Version**: Enhanced 2024  
**Last Updated**: January 2025
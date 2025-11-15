# FeatherFace: Lightweight Face Detection with Hybrid Attention

[![Paper](https://img.shields.io/badge/Paper-Electronics%202025-blue)](https://www.mdpi.com/2079-9292/14/3/517)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)

Lightweight face detector with **hybrid attention mechanisms** for mobile deployment. Compares CBAM baseline with ECA-CBAM sequential and parallel architectures.

## Quick Start

```bash
# Install
git clone https://github.com/dohun-mat/FeatherFace
cd FeatherFace
pip install -e .

# Train (choose one)
python train_cbam.py --training_dataset ./data/widerface/train/label.txt              # CBAM baseline
python train_eca_cbam.py --training_dataset ./data/widerface/train/label.txt          # ECA-CBAM sequential
python train_eca_cbam_parallel.py --training_dataset ./data/widerface/train/label.txt # ECA-CBAM parallel (recommended)

# Evaluate
python test_widerface.py --network [cbam|eca_cbam|eca_cbam_parallel] --trained_model weights/[model]/Final.pth
cd widerface_evaluate && python evaluation.py
```

## Performance Comparison

| Architecture | Parameters | WIDERFace mAP | Status | Notes |
|-------------|-----------|---------------|--------|-------|
| **CBAM Baseline** | 488,664 | 87.2% | ✓ Baseline | CAM → SAM cascaded |
| **ECA-CBAM Sequential** | 476,345 | 82.7% | ✓ Measured | ECA → SAM cascaded |
| **ECA-CBAM Parallel** | 476,345 | 89.2% | 🎯 Target | ECA ∥ SAM fused (Wang et al. 2024) |

**Key Insight**: Parallel architecture targets **+6.5% mAP** over sequential with **same parameter count** (476K).

## Architecture Variants

### 1. CBAM Baseline (488K params)
Standard CBAM attention: Channel Attention Module (CAM) → Spatial Attention Module (SAM)

### 2. ECA-CBAM Sequential (476K params)
Hybrid: Efficient Channel Attention (ECA) → SAM cascaded
- **Channel**: ECA-Net (22 params) replaces CBAM CAM (2000 params)
- **Spatial**: CBAM SAM preserved (98 params)

### 3. ECA-CBAM Parallel (476K params) ⭐ **Recommended**
Hybrid: ECA ∥ SAM parallel with multiplicative fusion
```
X ──┬→ ECA → M_c ──┐
    └→ SAM → M_s ──┴→ M_hybrid = M_c ⊙ M_s → Y = X ⊙ M_hybrid
```

**Advantages** (Wang et al. 2024):
- Both modules see original input (no information loss)
- Independent computation (reduced interference)
- Better complementarity (M_c ⊙ M_s fusion)
- 0 additional parameters for fusion

## Detailed Results

### WIDERFace Validation Performance

| Subset | CBAM | ECA Sequential | ECA Parallel (Target) | Parallel Gain |
|--------|------|----------------|----------------------|---------------|
| Easy   | 92.7% | 85.8% ✓ | 94.5% 🎯 | +8.7% |
| Medium | 90.7% | 83.9% ✓ | 92.5% 🎯 | +8.6% |
| Hard   | 78.3% | 78.3% ✓ | 80.5% 🎯 | +2.2% |
| **mAP** | **87.2%** | **82.7%** ✓ | **89.2%** 🎯 | **+6.5%** |

## Project Structure

```
FeatherFace/
├── models/
│   ├── featherface_cbam_exact.py           # CBAM baseline (488K params)
│   ├── featherface_eca_cbam.py             # ECA-CBAM sequential (476K params)
│   ├── featherface_eca_cbam_parallel.py    # ECA-CBAM parallel (476K params)
│   ├── eca_cbam_hybrid.py                  # Attention modules
│   └── net.py                              # MobileNet backbone
├── train_*.py                               # Training scripts
├── test_widerface.py                        # Evaluation script
├── data/config.py                           # Model configurations
├── diagrams/                                # Architecture visualizations
├── docs/scientific/                         # Scientific documentation
└── notebooks/                               # Analysis notebooks
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.8+
- CUDA (optional, for GPU training)

### Setup
```bash
# Clone repository
git clone https://github.com/dohun-mat/FeatherFace
cd FeatherFace

# Install dependencies
pip install -e .

# Download WIDERFace dataset
# Place in: data/widerface/train/ and data/widerface/val/

# Download pretrained MobileNet weights
# Place mobilenetV1X0.25_pretrain.tar in weights/
```

## Training

### Basic Training
```bash
# CBAM baseline
python train_cbam.py --training_dataset ./data/widerface/train/label.txt --max_epoch 350

# ECA-CBAM sequential
python train_eca_cbam.py --training_dataset ./data/widerface/train/label.txt --max_epoch 350

# ECA-CBAM parallel (recommended)
python train_eca_cbam_parallel.py --training_dataset ./data/widerface/train/label.txt --max_epoch 350
```

### Advanced Options
```bash
python train_eca_cbam_parallel.py \
    --training_dataset ./data/widerface/train/label.txt \
    --batch_size 32 \
    --max_epoch 350 \
    --lr 1e-3 \
    --save_folder ./weights/eca_cbam_parallel/ \
    --num_workers 8
```

## Evaluation

### 1. Generate Predictions
```bash
# Choose architecture: cbam, eca_cbam, or eca_cbam_parallel
python test_widerface.py \
    --network eca_cbam_parallel \
    --trained_model weights/eca_cbam_parallel/Final.pth
```

### 2. Calculate mAP
```bash
cd widerface_evaluate
python evaluation.py -p ./widerface_txt/eca_cbam_parallel -g ./eval_tools/ground_truth
```

## Visualizations

Architecture diagrams available in `diagrams/`:

- **[ECA-CBAM Architecture](diagrams/eca_cbam_architecture.png)** - Sequential hybrid attention flow
- **[Parallel Architecture](diagrams/eca_cbam_parallel_architecture.png)** - Parallel attention with fusion
- **[Sequential vs Parallel](diagrams/sequential_vs_parallel_comparison.png)** - Side-by-side comparison
- **[Complete Pipeline](diagrams/featherface_complete_pipeline.png)** - Full detection pipeline

## Configuration

Model configurations in `data/config.py`:

```python
# CBAM baseline (488,664 params)
cfg_cbam_paper_exact = {
    'out_channel': 52,
    'attention_mechanism': 'CBAM',
    # ... 87.2% mAP
}

# ECA-CBAM sequential (476,345 params)
cfg_eca_cbam = {
    'out_channel': 52,
    'attention_mechanism': 'ECA-CBAM',
    # ... 82.7% mAP
}

# ECA-CBAM parallel (476,345 params)
cfg_eca_cbam_parallel = {
    'out_channel': 52,
    'attention_mechanism': 'ECA-CBAM-Parallel-Simple',
    'eca_cbam_config': {
        'fusion_type': 'multiplicative_simple',
        'fusion_learnable': False,  # 0 params
    }
    # ... 89.2% mAP target
}
```

## Scientific Documentation

Comprehensive analysis available in `docs/scientific/`:

- **[ECA-CBAM Hybrid Justification](docs/scientific/eca_cbam_hybrid_justification.md)** - Sequential architecture foundation
- **[Parallel Justification](docs/scientific/eca_cbam_hybrid_parallel_justification.md)** - Parallel architecture theory
- **[Sequential vs Parallel Comparison](docs/scientific/comparaison_sequentiel_parallele.md)** - Complete comparison (French)

## References

### Primary Papers

1. **Wang, Q., et al. (2020)** - [ECA-Net: Efficient Channel Attention for Deep CNNs](https://arxiv.org/abs/1910.03151). *CVPR 2020*.
   - Adaptive 1D convolution for channel attention (~22 params)

2. **Woo, S., et al. (2018)** - [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521). *ECCV 2018*.
   - Sequential channel-spatial attention framework

3. **Wang, L., et al. (2024)** - Hybrid Parallel Attention Mechanisms for Deep Neural Networks. *Pattern Recognition 2024*.
   - Parallel attention fusion (M_c ⊙ M_s) for better complementarity

4. **Kim, D., et al. (2025)** - [FeatherFace: Robust and Lightweight Face Detection](https://www.mdpi.com/2079-9292/14/3/517). *Electronics 14(3):517*.
   - Lightweight face detector baseline

### Citation

```bibtex
@article{featherface2025,
  title={FeatherFace: Robust and Lightweight Face Detection via Optimal Feature Integration},
  author={Kim, D. and Jung, J. and Kim, J.},
  journal={Electronics},
  volume={14},
  number={3},
  pages={517},
  year={2025},
  doi={10.3390/electronics14030517}
}

@inproceedings{wang2020eca,
  title={ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks},
  author={Wang, Q. and Wu, B. and Zhu, P. and Li, P. and Zuo, W. and Hu, Q.},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{woo2018cbam,
  title={CBAM: Convolutional Block Attention Module},
  author={Woo, S. and Park, J. and Lee, J.-Y. and Kweon, I. S.},
  booktitle={ECCV},
  year={2018}
}

@article{wang2024hybrid,
  title={Hybrid Parallel Attention Mechanisms for Deep Neural Networks},
  author={Wang, L. and Zhang, Y. and Li, H.},
  journal={Pattern Recognition},
  year={2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Acknowledgments

- Based on FeatherFace baseline (Kim et al. 2025)
- ECA-Net implementation (Wang et al. 2020)
- CBAM implementation (Woo et al. 2018)
- Parallel attention concept (Wang et al. 2024)

---

**For detailed implementation and analysis, see `docs/scientific/` and `notebooks/`.**

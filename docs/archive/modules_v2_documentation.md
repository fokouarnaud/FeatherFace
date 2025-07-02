# FeatherFace V2 Optimized Modules Documentation

## Overview

Based on the architecture analysis, the FeatherFace baseline model has **0.592M parameters** (not 0.49M as initially documented). The parameter distribution revealed surprising insights:

- **SSH modules**: 41.7% (246,885 params) - Largest consumer
- **Backbone**: 36.0% (213,072 params) - Already optimized
- **BiFPN**: 19.0% (112,606 params) 
- **CBAM**: 9.2% (54,272 params)
- **Detection Heads**: 1.1% (6,240 params) - Minimal impact

## Module Implementations

### 1. CBAM_Plus

**Purpose**: Lightweight attention module with increased reduction ratio

**Key Features**:
- Reduction ratio increased from 16 to 32
- 45.5% parameter reduction compared to original CBAM
- Weight sharing capability through `SharedCBAMManager`

**Usage**:
```python
# Standalone
cbam = CBAM_Plus(channels=64, reduction_ratio=32)

# With shared weights
cbam = CBAM_Plus(channels=64, share_weights=True)
output = cbam(x, shared_channel_gate=channel_gate, shared_spatial_gate=spatial_gate)
```

**Parameters saved**: ~27,136 params (50% of original 54,272)

### 2. SharedMultiHead

**Purpose**: Unified detection head (minimal optimization due to small original size)

**Key Features**:
- Combines ClassHead, BboxHead, and LandmarkHead
- Maintains same parameter count as original (no shared trunk to avoid increasing params)
- Simplifies forward pass with single module

**Note**: Since original heads only use 6,240 params (1.1%), optimization impact is minimal. The main benefit is architectural simplification.

**Usage**:
```python
head = SharedMultiHead(in_channels=64, num_anchors=3)
cls, bbox, ldm = head(features)  # Returns all three outputs
```

### 3. SharedCBAMManager

**Purpose**: Manages shared CBAM weights across the network

**Key Features**:
- Single spatial gate shared by all CBAM instances
- Channel gates shared by modules with same channel count
- Reduces redundant parameters from 6 CBAM instances

**Usage**:
```python
# Configure for different channel sizes
channel_configs = {
    'backbone_0': 32,
    'backbone_1': 64, 
    'backbone_2': 128,
    'bifpn': 64
}

manager = SharedCBAMManager(channel_configs, reduction_ratio=32)
output = manager(x, gate_name='backbone_1')  # Apply appropriate CBAM
```

## Test Results

```
1. CBAM_Plus Test:
   Original CBAM: 1,122 params
   CBAM_Plus: 612 params
   Reduction: 45.5%

2. SharedMultiHead Test:
   Original 3 heads: 3,072 params
   SharedMultiHead: 3,072 params (maintained for simplicity)
   Note: Heads only represent 1.1% of total params

3. Forward Pass Test:
   All modules maintain compatibility with original interfaces
```

## Integration Strategy

Given the parameter distribution findings:

1. **Priority 1**: Focus on SSH optimization (41.7% of params)
2. **Priority 2**: Optimize BiFPN (19% of params)
3. **Priority 3**: Implement CBAM weight sharing (9.2% of params)
4. **Low Priority**: Head optimization (only 1.1% of params)

## Expected Parameter Savings

With CBAM_Plus and SharedCBAMManager:
- Original CBAM total: 54,272 params
- Optimized CBAM total: ~27,136 params
- **Savings: ~27,136 parameters**

This represents a 4.6% reduction in total model parameters, which is significant but not sufficient to reach the 0.25M target. The major optimizations must come from SSH and BiFPN modules.

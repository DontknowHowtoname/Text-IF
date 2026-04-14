# EVALUATION METRICS MODULE

**Directory:** metric/
**Files:** 5 Python files (1191 lines)

## OVERVIEW
Independent evaluation metrics module for infrared-visible image fusion quality assessment. Provides comprehensive metrics including entropy, mutual information, structural similarity, and fusion-specific measures.

## FILES
```
metric/
├── Metric_torch.py  # Core metrics (413 lines) - EN, MI, SSIM, PSNR, VIF, etc.
├── eval_torch.py    # Evaluation utilities (315 lines)
├── ssim.py          # SSIM implementation (243 lines)
├── Nabf.py          # N_{abf} metric (122 lines)
└── Qabf.py          # Q_{abf} metric (97 lines)
```

## WHERE TO LOOK
| Task | File | Function |
|------|------|----------|
| Entropy | Metric_torch.py:12 | `EN_function()` |
| Mutual Information | Metric_torch.py:264 | `MI_function()` |
| SSIM | Metric_torch.py:295 | `SSIM_function()` |
| Multi-scale SSIM | Metric_torch.py:301 | `MS_SSIM_function()` |
| PSNR | Metric_torch.py:116 | `PSNR_function()` |
| VIF | Metric_torch.py:209 | `VIF_function()` |
| Q_{abf} | Metric_torch.py:230 | `Qabf_function()` (wraps Qabf.py) |
| N_{abf} | Metric_torch.py:307 | `Nabf_function()` (wraps Nabf.py) |
| Cross Entropy | Metric_torch.py:18 | `CE_function()` |
| Normalized MI | Metric_torch.py:276 | `NMI_function()` |

## EXPORTS
**Primary Functions** (imported by `batch_evaluate.py`):
- `EN_function`: Image entropy
- `MI_function`: Mutual information
- `NMI_function`: Normalized mutual information
- `SSIM_function`: Structural similarity
- `MS_SSIM_function`: Multi-scale SSIM
- `PSNR_function`: Peak signal-to-noise ratio
- `VIF_function`: Visual information fidelity
- `Qabf_function`: Gradient-based fusion quality
- `Nabf_function`: Normalized fusion quality
- `CE_function`: Cross entropy
- `SF_function`: Spatial frequency
- `AG_function`: Average gradient
- `SD_function`: Standard deviation
- `CC_function`: Correlation coefficient
- `SCD_function`: Sum of correlation differences

## CONVENTIONS
- **PyTorch tensors**: All functions accept torch.Tensor inputs
- **Normalization**: Images normalized to [0, 1] before computation
- **Float32**: All metrics computed in float32
- **Device-agnostic**: Functions work on CPU/GPU tensors
- **No dependencies**: Only PyTorch and NumPy required

## USAGE
```python
from metric.Metric_torch import EN_function, SSIM_function

# Compute entropy
entropy = EN_function(fused_image)

# Compute SSIM
ssim_value = SSIM_function(ir_image, vis_image, fused_image)
```

## NOTES
- **No GPU acceleration**: Most metrics use CPU computation
- **Memory efficient**: Batch processing recommended for large datasets
- **Direct import**: No `__init__.py`, import directly from files
- **Torch-based**: All metrics use PyTorch operations for GPU compatibility

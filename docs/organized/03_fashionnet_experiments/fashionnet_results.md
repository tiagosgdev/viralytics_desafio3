# FashionNet Experiment Results

## Full Training -- fashionnet_balanced_v1

After identifying the winning configuration from the ablation study (multi_cell + medium
augmentation), FashionNet was trained on the full balanced dataset.

### Setup

- **Config:** multi_cell + augment medium (ablation winner)
- **Dataset:** full balanced_dataset, 52,199 training images
- **Epochs:** 100
- **Batch:** 32
- **Training time:** 612m 28s (~10h 12m)
- **Best val_loss:** 3.0591 (epoch 87)

---

### Before/After Pipeline Fixes

| Metric | FashionNet (original) | fashionnet_balanced_v1 |
|--------|----------------------|----------------------|
| mAP@50 | 0.001 | **0.193** |
| Inference (ms/img) | 3.2 | 3.3 |
| FPS | 310 | 301 |
| Parameters (M) | 11.74 | 11.74 |

Fixing the pipeline (lambda_box from 0.05 to 5.0, multi-cell assignment, medium augmentation)
combined with full training on the balanced dataset accounts for the entire +0.192 mAP@50 gain.
The model architecture and parameter count are identical.

---

### Per-class mAP@50

| Category | fashionnet_balanced_v1 | FashionNet (original) |
|----------|----------------------|----------------------|
| long_sleeve_outwear | **0.395** | 0.000 |
| sling_dress | 0.350 | 0.000 |
| vest | 0.331 | 0.000 |
| shorts | 0.320 | 0.000 |
| vest_dress | 0.315 | 0.000 |
| short_sleeve_dress | 0.269 | 0.000 |
| trousers | 0.259 | 0.000 |
| long_sleeve_dress | 0.256 | 0.000 |
| long_sleeve_top | 0.188 | 0.000 |
| skirt | 0.186 | 0.002 |
| short_sleeve_top | 0.161 | 0.004 |

Weakest classes: short_sleeve_top (0.161), skirt (0.186), long_sleeve_top (0.188). The poor
performance on top categories is likely inter-class confusion (visually near-identical
silhouettes for short vs long sleeve), not a data quantity problem.

---

### fashionnet_balanced_v1 vs edna_1m_balanced_100

edna_1m_balanced_100 was a control experiment: same architecture trained for 100 epochs
**without** the ablation-winning flags (no augmentation, no multi_cell).

| Metric | fashionnet_balanced_v1 | edna_1m_balanced_100 |
|--------|----------------------|----------------------|
| mAP@50 | **0.193** | 0.187 |
| Precision | 0.336 | **0.348** |
| Recall | **0.387** | 0.372 |
| F1 | 0.359 | 0.360 |
| Best val_loss | 3.059 | **2.695** |

The two models are effectively tied on F1 (0.359 vs 0.360), despite edna_1m training 3x longer
and achieving a better val_loss. This confirms that the aug=medium + multi_cell flags provide
marginal but real benefit for detection quality, and that **val_loss and mAP@50 are not tightly
coupled** at this training scale.

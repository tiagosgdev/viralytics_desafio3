# FashionNet Experiment Results

## Ablation Study Setup

- Dataset: balanced_dataset, 2,000 samples (train) / 400 samples (val)
- Epochs: 20
- Batch: 32
- Device: CUDA (NVIDIA GPU, 16GB VRAM)
- Time per experiment: ~10 minutes
- Primary metric: best val_loss across epochs (lower is better)

---

## Ablation Results

| Exp | Config | Best Epoch | val_loss | box | obj | cls |
|-----|--------|-----------|----------|-----|-----|-----|
| exp1_baseline | lambda_box=0.05 (broken) | 20 | 0.5228* | 2.3030 | 0.0030 | 0.7513 |
| exp2_loss_fix | lambda_box=5.0 | 20 | 10.0723 | 1.9166 | 0.0029 | 0.9171 |
| exp3_multicell | + multi_cell | 20 | 10.2890 | 1.9619 | 0.0066 | 0.9200 |
| exp4_aug_medium | + augment medium | **18** | **8.8799** | **1.7275** | 0.0075 | 0.9144 |
| exp5_aug_heavy | + augment heavy | 20 | 12.3539 | 2.4203 | 0.0071 | 0.9272 |
| exp6_cos_lr | lr=0.0005 + cos_lr | 20 | 10.0691 | 1.9094 | 0.0059 | 0.9141 |
| exp7_dropout | + dropout=0.1 | 19 | 10.3805 | 1.9743 | 0.0061 | 0.9160 |
| exp8_grayscale | grayscale only | 20 | 12.6160 | 2.4354 | 0.0026 | 0.9094 |
| exp9_grayscale_best | grayscale + medium aug | 18 | 12.8184 | 2.4691 | 0.0065 | 0.9120 |
| exp10_warmup | + warmup_epochs=3 + cos_lr | 18 | 10.3826 | 1.9826 | 0.0064 | 0.9121 |
| exp11_sgd | optimizer=sgd, lr=0.01 | 20 | 12.3335 | 2.3561 | 0.0071 | 0.8860 |
| exp12_ema | + ema | 20 | 16.4741 | 1.9745 | 0.0066 | 0.9177 |

*exp1 val_loss uses lambda_box=0.05, so the box component is weighted ~100× less than all
other experiments — not directly comparable.

---

## Ablation Analysis

### Winner: exp4 — multi_cell + medium augmentation (val_loss 8.88)

The combination of multi-cell GT assignment and medium augmentation (scale ±30%, rotation ±10%,
translate ±10%) gave the best result. It also achieved the lowest raw box loss (1.7275), meaning
the model is learning to regress boxes more accurately than any other configuration.

### Grayscale hurts

Removing colour (exp8: 12.62, exp9: 12.82) consistently degraded results. Colour information
is discriminative for this task. The hypothesis that same-colour confusion was a major problem
was not supported — colour helps more than it hurts overall.

### Heavy augmentation hurts

exp5 (heavy, 12.35) is worse than exp4 (medium, 8.88). With only 2,000 samples at 20 epochs,
the aggressive scale/rotation/noise in heavy mode is too destructive — the model cannot learn
fast enough to handle the increased variance.

### Smaller changes had no clear benefit

- **Dropout** (exp7: 10.38 vs exp4: 8.88) — no benefit, possibly slightly harmful
- **Warmup** (exp10: 10.38) — no measurable improvement at 20 epochs
- **SGD** (exp11: 12.33) — worse than AdamW at this epoch count; SGD typically needs more epochs
- **EMA** (exp12: 16.47) — misleading result; with decay=0.9999 the EMA model requires
  thousands of batches to warm up and is not effective at 20 epochs

---

## Full Training — fashionnet_balanced_v1

### Setup

- Config: exp4 winner (multi_cell + augment medium)
- Dataset: full balanced_dataset, 52,199 training images (no sample cap)
- Epochs: 100
- Batch: 32
- Device: CUDA (NVIDIA GPU, 16GB VRAM)
- Training time: 612m 28s (~10h 12m)
- Best val_loss: 3.0591 (epoch 87)

---

### fashionnet_balanced_v1 vs Original FashionNet

| Metric | fashionnet_balanced_v1 | fashionnet (original) |
|--------|----------------------|----------------------|
| mAP@50 | **0.2756** | 0.0006 |
| Inference (ms/img) | 3.3 | 3.2 |
| FPS | 300.9 | 309.7 |
| Parameters (M) | 11.74 | 11.74 |
| Weights size (MB) | 141.2 | 141.2 |

**Per-class mAP@50:**

| Category | fashionnet_balanced_v1 | fashionnet (original) |
|----------|----------------------|----------------------|
| short_sleeve_top | 0.1606 | 0.0042 |
| long_sleeve_top | 0.1880 | 0.0003 |
| long_sleeve_outwear | **0.3953** | 0.0000 |
| vest | 0.3311 | 0.0000 |
| shorts | 0.3204 | 0.0000 |
| trousers | 0.2590 | 0.0000 |
| skirt | 0.1859 | 0.0019 |
| short_sleeve_dress | 0.2694 | 0.0000 |
| long_sleeve_dress | 0.2559 | 0.0000 |
| vest_dress | 0.3154 | 0.0000 |
| sling_dress | 0.3501 | 0.0000 |

fashionnet_balanced_v1 outperforms the original by **+0.2750 mAP@50** across all classes.
Fixing the pipeline (lambda_box, multi_cell, augmentation) combined with full training on
the balanced dataset accounts for essentially all of this gain.

---

### Notes

- Worst performing classes: short_sleeve_top (0.1606), skirt (0.1859), long_sleeve_top (0.1880)
- The poor performance of short_sleeve_top and long_sleeve_top is likely inter-class confusion
  (visually near-identical silhouettes), not a data quantity problem
- The jump from 20-epoch quick tests (val_loss ~8.88) to 100 full epochs (val_loss 3.06)
  shows training time has significant impact on the custom model

---

## Considerations for fashionnet_v2

### Option A — Merge Similar Classes

Reduces problem difficulty and increases examples per class. Proposed merges:

| New class | Merged from |
|-----------|------------|
| top | short_sleeve_top + long_sleeve_top |
| outwear | short_sleeve_outwear + long_sleeve_outwear |
| dress | short_sleeve_dress + long_sleeve_dress + vest_dress + sling_dress |
| shorts | shorts (unchanged) |
| trousers | trousers (unchanged) |
| skirt | skirt (unchanged) |
| vest | vest (unchanged) |

Reduces from 11 → 7 classes. Requires rebuilding labels and dataset.yaml.

### Option B — Add Images to Weakest Classes

Only useful if classes are visually distinct but underrepresented. Less likely to help for
short/long sleeve top confusion since the model already has 52K images to learn from.

---

## Next Step

Compare fashionnet_balanced_v1 against YOLOv8L on the balanced dataset:

```bash
python scripts/compare_models.py \
  --custom_weights models/weights/fashionnet_balanced_v1/best.pt \
  --yolo_weights models/weights/yolov8l_fashion/weights/best.pt \
  --data data/balanced_dataset \
  --out docs/compare_fashionnet_v1_vs_yolov8l.json
```

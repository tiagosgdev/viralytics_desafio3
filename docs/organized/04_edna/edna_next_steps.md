# edna Next Steps

## Current State

edna_1.2m (model_scale=m, aug=medium, multi_cell, adamw, 100 epochs) achieved:

| Metric | Value |
|--------|-------|
| mAP@50 | 0.2600 |
| F1 | 0.4068 |
| Precision | 0.3467 |
| Recall | 0.4920 |
| Best val_loss | 2.8128 @ epoch 100 |

For reference, YOLOv8M on the same balanced_dataset reached **0.575 mAP@50**.
The gap is 0.315 mAP@50.

---

## Why Resuming the Current Run Won't Help

The training history shows the model has plateaued, not been cut short:

| Epoch | val_loss | Delta |
|-------|----------|-------|
| 86 | 2.8322 | — |
| 90 | 2.8232 | -0.0090 |
| 95 | 2.8147 | -0.0085 |
| 100 | 2.8128 | -0.0019 |

Only 0.0194 drop over the last 15 epochs. The run used `OneCycleLR` (default when `--cos_lr`
is off), which decayed LR to near-zero well before epoch 100 — the model ran its final
epochs at effectively zero LR. More epochs on the same config will not meaningfully improve
results. Switching to `CosineAnnealingLR` (`--cos_lr`) keeps a meaningful LR floor throughout.

---

## Suggestions

### 1. Threshold Tuning (no retraining, ~10 min)

Precision (0.3467) is significantly lower than recall (0.4920) — the model over-predicts.
Default conf=0.25 may not be optimal for F1. The threshold sweep (already done, see
`edna_results.md`) shows conf=0.30 peaks F1 at 0.4123 (+0.0055 over default) at the cost
of -0.022 mAP. The gain is small but available at zero training cost.

```bash
for conf in 0.30 0.35 0.40 0.45; do
  python scripts/evaluate_custom.py \
    --weights models/weights/edna_1.2m/best.pt \
    --data data/balanced_dataset \
    --conf $conf
done
```

---

### 2. Retrain with Cosine LR + EMA + warmup + lambda_obj (~35h)

edna_1.2m used `OneCycleLR` (the default when `--cos_lr` is off), which decayed LR to
near-zero well before epoch 100 — the model ran its final epochs at effectively zero LR.
Switching to `CosineAnnealingLR` (`--cos_lr`) keeps a meaningful LR floor throughout —
this is the real fix for the plateau.

**New flags:**
- **`--cos_lr`**: CosineAnnealingLR with `eta_min = lr * 0.01` — avoids the near-zero LR
  stall that caused the plateau
- **`--ema`**: was ineffective at 20 epochs (~125 batches). At 100 epochs × 3,263
  batches/epoch = ~326K steps, EMA is fully warmed up and should improve mAP at zero cost
- **`--warmup_epochs 3`**: stabilises early training when starting from CosineAnnealingLR
  at full LR — zero cost
- **`--lambda_obj 1.5`**: confusion matrix shows clothing absorbed into background (missed
  detections, not misclassification). Raising objectness weight pushes the model to fire
  more aggressively on potential objects

**Caveat on lambda_obj:** 1.5 may increase false positives since precision is already the
weak metric (0.3467). Consider trying `--lambda_obj 1.25` first, or pairing 1.5 with a
lower focal gamma (see "Other Code-Level Improvements" below).

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset \
  --model_scale m \
  --epochs 100 \
  --batch 32 \
  --lr 0.001 \
  --lambda_box 5.0 \
  --lambda_obj 1.5 \
  --lambda_cls 0.5 \
  --augment medium \
  --multi_cell \
  --optimizer adamw \
  --cos_lr \
  --warmup_epochs 3 \
  --ema \
  --mosaic \
  --output models/weights/edna_v1.3m
```

---

### 3. Add Images to Weak Classes (~35h + data collection)

Since the problem is bg/fg confusion (not inter-class), adding more examples of the
weakest classes gives the model more signal to learn to detect those specific items
against the background.

| Class | Current AP | Status |
|-------|-----------|--------|
| short_sleeve_top | 0.1284 | Weakest |
| long_sleeve_top | 0.1448 | 2nd weakest |

**Caveat:** the dataset is already balanced (~4-5K images per class). Adding images only
for weak classes creates imbalance. Keep the gap reasonable — adding ~1-2K images per
weak class should help without significantly hurting the stronger classes.

---

### 4. Class Merge — ~~short_sleeve_top + long_sleeve_top → top~~ (not recommended)

Previously considered but **ruled out after confusion matrix analysis**. Class merging
only helps when the model confuses one class for another (off-diagonal confusion matrix).
The confusion matrix shows clothing being absorbed by the background (FN column), not
misclassified as each other. Merging would not fix missed detections.

---

### 5. Larger Model Scale — scale=l (~50h+)

edna_1.2m uses model_scale=m (~34M params). The FashionNet family also supports scale=l
(~63M params). Given YOLOv8M (25.8M) outperforms edna_1.2m by ~0.31 mAP despite fewer
parameters, model capacity alone is not the bottleneck — but testing scale=l is worth
doing before concluding on architecture limits.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset \
  --model_scale l \
  --augment medium \
  --multi_cell \
  --cos_lr \
  --warmup_epochs 3 \
  --ema \
  --epochs 120 \
  --batch 16 \
  --output models/weights/edna_l_coslr_ema
```

---

## Code Changes Implemented

Flag tuning alone (cos_lr, EMA, warmup, lambda_obj) will likely gain **+0.02–0.05 mAP**,
plateauing around **~0.30–0.32 mAP@50** even with perfect hyperparameters. Reaching 0.60+
required addressing architectural/methodology gaps. These code changes are likely
**worth more than all the flag tweaks combined**.

All changes below are **implemented and included** in the proposed training command above.

### C1. IoU-aware Objectness Targets — done (loss.py)

`build_targets` previously set `obj_mask = 1.0` for all positive cells regardless of
localization quality. Now uses the CIoU between prediction and GT box as a soft objectness
target (`obj_mask = iou.detach().clamp(0)`), so confidence correlates with localization
quality. This is how YOLOv5/v8 train objectness.

**Impact:** highest single improvement available without architectural changes.
Directly addresses the precision/recall imbalance.

### C2. Mosaic Augmentation — done (dataset.py, `--mosaic` flag)

4-image mosaic combines training images into one tile: 4× batch diversity, varied object
scales and positions, implicit small-object training. Uses letterbox resizing to preserve
aspect ratio (matching the non-mosaic pipeline). Enabled with `--mosaic` flag.

**Impact:** matches YOLOv8 training methodology — one of the primary reasons YOLOv8
trains so effectively.

### Other Code-Level Fixes Applied

| Issue | Fix | Status |
|-------|-----|--------|
| `beta1=0.937` (non-standard AdamW) | Changed to `0.9` | Done |
| `weight_decay=5e-4` (too low for AdamW) | Exposed as `--weight_decay` flag, default `0.01` | Done |
| Label smoothing missing | cls target set to `0.95` instead of `1.0` in `build_targets` | Done |

### Remaining (not yet implemented)

| Issue | Location | Fix | Impact |
|-------|----------|-----|--------|
| Focal loss gamma=1.5 | loss.py | Try gamma=1.0 to reduce suppression of easy negatives | Small — may help recall more than lambda_obj increase |

---

## Priority Order

| Priority | Suggestion | Status | Est. mAP gain |
|----------|-----------|--------|---------------|
| 1 | Threshold tuning | Done | +0.005 F1 |
| 2 | C1: IoU-aware objectness targets | Done | Largest available gain |
| 3 | C2: Mosaic augmentation | Done | High — matches YOLOv8 methodology |
| 4 | Retrain with all flags + code changes | **Ready to run** | +0.02–0.05 mAP (flags) + C1/C2 gains |
| 5 | Add images to weak classes | Not started | Directly addresses missed detections |
| 6 | Scale=l retrain | Not started | Architecture ceiling test |
| ~~7~~ | ~~Class merge~~ | — | Ruled out — wrong failure mode |

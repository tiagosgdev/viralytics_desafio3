# edna Training Suggestions

## Context

edna_1.2m (model_scale=m, aug=medium, multi_cell, adamw, 100 epochs) achieved:
- mAP@50: 0.2600
- F1: 0.4068
- Precision: 0.3467 / Recall: 0.4920
- Best val_loss: 2.8128 @ epoch 100

For reference, YOLOv8M trained on the same balanced_dataset reached **0.575 mAP@50** (see `tests_balanced.md`). The gap is significant.

---

## Why Resuming Won't Help

The training history shows the model has plateaued, not been cut short:

| Epoch | val_loss | Δ |
|-------|----------|---|
| 86 | 2.8322 | — |
| 90 | 2.8232 | -0.0090 |
| 95 | 2.8147 | -0.0085 |
| 100 | 2.8128 | -0.0019 |

Only 0.0194 drop over the last 15 epochs. The run used `OneCycleLR` (default when `--cos_lr` is off), which decayed LR to near-zero well before epoch 100 — the model ran its final epochs at effectively zero LR. More epochs on the same config will not meaningfully improve results. Switching to `CosineAnnealingLR` (`--cos_lr`) keeps a meaningful LR floor throughout.

---

## Suggestions

### 1. Threshold Tuning (no retraining, quick)

Precision (0.3467) is significantly lower than recall (0.4920) — the model over-predicts. The default conf=0.25 may not be optimal. Evaluate at several thresholds to find the F1 sweet spot:

```bash
for conf in 0.30 0.35 0.40 0.45; do
  python scripts/evaluate_custom.py \
    --weights models/weights/edna_1.2m/best.pt \
    --data data/balanced_dataset \
    --conf $conf
done
```

Expected outcome: higher conf threshold will trade some recall for precision, likely improving F1 without any retraining.

---

### 2. Retrain with Cosine LR + EMA + warmup + lambda_obj

edna_1.2m used `OneCycleLR` (the default when `--cos_lr` is off), which decayed LR to
near-zero well before epoch 100 — the model ran its final epochs at effectively zero LR.
Switching to `CosineAnnealingLR` (`--cos_lr`) keeps a meaningful LR floor throughout —
this is the real fix for the plateau.

**New flags:**
- **`--cos_lr`**: CosineAnnealingLR with `eta_min = lr * 0.01` — avoids the near-zero stall
- **`--ema`**: ineffective at 20 epochs but fully valid at ~326K steps (100 epochs × 3,263 batches)
- **`--warmup_epochs 3`**: stabilises early training at full LR — zero cost
- **`--lambda_obj 1.5`**: confusion matrix shows clothing absorbed into background (missed
  detections, not misclassification). Raising objectness weight pushes the model to fire
  more aggressively on potential objects

**Caveat on lambda_obj:** 1.5 may increase false positives since precision is already the
weak metric (0.3467). Consider trying `--lambda_obj 1.25` first, or pairing 1.5 with a
lower focal gamma (see "Other Code-Level Improvements" below).

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset \
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
  --output models/weights/fashionnet_s_coslr_ema
```

---

### 3. Add Images to Weak Classes (~35h + data collection)

Since the problem is bg/fg confusion (not inter-class), adding more examples of the weakest
classes gives the model more signal to learn to detect those items against the background.

| Class | Current AP |
|-------|-----------|
| short_sleeve_top | 0.1284 |
| long_sleeve_top | 0.1448 |

**Caveat:** the dataset is already balanced (~4-5K images per class). Adding images only
for weak classes creates imbalance. Keep the gap reasonable — ~1-2K extra images per
weak class should help without significantly hurting the stronger classes.

---

### 4. Class Merge — ruled out

~~short_sleeve_top + long_sleeve_top → top~~

Previously considered but **ruled out after confusion matrix analysis**. Class merging
only helps when the model confuses one class for another (off-diagonal confusion matrix).
The confusion matrix shows clothing being absorbed by the background (FN column), not
misclassified between classes. Merging would not fix missed detections.

---

### 5. Larger Model Scale (~50h+)

edna_1.2m uses model_scale=m (~34M params). The FashionNet family also supports scale=l
(~63M params). Given YOLOv8M (25.8M params) outperforms edna_1.2m by ~0.31 mAP despite
fewer parameters, model capacity alone is not the bottleneck — but scale=l is worth
testing before concluding on architecture limits.

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

## Code Changes Required to Reach 0.60+ mAP@50

Flag tuning alone (cos_lr, EMA, warmup, lambda_obj) will likely gain **+0.02–0.05 mAP**,
plateauing around **~0.30–0.32 mAP@50** even with perfect hyperparameters. Reaching 0.60+
requires addressing two architectural/methodology gaps. These code changes are likely
**worth more than all the flag tweaks combined**.

### C1. IoU-aware Objectness Targets (loss.py — ~5 lines, highest impact)

Currently `build_targets` sets `obj_mask = 1.0` for all positive cells regardless of
localization quality. The model has no incentive to output lower confidence for poorly
localized predictions — a direct cause of the structural precision problem.

**Fix:** use CIoU between the current prediction and GT box as a soft objectness target
(`obj_mask = ciou.detach()`) instead of binary 1.0. This is how YOLOv5/v8 train
objectness — confidence should correlate with localization quality.

### C2. Mosaic Augmentation (dataset.py — moderate effort, high impact)

Mosaic combines 4 training images into one tile: 4× batch diversity, varied object scales
and positions, implicit small-object training. Absent from the current pipeline and one
of the primary reasons YOLOv8 trains so effectively.

### Other Code-Level Improvements

| Issue | Location | Fix | Impact |
|-------|----------|-----|--------|
| `beta1=0.937` (non-standard AdamW) | train_custom.py:269 | Change to `0.9` | Moderate |
| `weight_decay=5e-4` (too low) | train_custom.py:268 | Expose as flag, default `0.01` | Moderate |
| Label smoothing missing | loss.py build_targets | cls target `0.95` instead of `1.0` | Small |
| Focal gamma=1.5 | loss.py | Try `1.0` to reduce suppression of easy negatives | Small |

---

## Priority Order

| Priority | Suggestion | Effort | Est. mAP gain |
|----------|-----------|--------|---------------|
| 1 | Threshold tuning | ~10 min | +0.005 F1, done |
| 2 | **C1: IoU-aware objectness targets** | ~2h code | Largest available gain |
| 3 | **C2: Mosaic augmentation** | ~1 day code | High |
| 4 | Retrain with cos_lr + EMA + warmup + lambda_obj | ~35h | +0.02–0.05 mAP (flags only) |
| 5 | Retrain with C1 + C2 + all flags | ~35h | Required to reach 0.60+ |
| 6 | Add images to weak classes | ~35h + data | Addresses missed detections |
| 7 | Scale=l retrain | ~50h+ | Architecture ceiling test |
| ~~8~~ | ~~Class merge~~ | — | Ruled out — wrong failure mode |

# FashionNet Pipeline Fixes and Experiment Plan

## Context

FashionNet (custom from-scratch detector) initially achieved **0.009 mAP@50** on the sample
dataset, compared to **0.767 mAP@50** for the fine-tuned YOLOv8L. While the architecture gap
and lack of pretrained weights explain part of this, several bugs in the training pipeline were
artificially suppressing performance.

This document describes each identified issue, the implemented fix, and the 12-experiment
ablation plan used to validate the improvements.

---

## Identified Issues and Fixes

### 1. Box Loss Weight Too Low (Critical)

**Problem:** `lambda_box=0.05` in `loss.py` means the model receives almost no gradient signal
for bounding box regression. YOLOv5/v8 uses `box=7.5` by default. The model learned where
objects are (objectness) but never learned to draw accurate boxes around them, which directly
tanks mAP.

**Fix:** Default `lambda_box` changed to `5.0`.

---

### 2. Single-Cell Target Assignment

**Problem:** Each ground-truth box was assigned to exactly 1 grid cell (the cell containing its
center). This means ~95%+ of cells are negative, giving the model very few positive examples
per image. Modern detectors (YOLOv5/v8) assign each GT to 3–4 nearby cells.

**Fix:** Added `--multi_cell` flag. When enabled, each GT is also assigned to adjacent cells
when the center is near a cell boundary (within 0.5 grid units), increasing positive training
signal by ~2–3x.

---

### 3. NUM_CLASSES Mismatch

**Problem:** `model.py` hardcoded `NUM_CLASSES=13` (sample dataset), but the balanced dataset
has 11 classes. The model was wasting capacity on 2 unused output channels and receiving
confusing gradients.

**Fix:** `train_custom.py` now reads `nc` from `dataset.yaml` automatically. Can be overridden
with `--num_classes`.

---

### 4. Weak Augmentations

**Problem:** Augmentations were limited to horizontal flip + color jitter. For a from-scratch
model (no pretrained features), strong augmentation is critical to prevent overfitting.

**Fix:** Added `--augment` flag with three levels:
- `light` (previous default): horizontal flip + color jitter + rare grayscale
- `medium`: adds random scale (0.7–1.3×), rotation (±10°), translate (±10%)
- `heavy`: aggressive scale (0.5–1.5×), rotation (±15°), Gaussian noise, coarse dropout

---

### 5. No Dropout

**Problem:** No regularisation beyond weight decay. For a from-scratch model with limited data,
dropout in the detection head can help prevent overfitting.

**Fix:** Added optional `--dropout` flag (applied as Dropout2d before the prediction conv in
each detection head).

---

## New CLI Flags (train_custom.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--lambda_box` | 5.0 | Box loss weight (was 0.05) |
| `--lambda_obj` | 1.0 | Objectness loss weight |
| `--lambda_cls` | 0.5 | Classification loss weight |
| `--augment` | light | Augmentation level: light, medium, heavy |
| `--multi_cell` | off | Enable multi-cell GT assignment |
| `--num_classes` | auto | Read from dataset.yaml, or override |
| `--dropout` | 0.0 | Dropout rate in detection heads |
| `--cos_lr` | off | Use CosineAnnealingLR instead of OneCycleLR |
| `--grayscale` | off | Convert images to grayscale (3ch repeated) |
| `--warmup_epochs` | 0 | Linear LR warmup before main schedule (0 = disabled) |
| `--optimizer` | adamw | Optimizer: `adamw` or `sgd` (momentum=0.937, nesterov) |
| `--ema` | off | Exponential Moving Average of weights (used for val/inference) |

---

## Ablation Experiment Plan

All experiments use the balanced dataset with `--max_samples 2000 --epochs 20` for fast
iteration (~10 minutes each). Each adds one change over the previous to isolate individual impact.

### Experiment 1 — Baseline (fixed num_classes only)

Establishes baseline with correct num_classes but **old** lambda_box=0.05 (broken loss).

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --lambda_box 0.05 \
  --output models/weights/exp1_baseline
```

---

### Experiment 2 — Loss Weights Fix

Tests the impact of correcting box loss weight (0.05 → 5.0). Expected to be the single biggest improvement.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda \
  --output models/weights/exp2_loss_fix
```

---

### Experiment 3 — Loss Fix + Multi-Cell Assignment

Tests whether more positive training signal improves convergence.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell \
  --output models/weights/exp3_multicell
```

---

### Experiment 4 — Loss Fix + Multi-Cell + Medium Augmentation

Tests scale/rotation augmentation impact.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --output models/weights/exp4_aug_medium
```

---

### Experiment 5 — Loss Fix + Multi-Cell + Heavy Augmentation

Tests if heavy augmentation helps or hurts with limited samples.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment heavy \
  --output models/weights/exp5_aug_heavy
```

---

### Experiment 6 — Best Config + Lower LR + Cosine Schedule

Tests if slower LR with cosine annealing improves convergence stability.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --lr 0.0005 --cos_lr \
  --output models/weights/exp6_cos_lr
```

---

### Experiment 7 — Best Config + Dropout

Tests regularisation impact on a from-scratch model.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --dropout 0.1 \
  --output models/weights/exp7_dropout
```

---

### Experiment 8 — Grayscale Only

Tests if removing colour forces the model to learn shape/silhouette features, improving
discrimination between same-colour clothing types.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --grayscale \
  --output models/weights/exp8_grayscale
```

---

### Experiment 9 — Grayscale + Best Config

Combines grayscale with the best configuration from Experiments 3–7.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium --grayscale \
  --output models/weights/exp9_grayscale_best
```

---

### Experiment 10 — Best Config + Warmup

Tests if a 3-epoch linear warmup stabilises early training. Requires `--cos_lr` since
OneCycleLR has its own built-in warmup.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --cos_lr --warmup_epochs 3 \
  --output models/weights/exp10_warmup
```

---

### Experiment 11 — SGD + Momentum

Tests if SGD with momentum (standard for YOLO detectors) converges better than AdamW.
SGD is generally slower per epoch but can reach a better final mAP.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --optimizer sgd --lr 0.01 \
  --output models/weights/exp11_sgd
```

---

### Experiment 12 — Best Config + EMA

Tests if Exponential Moving Average of model weights improves validation mAP. EMA smooths
noisy weight updates and is used by default in YOLOv5/v8.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --ema \
  --output models/weights/exp12_ema
```

---

## How to Compare Results

After running all experiments, compare `val_loss` from each `history.json`:

```bash
for exp in exp1_baseline exp2_loss_fix exp3_multicell exp4_aug_medium exp5_aug_heavy exp6_cos_lr exp7_dropout exp8_grayscale exp9_grayscale_best exp10_warmup exp11_sgd exp12_ema; do
  echo "=== $exp ==="
  python -c "
import json
h = json.load(open('models/weights/$exp/history.json'))
best = min(h, key=lambda x: x['val_loss'])
print(f\"  Best epoch: {best['epoch']}, val_loss: {best['val_loss']:.4f}, box: {best['box']:.4f}, obj: {best['obj']:.4f}, cls: {best['cls']:.4f}\")
"
done
```

Once the best config is identified, run full training on the complete balanced dataset:

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --epochs 100 \
  --batch 32 --device cuda \
  <best flags from experiments> \
  --output models/weights/fashionnet_v2
```

Then evaluate against YOLOv8 with `scripts/compare_models.py`.

---

## Metrics

### Primary: mAP@50
A prediction is correct only if the predicted class is right AND the box overlaps the ground
truth by at least 50% IoU. This is the standard single-number summary for object detection.

### Per-class AP
The most important diagnostic metric. Identifies which clothing types are struggling, enabling
targeted investigation of inter-class confusion.

### Precision / Recall
- **Precision**: of all predicted boxes, what fraction were correct? High precision = few false positives.
- **Recall**: of all ground-truth objects, what fraction did the model find? High recall = few missed detections.
- There is always a precision/recall trade-off at a fixed threshold.

### val_loss (proxy only)
Useful for quick iteration. Lower is generally better, but val_loss can disagree with mAP when
confidence thresholds are poorly calibrated. Use val_loss to rank runs quickly; confirm with
actual mAP evaluation.

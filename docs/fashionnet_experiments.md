# FashionNet Experiment Plan

## Context

FashionNet (custom from-scratch detector) achieved **0.009 mAP@50** on the sample dataset (Test 6 in `tests_initial.md`), compared to **0.767 mAP@50** for the fine-tuned YOLOv8L. While the architecture gap and lack of pretrained weights explain part of this, several issues in the training pipeline are artificially suppressing performance.

This document describes the identified issues, the implemented fixes, and the experiment configurations to validate each improvement.

---

## Identified Issues

### 1. Box Loss Weight Too Low (Critical)

`lambda_box=0.05` in `loss.py` means the model receives almost no gradient signal for bounding box regression. For reference, YOLOv5/v8 uses `box=7.5` by default. The model learns where objects are (objectness) but never learns to draw accurate boxes around them, which directly tanks mAP.

**Fix:** Default `lambda_box` changed to `5.0`.

### 2. Single-Cell Target Assignment

Each ground-truth box is assigned to exactly 1 grid cell (the cell its center falls in). This means ~95%+ of cells are negative, giving the model very few positive examples per image. Modern detectors (YOLOv5/v8) assign each GT to 3-4 nearby cells.

**Fix:** Added `--multi_cell` flag. When enabled, each GT is also assigned to adjacent cells when the center is near a cell boundary (within 0.5 grid units), giving 1-4 positive cells per GT box. This increases positive training signal by ~2-3x.

### 3. NUM_CLASSES Mismatch

`model.py` hardcodes `NUM_CLASSES=13` (sample dataset) but the balanced dataset has 11 classes. The model wastes capacity on 2 unused output channels and may receive confusing gradients.

**Fix:** `train_custom.py` now reads `nc` from `dataset.yaml` automatically. Can be overridden with `--num_classes`.

### 4. Weak Augmentations

Current augmentations are limited to horizontal flip + color jitter. No scale variation, rotation, or spatial noise. For a model training from scratch (no pretrained features), strong augmentation is critical to prevent overfitting.

**Fix:** Added `--augment` flag with three levels:
- `light` (current): horizontal flip + color jitter + rare grayscale
- `medium`: adds random scale (0.7-1.3x), rotation (±10°), translate (±10%)
- `heavy`: aggressive scale (0.5-1.5x), rotation (±15°), Gaussian noise, coarse dropout

### 5. No Dropout

The model has no regularisation beyond weight decay. For a from-scratch model with limited data, dropout in the detection head helps prevent overfitting.

**Fix:** Added optional `--dropout` flag (applied as Dropout2d before the prediction conv in each detection head).

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

---

## Experiment Configurations

All experiments use the balanced dataset with `--max_samples 2000 --epochs 20` for fast iteration. Each adds one change over the previous to isolate individual impact.

### Experiment 1 — Baseline (fixed num_classes only)

Purpose: Establish baseline with correct num_classes but **old** lambda_box=0.05.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 16 --device mps --lambda_box 0.05 \
  --output models/weights/exp1_baseline
```

### Experiment 2 — Loss Weights Fix

Purpose: Test impact of corrected box loss weight (0.05 → 5.0). **Expected biggest single improvement.**

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 16 --device mps \
  --output models/weights/exp2_loss_fix
```

### Experiment 3 — Loss Fix + Multi-Cell Assignment

Purpose: Test if more positive training signal improves convergence.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 16 --device mps --multi_cell \
  --output models/weights/exp3_multicell
```

### Experiment 4 — Loss Fix + Multi-Cell + Medium Augmentation

Purpose: Test scale/rotation augmentation impact.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 16 --device mps --multi_cell --augment medium \
  --output models/weights/exp4_aug_medium
```

### Experiment 5 — Loss Fix + Multi-Cell + Heavy Augmentation

Purpose: Test if heavy augmentation helps or hurts with limited samples.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 16 --device mps --multi_cell --augment heavy \
  --output models/weights/exp5_aug_heavy
```

### Experiment 6 — Best Config + Lower LR + Cosine Schedule

Purpose: Test if slower learning rate with cosine annealing improves convergence stability.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 16 --device mps --multi_cell --augment medium \
  --lr 0.0005 --cos_lr \
  --output models/weights/exp6_cos_lr
```

### Experiment 7 — Best Config + Dropout

Purpose: Test regularisation impact on a from-scratch model.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 16 --device mps --multi_cell --augment medium \
  --dropout 0.1 \
  --output models/weights/exp7_dropout
```

---

## How to Compare Results

After running all experiments, compare `val_loss` from each `history.json`:

```bash
for exp in exp1_baseline exp2_loss_fix exp3_multicell exp4_aug_medium exp5_aug_heavy exp6_cos_lr exp7_dropout; do
  echo "=== $exp ==="
  python -c "
import json
h = json.load(open('models/weights/$exp/history.json'))
best = min(h, key=lambda x: x['val_loss'])
print(f\"  Best epoch: {best['epoch']}, val_loss: {best['val_loss']:.4f}, box: {best['box']:.4f}, obj: {best['obj']:.4f}, cls: {best['cls']:.4f}\")
"
done
```

Once the best config is identified, run a full training on the complete balanced dataset:

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset --epochs 100 \
  --batch 16 --device mps \
  <best flags from experiments> \
  --output models/weights/fashionnet_v2
```

Then evaluate against YOLOv8 with `scripts/compare_models.py`.

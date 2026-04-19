# FashionNet Pipeline: Issues Found and Fixes Applied

## Context

FashionNet is a custom single-shot detector built from scratch in pure PyTorch. It initially
achieved **0.009 mAP@50** on the sample dataset, compared to **0.767 mAP@50** for fine-tuned
YOLOv8L. While the architecture gap and lack of pretrained weights explain part of this
difference, several bugs in the training pipeline were artificially suppressing performance.

This document describes each identified issue, the fix applied, and the ablation study
used to validate the improvements.

---

## Identified Issues and Fixes

### 1. Box Loss Weight Too Low (Critical)

**Problem:** `lambda_box=0.05` in `loss.py` meant the model received almost no gradient signal
for bounding box regression. YOLOv5/v8 uses `box=7.5` by default. The model learned where
objects are (objectness) but never learned to draw accurate boxes, which directly tanks mAP.

**Fix:** Default `lambda_box` changed to `5.0`.

### 2. Single-Cell Target Assignment

**Problem:** Each ground-truth box was assigned to exactly 1 grid cell (the cell containing its
center). This means ~95%+ of cells are negative, giving the model very few positive examples
per image. Modern detectors (YOLOv5/v8) assign each GT to 3--4 nearby cells.

**Fix:** Added `--multi_cell` flag. When enabled, each GT is also assigned to adjacent cells
when the center is near a cell boundary (within 0.5 grid units), increasing positive training
signal by ~2--3x.

### 3. NUM_CLASSES Mismatch

**Problem:** `model.py` hardcoded `NUM_CLASSES=13` (sample dataset), but the balanced dataset
has 11 classes. The model wasted capacity on 2 unused output channels and received confusing gradients.

**Fix:** `train_custom.py` now reads `nc` from `dataset.yaml` automatically.

### 4. Weak Augmentations

**Problem:** Augmentations were limited to horizontal flip + color jitter. For a from-scratch
model with no pretrained features, strong augmentation is critical to prevent overfitting.

**Fix:** Added `--augment` flag with three levels:
- `light` (original default): horizontal flip + color jitter + rare grayscale
- `medium`: adds random scale (0.7--1.3x), rotation (+/-10 deg), translate (+/-10%)
- `heavy`: aggressive scale (0.5--1.5x), rotation (+/-15 deg), Gaussian noise, coarse dropout

### 5. No Dropout

**Problem:** No regularisation beyond weight decay.

**Fix:** Added optional `--dropout` flag (applied as Dropout2d before the prediction conv
in each detection head).

---

## Ablation Study

All experiments used the balanced dataset with `--max_samples 2000 --epochs 20` for fast
iteration (~10 minutes each). Each experiment adds one change over the previous to isolate
individual impact.

### Results

| Exp | Config | Best Epoch | val_loss | box_loss |
|-----|--------|-----------|----------|----------|
| 1 | lambda_box=0.05 (broken) | 20 | 0.52* | 2.303 |
| 2 | lambda_box=5.0 (fixed) | 20 | 10.07 | 1.917 |
| 3 | + multi_cell | 20 | 10.29 | 1.962 |
| **4** | **+ augment medium** | **18** | **8.88** | **1.728** |
| 5 | + augment heavy | 20 | 12.35 | 2.420 |
| 6 | lr=0.0005 + cos_lr | 20 | 10.07 | 1.909 |
| 7 | + dropout=0.1 | 19 | 10.38 | 1.974 |
| 8 | grayscale only | 20 | 12.62 | 2.435 |
| 9 | grayscale + medium aug | 18 | 12.82 | 2.469 |
| 10 | + warmup_epochs=3 + cos_lr | 18 | 10.38 | 1.983 |
| 11 | optimizer=sgd, lr=0.01 | 20 | 12.33 | 2.356 |
| 12 | + ema | 20 | 16.47 | 1.975 |

*Exp 1 val_loss uses lambda_box=0.05, so box loss is weighted ~100x less -- not directly comparable.

### Analysis

**Winner: Experiment 4** -- multi_cell + medium augmentation (val_loss 8.88). This combination
achieved the lowest raw box loss (1.728), meaning the model learned to regress boxes more
accurately than any other configuration.

**Grayscale hurts.** Removing colour (exp8: 12.62, exp9: 12.82) consistently degraded results.
Colour is discriminative for clothing detection.

**Heavy augmentation hurts.** Exp5 (12.35) was worse than exp4 (8.88). With only 2,000 samples
at 20 epochs, aggressive augmentation is too destructive for the model to learn effectively.

**Smaller changes had no clear benefit.** Dropout (exp7), warmup (exp10), SGD (exp11), and
EMA (exp12) showed no improvement at this training scale. EMA in particular requires thousands
of batches to warm up and was ineffective at 20 epochs.

The winning configuration (exp4: multi_cell + medium augmentation) was used for all subsequent
full-scale training runs.

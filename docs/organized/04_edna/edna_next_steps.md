# Dataset Analysis

Source: DeepFashion2 dataset, processed from raw annotations into YOLO format.

---

## Raw Dataset Summary

**Total items: 364,676** across 13 clothing categories.

| Category | Count | % of Total | Median Box Area | Median Aspect Ratio | % Heavy Occlusion (occ=3) |
|---|---:|---:|---:|---:|---:|
| short_sleeve_top | 84,201 | 23.1% | 0.1941 | 0.96 | 4.3% |
| trousers | 64,973 | 17.8% | 0.1608 | 0.61 | 7.6% |
| long_sleeve_top | 42,030 | 11.5% | 0.2029 | 0.90 | 1.9% |
| shorts | 40,783 | 11.2% | 0.1142 | 1.10 | 12.4% |
| skirt | 37,357 | 10.2% | 0.1571 | 0.97 | 6.6% |
| vest_dress | 21,301 | 5.8% | 0.3139 | 0.60 | 12.6% |
| short_sleeve_dress | 20,338 | 5.6% | 0.3338 | 0.66 | 3.3% |
| vest | 18,208 | 5.0% | 0.1523 | 0.83 | 8.2% |
| long_sleeve_outwear | 15,468 | 4.2% | 0.3546 | 0.75 | 0.7% |
| long_sleeve_dress | 9,384 | 2.6% | 0.3474 | 0.68 | 3.6% |
| sling_dress | 7,641 | 2.1% | 0.2892 | 0.57 | 8.1% |
| sling | 2,307 | 0.6% | 0.2001 | 0.82 | 12.7% |
| short_sleeve_outwear | 685 | 0.2% | 0.3097 | 0.75 | 1.2% |

---

## Balanced Dataset

**Total items: 84,051** across **11 classes**.

### Excluded Classes

Two classes were excluded due to insufficient sample counts for stratified balancing:

- `sling` â€” only 2,307 items (not enough for representative train/val/test)
- `short_sleeve_outwear` â€” only 685 items

### Sampling Strategy

Each of the 11 remaining classes was sampled down to **7,641 items**, stratified by occlusion
level to preserve the original occlusion distribution within each class.

### Occlusion Distribution per Class

| Category | occ1 (visible) | occ2 (partial) | occ3 (heavy) |
|---|---:|---:|---:|
| short_sleeve_top | 4,753 | 2,562 | 326 |
| long_sleeve_top | 5,013 | 2,479 | 149 |
| long_sleeve_outwear | 4,923 | 2,662 | 56 |
| vest | 4,134 | 2,877 | 630 |
| shorts | 2,293 | 4,397 | 951 |
| trousers | 1,546 | 5,513 | 582 |
| skirt | 2,582 | 4,558 | 501 |
| short_sleeve_dress | 4,116 | 3,275 | 250 |
| long_sleeve_dress | 4,582 | 2,782 | 277 |
| vest_dress | 3,418 | 3,260 | 963 |
| sling_dress | 4,541 | 2,484 | 616 |

---

## Train / Val / Test Split

Split ratio: **70% train / 15% val / 15% test** (split by image, not annotation).

| Category | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| short_sleeve_top | 5,349 | 1,143 | 1,149 | 7,641 |
| long_sleeve_top | 5,340 | 1,152 | 1,149 | 7,641 |
| long_sleeve_outwear | 5,343 | 1,157 | 1,141 | 7,641 |
| vest | 5,349 | 1,149 | 1,143 | 7,641 |
| shorts | 5,333 | 1,171 | 1,137 | 7,641 |
| trousers | 5,350 | 1,158 | 1,133 | 7,641 |
| skirt | 5,359 | 1,148 | 1,134 | 7,641 |
| short_sleeve_dress | 5,367 | 1,121 | 1,153 | 7,641 |
| long_sleeve_dress | 5,338 | 1,143 | 1,160 | 7,641 |
| vest_dress | 5,343 | 1,158 | 1,140 | 7,641 |
| sling_dress | 5,356 | 1,132 | 1,153 | 7,641 |
| **TOTAL** | **58,827** | **12,632** | **12,592** | **84,051** |

---

## Notes on Dataset Design Decisions

- Stratified occlusion sampling ensures the model trains on realistic distributions of
  visibility â€” not just fully visible garments.
- The 70/15/15 split is by image (not instance), so a single multi-garment image is
  entirely in one split, preventing label leakage.
- sling_dress has the fewest raw images (7,641), which is why it is the cap value for
  all other classes.

# YOLOv8 Experiment Results

All experiments in this document were run with `scripts/training/train.py` (Ultralytics API).
Tests 1â€“5 use the **sample dataset** (10k images, 13 classes, 970 val images).
The balanced dataset experiments follow at the end.

---

## Sample Dataset Experiments (Tests 1â€“5)

### Test 1 â€” YOLOv8L | batch=16 | ~1.311 hours

| Category | Images | Instances | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|--------|-----------|-----------|--------|--------|-----------|
| **all** | **970** | **1614** | **0.731** | **0.726** | **0.784** | **0.655** |
| short_sleeve_top | 228 | 231 | 0.823 | 0.806 | 0.873 | 0.747 |
| long_sleeve_top | 138 | 138 | 0.728 | 0.739 | 0.762 | 0.650 |
| short_sleeve_outwear | 78 | 79 | 0.677 | 0.734 | 0.789 | 0.667 |
| long_sleeve_outwear | 105 | 105 | 0.699 | 0.667 | 0.759 | 0.634 |
| vest | 93 | 94 | 0.708 | 0.734 | 0.823 | 0.662 |
| sling | 75 | 75 | 0.814 | 0.760 | 0.835 | 0.684 |
| shorts | 155 | 156 | 0.836 | 0.785 | 0.870 | 0.677 |
| trousers | 247 | 249 | 0.920 | 0.832 | 0.934 | 0.724 |
| skirt | 170 | 170 | 0.766 | 0.692 | 0.797 | 0.669 |
| short_sleeve_dress | 75 | 77 | 0.529 | 0.662 | 0.616 | 0.531 |
| long_sleeve_dress | 74 | 74 | 0.676 | 0.620 | 0.646 | 0.567 |
| vest_dress | 90 | 90 | 0.622 | 0.644 | 0.697 | 0.601 |
| sling_dress | 75 | 76 | 0.700 | 0.767 | 0.794 | 0.707 |

**Per-class mAP@50 (sorted):**

| Category | mAP@50 |
|----------|--------|
| trousers | 0.9149 |
| short_sleeve_top | 0.8598 |
| shorts | 0.8556 |
| sling | 0.8263 |
| vest | 0.8218 |
| sling_dress | 0.7906 |
| skirt | 0.7707 |
| short_sleeve_outwear | 0.7578 |
| long_sleeve_top | 0.7513 |
| long_sleeve_outwear | 0.7434 |
| vest_dress | 0.6786 |
| long_sleeve_dress | 0.6152 |
| short_sleeve_dress | 0.5886 |

**Summary:** mAP@50 0.7673 | mAP@50:95 0.6637 | Precision 0.7229 | Recall 0.7302

---

### Test 2 â€” YOLOv8L | batch=26 | ~1.316 hours

| Category | Images | Instances | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|--------|-----------|-----------|--------|--------|-----------|
| **all** | **970** | **1614** | **0.723** | **0.712** | **0.777** | **0.652** |
| short_sleeve_top | 228 | 231 | 0.790 | 0.779 | 0.857 | 0.729 |
| long_sleeve_top | 138 | 138 | 0.674 | 0.718 | 0.760 | 0.657 |
| short_sleeve_outwear | 78 | 79 | 0.726 | 0.684 | 0.785 | 0.696 |
| long_sleeve_outwear | 105 | 105 | 0.715 | 0.691 | 0.770 | 0.637 |
| vest | 93 | 94 | 0.742 | 0.787 | 0.852 | 0.697 |
| sling | 75 | 75 | 0.780 | 0.800 | 0.865 | 0.699 |
| shorts | 155 | 156 | 0.810 | 0.821 | 0.864 | 0.694 |
| trousers | 247 | 249 | 0.919 | 0.839 | 0.941 | 0.729 |
| skirt | 170 | 170 | 0.761 | 0.706 | 0.793 | 0.651 |
| short_sleeve_dress | 75 | 77 | 0.545 | 0.571 | 0.540 | 0.455 |
| long_sleeve_dress | 74 | 74 | 0.586 | 0.568 | 0.640 | 0.566 |
| vest_dress | 90 | 90 | 0.711 | 0.629 | 0.710 | 0.645 |
| sling_dress | 75 | 76 | 0.645 | 0.668 | 0.723 | 0.623 |

**Speed:** 0.1ms preprocess, 4.0ms inference, 0.3ms postprocess per image

**Summary:** mAP@50 0.7599 | mAP@50:95 0.6593 | Precision 0.7236 | Recall 0.7117

---

### Test 3 â€” YOLO-World Zero-Shot | yolov8s-worldv2 | conf=0.15

No fine-tuning â€” open-vocabulary detection via CLIP text embeddings.

| Category | Images | Instances | Precision | Recall | mAP@50 |
|----------|--------|-----------|-----------|--------|--------|
| **all** | **970** | **1614** | **0.235** | **0.385** | **0.1457** |
| short_sleeve_top | 228 | 231 | 0.485 | 0.476 | 0.2873 |
| long_sleeve_top | 138 | 138 | 0.240 | 0.543 | 0.1803 |
| short_sleeve_outwear | 78 | 79 | 0.100 | 0.025 | 0.0035 |
| long_sleeve_outwear | 105 | 105 | 0.211 | 0.076 | 0.0224 |
| vest | 93 | 94 | 0.089 | 0.043 | 0.0162 |
| sling | 75 | 75 | 0.029 | 0.040 | 0.0027 |
| shorts | 155 | 156 | 0.529 | 0.346 | 0.3115 |
| trousers | 247 | 249 | 0.718 | 0.594 | 0.5351 |
| skirt | 170 | 170 | 0.306 | 0.659 | 0.3320 |
| short_sleeve_dress | 75 | 77 | 0.154 | 0.078 | 0.0381 |
| long_sleeve_dress | 74 | 74 | 0.065 | 0.338 | 0.0400 |
| vest_dress | 90 | 90 | 0.205 | 0.089 | 0.0204 |
| sling_dress | 75 | 76 | 0.088 | 0.882 | 0.1048 |

**Why zero-shot performs poorly on fashion-specific classes:**
YOLO-World uses CLIP embeddings for open-vocabulary detection. CLIP was trained on
generic internet text. Common English terms like "trousers" and "shorts" have rich,
well-defined embeddings â€” those classes score best. Fashion-specific compound terms
like "sling", "vest_dress", and "long_sleeve_outwear" appear rarely or ambiguously in
CLIP's training corpus, resulting in near-zero mAP for those categories.

---

### Test 4 â€” YOLOv8L | batch=16 | ~1.227 hours (reproducibility check)

Full per-class results (same setup as Test 1, different run):

| Category | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|-----------|--------|--------|-----------|
| **all** | **0.723** | **0.730** | **0.767** | **0.664** |
| short_sleeve_top | 0.820 | 0.805 | 0.860 | 0.758 |
| long_sleeve_top | 0.721 | 0.739 | 0.751 | 0.658 |
| short_sleeve_outwear | 0.665 | 0.734 | 0.758 | 0.658 |
| long_sleeve_outwear | 0.688 | 0.672 | 0.743 | 0.650 |
| vest | 0.707 | 0.744 | 0.822 | 0.684 |
| sling | 0.816 | 0.769 | 0.826 | 0.711 |
| shorts | 0.839 | 0.788 | 0.856 | 0.698 |
| trousers | 0.919 | 0.835 | 0.915 | 0.742 |
| skirt | 0.753 | 0.700 | 0.771 | 0.679 |
| short_sleeve_dress | 0.521 | 0.662 | 0.589 | 0.521 |
| long_sleeve_dress | 0.663 | 0.622 | 0.615 | 0.553 |
| vest_dress | 0.600 | 0.644 | 0.679 | 0.603 |
| sling_dress | 0.686 | 0.776 | 0.791 | 0.715 |

**Speed:** 0.2ms preprocess, 6.6ms inference, 0.2ms postprocess per image

**Summary:** mAP@50 0.7673 | mAP@50:95 0.6637 | Precision 0.7229 | Recall 0.7302

---

### Test 5 â€” YOLOv8L | batch=16 | No Pretrained Weights | ~1.203 hours

Trained from scratch using `--no-pretrained` (random weight initialization, no COCO pretraining).

| Category | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|-----------|--------|--------|-----------|
| **all** | **0.641** | **0.668** | **0.697** | **0.587** |
| short_sleeve_top | 0.741 | 0.758 | 0.804 | 0.680 |
| long_sleeve_top | 0.633 | 0.681 | 0.689 | 0.567 |
| short_sleeve_outwear | 0.629 | 0.684 | 0.717 | 0.632 |
| long_sleeve_outwear | 0.616 | 0.638 | 0.719 | 0.618 |
| vest | 0.575 | 0.747 | 0.737 | 0.611 |
| sling | 0.796 | 0.667 | 0.779 | 0.644 |
| shorts | 0.752 | 0.788 | 0.849 | 0.683 |
| trousers | 0.907 | 0.803 | 0.901 | 0.716 |
| skirt | 0.656 | 0.706 | 0.713 | 0.589 |
| short_sleeve_dress | 0.507 | 0.494 | 0.503 | 0.430 |
| long_sleeve_dress | 0.506 | 0.541 | 0.537 | 0.474 |
| vest_dress | 0.522 | 0.594 | 0.559 | 0.493 |
| sling_dress | 0.498 | 0.579 | 0.559 | 0.490 |

**Speed:** 0.2ms preprocess, 6.4ms inference, 0.2ms postprocess per image

**Summary:** mAP@50 0.6974 | mAP@50:95 0.5867 | Precision 0.6414 | Recall 0.6676

---

## Cross-Test Comparisons

### Test 1 vs Test 2: Batch Size Impact (YOLOv8L, sample dataset)

| Metric | Test 1 (batch=16) | Test 2 (batch=26) |
|--------|-------------------|-------------------|
| mAP@50 | 0.7673 | 0.7599 |
| mAP@50:95 | 0.6637 | 0.6593 |
| Precision | 0.7229 | 0.7236 |
| Recall | 0.7302 | 0.7117 |
| Training time | ~1.311h | ~1.316h |

**Conclusion:** batch=16 marginally outperforms batch=26 with nearly identical training time.
The larger batch provided no benefit here. **batch=16 is the recommended configuration.**

---

### Test 1 vs Test 4: Reproducibility

| Metric | Test 1 | Test 4 |
|--------|--------|--------|
| mAP@50 | 0.7673 | 0.7673 |
| mAP@50:95 | 0.6637 | 0.6637 |
| Precision | 0.7229 | 0.7229 |
| Recall | 0.7302 | 0.7302 |

Results are identical. Confirms batch=16 is a **stable, reproducible configuration** for
YOLOv8L on this dataset. Test 4 completed slightly faster (1.227h vs 1.311h).

---

### Test 1 (pretrained) vs Test 5 (from scratch): Transfer Learning Value

| Metric | Test 1 (pretrained) | Test 5 (scratch) | Difference |
|--------|---------------------|------------------|------------|
| mAP@50 | 0.7673 | 0.6974 | -0.0699 |
| mAP@50:95 | 0.6637 | 0.5867 | -0.0770 |
| Precision | 0.7229 | 0.6414 | -0.0815 |
| Recall | 0.7302 | 0.6676 | -0.0626 |

COCO pretraining provides a **~7% mAP@50 advantage** at essentially no additional cost.
The gap is most pronounced on dress categories (short_sleeve_dress: 0.589 vs 0.503),
where limited training data makes feature transfer most valuable. Training time is nearly
identical (~1.2h either way).

---

### Test 1 (fine-tuned) vs Test 3 (zero-shot)

| Metric | YOLOv8L Fine-tuned | YOLO-World Zero-Shot |
|--------|-------------------|----------------------|
| mAP@50 | 0.7673 | 0.1457 |
| Precision | 0.7229 | 0.2353 |
| Recall | 0.7302 | 0.3854 |
| Best class | trousers (0.9149) | trousers (0.5351) |
| Worst class | short_sleeve_dress (0.5886) | sling (0.0027) |

Zero-shot mAP@50 (0.146) vs fine-tuned (0.767) â€” an expected ~5x gap. YOLO-World is
useful as a no-training baseline or for rapid prototyping, but fine-tuning is essential
for production-quality fashion detection.

---

### Test 6 â€” FashionNet (original) vs YOLOv8L

Side-by-side evaluation on the same 970-image validation set
(see `03_fashionnet_experiments/fashionnet_results.md` for FashionNet context).

| Metric | FashionNet (original) | YOLOv8L (fine-tuned) |
|--------|----------------------|----------------------|
| mAP@50 | 0.0091 | 0.7770 |
| Inference (ms/img) | 3.2 | 10.9 |
| FPS | 313.4 | 91.4 |
| Parameters (M) | 11.74 | 43.62 |
| Weights size (MB) | 141.2 | 87.6 |

**Per-class breakdown:**

| Category | FashionNet | YOLOv8L |
|----------|------------|---------|
| short_sleeve_top | 0.0250 | 0.8570 |
| long_sleeve_top | 0.0022 | 0.7562 |
| short_sleeve_outwear | 0.0198 | 0.7838 |
| long_sleeve_outwear | 0.0000 | 0.7679 |
| vest | 0.0099 | 0.8567 |
| sling | 0.0050 | 0.8649 |
| shorts | 0.0320 | 0.8647 |
| trousers | 0.0177 | 0.9397 |
| skirt | 0.0044 | 0.7932 |
| short_sleeve_dress | 0.0000 | 0.5445 |
| long_sleeve_dress | 0.0000 | 0.6385 |
| vest_dress | 0.0000 | 0.7096 |
| sling_dress | 0.0028 | 0.7245 |

The gap (0.768 mAP@50) is primarily explained by:
- **Broken loss weights** in FashionNet (lambda_box=0.05 â€” critical bug; see `fashionnet_pipeline_fixes.md`)
- **COCO pretraining** in YOLOv8L vs from-scratch FashionNet
- **Architectural maturity** â€” YOLOv8 benefits from years of design optimization

FashionNet is faster (3.2ms vs 10.9ms, ~3x) and uses fewer parameters,
but its detection quality before the pipeline fixes was near zero.

---

## Balanced Dataset â€” YOLOv8M Baseline

### Test â€” YOLOv8M | 50 epochs | batch=16 | ~7.429 hours

Evaluated on full balanced val split (11,186 images, 11 classes).

| Category | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|-----------|--------|--------|-----------|
| **all** | **0.558** | **0.678** | **0.575** | **0.521** |
| short_sleeve_top | 0.318 | 0.535 | 0.293 | 0.269 |
| long_sleeve_top | 0.432 | 0.632 | 0.408 | 0.367 |
| long_sleeve_outwear | 0.692 | 0.775 | 0.702 | 0.644 |
| vest | 0.632 | 0.711 | 0.641 | 0.559 |
| shorts | 0.523 | 0.687 | 0.547 | 0.464 |
| trousers | 0.394 | 0.595 | 0.400 | 0.351 |
| skirt | 0.446 | 0.657 | 0.438 | 0.392 |
| short_sleeve_dress | 0.623 | 0.710 | 0.693 | 0.646 |
| long_sleeve_dress | 0.642 | 0.731 | 0.723 | 0.684 |
| vest_dress | 0.619 | 0.693 | 0.664 | 0.609 |
| sling_dress | 0.818 | 0.729 | 0.816 | 0.741 |

**Speed:** 0.1ms preprocess, 4.0ms inference, 0.2ms postprocess per image

**Summary:** mAP@50 0.5750 | mAP@50:95 0.5207 | Precision 0.5581 | Recall 0.6778

**Why lower than sample dataset results:**
- Smaller model â€” YOLOv8M (25.8M params) vs YOLOv8L (43.6M params)
- Harder validation set â€” 11,186 images vs 970, more uniform class distribution
- short_sleeve_top (0.293) and trousers (0.400) dropped the most, likely due to
  increased visual confusion in the balanced, full-scale set

This YOLOv8M balanced result serves as the primary comparison target for all FashionNet models.

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
per image. Modern detectors (YOLOv5/v8) assign each GT to 3â€“4 nearby cells.

**Fix:** Added `--multi_cell` flag. When enabled, each GT is also assigned to adjacent cells
when the center is near a cell boundary (within 0.5 grid units), increasing positive training
signal by ~2â€“3x.

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
- `medium`: adds random scale (0.7â€“1.3Ã—), rotation (Â±10Â°), translate (Â±10%)
- `heavy`: aggressive scale (0.5â€“1.5Ã—), rotation (Â±15Â°), Gaussian noise, coarse dropout

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

### Experiment 1 â€” Baseline (fixed num_classes only)

Establishes baseline with correct num_classes but **old** lambda_box=0.05 (broken loss).

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --lambda_box 0.05 \
  --output models/weights/exp1_baseline
```

---

### Experiment 2 â€” Loss Weights Fix

Tests the impact of correcting box loss weight (0.05 â†’ 5.0). Expected to be the single biggest improvement.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda \
  --output models/weights/exp2_loss_fix
```

---

### Experiment 3 â€” Loss Fix + Multi-Cell Assignment

Tests whether more positive training signal improves convergence.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell \
  --output models/weights/exp3_multicell
```

---

### Experiment 4 â€” Loss Fix + Multi-Cell + Medium Augmentation

Tests scale/rotation augmentation impact.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --output models/weights/exp4_aug_medium
```

---

### Experiment 5 â€” Loss Fix + Multi-Cell + Heavy Augmentation

Tests if heavy augmentation helps or hurts with limited samples.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment heavy \
  --output models/weights/exp5_aug_heavy
```

---

### Experiment 6 â€” Best Config + Lower LR + Cosine Schedule

Tests if slower LR with cosine annealing improves convergence stability.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --lr 0.0005 --cos_lr \
  --output models/weights/exp6_cos_lr
```

---

### Experiment 7 â€” Best Config + Dropout

Tests regularisation impact on a from-scratch model.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --dropout 0.1 \
  --output models/weights/exp7_dropout
```

---

### Experiment 8 â€” Grayscale Only

Tests if removing colour forces the model to learn shape/silhouette features, improving
discrimination between same-colour clothing types.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --grayscale \
  --output models/weights/exp8_grayscale
```

---

### Experiment 9 â€” Grayscale + Best Config

Combines grayscale with the best configuration from Experiments 3â€“7.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium --grayscale \
  --output models/weights/exp9_grayscale_best
```

---

### Experiment 10 â€” Best Config + Warmup

Tests if a 3-epoch linear warmup stabilises early training. Requires `--cos_lr` since
OneCycleLR has its own built-in warmup.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --cos_lr --warmup_epochs 3 \
  --output models/weights/exp10_warmup
```

---

### Experiment 11 â€” SGD + Momentum

Tests if SGD with momentum (standard for YOLO detectors) converges better than AdamW.
SGD is generally slower per epoch but can reach a better final mAP.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --optimizer sgd --lr 0.01 \
  --output models/weights/exp11_sgd
```

---

### Experiment 12 â€” Best Config + EMA

Tests if Exponential Moving Average of model weights improves validation mAP. EMA smooths
noisy weight updates and is used by default in YOLOv5/v8.

```bash
python scripts/training/train_custom.py \
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
python scripts/training/train_custom.py \
  --data data/balanced_dataset --epochs 100 \
  --batch 32 --device cuda \
  <best flags from experiments> \
  --output models/weights/fashionnet_v2
```

Then evaluate against YOLOv8 with `scripts/evaluation/compare_models.py`.

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

*exp1 val_loss uses lambda_box=0.05, so the box component is weighted ~100Ã— less than all
other experiments â€” not directly comparable.

---

## Ablation Analysis

### Winner: exp4 â€” multi_cell + medium augmentation (val_loss 8.88)

The combination of multi-cell GT assignment and medium augmentation (scale Â±30%, rotation Â±10%,
translate Â±10%) gave the best result. It also achieved the lowest raw box loss (1.7275), meaning
the model is learning to regress boxes more accurately than any other configuration.

### Grayscale hurts

Removing colour (exp8: 12.62, exp9: 12.82) consistently degraded results. Colour information
is discriminative for this task. The hypothesis that same-colour confusion was a major problem
was not supported â€” colour helps more than it hurts overall.

### Heavy augmentation hurts

exp5 (heavy, 12.35) is worse than exp4 (medium, 8.88). With only 2,000 samples at 20 epochs,
the aggressive scale/rotation/noise in heavy mode is too destructive â€” the model cannot learn
fast enough to handle the increased variance.

### Smaller changes had no clear benefit

- **Dropout** (exp7: 10.38 vs exp4: 8.88) â€” no benefit, possibly slightly harmful
- **Warmup** (exp10: 10.38) â€” no measurable improvement at 20 epochs
- **SGD** (exp11: 12.33) â€” worse than AdamW at this epoch count; SGD typically needs more epochs
- **EMA** (exp12: 16.47) â€” misleading result; with decay=0.9999 the EMA model requires
  thousands of batches to warm up and is not effective at 20 epochs

---

## Full Training â€” fashionnet_balanced_v1

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

### Option A â€” Merge Similar Classes

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

Reduces from 11 â†’ 7 classes. Requires rebuilding labels and dataset.yaml.

### Option B â€” Add Images to Weakest Classes

Only useful if classes are visually distinct but underrepresented. Less likely to help for
short/long sleeve top confusion since the model already has 52K images to learn from.

---

## Next Step

Compare fashionnet_balanced_v1 against YOLOv8L on the balanced dataset:

```bash
python scripts/evaluation/compare_models.py \
  --custom_weights models/weights/fashionnet_balanced_v1/best.pt \
  --yolo_weights models/weights/yolov8l_fashion/weights/best.pt \
  --data data/balanced_dataset \
  --out docs/compare_fashionnet_v1_vs_yolov8l.json
```

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
| 86 | 2.8322 | â€” |
| 90 | 2.8232 | -0.0090 |
| 95 | 2.8147 | -0.0085 |
| 100 | 2.8128 | -0.0019 |

Only 0.0194 drop over the last 15 epochs. The run used `OneCycleLR` (default when `--cos_lr`
is off), which decayed LR to near-zero well before epoch 100 â€” the model ran its final
epochs at effectively zero LR. More epochs on the same config will not meaningfully improve
results. Switching to `CosineAnnealingLR` (`--cos_lr`) keeps a meaningful LR floor throughout.

---

## Suggestions

### 1. Threshold Tuning (no retraining, ~10 min)

Precision (0.3467) is significantly lower than recall (0.4920) â€” the model over-predicts.
Default conf=0.25 may not be optimal for F1. The threshold sweep (already done, see
`edna_results.md`) shows conf=0.30 peaks F1 at 0.4123 (+0.0055 over default) at the cost
of -0.022 mAP. The gain is small but available at zero training cost.

```bash
for conf in 0.30 0.35 0.40 0.45; do
  python scripts/evaluation/evaluate_custom.py \
    --weights models/weights/edna_1.2m/best.pt \
    --data data/balanced_dataset \
    --conf $conf
done
```

---

### 2. Retrain with Cosine LR + EMA + warmup + lambda_obj (~35h)

edna_1.2m used `OneCycleLR` (the default when `--cos_lr` is off), which decayed LR to
near-zero well before epoch 100 â€” the model ran its final epochs at effectively zero LR.
Switching to `CosineAnnealingLR` (`--cos_lr`) keeps a meaningful LR floor throughout â€”
this is the real fix for the plateau.

**New flags:**
- **`--cos_lr`**: CosineAnnealingLR with `eta_min = lr * 0.01` â€” avoids the near-zero LR
  stall that caused the plateau
- **`--ema`**: was ineffective at 20 epochs (~125 batches). At 100 epochs Ã— 3,263
  batches/epoch = ~326K steps, EMA is fully warmed up and should improve mAP at zero cost
- **`--warmup_epochs 3`**: stabilises early training when starting from CosineAnnealingLR
  at full LR â€” zero cost
- **`--lambda_obj 1.5`**: confusion matrix shows clothing absorbed into background (missed
  detections, not misclassification). Raising objectness weight pushes the model to fire
  more aggressively on potential objects

**Caveat on lambda_obj:** 1.5 may increase false positives since precision is already the
weak metric (0.3467). Consider trying `--lambda_obj 1.25` first, or pairing 1.5 with a
lower focal gamma (see "Other Code-Level Improvements" below).

```bash
python scripts/training/train_custom.py \
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
for weak classes creates imbalance. Keep the gap reasonable â€” adding ~1-2K images per
weak class should help without significantly hurting the stronger classes.

---

### 4. Class Merge â€” ~~short_sleeve_top + long_sleeve_top â†’ top~~ (not recommended)

Previously considered but **ruled out after confusion matrix analysis**. Class merging
only helps when the model confuses one class for another (off-diagonal confusion matrix).
The confusion matrix shows clothing being absorbed by the background (FN column), not
misclassified as each other. Merging would not fix missed detections.

---

### 5. Larger Model Scale â€” scale=l (~50h+)

edna_1.2m uses model_scale=m (~34M params). The FashionNet family also supports scale=l
(~63M params). Given YOLOv8M (25.8M) outperforms edna_1.2m by ~0.31 mAP despite fewer
parameters, model capacity alone is not the bottleneck â€” but testing scale=l is worth
doing before concluding on architecture limits.

```bash
python scripts/training/train_custom.py \
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

Flag tuning alone (cos_lr, EMA, warmup, lambda_obj) will likely gain **+0.02â€“0.05 mAP**,
plateauing around **~0.30â€“0.32 mAP@50** even with perfect hyperparameters. Reaching 0.60+
required addressing architectural/methodology gaps. These code changes are likely
**worth more than all the flag tweaks combined**.

All changes below are **implemented and included** in the proposed training command above.

### C1. IoU-aware Objectness Targets â€” done (loss.py)

`build_targets` previously set `obj_mask = 1.0` for all positive cells regardless of
localization quality. Now uses the CIoU between prediction and GT box as a soft objectness
target (`obj_mask = iou.detach().clamp(0)`), so confidence correlates with localization
quality. This is how YOLOv5/v8 train objectness.

**Impact:** highest single improvement available without architectural changes.
Directly addresses the precision/recall imbalance.

### C2. Mosaic Augmentation â€” done (dataset.py, `--mosaic` flag)

4-image mosaic combines training images into one tile: 4Ã— batch diversity, varied object
scales and positions, implicit small-object training. Uses letterbox resizing to preserve
aspect ratio (matching the non-mosaic pipeline). Enabled with `--mosaic` flag.

**Impact:** matches YOLOv8 training methodology â€” one of the primary reasons YOLOv8
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
| Focal loss gamma=1.5 | loss.py | Try gamma=1.0 to reduce suppression of easy negatives | Small â€” may help recall more than lambda_obj increase |

---

## Priority Order

| Priority | Suggestion | Status | Est. mAP gain |
|----------|-----------|--------|---------------|
| 1 | Threshold tuning | Done | +0.005 F1 |
| 2 | C1: IoU-aware objectness targets | Done | Largest available gain |
| 3 | C2: Mosaic augmentation | Done | High — matches YOLOv8 methodology |
| 4 | Retrain with all flags + code changes | Done (edna_1.3m) | Recall regression due to gr=0.5 |
| 5 | Background images (edna_1.4m) | Done | +0.0025 mAP, +0.13 precision, -0.077 recall vs 1.2m |
| 6 | edna_1.5m: fix focal_bce (.mean) + fewer bg images | **Next** | Recover recall while keeping precision gains |
| 7 | Add images to weak classes | Not started | Directly addresses missed detections |
| 8 | Scale=l retrain | Not started | Architecture ceiling test |
| ~~9~~ | ~~Class merge~~ | — | Ruled out — wrong failure mode |

# edna Training Results

edna is the FashionNet model family trained at larger scale or with different configurations
on the full balanced dataset. This document covers three training runs and their comparison.

All evaluations use `scripts/evaluation/evaluate_custom.py`, val split (11,186 images), conf=0.25, NMS IoU=0.45.

---

## edna_1m_balanced_100

### Setup

| Parameter | Value |
|-----------|-------|
| Config | default flags (no aug, no multi_cell) |
| Dataset | full balanced_dataset, 52,199 training images |
| Epochs | 100 |
| Batch | 32 (3,263 batches/epoch) |
| Device | CUDA, 16GB VRAM |
| Training time | 2,064m 44s (~34h 24m) |
| Best val_loss | 2.6953 (epoch 63) |
| Weights | `models/weights/edna_1m_balanced_100/best.pt` |

---

### fashionnet_balanced_v1 vs edna_1m_balanced_100

| Metric | fashionnet_balanced_v1 | edna_1m_balanced_100 |
|--------|----------------------|----------------------|
| mAP@50 | **0.1930** | 0.1869 |
| Precision | 0.3356 | **0.3479** |
| Recall | **0.3870** | 0.3723 |
| F1 | 0.3594 | **0.3597** |
| Best val_loss | 3.0591 | **2.6953** |
| Best epoch | 87 | 63 |
| Key flags | aug=medium, multi_cell | â€” |

**Per-class breakdown:**

| Category | fashionnet_balanced_v1 AP | edna_1m AP | fashionnet F1 | edna_1m F1 |
|----------|--------------------------|------------|---------------|------------|
| short_sleeve_top | 0.0860 | 0.0877 | 0.260 | 0.266 |
| long_sleeve_top | **0.0999** | 0.0709 | **0.259** | 0.234 |
| long_sleeve_outwear | 0.2823 | **0.2862** | 0.470 | **0.472** |
| vest | 0.2712 | **0.2727** | 0.398 | **0.431** |
| shorts | 0.2624 | **0.2781** | 0.429 | **0.439** |
| trousers | **0.2111** | 0.1950 | **0.405** | 0.386 |
| skirt | **0.1380** | 0.1179 | **0.298** | 0.275 |
| short_sleeve_dress | **0.1636** | 0.1620 | 0.338 | **0.340** |
| long_sleeve_dress | **0.1361** | 0.1304 | **0.307** | 0.300 |
| vest_dress | **0.2163** | 0.1792 | **0.399** | 0.383 |
| sling_dress | 0.2555 | **0.2756** | 0.370 | **0.416** |

**Analysis:** The two models are effectively tied on F1 (0.3594 vs 0.3597), despite
edna_1m_balanced_100 training 3Ã— longer and achieving a better val_loss (2.6953 vs 3.0591).
fashionnet_balanced_v1 edges out on mAP@50 and recall, while edna_1m wins on F1 and precision.

The aug=medium + multi_cell flags in fashionnet_balanced_v1 provide marginal but real benefit
for mAP. The significantly lower val_loss of edna_1m_balanced_100 does not translate into
better detection metrics â€” **val_loss and mAP@50 are not tightly coupled at this training scale.**

---

## edna_1.2m

### Setup

| Parameter | Value |
|-----------|-------|
| Config | aug=medium, multi_cell=true, model_scale=m (~34.07M params), optimizer=adamw |
| Dataset | full balanced_dataset, 52,199 training images |
| Epochs | 100 |
| Batch | 16 |
| Device | CUDA, 16GB VRAM |
| Training time | 2,093m 31s (~34h 53m) |
| Best val_loss | 2.8128 (epoch 100) |
| Weights | `models/weights/edna_1.2m/best.pt` |

### edna_1.2m Overall Metrics

| Metric | Value |
|--------|-------|
| mAP@50 | **0.2600** |
| Precision | 0.3467 |
| Recall | 0.4920 |
| F1 | **0.4068** |

### edna_1.2m Per-class Breakdown

| Category | AP | Precision | Recall | F1 |
|----------|----|-----------|--------|----|
| short_sleeve_top | 0.1284 | 0.215 | 0.440 | 0.289 |
| long_sleeve_top | 0.1448 | 0.329 | 0.338 | 0.334 |
| long_sleeve_outwear | **0.3734** | 0.524 | 0.552 | **0.537** |
| vest | 0.3516 | 0.368 | 0.594 | 0.455 |
| shorts | 0.3290 | 0.414 | 0.592 | 0.487 |
| trousers | 0.2463 | 0.318 | 0.604 | 0.417 |
| skirt | 0.2086 | 0.236 | 0.556 | 0.332 |
| short_sleeve_dress | 0.2705 | 0.404 | 0.445 | 0.424 |
| long_sleeve_dress | 0.2240 | 0.417 | 0.381 | 0.398 |
| vest_dress | 0.2706 | 0.406 | 0.478 | 0.439 |
| sling_dress | 0.3128 | 0.430 | 0.415 | 0.422 |

---

## 3-Way Comparison

| Experiment | mAP@50 | F1 | Best val_loss | Best epoch | Key flags |
|------------|--------|----|---------------|------------|-----------|
| fashionnet_balanced_v1 | 0.1930 | 0.3594 | 3.0591 | 87 | aug=medium, multi_cell |
| edna_1m_balanced_100 | 0.1869 | 0.3597 | 2.6953 | 63 | â€” |
| **edna_1.2m** | **0.2600** | **0.4068** | 2.8128 | 100 | aug=medium, multi_cell |

**edna_1.2m is a clear improvement over both previous versions:**
- +0.0670 mAP@50 over fashionnet_balanced_v1
- +0.0731 mAP@50 over edna_1m_balanced_100
- F1 improves by +0.0474 vs both

Recall jumps to 0.4920 (highest of the three), suggesting the medium-scale model with
aug=medium + multi_cell is better at finding objects. Precision (0.3467) remains the
lowest, meaning more false positives compared to the other two models.

Biggest class-level gains over edna_1m_balanced_100:
- long_sleeve_outwear: +0.0872 AP
- skirt: +0.0907 AP
- vest: +0.0789 AP
- shorts: +0.0509 AP
- short_sleeve_top: +0.0407 AP
- long_sleeve_top: +0.0739 AP

Scaling the model (scale=m vs default scale=s in edna_1m) combined with re-enabling
aug=medium and multi_cell accounts for the gain â€” consistent with the original exp4 finding.

---

## Threshold Tuning â€” edna_1.2m

Evaluated at conf=0.25 through 0.45 to test whether the precision/recall imbalance
could be corrected without retraining.

| conf | mAP@50 | Precision | Recall | F1 | Detections |
|------|--------|-----------|--------|----|------------|
| **0.25** | **0.2600** | 0.3467 | **0.4920** | 0.4068 | 17,237 |
| 0.30 | 0.2380 | 0.3923 | 0.4344 | **0.4123** | 13,448 |
| 0.35 | 0.2100 | 0.4355 | 0.3673 | 0.3985 | 10,244 |
| 0.40 | 0.1766 | 0.4835 | 0.2925 | 0.3645 | 7,349 |
| 0.45 | 0.1366 | 0.5349 | 0.2125 | 0.3042 | 4,825 |

**Conclusion:** The F1 peak is at conf=0.30 (+0.0055 over default), but at the cost of
-0.022 mAP@50. The gain is negligible. The low precision is structural â€” the model
genuinely produces false positives that no threshold can eliminate without a proportional
recall loss. Default conf=0.25 remains optimal for mAP; conf=0.30 is marginally better
for F1 only.

---

## Context: Gap to YOLOv8

For reference, YOLOv8M trained on the same balanced_dataset reached **0.575 mAP@50**
(50 epochs, see `02_yolo_experiments/yolo_results.md`). The gap to edna_1.2m is **0.315 mAP@50**.

Key differences explaining the gap:
1. YOLOv8M uses COCO-pretrained weights; edna family trains from scratch
2. YOLOv8 architecture is years more optimized (CSPDarknet, PANet, decoupled head)
3. YOLOv8M has ~25.8M params vs edna_1.2m at ~34M â€” more capacity alone does not close the gap

---

## Planned Next Runs â€” YOLOv8 on Balanced Dataset

All previous YOLOv8 weights were trained on `data/sample_dataset`, making them invalid
as fair comparisons against edna models. These three runs retrain YOLO on the same
balanced_dataset.

| Run | Model | Params | Purpose |
|-----|-------|--------|---------|
| yolov8n_balanced | yolov8n | ~3.2M | Size-matched comparison against edna_1.2m (~1.2M) |
| yolov8s_balanced | yolov8s | ~11M | Param-matched comparison against fashionnet family (~11.74M) |
| yolov8l_balanced | yolov8l | ~43.7M | Best-effort YOLO ceiling on this dataset |

### Training Commands

```bash
python scripts/training/train.py \
  --model yolov8n \
  --epochs 100 \
  --batch 16 \
  --data data/balanced_dataset/dataset.yaml \
  --patience 20

python scripts/training/train.py \
  --model yolov8s \
  --epochs 100 \
  --batch 16 \
  --data data/balanced_dataset/dataset.yaml \
  --patience 20

python scripts/training/train.py \
  --model yolov8l \
  --epochs 100 \
  --batch 16 \
  --data data/balanced_dataset/dataset.yaml \
  --patience 20
```

### Evaluation Commands

```bash
python scripts/evaluation/evaluate.py --weights models/weights/yolov8n_fashion/weights/best.pt \
  --data data/balanced_dataset/dataset.yaml
python scripts/evaluation/evaluate.py --weights models/weights/yolov8s_fashion/weights/best.pt \
  --data data/balanced_dataset/dataset.yaml
python scripts/evaluation/evaluate.py --weights models/weights/yolov8l_fashion/weights/best.pt \
  --data data/balanced_dataset/dataset.yaml
```

### Final Comparison Commands

```bash
# edna_1.2m vs yolov8n (size-matched)
python scripts/evaluation/compare_models.py \
  --custom_weights models/weights/edna_1.2m/best.pt \
  --yolo_weights   models/weights/yolov8n_fashion/weights/best.pt \
  --data           data/balanced_dataset \
  --out            docs/compare_edna_1.2m_vs_yolov8n.json

# edna_1.2m vs yolov8s (param-matched to FashionNet family)
python scripts/evaluation/compare_models.py \
  --custom_weights models/weights/edna_1.2m/best.pt \
  --yolo_weights   models/weights/yolov8s_fashion/weights/best.pt \
  --data           data/balanced_dataset \
  --out            docs/compare_edna_1.2m_vs_yolov8s.json

# edna_1.2m vs yolov8l (best-effort ceiling)
python scripts/evaluation/compare_models.py \
  --custom_weights models/weights/edna_1.2m/best.pt \
  --yolo_weights   models/weights/yolov8l_fashion/weights/best.pt \
  --data           data/balanced_dataset \
  --out            docs/compare_edna_1.2m_vs_yolov8l.json
```

# FashionNet Evaluation and Visualization Methodology

## Overview

FashionNet only tracks `val_loss` during training. To properly compare experiments and
analyze results, three components must be implemented in order:

```
1. postprocess.py       (no dependencies, enables everything else)
     â†“
2. evaluate_custom.py   (depends on postprocess.py + src/utils/metrics.py)
     â†“
3. visualize_results.py (depends on JSON output from evaluate_custom.py)
```

---

## Metrics

| Metric | Purpose |
|--------|---------|
| **mAP@50** | Standard detection metric. A prediction is correct only if class is right AND IoU >= 0.5. |
| **F1** (per-class + macro) | Harmonic mean of P and R. Better single summary than precision alone. |
| **Precision** | Of all predicted boxes, how many were correct? |
| **Recall** | Of all ground-truth objects, how many were found? |
| **Per-class AP** | Identifies which clothing categories are weak. |
| **Confusion matrix** | Shows misclassification patterns (e.g., vest_dress confused with vest). |

### Why F1 instead of just Precision?

Precision alone can be misleading. A model that predicts only 3 very confident boxes
achieves 100% precision but misses everything. F1 = `2 * P * R / (P + R)` penalises both
false positives and missed detections. Track all three (P, R, F1) but use F1 as the single
summary metric alongside mAP@50.

---

## Component 1: Post-processing Module

**File:** `src/custom_model/postprocess.py`

### `decode_predictions(raw_preds, img_size, conf_thresh, num_classes) -> List[Tensor]`

Converts FashionNet's 3 raw output tensors into detection boxes.

The model outputs `(B, 5+NC, gs, gs)` where `gs` is the grid size (80, 40, 20).
Stride = `img_size / gs` (8, 16, 32).

1. Permute from `(B, C, gs, gs)` to `(B, gs, gs, C)`
2. Generate grid offsets: `grid_x[j, i] = i`, `grid_y[j, i] = j` (meshgrid)
3. Decode center xy (channels 0â€“1):
   ```
   cx_norm = (sigmoid(raw_x) + grid_x) / gs
   cy_norm = (sigmoid(raw_y) + grid_y) / gs
   ```
4. Decode width/height (channels 2â€“3):
   ```
   w_norm = clamp(raw_w, min=0) / gs
   h_norm = clamp(raw_h, min=0) / gs
   ```
   Note: `raw_w/h` are in grid units (not log-space), matching how `build_targets` stores
   them. Clamp avoids negative dimensions.
5. Decode objectness (channel 4): `obj = sigmoid(raw_obj)`
6. Decode class scores (channels 5+): `cls_scores = sigmoid(raw_cls)`
7. Combine: `conf = obj * max(cls_scores)`
8. Filter: keep cells where `conf >= conf_thresh`
9. Concatenate detections from all 3 scales per image

Returns: list of length B, each `(D, 6)` tensor `[cx, cy, w, h, confidence, class_id]`
in normalized coords.

### `nms(detections, iou_thresh) -> Tensor`

Per-class NMS using `torchvision.ops.nms`:

1. Convert `(cx, cy, w, h)` to `(x1, y1, x2, y2)`
2. Apply class-offset trick: add `class_id * large_number` to coords so classes don't
   suppress each other
3. Call `torchvision.ops.nms(boxes, scores, iou_thresh)`

### `postprocess(raw_preds, img_size, conf_thresh, iou_thresh, num_classes, max_det)`

Top-level convenience: `decode_predictions` â†’ `nms` per image â†’ sort by confidence â†’ cap to `max_det`.

**Note on grid decoding:** `build_targets()` in `loss.py` stores box targets as
`(cx - cell_i, cy - cell_j, w, h)` in grid units. Channels 0â€“1 predict these offsets
(sigmoid applied), and channels 2â€“3 predict w/h directly in grid units (no exp/log transform).
Postprocessing must be the exact inverse of this encoding.

---

## Component 2: Evaluation Script

**File:** `scripts/evaluation/evaluate_custom.py`

### CLI Arguments

```
--weights       Path to FashionNet .pt checkpoint (required)
--data          Path to yolo/ directory (default: data/balanced_dataset)
--split         val or test (default: val)
--imgsz         Image size (default: 640)
--conf          Confidence threshold (default: 0.25)
--iou_thresh    NMS IoU threshold (default: 0.45)
--batch         Batch size (default: 16)
--device        cpu/cuda/mps/auto
--output_dir    Where to save metrics JSON (default: same dir as weights)
--num_classes   0 = auto-read from dataset.yaml
--model_scale   s/m/l (must match checkpoint)
```

### Logic Flow

1. **Load checkpoint** â€” read `config.json` from weights dir for `num_classes` and `model_scale`,
   instantiate `FashionNet(num_classes, scale)`, load `state_dict`
2. **Collect per-image GT** â€” iterate `FashionDataset` with val transforms (deterministic)
   to get post-augmentation boxes and classes per image
3. **Batched inference** â€” `model.eval()`, `torch.no_grad()`, call `postprocess()` on predictions
4. **Match predictions to GT** â€” IoU-based matching:
   - For **mAP/F1**: same-class matching only (standard)
   - For **confusion matrix**: class-agnostic matching by best IoU, then record `cm[gt_class, pred_class]`
5. **Compute metrics:**
   - Per-class AP using existing `per_class_ap` from `src/utils/metrics.py`
   - Per-class precision, recall, F1 from accumulated TP/FP/FN counts
   - `(NC+1) Ã— (NC+1)` confusion matrix (last row/col = "background"):
     - Matched prediction: `cm[gt_class, pred_class] += 1`
     - Unmatched prediction (FP): `cm[background_idx, pred_class] += 1`
     - Unmatched GT (FN): `cm[gt_class, background_idx] += 1`
6. **Save** `metrics.json` and print summary table

### Output JSON Format

```json
{
  "mAP50": 0.xxx,
  "precision": 0.xxx,
  "recall": 0.xxx,
  "F1": 0.xxx,
  "per_class": {
    "short_sleeve_top": {"AP": 0.xx, "precision": 0.xx, "recall": 0.xx, "F1": 0.xx},
    ...
  },
  "confusion_matrix": [[...]],
  "class_names": ["short_sleeve_top", ..., "background"],
  "config": { "weights": "...", "conf_thresh": 0.25, "split": "val", ... }
}
```

### Why Class-Agnostic Matching for the Confusion Matrix?

Standard mAP matching only pairs predictions with GT of the **same class**, so
misclassifications appear as both a FP and a FN. The confusion matrix needs class-agnostic
matching (best IoU regardless of class) to capture what the model *thinks* an object is vs
what it *actually* is.

### Why (NC+1) Ã— (NC+1)?

The extra row/column represents "background" (no detection). This captures:
- **FP:** model predicted something where there's nothing â†’ `cm[background, pred_class]`
- **FN:** model missed a real object â†’ `cm[gt_class, background]`
- **Misclassification:** model found the object but wrong class â†’ `cm[gt_class, pred_class]` (off-diagonal)

---

## Component 3: Visualization Script

**File:** `scripts/evaluation/visualize_results.py`

### CLI Arguments

```
--metrics_json   Path to metrics.json from evaluate_custom.py
--history_json   Path to history.json from train_custom.py
--output_dir     Where to save PNGs (default: results/plots)
--exp_dirs       Multiple experiment dirs for comparison table
--dpi            Image DPI (default: 150)
```

### Plot 1: Confusion Matrix Heatmap

- `(NC+1) Ã— (NC+1)` heatmap using `seaborn.heatmap`
- `annot=True`, `cmap="Blues"`, row-normalized
- x-axis = "Predicted", y-axis = "True"
- Includes "background" row/column for FP/FN
- Saves: `confusion_matrix.png`

### Plot 2: Training Loss Curves

- 2 subplots:
  - Left: `train_loss` and `val_loss` vs epoch
  - Right: `box`, `obj`, `cls` component losses vs epoch
- Saves: `training_curves.png`

### Plot 3: Per-class AP Bar Chart

- Horizontal bars, sorted descending by AP
- Color-coded: green (AP >= 0.5), yellow (0.3â€“0.5), red (< 0.3)
- Value labels on bars
- Vertical dashed line at mAP@50 (the mean)
- Saves: `per_class_ap.png`

### Plot 4: Per-class F1 Bar Chart

- Same structure as AP chart but for F1 scores
- Saves: `per_class_f1.png`

### Plot 5: Experiment Comparison Table

- Reads `metrics.json`, `history.json`, `config.json` from each experiment dir
- Columns: `Experiment | mAP@50 | F1 | Best val_loss | Best epoch | Key flags`
- Best value in each column highlighted in green
- Saves: `experiment_comparison.png`

### Usage Examples

```bash
# After training + evaluation:
python scripts/evaluation/visualize_results.py \
  --metrics_json models/weights/exp2_loss_fix/metrics.json \
  --history_json models/weights/exp2_loss_fix/history.json \
  --output_dir results/plots/exp2

# Compare all experiments:
python scripts/evaluation/visualize_results.py \
  --exp_dirs models/weights/exp1_baseline models/weights/exp2_loss_fix \
             models/weights/exp3_multicell models/weights/exp4_aug_medium \
  --output_dir results/plots/comparison
```

# Viralytics / FashionSense â€” Codebase Explanation

## Quick Start

```powershell
.\scripts\app\start_full_app.ps1
```

---

## 1. What This Repository Does

This is a small research system combining:

1. A computer-vision pipeline for clothing detection
2. A lightweight recommendation engine for complementary garments
3. A browser-facing application layer (FastAPI + custom frontend)
4. A natural-language clothing search subsystem (`LNIAGIA/`)
5. Supporting experimentation code for dataset preparation, training, evaluation, and comparison

The repository mixes two AI paradigms:
- **Perception** â€” detect what a user is wearing from an image or camera stream
- **Retrieval / reasoning** â€” search or recommend products based on semantic or symbolic rules

---

## 2. Architectural Decomposition

The repository is best understood as five layers.

### 2.1 Runtime Application Layer

The deployable prototype:

- `src/api/main.py`
- `src/api/schemas.py`
- `src/detection/*.py`
- `src/recommendations/*.py`
- `frontend/index.html`

Responsibilities: serving the frontend, exposing HTTP and WebSocket endpoints, loading a
detector once at startup, handling camera/image inference, converting detections into
product recommendations.

### 2.2 Vision Model and Data Pipeline Layer

Supports creating and evaluating the detector:

- `src/detection/converter.py`
- `scripts/data_prep/sample_dataset.py`
- `scripts/training/train.py`
- `scripts/evaluation/evaluate.py`
- `scripts/evaluation/evaluate_yolo_world.py`
- `scripts/data_prep/analyze_raw_dataset.py`

Classic ML pipeline: `raw dataset â†’ sampled subset â†’ converted annotations â†’ training â†’ evaluation`.

### 2.3 Custom Research Model Layer

The from-scratch detector implementation:

- `src/custom_model/dataset.py`
- `src/custom_model/model.py`
- `src/custom_model/loss.py`
- `scripts/training/train_custom.py`
- `scripts/evaluation/compare_models.py`

This layer exists because the project investigates a custom detector architecture and loss design,
not just consuming YOLOv8 as a black box.

### 2.4 Search Subsystem (`LNIAGIA/`)

A second mini-project inside the repository:

- `LNIAGIA/search_app.py`
- `LNIAGIA/llm_query_parser.py`
- `LNIAGIA/DB/models.py`
- `LNIAGIA/DB/SQLLite/DBManager.py`
- `LNIAGIA/DB/vector/*.py`

Goal: semantic clothing search from natural language, using an LLM for parsing, sentence
embeddings for retrieval, Qdrant for vector search, and metadata constraints for filtering.

### 2.5 Documentation and Experiment Artifacts

- `docs/*.md`, `docs/*.json`, `docs/*.png`
- `notebooks/*.ipynb`
- `runs/`, generated DB files, output JSON in `LNIAGIA/tests/output/`

Evidence, reports, and artifacts rather than active application logic.

---

## 3. Main Application Path

Core online path: `frontend â†’ FastAPI â†’ detector â†’ recommendation engine â†’ JSON/WebSocket response`

### 3.1 `src/api/main.py`

Orchestration module. Creates the FastAPI app, enables CORS, mounts static files, resolves
detector weights, creates singletons at startup, exposes endpoints for health, image detection,
live threshold updates, audio transcription, and camera streaming.

Key design choices:

**A) Weight discovery with `_find_weights()`**
Searches for fine-tuned weights in priority order: environment override â†’ large model â†’
medium â†’ small â†’ nano â†’ base `yolov8n.pt`. Decouples deployment from hard-coded filenames.

**B) Backend abstraction via `DETECTOR_BACKEND`**
Can choose between `FashionDetector` (fine-tuned YOLOv8) and `YOLOWorldDetector` (zero-shot).
This is a simple dependency injection pattern that makes the app inference-backend-agnostic.

**C) Startup-created shared objects**
Stores `detector`, `recommender`, `camera`, `whisper_model` as module-level globals initialized
at startup. Detection models are expensive to load; webcam logic needs shared state; Whisper
may trigger a one-time download.

**D) Whisper loading in a background task**
Speech-to-text is optional. Blocking startup on Whisper would delay API readiness. Background
loading preserves responsiveness for the critical path (image/camera detection).

**E) `/api/detect/image`**
Reads uploaded bytes â†’ OpenCV decode â†’ detect â†’ extract categories â†’ get recommendations â†’
draw annotations â†’ return detections + base64 frame.

**F) `/api/conf` and `/api/conf/{value}`**
Live confidence-threshold control. Demonstrates the precision/recall tradeoff interactively.

**G) `/api/transcribe`**
Accepts browser audio (WebM/Opus), converts to WAV via `ffmpeg`, transcribes with Faster-Whisper.

**H) `/ws/camera`**
Delegates to `CameraStream.run_session`. Keeps transport-level code in the API layer and
session/state-machine logic in a dedicated class.

### 3.2 `src/api/schemas.py`

Pydantic models for response payloads. Provides typed contract between backend and client,
automatic serialization validation, and self-documenting API structure.

Notable weakness: `DetectionResponse` uses `List[Dict[str, Any]]` instead of a typed
`List[DetectionItem]`, reducing type strictness.

---

## 4. Detection Subsystem

### 4.1 `src/detection/detector.py`

Main abstraction boundary. Contains category definitions, visualization colors,
`Detection` and `DetectionResult` dataclasses, `BaseDetector`, and `FashionDetector`.

**Why the dataclasses matter:** `Detection` and `DetectionResult` decouple the rest of the
codebase from Ultralytics' raw output types. This is an adapter-pattern design â€” alternative
backends can be swapped in more easily and testing becomes simpler.

**`FashionDetector`** wraps the Ultralytics `YOLO` object: loads weights, stores inference
thresholds, runs prediction, parses boxes into project-native dataclasses.

### 4.2 `src/detection/yolo_world.py`

Zero-shot / open-vocabulary backend. Loads `yolov8s-worldv2.pt`, injects the 13 clothing
categories via `set_classes`, uses a lower confidence threshold (zero-shot detectors have
weaker confidence calibration on task-specific categories).

Note: the code temporarily relaxes SSL verification to allow CLIP-related downloads.
In production this should be replaced with proper certificate configuration.

### 4.3 `src/detection/camera.py`

Real-time session implementation. Implements a state machine:
- `CAPTURING`
- `ANALYSING`
- `RESULTS`

**Multi-frame accumulation:** during capture, confidence per class is averaged across frames
rather than trusting one frame. Reduces sensitivity to temporary misdetections and smooths
out flicker. The brief analysing phase communicates progress and prevents jarring transitions.

### 4.4 `src/detection/converter.py`

Converts DeepFashion2-style annotations into YOLO label format. Reads sampled annotations
from `index.json`, splits train/val, clamps boxes to image boundaries, converts
`[x1, y1, x2, y2]` to normalized YOLO format, writes `dataset.yaml`.

---

## 5. Recommendation Subsystem

### 5.1 `src/recommendations/catalogue.py`

Defines a mock catalogue via a `CatalogueItem` dataclass and a hard-coded product list.
Replaces infrastructure dependence with an in-memory fixture for prototyping.

### 5.2 `src/recommendations/engine.py`

Rule-based recommender. Maps detected categories to complementary categories via `OUTFIT_RULES`,
accumulates rule scores, samples catalogue items within those categories, returns top `k`.

Why rule-based: explainable, computationally trivial, appropriate when there is no interaction
history or user-profile data. This is a symbolic recommender layered on top of a perceptual model.

Note: the module docstring mentions embedding similarity as a strategy, but the current
implementation is purely rule-based.

---

## 6. Custom Detector Research Path

### 6.1 `src/custom_model/dataset.py`

Adapts YOLO-format annotations into a PyTorch `Dataset` and `DataLoader`. Uses Albumentations
(handles bounding box transformation correctly), supports `light`, `medium`, and `heavy`
augmentation modes, uses a custom `collate_fn` because each image has a different number of boxes.

### 6.2 `src/custom_model/model.py`

Implements: basic convolution blocks (`ConvBnRelu`), residual blocks (`ResBlock`), CSP-style
blocks (`CSPBlock`), multi-scale backbone (`FashionBackbone`), FPN-like neck (`FashionNeck`),
anchor-free detection head (`DetectionHead`), `FashionNet`, and `TinyFashionNet`.

Architecture is clearly inspired by YOLO-family designs: downsampling backbone, multi-scale
features (P3, P4, P5), top-down fusion, per-scale prediction heads. This is intentional â€”
adapting successful detector ideas rather than reinventing from zero.

**`TinyFashionNet`** exists for pipeline verification rather than accuracy â€” a cheap model
to validate code paths quickly on CPU before committing to long GPU runs.

### 6.3 `src/custom_model/loss.py`

Implements: CIoU box loss, focal binary cross-entropy for objectness, BCE class loss, and
target assignment across scales.

- **CIoU**: IoU-only loss gives poor gradients when boxes don't overlap well; CIoU adds
  center-distance and aspect-ratio penalties for smoother optimization
- **Focal BCE for objectness**: dense detectors suffer severe foreground/background imbalance;
  focal loss down-weights easy negatives
- **`build_targets()`**: the `multi_cell` option assigns each GT to neighboring cells when
  near boundaries, increasing positive signal density. This is one of the most impactful
  changes in the pipeline (see `03_fashionnet_experiments/fashionnet_pipeline_fixes.md`)

---

## 7. Utility Code

### 7.1 `src/utils/metrics.py`

Evaluation utilities from first principles: IoU computation, greedy prediction-to-GT matching,
per-class AP (VOC-style 101-point interpolated), confusion matrix generation and plotting,
textual detection report, inference benchmarking.

Implementing metrics from first principles makes the evaluation methodology visible and
reusable for models outside Ultralytics (FashionNet, YOLO-World).

### 7.2 `src/utils/visualizer.py`

Extends visualization beyond bare bounding boxes. Histogram and blended annotation views
help reason about confidence distribution and display quality.

---

## 8. Frontend

### 8.1 `frontend/index.html`

Contains HTML structure, a large embedded style block, and a large embedded script block.
Single-file portability simplifies demo deployment and reduces bundling complexity.

The UI supports two modes: camera scanning and chat/voice interaction.

**Known gap:** the page expects `/api/chat`, but `src/api/main.py` does not define that
route. The chat UI is a stub or unfinished integration point.

### 8.2 `frontend/static/js/app.js`

A second frontend implementation for the scan flow. Appears auxiliary or legacy â€” `index.html`
already contains a full inline script. The source of truth for the active scan flow is the
inline JS in `index.html`.

### 8.3 `frontend/static/css/style.css`

Supplementary styling: utility classes, responsive rules, badges, toasts, skeletons.
Suggests the frontend evolved over time.

---

## 9. Training and Evaluation Scripts

### 9.1 `scripts/data_prep/sample_dataset.py`

Stratified dataset sampling from DeepFashion2 using pre-built CSV metadata. CSV-based
indexing avoids repeatedly parsing many raw annotation files. Class-balanced sampling
improves fairness across categories.

### 9.2 `scripts/training/train.py`

Fine-tunes YOLOv8 through the Ultralytics API. Encodes: model-scale selection, pretrained vs
from-scratch initialization, augmentation settings, optimizer/LR choices, early stopping.
This is the "engineering baseline" against which the custom detector is compared.

### 9.3 `scripts/training/train_custom.py`

Training loop for FashionNet. Exposes all experiment knobs: configurable loss weights,
augmentation intensity, multi-cell assignment, dropout, optimizer choice, grayscale ablations,
EMA, warmup, and scheduler variants. Not just a train loop â€” an experiment harness.

### 9.4 `scripts/evaluation/evaluate.py`

Evaluates fine-tuned YOLOv8 using Ultralytics validation flow. Post hoc evaluation needed
for specific checkpoints or confidence settings.

### 9.5 `scripts/evaluation/evaluate_yolo_world.py`

Evaluates zero-shot backend using the project's own metrics utilities. Necessary because
YOLO-World is used as a custom-configured detector, not a dataset-trained model.

### 9.6 `scripts/evaluation/compare_models.py`

Compares custom FashionNet vs YOLOv8 (or custom vs custom). Metrics include per-class mAP,
overall mAP, inference speed, parameter count, weight size. The core comparison tool for
the experimental section.

### 9.7 `scripts/data_prep/analyze_raw_dataset.py`

Exploratory data analysis on the original dataset. Analyzes class balance, box size, aspect
ratio, occlusion, and co-occurrence. Supports methodological rigor by providing empirical
justification for design decisions.

---

## 10. Tests

### 10.1 `tests/test_detector.py`

Validates basic detector behavior: return type, inference time, output shape, detection list
structure, bounding-box validity, confidence range. Sanity tests run on blank frames and base
weights â€” validates pipeline integrity more than semantic accuracy.

### 10.2 `tests/test_recommendations.py`

Tests: fallback behavior, `top_k`, required fields, non-duplication, expected dress/outwear
pairing, exclusion logic. Good unit tests because the rule-based recommender is deterministic
enough to verify meaningfully.

---

## 11. LNIAGIA Search Subsystem

### 11.1 `LNIAGIA/DB/models.py`

The ontology of the search system. Defines controlled vocabularies, field groups, realistic
generation constraints, brand/price distributions, and helper functions. Acts as domain schema,
generator configuration, and retrieval vocabulary source simultaneously.

### 11.2 `LNIAGIA/DB/SQLLite/DBManager.py`

Manages a simple SQLite database for item records. SQLite provides structured relational
storage, while Qdrant provides semantic retrieval â€” a common hybrid pattern.

### 11.3 `LNIAGIA/DB/vector/nl_mappings.py`

Maps compact symbolic values to richer natural-language descriptions and synonyms. Embedding
models work better with semantically rich text â€” `short_sleeve_top` becomes something like
"short sleeve top (t-shirt, tee)".

### 11.4 `LNIAGIA/DB/vector/description_generator.py`

Transforms structured catalog items into descriptive text for embedding. Bridges structured
data and semantic search through synthetic natural-language enrichment.

### 11.5 `LNIAGIA/DB/vector/VectorDBManager.py`

The retrieval engine. Handles embedding model loading (`BAAI/bge-base-en-v1.5`), Qdrant
collection management, vector indexing, plain semantic search, and filtered semantic search.

Uses `BGE_QUERY_PREFIX` because BGE models are optimized when queries include an instruction
prefix â€” a retrieval-quality optimization grounded in model-specific best practice.

**Strict vs non-strict search:** strict mode converts constraints into hard metadata filters;
non-strict mode retrieves broadly and penalizes mismatches. Real users often want negotiable
matches, not brittle exact filters.

Note: non-strict include-based soft boosting is currently commented out.

### 11.6 `LNIAGIA/llm_query_parser.py`

Uses Ollama with `qwen2.5:3b-instruct` to translate natural-language queries into structured
filters. LLM handles linguistic variability; controlled vocabulary constrains outputs;
validation cleans up invalid generations. A classic "LLM-to-symbolic-IR bridge."

### 11.7 `LNIAGIA/search_app.py`

CLI frontend for the search subsystem. Checks that the vector DB exists, loads the embedding
model, accepts user queries, invokes the LLM parser, asks whether to accept approximate
matches, runs filtered search, prints results.

---

## 12. Code Quality Observations

### Strengths

- Clear modular decomposition between API, detection, recommendations, custom model, and search
- Detector abstraction is well chosen and supports backend swapping cleanly
- Contains both engineering baselines and research-oriented custom implementations
- Evaluation and comparison utilities are unusually thoughtful for a student project
- Search subsystem shows strong awareness of hybrid symbolic + neural retrieval design

### Known Weaknesses

- `frontend/index.html` expects `/api/chat`, but the backend does not expose that route
- Frontend logic is split between inline JS and `app.js` â€” ambiguity about the source of truth
- Some Pydantic schemas are weaker than needed (nested models not fully enforced)
- `LNIAGIA/DB/models.py` is overloaded: schema + generator + business rules + search config
- Some conceptual claims (recommendation embeddings, soft include boosting) are broader than
  the currently active implementation

These are normal prototype-stage characteristics, not fatal flaws.

---

## 13. Technology Choices

| Technology | Why Used | Alternative |
|------------|----------|-------------|
| FastAPI | Async networking, schema-driven APIs, low boilerplate | Flask, Django |
| OpenCV + NumPy | Standard for CV prototypes, easy camera capture | PIL, imageio |
| Ultralytics YOLO | Strong baseline, fast iteration, easy model comparison | MMDetection, Detectron2 |
| Qdrant | Local deployment, metadata filtering, no external service needed | FAISS + custom layer |
| sentence-transformers (BGE) | Strong general-purpose retrieval embeddings | OpenAI embeddings |
| Ollama + qwen2.5:3b | Local LLM, no API key, handles linguistic variability | GPT-4, Mistral |

---

## 14. Recommended Reading Order

For a new researcher or evaluator:

1. `README.md`
2. `src/api/main.py`
3. `src/detection/detector.py`
4. `src/detection/camera.py`
5. `src/recommendations/engine.py`
6. `scripts/data_prep/sample_dataset.py`
7. `scripts/training/train.py`
8. `src/custom_model/model.py`
9. `src/custom_model/loss.py`
10. `scripts/training/train_custom.py`
11. `LNIAGIA/SEARCH_OVERVIEW.md`
12. `LNIAGIA/DB/vector/VectorDBManager.py`
13. `LNIAGIA/llm_query_parser.py`

This order moves from deployed behavior â†’ training methodology â†’ secondary retrieval research.

# Viralytics / FashionSense â€” Documentation Index

This directory contains the organized documentation for the FashionSense ML project:
a clothing detection system built on the DeepFashion2 dataset.

---

## Project Summary

The project investigates clothing detection from two angles:

1. **YOLOv8 fine-tuning** â€” strong engineering baseline using pretrained COCO weights
2. **FashionNet (custom)** â€” from-scratch detector research, progressively improved through
   a series of ablation experiments (FashionNet â†’ fashionnet_balanced_v1 â†’ edna family)

Both tracks are evaluated on the same balanced 11-class dataset derived from DeepFashion2.

---

## Directory Structure

```
docs/organized/
  README.md                       â† this file
  01_dataset/
    dataset_analysis.md           â† raw dataset stats, balancing methodology, train/val/test splits
  02_yolo_experiments/
    yolo_results.md               â† all YOLOv8 training runs (Tests 1â€“6), comparisons, conclusions
  03_fashionnet_experiments/
    fashionnet_pipeline_fixes.md  â† identified issues, fixes, CLI flags, 12-experiment ablation plan
    fashionnet_results.md         â† ablation results (20-epoch), full training (fashionnet_balanced_v1)
  04_edna/
    edna_results.md               â† edna_1m_balanced_100, edna_1.2m results and 3-way comparison
    edna_next_steps.md            â† threshold tuning, cos_lr+EMA, class merge, scale-up suggestions
  05_evaluation/
    evaluation_methodology.md     â† post-processing spec, mAP/F1/confusion matrix implementation plan
  06_codebase/
    codebase_explanation.md       â† full architectural walkthrough of all modules
```

---

## Key Results at a Glance

| Model | Dataset | mAP@50 | Notes |
|-------|---------|--------|-------|
| YOLOv8L (fine-tuned) | sample (10k) | 0.767 | Best YOLO on sample set |
| YOLOv8M (fine-tuned) | balanced (84k) | 0.575 | Balanced dataset baseline |
| YOLO-World (zero-shot) | sample (10k) | 0.146 | No fine-tuning |
| FashionNet (original) | sample (10k) | 0.009 | Broken loss weights |
| fashionnet_balanced_v1 | balanced (84k) | 0.276 | Fixed pipeline, 100 epochs |
| edna_1m_balanced_100 | balanced (84k) | 0.187 | Default flags, 100 epochs |
| edna_1.2m | balanced (84k) | 0.260 | scale=m, aug=medium, multi_cell |

---

## Reading Order

For a new reader, the recommended order is:

1. `01_dataset/dataset_analysis.md` â€” understand the data
2. `02_yolo_experiments/yolo_results.md` â€” understand the strong baseline
3. `03_fashionnet_experiments/fashionnet_pipeline_fixes.md` â€” understand why FashionNet was poor and what was fixed
4. `03_fashionnet_experiments/fashionnet_results.md` â€” see the fix impact
5. `04_edna/edna_results.md` â€” see the scaled-up model results
6. `04_edna/edna_next_steps.md` â€” understand what to do next
7. `05_evaluation/evaluation_methodology.md` â€” understand the evaluation infrastructure
8. `06_codebase/codebase_explanation.md` â€” understand the full system architecture

# Datset Analysis

---

## Raw Dataset Summary (364,676 items)

| Category | Count | % | Med Area | Med AR | % Occ=3 |
|---|---:|---:|---:|---:|---:|
| short_sleeve_top | 84,201 | 23.1% | 0.1941 | 0.96 | 4.3% |
| long_sleeve_top | 42,030 | 11.5% | 0.2029 | 0.90 | 1.9% |
| short_sleeve_outwear | 685 | 0.2% | 0.3097 | 0.75 | 1.2% |
| long_sleeve_outwear | 15,468 | 4.2% | 0.3546 | 0.75 | 0.7% |
| vest | 18,208 | 5.0% | 0.1523 | 0.83 | 8.2% |
| sling | 2,307 | 0.6% | 0.2001 | 0.82 | 12.7% |
| shorts | 40,783 | 11.2% | 0.1142 | 1.10 | 12.4% |
| trousers | 64,973 | 17.8% | 0.1608 | 0.61 | 7.6% |
| skirt | 37,357 | 10.2% | 0.1571 | 0.97 | 6.6% |
| short_sleeve_dress | 20,338 | 5.6% | 0.3338 | 0.66 | 3.3% |
| long_sleeve_dress | 9,384 | 2.6% | 0.3474 | 0.68 | 3.6% |
| vest_dress | 21,301 | 5.8% | 0.3139 | 0.60 | 12.6% |
| sling_dress | 7,641 | 2.1% | 0.2892 | 0.57 | 8.1% |
| **TOTAL** | **364,676** | | | | |

---

## Balanced Dataset (84,051 items, 11 classes)

Excluded: `sling` (2,307 items) and `short_sleeve_outwear` (685 items).
Sampled 7,641 items per class, stratified by occlusion level.

### Occlusion Distribution per Class

| Category | occ1 (visible) | occ2 (partial) | occ3 (heavy) |
|---|---:|---:|---:|
| short_sleeve_top | 4,753 | 2,562 | 326 |
| long_sleeve_top | 5,013 | 2,479 | 149 |
| long_sleeve_outwear | 4,923 | 2,662 | 56 |
| vest | 4,134 | 2,877 | 630 |
| shorts | 2,293 | 4,397 | 951 |
| trousers | 1,546 | 5,513 | 582 |
| skirt | 2,582 | 4,558 | 501 |
| short_sleeve_dress | 4,116 | 3,275 | 250 |
| long_sleeve_dress | 4,582 | 2,782 | 277 |
| vest_dress | 3,418 | 3,260 | 963 |
| sling_dress | 4,541 | 2,484 | 616 |

### Train / Val / Test Split (70/15/15 by image)

| Category | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| short_sleeve_top | 5,349 | 1,143 | 1,149 | 7,641 |
| long_sleeve_top | 5,340 | 1,152 | 1,149 | 7,641 |
| long_sleeve_outwear | 5,343 | 1,157 | 1,141 | 7,641 |
| vest | 5,349 | 1,149 | 1,143 | 7,641 |
| shorts | 5,333 | 1,171 | 1,137 | 7,641 |
| trousers | 5,350 | 1,158 | 1,133 | 7,641 |
| skirt | 5,359 | 1,148 | 1,134 | 7,641 |
| short_sleeve_dress | 5,367 | 1,121 | 1,153 | 7,641 |
| long_sleeve_dress | 5,338 | 1,143 | 1,160 | 7,641 |
| vest_dress | 5,343 | 1,158 | 1,140 | 7,641 |
| sling_dress | 5,356 | 1,132 | 1,153 | 7,641 |
| **TOTAL** | **58,827** | **12,632** | **12,592** | **84,051** |


# Viralytics / FashionSense Codebase Explanation

## Quick start

- `.\scripts\app\start_full_app.ps1`


## 1. What this repository is trying to do

This repository is not a single-purpose script. It is a small research system that combines:

1. A computer-vision pipeline for clothing detection.
2. A lightweight recommendation engine that suggests complementary garments.
3. A browser-facing application layer built with FastAPI and a custom frontend.
4. A persona-selection layer that switches between two model stacks: `Cruella` and `Edna`.
5. A natural-language clothing search subsystem in `LNIAGIA/`.
6. Supporting experimentation code for dataset preparation, training, evaluation, and model comparison.

At a master's-project level, the repository is interesting because it mixes two distinct AI paradigms:

- `Perception`: detect what a user is wearing from an image or camera stream.
- `Retrieval / reasoning over fashion metadata`: search or recommend products based on semantic or symbolic rules.

That makes the codebase a hybrid applied-AI system rather than just a model-training repo.

## 2. Architectural decomposition

The repository is best understood as five layers.

### 2.1 Runtime application layer

This is the deployable prototype:

- `src/api/main.py`
- `src/api/schemas.py`
- `src/detection/*.py`
- `src/recommendations/*.py`
- `frontend/index.html`

This layer is responsible for:

- serving the frontend,
- exposing HTTP and WebSocket endpoints,
- loading multiple detector backends at startup,
- routing requests by selected persona,
- handling camera/image inference,
- converting detections into product recommendations,
- passing persona-aware state into chat refinement.

### 2.2 Vision model and data pipeline layer

This layer supports creating and evaluating the detector:

- `src/detection/converter.py`
- `scripts/data_prep/sample_dataset.py`
- `scripts/training/train.py`
- `scripts/evaluation/evaluate.py`
- `scripts/evaluation/evaluate_yolo_world.py`
- `scripts/data_prep/analyze_raw_dataset.py`

This is a classic ML pipeline:

`raw dataset -> sampled subset -> converted annotations -> training -> evaluation`.

### 2.3 Custom research model layer

This is the from-scratch detector implementation:

- `src/custom_model/dataset.py`
- `src/custom_model/model.py`
- `src/custom_model/loss.py`
- `scripts/training/train_custom.py`
- `scripts/evaluation/compare_models.py`

This part exists because the project is not only consuming YOLOv8 as a black box; it also investigates a custom detector architecture and loss design.

### 2.4 Search subsystem (`LNIAGIA`)

This is effectively a second mini-project inside the repository:

- `LNIAGIA/search_app.py`
- `LNIAGIA/llm_query_parser.py`
- `LNIAGIA/DB/models.py`
- `LNIAGIA/DB/SQLLite/DBManager.py`
- `LNIAGIA/DB/vector/*.py`

Its goal is not visual detection. Its goal is semantic clothing search from natural language, using:

- an LLM for parsing,
- sentence embeddings for retrieval,
- Qdrant for vector search,
- metadata constraints for filtering.

### 2.5 Documentation and experiment artifacts

These are supporting outputs rather than core runtime code:

- `docs/*.md`
- `docs/*.json`
- `docs/*.png`
- `notebooks/*.ipynb`
- `runs/`
- generated DB files and output JSON files inside `LNIAGIA/tests/output/`

These files are still important academically, but they are mostly evidence, reports, and artifacts, not active application logic.

## 3. Main application path: how the deployed prototype works

The core online path is:

`frontend -> persona selection -> FastAPI -> persona-specific detector/parser -> recommendation/search response`.

### 3.1 `src/api/main.py`

This is the orchestration module. Its main role is not algorithmic novelty but system composition.

What it does:

- creates the FastAPI app,
- enables permissive CORS,
- mounts static files,
- resolves detector weights,
- creates singleton instances at startup,
- loads both the standard detector and the custom FashionNet detector,
- maintains persona-aware detector and camera registries,
- exposes endpoints for health, image detection, live threshold updates, audio transcription, and camera streaming.

Why this structure is used:

- FastAPI is a strong fit for ML prototypes because it gives typed endpoints, automatic docs, and async support with low ceremony.
- startup-time singleton loading avoids reloading the model per request, which would make inference unusably slow.
- WebSockets are appropriate for continuous camera interaction because the server can push frames/results incrementally.

Important design choices:

#### A) Weight discovery with `_find_weights()`

The file searches for fine-tuned weights in priority order:

- environment override,
- large model,
- medium,
- small,
- nano,
- and finally the base `yolov8n.pt`.

Why this is useful:

- it decouples deployment from hard-coded filenames,
- it allows the same app code to run across multiple experiment outcomes,
- it ensures the app still works even if no fine-tuned model is available.

Alternative:

- require a mandatory explicit config file or CLI flag.

Tradeoff:

- explicit config is cleaner and less implicit,
- but the current approach is friendlier for demos and rapid iteration.

#### B) Backend abstraction and persona routing

The current runtime architecture now has two overlapping ideas:

1. a generic backend abstraction for detector classes
2. a higher-level persona selection layer

At startup, the code prepares:

- a `Cruella` detector path, normally the trained YOLO detector
- an `Edna` detector path, normally the custom `FashionNetDetector`

This is stronger than the earlier single-detector startup design because the frontend can select a model family at runtime without restarting the app.

There is still a lower-level `DETECTOR_BACKEND` switch for the standard path, especially for YOLO-World experimentation, but the user-facing architectural concept is now persona-based rather than just "one detector backend."

#### C) Startup-created shared objects

The code stores:

- a default detector
- a persona-to-detector map
- a persona-to-camera map
- `recommender`
- `whisper_model`
- `search_service`

as module-level globals initialized during startup.

Why this is being used:

- detection models are large and expensive to load,
- webcam session logic needs shared state,
- Whisper may be slow to initialize and may trigger a one-time download.

Alternative:

- use `app.state` or a dependency-injection layer for all runtime services.

That would be more idiomatic FastAPI, but the current global registry pattern is still reasonable for a prototype and keeps the model-switching logic easy to follow.

#### D) Whisper loading in a background task

This is a notable system-design decision. The model is loaded asynchronously through `run_in_executor`.

Why:

- speech-to-text is optional functionality,
- blocking startup on Whisper would delay API readiness,
- a background load preserves responsiveness.

This is good engineering for an interactive demo: the critical path is image/camera detection, not voice chat.

#### E) `/api/detect/image` and `/api/mobile/scan`

These endpoints:

1. reads uploaded bytes,
2. decodes them with OpenCV,
3. resolves the selected persona,
4. runs the corresponding detector,
4. extracts unique categories,
5. gets recommendations,
6. draws annotations,
7. creates a persona-aware session,
8. returns structured detections plus a base64 frame.

Why OpenCV and base64:

- OpenCV is a natural choice because the detector already works on NumPy/OpenCV frames,
- base64 avoids dealing with separate binary image endpoints for the annotated preview.

Alternative:

- return raw boxes only and let the frontend draw overlays.

That would reduce bandwidth and make the frontend more flexible, but server-side annotation is simpler and guarantees consistent visualization.

#### F) `/api/conf` and `/api/conf/{value}`

These routes expose live confidence-threshold control.

Why this is pedagogically valuable:

- it surfaces an important detector hyperparameter to the user,
- it demonstrates the precision/recall tradeoff interactively,
- it helps explain why operating thresholds matter in applied ML systems.

#### G) `/api/transcribe`

This route accepts browser audio, stores it temporarily, converts it to WAV through `ffmpeg`, and transcribes it using Faster-Whisper.

Why it is implemented this way:

- browsers often record audio in WebM/Opus,
- Whisper-based libraries usually prefer PCM WAV or equivalent decoded audio.

The temporary-file approach is pragmatic and easy to debug, though not the most efficient.

Alternative:

- stream audio in-memory through `ffmpeg-python` or PyAV.

That would be more elegant, but significantly more complex.

#### H) `/ws/camera`

This endpoint now resolves the selected persona from the WebSocket query string and delegates the whole UX loop to the corresponding `CameraStream.run_session`.

That is a meaningful architectural improvement because it keeps the transport route stable while allowing different vision models to power the live scan flow.

### 3.2 `src/api/schemas.py`

This file defines Pydantic models for responses.

What it contributes:

- a typed contract between backend and client,
- automatic serialization validation,
- self-documenting API structure,
- explicit persona propagation through session, scan, and chat payloads.

Why Pydantic is being used:

- in FastAPI, Pydantic is the standard way to formalize API payloads.

One subtle weakness remains:

- `DetectionResponse` still uses `List[Dict[str, Any]]` instead of strict nested response models for detections and recommendations.

That reduces type strictness even though the persona-related contract is now clearer.

Alternative:

- use nested model classes directly.

That would be cleaner, safer, and better for generated docs.

## 4. Detection subsystem

### 4.1 `src/detection/detector.py`

This is the main abstraction boundary for detection.

It contains:

- category definitions,
- visualization colors,
- `Detection` and `DetectionResult` dataclasses,
- `BaseDetector`,
- `FashionDetector`.

#### Why the dataclasses matter

`Detection` and `DetectionResult` make the rest of the codebase independent of Ultralytics' raw output types.

This is a strong architectural decision because it creates an internal representation layer.

Benefits:

- the rest of the app is not tightly coupled to the YOLO library,
- alternative backends can be swapped in more easily,
- testing becomes simpler because results can be mocked with plain Python objects.

This is a textbook example of adapter-pattern thinking.

#### `BaseDetector`

This abstract base class defines the contract:

- subclasses must implement `detect(frame)`,
- all detectors inherit the `draw()` helper.

Why this is useful:

- both fine-tuned YOLOv8 and YOLO-World can be used interchangeably,
- code elsewhere can rely on polymorphism.

Alternative:

- skip the abstract class and just rely on duck typing.

That would still work in Python, but the current design is clearer for maintainability and for educational exposition.

#### `FashionDetector`

This class wraps the Ultralytics `YOLO` object.

What it does:

- loads weights,
- stores inference thresholds,
- runs prediction,
- parses Ultralytics boxes into project-native dataclasses.

Why this wrapper exists:

- to hide library-specific details from the rest of the system,
- to provide a stable project API regardless of detector backend,
- to centralize threshold/image-size configuration.

Alternative:

- call `YOLO.predict()` directly from the API route.

That would create tighter coupling and duplicate parsing logic. The wrapper is better.

### 4.2 `src/detection/fashionnet_detector.py`

This is a major new runtime bridge in the codebase.

Before this addition, the custom FashionNet model existed primarily as a research artifact for training and evaluation. The repository now contains an adapter that allows that model family to participate directly in the deployed application.

What it does:

- loads a FashionNet checkpoint,
- resolves configuration from checkpoint metadata when available,
- preprocesses OpenCV frames into the tensor format expected by FashionNet,
- runs the custom model,
- postprocesses raw outputs through the custom decoding/NMS pipeline,
- converts the results back into the same `Detection` / `DetectionResult` abstraction used elsewhere.

Why this file matters architecturally:

- it upgrades FashionNet from "offline experiment" to "runtime backend",
- it preserves the detector interface contract,
- it allows the frontend persona switch to map to a genuinely different vision stack.

This is a strong example of adapter-pattern thinking: a custom research model is made deployable by writing a translation layer rather than rewriting the rest of the app.

### 4.3 `src/detection/yolo_world.py`

This is the zero-shot/open-vocabulary backend.

Conceptually, it answers the research question:

"Can we perform clothing detection without fashion-specific fine-tuning?"

What it changes relative to `FashionDetector`:

- loads `yolov8s-worldv2.pt`,
- injects the project's 13 clothing categories through `set_classes`,
- uses a lower confidence threshold,
- parses results into the same internal dataclasses.

Why the lower threshold is justified:

- zero-shot detectors usually have weaker confidence calibration on task-specific categories than fully fine-tuned models.

Important note:

- the code temporarily relaxes SSL verification to allow the required CLIP-related download when configuring YOLO-World classes.

That is a pragmatic workaround, but from a production-security standpoint it is not ideal.

Alternative:

- pre-download dependencies in a controlled environment,
- or configure certificates correctly instead of bypassing verification.

From a research prototype perspective, the current approach prioritizes reproducibility under messy local environments.

### 4.4 `src/detection/camera.py`

This file implements the real-time session UX.

This is not just "camera code"; it is a state machine:

- `CAPTURING`
- `ANALYSING`
- `RESULTS`

Why this is good design:

- it imposes structure on what could otherwise become ad hoc WebSocket logic,
- it maps technical processing onto a comprehensible user experience.

#### Why accumulate detections over multiple frames

During capture, the code averages confidence per class across frames instead of trusting one frame.

This is an important applied-vision choice.

Benefits:

- reduces sensitivity to temporary misdetections,
- smooths out flicker,
- makes results more robust under small pose or lighting changes.

Alternative:

- use only the last frame,
- use temporal tracking,
- use a voting mechanism,
- use weighted exponential smoothing.

The current averaging approach is simple and sensible for a short scan window.

#### Why there is a separate analysing phase

The pause is only about half a second, but it matters UX-wise.

It:

- communicates progress,
- makes the system feel deliberate rather than abruptly jumping,
- creates time for recommendation generation without a jarring transition.

This is a subtle example of human-centered systems design.

### 4.5 `src/detection/converter.py`

This file converts DeepFashion2-style annotations into YOLO labels.

Why it exists:

- DeepFashion2 annotations are not directly in the Ultralytics training format,
- so the project needs a preprocessing bridge.

What it does:

- reads sampled annotations from `index.json`,
- splits train/val,
- clamps boxes to image boundaries,
- converts `[x1, y1, x2, y2]` into normalized YOLO format,
- writes `dataset.yaml`.

Why clamping and filtering invalid boxes matters:

- raw annotation pipelines often contain edge-case boxes,
- invalid boxes can poison training or crash downstream tools.

Alternative:

- use a library-specific dataset importer,
- or train directly in COCO-like format if the framework supports it.

But for Ultralytics YOLO, explicit YOLO-format conversion is the most straightforward route.

## 5. Recommendation subsystem

### 5.1 `src/recommendations/catalogue.py`

This layer is no longer best understood as a hard-coded Python fixture.

The recommendation catalogue is now designed around an editable external data source:

- `data/mock_store_catalogue_template.json`

Why this is important:

- it makes the mock store dynamic rather than code-embedded,
- it allows future stores to replace the catalogue without editing Python,
- it aligns the recommendation layer more closely with the attributes used by the search/parser subsystem.

The catalogue is therefore becoming an application-facing content layer rather than just a developer convenience.

### 5.2 `src/recommendations/engine.py`

This is a rule-based recommender.

Its logic:

1. map detected categories to complementary categories via `OUTFIT_RULES`,
2. accumulate rule scores,
3. sample catalogue items within those categories,
4. return the top `k`.

Why rule-based recommendations are used:

- they are explainable,
- computationally trivial,
- easy to debug,
- appropriate when there is no interaction history or user-profile data.

From an academic point of view, this is a symbolic recommender layered on top of a perceptual model.

That is a valid design choice because the recommendation problem here is not personalized recommendation at scale; it is contextual completion of an outfit.

Alternative approaches:

- collaborative filtering,
- content-based embedding similarity,
- graph-based outfit compatibility,
- learned stylistic compatibility scoring,
- CLIP-based multimodal recommendation.

Why those are not used here:

- they require more data, more modeling complexity, and often a real product corpus.

The present engine is intentionally interpretable and demonstrable.

One important note:

- the module docstring mentions embedding similarity as a strategy, but the current implementation is purely rule-based.

So the code reflects a simplified prototype rather than the full conceptual roadmap.

## 6. Custom detector research path

### 6.1 `src/custom_model/dataset.py`

This file adapts YOLO-format annotations into a PyTorch `Dataset` and `DataLoader`.

Core responsibilities:

- reading image/label pairs,
- applying Albumentations transforms,
- converting variable-length annotations into a collated target tensor.

Why Albumentations is used:

- it is strong for object-detection augmentation,
- especially because it can transform bounding boxes consistently with the image.

Why there are `light`, `medium`, and `heavy` augmentation modes:

- they support experimentation with regularization strength.

This is a practical research feature: augmentation intensity can materially affect small custom detectors trained from scratch.

Why a custom `collate_fn` is needed:

- each image has a different number of boxes,
- so default tensor stacking would fail.

This is standard for detection pipelines.

### 6.2 `src/custom_model/model.py`

This is the most research-oriented file in the repo.

It implements:

- basic convolution blocks,
- residual blocks,
- CSP-style blocks,
- a multi-scale backbone,
- an FPN-like neck,
- an anchor-free detection head,
- `FashionNet`,
- `TinyFashionNet`.

#### Why the architecture looks like this

Although it is described as custom, it is clearly inspired by modern one-stage detectors such as YOLO-family designs:

- downsampling backbone,
- multi-scale features,
- top-down fusion,
- per-scale prediction heads.

That is a sensible choice. Reinventing every design principle from zero would be academically weaker than intentionally adapting successful detector ideas.

#### `ConvBnRelu`, `ResBlock`, `CSPBlock`

These are the reusable structural primitives.

Why they exist:

- modularity,
- reduced repetition,
- clearer architectural semantics.

The CSP block is especially meaningful because it reflects awareness of compute/representation tradeoffs found in modern detectors.

#### `FashionBackbone`

This produces feature maps at three scales:

- P3,
- P4,
- P5.

Why multi-scale features are essential:

- clothing items vary in spatial scale,
- and some categories may appear as small localized regions while others span most of the person.

Alternative:

- single-scale detection,
- transformer-only encoder,
- pretrained backbone like ResNet/EfficientNet.

The current design prioritizes educational transparency and end-to-end control over raw performance.

#### `FashionNeck`

This is effectively an FPN/PAN-style fusion mechanism.

Why it is used:

- deep features have semantic richness but poor spatial precision,
- shallow features have better localization but weaker semantics,
- combining them improves detection performance across object sizes.

This is standard but important detector engineering.

#### `DetectionHead`

The head predicts:

- center offsets,
- width/height,
- objectness,
- class logits.

Why this matters:

- it turns dense feature maps into candidate detections,
- and does so in an anchor-free style.

Alternative:

- anchor-based heads,
- transformer decoder heads,
- center-based formulations like FCOS/CenterNet variants.

Anchor-free is a reasonable design because it reduces anchor-tuning complexity.

#### `TinyFashionNet`

This exists for pipeline verification rather than accuracy.

That is a very useful research engineering practice: maintain a cheap model that lets you validate code paths quickly on CPU before committing to long runs.

### 6.3 `src/custom_model/loss.py`

This file implements the custom detection loss.

Major components:

- CIoU box loss,
- focal binary cross-entropy for objectness,
- BCE class loss,
- target assignment across scales.

Why CIoU is used:

- IoU-only loss gives poor gradients when boxes do not overlap well,
- CIoU adds center-distance and aspect-ratio penalties,
- making optimization smoother.

Why focal BCE is used for objectness:

- dense detectors suffer severe foreground/background imbalance,
- most grid cells contain no object,
- focal loss down-weights easy negatives.

Why `build_targets()` is important

This function defines how ground truth is mapped to detection cells. In practice, target assignment is one of the most consequential pieces of a detector.

The optional `multi_cell` behavior is especially notable:

- it assigns a ground-truth object to neighboring cells when near boundaries,
- increasing positive signal density.

This is a practical approximation of richer assignment strategies.

Alternative assignment methods:

- anchor matching,
- dynamic label assignment such as SimOTA,
- center sampling,
- Hungarian matching.

Those are stronger or more modern in some contexts, but much harder to implement cleanly in a student project.

## 7. Utility code

### 7.1 `src/utils/metrics.py`

This file implements evaluation utilities from first principles.

Why this is academically useful:

- it makes the evaluation methodology visible,
- rather than fully outsourcing metrics to a framework.

Functions include:

- IoU computation,
- greedy prediction-to-ground-truth matching,
- per-class AP,
- confusion matrix generation and plotting,
- textual detection report,
- inference benchmarking.

The AP implementation is VOC-style 101-point interpolated AP. That is a legitimate and interpretable metric choice, though not identical to COCO's more exhaustive metric suite.

Alternative:

- rely entirely on Ultralytics' internal validation metrics.

The advantage of the custom metrics module is transparency and reuse for models outside Ultralytics, such as YOLO-World or FashionNet.

### 7.2 `src/utils/visualizer.py`

This module extends visualization beyond bare bounding boxes.

Why it exists:

- visualization is a debugging instrument in computer vision,
- not just a cosmetic feature.

The histogram and blended annotation views help reason about confidence distribution and display quality.

## 8. Frontend design and behavior

### 8.1 `frontend/index.html`

This file contains most of the active frontend implementation inline:

- HTML structure,
- a large embedded style block,
- a large embedded script block.

This is a deliberate prototype-oriented tradeoff.

Why this approach may have been chosen:

- single-file portability,
- easier demo deployment,
- reduced bundling complexity,
- simpler student iteration.

For a production frontend this would be too monolithic, but for a thesis prototype it is understandable.

The UI now supports three conceptual layers:

1. a landing screen for persona selection,
2. a camera-scanning mode,
3. a chat/voice refinement mode.

The camera experience is tightly aligned with the backend WebSocket state machine.

Important update:

- the page now uses a working `/api/chat` backend route,
- persona selection is stored client-side and propagated into scan/chat/session calls,
- the theme changes visually when the user selects `Cruella` or `Edna`,
- the user can return to the landing screen and reset the active session via a dedicated header control.

So the frontend should now be understood as a functioning integrated client rather than a partial interface stub.

### 8.2 `frontend/static/css/style.css`

This stylesheet is now a primary part of the runtime UI, not just supplementary decoration.

It encodes:

- the overall visual identity,
- responsive layout behavior,
- recommendation modal styling,
- persona-specific theming,
- interaction-state styling for scan/chat/recommendation components.

That matters because the current frontend no longer behaves like a neutral utility UI. It presents two distinct model personas visually as well as computationally.

## 9. Training and evaluation scripts

### 9.1 `scripts/data_prep/sample_dataset.py`

This script performs stratified dataset sampling from DeepFashion2 using pre-built CSV metadata.

Why this matters:

- the raw dataset is large,
- full-scale experimentation may be expensive or unnecessary for a master's prototype,
- class-balanced sampling improves fairness across categories.

Why CSV-based indexing is smart:

- parsing many raw annotation files repeatedly is slow,
- a consolidated dataframe makes sampling dramatically faster.

Alternative:

- use the entire dataset,
- or build a more sophisticated sampler with weighting by occlusion/scale.

The current method is a good compromise between practicality and statistical coverage.

### 9.2 `scripts/training/train.py`

This script fine-tunes YOLOv8 through the Ultralytics API.

What it encodes:

- model-scale selection,
- pretrained vs from-scratch initialization,
- augmentation settings,
- optimizer and learning-rate choices,
- early stopping.

Why YOLOv8 fine-tuning is being used:

- strong baseline,
- fast iteration,
- high quality with relatively little custom engineering.

The script is effectively the "engineering baseline" against which the custom detector is compared.

### 9.3 `scripts/training/train_custom.py`

This is the training loop for FashionNet.

Why this file is important:

- it operationalizes the custom architecture,
- exposes experiment knobs,
- handles scheduling, EMA, checkpoints, and history logging.

Noteworthy research features:

- configurable loss weights,
- augmentation intensity,
- multi-cell assignment,
- dropout,
- optimizer choice,
- grayscale ablations,
- EMA,
- warmup and scheduler variants.

This is exactly the kind of script one expects in an experimental thesis repo: it is not just a train loop, but an experiment harness.

### 9.4 `scripts/evaluation/evaluate.py`

This evaluates fine-tuned YOLOv8 models using the built-in Ultralytics validation flow and reports per-class AP.

Why it exists even though Ultralytics already validates during training:

- post hoc evaluation is often needed on specific checkpoints or confidence settings,
- and it creates a clearer reporting path for thesis tables.

### 9.5 `scripts/evaluation/evaluate_yolo_world.py`

This evaluates the zero-shot backend using the project's own metrics utilities rather than Ultralytics' task-specific validation.

Why this is necessary:

- YOLO-World is used here as a custom-configured detector rather than a dataset-trained model,
- so a custom evaluation loop provides consistent comparison against the fine-tuned and custom models.

### 9.6 `scripts/evaluation/compare_models.py`

This is one of the academically strongest scripts in the repo.

It compares:

- custom FashionNet,
- YOLOv8 baseline,
- or even custom-vs-custom checkpoints.

Metrics include:

- per-class mAP,
- overall mAP,
- inference speed,
- parameter count,
- weight size.

This is exactly what a comparative experimental section in a dissertation needs.

### 9.7 `scripts/data_prep/analyze_raw_dataset.py`

This performs exploratory data analysis on the original dataset and produces figures.

Why this matters academically:

- dataset properties strongly influence model behavior,
- analyzing class balance, box size, aspect ratio, occlusion, and co-occurrence gives empirical justification for design decisions.

In other words, this script supports methodological rigor, not just convenience.

## 10. Test files

### 10.1 `tests/test_detector.py`

These tests validate basic detector behavior:

- return type,
- inference time existence,
- output shape,
- detection list structure,
- bounding-box validity,
- confidence range.

These are sanity tests rather than deep correctness tests.

Why that still matters:

- for ML systems, many failures are interface failures rather than theorem-level logical bugs,
- so smoke tests are valuable.

One limitation:

- the tests run on blank frames and base weights,
- so they validate pipeline integrity more than semantic accuracy.

### 10.2 `tests/test_recommendations.py`

These test:

- fallback behavior,
- `top_k`,
- required fields,
- non-duplication,
- expected dress/outwear pairing,
- exclusion logic.

These are good unit tests because the recommendation engine is deterministic enough in structure to verify meaningfully.

## 11. The `LNIAGIA` search subsystem in detail

This subsystem is conceptually separate from the image-based app.

It solves a different problem:

"Given a natural-language clothing request, retrieve suitable items from a structured catalog."

### 11.1 `LNIAGIA/DB/models.py`

This file is the ontology of the search system.

It defines:

- controlled vocabularies,
- field groups,
- realistic generation constraints,
- brand/price distributions,
- helper functions,
- the canonical set of filterable fields.

Why this file is foundational:

- it acts as domain schema,
- generator configuration,
- and retrieval vocabulary source all at once.

In database terms, it is part schema definition, part synthetic-data prior, and part business-rule layer.

This is powerful, but also creates tight coupling: many other modules depend on it as a single source of truth.

Alternative:

- split schema, generation rules, and search config into separate files.

That would improve separation of concerns, but the current single-file ontology is easier to navigate in a student project.

### 11.2 `LNIAGIA/DB/SQLLite/DBManager.py`

This manages a simple SQLite database for item records.

Why it exists alongside the vector DB:

- SQLite provides structured relational storage,
- Qdrant provides semantic retrieval.

This is a common hybrid pattern:

- relational storage for exact records,
- vector storage for semantic similarity.

### 11.3 `LNIAGIA/DB/vector/nl_mappings.py`

This maps compact symbolic values to richer natural-language descriptions and synonyms.

Why this is clever:

- embedding models work better with semantically rich text than with terse categorical tokens,
- so `short_sleeve_top` becomes something more linguistically meaningful like "short sleeve top (t-shirt, tee)".

This improves retrieval quality without changing the structured metadata.

### 11.4 `LNIAGIA/DB/vector/description_generator.py`

This transforms structured catalog items into descriptive text for embedding.

Why it is being used:

- vector search quality is only as good as the text representation being embedded,
- rich descriptions help the embedding model capture style, material, audience, and occasion semantics.

This is a standard retrieval trick: use synthetic natural-language enrichment to bridge structured data and semantic search.

### 11.5 `LNIAGIA/DB/vector/VectorDBManager.py`

This is the retrieval engine.

It handles:

- embedding model loading,
- Qdrant collection management,
- vector indexing,
- plain semantic search,
- filtered semantic search,
- strict vs soft exclusion behavior.

Why BGE is used:

- `BAAI/bge-base-en-v1.5` is a strong general-purpose embedding model for retrieval.

Why Qdrant local storage is used:

- easy local deployment,
- metadata filtering support,
- no external service requirement.

Why the `BGE_QUERY_PREFIX` matters:

- BGE models are optimized when queries are phrased with an instruction prefix,
- so this is a retrieval-quality optimization grounded in model-specific best practice.

#### Strict vs non-strict search

This is the central design idea.

- strict mode converts include/exclude constraints into hard metadata filters,
- non-strict mode retrieves more broadly and then penalizes mismatches.

This is a good information-retrieval design because real users often want a negotiable match, not a brittle exact filter.

One caveat:

- the code currently penalizes exclusions but leaves include-based soft boosting commented out.

So the non-strict mode is partially implemented relative to its conceptual ambition.

### 11.6 `LNIAGIA/llm_query_parser.py`

This uses Ollama with `qwen2.5:3b-instruct` to translate natural-language queries into structured filters.

Why this architecture is appealing:

- the LLM handles linguistic variability,
- the controlled vocabulary constrains outputs,
- validation cleans up invalid generations.

This is a classic "LLM-to-symbolic-IR bridge."

Why validation matters:

- LLMs are generative and not guaranteed to obey schemas perfectly,
- downstream retrieval needs clean, valid field values.

The parser therefore acts as a probabilistic front end, while `_validate()` converts it into a more deterministic system component.

In the current integrated application, this LLM path is specifically associated with the `Cruella` persona.

### 11.7 `LNIAGIA/search_app.py`

This is the CLI frontend for the search subsystem.

It:

- checks that the vector DB exists,
- loads the embedding model,
- accepts user queries,
- invokes the LLM parser,
- asks whether the user accepts approximate matches,
- runs filtered search,
- prints results.

From a system-design perspective, this file is the user-facing glue for the retrieval experiment.

### 11.8 `src/api/custom_text_parser.py`

This file introduces a non-LLM parsing path for the integrated app.

Its purpose is not to outperform the LLM parser linguistically; its purpose is to give the `Edna` persona a distinct text-processing behavior that is local, deterministic, and based on handcrafted matching rules.

What it does:

- parses messages against known mappings and vocabulary,
- handles simple include/exclude detection,
- supports generic garment synonyms,
- performs lightweight refinement merging.

Why this matters:

- it creates a meaningful model-family distinction between the two personas,
- it allows the app to demonstrate a custom NLP-style path instead of always routing through the LLM,
- it offers a controllable fallback when the project does not yet expose a separate trained text model checkpoint at runtime.

## 12. Code quality observations and implicit design tradeoffs

### 12.1 Strengths

- There is clear modular decomposition between API, detection, recommendations, custom model, and search.
- The detector abstraction is well chosen and supports backend swapping cleanly.
- The repository contains both engineering baselines and research-oriented custom implementations.
- Evaluation and comparison utilities are unusually thoughtful for a student project.
- The search subsystem shows strong awareness of hybrid symbolic + neural retrieval design.

### 12.2 Weaknesses or mismatches

- Some Pydantic schemas are weaker than they could be because nested models are not fully enforced.
- `LNIAGIA/DB/models.py` is powerful but overloaded; it acts as schema, generator, business rules, and search config simultaneously.
- The `Edna` text path is currently implemented through a custom deterministic parser adapter rather than a separately deployed trained text model artifact.
- Some conceptual claims in comments or docs are broader than the currently active implementation, especially around recommendation embeddings and soft include boosting.

These are normal prototype-stage characteristics, not fatal flaws.

## 13. Why the chosen technologies make sense

### 13.1 FastAPI

Used because it provides:

- asynchronous networking,
- schema-driven APIs,
- low boilerplate,
- clean Python integration.

Alternative:

- Flask,
- Django,
- Starlette directly.

FastAPI is the best fit among these for an ML demo API.

### 13.2 OpenCV + NumPy

Used because:

- detector input/output is image-array centric,
- camera capture and JPEG encoding are easy,
- the ecosystem is standard for CV prototypes.

Alternative:

- PIL only,
- imageio,
- browser-side capture with canvas overlays.

OpenCV remains the pragmatic choice.

### 13.3 Ultralytics YOLO

Used because:

- it gives a strong baseline quickly,
- training and inference APIs are streamlined,
- model variants are easy to compare.

Alternative:

- MMDetection,
- Detectron2,
- torchvision detection models.

Ultralytics optimizes for speed of experimentation, which aligns with the project.

### 13.4 Qdrant + sentence-transformers + Ollama

This stack makes sense for a local semantic search prototype because it is:

- open-source friendly,
- locally runnable,
- relatively easy to compose.

Alternative:

- Elasticsearch + dense vectors,
- FAISS + custom metadata layer,
- hosted embeddings plus a hosted vector DB.

The current choices optimize for independence and reproducibility.

## 14. How to read the repo efficiently

If a new researcher or evaluator wanted to understand the repo quickly, the best order is:

1. `README.md`
2. `src/api/main.py`
3. `src/detection/detector.py`
4. `src/detection/camera.py`
5. `src/recommendations/engine.py`
6. `scripts/data_prep/sample_dataset.py`
7. `scripts/training/train.py`
8. `src/custom_model/model.py`
9. `src/custom_model/loss.py`
10. `scripts/training/train_custom.py`
11. `LNIAGIA/SEARCH_OVERVIEW.md`
12. `LNIAGIA/DB/vector/VectorDBManager.py`
13. `LNIAGIA/llm_query_parser.py`

That order moves from deployed behavior to training methodology to secondary retrieval research.

## 15. Final interpretation

The repository should be understood as a hybrid master's-level applied AI project with two complementary themes:

- `visual understanding of clothing` through detection,
- `semantic understanding of clothing` through search and recommendation.

Its strongest qualities are:

- modular detector abstraction,
- meaningful evaluation tooling,
- a transparent custom model path,
- and a credible hybrid search architecture.

Its main limitations are typical of an evolving research prototype:

- some UI/backend mismatch,
- some unfinished integration surfaces,
- and a few files that carry both active logic and experimental residue.

Even with those limitations, the code clearly demonstrates thoughtful design decisions rather than random assembly. The repository is using widely accepted engineering patterns for ML systems, while also exposing enough internal implementation detail to support academic explanation, comparison, and critique.

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

| Epoch | val_loss | Î” |
|-------|----------|---|
| 86 | 2.8322 | â€” |
| 90 | 2.8232 | -0.0090 |
| 95 | 2.8147 | -0.0085 |
| 100 | 2.8128 | -0.0019 |

Only 0.0194 drop over the last 15 epochs. The run used `OneCycleLR` (default when `--cos_lr` is off), which decayed LR to near-zero well before epoch 100 â€” the model ran its final epochs at effectively zero LR. More epochs on the same config will not meaningfully improve results. Switching to `CosineAnnealingLR` (`--cos_lr`) keeps a meaningful LR floor throughout.

---

## Suggestions

### 1. Threshold Tuning (no retraining, quick)

Precision (0.3467) is significantly lower than recall (0.4920) â€” the model over-predicts. The default conf=0.25 may not be optimal. Evaluate at several thresholds to find the F1 sweet spot:

```bash
for conf in 0.30 0.35 0.40 0.45; do
  python scripts/evaluation/evaluate_custom.py \
    --weights models/weights/edna_1.2m/best.pt \
    --data data/balanced_dataset \
    --conf $conf
done
```

Expected outcome: higher conf threshold will trade some recall for precision, likely improving F1 without any retraining.

---

### 2. Retrain with Cosine LR + EMA + warmup + lambda_obj

edna_1.2m used `OneCycleLR` (the default when `--cos_lr` is off), which decayed LR to
near-zero well before epoch 100 â€” the model ran its final epochs at effectively zero LR.
Switching to `CosineAnnealingLR` (`--cos_lr`) keeps a meaningful LR floor throughout â€”
this is the real fix for the plateau.

**New flags:**
- **`--cos_lr`**: CosineAnnealingLR with `eta_min = lr * 0.01` â€” avoids the near-zero stall
- **`--ema`**: ineffective at 20 epochs but fully valid at ~326K steps (100 epochs Ã— 3,263 batches)
- **`--warmup_epochs 3`**: stabilises early training at full LR â€” zero cost
- **`--lambda_obj 1.5`**: confusion matrix shows clothing absorbed into background (missed
  detections, not misclassification). Raising objectness weight pushes the model to fire
  more aggressively on potential objects

**Caveat on lambda_obj:** 1.5 may increase false positives since precision is already the
weak metric (0.3467). Consider trying `--lambda_obj 1.25` first, or pairing 1.5 with a
lower focal gamma (see "Other Code-Level Improvements" below).

```bash
python scripts/training/train_custom.py \
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

Since the problem is bg/fg confusion (not inter-class), adding more examples of the weakest
classes gives the model more signal to learn to detect those items against the background.

| Class | Current AP |
|-------|-----------|
| short_sleeve_top | 0.1284 |
| long_sleeve_top | 0.1448 |

**Caveat:** the dataset is already balanced (~4-5K images per class). Adding images only
for weak classes creates imbalance. Keep the gap reasonable â€” ~1-2K extra images per
weak class should help without significantly hurting the stronger classes.

---

### 4. Class Merge â€” ruled out

~~short_sleeve_top + long_sleeve_top â†’ top~~

Previously considered but **ruled out after confusion matrix analysis**. Class merging
only helps when the model confuses one class for another (off-diagonal confusion matrix).
The confusion matrix shows clothing being absorbed by the background (FN column), not
misclassified between classes. Merging would not fix missed detections.

---

### 5. Larger Model Scale (~50h+)

edna_1.2m uses model_scale=m (~34M params). The FashionNet family also supports scale=l
(~63M params). Given YOLOv8M (25.8M params) outperforms edna_1.2m by ~0.31 mAP despite
fewer parameters, model capacity alone is not the bottleneck â€” but scale=l is worth
testing before concluding on architecture limits.

```bash
python scripts/training/train_custom.py \
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

Flag tuning alone (cos_lr, EMA, warmup, lambda_obj) will likely gain **+0.02â€“0.05 mAP**,
plateauing around **~0.30â€“0.32 mAP@50** even with perfect hyperparameters. Reaching 0.60+
required addressing architectural/methodology gaps. These code changes are likely
**worth more than all the flag tweaks combined**.

All changes below are **implemented and included** in the proposed training command above.

### C1. IoU-aware Objectness Targets â€” done (loss.py)

`build_targets` previously set `obj_mask = 1.0` for all positive cells regardless of
localization quality. Now uses CIoU between prediction and GT box as a soft objectness
target (`obj_mask = iou.detach().clamp(0)`), so confidence correlates with localization
quality. This is how YOLOv5/v8 train objectness.

### C2. Mosaic Augmentation â€” done (dataset.py, `--mosaic` flag)

4-image mosaic combines training images into one tile: 4x batch diversity, varied object
scales and positions, implicit small-object training. Uses letterbox resizing to preserve
aspect ratio. Enabled with `--mosaic` flag.

### Other Code-Level Fixes Applied

| Issue | Fix | Status |
|-------|-----|--------|
| `beta1=0.937` (non-standard AdamW) | Changed to `0.9` | Done |
| `weight_decay=5e-4` (too low for AdamW) | Exposed as `--weight_decay` flag, default `0.01` | Done |
| Label smoothing missing | cls target set to `0.95` instead of `1.0` in `build_targets` | Done |

### Remaining (not yet implemented)

| Issue | Location | Fix | Impact |
|-------|----------|-----|--------|
| Focal loss gamma=1.5 | loss.py | Try gamma=1.0 to reduce suppression of easy negatives | Small |

---

## Priority Order

| Priority | Suggestion | Status | Est. mAP gain |
|----------|-----------|--------|---------------|
| 1 | Threshold tuning | Done | +0.005 F1 |
| 2 | C1: IoU-aware objectness targets | Done | Largest available gain |
| 3 | C2: Mosaic augmentation | Done | High |
| 4 | Retrain with all flags + code changes | **Ready to run** | +0.02â€“0.05 mAP (flags) + C1/C2 gains |
| 5 | Add images to weak classes | Not started | Addresses missed detections |
| 6 | Scale=l retrain | Not started | Architecture ceiling test |
| ~~7~~ | ~~Class merge~~ | â€” | Ruled out â€” wrong failure mode |

# FashionNet Evaluation & Visualization Plan

## Overview

FashionNet currently only tracks `val_loss` during training. To properly compare experiments and analyze results, we need:

1. **Post-processing** â€” decode raw grid outputs into usable detection boxes
2. **Evaluation** â€” compute mAP@50, F1, precision, recall, confusion matrix
3. **Visualization** â€” generate plots for analysis

These must be implemented in order (each depends on the previous).

---

## Metrics to Compute

| Metric | Why |
|--------|-----|
| **mAP@50** | Standard detection metric. A prediction is correct only if class is right AND IoU >= 0.5. |
| **F1** (per-class + macro) | Harmonic mean of precision and recall. Better single summary than precision alone â€” precision can look great if the model only predicts few high-confidence boxes. |
| **Precision** | Of all predicted boxes, how many were correct? |
| **Recall** | Of all ground-truth objects, how many were found? |
| **Per-class AP** | Identifies which clothing categories are weak. |
| **Confusion matrix** | Shows misclassification patterns (e.g., vest_dress confused with vest). |

---

## Component 1: Post-processing Module

**File:** `src/custom_model/postprocess.py`

### `decode_predictions(raw_preds, img_size, conf_thresh, num_classes) -> List[Tensor]`

Converts FashionNet's 3 raw output tensors into detection boxes.

**Grid decoding math (per scale):**

The model outputs `(B, 5+NC, gs, gs)` where `gs` is the grid size (80, 40, 20). The stride = `img_size / gs` (8, 16, 32).

1. Permute from `(B, C, gs, gs)` to `(B, gs, gs, C)`
2. Generate grid offsets: `grid_x[j, i] = i`, `grid_y[j, i] = j` (meshgrid)
3. Decode center xy (channels 0-1):
   ```
   cx_norm = (sigmoid(raw_x) + grid_x) / gs
   cy_norm = (sigmoid(raw_y) + grid_y) / gs
   ```
4. Decode width/height (channels 2-3):
   ```
   w_norm = clamp(raw_w, min=0) / gs
   h_norm = clamp(raw_h, min=0) / gs
   ```
   Note: `raw_w/h` are in grid units (not log-space), matching how `build_targets` stores them. Clamp to avoid negative dimensions.
5. Decode objectness (channel 4): `obj = sigmoid(raw_obj)`
6. Decode class scores (channels 5+): `cls_scores = sigmoid(raw_cls)`
7. Combine: `conf = obj * max(cls_scores)`
8. Filter: keep cells where `conf >= conf_thresh`
9. Concatenate detections from all 3 scales per image

Returns: list of length B, each `(D, 6)` tensor `[cx, cy, w, h, confidence, class_id]` in normalized coords.

### `nms(detections, iou_thresh) -> Tensor`

Per-class NMS using `torchvision.ops.nms`:

1. Convert `(cx, cy, w, h)` to `(x1, y1, x2, y2)`
2. Apply class-offset trick: add `class_id * large_number` to coords so classes don't suppress each other
3. Call `torchvision.ops.nms(boxes, scores, iou_thresh)`

### `postprocess(raw_preds, img_size, conf_thresh, iou_thresh, num_classes, max_det) -> List[Tensor]`

Top-level convenience: `decode_predictions` â†’ `nms` per image â†’ sort by confidence â†’ cap to `max_det`.

---

## Component 2: Evaluation Script

**File:** `scripts/evaluation/evaluate_custom.py`

### CLI Arguments

```
--weights       Path to FashionNet .pt checkpoint (required)
--data          Path to yolo/ directory (default: data/balanced_dataset)
--split         val or test (default: val)
--imgsz         Image size (default: 640)
--conf          Confidence threshold (default: 0.25)
--iou_thresh    NMS IoU threshold (default: 0.45)
--batch         Batch size (default: 16)
--device        cpu/cuda/mps/auto
--output_dir    Where to save metrics JSON (default: same dir as weights)
--num_classes   0 = auto-read from dataset.yaml
--model_scale   s/m/l (must match checkpoint)
```

### Logic Flow

1. **Load checkpoint** â€” read `config.json` from weights dir for `num_classes` and `model_scale`, instantiate `FashionNet(num_classes, scale)`, load `state_dict`
2. **Collect per-image GT** â€” iterate `FashionDataset` with val transforms (deterministic) to get post-augmentation boxes and classes per image
3. **Batched inference** â€” `model.eval()`, `torch.no_grad()`, call `postprocess()` on predictions
4. **Match predictions to GT** â€” IoU-based matching:
   - For **mAP/F1**: same-class matching only (standard)
   - For **confusion matrix**: class-agnostic matching by best IoU, then record `cm[gt_class, pred_class]`
5. **Compute metrics:**
   - Per-class AP using existing `per_class_ap` from `src/utils/metrics.py`
   - Per-class precision, recall, F1 from accumulated TP/FP/FN counts
   - `(NC+1) x (NC+1)` confusion matrix (last row/col = "background"):
     - Matched prediction: `cm[gt_class, pred_class] += 1` (captures correct + misclassified)
     - Unmatched prediction (FP): `cm[background_idx, pred_class] += 1`
     - Unmatched GT (FN): `cm[gt_class, background_idx] += 1`
6. **Save** `metrics.json` and print summary table

### Output JSON Format

```json
{
  "mAP50": 0.xxx,
  "precision": 0.xxx,
  "recall": 0.xxx,
  "F1": 0.xxx,
  "per_class": {
    "short_sleeve_top": {"AP": 0.xx, "precision": 0.xx, "recall": 0.xx, "F1": 0.xx},
    ...
  },
  "confusion_matrix": [[...]],
  "class_names": ["short_sleeve_top", ..., "background"],
  "config": { "weights": "...", "conf_thresh": 0.25, "split": "val", ... }
}
```

---

## Component 3: Visualization Script

**File:** `scripts/evaluation/visualize_results.py`

### CLI Arguments

```
--metrics_json   Path to metrics.json from evaluate_custom.py
--history_json   Path to history.json from train_custom.py
--output_dir     Where to save PNGs (default: results/plots)
--exp_dirs       Multiple experiment dirs for comparison table
--dpi            Image DPI (default: 150)
```

### Plot 1: Confusion Matrix Heatmap

`plot_confusion_matrix_heatmap(metrics, output_dir, dpi)`

- `(NC+1) x (NC+1)` heatmap using `seaborn.heatmap`
- `annot=True`, `cmap="Blues"`, row-normalized option
- x-axis = "Predicted", y-axis = "True"
- Includes "background" row/column for FP/FN
- Saves: `confusion_matrix.png`

### Plot 2: Training Loss Curves

`plot_training_curves(history, output_dir, dpi)`

- 2 subplots side-by-side:
  - **Left:** `train_loss` and `val_loss` vs epoch (2 lines)
  - **Right:** `box`, `obj`, `cls` component losses vs epoch (3 lines)
- Saves: `training_curves.png`

### Plot 3: Per-class AP Bar Chart

`plot_per_class_ap(metrics, output_dir, dpi)`

- Horizontal bars, sorted descending by AP
- Color-coded: green (AP >= 0.5), yellow (0.3-0.5), red (< 0.3)
- Value labels on bars
- Vertical dashed line at mAP@50 (the mean)
- Saves: `per_class_ap.png`

### Plot 4: Per-class F1 Bar Chart

`plot_per_class_f1(metrics, output_dir, dpi)`

- Same structure as AP chart but for F1 scores
- Saves: `per_class_f1.png`

### Plot 5: Experiment Comparison Table

`plot_experiment_comparison(exp_dirs, output_dir, dpi)`

- Reads `metrics.json`, `history.json`, `config.json` from each experiment dir
- Columns: `Experiment | mAP@50 | F1 | Best val_loss | Best epoch | Key flags`
- Best value in each column highlighted in green
- Renders using `matplotlib.table`
- Also prints to stdout as formatted text
- Saves: `experiment_comparison.png`

### Usage Examples

```bash
# After training + evaluation:
python scripts/evaluation/visualize_results.py \
  --metrics_json models/weights/exp2_loss_fix/metrics.json \
  --history_json models/weights/exp2_loss_fix/history.json \
  --output_dir results/plots/exp2

# Compare all experiments:
python scripts/evaluation/visualize_results.py \
  --exp_dirs models/weights/exp1_baseline models/weights/exp2_loss_fix \
             models/weights/exp3_multicell models/weights/exp4_aug_medium \
  --output_dir results/plots/comparison
```

---

## Implementation Order

```
1. postprocess.py       (no dependencies, enables everything else)
     â†“
2. evaluate_custom.py   (depends on postprocess.py + src/utils/metrics.py)
     â†“
3. visualize_results.py (depends on JSON output from evaluate_custom.py)
```

---

## Key Design Decisions

### Why F1 instead of just Precision?

Precision alone can be misleading. A model that only predicts 3 very confident boxes gets 100% precision but misses everything. F1 = `2 * P * R / (P + R)` penalises both false positives and missed detections equally. We track all three (P, R, F1) but use F1 as the single summary metric alongside mAP@50.

### Why class-agnostic matching for the confusion matrix?

Standard mAP matching only pairs predictions with GT of the **same class**, so misclassifications appear as both a FP and a FN. The confusion matrix needs class-agnostic matching (best IoU regardless of class) to capture what the model *thinks* an object is vs what it *actually* is.

### Why (NC+1) x (NC+1) confusion matrix?

The extra row/column represents "background" (no detection). This captures:
- **FP:** model predicted something where there's nothing â†’ `cm[background, pred_class]`
- **FN:** model missed a real object â†’ `cm[gt_class, background]`
- **Misclassification:** model found the object but called it the wrong class â†’ `cm[gt_class, pred_class]` (off-diagonal)

### Grid decoding matches loss encoding

The `build_targets` in `loss.py` stores box targets as `(cx - cell_i, cy - cell_j, w, h)` in grid units. The model's channels 0-1 predict these offsets (sigmoid applied), and channels 2-3 predict w/h directly in grid units (no exp/log transform). Postprocessing must be the exact inverse of this encoding.

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
- `medium`: adds random scale (0.7-1.3x), rotation (Â±10Â°), translate (Â±10%)
- `heavy`: aggressive scale (0.5-1.5x), rotation (Â±15Â°), Gaussian noise, coarse dropout

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
| `--grayscale` | off | Convert images to grayscale (3ch repeated) â€” tests shape vs colour |
| `--warmup_epochs` | 0 | Linear LR warmup before main schedule (0 = disabled) |
| `--optimizer` | adamw | Optimizer: `adamw` or `sgd` (momentum=0.937, nesterov) |
| `--ema` | off | Exponential Moving Average of weights (used for val/inference) |

---

## Experiment Configurations

All experiments use the balanced dataset with `--max_samples 2000 --epochs 20` for fast iteration. Compare results using `val_loss` from `history.json` (see How to Compare below). Each adds one change over the previous to isolate individual impact.

### Experiment 1 â€” Baseline (fixed num_classes only)

Purpose: Establish baseline with correct num_classes but **old** lambda_box=0.05.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --lambda_box 0.05 \
  --output models/weights/exp1_baseline
```

---

### Experiment 2 â€” Loss Weights Fix

Purpose: Test impact of corrected box loss weight (0.05 â†’ 5.0). **Expected biggest single improvement.**

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda \
  --output models/weights/exp2_loss_fix
```

---

### Experiment 3 â€” Loss Fix + Multi-Cell Assignment

Purpose: Test if more positive training signal improves convergence.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell \
  --output models/weights/exp3_multicell
```

---

### Experiment 4 â€” Loss Fix + Multi-Cell + Medium Augmentation

Purpose: Test scale/rotation augmentation impact.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --output models/weights/exp4_aug_medium
```

---

### Experiment 5 â€” Loss Fix + Multi-Cell + Heavy Augmentation

Purpose: Test if heavy augmentation helps or hurts with limited samples.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment heavy \
  --output models/weights/exp5_aug_heavy
```

---

### Experiment 6 â€” Best Config + Lower LR + Cosine Schedule

Purpose: Test if slower learning rate with cosine annealing improves convergence stability.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --lr 0.0005 --cos_lr \
  --output models/weights/exp6_cos_lr
```

---

### Experiment 7 â€” Best Config + Dropout

Purpose: Test regularisation impact on a from-scratch model.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --dropout 0.1 \
  --output models/weights/exp7_dropout
```

---

### Experiment 8 â€” Grayscale Only

Purpose: Test if removing colour information forces the model to learn shape/silhouette features, improving discrimination between same-colour clothing types. Uses the loss fix from Exp 2 but no other changes, to isolate the grayscale effect.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --grayscale \
  --output models/weights/exp8_grayscale
```

---

### Experiment 9 â€” Grayscale + Best Config

Purpose: Combine grayscale with the best configuration from Experiments 3-7. Replace flags below with whichever config won.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium --grayscale \
  --output models/weights/exp9_grayscale_best
```

---

### Experiment 10 â€” Best Config + Warmup

Purpose: Test if a 3-epoch linear warmup stabilises early training for a from-scratch model. Requires `--cos_lr` since OneCycleLR has its own built-in warmup.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --cos_lr --warmup_epochs 3 \
  --output models/weights/exp10_warmup
```

---

### Experiment 11 â€” SGD + Momentum

Purpose: Test if SGD with momentum (standard for YOLO detectors) converges better than AdamW for from-scratch CNN training. SGD is generally slower per epoch but can reach a better final mAP.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --optimizer sgd --lr 0.01 \
  --output models/weights/exp11_sgd
```

---

### Experiment 12 â€” Best Config + EMA

Purpose: Test if Exponential Moving Average of model weights improves validation mAP at effectively zero training cost. EMA smooths out noisy weight updates and is used by default in YOLOv5/v8.

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --max_samples 2000 --epochs 20 \
  --batch 32 --device cuda --multi_cell --augment medium \
  --ema \
  --output models/weights/exp12_ema
```

---

## Metrics to Track

### Primary: mAP@50
A prediction is correct only if the predicted class is right AND the box overlaps the ground truth by â‰¥ 50% (IoU â‰¥ 0.50). This is the standard single-number summary for object detection. Higher = better.

### Per-class AP
The most important metric for diagnosing the same-colour confusion issue. Instead of one average number, you get AP for each class individually (e.g., shorts: 0.45, trousers: 0.62). Compare per-class AP between colour and grayscale experiments to see which clothing types benefit from removing colour.

### Precision / Recall
- **Precision**: of all boxes the model predicted, what fraction were correct? High precision = few false positives.
- **Recall**: of all ground-truth objects, what fraction did the model find? High recall = few missed detections.
- There is always a trade-off â€” a model that predicts everything has high recall but low precision.

### val_loss (proxy only)
Useful for quick iteration during training. Lower is generally better, but val_loss can disagree with mAP â€” a model with lower loss can still produce worse mAP if its confidence thresholds are poorly calibrated. Use val_loss to compare runs quickly, but confirm the winner with actual mAP evaluation.

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

Once the best config is identified, run a full training on the complete balanced dataset:

```bash
python scripts/training/train_custom.py \
  --data data/balanced_dataset --epochs 100 \
  --batch 32 --device cuda \
  <best flags from experiments> \
  --output models/weights/fashionnet_v2
```

Then evaluate against YOLOv8 with `scripts/evaluation/compare_models.py`.

# FashionNet Experiment Results

## Setup

- Dataset: balanced_dataset, 2000 samples (train) / 400 samples (val)
- Epochs: 20
- Batch: 32
- Device: CUDA (NVIDIA GPU, 16GB VRAM)
- Time per experiment: ~10 minutes
- Metric: best val_loss across epochs (lower is better)

---

## Results

| Exp | Config | Best Epoch | val_loss | box | obj | cls |
|-----|--------|-----------|----------|-----|-----|-----|
| exp1_baseline | lambda_box=0.05 (broken) | 20 | 0.5228* | 2.3030 | 0.0030 | 0.7513 |
| exp2_loss_fix | lambda_box=5.0 | 20 | 10.0723 | 1.9166 | 0.0029 | 0.9171 |
| exp3_multicell | + multi_cell | 20 | 10.2890 | 1.9619 | 0.0066 | 0.9200 |
| exp4_aug_medium | + augment medium | 18 | **8.8799** | 1.7275 | 0.0075 | 0.9144 |
| exp5_aug_heavy | + augment heavy | 20 | 12.3539 | 2.4203 | 0.0071 | 0.9272 |
| exp6_cos_lr | lr=0.0005 + cos_lr | 20 | 10.0691 | 1.9094 | 0.0059 | 0.9141 |
| exp7_dropout | + dropout=0.1 | 19 | 10.3805 | 1.9743 | 0.0061 | 0.9160 |
| exp8_grayscale | grayscale only | 20 | 12.6160 | 2.4354 | 0.0026 | 0.9094 |
| exp9_grayscale_best | grayscale + medium aug | 18 | 12.8184 | 2.4691 | 0.0065 | 0.9120 |
| exp10_warmup | + warmup_epochs=3 + cos_lr | 18 | 10.3826 | 1.9826 | 0.0064 | 0.9121 |
| exp11_sgd | optimizer=sgd, lr=0.01 | 20 | 12.3335 | 2.3561 | 0.0071 | 0.8860 |
| exp12_ema | + ema | 20 | 16.4741 | 1.9745 | 0.0066 | 0.9177 |

*exp1 val_loss uses lambda_box=0.05 so the box component is weighted ~100x less than all other experiments â€” not directly comparable.

---

## Analysis

### Winner: exp4 â€” multi_cell + medium augmentation (val_loss 8.88)

The combination of multi-cell GT assignment and medium augmentation (scale Â±30%, rotation Â±10%, translate Â±10%) gave the best result. This also had the lowest raw box loss (1.7275), meaning the model is learning to regress boxes more accurately.

### Grayscale hurts

Removing colour (exp8: 12.62, exp9: 12.82) consistently made results worse. Colour information is discriminative for this task â€” clothing categories differ in shape but colour also provides useful signal. The original hypothesis was that same-colour confusion was a problem, but the data suggests colour helps more than it hurts overall.

### Heavy augmentation hurts

exp5 (heavy) at 12.35 is worse than exp4 (medium) at 8.88. With only 2000 samples at 20 epochs, the aggressive scale/rotation/noise in heavy mode is too destructive â€” the model can't learn fast enough to handle the increased variance.

### Smaller changes had no clear benefit

- **Dropout** (exp7: 10.38 vs exp4: 8.88) â€” no benefit, possibly slightly harmful
- **Warmup** (exp10: 10.38) â€” no measurable improvement at 20 epochs
- **SGD** (exp11: 12.33) â€” worse than AdamW at this epoch count; SGD needs more epochs
- **EMA** (exp12: 16.47) â€” misleading result; with decay=0.9999 the EMA model needs thousands of batches to warm up. Not useful at 20 epochs.

---

---

## Full Training â€” fashionnet_balanced_v1

### Setup

- Config: exp4 winner (multi_cell + augment medium)
- Dataset: full balanced_dataset, 52,199 training images (no sample cap)
- Epochs: 100
- Batch: 32
- Device: CUDA (NVIDIA GPU, 16GB VRAM)
- Training time: 612m 28s (~10h 12m)
- Best val_loss: 3.0591 (epoch 87)

### Comparison vs Original FashionNet

| Metric | fashionnet_balanced_v1 | fashionnet (original) |
|--------|----------------------|----------------------|
| mAP@50 (overall) | **0.2756** | 0.0006 |
| Inference (ms/img) | 3.3 | 3.2 |
| FPS | 300.9 | 309.7 |
| Parameters (M) | 11.74 | 11.74 |
| Weights size (MB) | 141.2 | 141.2 |

### Per-class mAP@50

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

fashionnet_balanced_v1 outperforms the original by **0.2750 mAP@50** across all classes. The improvements from fixing the pipeline (lambda_box, multi_cell, augmentation) combined with full training account for essentially all of this gain.

---

### Notes

- Worst performing classes: short_sleeve_top (0.1606), long_sleeve_top (0.1880), skirt (0.1859)
- short_sleeve_top and long_sleeve_top being the two worst is likely inter-class confusion (visually almost identical) rather than a data quantity problem
- The jump from 20-epoch quick tests (val_loss ~8.88) to 100 full epochs (val_loss 3.06) shows training time has significant impact

---

## Considerations for v2

### Option A â€” Merge similar classes
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

Reduces from 11 â†’ 7 classes. Requires rebuilding labels and dataset.yaml.

### Option B â€” Add images to weakest classes
Only useful if classes are visually distinct but underrepresented. Less likely to help for short/long sleeve top confusion since the model already has 52K images to learn from.

---

## Next Step

Compare fashionnet_balanced_v1 against YOLOv8L:

```bash
python scripts/evaluation/compare_models.py \
  --custom_weights models/weights/fashionnet_balanced_v1/best.pt \
  --yolo_weights models/weights/yolov8l_fashion/weights/best.pt \
  --data data/balanced_dataset \
  --out docs/compare_fashionnet_v1_vs_yolov8l.json
```

----------------------
-------  EDNA --------
----------------------

## Full Training â€” edna_1m_balanced_100

### Setup

- Config: no explicit flags (default settings)
- Dataset: full balanced_dataset, 52,199 training images (no sample cap)
- Epochs: 100
- Batch: 32 (3263 batches/epoch)
- Device: CUDA (NVIDIA GPU, 16GB VRAM)
- Training time: 2064m 44s (~34h 24m)
- Best val_loss: 2.6953 (epoch 63)
- Weights: `models/weights/edna_1m_balanced_100/best.pt`

### Evaluation â€” fashionnet_balanced_v1 vs edna_1m_balanced_100

Evaluated with `scripts/evaluation/evaluate_custom.py`, val split (11,186 images), conf=0.25, NMS IoU=0.45.

| Metric | fashionnet_balanced_v1 | edna_1m_balanced_100 |
|--------|----------------------|----------------------|
| mAP@50 | **0.1930** | 0.1869 |
| Precision | 0.3356 | **0.3479** |
| Recall | **0.3870** | 0.3723 |
| F1 | 0.3594 | **0.3597** |
| Best val_loss | 3.0591 | **2.6953** |
| Best epoch | 87 | 63 |
| Key flags | aug=medium, multi_cell | â€” |

### Per-class breakdown

| Category | fashionnet_balanced_v1 | edna_1m_balanced_100 |
|----------|----------------------|----------------------|
| short_sleeve_top | AP=0.0860  P=0.227  R=0.303  F1=0.260 | AP=0.0877  P=0.225  R=0.324  F1=0.266 |
| long_sleeve_top | AP=0.0999  P=0.324  R=0.216  F1=0.259 | AP=0.0709  P=0.340  R=0.178  F1=0.234 |
| long_sleeve_outwear | AP=0.2823  P=0.526  R=0.425  F1=0.470 | AP=0.2862  P=0.534  R=0.422  F1=0.472 |
| vest | AP=0.2712  P=0.338  R=0.482  F1=0.398 | AP=0.2727  P=0.376  R=0.504  F1=0.431 |
| shorts | AP=0.2624  P=0.369  R=0.510  F1=0.429 | AP=0.2781  P=0.381  R=0.519  F1=0.439 |
| trousers | AP=0.2111  P=0.322  R=0.548  F1=0.405 | AP=0.1950  P=0.327  R=0.471  F1=0.386 |
| skirt | AP=0.1380  P=0.224  R=0.443  F1=0.298 | AP=0.1179  P=0.216  R=0.377  F1=0.275 |
| short_sleeve_dress | AP=0.1636  P=0.392  R=0.297  F1=0.338 | AP=0.1620  P=0.352  R=0.329  F1=0.340 |
| long_sleeve_dress | AP=0.1361  P=0.422  R=0.242  F1=0.307 | AP=0.1304  P=0.420  R=0.234  F1=0.300 |
| vest_dress | AP=0.2163  P=0.404  R=0.395  F1=0.399 | AP=0.1792  P=0.417  R=0.354  F1=0.383 |
| sling_dress | AP=0.2555  P=0.367  R=0.373  F1=0.370 | AP=0.2756  P=0.488  R=0.363  F1=0.416 |

### Analysis

The two models are effectively tied on F1 (0.3594 vs 0.3597), despite edna_1m_balanced_100 training 3x longer and achieving a better val_loss (2.6953 vs 3.0591). fashionnet_balanced_v1 edges out on mAP@50 (0.1930 vs 0.1869) and recall.

edna_1m_balanced_100 wins on: long_sleeve_outwear, vest, shorts, sling_dress (slightly better AP/F1).  
fashionnet_balanced_v1 wins on: long_sleeve_top (AP 0.0999 vs 0.0709), trousers, skirt, overall mAP.

The aug=medium + multi_cell flags in fashionnet_balanced_v1 appear to provide marginal but real benefit for mAP, particularly for harder long-tail classes. The significantly lower val_loss of edna_1m_balanced_100 does not translate into better detection metrics, suggesting val_loss and mAP@50 are not tightly coupled at this scale.

---

## Full Training â€” edna_1.2m

### Setup

- Config: aug=medium, multi_cell=true, model_scale=m (34.07M params), optimizer=adamw
- Dataset: full balanced_dataset, 52,199 training images (no sample cap)
- Epochs: 100
- Batch: 16
- Device: CUDA (NVIDIA GPU, 16GB VRAM)
- Training time: 2093m 31s (~34h 53m)
- Best val_loss: 2.8128 (epoch 100)
- Weights: `models/weights/edna_1.2m/best.pt`

### Evaluation â€” edna_1.2m

Evaluated with `scripts/evaluation/evaluate_custom.py`, val split (11,186 images), conf=0.25, NMS IoU=0.45.

| Metric | edna_1.2m |
|--------|-----------|
| mAP@50 | **0.2600** |
| Precision | 0.3467 |
| Recall | 0.4920 |
| F1 | **0.4068** |
| Best val_loss | 2.8128 |
| Best epoch | 100 |
| Key flags | aug=medium, multi_cell |

### Per-class breakdown

| Category | AP | P | R | F1 |
|----------|----|---|---|----|
| short_sleeve_top | 0.1284 | 0.215 | 0.440 | 0.289 |
| long_sleeve_top | 0.1448 | 0.329 | 0.338 | 0.334 |
| long_sleeve_outwear | **0.3734** | 0.524 | 0.552 | **0.537** |
| vest | 0.3516 | 0.368 | 0.594 | 0.455 |
| shorts | 0.3290 | 0.414 | 0.592 | 0.487 |
| trousers | 0.2463 | 0.318 | 0.604 | 0.417 |
| skirt | 0.2086 | 0.236 | 0.556 | 0.332 |
| short_sleeve_dress | 0.2705 | 0.404 | 0.445 | 0.424 |
| long_sleeve_dress | 0.2240 | 0.417 | 0.381 | 0.398 |
| vest_dress | 0.2706 | 0.406 | 0.478 | 0.439 |
| sling_dress | 0.3128 | 0.430 | 0.415 | 0.422 |

### 3-Way Comparison

| Experiment | mAP@50 | F1 | Best val_loss | Best epoch | Key flags |
|------------|--------|----|---------------|------------|-----------|
| fashionnet_balanced_v1 | 0.1930 | 0.3594 | 3.0591 | 87 | aug=medium, multi_cell |
| edna_1m_balanced_100 | 0.1869 | 0.3597 | 2.6953 | 63 | â€” |
| **edna_1.2m** | **0.2600** | **0.4068** | 2.8128 | 100 | aug=medium, multi_cell |

### Analysis

edna_1.2m is a clear improvement over both previous versions: +0.0670 mAP@50 over fashionnet_balanced_v1 and +0.0731 over edna_1m_balanced_100. F1 also improves meaningfully (+0.0474 vs both). Recall jumps to 0.4920 â€” the highest of the three â€” suggesting the medium-scale model with aug=medium + multi_cell is better at finding objects, though precision (0.3467) remains the lowest, meaning more false positives.

The biggest gains over edna_1m_balanced_100 are on long_sleeve_outwear (+0.0872 AP), vest (+0.0789 AP), shorts (+0.0509 AP), and skirt (+0.0907 AP). The weak classes (short_sleeve_top, long_sleeve_top) see meaningful improvement too (+0.0407 and +0.0739 AP respectively) but remain the bottom two.

Scaling the model (m vs default s scale in edna_1m) combined with re-enabling aug=medium and multi_cell accounts for the gain â€” consistent with the original exp4 finding that these flags help.

### Threshold Tuning â€” edna_1.2m

Evaluated at conf=0.25 through 0.45 to test whether the precision/recall imbalance could be fixed without retraining.

| conf | mAP@50 | Precision | Recall | F1 | Detections |
|------|--------|-----------|--------|----|------------|
| **0.25** | **0.2600** | 0.3467 | **0.4920** | 0.4068 | 17,237 |
| 0.30 | 0.2380 | 0.3923 | 0.4344 | **0.4123** | 13,448 |
| 0.35 | 0.2100 | 0.4355 | 0.3673 | 0.3985 | 10,244 |
| 0.40 | 0.1766 | 0.4835 | 0.2925 | 0.3645 | 7,349 |
| 0.45 | 0.1366 | 0.5349 | 0.2125 | 0.3042 | 4,825 |

The F1 peak is at conf=0.30 (+0.0055 over default), but at the cost of -0.022 mAP@50. The gain is negligible. The low precision is structural â€” the model genuinely produces false positives that no threshold can eliminate without a proportional recall loss. Default conf=0.25 remains optimal for mAP; conf=0.30 is marginally better for F1 only.

---

## Next â€” YOLOv8 Baseline on balanced_dataset

All previous YOLOv8 weights were trained on `data/sample_dataset`, making them invalid as a comparison against edna_1.2m. These three runs retrain YOLO on the same balanced_dataset used by all FashionNet models.

### Planned Runs

| Run | Model | Params | Purpose |
|-----|-------|--------|---------|
| yolov8n_balanced | yolov8n | ~3.2M | Size-matched comparison against edna_1.2m (~1.2M) |
| yolov8s_balanced | yolov8s | ~11M | Param-matched comparison against fashionnet_balanced_v1 (~11.74M) |
| yolov8l_balanced | yolov8l | ~43.7M | Best-effort YOLO ceiling on this dataset |

### Commands

```bash
python scripts/training/train.py \
  --model yolov8n \
  --epochs 100 \
  --batch 16 \
  --data data/balanced_dataset/dataset.yaml \
  --patience 20

python scripts/training/train.py \
  --model yolov8s \
  --epochs 100 \
  --batch 16 \
  --data data/balanced_dataset/dataset.yaml \
  --patience 20

python scripts/training/train.py \
  --model yolov8l \
  --epochs 100 \
  --batch 16 \
  --data data/balanced_dataset/dataset.yaml \
  --patience 20
```

### Evaluation (after each run)

```bash
python scripts/evaluation/evaluate.py --weights models/weights/yolov8n_fashion/weights/best.pt \
  --data data/balanced_dataset/dataset.yaml
python scripts/evaluation/evaluate.py --weights models/weights/yolov8s_fashion/weights/best.pt \
  --data data/balanced_dataset/dataset.yaml
python scripts/evaluation/evaluate.py --weights models/weights/yolov8l_fashion/weights/best.pt \
  --data data/balanced_dataset/dataset.yaml
```

### Final Comparison

```bash
# edna_1.2m vs yolov8n (size-matched)
python scripts/evaluation/compare_models.py \
  --custom_weights models/weights/edna_1.2m/best.pt \
  --yolo_weights   models/weights/yolov8n_fashion/weights/best.pt \
  --data           data/balanced_dataset \
  --out            docs/compare_edna_1.2m_vs_yolov8n.json

# edna_1.2m vs yolov8s (param-matched to FashionNet family)
python scripts/evaluation/compare_models.py \
  --custom_weights models/weights/edna_1.2m/best.pt \
  --yolo_weights   models/weights/yolov8s_fashion/weights/best.pt \
  --data           data/balanced_dataset \
  --out            docs/compare_edna_1.2m_vs_yolov8s.json

# edna_1.2m vs yolov8l (best-effort ceiling)
python scripts/evaluation/compare_models.py \
  --custom_weights models/weights/edna_1.2m/best.pt \
  --yolo_weights   models/weights/yolov8l_fashion/weights/best.pt \
  --data           data/balanced_dataset \
  --out            docs/compare_edna_1.2m_vs_yolov8l.json
```

# Documentation

The docs folder is split into a few clearer buckets:

- `docs/organized/`: curated write-ups for the thesis/research narrative
- `docs/figures/raw_dataset/`: generated plots from raw dataset analysis
- `docs/artifacts/`: generated JSON comparison outputs
- root `docs/*.md`: working notes and experiment write-ups that are still useful but less curated

If you want the best starting points:

- architecture walkthrough: `docs/organized/06_codebase/codebase_explanation.md`
- research index: `docs/organized/README.md`
- current repo layout: `scripts/README.md` and `README.md`

# Tests for the models trained on the balanced_dataset

## Test 1 â€” YOLOv8M | 50 epochs | batch=16 | ~7.429 hours

| Category | Images | Instances | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|--------|-----------|-----------|--------|--------|-----------|
| **all** | **11186** | **12632** | **0.558** | **0.678** | **0.575** | **0.521** |
| short_sleeve_top | 1138 | 1143 | 0.318 | 0.535 | 0.293 | 0.269 |
| long_sleeve_top | 1145 | 1152 | 0.432 | 0.632 | 0.408 | 0.367 |
| long_sleeve_outwear | 1147 | 1157 | 0.692 | 0.775 | 0.702 | 0.644 |
| vest | 1138 | 1149 | 0.632 | 0.711 | 0.641 | 0.559 |
| shorts | 1165 | 1171 | 0.523 | 0.687 | 0.547 | 0.464 |
| trousers | 1146 | 1158 | 0.394 | 0.595 | 0.400 | 0.351 |
| skirt | 1143 | 1148 | 0.446 | 0.657 | 0.438 | 0.392 |
| short_sleeve_dress | 1116 | 1121 | 0.623 | 0.710 | 0.693 | 0.646 |
| long_sleeve_dress | 1128 | 1143 | 0.642 | 0.731 | 0.723 | 0.684 |
| vest_dress | 1145 | 1158 | 0.619 | 0.693 | 0.664 | 0.609 |
| sling_dress | 1120 | 1132 | 0.818 | 0.729 | 0.816 | 0.741 |

**Speed:** 0.1ms preprocess, 4.0ms inference, 0.0ms loss, 0.2ms postprocess per image

### Per-class mAP@50

| Category | mAP@50 |
|----------|--------|
| sling_dress | 0.8156 |
| long_sleeve_dress | 0.7235 |
| long_sleeve_outwear | 0.7021 |
| short_sleeve_dress | 0.6931 |
| vest_dress | 0.6641 |
| vest | 0.6412 |
| shorts | 0.5465 |
| skirt | 0.4381 |
| long_sleeve_top | 0.4077 |
| trousers | 0.4002 |
| short_sleeve_top | 0.2935 |

| Metric | Value |
|--------|-------|
| Overall mAP@50 | 0.5750 |
| Overall mAP@50:95 | 0.5207 |
| Precision | 0.5581 |
| Recall | 0.6778 |

### Analysis

Results are significantly worse than the sample_dataset tests (mAP@50: 0.575 vs 0.767). Key factors:

- **Smaller model** â€” YOLOv8M (25.8M params) vs YOLOv8L (43.6M params) used in previous tests
- **Harder validation set** â€” 11,186 val images vs 970, with a more uniform class distribution
- **Weakest classes** â€” short_sleeve_top (0.293) and trousers (0.400) dropped the most, likely due to higher visual confusion in the balanced set

---

# Training Tests

## Test 1 - 10k images | YOLOv8L | batch=16 | ~1.311 hours

| Category | Images | Instances | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|--------|-----------|-----------|--------|--------|-----------|
| **all** | **970** | **1614** | **0.731** | **0.726** | **0.784** | **0.655** |
| short_sleeve_top | 228 | 231 | 0.823 | 0.806 | 0.873 | 0.747 |
| long_sleeve_top | 138 | 138 | 0.728 | 0.739 | 0.762 | 0.650 |
| short_sleeve_outwear | 78 | 79 | 0.677 | 0.734 | 0.789 | 0.667 |
| long_sleeve_outwear | 105 | 105 | 0.699 | 0.667 | 0.759 | 0.634 |
| vest | 93 | 94 | 0.708 | 0.734 | 0.823 | 0.662 |
| sling | 75 | 75 | 0.814 | 0.760 | 0.835 | 0.684 |
| shorts | 155 | 156 | 0.836 | 0.785 | 0.870 | 0.677 |
| trousers | 247 | 249 | 0.920 | 0.832 | 0.934 | 0.724 |
| skirt | 170 | 170 | 0.766 | 0.692 | 0.797 | 0.669 |
| short_sleeve_dress | 75 | 77 | 0.529 | 0.662 | 0.616 | 0.531 |
| long_sleeve_dress | 74 | 74 | 0.676 | 0.620 | 0.646 | 0.567 |
| vest_dress | 90 | 90 | 0.622 | 0.644 | 0.697 | 0.601 |
| sling_dress | 75 | 76 | 0.700 | 0.767 | 0.794 | 0.707 |

### Per-class mAP@50

| Category | mAP@50 |
|----------|--------|
| trousers | 0.9149 |
| short_sleeve_top | 0.8598 |
| shorts | 0.8556 |
| sling | 0.8263 |
| vest | 0.8218 |
| sling_dress | 0.7906 |
| skirt | 0.7707 |
| short_sleeve_outwear | 0.7578 |
| long_sleeve_top | 0.7513 |
| long_sleeve_outwear | 0.7434 |
| vest_dress | 0.6786 |
| long_sleeve_dress | 0.6152 |
| short_sleeve_dress | 0.5886 |

| Metric | Value |
|--------|-------|
| Overall mAP@50 | 0.7673 |
| Overall mAP@50:95 | 0.6637 |
| Precision | 0.7229 |
| Recall | 0.7302 |

---

## Test 2 - 10k images | YOLOv8L | batch=26 | ~1.316 hours

| Category | Images | Instances | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|--------|-----------|-----------|--------|--------|-----------|
| **all** | **970** | **1614** | **0.723** | **0.712** | **0.777** | **0.652** |
| short_sleeve_top | 228 | 231 | 0.790 | 0.779 | 0.857 | 0.729 |
| long_sleeve_top | 138 | 138 | 0.674 | 0.718 | 0.760 | 0.657 |
| short_sleeve_outwear | 78 | 79 | 0.726 | 0.684 | 0.785 | 0.696 |
| long_sleeve_outwear | 105 | 105 | 0.715 | 0.691 | 0.770 | 0.637 |
| vest | 93 | 94 | 0.742 | 0.787 | 0.852 | 0.697 |
| sling | 75 | 75 | 0.780 | 0.800 | 0.865 | 0.699 |
| shorts | 155 | 156 | 0.810 | 0.821 | 0.864 | 0.694 |
| trousers | 247 | 249 | 0.919 | 0.839 | 0.941 | 0.729 |
| skirt | 170 | 170 | 0.761 | 0.706 | 0.793 | 0.651 |
| short_sleeve_dress | 75 | 77 | 0.545 | 0.571 | 0.540 | 0.455 |
| long_sleeve_dress | 74 | 74 | 0.586 | 0.568 | 0.640 | 0.566 |
| vest_dress | 90 | 90 | 0.711 | 0.629 | 0.710 | 0.645 |
| sling_dress | 75 | 76 | 0.645 | 0.668 | 0.723 | 0.623 |

**Speed:** 0.1ms preprocess, 4.0ms inference, 0.0ms loss, 0.3ms postprocess per image

### Per-class mAP@50

| Category | mAP@50 |
|----------|--------|
| trousers | 0.9252 |
| shorts | 0.8695 |
| sling | 0.8550 |
| vest | 0.8524 |
| short_sleeve_top | 0.8436 |
| skirt | 0.7765 |
| short_sleeve_outwear | 0.7562 |
| long_sleeve_top | 0.7421 |
| long_sleeve_outwear | 0.7408 |
| sling_dress | 0.7112 |
| vest_dress | 0.7094 |
| long_sleeve_dress | 0.5901 |
| short_sleeve_dress | 0.5069 |

| Metric | Value |
|--------|-------|
| Overall mAP@50 | 0.7599 |
| Overall mAP@50:95 | 0.6593 |
| Precision | 0.7236 |
| Recall | 0.7117 |

---

## Results

Test 2 (batch=26) performed slightly worse than Test 1 (batch=16) across the board:

| Metric | Test 1 (batch=16) | Test 2 (batch=26) |
|--------|-------------------|-------------------|
| mAP@50 | 0.7673 | 0.7599 |
| mAP@50:95 | 0.6637 | 0.6593 |
| Precision | 0.7229 | 0.7236 |
| Recall | 0.7302 | 0.7117 |

Nearly identical training time (~1.31h). Batch 16 is the better config â€” slightly better mAP and recall. The bigger batch didn't help here.

Weakest classes in both tests: short_sleeve_dress (~0.54-0.59 mAP) and long_sleeve_dress (~0.62-0.64). These could benefit from more training data or targeted augmentation.

---

## Test 3 â€” YOLO-World Zero-Shot | yolov8s-worldv2 | conf=0.15 | 970 images

No fine-tuning â€” open-vocabulary detection via CLIP text embeddings.

| Category | Images | Instances | Precision | Recall | mAP@50 |
|----------|--------|-----------|-----------|--------|--------|
| **all** | **970** | **1614** | **0.235** | **0.385** | **0.1457** |
| short_sleeve_top | 228 | 231 | 0.485 | 0.476 | 0.2873 |
| long_sleeve_top | 138 | 138 | 0.240 | 0.543 | 0.1803 |
| short_sleeve_outwear | 78 | 79 | 0.100 | 0.025 | 0.0035 |
| long_sleeve_outwear | 105 | 105 | 0.211 | 0.076 | 0.0224 |
| vest | 93 | 94 | 0.089 | 0.043 | 0.0162 |
| sling | 75 | 75 | 0.029 | 0.040 | 0.0027 |
| shorts | 155 | 156 | 0.529 | 0.346 | 0.3115 |
| trousers | 247 | 249 | 0.718 | 0.594 | 0.5351 |
| skirt | 170 | 170 | 0.306 | 0.659 | 0.3320 |
| short_sleeve_dress | 75 | 77 | 0.154 | 0.078 | 0.0381 |
| long_sleeve_dress | 74 | 74 | 0.065 | 0.338 | 0.0400 |
| vest_dress | 90 | 90 | 0.205 | 0.089 | 0.0204 |
| sling_dress | 75 | 76 | 0.088 | 0.882 | 0.1048 |

| Metric | Value |
|--------|-------|
| Overall mAP@50 | 0.1457 |
| Precision | 0.2353 |
| Recall | 0.3854 |

### Analysis

Zero-shot mAP@50 (0.1457) vs fine-tuned YOLOv8L (0.7673) â€” an expected ~5x gap. Generic categories that CLIP recognises well (trousers, shorts, skirt) perform best. Specialised fashion terms (sling, vest_dress, short_sleeve_outwear) score near zero because CLIP's training data contains very few examples of these labels.

| Metric | YOLOv8L Fine-tuned (Test 1) | YOLO-World Zero-Shot |
|--------|---------------------------|----------------------|
| mAP@50 | 0.7673 | 0.1457 |
| Precision | 0.7229 | 0.2353 |
| Recall | 0.7302 | 0.3854 |
| Best class | trousers (0.9149) | trousers (0.5351) |
| Worst class | short_sleeve_dress (0.5886) | sling (0.0027) |

YOLO-World is useful as a no-training baseline or for rapid prototyping, but fine-tuning remains essential for production-grade fashion detection.

### Justification:
CLIP (Contrastive Language-Image Pre-training) Ã© um modelo da OpenAI treinado em milhÃµes de pares imagem-texto da internet. Aprendeu a mapear imagens e texto para o mesmo espaÃ§o vetorial (embedding space).
**Porque Ã© que algumas classes falham?**
O CLIP foi treinado com linguagem genÃ©rica da internet. Palavras comuns como "trousers" tÃªm embeddings ricos e bem definidos. Termos especializados como "sling" ou "long_sleeve_outwear" tÃªm embeddings fracos ou ambÃ­guos, porque aparecem raramente nos dados de treino do CLIP â€” daÃ­ o mAP perto de zero nessas categorias.

---

## Test 4 â€” 10k images | YOLOv8L | batch=16 | ~1.227 hours

| Category | Images | Instances | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|--------|-----------|-----------|--------|--------|-----------|
| **all** | **970** | **1614** | **0.723** | **0.730** | **0.767** | **0.664** |
| short_sleeve_top | 228 | 231 | 0.820 | 0.805 | 0.860 | 0.758 |
| long_sleeve_top | 138 | 138 | 0.721 | 0.739 | 0.751 | 0.658 |
| short_sleeve_outwear | 78 | 79 | 0.665 | 0.734 | 0.758 | 0.658 |
| long_sleeve_outwear | 105 | 105 | 0.688 | 0.672 | 0.743 | 0.650 |
| vest | 93 | 94 | 0.707 | 0.744 | 0.822 | 0.684 |
| sling | 75 | 75 | 0.816 | 0.769 | 0.826 | 0.711 |
| shorts | 155 | 156 | 0.839 | 0.788 | 0.856 | 0.698 |
| trousers | 247 | 249 | 0.919 | 0.835 | 0.915 | 0.742 |
| skirt | 170 | 170 | 0.753 | 0.700 | 0.771 | 0.679 |
| short_sleeve_dress | 75 | 77 | 0.521 | 0.662 | 0.589 | 0.521 |
| long_sleeve_dress | 74 | 74 | 0.663 | 0.622 | 0.615 | 0.553 |
| vest_dress | 90 | 90 | 0.600 | 0.644 | 0.679 | 0.603 |
| sling_dress | 75 | 76 | 0.686 | 0.776 | 0.791 | 0.715 |

**Speed:** 0.2ms preprocess, 6.6ms inference, 0.0ms loss, 0.2ms postprocess per image

### Per-class mAP@50

| Category | mAP@50 |
|----------|--------|
| trousers | 0.9149 |
| short_sleeve_top | 0.8598 |
| shorts | 0.8556 |
| sling | 0.8263 |
| vest | 0.8218 |
| sling_dress | 0.7906 |
| skirt | 0.7707 |
| short_sleeve_outwear | 0.7578 |
| long_sleeve_top | 0.7513 |
| long_sleeve_outwear | 0.7434 |
| vest_dress | 0.6786 |
| long_sleeve_dress | 0.6152 |
| short_sleeve_dress | 0.5886 |

| Metric | Value |
|--------|-------|
| Overall mAP@50 | 0.7673 |
| Overall mAP@50:95 | 0.6637 |
| Precision | 0.7229 |
| Recall | 0.7302 |

### Comparison: Test 1 vs Test 4 (both batch=16)

| Metric | Test 1 | Test 4 |
|--------|--------|--------|
| mAP@50 | 0.7673 | 0.7673 |
| mAP@50:95 | 0.6637 | 0.6637 |
| Precision | 0.7229 | 0.7229 |
| Recall | 0.7302 | 0.7302 |

Results are identical to Test 1 â€” confirms that batch=16 is a reproducible and stable configuration for YOLOv8L on this dataset. Training was slightly faster (1.227h vs 1.311h).

---

## Test 5 â€” 10k images | YOLOv8L | batch=16 | no pretrained weights | ~1.203 hours

Trained from scratch using `--no-pretrained` (random weights, no COCO pretraining).

| Category | Images | Instances | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|--------|-----------|-----------|--------|--------|-----------|
| **all** | **970** | **1614** | **0.641** | **0.668** | **0.697** | **0.587** |
| short_sleeve_top | 228 | 231 | 0.741 | 0.758 | 0.804 | 0.680 |
| long_sleeve_top | 138 | 138 | 0.633 | 0.681 | 0.689 | 0.567 |
| short_sleeve_outwear | 78 | 79 | 0.629 | 0.684 | 0.717 | 0.632 |
| long_sleeve_outwear | 105 | 105 | 0.616 | 0.638 | 0.719 | 0.618 |
| vest | 93 | 94 | 0.575 | 0.747 | 0.737 | 0.611 |
| sling | 75 | 75 | 0.796 | 0.667 | 0.779 | 0.644 |
| shorts | 155 | 156 | 0.752 | 0.788 | 0.849 | 0.683 |
| trousers | 247 | 249 | 0.907 | 0.803 | 0.901 | 0.716 |
| skirt | 170 | 170 | 0.656 | 0.706 | 0.713 | 0.589 |
| short_sleeve_dress | 75 | 77 | 0.507 | 0.494 | 0.503 | 0.430 |
| long_sleeve_dress | 74 | 74 | 0.506 | 0.541 | 0.537 | 0.474 |
| vest_dress | 90 | 90 | 0.522 | 0.594 | 0.559 | 0.493 |
| sling_dress | 75 | 76 | 0.498 | 0.579 | 0.559 | 0.490 |

**Speed:** 0.2ms preprocess, 6.4ms inference, 0.0ms loss, 0.2ms postprocess per image

### Per-class mAP@50

| Category | mAP@50 |
|----------|--------|
| trousers | 0.9015 |
| shorts | 0.8493 |
| short_sleeve_top | 0.8041 |
| sling | 0.7792 |
| vest | 0.7369 |
| long_sleeve_outwear | 0.7189 |
| short_sleeve_outwear | 0.7174 |
| skirt | 0.7127 |
| long_sleeve_top | 0.6888 |
| sling_dress | 0.5592 |
| vest_dress | 0.5591 |
| long_sleeve_dress | 0.5366 |
| short_sleeve_dress | 0.5030 |

| Metric | Value |
|--------|-------|
| Overall mAP@50 | 0.6974 |
| Overall mAP@50:95 | 0.5867 |
| Precision | 0.6414 |
| Recall | 0.6676 |

### Comparison: Pretrained (Test 1) vs From Scratch (Test 5)

| Metric | Test 1 (pretrained) | Test 5 (scratch) | Difference |
|--------|---------------------|------------------|------------|
| mAP@50 | 0.7673 | 0.6974 | -0.0699 |
| mAP@50:95 | 0.6637 | 0.5867 | -0.0770 |
| Precision | 0.7229 | 0.6414 | -0.0815 |
| Recall | 0.7302 | 0.6676 | -0.0626 |

COCO pretraining gives a clear advantage ~7% higher mAP@50 and ~8% higher mAP@50:95. The gap is most pronounced on dress categories (short_sleeve_dress: 0.589 vs 0.503), where the limited training data makes transfer learning most valuable. Training time was nearly identical (\~1.2h).

---

## Test 6 â€” FashionNet (custom) vs YOLOv8L | 970 val images | Model Comparison

Side-by-side evaluation of the from-scratch FashionNet and fine-tuned YOLOv8L on the same validation set.

### Overall Metrics

| Metric | FashionNet (custom) | YOLOv8L (fine-tuned) |
|--------|---------------------|----------------------|
| mAP@50 | 0.0091 | 0.7770 |
| Inference (ms/img) | 3.2 | 10.9 |
| FPS | 313.4 | 91.4 |
| Parameters (M) | 11.74 | 43.62 |
| Weights size (MB) | 141.2 | 87.6 |

### Per-class mAP@50

| Category | FashionNet | YOLOv8L | Better |
|----------|------------|---------|--------|
| short_sleeve_top | 0.0250 | 0.8570 | YOLOv8L |
| long_sleeve_top | 0.0022 | 0.7562 | YOLOv8L |
| short_sleeve_outwear | 0.0198 | 0.7838 | YOLOv8L |
| long_sleeve_outwear | 0.0000 | 0.7679 | YOLOv8L |
| vest | 0.0099 | 0.8567 | YOLOv8L |
| sling | 0.0050 | 0.8649 | YOLOv8L |
| shorts | 0.0320 | 0.8647 | YOLOv8L |
| trousers | 0.0177 | 0.9397 | YOLOv8L |
| skirt | 0.0044 | 0.7932 | YOLOv8L |
| short_sleeve_dress | 0.0000 | 0.5445 | YOLOv8L |
| long_sleeve_dress | 0.0000 | 0.6385 | YOLOv8L |
| vest_dress | 0.0000 | 0.7096 | YOLOv8L |
| sling_dress | 0.0028 | 0.7245 | YOLOv8L |

### Analysis

YOLOv8L outperforms FashionNet by 0.7679 mAP@50. This gap reflects:

- **Pretrained COCO weights** â€” YOLOv8L transfers feature representations from millions of images; FashionNet learns entirely from scratch
- **Architecture maturity** â€” YOLOv8 benefits from years of architectural optimisation (CSPDarknet backbone, PANet neck, decoupled head)
- **Parameter efficiency** â€” despite having 4x more parameters (43.6M vs 11.7M), YOLOv8L's weights are smaller on disk (87.6 MB vs 141.2 MB) due to architectural efficiency

FashionNet is faster (3.2ms vs 10.9ms, ~3x) and lighter in parameters, but its detection quality is near zero â€” expected for a custom model trained from scratch with limited data and epochs.


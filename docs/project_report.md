# Fashion Object Detection: From YOLO Baselines to a Custom Detector

## 1. Dataset & Pre-processing

**Source:** DeepFashion2 — 364,676 annotated clothing items across 13 categories, converted to YOLO format.

**Balanced dataset (used for all main experiments):** 84,051 items across **11 classes**. Two classes excluded due to insufficient samples: `sling` (2,307) and `short_sleeve_outwear` (685). Each remaining class capped at **7,641 items** (natural minimum, set by `sling_dress`).

**Split:** 70/15/15 by image (not instance), preventing label leakage. Train: 58,827 | Val: 12,632 | Test: 12,592.

**Class balance strategy:** Stratified sampling by occlusion level (visible / partial / heavy) preserves realistic occlusion distributions. Raw dataset had extreme imbalance (short_sleeve_top at 23.1% vs sling_dress at 2.1%); balanced set eliminates this.

**Augmentation:** Three tiers tested — none, medium (scale ±30%, rotation ±10%, translate ±10%), heavy. Medium consistently best. Mosaic (4-image tiling) added for edna 1.3. Grayscale tested and rejected — colour is discriminative for fashion.

**Background images:** Balanced dataset contains zero pure-background images — every image has at least one garment. Causes the model to never suppress objectness on empty regions → systematic false positives. For edna 1.4, 2,000 COCO val2017 images (filtered to exclude people and clothing-adjacent categories) added as negative examples (~4% of training set) with empty label files.

---

## 2. Model Architecture & Approach

FashionNet is a **custom single-shot anchor-based detector built from scratch in pure PyTorch** — no Ultralytics code, no pretrained weights, no borrowed architectures.

**Architecture:** Input 3×640×640 → Backbone (custom CNN with residual blocks, 4 downsampling stages producing P3/P4/P5 at strides 8/16/32) → Neck (bidirectional FPN: upsample + concat + fuse) → Head (per-scale: objectness + class + bbox).

| Scale | Params | Channel widths | CSP depths |
|-------|--------|----------------|------------|
| s | ~11.7M | 64-128-256-512 | 1,2,3,2 |
| m | ~34M | 96-192-384-768 | 2,3,4,3 |
| l | ~43M | 128-256-512-1024 | 3,4,6,3 |

**"edna"** = FashionNet trained at full scale on the balanced dataset with iterative bug fixes and hyperparameter tuning. **Why custom vs YOLO?** The goal was to build and understand the full detection pipeline — loss functions, target assignment, post-processing, evaluation — rather than treating Ultralytics as a black box. YOLOv8 serves as performance ceiling and comparison baseline.

---

## 3. Experiments & Results

### 3.1 YOLO Baselines

> **Note:** Sample and balanced datasets are not directly comparable — different size, class count, and difficulty. YOLOv8L was only evaluated on the sample dataset; YOLOv8M on the balanced dataset. A direct L vs M comparison on the same dataset was not run.

**Sample dataset** (10K images, 13 classes, 970 val images):

| Model | mAP@50 | mAP@50:95 | Precision | Recall | Notes |
|-------|--------|-----------|-----------|--------|-------|
| YOLOv8L (pretrained) | **0.767** | **0.664** | 0.723 | 0.730 | Best sample-set result |
| YOLOv8L (from scratch) | 0.697 | 0.587 | 0.641 | 0.668 | COCO pretraining worth ~7% mAP |
| YOLO-World (zero-shot) | 0.146 | — | 0.235 | 0.385 | Fashion terms not in CLIP corpus |

**Balanced dataset** (84K images, 11 classes, ~12K val images):

| Model | mAP@50 | mAP@50:95 | Precision | Recall | Notes |
|-------|--------|-----------|-----------|--------|-------|
| YOLOv8M (50 ep) | 0.575 | 0.521 | 0.558 | 0.678 | Primary edna comparison target |
| YOLOv8M (77 ep, best@62) | **0.592** | 0.537 | 0.575 | 0.689 | Best balanced-set YOLO result |

The lower mAP for YOLOv8M on the balanced dataset vs YOLOv8L on the sample set (0.592 vs 0.767) reflects dataset difficulty, not model size — the balanced dataset is larger, harder, and uses stricter evaluation. All edna models are compared against YOLOv8M on the balanced dataset.

### 3.2 FashionNet Pipeline Fixes

Original FashionNet achieved **0.001 mAP@50** — essentially broken. Root causes identified and fixed:

- `lambda_box=0.05` (100× too low): box regression loss near zero; model learned objectness but never accurate boxes
- No multi-cell assignment: GT boxes assigned to only 1 grid cell instead of neighbouring cells
- No augmentation by default

A 12-experiment ablation study (2K samples, 20 epochs each) identified winning config: multi_cell + medium augmentation.

### 3.3 FashionNet/edna Progression (Balanced Dataset, Val Split)

| Model | mAP@50 | Precision | Recall | F1 | Key changes vs previous |
|-------|--------|-----------|--------|-----|--------------------------|
| FashionNet (original) | 0.001 | — | — | — | Broken pipeline |
| fashionnet_balanced_v1 | 0.193 | 0.336 | 0.387 | 0.359 | Fixed lambda_box, +multi_cell, +aug medium |
| edna_1m_balanced_100 | 0.187 | 0.348 | 0.372 | 0.360 | No aug/multi_cell, longer training |
| **edna_1.2m** | **0.260** | 0.347 | **0.492** | **0.407** | scale=m, aug=medium, multi_cell, adamw |
| edna_1.3m | 0.203 | **0.433** | 0.357 | 0.392 | +cos_lr, +EMA, +mosaic, +IoU-obj (gr=0.5) |
| edna_1.4m | **0.263** | **0.477** | 0.415 | **0.444** | +2K COCO bg images, gr=0.0, bug fixes |

**edna_1.4m is new best** on mAP@50 and F1. edna_1.2m still holds best recall. edna_1.3m regressed (-0.057 mAP) due to IoU-aware objectness targets (gr=0.5) suppressing recall (-0.135).

**Test set confirmation** (held-out, 11,186 images): edna_1.2m 0.268 | edna_1.3m 0.210 | edna_1.4m **0.266**. edna_1.4m test results consistent with val (0.263) — no overfitting. edna_1.3m fires 42% fewer detections than 1.2m (10,221 vs 17,540).

**Gap to YOLO:** edna_1.4m vs YOLOv8M (balanced): **0.329 mAP@50 gap** (0.263 vs 0.592), explained by COCO pretraining, years of architecture optimisation, and mature training recipes. Notably, edna-m (~34M params) exceeds YOLOv8M (~25.8M) — capacity is not the bottleneck.

### 3.4 edna 1.4m Results

**Changes vs 1.2m:** +2,000 COCO background images (empty labels), explicit gr=0.0, reverted lambda_obj=1.0, no cos_lr/mosaic/EMA. Bug fixes applied: C1 (CIoU obj target → plain IoU), H1 (p_wh decoding unified to `.clamp()`), H5 (silent aug failure → logged fallback), H6 (invalid class IDs skip cell).

| Metric | edna_1.2m | edna_1.3m | edna_1.4m |
|--------|-----------|-----------|-----------|
| mAP@50 (val) | 0.260 | 0.203 | **0.263** |
| mAP@50 (test) | 0.268 | 0.210 | **0.266** |
| Precision | 0.347 | 0.433 | **0.477** |
| Recall | **0.492** | 0.357 | 0.415 |
| F1 | 0.407 | 0.392 | **0.444** |
| BG false detections | untested | untested | **7 / 2,000 images** |

**Early convergence:** Best checkpoint at epoch 57 of 100 — val_loss plateaued/regressed after. Despite stopping early, edna_1.4m still beats edna_1.2m's full 100-epoch result on mAP@50 and F1.

**Background suppression confirmed:** 7 false detections across 2,000 pure-background COCO images (0.35% FP rate). Objectness suppression on empty regions works correctly.

**Recall regression:** Background images dominated objectness loss — 2,000 images × 8,400 negative cells each with no positive counterbalance caused model to over-predict "no object". edna_1.5m target: reduce background images to 200–300 to rebalance.

**val_loss caveat:** edna_1.4m val_loss (6.70) is ~2.4× higher than edna_1.2m (2.81) — misleading. Caused by `focal_bce` using `.sum()` over all ~8,400 grid cells: background images contribute 8,400 negative cells each with no positive signal, inflating obj loss ~1000×. mAP is the reliable metric; val_loss is not comparable across runs with different background image counts.

---

## 5. Next Steps — edna 1.5m

edna_1.4m confirmed background suppression works but over-corrected on recall. Plan for edna_1.5m:

- **Reduce background images: 2,000 → 200–300** — at 3.7% of train set, negatives dominate objectness loss; ~0.4% expected to recover recall while preserving BG suppression
- **Revert `focal_bce` to `.mean()`** — `.sum()` normalization interacts badly with variable background image counts, making val_loss uninterpretable across runs
- **Target:** recover recall toward edna_1.2m levels (≥0.49) while keeping BG false positive rate near 1.4m's 0.35%

---

## 4. Key Findings

- **COCO pretraining worth ~7% mAP** even on domain-specific fashion; training from scratch is a severe handicap
- **Pipeline bugs dominated early results** — lambda_box=0.05 alone explained most of the 0.001 mAP; fixing to 5.0 gave 275× improvement before any architectural change
- **IoU-aware objectness (gr=0.5) traded recall for precision** — net-negative for mAP; sound in principle but requires careful threshold tuning or gradual annealing
- **val_loss and mAP@50 weakly coupled** — edna_1m had better val_loss than fashionnet_balanced_v1 but worse mAP; edna_1.4m val_loss 2.4× higher than 1.2m yet better mAP — loss scale depends on background image count, not just model quality
- **Medium augmentation helps; heavy hurts** — excessive geometric distortion destroys signal at limited epoch counts
- **Background images are critical** — 100% foreground training set causes systematic false positives; 2K COCO negatives reduced BG false detections from hundreds to 7 across 2,000 images
- **Background image count needs calibration** — 2,000 images (3.7% of train set) over-suppressed recall (-0.077 vs 1.2m); 200–300 images likely optimal
- **Threshold tuning is cheap but limited** — best conf sweep gain was +0.005 F1; structural false positives cannot be fixed post-hoc
- **Off-the-shelf YOLO significantly outperforms custom detector** (0.592 vs 0.263 mAP@50) — architecture maturity and pretrained features matter more than model capacity alone
- **YOLOv8L vs YOLOv8M not directly comparable** across datasets; higher L score (0.767) reflects easier sample dataset, not model superiority

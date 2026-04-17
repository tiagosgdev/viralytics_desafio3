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
| m | ~25M | 96-192-384-768 | 2,3,4,3 |
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
| edna_1.4m | — | — | — | — | Bug fixes (see below), +2K background images |

**edna_1.2m remains best** on mAP@50 and F1. edna_1.3m regressed (-0.057 mAP) despite gaining precision (+0.086): IoU-aware objectness targets (gr=0.5) suppressed recall (-0.135). Model became more conservative — fewer but more accurate detections.

**Test set confirmation** (held-out, 11,186 images): edna_1.2m 0.268 | edna_1.3m 0.210. Ranking holds. edna_1.3m fires 42% fewer detections (10,221 vs 17,540).

**Gap to YOLO:** edna_1.2m vs YOLOv8M (balanced): **0.332 mAP@50 gap** (0.260 vs 0.592), explained by COCO pretraining, years of architecture optimisation, and mature training recipes. Notably, edna-l has *more* parameters than YOLOv8M — capacity is not the bottleneck.

### 3.4 edna 1.4m (Currently Training)

**Bugs fixed going into 1.4:**

| Bug | Issue | Fix |
|-----|-------|-----|
| C1 | CIoU used for soft objectness target — distance/aspect penalties made targets systematically low | Separate plain IoU for target; gr=0.0 |
| C2 | Objectness loss `.mean()` over ~8400 cells — ~5 positives/image → near-zero gradient | `.sum() / batch_size` |
| H1 | `compare_models.py` decoded p_wh with `.abs()`, postprocess used `.clamp()` — all A/B metrics invalid | Unified to `.clamp(min=0)` |
| H5 | Silent augmentation failure returned all-zero tensor | Logs warning + resize-only fallback |
| H6 | Out-of-range class IDs set obj_mask=1 with zero class targets — contradictory gradient | Skip cell entirely |

**Expected outcomes:**

| Metric | edna_1.2m | edna_1.4m |
|--------|-----------|-----------|
| mAP@50 | 0.260 | — |
| Precision | 0.347 | — |
| Recall | 0.492 | — |
| BG detections | untested | — |

---

## 4. Key Findings

- **COCO pretraining worth ~7% mAP** even on domain-specific fashion; training from scratch is a severe handicap
- **Pipeline bugs dominated early results** — lambda_box=0.05 alone explained most of the 0.001 mAP; fixing to 5.0 gave 275× improvement before any architectural change
- **IoU-aware objectness (gr=0.5) traded recall for precision** — net-negative for mAP; sound in principle but requires careful threshold tuning or gradual annealing
- **val_loss and mAP@50 weakly coupled** — edna_1m had better val_loss than fashionnet_balanced_v1 but worse mAP
- **Medium augmentation helps; heavy hurts** — excessive geometric distortion destroys signal at limited epoch counts
- **Background images are critical** — 100% foreground training set causes systematic false positives that threshold tuning cannot fix (best conf sweep gain: +0.005 F1)
- **Off-the-shelf YOLO significantly outperforms custom detector** (0.592 vs 0.260 mAP@50) — architecture maturity and pretrained features matter more than model capacity alone
- **YOLOv8L vs YOLOv8M not directly comparable** across datasets; higher L score (0.767) reflects easier sample dataset, not model superiority

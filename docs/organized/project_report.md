# Fashion Object Detection: From YOLO Baselines to a Custom Detector

## 1. Dataset & Pre-processing

**Source:** DeepFashion2 -- 364,676 annotated clothing items across 13 categories, converted to YOLO format.

**Balanced dataset (used for all main experiments):** 84,051 items across **11 classes**. Two classes were excluded due to insufficient samples: `sling` (2,307) and `short_sleeve_outwear` (685). Each remaining class was capped at **7,641 items** (the natural minimum, set by `sling_dress`).

**Split:** 70/15/15 by image (not instance), preventing label leakage. Train: 58,827 | Val: 12,632 | Test: 12,592.

**Class balance strategy:** Stratified sampling by occlusion level (visible / partial / heavy) preserves realistic occlusion distributions within each class. The raw dataset had extreme imbalance (short_sleeve_top at 23.1% vs sling_dress at 2.1%); the balanced set eliminates this.

**Augmentation:** Three tiers tested in ablation -- none, medium (scale +/-30%, rotation +/-10%, translate +/-10%), and heavy. Medium augmentation was consistently the best. Mosaic augmentation (4-image tiling, matching YOLOv8 methodology) was added for edna 1.3. Grayscale was tested and rejected -- colour is discriminative for fashion.

**Background images:** The balanced dataset contains zero pure-background images. Every image has at least one garment. This causes the model to never learn to suppress objectness on background regions, leading to false positives. For edna 1.4, 2,000 COCO val2017 images (filtered to exclude people and clothing-adjacent categories) are added as negative examples (~4% of training set) with empty label files.

---

## 2. Model Architecture & Approach

### FashionNet / edna

FashionNet is a **custom single-shot anchor-based detector built from scratch in pure PyTorch** -- no Ultralytics code, no pretrained weights, no borrowed architectures.

**Architecture:** Input 3x640x640 -> Backbone (custom CNN with residual blocks, 4 downsampling stages producing P3/P4/P5 at strides 8/16/32) -> Neck (bidirectional FPN: upsample + concat + fuse) -> Head (per-scale predictions: objectness + class + bbox). Uses CSP blocks, ConvBnReLU units, and LeakyReLU activations.

**Scales:** Three model sizes via `--model_scale`:
| Scale | Params | Channel widths | CSP depths |
|-------|--------|---------------|------------|
| s | ~11.7M | 64-128-256-512 | 1,2,3,2 |
| m | ~25M | 96-192-384-768 | 2,3,4,3 |
| l | ~43M | 128-256-512-1024 | 3,4,6,3 |

**"edna"** is the name for FashionNet trained at full scale on the balanced dataset with iterative bug fixes and hyperparameter tuning.

### Why custom vs off-the-shelf YOLO?

The project goal was to understand and build a detection pipeline end-to-end -- loss functions, target assignment, post-processing, evaluation -- rather than relying on Ultralytics as a black box. YOLOv8 serves as the performance ceiling and comparison baseline.

---

## 3. Experiments & Results

### 3.1 YOLO Baselines

**Sample dataset** (10K images, 13 classes, 970 val images):

| Model | mAP@50 | mAP@50:95 | Precision | Recall | Notes |
|-------|--------|-----------|-----------|--------|-------|
| YOLOv8L (pretrained) | **0.767** | **0.664** | 0.723 | 0.730 | Best sample-set result |
| YOLOv8L (from scratch) | 0.697 | 0.587 | 0.641 | 0.668 | COCO pretraining worth ~7% mAP |
| YOLO-World (zero-shot) | 0.146 | -- | 0.235 | 0.385 | Fashion terms not in CLIP corpus |

**Balanced dataset** (84K images, 11 classes, ~12K val images):

| Model | mAP@50 | mAP@50:95 | Precision | Recall | Notes |
|-------|--------|-----------|-----------|--------|-------|
| YOLOv8M (50 ep) | 0.575 | 0.521 | 0.558 | 0.678 | Primary comparison target |
| YOLOv8M (77 ep, best@62) | **0.592** | -- | 0.575 | 0.689 | Best balanced-set YOLO result |

The YOLOv8L sample-set result (0.767 mAP@50) is not directly comparable to balanced-set models due to different datasets, but the YOLOv8M balanced result (**0.592**) significantly **outperforms every edna variant** tested so far.

### 3.2 FashionNet Pipeline Fixes

The original FashionNet achieved **0.009 mAP@50** -- essentially broken. Root causes identified and fixed:

- **lambda_box = 0.05** (vs correct 5.0): box regression loss was weighted 100x too low; model learned objectness but never learned accurate boxes
- **Multi-cell assignment**: GT boxes assigned to only 1 grid cell instead of multiple neighbouring cells
- **Augmentation**: none by default

A 12-experiment ablation study (2K samples, 20 epochs each) identified the winning config: multi_cell + medium augmentation (val_loss 8.88, best of all configs).

### 3.3 FashionNet/edna Progression (Balanced Dataset, Val Split)

| Model | mAP@50 | Precision | Recall | F1 | Key changes vs previous |
|-------|--------|-----------|--------|----|----|
| FashionNet (original) | 0.001 | -- | -- | -- | Broken pipeline |
| fashionnet_balanced_v1 | 0.193 | 0.336 | 0.387 | 0.359 | Fixed lambda_box, +multi_cell, +aug medium |
| edna_1m_balanced_100 | 0.187 | 0.348 | 0.372 | 0.360 | No aug/multi_cell, longer training |
| **edna_1.2m** | **0.260** | 0.347 | **0.492** | **0.407** | scale=m, aug=medium, multi_cell, adamw |
| edna_1.3m | 0.203 | **0.433** | 0.357 | 0.392 | +cos_lr, +EMA, +mosaic, +IoU-obj (gr=0.5), +label smoothing |
| edna_1.4m | -- | -- | -- | -- | Bug fixes (see below), +2K background images |

**edna_1.2m remains the best** on mAP@50 and F1. edna_1.3m regressed on mAP (-0.057) despite gaining precision (+0.086), because IoU-aware objectness targets (gr=0.5) suppressed recall (-0.135). The model became more conservative -- fewer but more accurate detections.

**Test set confirmation** (held-out, 11,186 images): edna_1.2m: 0.268 mAP@50 | edna_1.3m: 0.210 mAP@50. Ranking holds. edna_1.3m fires 42% fewer detections (10,221 vs 17,540).

**Gap to YOLO:** edna_1.2m (best edna) vs YOLOv8M (best balanced YOLO): **0.332 mAP@50 gap** (0.260 vs 0.592). The YOLOv8M baseline outperforms all edna variants by a wide margin, explained by COCO pretraining, years of architecture optimization (CSPDarknet, PANet, decoupled head), and mature training recipes.

### 3.4 edna 1.4m (Currently Training)

**Bugs fixed going into 1.4:**

| Bug ID | Issue | Effect |
|--------|-------|--------|
| C1 | CIoU-aware objectness target (gr=0.5) | Soft obj targets (~0.55-0.70 instead of 1.0) caused recall collapse at conf=0.25. **Fix:** gr=0.0 (disable) |
| C2 | Objectness normalization via mosaic | Mosaic changed training distribution significantly. **Fix:** reverted, isolated one variable at a time |
| H1 | Decoding mismatch between train/inference | Predictions decoded differently in loss vs postprocess |
| H5 | Silent augmentation failure | Augmentation pipeline silently failed on certain image formats |
| H6 | Invalid class IDs | Class ID mismatch between dataset config and model output |

**Key design decisions for 1.4:** `--gr 0.0` (confirmed root cause of 1.3m recall regression); `--lambda_obj 1.0` (back to default); no cosine LR (revert to OneCycleLR as in 1.2m); no mosaic (isolate background-image variable); 85 epochs; +2,000 COCO background images.

**Expected results:**

| Metric | edna_1.2m | edna_1.4m | Target |
|--------|-----------|-----------|--------|
| mAP@50 | 0.260 | -- | > 0.260 |
| Precision | 0.347 | -- | >= 0.347 |
| Recall | 0.492 | -- | >= 0.492 |
| BG detections | untested | -- | near zero |

---

## 4. Key Findings / Lessons

- **COCO pretraining is worth ~7% mAP** even on a domain-specific fashion task; training from scratch is a severe handicap.
- **Pipeline bugs dominated early results.** lambda_box=0.05 alone explained most of the 0.001 mAP -- fixing it to 5.0 gave a 275x improvement before any architectural changes.
- **IoU-aware objectness (gr=0.5) traded recall for precision** but net-negative for mAP. The change is sound in principle (YOLOv5/v8 use it) but requires careful threshold tuning or gradual annealing.
- **val_loss and mAP@50 are weakly coupled** at this training scale. edna_1m had a better val_loss than fashionnet_balanced_v1 but worse mAP.
- **Medium augmentation helps; heavy augmentation hurts.** Excessive geometric distortion destroys training signal at limited epoch counts.
- **Background images are critical** when 100% of training images contain foreground objects. The model never learns to suppress objectness on empty regions, causing systematic false positives.
- **Threshold tuning is cheap but limited.** Best F1 improvement from sweeping conf threshold was +0.005 -- structural false positives cannot be fixed post-hoc.
- **Off-the-shelf YOLO significantly outperforms the custom detector** (0.592 vs 0.260 mAP@50), confirming that architecture maturity and pretrained features matter more than model capacity alone (edna has more parameters than YOLOv8M).

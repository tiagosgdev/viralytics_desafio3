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

Nearly identical training time (~1.31h). Batch 16 is the better config — slightly better mAP and recall. The bigger batch didn't help here.

Weakest classes in both tests: short_sleeve_dress (~0.54-0.59 mAP) and long_sleeve_dress (~0.62-0.64). These could benefit from more training data or targeted augmentation.

---

## Test 3 — YOLO-World Zero-Shot | yolov8s-worldv2 | conf=0.15 | 970 images

No fine-tuning — open-vocabulary detection via CLIP text embeddings.

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

Zero-shot mAP@50 (0.1457) vs fine-tuned YOLOv8L (0.7673) — an expected ~5x gap. Generic categories that CLIP recognises well (trousers, shorts, skirt) perform best. Specialised fashion terms (sling, vest_dress, short_sleeve_outwear) score near zero because CLIP's training data contains very few examples of these labels.

| Metric | YOLOv8L Fine-tuned (Test 1) | YOLO-World Zero-Shot |
|--------|---------------------------|----------------------|
| mAP@50 | 0.7673 | 0.1457 |
| Precision | 0.7229 | 0.2353 |
| Recall | 0.7302 | 0.3854 |
| Best class | trousers (0.9149) | trousers (0.5351) |
| Worst class | short_sleeve_dress (0.5886) | sling (0.0027) |

YOLO-World is useful as a no-training baseline or for rapid prototyping, but fine-tuning remains essential for production-grade fashion detection.

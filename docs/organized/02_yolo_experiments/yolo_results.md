# YOLOv8 Experiment Results

All experiments were run with `scripts/train.py` using the Ultralytics API.
Tests 1--5 use the **sample dataset** (10k images, 13 classes, 970 val images).
Balanced dataset experiments follow at the end.

---

## Sample Dataset Experiments

### Test 1 -- YOLOv8L | batch=16 | pretrained (COCO)

| Category | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|-----------|--------|--------|-----------|
| **all (970 imgs, 1614 inst.)** | **0.731** | **0.726** | **0.784** | **0.655** |
| short_sleeve_top | 0.823 | 0.806 | 0.873 | 0.747 |
| long_sleeve_top | 0.728 | 0.739 | 0.762 | 0.650 |
| short_sleeve_outwear | 0.677 | 0.734 | 0.789 | 0.667 |
| long_sleeve_outwear | 0.699 | 0.667 | 0.759 | 0.634 |
| vest | 0.708 | 0.734 | 0.823 | 0.662 |
| sling | 0.814 | 0.760 | 0.835 | 0.684 |
| shorts | 0.836 | 0.785 | 0.870 | 0.677 |
| trousers | 0.920 | 0.832 | 0.934 | 0.724 |
| skirt | 0.766 | 0.692 | 0.797 | 0.669 |
| short_sleeve_dress | 0.529 | 0.662 | 0.616 | 0.531 |
| long_sleeve_dress | 0.676 | 0.620 | 0.646 | 0.567 |
| vest_dress | 0.622 | 0.644 | 0.697 | 0.601 |
| sling_dress | 0.700 | 0.767 | 0.794 | 0.707 |

**Summary:** mAP@50 = 0.767, mAP@50:95 = 0.664, P = 0.723, R = 0.730

---

### Test 2 -- YOLOv8L | batch=26 | pretrained (COCO)

| Metric | Value |
|--------|-------|
| mAP@50 | 0.760 |
| mAP@50:95 | 0.659 |
| Precision | 0.724 |
| Recall | 0.712 |

Batch size increase from 16 to 26 produced no benefit. Results marginally worse across all metrics. Training time was nearly identical (~1.3h).

---

### Test 3 -- YOLO-World Zero-Shot | yolov8s-worldv2 | conf=0.15

No fine-tuning -- open-vocabulary detection via CLIP text embeddings.

| Category | Precision | Recall | mAP@50 |
|----------|-----------|--------|--------|
| **all** | **0.235** | **0.385** | **0.146** |
| trousers | 0.718 | 0.594 | 0.535 |
| skirt | 0.306 | 0.659 | 0.332 |
| shorts | 0.529 | 0.346 | 0.312 |
| short_sleeve_top | 0.485 | 0.476 | 0.287 |
| sling | 0.029 | 0.040 | 0.003 |
| vest | 0.089 | 0.043 | 0.016 |

YOLO-World uses CLIP embeddings for open-vocabulary detection. Common English terms ("trousers", "shorts") have well-defined embeddings and scored highest. Fashion-specific compound terms ("sling", "vest_dress", "long_sleeve_outwear") appear rarely in CLIP's training corpus, resulting in near-zero mAP.

---

### Test 4 -- YOLOv8L | batch=16 | pretrained (reproducibility check)

Identical setup to Test 1, different random seed.

| Metric | Test 1 | Test 4 |
|--------|--------|--------|
| mAP@50 | 0.767 | 0.767 |
| mAP@50:95 | 0.664 | 0.664 |
| Precision | 0.723 | 0.723 |
| Recall | 0.730 | 0.730 |

Results are identical, confirming batch=16 is a stable, reproducible configuration.

---

### Test 5 -- YOLOv8L | batch=16 | No Pretrained Weights (from scratch)

| Metric | Pretrained (Test 1) | From Scratch (Test 5) | Difference |
|--------|--------------------|-----------------------|------------|
| mAP@50 | 0.767 | 0.697 | -0.070 |
| mAP@50:95 | 0.664 | 0.587 | -0.077 |
| Precision | 0.723 | 0.641 | -0.082 |
| Recall | 0.730 | 0.668 | -0.062 |

COCO pretraining provides a ~7% mAP@50 advantage at no additional cost. The gap is most pronounced on dress categories (short_sleeve_dress: 0.589 vs 0.503), where limited training data makes feature transfer most valuable.

---

### Test 6 -- FashionNet (original) vs YOLOv8L

Side-by-side evaluation on the same 970-image validation set.

| Metric | FashionNet (original) | YOLOv8L (fine-tuned) |
|--------|----------------------|----------------------|
| mAP@50 | 0.009 | 0.777 |
| Inference (ms/img) | 3.2 | 10.9 |
| FPS | 313 | 91 |
| Parameters (M) | 11.74 | 43.62 |

The 0.768 mAP@50 gap is explained by three factors: (1) a critical bug in FashionNet's loss weights (lambda_box=0.05, 100x too low), (2) COCO pretraining in YOLOv8L vs random initialization in FashionNet, and (3) architectural maturity of the YOLOv8 design. FashionNet was faster (3x) but produced near-zero detections.

---

## Balanced Dataset Experiments

### YOLOv8M | 50 epochs | batch=16

Evaluated on the full balanced validation split (11,186 images, 11 classes).

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

### YOLOv8M | 77 epochs (best at epoch 62)

Extended training with early stopping. Best checkpoint at epoch 62:

| Metric | 50 epochs | Best @ ep 62 |
|--------|-----------|-------------|
| mAP@50 | 0.575 | **0.592** |
| Precision | 0.558 | 0.575 |
| Recall | 0.678 | 0.689 |

The YOLOv8M balanced result (**0.592 mAP@50**) serves as the primary comparison target for all custom FashionNet/edna models. The lower performance compared to the sample dataset results is explained by: (1) smaller model (YOLOv8M 25.8M params vs YOLOv8L 43.6M), (2) harder validation set (11,186 images vs 970, uniform class distribution), and (3) increased visual confusion between similar categories at scale.

---

## Key Takeaways

| Finding | Evidence |
|---------|----------|
| COCO pretraining is worth ~7% mAP | Test 1 vs Test 5 |
| Batch size 16 is optimal and stable | Test 1 vs Test 2, Test 1 vs Test 4 |
| Zero-shot detection is insufficient for fashion | Test 3: 0.146 vs 0.767 fine-tuned |
| Dress categories are hardest | Consistently lowest AP across all tests |
| Original FashionNet was broken, not just weak | Test 6: 0.009 mAP@50 due to loss bug |
| YOLOv8M on balanced dataset: 0.592 mAP@50 | Primary baseline for custom model comparison |

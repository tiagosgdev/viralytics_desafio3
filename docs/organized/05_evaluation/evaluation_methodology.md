# Evaluation Methodology

## Overview

All models (YOLOv8, FashionNet/edna) are evaluated using the same metrics, thresholds,
and dataset splits to ensure fair comparison. YOLOv8 models are evaluated via the Ultralytics
validation API; custom models use a dedicated evaluation script (`scripts/evaluate_custom.py`)
that implements equivalent metrics from first principles.

---

## Dataset Splits

| Split | Images | Purpose |
|-------|--------|---------|
| Train | 58,827 | Model training |
| Val | 12,632 | Hyperparameter selection, model comparison |
| Test | 12,592 | Final held-out evaluation |

Splits are by image (not instance), so a single multi-garment image is entirely in one split,
preventing label leakage. All 11 classes are represented equally in each split.

---

## Metrics

### Primary: mAP@50

A prediction is correct only if the predicted class is right AND the bounding box overlaps
the ground truth by at least 50% IoU. Per-class Average Precision is computed using VOC-style
101-point interpolation, then averaged across all 11 classes (mean Average Precision).

This is the standard single-number summary for object detection and the primary metric used
to rank all models in this project.

### Secondary: mAP@50:95

Average of mAP at IoU thresholds from 0.50 to 0.95 in steps of 0.05. A stricter metric
that penalizes imprecise localization. Reported for YOLOv8 models; not always available
for custom models due to evaluation script differences.

### Precision, Recall, F1

- **Precision:** of all predicted boxes, what fraction were correct
- **Recall:** of all ground-truth objects, what fraction did the model find
- **F1:** harmonic mean of precision and recall (2 * P * R / (P + R))

Precision alone can be misleading -- a model predicting only 3 confident boxes achieves
100% precision but misses everything. F1 penalizes both false positives and missed detections.

### Per-class AP

Identifies which clothing categories the model struggles with. This is the most important
diagnostic metric for understanding inter-class confusion patterns and guiding improvements.

### Confusion Matrix

Shows misclassification patterns. The matrix is (NC+1) x (NC+1), where the extra row/column
represents "background":
- **False positive:** model predicted something where nothing exists -- recorded as `cm[background, pred_class]`
- **False negative:** model missed a real object -- recorded as `cm[gt_class, background]`
- **Misclassification:** model found the object but assigned wrong class -- off-diagonal `cm[gt_class, pred_class]`

The confusion matrix uses class-agnostic IoU matching (best IoU regardless of class) rather
than same-class matching. This captures what the model thinks an object is vs what it actually
is, rather than treating misclassifications as independent FP + FN events.

---

## Default Thresholds

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Confidence threshold | 0.25 | Standard for YOLO-family detectors |
| NMS IoU threshold | 0.45 | Standard for YOLO-family detectors |
| Max detections | 300 | Per-image cap |

These thresholds are held constant across all model evaluations for fair comparison.
Threshold sensitivity analysis was performed on edna_1.2m (see edna_results.md).

---

## Post-processing Pipeline (Custom Models)

Custom models output raw grid predictions that must be decoded before evaluation:

1. **Decode predictions:** Convert raw `(B, 5+NC, gs, gs)` tensors from 3 scales (stride 8, 16, 32) into `(cx, cy, w, h, confidence, class_id)` in normalized coordinates
2. **Filter by confidence:** Keep cells where `obj * max(cls_scores) >= conf_thresh`
3. **Non-maximum suppression:** Per-class NMS using `torchvision.ops.nms` with class-offset trick (add `class_id * large_number` to coordinates so classes don't suppress each other)
4. **Cap detections:** Sort by confidence, keep top `max_det` per image

The decoding is the exact inverse of the target encoding in `build_targets()`: channels 0--1
are sigmoid-decoded center offsets in grid units; channels 2--3 are width/height directly
in grid units (no log-space transform).

---

## Evaluation Scripts

### YOLOv8: `scripts/evaluate.py`

Uses Ultralytics' built-in validation flow. Evaluates a specific checkpoint on a given
dataset split with configurable confidence threshold.

### Custom models: `scripts/evaluate_custom.py`

Loads a FashionNet/edna checkpoint, reads model configuration (num_classes, model_scale)
from the weights directory, runs batched inference with deterministic validation transforms,
matches predictions to ground truth via IoU, and computes all metrics.

Output: `metrics.json` containing mAP@50, per-class AP, precision, recall, F1, confusion
matrix, and evaluation configuration.

### Model comparison: `scripts/compare_models.py`

Compares custom FashionNet vs YOLOv8 (or custom vs custom) on the same dataset. Reports
per-class mAP, overall mAP, inference speed, parameter count, and weight file size.

### Metrics implementation: `src/utils/metrics.py`

All evaluation metrics are implemented from first principles: IoU computation, greedy
prediction-to-GT matching, per-class AP (VOC-style 101-point interpolation), confusion
matrix generation. This ensures the evaluation methodology is transparent and consistent
across all model types, independent of the Ultralytics framework.

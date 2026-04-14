# FashionNet Evaluation and Visualization Methodology

## Overview

FashionNet only tracks `val_loss` during training. To properly compare experiments and
analyze results, three components must be implemented in order:

```
1. postprocess.py       (no dependencies, enables everything else)
     ↓
2. evaluate_custom.py   (depends on postprocess.py + src/utils/metrics.py)
     ↓
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
3. Decode center xy (channels 0–1):
   ```
   cx_norm = (sigmoid(raw_x) + grid_x) / gs
   cy_norm = (sigmoid(raw_y) + grid_y) / gs
   ```
4. Decode width/height (channels 2–3):
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

Top-level convenience: `decode_predictions` → `nms` per image → sort by confidence → cap to `max_det`.

**Note on grid decoding:** `build_targets()` in `loss.py` stores box targets as
`(cx - cell_i, cy - cell_j, w, h)` in grid units. Channels 0–1 predict these offsets
(sigmoid applied), and channels 2–3 predict w/h directly in grid units (no exp/log transform).
Postprocessing must be the exact inverse of this encoding.

---

## Component 2: Evaluation Script

**File:** `scripts/evaluate_custom.py`

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

1. **Load checkpoint** — read `config.json` from weights dir for `num_classes` and `model_scale`,
   instantiate `FashionNet(num_classes, scale)`, load `state_dict`
2. **Collect per-image GT** — iterate `FashionDataset` with val transforms (deterministic)
   to get post-augmentation boxes and classes per image
3. **Batched inference** — `model.eval()`, `torch.no_grad()`, call `postprocess()` on predictions
4. **Match predictions to GT** — IoU-based matching:
   - For **mAP/F1**: same-class matching only (standard)
   - For **confusion matrix**: class-agnostic matching by best IoU, then record `cm[gt_class, pred_class]`
5. **Compute metrics:**
   - Per-class AP using existing `per_class_ap` from `src/utils/metrics.py`
   - Per-class precision, recall, F1 from accumulated TP/FP/FN counts
   - `(NC+1) × (NC+1)` confusion matrix (last row/col = "background"):
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

### Why (NC+1) × (NC+1)?

The extra row/column represents "background" (no detection). This captures:
- **FP:** model predicted something where there's nothing → `cm[background, pred_class]`
- **FN:** model missed a real object → `cm[gt_class, background]`
- **Misclassification:** model found the object but wrong class → `cm[gt_class, pred_class]` (off-diagonal)

---

## Component 3: Visualization Script

**File:** `scripts/visualize_results.py`

### CLI Arguments

```
--metrics_json   Path to metrics.json from evaluate_custom.py
--history_json   Path to history.json from train_custom.py
--output_dir     Where to save PNGs (default: results/plots)
--exp_dirs       Multiple experiment dirs for comparison table
--dpi            Image DPI (default: 150)
```

### Plot 1: Confusion Matrix Heatmap

- `(NC+1) × (NC+1)` heatmap using `seaborn.heatmap`
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
- Color-coded: green (AP >= 0.5), yellow (0.3–0.5), red (< 0.3)
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
python scripts/visualize_results.py \
  --metrics_json models/weights/exp2_loss_fix/metrics.json \
  --history_json models/weights/exp2_loss_fix/history.json \
  --output_dir results/plots/exp2

# Compare all experiments:
python scripts/visualize_results.py \
  --exp_dirs models/weights/exp1_baseline models/weights/exp2_loss_fix \
             models/weights/exp3_multicell models/weights/exp4_aug_medium \
  --output_dir results/plots/comparison
```

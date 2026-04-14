# edna Training Results

edna is the FashionNet model family trained at larger scale or with different configurations
on the full balanced dataset. This document covers three training runs and their comparison.

All evaluations use `scripts/evaluate_custom.py`, val split (11,186 images), conf=0.25, NMS IoU=0.45.

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
| Key flags | aug=medium, multi_cell | — |

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
edna_1m_balanced_100 training 3× longer and achieving a better val_loss (2.6953 vs 3.0591).
fashionnet_balanced_v1 edges out on mAP@50 and recall, while edna_1m wins on F1 and precision.

The aug=medium + multi_cell flags in fashionnet_balanced_v1 provide marginal but real benefit
for mAP. The significantly lower val_loss of edna_1m_balanced_100 does not translate into
better detection metrics — **val_loss and mAP@50 are not tightly coupled at this training scale.**

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
| edna_1m_balanced_100 | 0.1869 | 0.3597 | 2.6953 | 63 | — |
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
aug=medium and multi_cell accounts for the gain — consistent with the original exp4 finding.

---

## Threshold Tuning — edna_1.2m

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
-0.022 mAP@50. The gain is negligible. The low precision is structural — the model
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
3. YOLOv8M has ~25.8M params vs edna_1.2m at ~34M — more capacity alone does not close the gap

---

## Planned Next Runs — YOLOv8 on Balanced Dataset

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
python scripts/train.py \
  --model yolov8n \
  --epochs 100 \
  --batch 16 \
  --data data/balanced_dataset/dataset.yaml \
  --patience 20

python scripts/train.py \
  --model yolov8s \
  --epochs 100 \
  --batch 16 \
  --data data/balanced_dataset/dataset.yaml \
  --patience 20

python scripts/train.py \
  --model yolov8l \
  --epochs 100 \
  --batch 16 \
  --data data/balanced_dataset/dataset.yaml \
  --patience 20
```

### Evaluation Commands

```bash
python scripts/evaluate.py --weights models/weights/yolov8n_fashion/weights/best.pt \
  --data data/balanced_dataset/dataset.yaml
python scripts/evaluate.py --weights models/weights/yolov8s_fashion/weights/best.pt \
  --data data/balanced_dataset/dataset.yaml
python scripts/evaluate.py --weights models/weights/yolov8l_fashion/weights/best.pt \
  --data data/balanced_dataset/dataset.yaml
```

### Final Comparison Commands

```bash
# edna_1.2m vs yolov8n (size-matched)
python scripts/compare_models.py \
  --custom_weights models/weights/edna_1.2m/best.pt \
  --yolo_weights   models/weights/yolov8n_fashion/weights/best.pt \
  --data           data/balanced_dataset \
  --out            docs/compare_edna_1.2m_vs_yolov8n.json

# edna_1.2m vs yolov8s (param-matched to FashionNet family)
python scripts/compare_models.py \
  --custom_weights models/weights/edna_1.2m/best.pt \
  --yolo_weights   models/weights/yolov8s_fashion/weights/best.pt \
  --data           data/balanced_dataset \
  --out            docs/compare_edna_1.2m_vs_yolov8s.json

# edna_1.2m vs yolov8l (best-effort ceiling)
python scripts/compare_models.py \
  --custom_weights models/weights/edna_1.2m/best.pt \
  --yolo_weights   models/weights/yolov8l_fashion/weights/best.pt \
  --data           data/balanced_dataset \
  --out            docs/compare_edna_1.2m_vs_yolov8l.json
```

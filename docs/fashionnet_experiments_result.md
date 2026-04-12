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

*exp1 val_loss uses lambda_box=0.05 so the box component is weighted ~100x less than all other experiments — not directly comparable.

---

## Analysis

### Winner: exp4 — multi_cell + medium augmentation (val_loss 8.88)

The combination of multi-cell GT assignment and medium augmentation (scale ±30%, rotation ±10%, translate ±10%) gave the best result. This also had the lowest raw box loss (1.7275), meaning the model is learning to regress boxes more accurately.

### Grayscale hurts

Removing colour (exp8: 12.62, exp9: 12.82) consistently made results worse. Colour information is discriminative for this task — clothing categories differ in shape but colour also provides useful signal. The original hypothesis was that same-colour confusion was a problem, but the data suggests colour helps more than it hurts overall.

### Heavy augmentation hurts

exp5 (heavy) at 12.35 is worse than exp4 (medium) at 8.88. With only 2000 samples at 20 epochs, the aggressive scale/rotation/noise in heavy mode is too destructive — the model can't learn fast enough to handle the increased variance.

### Smaller changes had no clear benefit

- **Dropout** (exp7: 10.38 vs exp4: 8.88) — no benefit, possibly slightly harmful
- **Warmup** (exp10: 10.38) — no measurable improvement at 20 epochs
- **SGD** (exp11: 12.33) — worse than AdamW at this epoch count; SGD needs more epochs
- **EMA** (exp12: 16.47) — misleading result; with decay=0.9999 the EMA model needs thousands of batches to warm up. Not useful at 20 epochs.

---

---

## Full Training — fashionnet_balanced_v1

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

### Option A — Merge similar classes
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

Reduces from 11 → 7 classes. Requires rebuilding labels and dataset.yaml.

### Option B — Add images to weakest classes
Only useful if classes are visually distinct but underrepresented. Less likely to help for short/long sleeve top confusion since the model already has 52K images to learn from.

---

## Next Step

Compare fashionnet_balanced_v1 against YOLOv8L:

```bash
python scripts/compare_models.py \
  --custom_weights models/weights/fashionnet_balanced_v1/best.pt \
  --yolo_weights models/weights/yolov8l_fashion/weights/best.pt \
  --data data/balanced_dataset \
  --out docs/compare_fashionnet_v1_vs_yolov8l.json
```

----------------------
-------  EDNA --------
----------------------

## Full Training — edna_1m_balanced_100

### Setup

- Config: no explicit flags (default settings)
- Dataset: full balanced_dataset, 52,199 training images (no sample cap)
- Epochs: 100
- Batch: 32 (3263 batches/epoch)
- Device: CUDA (NVIDIA GPU, 16GB VRAM)
- Training time: 2064m 44s (~34h 24m)
- Best val_loss: 2.6953 (epoch 63)
- Weights: `models/weights/edna_1m_balanced_100/best.pt`

### Evaluation — fashionnet_balanced_v1 vs edna_1m_balanced_100

Evaluated with `scripts/evaluate_custom.py`, val split (11,186 images), conf=0.25, NMS IoU=0.45.

| Metric | fashionnet_balanced_v1 | edna_1m_balanced_100 |
|--------|----------------------|----------------------|
| mAP@50 | **0.1930** | 0.1869 |
| Precision | 0.3356 | **0.3479** |
| Recall | **0.3870** | 0.3723 |
| F1 | 0.3594 | **0.3597** |
| Best val_loss | 3.0591 | **2.6953** |
| Best epoch | 87 | 63 |
| Key flags | aug=medium, multi_cell | — |

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

## Full Training — edna_1.2m

### Setup

- Config: aug=medium, multi_cell=true, model_scale=m (~1.2M params), optimizer=adamw
- Dataset: full balanced_dataset, 52,199 training images (no sample cap)
- Epochs: 100
- Batch: 16
- Device: CUDA (NVIDIA GPU, 16GB VRAM)
- Training time: 2093m 31s (~34h 53m)
- Best val_loss: 2.8128 (epoch 100)
- Weights: `models/weights/edna_1.2m/best.pt`

### Evaluation — edna_1.2m

Evaluated with `scripts/evaluate_custom.py`, val split (11,186 images), conf=0.25, NMS IoU=0.45.

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
| edna_1m_balanced_100 | 0.1869 | 0.3597 | 2.6953 | 63 | — |
| **edna_1.2m** | **0.2600** | **0.4068** | 2.8128 | 100 | aug=medium, multi_cell |

### Analysis

edna_1.2m is a clear improvement over both previous versions: +0.0670 mAP@50 over fashionnet_balanced_v1 and +0.0731 over edna_1m_balanced_100. F1 also improves meaningfully (+0.0474 vs both). Recall jumps to 0.4920 — the highest of the three — suggesting the medium-scale model with aug=medium + multi_cell is better at finding objects, though precision (0.3467) remains the lowest, meaning more false positives.

The biggest gains over edna_1m_balanced_100 are on long_sleeve_outwear (+0.0872 AP), vest (+0.0789 AP), shorts (+0.0509 AP), and skirt (+0.0907 AP). The weak classes (short_sleeve_top, long_sleeve_top) see meaningful improvement too (+0.0407 and +0.0739 AP respectively) but remain the bottom two.

Scaling the model (m vs default s scale in edna_1m) combined with re-enabling aug=medium and multi_cell accounts for the gain — consistent with the original exp4 finding that these flags help.

---

## Next — YOLOv8 Baseline on balanced_dataset

All previous YOLOv8 weights were trained on `data/sample_dataset`, making them invalid as a comparison against edna_1.2m. These three runs retrain YOLO on the same balanced_dataset used by all FashionNet models.

### Planned Runs

| Run | Model | Params | Purpose |
|-----|-------|--------|---------|
| yolov8n_balanced | yolov8n | ~3.2M | Size-matched comparison against edna_1.2m (~1.2M) |
| yolov8s_balanced | yolov8s | ~11M | Param-matched comparison against fashionnet_balanced_v1 (~11.74M) |
| yolov8l_balanced | yolov8l | ~43.7M | Best-effort YOLO ceiling on this dataset |

### Commands

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

### Evaluation (after each run)

```bash
python scripts/evaluate.py --weights models/weights/yolov8n_fashion/weights/best.pt \
  --data data/balanced_dataset/dataset.yaml
python scripts/evaluate.py --weights models/weights/yolov8s_fashion/weights/best.pt \
  --data data/balanced_dataset/dataset.yaml
python scripts/evaluate.py --weights models/weights/yolov8l_fashion/weights/best.pt \
  --data data/balanced_dataset/dataset.yaml
```

### Final Comparison

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

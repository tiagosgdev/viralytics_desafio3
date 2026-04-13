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
| 86 | 2.8322 | — |
| 90 | 2.8232 | -0.0090 |
| 95 | 2.8147 | -0.0085 |
| 100 | 2.8128 | -0.0019 |

Only 0.0194 drop over the last 15 epochs. The flat AdamW with no LR schedule has stalled
at a local minimum. More epochs on the same config will not meaningfully improve results.

---

## Suggestions

### 1. Threshold Tuning (no retraining, ~10 min)

Precision (0.3467) is significantly lower than recall (0.4920) — the model over-predicts.
Default conf=0.25 may not be optimal for F1. The threshold sweep (already done, see
`edna_results.md`) shows conf=0.30 peaks F1 at 0.4123 (+0.0055 over default) at the cost
of -0.022 mAP. The gain is small but available at zero training cost.

```bash
for conf in 0.30 0.35 0.40 0.45; do
  python scripts/evaluate_custom.py \
    --weights models/weights/edna_1.2m/best.pt \
    --data data/balanced_dataset \
    --conf $conf
done
```

---

### 2. Retrain with Cosine LR + EMA (~35h)

The plateau is a scheduling problem. Flat AdamW with no decay stalls in later epochs.
Two additions address this:

- **`--cos_lr`**: cosine annealing gradually reduces LR, allowing finer convergence
  in later epochs instead of stalling
- **`--ema`**: was misleading in the 20-epoch ablations (decay=0.9999 needs ~10K steps to
  warm up). At 100 epochs × 3,263 batches/epoch = ~326K steps, EMA is fully effective and
  should improve validation mAP at essentially zero cost

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset \
  --model_scale m \
  --augment medium \
  --multi_cell \
  --cos_lr \
  --ema \
  --epochs 120 \
  --batch 16 \
  --output models/weights/edna_1.2m_coslr_ema
```

---

### 3. Class Merge — short_sleeve_top + long_sleeve_top → top (~35h + label rebuild)

short_sleeve_top and long_sleeve_top have been the **bottom two classes across every model**:

| Model | short_sleeve_top AP | long_sleeve_top AP |
|-------|--------------------|--------------------|
| fashionnet_balanced_v1 | 0.0860 | 0.0999 |
| edna_1m_balanced_100 | 0.0877 | 0.0709 |
| edna_1.2m | 0.1284 | 0.1448 |
| YOLOv8M (balanced) | 0.2935 | 0.4077 |

This is a structural problem — the classes are visually near-identical (same silhouette,
only sleeve length differs). No amount of training tweaks will reliably resolve inter-class
confusion at this level.

**Proposed merge:** combine into a single `top` class, reducing 11 → 10 classes.

Steps required:
1. Rebuild labels — remap class IDs in all annotation files
2. Update `dataset.yaml` — nc: 10, remove one name
3. Full retrain from scratch on the merged dataset

Expected outcome: the merged `top` class will have 2× the training examples and no
ambiguity between short and long sleeve. The mAP denominator shrinks by one class but
the overall score should rise meaningfully.

---

### 4. Larger Model Scale — scale=l (~50h+)

edna_1.2m uses model_scale=m (~34M params). The FashionNet family also supports scale=l.
Given YOLOv8M (25.8M) outperforms edna_1.2m by ~0.31 mAP despite fewer parameters, model
capacity alone is not the bottleneck — but testing scale=l is worth doing before concluding
on architecture limits.

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset \
  --model_scale l \
  --augment medium \
  --multi_cell \
  --cos_lr \
  --ema \
  --epochs 120 \
  --batch 16 \
  --output models/weights/edna_l_coslr_ema
```

---

## Priority Order

| Priority | Suggestion | Effort | Expected gain |
|----------|-----------|--------|---------------|
| 1 | Threshold tuning | ~10 min | Small F1 gain (+0.005), no retraining |
| 2 | Retrain with cos_lr + EMA | ~35h | Moderate mAP improvement |
| 3 | Class merge (top) | ~35h + label rebuild | Likely largest single gain |
| 4 | Scale=l retrain | ~50h+ | Unknown, architecture ceiling test |

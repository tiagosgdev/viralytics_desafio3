# FashionNet/edna — Code Review

**Branch:** code-review  
**Reviewed by:** Opus agent  
**Scope:** `src/custom_model/`, `scripts/training/`, `scripts/evaluation/`, `src/utils/`

---

## CRITICAL

### C1. `loss.py:248` — Objectness target uses CIoU (not plain IoU)

```python
obj_mask[obj_mask_bool] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0)
```

`iou` here is **CIoU** (range ~`[-1, 1]`), not plain IoU. CIoU includes distance and aspect-ratio penalties so it is systematically *lower* than actual overlap. With `gr=0.5`, a prediction with real IoU≈0.7 gets soft target ≈ 0.5 + 0.5×0.55 ≈ 0.78 — focal BCE then pulls obj logit toward 0.78, which after multiplying by class score often drops below `conf=0.25`. **This is the confirmed root cause of edna_1.3m recall regression.**

**Fix:**
```python
iou_loss = bbox_iou(pred_boxes, tgt_boxes, ciou=True)   # for box loss only
loss_box = loss_box + (1 - iou_loss).mean()

with torch.no_grad():
    iou_target = bbox_iou(pred_boxes, tgt_boxes, ciou=False).clamp(0)
obj_mask[obj_mask_bool] = (1.0 - self.gr) + self.gr * iou_target
```

---

### C2. `loss.py:88, 256` — Objectness loss positives drowned by mean-reduction over all cells

Focal BCE is `mean()`-reduced over all ~8400 cells. With ~5 positives per image and `alpha=0.25` (down-weights positives), positive contribution is `(0.25 × few) / 8400` of the total loss. Positive gradient is negligible, objectness systematically underfits.

**Fix:** Sum then divide by batch size:
```python
obj_l = alpha_t * (1 - p_t) ** gamma * bce      # (B, gs, gs)
loss_obj = loss_obj + obj_l.sum() / pred.shape[0]
```
Also consider raising `lambda_obj` after fixing normalization, and dropping `alpha` toward 0.5 for objectness specifically.

---

## HIGH

### H1. `loss.py:233` + `postprocess.py:67` + `compare_models.py:81` — Raw `p_wh` allows negative w/h; decoders disagree

`p_wh = p[..., 2:4]` is fed raw into `bbox_iou`. Negative predictions produce inverted/zero-area boxes. **Worse:** two decoders exist with different semantics:
- `postprocess.py:67`: `.clamp(min=0)` — zero-area if negative
- `compare_models.py:81`: `.abs()` — valid box with flipped sign

Any A/B comparison between these is invalid — same checkpoint, different mAP.

**Fix:** Pick one (clamp). Apply in both scripts. In the loss:
```python
p_wh = p[..., 2:4].clamp(min=0)
```
Or use `F.softplus`. Have `compare_models.py` call `postprocess.postprocess()` directly instead of duplicating decoding logic.

---

### H2. `loss.py:164` — Class target 0.95 caps max confidence below threshold

```python
target_cls[bi, cj, ci, cls] = 0.95
```

BCE optimal logit at convergence = `logit(0.95) ≈ 2.94` → max class sigmoid = 0.95. With soft objectness from `gr=0.5` (obj≈0.6), `conf = obj × cls` maxes at `0.6 × 0.95 ≈ 0.57`. Many TPs land near this cap, close to the 0.25 threshold. Asymmetric smoothing — positives hurt, negatives untouched.

**Fix:** Either `1.0` (no smoothing) or symmetric:
```python
SMOOTH = 0.05
target_cls[bi, cj, ci, :] = SMOOTH              # negatives
target_cls[bi, cj, ci, cls] = 1.0 - SMOOTH      # positive
```

---

### H3. `loss.py:222-260` — Losses summed across 3 scales without per-scale normalization

Box and obj losses accumulate raw across P3/P4/P5. No per-scale balance weights. P5 (large-scale) objectness dominates because most cells are always background.

**Fix:**
```python
SCALE_WEIGHTS = [4.0, 1.0, 0.4]   # P3, P4, P5
...
loss_obj = loss_obj + SCALE_WEIGHTS[i] * focal_bce(...)
```

---

### H4. `loss.py:124-168` — All GTs assigned to all 3 scales; no scale stratification

A 600px object is assigned to a P3 (stride 8) cell — regression target is ~75 cells wide, impossible. A 20px object at P5 (stride 32) spans <1 cell. Both hurt precision and recall on extreme sizes.

**Fix:** Only assign GT to the scale where its pixel size falls in range:
```python
size_thresholds = [(0, 64), (64, 256), (256, 9999)]  # P3, P4, P5
obj_size = max(w, h) * img_size
# assign to one scale only
```

---

### H5. `dataset.py:260-272` — Silent augmentation failure returns all-zero image

```python
except Exception:
    img_t = torch.zeros(3, self.img_size, self.img_size)
    boxes, classes = [], []
```

All-zero tensor is visually identical across failed samples — model can memorize the signature. Hides bugs. Indistinguishable from real background images added for FP suppression.

**Fix:**
```python
except Exception as e:
    print(f"[dataset] aug failed for {self.samples[idx][0]}: {e}", file=sys.stderr)
    aug = get_val_transforms(self.img_size)(image=img, bboxes=[], class_labels=[])
    img_t = aug['image']
    boxes, classes = [], []
```

---

### H6. `loss.py:130-165` — Out-of-range class IDs set `obj_mask` but not `target_cls`

In `build_targets`, `if cls < num_classes:` check only guards `target_cls` assignment, not `obj_mask`. A label with `cls >= num_classes` still marks the cell as positive (obj_mask=1) with all-zero class targets — trains the model to output zero classification confidence on that cell.

**Fix:** Move the guard up so the entire assignment is skipped:
```python
if cls >= num_classes:
    continue
obj_mask[bi, cj, ci] = 1
...
```

---

### H7. `compare_models.py:92` — Confidence filter uses obj-only, not obj×cls

`postprocess.py:76` filters by `obj * max_cls`. `compare_models.py:92` filters by `p_obj > conf_thresh` (objectness alone). Different semantics — any mAP from `compare_models` is not comparable to `evaluate_custom`.

**Fix:** Replace custom decoder in `compare_models.py` with a call to `postprocess.postprocess()`.

---

## MEDIUM

### M1. `model.py:266-268` — Head bias initialized to 0, not the objectness prior

With `bias=0`, `sigmoid(0)=0.5` for objectness on random init. Focal loss gradient is huge and dominated by background cells. RetinaNet trick: set bias so initial `sigmoid ≈ 0.01`.

**Fix:**
```python
import math
bias = self.pred.bias.view(-1)
bias.data[4] = -math.log((1 - 0.01) / 0.01)   # objectness
bias.data[5:] = -math.log((1 - 0.01) / 0.01)  # classes
```

---

### M2. `evaluate_custom.py:357` — `macro_prec` is actually micro-average

Variable names at line 357 say `macro_prec`/`macro_rec` but the computation is sum(TP)/sum(TP+FP) — that is **micro**. Misleading label.

**Fix:** Rename to `micro_prec`/`micro_rec`, or compute true macro as `np.mean([m['precision'] for m in per_class_metrics.values()])`.

---

### M3. `train_custom.py:317-329` — Resume discards EMA history and `best_val`

After `--resume`, `best_val` stays `inf` (next epoch always overwrites `best.pt`), and EMA is rebuilt from loaded weights but with `updates=0` — warmup restarts, ~2000 post-resume steps barely smooth.

**Fix:**
```python
start_epoch = ckpt['epoch'] + 1
best_val = ckpt.get('metrics', {}).get('val_loss', float('inf'))
if ema is not None and ckpt.get('ema'):
    ema.ema.load_state_dict(ckpt['ema'])
    ema.updates = ckpt.get('ema_updates', 10000)
```

---

### M4. `evaluate_custom.py:171-175` — Confusion matrix O(N_pred × N_gt) pure Python loop

~25M Python `iou()` calls over a val pass of 17K images. Dominates eval runtime.

**Fix:**
```python
import torchvision.ops as tv_ops
pb = torch.tensor(pred_boxes); gb = torch.tensor(gt_boxes)
iou_matrix = tv_ops.box_iou(pb, gb).numpy()
```

---

## LOW

| ID | Location | Issue |
|----|----------|-------|
| L1 | `dataset.py:144` | `random.seed(42)` mutates global RNG. Use `random.Random(42)` instance. |
| L2 | `train_custom.py:161` | `clip_grad_norm_(..., max_norm=10.0)` too high. YOLOv5/v8 use 1.0–4.0. |
| L3 | `train_custom.py:309` | Resume + OneCycleLR total_steps mismatch — schedule is wrong after resume. |
| L4 | `postprocess.py:139` | Class-offset NMS trick breaks if boxes in pixel space >10000px. Use `tv_ops.batched_nms`. |
| L5 | `compare_models.py:380` | `torch.load` without `weights_only=True` — security concern with untrusted paths. |
| L6 | `dataset.py:317` | `pin_memory=torch.cuda.is_available()` hardcoded; MPS path gets wasted pin_memory. |

---

## Top Fixes by Expected Impact

| Priority | File | Issue | Expected Gain |
|----------|------|-------|---------------|
| 1 | `loss.py:248` | C1: use plain IoU for obj target | Directly fixes recall regression |
| 2 | `loss.py:164` | H2: class target 0.95 → 1.0 or symmetric | Restores max conf above threshold |
| 3 | `loss.py:88, 256` | C2: mean→sum/batch for obj loss | Objectness learns positives |
| 4 | `loss.py:233` / `postprocess.py` | H1: unify p_wh decode (clamp in both) | Fixes A/B comparison validity |
| 5 | `loss.py:124-168` | H4: scale stratification | Expected mAP bump esp. small/large objects |
| 6 | `model.py:266` | M1: obj prior bias | Stabilizes early training |

Fixes 1+2 alone should recover the recall lost in edna_1.3m. Fixes 3+4+5+6 should push the next run materially above the current best (mAP@50=0.2600, R=0.4920).

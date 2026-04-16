# edna_1.4m — Training Plan

**Goal:** Add background-only images to fix bg/fg confusion. Retrain edna on balanced dataset + negatives.

**Window:** Friday 09:30 → Sunday 12:00 (~50h)

---

## What Caused edna_1.3m Recall Regression

**Primary culprit: C1 — IoU-aware objectness targets (`gr=0.5`)**, not `lambda_obj=1.5`.

`loss.py:248` — with `gr=0.5`, the positive-cell objectness target becomes:
```
obj_target = 0.5 + 0.5 * iou.clamp(0)
```
Early-training IoU is typically 0.1–0.4, so targets drop to ~0.55–0.70 instead of 1.0.
At conf=0.25 threshold, many true positives fall below threshold → recall collapses.

`lambda_obj=1.5` amplified the effect (larger loss weight on already-soft targets) but was not the root cause.
`cos_lr`, `EMA`, and `mosaic` are unlikely causes — they don't produce an asymmetric ±0.13 P/R swing.

**edna_1.4m fix:** `--gr 0.0` (explicit, disables C1). This is already the default in `train_custom.py` but must be stated clearly to avoid future confusion.

---

## Root Cause

Entire balanced dataset has clothing in every image — zero pure-background examples.
Model never learns to suppress objectness on background regions.
Confusion matrix confirms: clothing absorbed into background (FN), not misclassified between classes.

---

## Step 1 — Download Background Images (~1–2h)

Target: **2,000 images** (~4% of 58K train set). 500 is too weak a signal (1%) to meaningfully shift objectness distribution.

Run this script locally to download 2,000 COCO val2017 images with no people or clothing:

```python
import requests, json, os, shutil
from pathlib import Path

print("Downloading COCO annotations...")
os.system("wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -q")
os.system("unzip -q annotations_trainval2017.zip annotations/instances_val2017.json")

with open("annotations/instances_val2017.json") as f:
    coco = json.load(f)

# Exclude: person + clothing-adjacent categories
EXCLUDE = {1, 27, 28, 31, 32}  # person, backpack, umbrella, handbag, tie

ann_by_img = {}
for ann in coco['annotations']:
    ann_by_img.setdefault(ann['image_id'], set()).add(ann['category_id'])

clean_imgs = [
    img for img in coco['images']
    if not ann_by_img.get(img['id'], set()) & EXCLUDE
][:2000]

print(f"Found {len(clean_imgs)} clean background images")

out = Path("bg_images")
out.mkdir(exist_ok=True)
for i, img in enumerate(clean_imgs):
    url = img['coco_url']
    fname = out / f"bg_{img['id']}.jpg"
    r = requests.get(url, stream=True)
    with open(fname, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    if i % 50 == 0:
        print(f"  {i}/{len(clean_imgs)}")

print("Done.")
```

---

## Step 2 — Add to Dataset (~15 min)

```bash
# Copy images into train split
cp bg_images/*.jpg data/balanced_dataset/images/train/

# Create empty label files (empty = no objects = background)
for f in bg_images/*.jpg; do
    touch data/balanced_dataset/labels/train/$(basename $f .jpg).txt
done
```

Verify dataset loader handles them (it does — `dataset.py` line 161 skips missing/empty labels gracefully):

```bash
# Should show ~2000 empty files
find data/balanced_dataset/labels/train -name "bg_*.txt" -empty | wc -l
```

---

## Step 3 — Train (~42h)

```bash
python scripts/train_custom.py \
  --data data/balanced_dataset \
  --model_scale m \
  --epochs 85 \
  --batch 32 \
  --lr 0.001 \
  --lambda_box 5.0 \
  --lambda_obj 1.0 \
  --lambda_cls 0.5 \
  --gr 0.0 \
  --augment medium \
  --multi_cell \
  --optimizer adamw \
  --weight_decay 0.01 \
  --device cuda \
  --output models/weights/edna_1.4m
```

**Key decisions:**
- `--gr 0.0` — explicit. Disables C1 IoU-aware objectness targets. C1 is the confirmed cause of edna_1.3m recall regression, not lambda_obj.
- `--lambda_obj 1.0` — back to default. 1.5 amplified the C1 effect but is not root cause.
- No `--cos_lr` — revert to OneCycleLR (same as edna_1.2m which holds the best mAP).
- No `--mosaic` — isolate the background image variable. One change at a time.
- 85 epochs — edna_1.2m was ~21 min/epoch → 85 epochs ≈ ~30h.

**Timeline:**
| | |
|--|--|
| Start | Friday ~11:00 (after setup) |
| End | Saturday ~17:00 (~30h training) |
| Buffer | ~19h for eval + docs |

---

## Step 4 — Evaluate (~30 min)

```bash
# Standard val eval
python scripts/evaluate_custom.py \
  --weights models/weights/edna_1.4m/best.pt \
  --data data/balanced_dataset \
  --conf 0.25
```

**Also run eval on background images only** — directly verifies the fix.
`--split` only accepts `val`/`test` so create a temp dataset:

```bash
mkdir -p data/bg_test/images/test
mkdir -p data/bg_test/labels/test
cp data/balanced_dataset/images/train/bg_*.jpg data/bg_test/images/test/
for f in data/bg_test/images/test/*.jpg; do
    touch data/bg_test/labels/test/$(basename $f .jpg).txt
done
cp data/balanced_dataset/dataset.yaml data/bg_test/dataset.yaml

python scripts/evaluation/evaluate_custom.py \
  --weights models/weights/edna_1.4m/best.pt \
  --data data/bg_test \
  --split test \
  --conf 0.25
```

Total detections should be near zero. High detections = objectness fix didn't work.

Compare against edna_1.2m baseline:

| Metric | edna_1.2m | edna_1.4m | Δ |
|--------|-----------|-----------|---|
| mAP@50 | 0.2600 | — | — |
| Precision | 0.3467 | — | — |
| Recall | 0.4920 | — | — |
| F1 | 0.4068 | — | — |
| Detections (val) | 17,237 | — | — |

Fill in after eval.

**Success criteria:**
- mAP@50 > 0.2600
- Recall ≥ 0.4920 (not regressed)
- Precision ≥ 0.3467 (background images should suppress FPs)
- Near-zero detections on background-only images

---

## Step 5 — Document Results

Add results to `docs/organized/04_edna/edna_results.md` under new `## edna_1.4m` section.
Update priority table in `edna_next_steps.md`.

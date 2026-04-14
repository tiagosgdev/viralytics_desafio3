# Viralytics / FashionSense — Documentation Index

This directory contains the organized documentation for the FashionSense ML project:
a clothing detection system built on the DeepFashion2 dataset.

---

## Project Summary

The project investigates clothing detection from two angles:

1. **YOLOv8 fine-tuning** — strong engineering baseline using pretrained COCO weights
2. **FashionNet (custom)** — from-scratch detector research, progressively improved through
   a series of ablation experiments (FashionNet → fashionnet_balanced_v1 → edna family)

Both tracks are evaluated on the same balanced 11-class dataset derived from DeepFashion2.

---

## Directory Structure

```
docs/organized/
  README.md                       ← this file
  01_dataset/
    dataset_analysis.md           ← raw dataset stats, balancing methodology, train/val/test splits
  02_yolo_experiments/
    yolo_results.md               ← all YOLOv8 training runs (Tests 1–6), comparisons, conclusions
  03_fashionnet_experiments/
    fashionnet_pipeline_fixes.md  ← identified issues, fixes, CLI flags, 12-experiment ablation plan
    fashionnet_results.md         ← ablation results (20-epoch), full training (fashionnet_balanced_v1)
  04_edna/
    edna_results.md               ← edna_1m_balanced_100, edna_1.2m results and 3-way comparison
    edna_next_steps.md            ← threshold tuning, cos_lr+EMA, class merge, scale-up suggestions
  05_evaluation/
    evaluation_methodology.md     ← post-processing spec, mAP/F1/confusion matrix implementation plan
  06_codebase/
    codebase_explanation.md       ← full architectural walkthrough of all modules
```

---

## Key Results at a Glance

| Model | Dataset | mAP@50 | Notes |
|-------|---------|--------|-------|
| YOLOv8L (fine-tuned) | sample (10k) | 0.767 | Best YOLO on sample set |
| YOLOv8M (fine-tuned) | balanced (84k) | 0.575 | Balanced dataset baseline |
| YOLO-World (zero-shot) | sample (10k) | 0.146 | No fine-tuning |
| FashionNet (original) | sample (10k) | 0.009 | Broken loss weights |
| fashionnet_balanced_v1 | balanced (84k) | 0.276 | Fixed pipeline, 100 epochs |
| edna_1m_balanced_100 | balanced (84k) | 0.187 | Default flags, 100 epochs |
| edna_1.2m | balanced (84k) | 0.260 | scale=m, aug=medium, multi_cell |

---

## Reading Order

For a new reader, the recommended order is:

1. `01_dataset/dataset_analysis.md` — understand the data
2. `02_yolo_experiments/yolo_results.md` — understand the strong baseline
3. `03_fashionnet_experiments/fashionnet_pipeline_fixes.md` — understand why FashionNet was poor and what was fixed
4. `03_fashionnet_experiments/fashionnet_results.md` — see the fix impact
5. `04_edna/edna_results.md` — see the scaled-up model results
6. `04_edna/edna_next_steps.md` — understand what to do next
7. `05_evaluation/evaluation_methodology.md` — understand the evaluation infrastructure
8. `06_codebase/codebase_explanation.md` — understand the full system architecture

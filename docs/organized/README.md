# FashionSense -- Documentation Index

This directory contains the organized documentation for the FashionSense project:
a clothing detection system built on the DeepFashion2 dataset, investigating both
off-the-shelf YOLOv8 fine-tuning and a custom detector built from scratch in pure PyTorch.

---

## Directory Structure

```
docs/organized/
  README.md                       <- this file
  01_dataset/
    dataset_analysis.md           <- raw dataset stats, balancing methodology, train/val/test splits
  02_yolo_experiments/
    yolo_results.md               <- all YOLOv8 training runs, comparisons, conclusions
  03_fashionnet_experiments/
    fashionnet_pipeline_fixes.md  <- bugs found in original FashionNet, fixes applied, ablation study
    fashionnet_results.md         <- ablation results, full training (fashionnet_balanced_v1)
  04_edna/
    edna_results.md               <- edna 1.2m, 1.3m, 1.4m results with full metrics and analysis
  05_evaluation/
    evaluation_methodology.md     <- metrics, thresholds, evaluation infrastructure
  06_codebase/
    codebase_explanation.md       <- system architecture and module-level walkthrough
```

---

## Key Results Summary

| Model | Dataset | mAP@50 | F1 | Notes |
|-------|---------|--------|-----|-------|
| YOLOv8L (fine-tuned, pretrained) | sample (10k) | 0.767 | -- | Best YOLO on sample set |
| YOLOv8M (fine-tuned, pretrained) | balanced (84k) | 0.592 | -- | Best YOLO on balanced set |
| YOLO-World (zero-shot) | sample (10k) | 0.146 | -- | No fine-tuning |
| FashionNet (original, broken) | sample (10k) | 0.001 | -- | lambda_box bug |
| fashionnet_balanced_v1 | balanced (84k) | 0.193 | 0.359 | Fixed pipeline, 100 epochs |
| edna_1.2m | balanced (84k) | 0.260 | 0.407 | Best edna recall |
| edna_1.3m | balanced (84k) | 0.203 | 0.392 | IoU-obj regression |
| edna_1.4m | balanced + 2k bg (54k) | 0.263 | 0.444 | Best precision (0.477) |
| **edna_1.5m** | balanced + 1k bg (53k) | **0.272** (test) | **0.447** | Best test mAP, recall recovered |

---

## Reading Order

1. `01_dataset/dataset_analysis.md` -- understand the data
2. `02_yolo_experiments/yolo_results.md` -- understand the YOLO baseline
3. `03_fashionnet_experiments/fashionnet_pipeline_fixes.md` -- understand why FashionNet was broken and what was fixed
4. `03_fashionnet_experiments/fashionnet_results.md` -- see the fix impact
5. `04_edna/edna_results.md` -- see the scaled-up custom model results
6. `05_evaluation/evaluation_methodology.md` -- understand the evaluation infrastructure
7. `06_codebase/codebase_explanation.md` -- understand the full system architecture

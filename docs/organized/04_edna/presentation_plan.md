# FashionNet — 5-Minute Technical Presentation Plan

## Overview

Total time: **5 minutes**. Four sections, each centred on one code file.

Narrative arc: **architecture (what) → loss (how it learns) → data pipeline (what it eats) → inference (what comes out)**

---

## Section 1 — Architecture: Built From Scratch, Multi-Scale (1 min 30 s)

**File:** `src/custom_model/model.py`

### 1a — Building blocks (`CSPBlock`, lines 102–119)

Show `CSPBlock.forward()` at lines 116–119.

> "The backbone is built from Cross Stage Partial blocks. Each CSP block splits the feature map into two branches — one goes through a stack of residual blocks, the other skips them entirely. They are concatenated and fused. This halves the computation of the residual path while preserving representational capacity, which is critical when training from scratch without pretrained weights."

### 1b — Full forward pass (`FashionNet.forward`, lines 308–315)

Show the three-line `FashionNet.forward()` method.

> "The full model is three components in sequence: a 4-stage CNN backbone that produces feature maps at strides 8, 16, and 32 (P3, P4, P5), a bidirectional FPN neck that fuses semantic and spatial information across all three scales, and three independent anchor-free detection heads. Each head outputs a tensor of shape (B, 18, gs, gs) — 4 box values, 1 objectness score, and 13 class scores per grid cell. The three scales let us detect small garments at 80×80 and large garments at 20×20."

**Why this matters:** Fully custom architecture. No pretrained backbone. Designed from scratch for 13-class DeepFashion2 detection.

---

## Section 2 — Loss Function: CIoU + Focal BCE (1 min 15 s)

**File:** `src/custom_model/loss.py`

### 2a — CIoU (`bbox_iou`, lines 58–71)

Show the CIoU penalty terms at lines 58–71.

> "We use Complete IoU loss instead of plain IoU. CIoU adds two penalty terms: one penalises the distance between predicted and target box centres, another penalises aspect ratio mismatch. These give much smoother gradients than plain IoU, especially early in training when predicted boxes are far from ground truth — which matters enormously when training from random initialisation."

### 2b — Focal BCE for objectness (`focal_bce`, lines 78–89)

Show the full `focal_bce` function.

> "Around 95% of grid cells are background. Standard BCE lets the model coast by predicting 'no object' everywhere and still get a low loss. Focal loss down-weights easy negatives via the `(1 - p_t)^gamma` term — well-classified background cells contribute almost nothing, forcing the model to focus on hard positives and hard negatives. This is what gets objectness training to converge."

**Why this matters:** The loss function is the main lever for training a from-scratch detector. Bad loss design = zero useful detections.

---

## Section 3 — Data Pipeline: Mosaic Augmentation (1 min)

**File:** `src/custom_model/dataset.py`

### 3 — Mosaic (`_mosaic4`, lines 171–249)

Show lines 186–192 (random centre + canvas creation) and lines 244–246 (area-based box filtering).

> "Mosaic augmentation stitches four training images into a single tile by picking a random dividing point and placing one quadrant from each image. This forces the model to see objects at unusual positions and varying scales in every batch — especially important when there are no pretrained features to rely on. Boxes that lose more than 70% of their area after clipping are discarded, matching the `min_visibility=0.3` threshold used elsewhere in the pipeline."

**Why this matters:** With no pretrained weights, the model relies entirely on the training data. Mosaic is the strongest regulariser in the pipeline — effectively quadrupling scene diversity per batch.

---

## Section 4 — Inference: Grid Decode + NMS (1 min 15 s)

**File:** `src/custom_model/postprocess.py`

### 4 — Decoding and NMS (`decode_predictions` lines 26–107, `nms` lines 110–144)

Show lines 66–69 (sigmoid/clamp decode) and lines 139–143 (class-offset NMS trick).

> "At inference, each grid cell's raw logits are decoded back to normalised coordinates: sigmoid maps centre offsets to (0,1), and width/height are clamped positive. Combined confidence is objectness × max class score — cells below the threshold are dropped immediately. For NMS, we use the class-offset trick: each class's boxes are shifted by `class_id × 10000` so that a single NMS pass never suppresses detections across different garment categories — a shirt should never suppress a nearby pair of trousers."

**Why this matters:** This is where raw model output becomes usable detections. The class-offset trick is a clean engineering detail that shows intentional design.

---

## Timing Summary

| Section | Topic | Time |
|---|---|---|
| 1 | Architecture — CSPBlock + FashionNet forward | 1:30 |
| 2 | Loss — CIoU + focal BCE | 1:15 |
| 3 | Data pipeline — mosaic augmentation | 1:00 |
| 4 | Inference — decode + NMS | 1:15 |
| **Total** | | **5:00** |

---

## Preparation Tips

- **Open all four files before starting.** Do not navigate during the presentation. Use split tabs or a prepared workspace.
- **Zoom in on the specific lines listed above.** The audience must be able to read the code. Do not show the full file.
- **Thread the "from scratch" narrative throughout**: CSP blocks enable deep training without pretraining; CIoU gives gradients from random init; mosaic compensates for no transfer learning; the decode pipeline is fully custom, not borrowed from Ultralytics.
- **If running short on time**: cut Section 3 (mosaic). If running long: shorten Section 1b — just point at the three-line forward method without reading it out.

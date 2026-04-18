# CV Architecture Overview

## Data Pipeline

```
DeepFashion2 raw → converter.py → YOLO format (images/ + labels/)
                                         ↓
                              dataset.py → FashionDataset
                                         ↓
                              collate_fn → batches: (B,3,640,640) images + (N,6) targets
```

**converter.py** (`src/detection/converter.py`): reads `index.json`, converts `[x1,y1,x2,y2]` → YOLO normalised `[cx,cy,w,h]`, splits 85/15 train/val.

**dataset.py** (`src/custom_model/dataset.py`): albumentations pipeline (light/medium/heavy), mosaic4 augmentation (4 images into 1 tile), outputs `(img_tensor, boxes, classes, path)`.

---

## 3 Detector Backends

| Class | File | Method |
|---|---|---|
| `FashionDetector` | `src/detection/detector.py` | Fine-tuned YOLOv8 via Ultralytics |
| `FashionNetDetector` | `src/detection/fashionnet_detector.py` | Custom FashionNet weights |
| `YOLOWorldDetector` | `src/detection/yolo_world.py` | Zero-shot YOLO-World + CLIP |

All inherit `BaseDetector` → all implement `detect(frame: np.ndarray) → DetectionResult`.

---

## FashionNet Architecture (`src/custom_model/model.py`)

```
Input (3×640×640)
    ↓
FashionBackbone  → P3 (80×80), P4 (40×40), P5 (20×20)
    stem: 640→320 (2× ConvBnRelu)
    4 stages: each ConvBnRelu (stride=2) + CSPBlock

FashionNeck (bidirectional FPN)
    top-down:   P5→fuse P4, P4→fuse P3  (upsample + concat + CSP)
    bottom-up:  refine P4, P5 after top-down pass

DetectionHead × 3 (one per scale)
    2× ConvBnRelu → Conv2d(5 + NC)
    outputs: [cx_offset, cy_offset, w, h, obj, class_0..N]
```

Building blocks: `ConvBnRelu` (Conv+BN+LeakyReLU), `ResBlock` (residual), `CSPBlock` (split→blocks→concat, reduces compute).

Model scales:

| Scale | Params | Channels | CSP depths |
|---|---|---|---|
| s | ~11.7M | 64→128→256→512 | 1,2,3,2 |
| m | ~25M | 96→192→384→768 | 2,3,4,3 |
| l | ~43M | 128→256→512→1024 | 3,4,6,3 |

`TinyFashionNet` = stripped version (~400K params) for CPU pipeline testing only.

---

## Loss (`src/custom_model/loss.py`)

```
Total = 5.0 × L_box + 1.0 × L_obj + 0.5 × L_cls
```

- **L_box**: CIoU on positive cells only. CIoU adds center distance + aspect ratio penalty → smoother gradients than plain IoU.
- **L_obj**: focal BCE on ALL cells (γ=1.5). Critical — ~95% cells are background so focal weight down-weights easy negatives.
- **L_cls**: plain BCE on positives only. Multi-label (one image can have multiple garment types).

`build_targets()` assigns GT boxes to grid cells. Normalised `[cx,cy,w,h]` → scaled to grid units, stored as `(cx - cell_i, cy - cell_j, w, h)` (offsets). `multi_cell=True` assigns each GT to center cell + up to 2 adjacent cells → 2-3× more positive signal.

---

## Postprocess (`src/custom_model/postprocess.py`)

```
raw preds (3 tensors) → decode_predictions() → per-image candidates
                                              ↓
                                         nms() (per-class, class-offset trick)
                                              ↓
                                    sort by conf, top max_det
```

Decoding is exact inverse of `build_targets`: `sigmoid(xy) + grid_offset / gs` → normalised coords. Output: `[cx, cy, w, h, conf, class_id]` normalised 0–1.

**Class-offset NMS trick** (`nms()` line 141): `boxes_shifted = boxes + class_id * 10000` so classes never suppress each other.

---

## Inference Path (`FashionNetDetector.detect`)

```
BGR frame
→ RGB + letterbox pad to 640×640 (gray canvas = 114)
→ normalize (ImageNet mean/std)
→ unsqueeze → model(tensor) → 3 raw pred tensors
→ postprocess() → [cx,cy,w,h,conf,cls] normalised
→ unscale: remove pad offset, divide by scale → pixel coords
→ filter degenerate boxes (x2<=x1 or y2<=y1)
→ DetectionResult
```

---

## Camera Pipeline (`src/detection/camera.py`)

State machine: `READY → CAPTURING (5s) → ANALYSING → RESULTS`

During capture: runs detector every ~80ms, **averages confidence per class** across all frames. At end: filters by `conf_thres`, picks top 5 dominant categories, passes to `RecommendationEngine`.

---

## Training Script (`scripts/training/train_custom.py`)

Standard PyTorch loop. Key details:

- **Schedulers**: `OneCycleLR` (default, per-batch) or `CosineAnnealingLR` (per-epoch) + optional linear warmup.
- **ModelEMA**: EMA copy of weights with decay warmup (used for val + inference checkpoints).
- **Checkpoints**: saves `best.pt` (by val_loss), `last.pt`, `history.json`, `config.json` every epoch.
- **Resume**: `--resume path/to/ckpt.pt`.

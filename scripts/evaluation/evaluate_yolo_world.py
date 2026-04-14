"""
scripts/evaluation/evaluate_yolo_world.py
──────────────────────────────
Evaluate the YOLO-World zero-shot detector on the validation set
using the metrics from src/utils/metrics.py.

Usage:
    python scripts/evaluation/evaluate_yolo_world.py
    python scripts/evaluation/evaluate_yolo_world.py --data data/sample_dataset/yolo/dataset.yaml
    python scripts/evaluation/evaluate_yolo_world.py --conf 0.10 --iou 0.50
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import yaml

from src.detection.yolo_world import YOLOWorldDetector
from src.utils.metrics import (
    CATEGORY_NAMES,
    match_predictions,
    per_class_ap,
    detection_report,
)


def load_yolo_labels(label_path: Path, img_w: int, img_h: int):
    """
    Read a YOLO-format label file and convert to absolute [x1,y1,x2,y2].

    Each line: class_id cx cy w h  (all normalised 0-1)
    """
    boxes, classes = [], []
    if not label_path.exists():
        return boxes, classes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        cls = int(parts[0])
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        boxes.append([x1, y1, x2, y2])
        classes.append(cls)
    return boxes, classes


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO-World zero-shot detector")
    parser.add_argument("--data", default="data/sample_dataset/yolo/dataset.yaml",
                        help="Path to dataset.yaml")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Confidence threshold")
    parser.add_argument("--iou",  type=float, default=0.50,
                        help="IoU threshold for AP matching")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    # Resolve val images directory
    ds_root = data_yaml.parent
    val_rel = cfg.get("val", "images/val")
    val_images_dir = ds_root / val_rel
    if not val_images_dir.exists():
        raise FileNotFoundError(f"Validation images dir not found: {val_images_dir}")

    # Corresponding labels directory
    val_labels_dir = Path(str(val_images_dir).replace("/images/", "/labels/"))

    print(f"\n📊  Evaluating YOLO-World (zero-shot)")
    print(f"     conf={args.conf}  iou={args.iou}  imgsz={args.imgsz}")
    print(f"     val images: {val_images_dir}\n")

    detector = YOLOWorldDetector(conf_thres=args.conf, imgsz=args.imgsz)

    all_preds, all_gts = [], []
    img_paths = sorted(
        p for p in val_images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    for i, img_path in enumerate(img_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        # Ground truth
        label_path = val_labels_dir / (img_path.stem + ".txt")
        gt_boxes, gt_classes = load_yolo_labels(label_path, w, h)

        # Prediction
        result = detector.detect(frame)
        pred_boxes   = [d.bbox for d in result.detections]
        pred_classes = [d.class_id for d in result.detections]
        pred_scores  = [d.confidence for d in result.detections]

        all_preds.append({"boxes": pred_boxes, "classes": pred_classes, "scores": pred_scores})
        all_gts.append({"boxes": gt_boxes, "classes": gt_classes})

        if (i + 1) % 50 == 0 or (i + 1) == len(img_paths):
            print(f"  Processed {i + 1}/{len(img_paths)} images")

    # ── Per-class precision / recall / instances ─────────────────────────
    num_classes = len(CATEGORY_NAMES)
    class_tp  = {c: 0 for c in range(num_classes)}
    class_fp  = {c: 0 for c in range(num_classes)}
    class_ngt = {c: 0 for c in range(num_classes)}
    # Count images that contain each class
    class_imgs = {c: 0 for c in range(num_classes)}

    for preds, gts in zip(all_preds, all_gts):
        tp_flags, fn_flags = match_predictions(
            preds["boxes"], preds["classes"], preds["scores"],
            gts["boxes"],   gts["classes"],
            args.iou,
        )
        for i, cls in enumerate(preds["classes"]):
            if 0 <= cls < num_classes:
                if tp_flags[i]:
                    class_tp[cls] += 1
                else:
                    class_fp[cls] += 1
        for cls in gts["classes"]:
            if 0 <= cls < num_classes:
                class_ngt[cls] += 1
        for cls in set(gts["classes"]):
            if 0 <= cls < num_classes:
                class_imgs[cls] += 1

    # ── AP ─────────────────────────────────────────────────────────────
    ap_dict = per_class_ap(all_preds, all_gts, iou_thresh=args.iou)

    # ── Print full table ───────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print(f"  {'Category':<25} {'Images':>6} {'Instances':>10} "
          f"{'Precision':>10} {'Recall':>8} {'mAP@50':>8}")
    print(f"{'─'*90}")

    total_tp = total_fp = total_ngt = 0
    valid_aps = []

    for cls in range(num_classes):
        name = CATEGORY_NAMES[cls]
        tp   = class_tp[cls]
        fp   = class_fp[cls]
        ngt  = class_ngt[cls]
        imgs = class_imgs[cls]
        ap   = ap_dict.get(cls, float("nan"))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / ngt if ngt > 0 else 0.0

        total_tp  += tp
        total_fp  += fp
        total_ngt += ngt
        if not np.isnan(ap):
            valid_aps.append(ap)

        ap_str = f"{ap:.4f}" if not np.isnan(ap) else "N/A"
        print(f"  {name:<25} {imgs:>6} {ngt:>10} "
              f"{prec:>10.3f} {rec:>8.3f} {ap_str:>8}")

    mAP       = float(np.mean(valid_aps)) if valid_aps else 0.0
    total_pre = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_rec = total_tp / total_ngt if total_ngt > 0 else 0.0

    print(f"{'─'*90}")
    print(f"  {'all':<25} {len(all_preds):>6} {total_ngt:>10} "
          f"{total_pre:>10.3f} {total_rec:>8.3f} {mAP:>8.4f}")
    print(f"{'─'*90}")

    print(f"\n   Overall mAP@50  : {mAP:.4f}")
    print(f"   Precision       : {total_pre:.4f}")
    print(f"   Recall          : {total_rec:.4f}")
    print(f"   Images evaluated: {len(all_preds)}\n")


if __name__ == "__main__":
    main()

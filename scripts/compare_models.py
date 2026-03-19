"""
scripts/compare_models.py
──────────────────────────
Evaluates both FashionNet (custom) and YOLOv8 (fine-tuned) on the
same validation set and produces a side-by-side comparison report.

Metrics compared:
  - mAP@50 per class and overall
  - Inference speed (ms/image, FPS)
  - Parameter count
  - Model file size

Usage:
    python scripts/compare_models.py \
        --yolo_weights models/weights/yolov8n_fashion/weights/best.pt \
        --custom_weights models/weights/fashionnet/best.pt \
        --data data/sample_dataset/yolo
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.custom_model.model   import FashionNet
from src.custom_model.dataset import FashionDataset, get_val_transforms, collate_fn
from src.utils.metrics        import iou, detection_report, per_class_ap

CATEGORY_NAMES = [
    "short_sleeve_top", "long_sleeve_top", "short_sleeve_outwear",
    "long_sleeve_outwear", "vest", "sling", "shorts", "trousers",
    "skirt", "short_sleeve_dress", "long_sleeve_dress",
    "vest_dress", "sling_dress",
]


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def decode_fashionnet_output(
    preds: list,
    conf_thresh: float = 0.25,
    nms_thresh:  float = 0.45,
    img_size:    int   = 640,
):
    """
    Convert FashionNet raw output tensors to a list of detections per image.
    Returns list of dicts: {boxes: [[x1,y1,x2,y2],...], scores: [...], classes: [...]}
    """
    device     = preds[0].device
    strides    = [img_size // p.shape[-1] for p in preds]
    B          = preds[0].shape[0]
    all_dets   = [{"boxes": [], "scores": [], "classes": []} for _ in range(B)]

    for pred, stride in zip(preds, strides):
        B, C, gs, _ = pred.shape
        p = pred.permute(0, 2, 3, 1)   # (B, gs, gs, 5+NC)

        p_xy  = torch.sigmoid(p[..., :2])
        p_wh  = p[..., 2:4] * stride
        p_obj = torch.sigmoid(p[..., 4])
        p_cls = torch.sigmoid(p[..., 5:])

        # Build grid offsets
        gy, gx = torch.meshgrid(
            torch.arange(gs, device=device, dtype=torch.float32),
            torch.arange(gs, device=device, dtype=torch.float32),
            indexing='ij'
        )

        for bi in range(B):
            obj  = p_obj[bi]   # (gs, gs)
            mask = obj > conf_thresh
            if not mask.any():
                continue

            cx = (p_xy[bi, ..., 0][mask] + gx[mask]) * stride
            cy = (p_xy[bi, ..., 1][mask] + gy[mask]) * stride
            w  = p_wh[bi, ..., 0][mask]
            h  = p_wh[bi, ..., 1][mask]

            scores_cls, classes = p_cls[bi][mask].max(dim=-1)
            scores = obj[mask] * scores_cls

            keep = scores > conf_thresh
            if not keep.any():
                continue

            boxes = torch.stack([
                cx[keep] - w[keep]/2,
                cy[keep] - h[keep]/2,
                cx[keep] + w[keep]/2,
                cy[keep] + h[keep]/2,
            ], dim=1)

            all_dets[bi]["boxes"].extend(boxes.cpu().tolist())
            all_dets[bi]["scores"].extend(scores[keep].cpu().tolist())
            all_dets[bi]["classes"].extend(classes[keep].cpu().tolist())

    return all_dets


def benchmark_speed(model_fn, device, n_runs=100, img_size=640):
    """Measure average inference time in ms over n_runs."""
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    # Warmup
    for _ in range(10):
        model_fn(dummy)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model_fn(dummy)
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fashionnet(weights_path, data_dir, device, img_size=640, batch=8):
    """Run FashionNet on validation set, collect predictions and ground truths."""
    model = FashionNet()
    ckpt  = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    val_ds = FashionDataset(data_dir, "val", img_size, get_val_transforms(img_size))
    val_dl = DataLoader(val_ds, batch_size=batch, collate_fn=collate_fn, num_workers=0)

    all_preds, all_gts = [], []

    with torch.no_grad():
        for images, targets in tqdm(val_dl, desc="FashionNet eval"):
            images = images.to(device)
            B      = images.shape[0]
            preds  = model(images)
            dets   = decode_fashionnet_output(preds, img_size=img_size)

            for bi in range(B):
                # Predictions
                all_preds.append({
                    "boxes":   dets[bi]["boxes"],
                    "classes": dets[bi]["classes"],
                    "scores":  dets[bi]["scores"],
                })
                # Ground truths for this image
                gt_mask = (targets[:, 0] == bi)
                gt_t    = targets[gt_mask]
                gt_boxes, gt_classes = [], []
                for row in gt_t:
                    _, cls, cx, cy, w, h = row.tolist()
                    # Convert normalised → pixel
                    x1 = (cx - w/2) * img_size
                    y1 = (cy - h/2) * img_size
                    x2 = (cx + w/2) * img_size
                    y2 = (cy + h/2) * img_size
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_classes.append(int(cls))
                all_gts.append({"boxes": gt_boxes, "classes": gt_classes})

    # Speed
    mean_ms, std_ms = benchmark_speed(
        lambda x: model(x), device, n_runs=50, img_size=img_size
    )

    ap_dict = per_class_ap(all_preds, all_gts, iou_thresh=0.50)

    return {
        "ap_per_class": ap_dict,
        "map50":        float(np.nanmean(list(ap_dict.values()))),
        "mean_ms":      round(mean_ms, 2),
        "std_ms":       round(std_ms, 2),
        "fps":          round(1000 / mean_ms, 1),
        "params":       sum(p.numel() for p in model.parameters()),
        "size_mb":      round(Path(weights_path).stat().st_size / 1e6, 2),
    }


def evaluate_yolo(weights_path, data_dir, device_str, img_size=640):
    """Run YOLOv8 validation and collect metrics."""
    model = YOLO(weights_path)
    yaml  = str(Path(data_dir) / "dataset.yaml")

    results  = model.val(data=yaml, imgsz=img_size, verbose=False)
    map50    = float(results.box.map50)
    ap_list  = results.box.ap50.tolist() if hasattr(results.box.ap50, 'tolist') else []
    ap_dict  = {i: float(ap_list[i]) if i < len(ap_list) else float('nan')
                for i in range(13)}

    # Speed benchmark
    device = torch.device(device_str if device_str else "cpu")
    dummy  = torch.zeros(1, 3, img_size, img_size)
    def yolo_fn(x):
        model.predict(x.numpy()[0].transpose(1,2,0).astype(np.uint8),
                      verbose=False)
    mean_ms, std_ms = benchmark_speed(yolo_fn, torch.device('cpu'), n_runs=30)

    return {
        "ap_per_class": ap_dict,
        "map50":        map50,
        "mean_ms":      round(mean_ms, 2),
        "std_ms":       round(std_ms, 2),
        "fps":          round(1000 / max(mean_ms, 1), 1),
        "params":       sum(p.numel() for p in model.model.parameters()),
        "size_mb":      round(Path(weights_path).stat().st_size / 1e6, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(custom: dict, yolo: dict):
    print("\n" + "═"*70)
    print("  MODEL COMPARISON REPORT")
    print("═"*70)

    print(f"\n{'Metric':<30} {'FashionNet (custom)':>22} {'YOLOv8n':>12}")
    print("─"*70)

    def row(label, c_val, y_val, fmt=".4f", higher_better=True):
        c_str = format(c_val, fmt) if isinstance(c_val, float) else str(c_val)
        y_str = format(y_val, fmt) if isinstance(y_val, float) else str(y_val)
        if higher_better:
            winner = "✅" if c_val >= y_val else "  "
        else:
            winner = "✅" if c_val <= y_val else "  "
        print(f"  {label:<28} {c_str:>22} {y_str:>12}  {winner}")

    row("mAP@50 (overall)",     custom["map50"],    yolo["map50"])
    row("Inference (ms/img)",   custom["mean_ms"],  yolo["mean_ms"],   fmt=".1f", higher_better=False)
    row("FPS",                  custom["fps"],      yolo["fps"],       fmt=".1f")
    row("Parameters (M)",       custom["params"]/1e6, yolo["params"]/1e6, fmt=".2f", higher_better=False)
    row("Weights size (MB)",    custom["size_mb"],  yolo["size_mb"],   fmt=".1f", higher_better=False)

    print("\n── Per-class mAP@50 ──────────────────────────────────────────────")
    print(f"  {'Category':<30} {'FashionNet':>12} {'YOLOv8n':>12}  {'Better':>8}")
    print("  " + "─"*66)

    for i, name in enumerate(CATEGORY_NAMES):
        c_ap = custom["ap_per_class"].get(i, float('nan'))
        y_ap = yolo["ap_per_class"].get(i, float('nan'))
        if np.isnan(c_ap) and np.isnan(y_ap):
            continue
        c_str  = f"{c_ap:.4f}" if not np.isnan(c_ap) else "  N/A"
        y_str  = f"{y_ap:.4f}" if not np.isnan(y_ap) else "  N/A"
        winner = "FashionNet" if (not np.isnan(c_ap) and not np.isnan(y_ap) and c_ap > y_ap) \
                 else "YOLOv8n" if not np.isnan(y_ap) else ""
        print(f"  {name:<30} {c_str:>12} {y_str:>12}  {winner:>8}")

    print("\n" + "═"*70)
    diff = custom["map50"] - yolo["map50"]
    if diff < 0:
        print(f"  YOLOv8n outperforms FashionNet by {abs(diff):.4f} mAP@50")
        print(f"  This gap reflects pretrained COCO weights, architectural")
        print(f"  optimisations, and years of engineering in Ultralytics.")
    else:
        print(f"  FashionNet outperforms YOLOv8n by {diff:.4f} mAP@50 🎉")
    print("═"*70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_weights",   default="models/weights/yolov8n_fashion/weights/best.pt")
    parser.add_argument("--custom_weights", default="models/weights/fashionnet/best.pt")
    parser.add_argument("--data",           default="data/sample_dataset/yolo")
    parser.add_argument("--imgsz",          type=int, default=640)
    parser.add_argument("--batch",          type=int, default=8)
    parser.add_argument("--device",         default="")
    parser.add_argument("--out",            default="docs/comparison.json")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else "cpu")
    print(f"\nDevice: {device}")

    print("\n[1/2] Evaluating FashionNet (custom)...")
    custom_results = evaluate_fashionnet(
        args.custom_weights, args.data, device, args.imgsz, args.batch
    )

    print("\n[2/2] Evaluating YOLOv8n...")
    yolo_results = evaluate_yolo(
        args.yolo_weights, args.data, args.device, args.imgsz
    )

    print_comparison(custom_results, yolo_results)

    # Save JSON for notebook / thesis tables
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"fashionnet": custom_results, "yolov8n": yolo_results}, f, indent=2)
    print(f"Results saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
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

import yaml
import cv2
import numpy as np
import torch
import torchvision.ops as tv_ops
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.custom_model.model   import FashionNet
from src.custom_model.dataset import FashionDataset, get_val_transforms, collate_fn
from src.utils.metrics        import iou, detection_report, per_class_ap


def load_category_names(data_dir: str) -> list:
    yaml_path = Path(data_dir) / "dataset.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["names"]


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def decode_fashionnet_output(
    preds: list,
    conf_thresh: float = 0.001,
    nms_thresh:  float = 0.45,
    img_size:    int   = 640,
    max_det:     int   = 300,
):
    """
    Convert FashionNet raw output tensors to a list of detections per image.
    Accumulates candidates across all 3 scales, keeps top-max_det by score,
    then applies per-class NMS.
    Returns list of dicts: {boxes: [[x1,y1,x2,y2],...], scores: [...], classes: [...]}
    """
    device  = preds[0].device
    strides = [img_size // p.shape[-1] for p in preds]
    B       = preds[0].shape[0]

    # Accumulate raw candidates across scales: one list per image
    raw_boxes   = [[] for _ in range(B)]
    raw_scores  = [[] for _ in range(B)]
    raw_classes = [[] for _ in range(B)]

    for pred, stride in zip(preds, strides):
        _, C, gs, _ = pred.shape
        p = pred.permute(0, 2, 3, 1)   # (B, gs, gs, 5+NC)

        p_xy  = torch.sigmoid(p[..., :2])
        p_wh  = p[..., 2:4].abs() * stride
        p_obj = torch.sigmoid(p[..., 4])
        p_cls = torch.sigmoid(p[..., 5:])

        gy, gx = torch.meshgrid(
            torch.arange(gs, device=device, dtype=torch.float32),
            torch.arange(gs, device=device, dtype=torch.float32),
            indexing='ij',
        )

        for bi in range(B):
            mask = p_obj[bi] > conf_thresh
            if not mask.any():
                continue

            cx = (p_xy[bi, ..., 0][mask] + gx[mask]) * stride
            cy = (p_xy[bi, ..., 1][mask] + gy[mask]) * stride
            w  = p_wh[bi, ..., 0][mask]
            h  = p_wh[bi, ..., 1][mask]

            scores_cls, classes = p_cls[bi][mask].max(dim=-1)
            scores = p_obj[bi][mask] * scores_cls

            keep = scores > conf_thresh
            if not keep.any():
                continue

            boxes = torch.stack([
                cx[keep] - w[keep] / 2,
                cy[keep] - h[keep] / 2,
                cx[keep] + w[keep] / 2,
                cy[keep] + h[keep] / 2,
            ], dim=1)

            raw_boxes[bi].append(boxes)
            raw_scores[bi].append(scores[keep])
            raw_classes[bi].append(classes[keep])

    # Per-image: top-K cap → per-class NMS
    all_dets = []
    for bi in range(B):
        if not raw_scores[bi]:
            all_dets.append({"boxes": [], "scores": [], "classes": []})
            continue

        boxes_t   = torch.cat(raw_boxes[bi],   dim=0)   # (N, 4)
        scores_t  = torch.cat(raw_scores[bi],  dim=0)   # (N,)
        classes_t = torch.cat(raw_classes[bi], dim=0)   # (N,)

        # Keep top max_det candidates before NMS
        if scores_t.shape[0] > max_det:
            topk = scores_t.topk(max_det).indices
            boxes_t, scores_t, classes_t = boxes_t[topk], scores_t[topk], classes_t[topk]

        # Per-class NMS
        kept_boxes, kept_scores, kept_classes = [], [], []
        for cls_id in classes_t.unique():
            m = classes_t == cls_id
            keep_idx = tv_ops.nms(boxes_t[m], scores_t[m], nms_thresh)
            kept_boxes.append(boxes_t[m][keep_idx])
            kept_scores.append(scores_t[m][keep_idx])
            kept_classes.extend([cls_id.item()] * keep_idx.shape[0])

        if kept_boxes:
            all_dets.append({
                "boxes":   torch.cat(kept_boxes).cpu().tolist(),
                "scores":  torch.cat(kept_scores).cpu().tolist(),
                "classes": kept_classes,
            })
        else:
            all_dets.append({"boxes": [], "scores": [], "classes": []})

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
    ckpt = torch.load(weights_path, map_location=device)

    # Resolve num_classes and model_scale: prefer config.json, then infer from checkpoint
    config_path = Path(weights_path).parent / "config.json"
    scale = "s"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        num_classes = cfg.get("num_classes_resolved", 13)
        scale = cfg.get("model_scale", "s")
    else:
        head_weight = ckpt['model'].get('head_p3.pred.weight')
        if head_weight is not None:
            num_classes = head_weight.shape[0] - 5
        else:
            num_classes = len(load_category_names(data_dir))

    model = FashionNet(num_classes=num_classes, scale=scale)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    grayscale = config_path.exists() and json.load(open(config_path)).get("grayscale", False)
    val_ds = FashionDataset(data_dir, "val", img_size, get_val_transforms(img_size, grayscale=grayscale))
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

    # ── Diagnostics ───────────────────────────────────────────────────
    total_dets = sum(len(p["boxes"]) for p in all_preds)
    total_gts  = sum(len(g["boxes"]) for g in all_gts)
    print(f"\n  [DEBUG] Total detections produced: {total_dets}")
    print(f"  [DEBUG] Total ground-truth boxes:  {total_gts}")

    if total_dets > 0:
        all_scores = [s for p in all_preds for s in p["scores"]]
        all_cls    = [c for p in all_preds for c in p["classes"]]
        all_widths = [abs(b[2] - b[0]) for p in all_preds for b in p["boxes"]]
        all_heights= [abs(b[3] - b[1]) for p in all_preds for b in p["boxes"]]
        neg_w = sum(1 for p in all_preds for b in p["boxes"] if b[2] < b[0])
        neg_h = sum(1 for p in all_preds for b in p["boxes"] if b[3] < b[1])

        print(f"  [DEBUG] Score  range: [{min(all_scores):.4f}, {max(all_scores):.4f}]  "
              f"mean={np.mean(all_scores):.4f}")
        print(f"  [DEBUG] Width  range: [{min(all_widths):.1f}, {max(all_widths):.1f}]  "
              f"mean={np.mean(all_widths):.1f}")
        print(f"  [DEBUG] Height range: [{min(all_heights):.1f}, {max(all_heights):.1f}]  "
              f"mean={np.mean(all_heights):.1f}")
        print(f"  [DEBUG] Inverted boxes (x1>x2): {neg_w},  (y1>y2): {neg_h}")
        print(f"  [DEBUG] Classes predicted: {sorted(set(all_cls))}")
    else:
        # Check raw objectness scores to understand why nothing passed
        print("  [DEBUG] No detections! Checking raw objectness scores...")
        model.eval()
        with torch.no_grad():
            sample_imgs = next(iter(val_dl))[0][:1].to(device)
            raw_preds = model(sample_imgs)
            for i, rp in enumerate(raw_preds):
                p = rp.permute(0, 2, 3, 1)
                obj = torch.sigmoid(p[..., 4])
                print(f"    Scale {i} objectness: min={obj.min():.4f}  "
                      f"max={obj.max():.4f}  mean={obj.mean():.4f}  "
                      f">{0.25}: {(obj > 0.25).sum().item()} cells")
    print()

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
    nc       = len(load_category_names(data_dir))
    map50    = float(results.box.map50)
    ap_list  = results.box.ap50.tolist() if hasattr(results.box.ap50, 'tolist') else []
    ap_dict  = {i: float(ap_list[i]) if i < len(ap_list) else float('nan')
                for i in range(nc)}

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

def print_comparison(custom: dict, yolo: dict, custom_name: str, yolo_name: str,
                     category_names: list):
    print("\n" + "═"*70)
    print("  MODEL COMPARISON REPORT")
    print("═"*70)

    print(f"\n{'Metric':<30} {custom_name:>22} {yolo_name:>12}")
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
    print(f"  {'Category':<30} {custom_name:>12} {yolo_name:>12}  {'Better':>8}")
    print("  " + "─"*66)

    # Build a name lookup covering both models (one may have more classes)
    all_class_ids = sorted(set(custom["ap_per_class"].keys()) | set(yolo["ap_per_class"].keys()))
    for i in all_class_ids:
        name = category_names[i] if i < len(category_names) else f"class_{i}"
        c_ap = custom["ap_per_class"].get(i, float('nan'))
        y_ap = yolo["ap_per_class"].get(i, float('nan'))
        if np.isnan(c_ap) and np.isnan(y_ap):
            continue
        c_str  = f"{c_ap:.4f}" if not np.isnan(c_ap) else "   N/A"
        y_str  = f"{y_ap:.4f}" if not np.isnan(y_ap) else "   N/A"
        if not np.isnan(c_ap) and not np.isnan(y_ap) and c_ap != y_ap:
            winner = custom_name if c_ap > y_ap else yolo_name
        else:
            winner = ""
        print(f"  {name:<30} {c_str:>12} {y_str:>12}  {winner:>8}")

    print("\n" + "═"*70)
    diff = custom["map50"] - yolo["map50"]
    winner   = custom_name if diff >= 0 else yolo_name
    loser    = yolo_name   if diff >= 0 else custom_name
    margin   = abs(diff)
    print(f"  {winner} outperforms {loser} by {margin:.4f} mAP@50")
    print("═"*70 + "\n")


def is_fashionnet_checkpoint(path: str) -> bool:
    """Detect if a weights file is a FashionNet checkpoint (has 'model' key)."""
    try:
        ckpt = torch.load(path, map_location="cpu")
        return isinstance(ckpt, dict) and "model" in ckpt
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_weights",   default="models/weights/yolov8n_fashion/weights/best.pt",
                        help="YOLOv8 weights OR a previous FashionNet checkpoint for exp-vs-exp comparison")
    parser.add_argument("--custom_weights", default="models/weights/fashionnet/best.pt")
    parser.add_argument("--data",           default="data/sample_dataset/yolo")
    parser.add_argument("--imgsz",          type=int, default=640)
    parser.add_argument("--batch",          type=int, default=8)
    parser.add_argument("--device",         default="")
    parser.add_argument("--out",            default="docs/comparison.json")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    category_names = load_category_names(args.data)
    print(f"Classes ({len(category_names)}): {', '.join(category_names)}")

    # Derive display names from weight paths: pick first parent that isn't "weights"
    def _model_name(p):
        for part in reversed(Path(p).parts[:-1]):
            if part.lower() != "weights":
                return part
        return Path(p).stem

    custom_name = _model_name(args.custom_weights)
    ref_name    = _model_name(args.yolo_weights)

    print(f"\n[1/2] Evaluating {custom_name}...")
    custom_results = evaluate_fashionnet(
        args.custom_weights, args.data, device, args.imgsz, args.batch
    )

    print(f"\n[2/2] Evaluating {ref_name}...")
    if is_fashionnet_checkpoint(args.yolo_weights):
        ref_results = evaluate_fashionnet(
            args.yolo_weights, args.data, device, args.imgsz, args.batch
        )
    else:
        ref_results = evaluate_yolo(
            args.yolo_weights, args.data, args.device, args.imgsz
        )

    print_comparison(custom_results, ref_results, custom_name, ref_name,
                     category_names)

    # Save JSON for notebook / thesis tables
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({custom_name: custom_results, ref_name: ref_results}, f, indent=2)
    print(f"Results saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
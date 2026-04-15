"""
scripts/evaluation/evaluate_custom.py
──────────────────────────
Standalone evaluation script for FashionNet checkpoints.

Computes mAP@50, F1, precision, recall, per-class metrics, and a
(NC+1)x(NC+1) confusion matrix (with background row/column for FP/FN).

Outputs metrics.json alongside the weights (or to --output_dir).

Usage:
    python scripts/evaluation/evaluate_custom.py --weights models/weights/fashionnet/best.pt
    python scripts/evaluation/evaluate_custom.py --weights models/weights/fashionnet/best.pt --split test --conf 0.25
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.custom_model.model import FashionNet, TinyFashionNet
from src.custom_model.dataset import (
    FashionDataset, collate_fn, get_val_transforms, CATEGORY_NAMES,
)
from src.custom_model.postprocess import postprocess
from src.utils.metrics import per_class_ap, iou


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a FashionNet checkpoint")
    p.add_argument("--weights",     required=True, help="Path to .pt checkpoint")
    p.add_argument("--data",        default="data/balanced_dataset",
                   help="Path to yolo/ directory")
    p.add_argument("--split",       default="val", choices=["val", "test"])
    p.add_argument("--imgsz",       type=int, default=640)
    p.add_argument("--conf",        type=float, default=0.25,
                   help="Confidence threshold")
    p.add_argument("--iou_thresh",  type=float, default=0.45,
                   help="NMS IoU threshold")
    p.add_argument("--batch",       type=int, default=16)
    p.add_argument("--device",      default="",
                   help="cpu / cuda / mps / auto")
    p.add_argument("--output_dir",  default="",
                   help="Where to save metrics.json (default: same dir as weights)")
    p.add_argument("--num_classes",  type=int, default=0,
                   help="0 = auto-read from config.json / dataset.yaml")
    return p.parse_args()


def pick_device(requested: str) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(weights_path: str, device: torch.device, num_classes_override: int = 0):
    """Load FashionNet from checkpoint, resolving num_classes and model type."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    weights_dir = Path(weights_path).parent
    config_path = weights_dir / "config.json"

    # Resolve num_classes
    num_classes = num_classes_override
    is_fast = False
    grayscale = False

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if not num_classes:
            num_classes = config.get("num_classes_resolved", 13)
        is_fast = config.get("fast", False)
        grayscale = config.get("grayscale", False)
    elif not num_classes:
        # Infer from checkpoint head weight shape
        head_weight = ckpt["model"].get("head_p3.pred.weight")
        if head_weight is not None:
            num_classes = head_weight.shape[0] - 5
        else:
            num_classes = 13

    # Instantiate model
    if is_fast:
        model = TinyFashionNet(num_classes=num_classes)
    else:
        dropout = 0.0
        scale = "s"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            dropout = cfg.get("dropout", 0.0)
            scale = cfg.get("model_scale", "s")
        model = FashionNet(num_classes=num_classes, dropout=dropout, scale=scale)

    # Load weights (prefer EMA if available)
    if "ema" in ckpt and ckpt["ema"]:
        model.load_state_dict(ckpt["ema"])
        print("  Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])

    model.to(device).eval()
    return model, num_classes, grayscale


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix helpers (class-agnostic matching)
# ─────────────────────────────────────────────────────────────────────────────

def build_detection_confusion_matrix(
    all_preds, all_gts, num_classes, iou_thresh=0.50,
):
    """
    Build (NC+1) x (NC+1) confusion matrix using class-agnostic IoU matching.

    Last row/col index = background.
    - Matched pred:   cm[gt_class, pred_class] += 1
    - Unmatched pred:  cm[background, pred_class] += 1   (FP)
    - Unmatched GT:    cm[gt_class, background] += 1     (FN)
    """
    bg = num_classes  # background index
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for preds, gts in zip(all_preds, all_gts):
        pred_boxes = preds["boxes"]
        pred_classes = preds["classes"]
        pred_scores = preds["scores"]
        gt_boxes = gts["boxes"]
        gt_classes = gts["classes"]

        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)

        if n_gt == 0 and n_pred == 0:
            continue

        if n_gt == 0:
            # All predictions are FP
            for pc in pred_classes:
                cm[bg, pc] += 1
            continue

        if n_pred == 0:
            # All GT are FN
            for gc in gt_classes:
                cm[gc, bg] += 1
            continue

        # Compute IoU matrix (n_pred x n_gt) — class-agnostic
        iou_matrix = np.zeros((n_pred, n_gt))
        for pi in range(n_pred):
            for gi in range(n_gt):
                iou_matrix[pi, gi] = iou(pred_boxes[pi], gt_boxes[gi])

        # Greedy matching by descending prediction confidence
        matched_gt = set()
        matched_pred = set()
        order = sorted(range(n_pred), key=lambda i: pred_scores[i], reverse=True)

        for pi in order:
            best_iou = iou_thresh - 1e-9
            best_gi = -1
            for gi in range(n_gt):
                if gi in matched_gt:
                    continue
                if iou_matrix[pi, gi] > best_iou:
                    best_iou = iou_matrix[pi, gi]
                    best_gi = gi
            if best_gi >= 0:
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                cm[gt_classes[best_gi], pred_classes[pi]] += 1

        # Unmatched predictions → FP
        for pi in range(n_pred):
            if pi not in matched_pred:
                cm[bg, pred_classes[pi]] += 1

        # Unmatched GT → FN
        for gi in range(n_gt):
            if gi not in matched_gt:
                cm[gt_classes[gi], bg] += 1

    return cm


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"\n  Device: {device}")

    # Load model
    model, num_classes, grayscale = load_model(
        args.weights, device, args.num_classes,
    )
    print(f"  Classes: {num_classes}")
    print(f"  Conf threshold: {args.conf}")
    print(f"  NMS IoU threshold: {args.iou_thresh}")

    # Resolve class names
    data_yaml = Path(args.data) / "dataset.yaml"
    if data_yaml.exists():
        with open(data_yaml) as f:
            ds_cfg = yaml.safe_load(f)
        class_names = ds_cfg.get("names", CATEGORY_NAMES[:num_classes])
    else:
        class_names = CATEGORY_NAMES[:num_classes]

    # Build dataloader
    val_ds = FashionDataset(
        args.data, args.split, args.imgsz,
        transforms=get_val_transforms(args.imgsz, grayscale=grayscale),
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── Inference ─────────────────────────────────────────────────────────
    all_preds, all_gts = [], []

    with torch.no_grad():
        for images, targets in tqdm(val_dl, desc=f"Evaluating ({args.split})"):
            images = images.to(device)
            B = images.shape[0]
            raw_preds = model(images)

            dets_batch = postprocess(
                raw_preds,
                img_size=args.imgsz,
                conf_thresh=args.conf,
                iou_thresh=args.iou_thresh,
                num_classes=num_classes,
                max_det=300,
            )

            for bi in range(B):
                dets = dets_batch[bi]

                # Convert normalised (cx,cy,w,h) → pixel (x1,y1,x2,y2)
                if dets.shape[0] > 0:
                    cx = dets[:, 0] * args.imgsz
                    cy = dets[:, 1] * args.imgsz
                    w = dets[:, 2] * args.imgsz
                    h = dets[:, 3] * args.imgsz
                    pred_boxes = torch.stack([
                        cx - w / 2, cy - h / 2,
                        cx + w / 2, cy + h / 2,
                    ], dim=1).cpu().tolist()
                    pred_scores = dets[:, 4].cpu().tolist()
                    pred_classes = dets[:, 5].int().cpu().tolist()
                else:
                    pred_boxes, pred_scores, pred_classes = [], [], []

                all_preds.append({
                    "boxes": pred_boxes,
                    "classes": pred_classes,
                    "scores": pred_scores,
                })

                # Ground truths for this image
                gt_mask = targets[:, 0] == bi
                gt_t = targets[gt_mask]
                gt_boxes, gt_classes = [], []
                for row in gt_t:
                    _, cls, cx, cy, w, h = row.tolist()
                    x1 = (cx - w / 2) * args.imgsz
                    y1 = (cy - h / 2) * args.imgsz
                    x2 = (cx + w / 2) * args.imgsz
                    y2 = (cy + h / 2) * args.imgsz
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_classes.append(int(cls))
                all_gts.append({"boxes": gt_boxes, "classes": gt_classes})

    # ── Compute metrics ───────────────────────────────────────────────────
    total_dets = sum(len(p["boxes"]) for p in all_preds)
    total_gts = sum(len(g["boxes"]) for g in all_gts)
    print(f"\n  Total detections: {total_dets}")
    print(f"  Total ground truths: {total_gts}")

    # Per-class AP
    ap_dict = per_class_ap(all_preds, all_gts, iou_thresh=0.50, num_classes=num_classes)
    valid_aps = [v for v in ap_dict.values() if not np.isnan(v)]
    mAP50 = float(np.mean(valid_aps)) if valid_aps else 0.0

    # Per-class precision, recall, F1 from accumulated TP/FP/FN
    class_tp = {c: 0 for c in range(num_classes)}
    class_fp = {c: 0 for c in range(num_classes)}
    class_fn = {c: 0 for c in range(num_classes)}

    from src.utils.metrics import match_predictions
    for preds, gts in zip(all_preds, all_gts):
        tp_flags, fn_flags = match_predictions(
            preds["boxes"], preds["classes"], preds["scores"],
            gts["boxes"], gts["classes"], iou_thresh=0.50,
        )
        for i, cls in enumerate(preds["classes"]):
            if cls < num_classes:
                if tp_flags[i]:
                    class_tp[cls] += 1
                else:
                    class_fp[cls] += 1
        for i, cls in enumerate(gts["classes"]):
            if cls < num_classes and fn_flags[i]:
                class_fn[cls] += 1

    # Aggregate
    per_class_metrics = {}
    total_tp = total_fp = total_fn = 0

    for c in range(num_classes):
        tp, fp, fn = class_tp[c], class_fp[c], class_fn[c]
        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        ap = ap_dict.get(c, float("nan"))

        name = class_names[c] if c < len(class_names) else f"class_{c}"
        per_class_metrics[name] = {
            "AP": round(ap, 4) if not np.isnan(ap) else None,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "F1": round(f1, 4),
        }

    # Macro averages
    macro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    macro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec) if (macro_prec + macro_rec) > 0 else 0.0

    # Confusion matrix (class-agnostic matching)
    cm = build_detection_confusion_matrix(all_preds, all_gts, num_classes, iou_thresh=0.50)

    # ── Build output ──────────────────────────────────────────────────────
    metrics = {
        "mAP50": round(mAP50, 4),
        "precision": round(macro_prec, 4),
        "recall": round(macro_rec, 4),
        "F1": round(macro_f1, 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names[:num_classes]) + ["background"],
        "config": {
            "weights": str(args.weights),
            "conf_thresh": args.conf,
            "iou_thresh": args.iou_thresh,
            "split": args.split,
            "imgsz": args.imgsz,
            "num_classes": num_classes,
            "total_images": len(all_preds),
            "total_detections": total_dets,
            "total_ground_truths": total_gts,
        },
    }

    # ── Save ──────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.weights).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FashionNet Evaluation Results ({args.split})")
    print(f"{'='*60}")
    print(f"  mAP@50    : {mAP50:.4f}")
    print(f"  Precision : {macro_prec:.4f}")
    print(f"  Recall    : {macro_rec:.4f}")
    print(f"  F1        : {macro_f1:.4f}")
    print(f"{'─'*60}")

    for name, m in per_class_metrics.items():
        ap_str = f"{m['AP']:.4f}" if m["AP"] is not None else "  N/A"
        print(f"  {name:<28}  AP={ap_str}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['F1']:.3f}")

    print(f"{'─'*60}")
    print(f"  Saved: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

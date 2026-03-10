"""
src/utils/metrics.py
────────────────────
Evaluation helpers for the FashionSense detection system.

Provides:
  - per_class_ap()        compute AP per class from raw predictions
  - confusion_matrix()    build and plot a confusion matrix
  - detection_report()    pretty-print a classification-style report
  - iou()                 intersection-over-union for two boxes
  - match_predictions()   match predicted boxes to ground-truth boxes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional — only needed for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

CATEGORY_NAMES = [
    "short_sleeve_top", "long_sleeve_top", "short_sleeve_outwear",
    "long_sleeve_outwear", "vest", "sling", "shorts", "trousers",
    "skirt", "short_sleeve_dress", "long_sleeve_dress",
    "vest_dress", "sling_dress",
]


# ── Box utilities ──────────────────────────────────────────────────────────

def iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Compute Intersection-over-Union for two boxes in [x1,y1,x2,y2] format.

    Parameters
    ----------
    boxA, boxB : [x1, y1, x2, y2]

    Returns
    -------
    float in [0, 1]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def match_predictions(
    pred_boxes:   List[List[float]],
    pred_classes: List[int],
    pred_scores:  List[float],
    gt_boxes:     List[List[float]],
    gt_classes:   List[int],
    iou_thresh:   float = 0.50,
) -> Tuple[List[bool], List[bool]]:
    """
    Greedy matching of predictions to ground-truth boxes.

    Returns
    -------
    tp : List[bool]  — True-positive flag per prediction
    fn : List[bool]  — False-negative flag per ground-truth
    """
    matched_gt = [False] * len(gt_boxes)
    tp         = [False] * len(pred_boxes)

    # Sort predictions by descending confidence
    order = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)

    for pi in order:
        best_iou = iou_thresh - 1e-9
        best_gi  = -1
        for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
            if matched_gt[gi]:
                continue
            if gc != pred_classes[pi]:
                continue
            v = iou(pred_boxes[pi], gb)
            if v > best_iou:
                best_iou = v
                best_gi  = gi
        if best_gi >= 0:
            tp[pi]          = True
            matched_gt[best_gi] = True

    fn = [not m for m in matched_gt]
    return tp, fn


# ── Average Precision ──────────────────────────────────────────────────────

def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """VOC 2010+ 101-point interpolated AP."""
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p / 101
    return float(ap)


def per_class_ap(
    all_preds: List[Dict],
    all_gts:   List[Dict],
    iou_thresh: float = 0.50,
    num_classes: int  = 13,
) -> Dict[int, float]:
    """
    Compute per-class Average Precision.

    Parameters
    ----------
    all_preds : list of dicts per image:
        { 'boxes': [[x1,y1,x2,y2],...], 'classes': [int,...], 'scores': [float,...] }
    all_gts : list of dicts per image:
        { 'boxes': [[x1,y1,x2,y2],...], 'classes': [int,...] }
    iou_thresh : IoU threshold for a match (default 0.5)
    num_classes : number of classes

    Returns
    -------
    Dict[class_id → AP]
    """
    # Accumulate TP/FP/n_gt per class
    class_tp:  Dict[int, List[Tuple[float, bool]]] = {c: [] for c in range(num_classes)}
    class_ngt: Dict[int, int] = {c: 0 for c in range(num_classes)}

    for preds, gts in zip(all_preds, all_gts):
        tp_flags, _ = match_predictions(
            preds["boxes"], preds["classes"], preds["scores"],
            gts["boxes"],  gts["classes"],
            iou_thresh,
        )
        for i, cls in enumerate(preds["classes"]):
            class_tp[cls].append((preds["scores"][i], tp_flags[i]))
        for cls in gts["classes"]:
            class_ngt[cls] += 1

    ap_per_class: Dict[int, float] = {}
    for cls in range(num_classes):
        entries = sorted(class_tp[cls], key=lambda x: x[0], reverse=True)
        ngt     = class_ngt[cls]
        if ngt == 0:
            ap_per_class[cls] = float("nan")
            continue
        tp_cum = np.cumsum([1 if e[1] else 0 for e in entries])
        fp_cum = np.cumsum([0 if e[1] else 1 for e in entries])
        rec    = tp_cum / ngt
        prec   = tp_cum / (tp_cum + fp_cum + 1e-9)
        ap_per_class[cls] = _voc_ap(rec, prec)

    return ap_per_class


# ── Confusion matrix ────────────────────────────────────────────────────────

def build_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int = 13,
) -> np.ndarray:
    """
    Build a confusion matrix from flat class lists.

    Parameters
    ----------
    y_true, y_pred : integer class ids (0-indexed)

    Returns
    -------
    np.ndarray of shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def plot_confusion_matrix(
    cm:          np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize:   bool = True,
    save_path:   Optional[str] = None,
    figsize:     Tuple[int, int] = (14, 12),
) -> None:
    """
    Plot a colour-coded confusion matrix using matplotlib.

    Parameters
    ----------
    cm         : confusion matrix (output of build_confusion_matrix)
    normalize  : show proportions instead of raw counts
    save_path  : if given, saves the figure to this path
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for plotting. pip install matplotlib")

    names = class_names or CATEGORY_NAMES
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot  = np.where(row_sums > 0, cm / row_sums, 0).astype(float)
        fmt      = ".2f"
    else:
        cm_plot = cm.astype(float)
        fmt     = "d"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(names)),
        yticks=np.arange(len(names)),
        xticklabels=[n.replace("_", "\n") for n in names],
        yticklabels=[n.replace("_", " ") for n in names],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix" + (" (normalised)" if normalize else ""),
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_plot.max() / 2.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = cm_plot[i, j]
            if val == 0:
                continue
            text = format(int(cm[i, j]), "d") if not normalize else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾  Confusion matrix saved → {save_path}")
    plt.show()


# ── Detection report ────────────────────────────────────────────────────────

def detection_report(
    ap_dict:    Dict[int, float],
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Pretty-print a per-class AP report (similar to sklearn classification_report).

    Parameters
    ----------
    ap_dict : output of per_class_ap()

    Returns
    -------
    Formatted string report
    """
    names  = class_names or CATEGORY_NAMES
    lines  = []
    lines.append(f"\n{'─'*54}")
    lines.append(f"  {'Category':<30}  {'AP@50':>8}  {'Grade':>6}")
    lines.append(f"{'─'*54}")

    valid_aps = [v for v in ap_dict.values() if not np.isnan(v)]
    mean_ap   = float(np.mean(valid_aps)) if valid_aps else 0.0

    for cls_id in sorted(ap_dict):
        ap   = ap_dict[cls_id]
        name = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
        if np.isnan(ap):
            lines.append(f"  {name:<30}  {'N/A':>8}  {'—':>6}")
        else:
            grade = "✅" if ap >= 0.5 else "⚠️ " if ap >= 0.3 else "❌"
            lines.append(f"  {name:<30}  {ap:>8.4f}  {grade:>6}")

    lines.append(f"{'─'*54}")
    lines.append(f"  {'mAP@50':<30}  {mean_ap:>8.4f}")
    lines.append(f"{'─'*54}\n")
    return "\n".join(lines)


# ── Frame-level speed benchmark ────────────────────────────────────────────

def benchmark_inference(
    detector,
    num_frames: int = 100,
    frame_hw:   Tuple[int, int] = (640, 640),
) -> Dict[str, float]:
    """
    Run N blank frames through the detector and report latency statistics.

    Parameters
    ----------
    detector   : FashionDetector instance
    num_frames : how many frames to benchmark
    frame_hw   : (height, width) of test frames

    Returns
    -------
    Dict with keys: mean_ms, std_ms, min_ms, max_ms, fps
    """
    import time
    import numpy as np

    h, w   = frame_hw
    frame  = np.zeros((h, w, 3), dtype=np.uint8)
    times  = []

    print(f"⏱️  Benchmarking {num_frames} frames at {w}×{h} …")
    for _ in range(num_frames):
        t0 = time.perf_counter()
        detector.detect(frame)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    stats = {
        "mean_ms": float(times.mean()),
        "std_ms":  float(times.std()),
        "min_ms":  float(times.min()),
        "max_ms":  float(times.max()),
        "fps":     float(1000 / times.mean()),
    }
    print(f"   Mean: {stats['mean_ms']:.1f} ms  |  "
          f"Std: {stats['std_ms']:.1f} ms  |  "
          f"FPS: {stats['fps']:.1f}")
    return stats

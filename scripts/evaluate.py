"""
scripts/evaluate.py
───────────────────
Evaluates a trained model on the validation set and prints per-class metrics.

Usage:
    python scripts/evaluate.py --weights models/weights/yolov8s_fashion/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO
import pandas as pd


CATEGORY_NAMES = [
    "short_sleeve_top", "long_sleeve_top", "short_sleeve_outwear",
    "long_sleeve_outwear", "vest", "sling", "shorts", "trousers",
    "skirt", "short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--data",    default="data/sample_dataset/yolo/dataset.yaml")
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--conf",    type=float, default=0.25)
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = YOLO(args.weights)

    print(f"\n📊  Evaluating: {args.weights}\n")
    metrics = model.val(
        data  = args.data,
        imgsz = args.imgsz,
        conf  = args.conf,
        verbose = True,
    )

    # Per-class mAP table
    print("\n── Per-class mAP@50 ──────────────────────────────")
    rows = []
    for i, name in enumerate(CATEGORY_NAMES):
        try:
            ap = metrics.box.ap50[i]
        except Exception:
            ap = float("nan")
        rows.append({"Category": name, "mAP@50": round(ap, 4)})

    df = pd.DataFrame(rows).sort_values("mAP@50", ascending=False)
    print(df.to_string(index=False))

    print(f"\n   Overall mAP@50   : {metrics.box.map50:.4f}")
    print(f"   Overall mAP@50:95: {metrics.box.map:.4f}")
    print(f"   Precision        : {metrics.box.mp:.4f}")
    print(f"   Recall           : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()

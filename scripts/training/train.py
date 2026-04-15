"""
scripts/training/train.py
────────────────
Fine-tunes a YOLOv8 model on the converted DeepFashion2 sample.

Usage:
    python scripts/training/train.py --model yolov8s --epochs 50 --batch 16
    python scripts/training/train.py --model yolov8n --epochs 30 --batch 32   # faster / lighter
    python scripts/training/train.py --model yolov8s --epochs 50 --no-pretrained  # train from scratch
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="yolov8s",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
        help="YOLOv8 variant (n=nano … l=large)",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (0 to disable)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--data", default="data/sample_dataset/yolo/dataset.yaml")
    p.add_argument("--output_dir", default="models/weights")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--device", default="0", help="'0' (CUDA GPU), 'cpu', 'mps' (Apple GPU), '0,1'"
    )
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch (random weights, no COCO pretraining)",
    )
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    return p.parse_args()


def main():
    args = parse_args()

    if args.wandb:
        import wandb

        wandb.init(project="fashionsense", config=vars(args))

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {data_path}\n"
            "Run the converter first:\n"
            '  python -c "from src.detection.converter import DeepFashion2ToYOLO; '
            "DeepFashion2ToYOLO('data/sample_dataset').convert()\""
        )

    # ── Load model ────────────────────────────────────────────────────────
    if args.no_pretrained:
        # Architecture only — random weights, no COCO pretraining
        model = YOLO(f"{args.model}.yaml")
        tag = "from scratch"
    else:
        # COCO pretrained weights → fine-tune on fashion
        model = YOLO(f"{args.model}.pt")
        tag = "pretrained"
    print(
        f"\n🚀  Starting training ({tag}): {args.model}  |  epochs={args.epochs}  |  "
        f"batch={args.batch}  |  patience={args.patience}\n"
    )
    

    # ── Train ─────────────────────────────────────────────────────────────
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device or None,
        project=str(Path(args.output_dir).resolve()),
        name=(
            f"{args.model}_fashion"
            if not args.no_pretrained
            else f"{args.model}_fashion_scratch"
        ),
        exist_ok=False,
        patience=args.patience,
        # Augmentation settings (important for clothing variety)
        hsv_h=0.015,  # hue shift
        hsv_s=0.7,  # saturation shift
        hsv_v=0.4,  # brightness shift
        flipud=0.0,  # no vertical flip (clothes are upright)
        fliplr=0.5,  # horizontal flip — valid for clothing
        mosaic=1.0,  # mosaic augmentation
        mixup=0.1,
        # Optimiser
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        # Save best checkpoint
        save=True,
        save_period=10,
    )

    run_name = (
        f"{args.model}_fashion"
        if not args.no_pretrained
        else f"{args.model}_fashion_scratch"
    )
    best = Path(args.output_dir) / run_name / "weights" / "best.pt"
    print(f"\n✅  Training complete!")
    print(f"   Best weights → {best.resolve()}")
    print(f"\n   Quick test:")
    print(f"   python scripts/evaluation/evaluate.py --weights {best}")


if __name__ == "__main__":
    main()

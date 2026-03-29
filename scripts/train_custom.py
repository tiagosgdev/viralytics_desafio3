"""
scripts/train_custom.py
────────────────────────
Training script for FashionNet (custom from-scratch model).

Usage:
    python scripts/train_custom.py --epochs 50 --batch 8 --device cpu
    python scripts/train_custom.py --epochs 50 --batch 16 --device cuda  # GPU
    python scripts/train_custom.py --epochs 50 --batch 16 --device mps   # Apple

Saves checkpoints to: models/weights/fashionnet/
"""

import argparse
import json
import time
from pathlib import Path

import yaml
from tqdm import tqdm

import copy

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LinearLR, SequentialLR

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.custom_model.model   import FashionNet, TinyFashionNet
from src.custom_model.loss    import FashionNetLoss
from src.custom_model.dataset import build_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description="Train FashionNet from scratch")
    p.add_argument("--epochs",   type=int,   default=50)
    p.add_argument("--batch",    type=int,   default=8,
                   help="Batch size — use 4-8 on CPU, 16+ on GPU")
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--device",   default="",
                   help="'cpu', 'cuda', 'mps', or '' for auto")
    p.add_argument("--workers",  type=int,   default=0,
                   help="DataLoader workers (0 = main process, safe on Windows)")
    p.add_argument("--data",     default="data/sample_dataset/yolo",
                   help="Path to yolo/ directory containing images/ and labels/")
    p.add_argument("--output",   default="models/weights/fashionnet")
    p.add_argument("--resume",   default="",
                   help="Path to checkpoint .pt to resume from")
    p.add_argument("--fast",     action="store_true",
                   help="Use TinyFashionNet (fewer channels) for quick testing")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Cap dataset size for quick testing (0 = use all)")

    # Experiment flags
    p.add_argument("--lambda_box", type=float, default=5.0,
                   help="Box loss weight (old default was 0.05)")
    p.add_argument("--lambda_obj", type=float, default=1.0,
                   help="Objectness loss weight")
    p.add_argument("--lambda_cls", type=float, default=0.5,
                   help="Classification loss weight")
    p.add_argument("--augment",  default="light",
                   choices=["light", "medium", "heavy"],
                   help="Augmentation intensity level")
    p.add_argument("--multi_cell", action="store_true",
                   help="Assign each GT to multiple nearby grid cells")
    p.add_argument("--num_classes", type=int, default=0,
                   help="Number of classes (0 = auto-read from dataset.yaml)")
    p.add_argument("--dropout",  type=float, default=0.0,
                   help="Dropout rate in detection heads (0 = disabled)")
    p.add_argument("--cos_lr",   action="store_true",
                   help="Use CosineAnnealingLR instead of OneCycleLR")
    p.add_argument("--grayscale", action="store_true",
                   help="Convert images to grayscale (3ch repeated) — tests shape vs colour")
    p.add_argument("--warmup_epochs", type=int, default=0,
                   help="Linear LR warmup epochs before main schedule (0 = disabled)")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"],
                   help="Optimizer: adamw (default) or sgd (momentum=0.937)")
    p.add_argument("--ema", action="store_true",
                   help="Exponential Moving Average of model weights (used for val/inference)")
    return p.parse_args()


def pick_device(requested: str) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path: Path,
                    ema=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict() if scheduler else None,
        "metrics":    metrics,
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)


class ModelEMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs,
                    batch_scheduler=None, ema=None):
    model.train()
    total_loss = box_loss = obj_loss = cls_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]",
                unit="batch", dynamic_ncols=True, leave=True)

    for i, (images, targets) in enumerate(pbar):
        images  = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss, components = criterion(preds, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        if batch_scheduler is not None:
            batch_scheduler.step()

        total_loss += loss.item()
        box_loss   += components['box']
        obj_loss   += components['obj']
        cls_loss   += components['cls']

        # Update progress bar with live loss values
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'box':  f'{components["box"]:.3f}',
            'obj':  f'{components["obj"]:.3f}',
            'cls':  f'{components["cls"]:.3f}',
        })

    n = len(loader)
    return {
        "loss": total_loss / n,
        "box":  box_loss   / n,
        "obj":  obj_loss   / n,
        "cls":  cls_loss   / n,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for images, targets in tqdm(loader, desc="  [val] ", unit="batch", dynamic_ncols=True, leave=False):
        images  = images.to(device)
        targets = targets.to(device)
        preds   = model(images)
        loss, _ = criterion(preds, targets)
        total_loss += loss.item()

    return {"val_loss": total_loss / max(len(loader), 1)}


def main():
    args   = parse_args()
    device = pick_device(args.device)
    out    = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── Resolve num_classes ──────────────────────────────────────────────
    num_classes = args.num_classes
    if num_classes == 0:
        yaml_path = Path(args.data) / "dataset.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            num_classes = cfg.get("nc", 13)
        else:
            num_classes = 13

    print(f"\n{'='*55}")
    print(f"  FashionNet — training from scratch")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch     : {args.batch}")
    print(f"  Image     : {args.imgsz}×{args.imgsz}")
    print(f"  Classes   : {num_classes}")
    print(f"  Loss wts  : box={args.lambda_box} obj={args.lambda_obj} cls={args.lambda_cls}")
    print(f"  Augment   : {args.augment}")
    print(f"  Multi-cell: {args.multi_cell}")
    print(f"  Dropout   : {args.dropout}")
    print(f"  Optimizer : {args.optimizer}")
    print(f"  Scheduler : {'CosineAnnealing' if args.cos_lr else 'OneCycleLR'}")
    print(f"  Warmup    : {args.warmup_epochs} epochs")
    print(f"  EMA       : {args.ema}")
    print(f"  Grayscale : {args.grayscale}")
    if args.fast:
        print(f"  Mode      : FAST (TinyFashionNet + capped samples)")
    if args.max_samples:
        print(f"  Samples   : max {args.max_samples} per split")
    print(f"{'='*55}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    train_dl, val_dl = build_dataloaders(
        args.data, args.imgsz, args.batch, args.workers,
        max_samples=args.max_samples,
        augment_level=args.augment,
        grayscale=args.grayscale,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    if args.fast:
        model = TinyFashionNet(num_classes=num_classes).to(device)
        print(f"  Mode: FAST (TinyFashionNet — reduced channels)")
    else:
        model = FashionNet(num_classes=num_classes, dropout=args.dropout).to(device)
    print(f"  Parameters: {model.count_parameters():,}")

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = FashionNetLoss(
        num_classes=num_classes,
        lambda_box=args.lambda_box,
        lambda_obj=args.lambda_obj,
        lambda_cls=args.lambda_cls,
        img_size=args.imgsz,
        multi_cell=args.multi_cell,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=0.937, weight_decay=5e-4, nesterov=True,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr,
            weight_decay=5e-4, betas=(0.937, 0.999),
        )

    # ── Scheduler ─────────────────────────────────────────────────────────
    if args.cos_lr:
        main_sched = CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs,
            eta_min=args.lr * 0.01,
        )
        if args.warmup_epochs > 0:
            warmup_sched = LinearLR(
                optimizer, start_factor=0.01, total_iters=args.warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer, [warmup_sched, main_sched],
                milestones=[args.warmup_epochs],
            )
        else:
            scheduler = main_sched
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr       = args.lr,
            total_steps  = args.epochs * len(train_dl),
            pct_start    = 0.1,
            anneal_strategy = 'cos',
        )

    # ── EMA ───────────────────────────────────────────────────────────────
    ema = ModelEMA(model) if args.ema else None

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch  = 1
    best_val     = float('inf')
    history      = []

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scheduler') and scheduler:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed from epoch {ckpt['epoch']}")

    # ── Save experiment config ────────────────────────────────────────────
    config = vars(args)
    config["num_classes_resolved"] = num_classes
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n  Starting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.time()

        # OneCycleLR steps per batch; CosineAnnealingLR steps per epoch
        batch_sched = scheduler if not args.cos_lr else None
        train_metrics = train_one_epoch(
            model, train_dl, criterion, optimizer, device, epoch, args.epochs,
            batch_scheduler=batch_sched, ema=ema,
        )
        if args.cos_lr:
            scheduler.step()

        # Validate with EMA model if available, otherwise raw model
        val_model = ema.ema if ema else model
        val_metrics = validate(val_model, val_dl, criterion, device)

        epoch_time = time.time() - t_epoch
        row = {**train_metrics, **val_metrics, "epoch": epoch, "time_s": round(epoch_time, 1)}
        history.append(row)

        print(f"\nEpoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_metrics['loss']:.4f}  "
              f"val_loss={val_metrics['val_loss']:.4f}  "
              f"({epoch_time:.0f}s)\n")

        # Save latest
        save_checkpoint(model, optimizer, scheduler, epoch, row, out / "last.pt", ema=ema)

        # Save best
        if val_metrics['val_loss'] < best_val:
            best_val = val_metrics['val_loss']
            save_checkpoint(model, optimizer, scheduler, epoch, row, out / "best.pt", ema=ema)
            print(f"  ✅ New best val_loss={best_val:.4f} → saved best.pt\n")

        # Save history
        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Best val_loss : {best_val:.4f}")
    print(f"  Weights saved : {(out / 'best.pt').resolve()}")
    print(f"  Run comparison: python scripts/compare_models.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
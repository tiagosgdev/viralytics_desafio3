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

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

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
    return p.parse_args()


def pick_device(requested: str) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict() if scheduler else None,
        "metrics":    metrics,
    }, path)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
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

    print(f"\n{'='*55}")
    print(f"  FashionNet — training from scratch")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Image   : {args.imgsz}×{args.imgsz}")
    if args.fast:
        print(f"  Mode    : FAST (TinyFashionNet + capped samples)")
    if args.max_samples:
        print(f"  Samples : max {args.max_samples} per split")
    print(f"{'='*55}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    train_dl, val_dl = build_dataloaders(
        args.data, args.imgsz, args.batch, args.workers,
        max_samples=args.max_samples,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    if args.fast:
        model = TinyFashionNet().to(device)
        print(f"  Mode: FAST (TinyFashionNet — reduced channels)")
    else:
        model = FashionNet().to(device)
    print(f"  Parameters: {model.count_parameters():,}")

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = FashionNetLoss(img_size=args.imgsz)

    # ── Optimiser — AdamW + OneCycleLR ────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=5e-4, betas=(0.937, 0.999)
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr       = args.lr,
        total_steps  = args.epochs * len(train_dl),
        pct_start    = 0.1,          # 10% warmup
        anneal_strategy = 'cos',
    )

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

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n  Starting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.time()

        train_metrics = train_one_epoch(
            model, train_dl, criterion, optimizer, device, epoch, args.epochs
        )
        if scheduler:
            scheduler.step()

        val_metrics = validate(model, val_dl, criterion, device)

        epoch_time = time.time() - t_epoch
        row = {**train_metrics, **val_metrics, "epoch": epoch, "time_s": round(epoch_time, 1)}
        history.append(row)

        print(f"\nEpoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_metrics['loss']:.4f}  "
              f"val_loss={val_metrics['val_loss']:.4f}  "
              f"({epoch_time:.0f}s)\n")

        # Save latest
        save_checkpoint(model, optimizer, scheduler, epoch, row, out / "last.pt")

        # Save best
        if val_metrics['val_loss'] < best_val:
            best_val = val_metrics['val_loss']
            save_checkpoint(model, optimizer, scheduler, epoch, row, out / "best.pt")
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
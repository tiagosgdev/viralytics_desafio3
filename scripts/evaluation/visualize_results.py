"""
scripts/evaluation/visualize_results.py
─────────────────────────────
Visualization script for FashionNet evaluation results.

Generates:
  1. Confusion matrix heatmap (with background row/col)
  2. Training loss curves (total + component losses)
  3. Per-class AP bar chart
  4. Per-class F1 bar chart
  5. Experiment comparison table

Usage:
    # Single experiment:
    python scripts/evaluation/visualize_results.py \
        --metrics_json models/weights/fashionnet/metrics.json \
        --history_json models/weights/fashionnet/history.json \
        --output_dir results/plots/fashionnet

    # Compare experiments:
    python scripts/evaluation/visualize_results.py \
        --exp_dirs models/weights/exp1 models/weights/exp2 \
        --output_dir results/plots/comparison
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualize FashionNet results")
    p.add_argument("--metrics_json", default="",
                   help="Path to metrics.json from evaluate_custom.py")
    p.add_argument("--history_json", default="",
                   help="Path to history.json from train_custom.py")
    p.add_argument("--output_dir",  default="results/plots",
                   help="Where to save PNGs")
    p.add_argument("--exp_dirs",    nargs="*", default=[],
                   help="Multiple experiment dirs for comparison table")
    p.add_argument("--dpi",         type=int, default=150)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Confusion Matrix Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix_heatmap(metrics, output_dir, dpi=150):
    """(NC+1) x (NC+1) heatmap with background row/column."""
    cm = np.array(metrics["confusion_matrix"])
    names = metrics["class_names"]

    # Short display names
    short_names = [n.replace("_", "\n") for n in names]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.9), max(8, len(names) * 0.8)))

    if HAS_SNS:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=short_names, yticklabels=[n.replace("_", " ") for n in names],
            ax=ax, linewidths=0.5,
        )
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(short_names, fontsize=7)
        ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7)
        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black", fontsize=6)

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Detection Confusion Matrix", fontsize=13)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    plt.tight_layout()

    out = Path(output_dir) / "confusion_matrix.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Training Loss Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history, output_dir, dpi=150):
    """Two subplots: total loss + component losses."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: train vs val loss
    ax1.plot(epochs, train_loss, label="train_loss", linewidth=1.5)
    ax1.plot(epochs, val_loss, label="val_loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: component losses
    has_components = all(k in history[0] for k in ("box", "obj", "cls"))
    if has_components:
        ax2.plot(epochs, [h["box"] for h in history], label="box", linewidth=1.5)
        ax2.plot(epochs, [h["obj"] for h in history], label="obj", linewidth=1.5)
        ax2.plot(epochs, [h["cls"] for h in history], label="cls", linewidth=1.5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss Component")
        ax2.set_title("Loss Components")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No component losses recorded",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Loss Components (N/A)")

    plt.tight_layout()

    out = Path(output_dir) / "training_curves.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Per-class AP Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_ap(metrics, output_dir, dpi=150):
    """Horizontal bar chart of per-class AP, sorted descending, color-coded."""
    per_class = metrics["per_class"]
    mAP50 = metrics["mAP50"]

    # Sort by AP descending
    items = [(name, m["AP"]) for name, m in per_class.items() if m["AP"] is not None]
    items.sort(key=lambda x: x[1], reverse=True)

    if not items:
        print("  No per-class AP data to plot.")
        return

    names = [x[0] for x in items]
    aps = [x[1] for x in items]
    colors = ["#2ecc71" if ap >= 0.5 else "#f39c12" if ap >= 0.3 else "#e74c3c" for ap in aps]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, aps, color=colors, edgecolor="white", height=0.7)

    # Value labels
    for bar, ap in zip(bars, aps):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{ap:.3f}", va="center", fontsize=9)

    # mAP line
    ax.axvline(x=mAP50, color="navy", linestyle="--", linewidth=1.2, label=f"mAP@50 = {mAP50:.3f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("AP@50")
    ax.set_title("Per-class Average Precision")
    ax.set_xlim(0, min(1.15, max(aps) + 0.15))
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out = Path(output_dir) / "per_class_ap.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Per-class F1 Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_f1(metrics, output_dir, dpi=150):
    """Horizontal bar chart of per-class F1, sorted descending, color-coded."""
    per_class = metrics["per_class"]

    items = [(name, m["F1"]) for name, m in per_class.items()]
    items.sort(key=lambda x: x[1], reverse=True)

    if not items:
        print("  No per-class F1 data to plot.")
        return

    names = [x[0] for x in items]
    f1s = [x[1] for x in items]
    colors = ["#2ecc71" if f1 >= 0.5 else "#f39c12" if f1 >= 0.3 else "#e74c3c" for f1 in f1s]

    macro_f1 = metrics["F1"]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, f1s, color=colors, edgecolor="white", height=0.7)

    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{f1:.3f}", va="center", fontsize=9)

    ax.axvline(x=macro_f1, color="navy", linestyle="--", linewidth=1.2, label=f"Macro F1 = {macro_f1:.3f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-class F1 Score")
    ax.set_xlim(0, min(1.15, max(f1s) + 0.15) if f1s else 1.0)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out = Path(output_dir) / "per_class_f1.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Experiment Comparison Table
# ─────────────────────────────────────────────────────────────────────────────

def plot_experiment_comparison(exp_dirs, output_dir, dpi=150):
    """
    Read metrics.json, history.json, config.json from each experiment dir.
    Render a comparison table and print to stdout.
    """
    rows = []

    for exp_dir in exp_dirs:
        exp_dir = Path(exp_dir)
        name = exp_dir.name

        metrics_path = exp_dir / "metrics.json"
        history_path = exp_dir / "history.json"
        config_path = exp_dir / "config.json"

        mAP50 = f1 = best_val = best_epoch = ""
        key_flags = ""

        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            mAP50 = f"{m.get('mAP50', 0):.4f}"
            f1 = f"{m.get('F1', 0):.4f}"

        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            if history:
                val_losses = [h["val_loss"] for h in history]
                best_idx = int(np.argmin(val_losses))
                best_val = f"{val_losses[best_idx]:.4f}"
                best_epoch = str(history[best_idx]["epoch"])

        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            flags = []
            if cfg.get("augment", "light") != "light":
                flags.append(f"aug={cfg['augment']}")
            if cfg.get("multi_cell"):
                flags.append("multi_cell")
            if cfg.get("ema"):
                flags.append("ema")
            if cfg.get("cos_lr"):
                flags.append("cos_lr")
            if cfg.get("dropout", 0) > 0:
                flags.append(f"drop={cfg['dropout']}")
            if cfg.get("grayscale"):
                flags.append("gray")
            key_flags = ", ".join(flags) if flags else "-"

        rows.append([name, mAP50, f1, best_val, best_epoch, key_flags])

    if not rows:
        print("  No experiment directories found.")
        return

    headers = ["Experiment", "mAP@50", "F1", "Best val_loss", "Best epoch", "Key flags"]

    # Print to stdout
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep = "-" * len(header_line)

    print(f"\n{sep}")
    print(header_line)
    print(sep)
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, col_widths)))
    print(f"{sep}\n")

    # Find best values for highlighting
    def safe_float(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    best_map = max((safe_float(r[1]) for r in rows if safe_float(r[1]) is not None), default=None)
    best_f1 = max((safe_float(r[2]) for r in rows if safe_float(r[2]) is not None), default=None)
    best_vloss = min((safe_float(r[3]) for r in rows if safe_float(r[3]) is not None), default=None)

    # Render as matplotlib table
    fig, ax = plt.subplots(figsize=(max(12, len(headers) * 2.2), max(3, len(rows) * 0.6 + 1.5)))
    ax.axis("off")
    ax.set_title("Experiment Comparison", fontsize=14, fontweight="bold", pad=20)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#34495e")
        cell.set_text_props(color="white", fontweight="bold")

    # Highlight best values in green
    for i, row in enumerate(rows, start=1):
        if safe_float(row[1]) is not None and safe_float(row[1]) == best_map:
            table[i, 1].set_facecolor("#d5f5e3")
        if safe_float(row[2]) is not None and safe_float(row[2]) == best_f1:
            table[i, 2].set_facecolor("#d5f5e3")
        if safe_float(row[3]) is not None and safe_float(row[3]) == best_vloss:
            table[i, 3].set_facecolor("#d5f5e3")

    plt.tight_layout()

    out = Path(output_dir) / "experiment_comparison.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Output directory: {output_dir}\n")

    # Single experiment plots
    if args.metrics_json:
        with open(args.metrics_json) as f:
            metrics = json.load(f)
        plot_confusion_matrix_heatmap(metrics, output_dir, args.dpi)
        plot_per_class_ap(metrics, output_dir, args.dpi)
        plot_per_class_f1(metrics, output_dir, args.dpi)

    if args.history_json:
        with open(args.history_json) as f:
            history = json.load(f)
        plot_training_curves(history, output_dir, args.dpi)

    # Multi-experiment comparison
    if args.exp_dirs:
        plot_experiment_comparison(args.exp_dirs, output_dir, args.dpi)

    if not args.metrics_json and not args.history_json and not args.exp_dirs:
        print("  Nothing to plot. Provide --metrics_json, --history_json, or --exp_dirs.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt

from .data_utils import ensure_dir


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def plot_folder_metrics(
    folder_size: int,
    model_rows: Sequence[Mapping[str, float]],
    latency_map: Mapping[str, Sequence[float]],
    per_key_map: Mapping[str, Mapping[str, Mapping[str, float]]],
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)

    model_names = [str(row["model"]) for row in model_rows]

    metric_keys = [
        "structured_micro_f1",
        "structured_macro_f1",
        "negation_accuracy",
        "strict_exact_match_rate",
    ]
    metric_labels = ["Micro F1", "Macro F1", "Negation Acc", "Exact Match"]

    x = list(range(len(metric_keys)))
    width = 0.35 if len(model_rows) > 1 else 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, row in enumerate(model_rows):
        y = [float(row.get(key, 0.0)) for key in metric_keys]
        offsets = [pos + (idx - (len(model_rows) - 1) / 2) * width for pos in x]
        ax.bar(offsets, y, width=width, label=str(row["model"]))

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Folder {folder_size}: Quality Metrics by Model")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "quality_bars.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    latency_values = [list(latency_map.get(name, [])) for name in model_names]
    ax.boxplot(latency_values, labels=model_names, showfliers=False)
    ax.set_title(f"Folder {folder_size}: Inference Latency Distribution")
    ax.set_ylabel("Latency (ms)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "latency_boxplot.png", dpi=160)
    plt.close(fig)

    all_keys = sorted({key for model_name in per_key_map for key in per_key_map[model_name].keys()})
    if all_keys:
        matrix: List[List[float]] = []
        for model_name in model_names:
            row_vals = []
            key_scores = per_key_map.get(model_name, {})
            for key in all_keys:
                row_vals.append(float(key_scores.get(key, {}).get("f1", 0.0)))
            matrix.append(row_vals)

        fig, ax = plt.subplots(figsize=(max(8, len(all_keys) * 0.6), 4.5))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(all_keys)))
        ax.set_xticklabels(all_keys, rotation=45, ha="right")
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names)
        ax.set_title(f"Folder {folder_size}: Per-Key F1 Heatmap")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(output_dir / "per_key_f1_heatmap.png", dpi=160)
        plt.close(fig)


def plot_global_learning_curves(global_rows: Sequence[Mapping[str, float]], output_dir: Path) -> None:
    ensure_dir(output_dir)

    grouped: Dict[str, List[Mapping[str, float]]] = defaultdict(list)
    for row in global_rows:
        grouped[str(row["model"])].append(row)

    for model, rows in grouped.items():
        rows.sort(key=lambda item: int(item["folder_size"]))

    metrics = [
        ("structured_micro_f1", "Micro F1"),
        ("structured_macro_f1", "Macro F1"),
        ("negation_accuracy", "Negation Accuracy"),
        ("strict_exact_match_rate", "Exact Match"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for axis, (metric_key, metric_label) in zip(axes_flat, metrics):
        for model, rows in grouped.items():
            x = [int(item["folder_size"]) for item in rows]
            y = [float(item.get(metric_key, 0.0)) for item in rows]
            axis.plot(x, y, marker="o", label=model)
        axis.set_title(metric_label)
        axis.set_xlabel("Dataset Size")
        axis.set_ylabel("Score")
        axis.set_ylim(0.0, 1.0)
        axis.grid(linestyle="--", alpha=0.3)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=max(2, len(handles)))

    fig.suptitle("Learning Curves Across Dataset Sizes", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(output_dir / "learning_curves.png", dpi=170)
    plt.close(fig)


def plot_global_latency_vs_quality(global_rows: Sequence[Mapping[str, float]], output_dir: Path) -> None:
    ensure_dir(output_dir)

    grouped: Dict[str, List[Mapping[str, float]]] = defaultdict(list)
    for row in global_rows:
        grouped[str(row["model"])].append(row)

    fig, ax = plt.subplots(figsize=(9, 6))

    for model, rows in grouped.items():
        x = [float(row.get("latency_p95_ms", 0.0)) for row in rows]
        y = [float(row.get("quality_score", 0.0)) for row in rows]
        sizes = [int(row.get("folder_size", 0)) for row in rows]

        ax.scatter(x, y, s=80, alpha=0.8, label=model)
        for xpos, ypos, size in zip(x, y, sizes):
            ax.annotate(str(size), (xpos, ypos), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_xlabel("Latency p95 (ms)")
    ax.set_ylabel("Quality Score")
    ax.set_title("Quality vs Latency (higher is better, lower latency is better)")
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "quality_vs_latency.png", dpi=170)
    plt.close(fig)


def plot_global_delta(global_rows: Sequence[Mapping[str, float]], output_dir: Path) -> None:
    ensure_dir(output_dir)

    by_size: Dict[int, Dict[str, Mapping[str, float]]] = defaultdict(dict)
    for row in global_rows:
        size = int(row.get("folder_size", 0))
        by_size[size][str(row.get("model"))] = row

    sizes = sorted(by_size.keys())
    if not sizes:
        return

    if not any("rule_based" in model_map and "crf" in model_map for model_map in by_size.values()):
        return

    metrics = [
        ("structured_micro_f1", "Delta Micro F1"),
        ("structured_macro_f1", "Delta Macro F1"),
        ("quality_score", "Delta Quality Score"),
        ("tradeoff_score", "Delta Tradeoff Score"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for axis, (metric_key, title) in zip(axes_flat, metrics):
        deltas = []
        valid_sizes = []

        for size in sizes:
            model_map = by_size[size]
            if "rule_based" not in model_map or "crf" not in model_map:
                continue
            crf_value = float(model_map["crf"].get(metric_key, 0.0))
            rule_value = float(model_map["rule_based"].get(metric_key, 0.0))
            valid_sizes.append(size)
            deltas.append(crf_value - rule_value)

        if not valid_sizes:
            continue

        colors = ["#2ca02c" if val >= 0 else "#d62728" for val in deltas]
        axis.bar([str(size) for size in valid_sizes], deltas, color=colors)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Dataset Size")
        axis.set_ylabel("CRF - RuleBased")
        axis.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("CRF vs Rule-Based Delta Across Sizes", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "crf_minus_rule_delta.png", dpi=170)
    plt.close(fig)

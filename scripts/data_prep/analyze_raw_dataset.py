"""
scripts/data_prep/analyze_raw_dataset.py
──────────────────────────────
Analyse the raw DeepFashion2 dataset from the pre-built CSV dataframes.
Produces 7 figures in docs/figures/raw_dataset/ and prints a per-class summary table.

Usage:
    python scripts/data_prep/analyze_raw_dataset.py
"""

import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── DeepFashion2 category map ──────────────────────────────────────────────
CATEGORY_MAP = {
    1: "short_sleeve_top",
    2: "long_sleeve_top",
    3: "short_sleeve_outwear",
    4: "long_sleeve_outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeve_dress",
    11: "long_sleeve_dress",
    12: "vest_dress",
    13: "sling_dress",
}


# ── Data loading ───────────────────────────────────────────────────────────

def load_and_prepare(train_csv: str, val_csv: str) -> pd.DataFrame:
    print("📂  Loading CSVs …")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_train["split"] = "train"
    df_val["split"] = "val"
    df = pd.concat([df_train, df_val], ignore_index=True)

    df["img_id"] = df["path"].apply(lambda p: Path(p).stem)
    df["category_name_mapped"] = df["category_id"].map(CATEGORY_MAP)

    # Parse bounding box
    coords = df["b_box"].apply(ast.literal_eval)
    df["x1"] = coords.str[0]
    df["y1"] = coords.str[1]
    df["x2"] = coords.str[2]
    df["y2"] = coords.str[3]
    df["bbox_w"] = df["x2"] - df["x1"]
    df["bbox_h"] = df["y2"] - df["y1"]
    df["rel_bbox_area"] = (df["bbox_w"] * df["bbox_h"]) / (df["img_width"] * df["img_height"])
    df["aspect_ratio"] = df["bbox_w"] / df["bbox_h"].replace(0, np.nan)

    print(f"   ✅ {len(df)} items across {df['img_id'].nunique()} images")
    return df


# ── Plot functions ─────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, output_dir: Path):
    print("📊  Plotting class distribution …")
    counts = df.groupby(["category_name_mapped", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(
        [CATEGORY_MAP[i] for i in sorted(CATEGORY_MAP)], fill_value=0
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    counts.plot.bar(ax=ax)
    ax.set_title("Class Distribution (Train vs Val)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "raw_class_distribution.png", dpi=150)
    plt.close(fig)


def plot_cooccurrence_matrix(df: pd.DataFrame, output_dir: Path):
    print("📊  Plotting co-occurrence matrix …")
    cats = sorted(CATEGORY_MAP.keys())
    cat_names = [CATEGORY_MAP[c] for c in cats]
    n = len(cats)
    matrix = np.zeros((n, n), dtype=int)

    for _, group in df.groupby("img_id"):
        present = sorted(group["category_id"].unique())
        for i, ci in enumerate(present):
            for cj in present[i:]:
                idx_i = cats.index(ci)
                idx_j = cats.index(cj)
                matrix[idx_i, idx_j] += 1
                if idx_i != idx_j:
                    matrix[idx_j, idx_i] += 1

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix, xticklabels=cat_names, yticklabels=cat_names,
        annot=True, fmt="d", cmap="YlOrRd", ax=ax,
    )
    ax.set_title("Category Co-occurrence Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "raw_cooccurrence_matrix.png", dpi=150)
    plt.close(fig)


def plot_bbox_size_distribution(df: pd.DataFrame, output_dir: Path):
    print("📊  Plotting bbox size distribution …")
    order = [CATEGORY_MAP[i] for i in sorted(CATEGORY_MAP)]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=df, x="category_name_mapped", y="rel_bbox_area",
        order=order, ax=ax, showfliers=False,
    )
    ax.set_title("Relative Bounding Box Area per Class")
    ax.set_xlabel("Category")
    ax.set_ylabel("Relative BBox Area (bbox / image)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "raw_bbox_size_distribution.png", dpi=150)
    plt.close(fig)


def plot_resolution_distribution(df: pd.DataFrame, output_dir: Path):
    print("📊  Plotting resolution distribution …")
    img_df = df.drop_duplicates(subset="img_id")[["img_width", "img_height"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist2d(img_df["img_width"], img_df["img_height"], bins=50, cmap="viridis")
    ax.set_title("Image Resolution Distribution")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    fig.colorbar(ax.collections[0], ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(output_dir / "raw_resolution_distribution.png", dpi=150)
    plt.close(fig)


def plot_occlusion_distribution(df: pd.DataFrame, output_dir: Path):
    print("📊  Plotting occlusion distribution …")
    order = [CATEGORY_MAP[i] for i in sorted(CATEGORY_MAP)]
    occ_counts = (
        df.groupby(["category_name_mapped", "occlusion"])
        .size()
        .unstack(fill_value=0)
    )
    occ_counts = occ_counts.reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 7))
    occ_counts.plot.bar(stacked=True, ax=ax, colormap="RdYlGn_r")
    ax.set_title("Occlusion Level per Class (1=none, 2=partial, 3=heavy)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Occlusion")
    fig.tight_layout()
    fig.savefig(output_dir / "raw_occlusion_per_class.png", dpi=150)
    plt.close(fig)


def plot_scale_distribution(df: pd.DataFrame, output_dir: Path):
    print("📊  Plotting scale distribution …")
    order = [CATEGORY_MAP[i] for i in sorted(CATEGORY_MAP)]
    scale_counts = (
        df.groupby(["category_name_mapped", "scale"])
        .size()
        .unstack(fill_value=0)
    )
    scale_counts = scale_counts.reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 7))
    scale_counts.plot.bar(stacked=True, ax=ax, colormap="coolwarm")
    ax.set_title("Scale Level per Class (1=small, 2=medium, 3=large)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Scale")
    fig.tight_layout()
    fig.savefig(output_dir / "raw_scale_per_class.png", dpi=150)
    plt.close(fig)


def plot_aspect_ratio_distribution(df: pd.DataFrame, output_dir: Path):
    print("📊  Plotting aspect ratio distribution …")
    order = [CATEGORY_MAP[i] for i in sorted(CATEGORY_MAP)]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=df, x="category_name_mapped", y="aspect_ratio",
        order=order, ax=ax, showfliers=False,
    )
    ax.set_title("Bounding Box Aspect Ratio per Class (w/h)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Aspect Ratio (w / h)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "raw_aspect_ratio_per_class.png", dpi=150)
    plt.close(fig)


# ── Summary table ──────────────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame):
    print("\n" + "=" * 90)
    print(f"{'Category':<25} {'Count':>8} {'%':>7} {'Med Area':>10} {'Med AR':>8} {'% Occ=3':>8}")
    print("-" * 90)

    total = len(df)
    for cat_id in sorted(CATEGORY_MAP):
        name = CATEGORY_MAP[cat_id]
        sub = df[df["category_id"] == cat_id]
        count = len(sub)
        pct = 100 * count / total if total else 0
        med_area = sub["rel_bbox_area"].median()
        med_ar = sub["aspect_ratio"].median()
        pct_occ3 = 100 * (sub["occlusion"] == 3).sum() / count if count else 0
        print(f"{name:<25} {count:>8d} {pct:>6.1f}% {med_area:>10.4f} {med_ar:>8.2f} {pct_occ3:>7.1f}%")

    print("=" * 90)
    print(f"{'TOTAL':<25} {total:>8d}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse the raw DeepFashion2 dataset from pre-built CSVs"
    )
    parser.add_argument(
        "--train_csv",
        default="data/raw/DeepFashion2/img_info_dataframes/train.csv",
    )
    parser.add_argument(
        "--val_csv",
        default="data/raw/DeepFashion2/img_info_dataframes/validation.csv",
    )
    parser.add_argument("--output_dir", default="docs/figures/raw_dataset")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(args.train_csv, args.val_csv)

    plot_class_distribution(df, output_dir)
    plot_cooccurrence_matrix(df, output_dir)
    plot_bbox_size_distribution(df, output_dir)
    plot_resolution_distribution(df, output_dir)
    plot_occlusion_distribution(df, output_dir)
    plot_scale_distribution(df, output_dir)
    plot_aspect_ratio_distribution(df, output_dir)

    print_summary_table(df)
    print(f"\n🎉  All figures saved to {output_dir.resolve()}/")


if __name__ == "__main__":
    main()

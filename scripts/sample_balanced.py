"""
scripts/sample_balanced.py
──────────────────────────
Balanced sampler for DeepFashion2.

Drops rare classes (sling, short_sleeve_outwear), samples equal items per
remaining class with occlusion-stratified sampling, splits into
train/val/test (70/15/15), copies images, and produces YOLO-format output.

Usage:
    python scripts/sample_balanced.py \
        --train_csv data/raw/DeepFashion2/img_info_dataframes/train.csv \
        --val_csv   data/raw/DeepFashion2/img_info_dataframes/validation.csv \
        --img_dirs  data/raw/train/image data/raw/validation/image \
        --output_dir data/balanced_dataset \
        --n_per_class 7641 \
        --seed 42
"""

import argparse
import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Categories (excluding sling=6 and short_sleeve_outwear=3) ─────────────
EXCLUDED_CATEGORIES = {3, 6}  # short_sleeve_outwear, sling

CATEGORY_MAP = {
    1: "short_sleeve_top",
    2: "long_sleeve_top",
    4: "long_sleeve_outwear",
    5: "vest",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeve_dress",
    11: "long_sleeve_dress",
    12: "vest_dress",
    13: "sling_dress",
}

# YOLO class index (0-based, ordered by original category_id)
YOLO_CLASS_MAP = {cat_id: i for i, cat_id in enumerate(sorted(CATEGORY_MAP))}


# ── Data loading ───────────────────────────────────────────────────────────

def load_and_filter(train_csv: str, val_csv: str) -> pd.DataFrame:
    """Load both CSVs, concatenate, and drop excluded categories."""
    print("📂  Loading CSVs …")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df = pd.concat([df_train, df_val], ignore_index=True)

    df["img_id"] = df["path"].apply(lambda p: Path(p).stem)

    # Drop excluded categories
    before = len(df)
    df = df[~df["category_id"].isin(EXCLUDED_CATEGORIES)].copy()
    print(f"   Dropped {before - len(df)} items from excluded categories "
          f"(sling, short_sleeve_outwear)")
    print(f"   {len(df)} items across {df['img_id'].nunique()} images, "
          f"{len(CATEGORY_MAP)} classes")
    return df


# ── Sampling ───────────────────────────────────────────────────────────────

def sample_balanced(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    """
    Sample n_per_class items per category, stratified by occlusion level.
    Works at the annotation (row) level — one image may contribute multiple items.
    """
    print(f"\n📊  Balanced sampling ({n_per_class}/class, stratified by occlusion) …")
    sampled_parts = []

    for cat_id in sorted(CATEGORY_MAP):
        name = CATEGORY_MAP[cat_id]
        cat_df = df[df["category_id"] == cat_id]
        available = len(cat_df)
        n = min(n_per_class, available)

        # Stratified sample preserving occlusion distribution
        if n < available:
            cat_sampled, _ = train_test_split(
                cat_df, train_size=n, stratify=cat_df["occlusion"],
                random_state=seed,
            )
        else:
            cat_sampled = cat_df

        sampled_parts.append(cat_sampled)
        occ_dist = cat_sampled["occlusion"].value_counts().sort_index()
        occ_str = "  ".join(f"occ{k}={v}" for k, v in occ_dist.items())
        print(f"   [{cat_id:>2}] {name:<25} avail={available:>6}  sampled={n:>5}  {occ_str}")

    result = pd.concat(sampled_parts, ignore_index=True)
    print(f"\n   ✅ Total sampled items: {len(result)}  "
          f"({result['img_id'].nunique()} unique images)")
    return result


# ── Splitting ──────────────────────────────────────────────────────────────

def split_by_image(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """
    Split into train/val/test (70/15/15) at the IMAGE level,
    stratified by the dominant category per image to keep class balance.
    """
    print("\n✂️  Splitting 70/15/15 by image …")

    # Determine dominant category per image (most annotations)
    img_meta = (
        df.groupby("img_id")["category_id"]
        .agg(lambda x: x.value_counts().index[0])
        .rename("dominant_cat")
        .reset_index()
    )

    train_imgs, temp_imgs = train_test_split(
        img_meta, test_size=0.30, stratify=img_meta["dominant_cat"],
        random_state=seed,
    )
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.50, stratify=temp_imgs["dominant_cat"],
        random_state=seed,
    )

    splits = {}
    for name, img_df in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        split_df = df[df["img_id"].isin(img_df["img_id"])].copy()
        splits[name] = split_df
        print(f"   {name:<5}: {split_df['img_id'].nunique():>5} images, "
              f"{len(split_df):>6} items")

    return splits


# ── Output ─────────────────────────────────────────────────────────────────

def copy_images(splits: dict, img_dirs: list[str], output_dir: Path):
    """Copy images into output_dir/images/{train,val,test}/."""
    # Build a lookup of img_id → source path
    src_lookup = {}
    for d in img_dirs:
        p = Path(d)
        if p.is_dir():
            for f in p.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    src_lookup[f.stem] = f

    print(f"\n📋  Copying images ({len(src_lookup)} available in source dirs) …")
    missing = []
    for split_name, split_df in splits.items():
        dst = output_dir / "images" / split_name
        dst.mkdir(parents=True, exist_ok=True)
        for img_id in tqdm(split_df["img_id"].unique(), desc=split_name):
            src = src_lookup.get(img_id)
            if src is None:
                missing.append(img_id)
                continue
            shutil.copy2(src, dst / src.name)

    if missing:
        print(f"   ⚠️  {len(missing)} images not found (first 5: {missing[:5]})")
    else:
        print("   ✅ All images copied")


def write_yolo_labels(splits: dict, output_dir: Path):
    """Write YOLO-format label .txt files and dataset.yaml."""
    print("\n📝  Writing YOLO labels …")
    for split_name, split_df in splits.items():
        label_dir = output_dir / "labels" / split_name
        label_dir.mkdir(parents=True, exist_ok=True)

        for img_id, group in split_df.groupby("img_id"):
            lines = []
            for _, row in group.iterrows():
                bbox = ast.literal_eval(row["b_box"])
                x1, y1, x2, y2 = bbox
                img_w, img_h = row["img_width"], row["img_height"]

                # YOLO format: class cx cy w h (all normalised 0-1)
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h

                cls = YOLO_CLASS_MAP[int(row["category_id"])]
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(label_dir / f"{img_id}.txt", "w") as f:
                f.write("\n".join(lines) + "\n")

    # dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    names = [CATEGORY_MAP[c] for c in sorted(CATEGORY_MAP)]
    abs_path = output_dir.resolve()
    yaml_content = (
        f"path: {abs_path}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(names)}\n"
        f"names:\n"
    )
    for n in names:
        yaml_content += f"- {n}\n"

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"   ✅ Labels written, dataset.yaml at {yaml_path}")


def write_index(splits: dict, output_dir: Path):
    """Write index.json with all annotations (for compatibility)."""
    index = {}
    for split_df in splits.values():
        for img_id, group in split_df.groupby("img_id"):
            items = []
            for _, row in group.iterrows():
                bbox = ast.literal_eval(row["b_box"])
                items.append({
                    "category_id": int(row["category_id"]),
                    "bounding_box": bbox,
                })
            index[img_id] = items

    with open(output_dir / "index.json", "w") as f:
        json.dump(index, f)


def print_split_summary(splits: dict):
    """Print per-class counts for each split."""
    print("\n" + "=" * 75)
    header = f"{'Category':<25}"
    for s in splits:
        header += f" {s:>8}"
    header += f" {'Total':>8}"
    print(header)
    print("-" * 75)

    for cat_id in sorted(CATEGORY_MAP):
        name = CATEGORY_MAP[cat_id]
        row = f"{name:<25}"
        total = 0
        for split_df in splits.values():
            n = (split_df["category_id"] == cat_id).sum()
            total += n
            row += f" {n:>8}"
        row += f" {total:>8}"
        print(row)

    print("-" * 75)
    row = f"{'TOTAL':<25}"
    grand = 0
    for split_df in splits.values():
        n = len(split_df)
        grand += n
        row += f" {n:>8}"
    row += f" {grand:>8}"
    print(row)
    print("=" * 75)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Balanced sampler for DeepFashion2 (occlusion-stratified, 70/15/15 split)"
    )
    parser.add_argument(
        "--train_csv",
        default="data/raw/DeepFashion2/img_info_dataframes/train.csv",
    )
    parser.add_argument(
        "--val_csv",
        default="data/raw/DeepFashion2/img_info_dataframes/validation.csv",
    )
    parser.add_argument(
        "--img_dirs", nargs="+",
        default=["data/raw/train/image", "data/raw/validation/image"],
        help="Directories containing source images",
    )
    parser.add_argument("--output_dir", default="data/balanced_dataset")
    parser.add_argument("--n_per_class", type=int, default=7641)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & filter
    df = load_and_filter(args.train_csv, args.val_csv)

    # 2. Balanced sample (occlusion-stratified)
    sampled = sample_balanced(df, args.n_per_class, args.seed)

    # 3. Split 70/15/15 by image
    splits = split_by_image(sampled, args.seed)

    # 4. Summary
    print_split_summary(splits)

    # 5. Copy images & write labels
    copy_images(splits, args.img_dirs, output_dir)
    write_yolo_labels(splits, output_dir)
    write_index(splits, output_dir)

    print(f"\n🎉  Balanced dataset ready at {output_dir.resolve()}/")


if __name__ == "__main__":
    main()

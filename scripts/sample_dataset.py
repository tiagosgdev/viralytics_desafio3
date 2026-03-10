"""
scripts/sample_dataset.py
─────────────────────────
Stratified sampler for DeepFashion2.
Produces equal N images per clothing category without data manipulation.

Uses the pre-built CSV dataframes instead of reading individual JSON files,
reducing annotation parsing from minutes to seconds.

Usage:
    python scripts/sample_dataset.py \
        --data_dir  data/raw \
        --csv_path  data/raw/DeepFashion2/img_info_dataframes/train.csv \
        --output_dir data/sample_dataset \
        --n_per_class 500 \
        --seed 42
"""

import argparse
import ast
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

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


def parse_csv_annotations(csv_path: Path) -> dict:
    """
    Read the pre-built CSV dataframe (one row per clothing item).
    Returns: { image_id: {category_ids: [...], items: [{category_id, bounding_box}, ...]} }
    """
    print(f"📂  Reading annotations from {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Extract image ID from path (e.g. "...train/image/039350.jpg" → "039350")
    df["img_id"] = df["path"].apply(lambda p: Path(p).stem)

    image_index = {}
    for img_id, group in df.groupby("img_id"):
        cats = set()
        items = []
        for _, row in group.iterrows():
            cat_id = int(row["category_id"])
            bbox = ast.literal_eval(row["b_box"])  # "[x1, y1, x2, y2]" → list
            cats.add(cat_id)
            items.append({"category_id": cat_id, "bounding_box": bbox})

        image_index[img_id] = {
            "category_ids": list(cats),
            "items": items,
        }

    print(f"   Loaded {len(image_index)} images, {len(df)} items")
    return image_index


def stratified_sample(image_index: dict, n_per_class: int, seed: int) -> set:
    """
    For every category pick up to n_per_class images that contain it.
    An image can be selected because it matches multiple categories — that's fine,
    it simply broadens coverage and preserves multi-label integrity.
    """
    random.seed(seed)

    cat_to_imgs = defaultdict(list)
    for img_id, meta in image_index.items():
        for cat_id in meta["category_ids"]:
            cat_to_imgs[cat_id].append(img_id)

    sampled = set()
    print("\n📊  Stratified sampling:")
    for cat_id in sorted(cat_to_imgs):
        pool = cat_to_imgs[cat_id]
        k = min(n_per_class, len(pool))
        chosen = random.sample(pool, k)
        sampled.update(chosen)
        print(f"   [{cat_id:>2}] {CATEGORY_MAP.get(cat_id, '?'):<28}  "
              f"available={len(pool):>5}  sampled={k:>4}")

    return sampled


def copy_sample(
    image_index: dict,
    sampled_ids: set,
    data_dir: Path,
    output_dir: Path,
    split: str = "train",
):
    img_src = data_dir / split / "image"
    img_dst = output_dir / "images"
    img_dst.mkdir(parents=True, exist_ok=True)

    missing_imgs = []
    copied = 0

    print(f"\n📋  Copying {len(sampled_ids)} images …")
    for img_id in tqdm(sampled_ids, desc="Copying"):
        src_img = img_src / f"{img_id}.jpg"
        if not src_img.exists():
            missing_imgs.append(img_id)
            continue
        shutil.copy2(src_img, img_dst / f"{img_id}.jpg")
        copied += 1

    # Single index.json with all annotations (used by converter.py)
    index = {}
    for img_id in sampled_ids:
        if img_id in missing_imgs:
            continue
        index[img_id] = image_index[img_id]["items"]
    with open(output_dir / "index.json", "w") as f:
        json.dump(index, f)

    print(f"\n✅  Done!  Copied: {copied}  Missing images: {len(missing_imgs)}")
    if missing_imgs:
        print(f"   ⚠️  Missing: {missing_imgs[:5]} {'…' if len(missing_imgs) > 5 else ''}")

    return copied


def main():
    parser = argparse.ArgumentParser(description="Stratified sampler for DeepFashion2")
    parser.add_argument("--data_dir",    default="data/raw",            help="Root of raw DeepFashion2")
    parser.add_argument("--csv_path",    default="data/raw/DeepFashion2/img_info_dataframes/train.csv",
                        help="Path to CSV dataframe with annotations")
    parser.add_argument("--output_dir",  default="data/sample_dataset", help="Where to write sample")
    parser.add_argument("--n_per_class", type=int, default=500,         help="Images per category")
    parser.add_argument("--split",       default="train",               help="train / validation")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    csv_path   = Path(args.csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    image_index  = parse_csv_annotations(csv_path)
    sampled_ids  = stratified_sample(image_index, args.n_per_class, args.seed)

    print(f"\n🔢  Total unique images selected: {len(sampled_ids)}")

    copy_sample(image_index, sampled_ids, data_dir, output_dir, args.split)

    print(f"\n🎉  Sample dataset ready at: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

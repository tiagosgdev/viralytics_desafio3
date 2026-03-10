"""
src/detection/converter.py
──────────────────────────
Converts DeepFashion2 annotations → YOLO format (.txt label files)
and generates the dataset.yaml required by Ultralytics.

Reads annotations from index.json (produced by sample_dataset.py)
instead of individual per-image JSON files.

YOLO label format (per line):
    class_id  x_center  y_center  width  height
    (all values normalised 0-1 relative to image dimensions)

Usage:
    from src.detection.converter import DeepFashion2ToYOLO
    DeepFashion2ToYOLO("data/sample_dataset").convert()
"""

import json
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

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

# YOLO class ids are 0-indexed
YOLO_CLASS_MAP = {df2_id: df2_id - 1 for df2_id in CATEGORY_MAP}


class DeepFashion2ToYOLO:
    """
    Converts a sampled DeepFashion2 directory to YOLO-ready structure:

    sample_dataset/
    ├── images/
    ├── index.json
    └── yolo/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── dataset.yaml
    """

    def __init__(self, sample_dir: str, val_ratio: float = 0.15, seed: int = 42):
        self.sample_dir = Path(sample_dir)
        self.img_dir    = self.sample_dir / "images"
        self.yolo_dir   = self.sample_dir / "yolo"
        self.val_ratio  = val_ratio
        self.seed       = seed

    # ── public ──────────────────────────────────────────────────────────────

    def convert(self):
        print("🔄  Converting DeepFashion2 → YOLO format …")
        self._prepare_dirs()

        index_path = self.sample_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"index.json not found in {self.sample_dir}")

        with open(index_path) as f:
            index = json.load(f)

        # Train / val split
        img_ids = sorted(index.keys())
        random.seed(self.seed)
        random.shuffle(img_ids)
        n_val = max(1, int(len(img_ids) * self.val_ratio))
        val_set = set(img_ids[:n_val])
        print(f"   Train: {len(img_ids) - n_val}  |  Val: {n_val}")

        skipped = 0
        for img_id in tqdm(img_ids, desc="Converting"):
            split = "val" if img_id in val_set else "train"
            ok = self._convert_one(index[img_id], img_id, split)
            if not ok:
                skipped += 1

        self._write_yaml()
        print(f"\n✅  Conversion complete!  Skipped (missing images): {skipped}")
        print(f"   YOLO dataset at: {self.yolo_dir.resolve()}")
        print(f"   dataset.yaml  at: {(self.yolo_dir / 'dataset.yaml').resolve()}")

    # ── private ─────────────────────────────────────────────────────────────

    def _prepare_dirs(self):
        for split in ("train", "val"):
            (self.yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _convert_one(self, items: list, img_id: str, split: str) -> bool:
        src_img = self.img_dir / f"{img_id}.jpg"
        if not src_img.exists():
            return False

        # Get image dimensions
        with Image.open(src_img) as im:
            W, H = im.size

        lines = []
        for item in items:
            cat_id = item.get("category_id")
            bbox   = item.get("bounding_box")  # [x1, y1, x2, y2]

            if cat_id not in YOLO_CLASS_MAP or not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            if x2 <= x1 or y2 <= y1:
                continue  # degenerate box

            # YOLO normalised coords
            xc = ((x1 + x2) / 2) / W
            yc = ((y1 + y2) / 2) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H

            yolo_cls = YOLO_CLASS_MAP[cat_id]
            lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            return False  # no valid boxes

        # Write label file
        lbl_path = self.yolo_dir / "labels" / split / f"{img_id}.txt"
        lbl_path.write_text("\n".join(lines))

        # Copy image
        dst_img = self.yolo_dir / "images" / split / f"{img_id}.jpg"
        shutil.copy2(src_img, dst_img)

        return True

    def _write_yaml(self):
        yaml_path = self.yolo_dir / "dataset.yaml"
        data = {
            "path":  str(self.yolo_dir.resolve()),
            "train": "images/train",
            "val":   "images/val",
            "nc":    len(CATEGORY_MAP),
            "names": [CATEGORY_MAP[i] for i in sorted(CATEGORY_MAP)],
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

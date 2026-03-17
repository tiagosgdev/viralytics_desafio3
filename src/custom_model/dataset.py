"""
src/custom_model/dataset.py
────────────────────────────
PyTorch Dataset that reads the YOLO-format labels already created by
converter.py — so no re-annotation is needed.

Label format (one .txt per image, each line):
    class_id  cx  cy  w  h    (all normalised 0-1)
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


CATEGORY_NAMES = [
    "short_sleeve_top", "long_sleeve_top", "short_sleeve_outwear",
    "long_sleeve_outwear", "vest", "sling", "shorts", "trousers",
    "skirt", "short_sleeve_dress", "long_sleeve_dress",
    "vest_dress", "sling_dress",
]


def get_train_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015, p=0.8),
        A.ToGray(p=0.01),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo', label_fields=['class_labels'], min_visibility=0.3
    ))


def get_val_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo', label_fields=['class_labels'], min_visibility=0.3
    ))


class FashionDataset(Dataset):
    """
    Reads YOLO-format image/label pairs.

    Directory structure expected:
        yolo/
          images/train/*.jpg
          images/val/*.jpg
          labels/train/*.txt
          labels/val/*.txt
    """

    def __init__(
        self,
        yolo_dir:   str,
        split:      str = "train",
        img_size:   int = 640,
        transforms: Optional[A.Compose] = None,
        max_samples: int = 0,
    ):
        self.img_size   = img_size
        self.transforms = transforms or (
            get_train_transforms(img_size) if split == "train"
            else get_val_transforms(img_size)
        )

        img_dir = Path(yolo_dir) / "images" / split
        lbl_dir = Path(yolo_dir) / "labels" / split

        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        # Match image files to their label files
        self.samples: List[Tuple[Path, Optional[Path]]] = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            self.samples.append((img_path, lbl_path if lbl_path.exists() else None))

        if max_samples and 0 < max_samples < len(self.samples):
            import random
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)
            print(f"  FashionDataset [{split}]: {len(self.samples)} images (capped from full set)")
        else:
            print(f"  FashionDataset [{split}]: {len(self.samples)} images found")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]

        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load labels
        boxes, classes = [], []
        if lbl_path and lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, w, h = parts
                        boxes.append([float(cx), float(cy), float(w), float(h)])
                        classes.append(int(cls))

        # Apply transforms
        try:
            aug = self.transforms(
                image=img,
                bboxes=boxes,
                class_labels=classes,
            )
            img_t  = aug['image']        # (3, H, W) tensor
            boxes  = list(aug['bboxes'])
            classes= list(aug['class_labels'])
        except Exception:
            # Fallback: return image without boxes if augmentation fails
            img_t  = torch.zeros(3, self.img_size, self.img_size)
            boxes, classes = [], []

        return img_t, boxes, classes, str(img_path)


def collate_fn(batch):
    """
    Custom collate that handles variable-length box lists per image.
    Returns:
        images  : (B, 3, H, W)
        targets : (N, 6) — [batch_idx, cls, cx, cy, w, h]
    """
    images, targets = [], []

    for batch_idx, (img, boxes, classes, _) in enumerate(batch):
        images.append(img)
        for box, cls in zip(boxes, classes):
            cx, cy, w, h = box
            targets.append([batch_idx, cls, cx, cy, w, h])

    images = torch.stack(images)
    targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 6))
    return images, targets


def build_dataloaders(
    yolo_dir:    str,
    img_size:    int = 640,
    batch_size:  int = 16,
    workers:     int = 0,
    max_samples: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders."""
    val_cap  = max(1, max_samples // 5) if max_samples else 0
    train_ds = FashionDataset(yolo_dir, "train", img_size, max_samples=max_samples)
    val_ds   = FashionDataset(yolo_dir, "val",   img_size, max_samples=val_cap)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=workers, pin_memory=torch.cuda.is_available(),
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=workers, pin_memory=torch.cuda.is_available(),
    )
    return train_dl, val_dl
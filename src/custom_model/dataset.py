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


def get_train_transforms(img_size: int = 640, level: str = "light",
                         grayscale: bool = False) -> A.Compose:
    """
    Training augmentations at three intensity levels.

    - light  : horizontal flip + color jitter (original)
    - medium : + random scale/rotation/translate
    - heavy  : + aggressive scale/rotation, Gaussian noise, coarse dropout
    """
    common_pre = [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
    ]
    grayscale_tf = [A.ToGray(p=1.0)] if grayscale else []

    common_post = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    if level == "heavy":
        spatial = [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.5, 1.5), rotate=(-15, 15), translate_percent=(-0.1, 0.1),
                     border_mode=cv2.BORDER_CONSTANT, p=0.7),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.7, hue=0.02, p=0.8),
            A.ToGray(p=0.05),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            A.CoarseDropout(num_holes_range=(4, 8),
                            hole_height_range=(16, 32),
                            hole_width_range=(16, 32),
                            fill=0, p=0.3),
        ]
    elif level == "medium":
        spatial = [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.7, 1.3), rotate=(-10, 10), translate_percent=(-0.1, 0.1),
                     border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015, p=0.8),
            A.ToGray(p=0.02),
        ]
    else:  # light
        spatial = [
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015, p=0.8),
            A.ToGray(p=0.01),
        ]

    return A.Compose(
        common_pre + spatial + grayscale_tf + common_post,
        bbox_params=A.BboxParams(
            format='yolo', label_fields=['class_labels'], min_visibility=0.3
        ),
    )


def get_val_transforms(img_size: int = 640, grayscale: bool = False) -> A.Compose:
    grayscale_tf = [A.ToGray(p=1.0)] if grayscale else []
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
        *grayscale_tf,
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
        augment_level: str = "light",
        grayscale: bool = False,
        mosaic: bool = False,
    ):
        self.img_size   = img_size
        self.mosaic     = mosaic and (split == "train")
        self.transforms = transforms or (
            get_train_transforms(img_size, augment_level, grayscale=grayscale)
            if split == "train"
            else get_val_transforms(img_size, grayscale=grayscale)
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

    def _load_raw(self, idx: int):
        """Load image and labels without any transforms."""
        img_path, lbl_path = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, classes = [], []
        if lbl_path and lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, w, h = parts
                        boxes.append([float(cx), float(cy), float(w), float(h)])
                        classes.append(int(cls))
        return img, boxes, classes

    def _mosaic4(self, idx: int):
        """
        Combine 4 images into a single mosaic tile.

        Letterbox-resizes each image to img_size×img_size (preserving aspect
        ratio), picks a random center point, and places one quadrant from each
        image into the output canvas.
        Bounding boxes are shifted and clipped accordingly.
        """
        s = self.img_size
        yc = int(np.random.uniform(s * 0.25, s * 0.75))
        xc = int(np.random.uniform(s * 0.25, s * 0.75))

        indices = [idx] + [np.random.randint(0, len(self.samples)) for _ in range(3)]

        mosaic_img = np.full((s, s, 3), 114, dtype=np.uint8)
        mosaic_boxes: List = []
        mosaic_classes: List = []

        for i, index in enumerate(indices):
            img, boxes, classes = self._load_raw(index)

            # Letterbox resize: preserve aspect ratio, pad with gray (114)
            h0, w0 = img.shape[:2]
            scale = s / max(h0, w0)
            new_w, new_h = int(w0 * scale), int(h0 * scale)
            img = cv2.resize(img, (new_w, new_h))
            pad_x, pad_y = (s - new_w) // 2, (s - new_h) // 2
            letterboxed = np.full((s, s, 3), 114, dtype=np.uint8)
            letterboxed[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img
            img = letterboxed

            # Adjust YOLO boxes for letterbox padding
            boxes = [
                [(cx * new_w + pad_x) / s, (cy * new_h + pad_y) / s,
                 bw * new_w / s, bh * new_h / s]
                for cx, cy, bw, bh in boxes
            ]

            # Determine source crop (b) and canvas placement (a) per quadrant
            if i == 0:      # top-left
                x1a, y1a, x2a, y2a = 0, 0, xc, yc
                x1b, y1b, x2b, y2b = s - xc, s - yc, s, s
            elif i == 1:    # top-right
                x1a, y1a, x2a, y2a = xc, 0, s, yc
                x1b, y1b, x2b, y2b = 0, s - yc, s - xc, s
            elif i == 2:    # bottom-left
                x1a, y1a, x2a, y2a = 0, yc, xc, s
                x1b, y1b, x2b, y2b = s - xc, 0, s, s - yc
            else:           # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, s, s
                x1b, y1b, x2b, y2b = 0, 0, s - xc, s - yc

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            # Shift boxes: pixel offset from source crop to canvas placement
            offset_x = (x1a - x1b) / s
            offset_y = (y1a - y1b) / s

            for box, cls in zip(boxes, classes):
                cx, cy, bw, bh = box
                new_cx = cx + offset_x
                new_cy = cy + offset_y

                # Clip to canvas [0, 1]
                x1 = max(new_cx - bw / 2, 0.0)
                y1 = max(new_cy - bh / 2, 0.0)
                x2 = min(new_cx + bw / 2, 1.0)
                y2 = min(new_cy + bh / 2, 1.0)
                new_w = x2 - x1
                new_h = y2 - y1

                # Keep box if ≥30% of original area remains (matches min_visibility)
                orig_area = bw * bh
                if orig_area > 0 and (new_w * new_h) / orig_area >= 0.3:
                    mosaic_boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, new_w, new_h])
                    mosaic_classes.append(cls)

        return mosaic_img, mosaic_boxes, mosaic_classes

    def __getitem__(self, idx: int):
        img_path = str(self.samples[idx][0])

        if self.mosaic:
            img, boxes, classes = self._mosaic4(idx)
        else:
            img, boxes, classes = self._load_raw(idx)

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
        except Exception as e:
            import warnings
            warnings.warn(f"Augmentation failed for {img_path}: {e!r} — falling back to resize-only")
            fallback = A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(self.img_size, self.img_size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
            aug    = fallback(image=img, bboxes=boxes, class_labels=classes)
            img_t  = aug['image']
            boxes  = list(aug['bboxes'])
            classes = list(aug['class_labels'])

        return img_t, boxes, classes, img_path


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
    augment_level: str = "light",
    grayscale: bool = False,
    mosaic: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders."""
    val_cap  = max(1, max_samples // 5) if max_samples else 0
    train_ds = FashionDataset(yolo_dir, "train", img_size, max_samples=max_samples,
                              augment_level=augment_level, grayscale=grayscale,
                              mosaic=mosaic)
    val_ds   = FashionDataset(yolo_dir, "val",   img_size, max_samples=val_cap,
                              grayscale=grayscale)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=workers, pin_memory=torch.cuda.is_available(),
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=workers, pin_memory=torch.cuda.is_available(),
    )
    return train_dl, val_dl
"""
src/detection/yolo_world.py
───────────────────────────
YOLO-World zero-shot detector for fashion categories.
Uses open-vocabulary detection — no fine-tuning required.
"""

from __future__ import annotations

import ssl
import time
from typing import List

import numpy as np
from ultralytics import YOLO

from src.detection.detector import (
    BaseDetector,
    CATEGORY_COLORS,
    CATEGORY_NAMES,
    Detection,
    DetectionResult,
)

# Map YOLO-World class index (set_classes order) → original category id
YOLO_WORLD_CLASSES = [
    "short sleeve top",
    "long sleeve top",
    "short sleeve outwear",
    "long sleeve outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short sleeve dress",
    "long sleeve dress",
    "vest dress",
    "sling dress",
]


class YOLOWorldDetector(BaseDetector):
    """
    Zero-shot fashion detector powered by YOLO-World.

    Loads ``yolov8s-worldv2.pt`` and configures it with the 13 fashion
    categories via ``set_classes()``.  No fine-tuned weights needed.

    Default ``conf_thres`` is lower (0.15) because zero-shot models
    produce lower raw confidences than fine-tuned ones.
    """

    def __init__(
        self,
        weights:    str   = "yolov8s-worldv2.pt",
        conf_thres: float = 0.15,
        iou_thres:  float = 0.45,
        device:     str   = "",
        imgsz:      int   = 640,
    ):
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres
        self.imgsz      = imgsz

        print(f"🔍  Loading YOLO-World model: {weights}")
        self.model = YOLO(weights)

        # CLIP download may fail behind proxies / on macOS without certs installed.
        # Temporarily allow unverified SSL so set_classes() can fetch the CLIP model.
        _default_ctx = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            self.model.set_classes(YOLO_WORLD_CLASSES)
        finally:
            ssl._create_default_https_context = _default_ctx

        if device:
            self.model.to(device)
        print("✅  YOLO-World ready")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run zero-shot inference on a BGR frame."""
        t0 = time.perf_counter()

        results = self.model.predict(
            source  = frame,
            conf    = self.conf_thres,
            iou     = self.iou_thres,
            imgsz   = self.imgsz,
            verbose = False,
        )

        inference_ms = (time.perf_counter() - t0) * 1000
        detections   = self._parse(results)

        return DetectionResult(
            detections   = detections,
            frame_shape  = frame.shape,
            inference_ms = inference_ms,
            timestamp    = time.time(),
        )

    def _parse(self, results) -> List[Detection]:
        detections = []
        for r in results:
            for box in r.boxes:
                # class id from set_classes order matches CATEGORY_NAMES order
                class_id = int(box.cls.item())
                detections.append(Detection(
                    class_id   = class_id,
                    class_name = CATEGORY_NAMES.get(class_id, f"class_{class_id}"),
                    confidence = float(box.conf.item()),
                    bbox       = [int(v) for v in box.xyxy[0].tolist()],
                    color      = CATEGORY_COLORS.get(class_id, (0, 255, 0)),
                ))
        return detections
